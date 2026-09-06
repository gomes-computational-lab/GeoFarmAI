"""Typed YAML translation for GeoFarmAI's canonical scientific API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

from pyproj import CRS
from pyproj.exceptions import CRSError
import yaml

from geofarmai.data import (
    CoordinateSpec,
    DataSource,
    FieldDataset,
    VariableRole,
    VariableSpec,
)
from geofarmai.exceptions import (
    ConfigurationError,
    DataModelError,
    ModelConfigurationError,
    OutcomeConfigurationError,
)
from geofarmai.model import GeoFarmModel
from geofarmai.result import GeoFarmResult


_SOURCE_KEYS = {"id", "path", "coordinates", "variables", "metadata"}
_VARIABLE_KEYS = {"role", "domain", "units", "description"}
_ANALYSIS_KEYS = {
    "harmonization",
    "decomposition",
    "clustering",
    "selection",
    "random_state",
}
_HARMONIZATION_KEYS = {
    "strategy",
    "resolution",
    "method",
    "target_crs",
    "location_tolerance",
    "min_cell_size",
    "max_cell_size",
    "buffer_m",
}
_DECOMPOSITION_KEYS = {"method", "engine", "n_components"}
_CLUSTERING_KEYS = {"methods", "k", "weights_k", "silhouette_sample_size"}
_SELECTION_KEYS = {"metric", "outcome"}
_HARMONIZATION_STRATEGIES = {"auto", "direct", "grid"}
_RECONCILIATION_METHODS = {"idw", "nearest", "buffer_mean"}
_MISSING = object()


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"Duplicate YAML key {key!r} is not allowed.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class SourceConfiguration:
    """Declarative configuration for one stable canonical data source."""

    source_id: str
    configured_path: str
    resolved_path: Path
    coordinates: CoordinateSpec
    variables: tuple[VariableSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> DataSource:
        metadata = dict(self.metadata)
        metadata["geofarmai_configuration"] = {
            "configured_path": self.configured_path,
            "resolved_path": str(self.resolved_path),
        }
        try:
            return DataSource.from_csv(
                self.resolved_path,
                source_id=self.source_id,
                variables=self.variables,
                coordinates=self.coordinates,
                metadata=metadata,
            )
        except DataModelError as exc:
            raise ConfigurationError(
                f"Invalid source {self.source_id!r}: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class AnalysisConfiguration:
    """Typed model options translated directly to ``GeoFarmModel``."""

    model_options: Mapping[str, Any]

    def build_model(self) -> GeoFarmModel:
        return GeoFarmModel(**dict(self.model_options))


@dataclass(frozen=True, slots=True)
class GeoFarmConfiguration:
    """A loaded YAML document that builds canonical data and model objects."""

    configuration_path: Path
    sources: tuple[SourceConfiguration, ...]
    analysis: AnalysisConfiguration
    metadata: Mapping[str, Any] = field(default_factory=dict)
    legacy: bool = False
    _legacy_dataset: FieldDataset | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def model_options(self) -> dict[str, Any]:
        return dict(self.analysis.model_options)

    def build_dataset(self) -> FieldDataset:
        if self._legacy_dataset is not None:
            return self._legacy_dataset
        return FieldDataset.from_sources([source.build() for source in self.sources])

    def build_model(self) -> GeoFarmModel:
        return self.analysis.build_model()

    def build(self) -> tuple[FieldDataset, GeoFarmModel]:
        return self.build_dataset(), self.build_model()

    def run(self) -> GeoFarmResult:
        data, model = self.build()
        return model.fit(data)


def load_config(path: str | Path) -> GeoFarmConfiguration:
    """Load the source-based YAML schema relative to the YAML file itself."""

    config_path, document = _load_document(path)
    _reject_unknown(document, {"sources", "analysis", "metadata"}, "top level")
    if "sources" not in document:
        raise ConfigurationError("Configuration requires a top-level 'sources' list.")
    raw_sources = document["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigurationError("'sources' must be a non-empty list.")

    sources = tuple(
        _parse_source(item, index, config_path.parent)
        for index, item in enumerate(raw_sources)
    )
    source_ids = [source.source_id for source in sources]
    duplicates = sorted(
        source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
    )
    if duplicates:
        raise ConfigurationError(f"Duplicate source IDs are not allowed: {duplicates}.")

    if "analysis" not in document:
        raise ConfigurationError("Configuration requires a top-level 'analysis' mapping.")
    analysis = _parse_analysis(document["analysis"], sources)
    metadata = document.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("Top-level 'metadata' must be a mapping when provided.")
    return GeoFarmConfiguration(
        configuration_path=config_path,
        sources=sources,
        analysis=analysis,
        metadata=dict(metadata),
    )


def load_legacy_config(path: str | Path) -> GeoFarmConfiguration:
    """Translate the historical project-level YAML schema in isolation."""

    config_path, document = _load_document(path)
    if "project" not in document:
        raise ConfigurationError(
            "Legacy configuration requires a top-level 'project' mapping."
        )
    legacy_root = _legacy_root(config_path, document)
    try:
        dataset = _build_legacy_dataset(document, legacy_root)
    except (DataModelError, OutcomeConfigurationError) as exc:
        raise ConfigurationError(f"Invalid legacy configuration: {exc}") from exc

    source_configs: list[SourceConfiguration] = []
    project = document["project"]
    for source in dataset.sources:
        section_name = str(source.metadata.get("legacy_section", source.source_id))
        section = project.get(section_name, {})
        configured_path = str(section.get("path", source.path))
        source.metadata["geofarmai_configuration"] = {
            "configured_path": configured_path,
            "resolved_path": str(source.path),
            "legacy": True,
            "identifier_synthesis": False,
        }
        source_configs.append(
            SourceConfiguration(
                source_id=source.source_id,
                configured_path=configured_path,
                resolved_path=Path(source.path),
                coordinates=source.coordinates,
                variables=tuple(source.variables),
                metadata=dict(source.metadata),
            )
        )

    options = _legacy_model_options(document, dataset)
    return GeoFarmConfiguration(
        configuration_path=config_path,
        sources=tuple(source_configs),
        analysis=AnalysisConfiguration(options),
        metadata={
            "legacy_adapter": True,
            "legacy_identifier_synthesis": (
                "Not applied by the canonical adapter; the unchanged legacy pipeline "
                "still synthesizes sample_id when required."
            ),
            "legacy_configured_random_states": tuple(
                document.get("clustering", {}).get("seeds", [42])
            ),
            "legacy_random_state_policy": (
                "GeoFarmModel uses the first configured legacy seed. Use the unchanged "
                "legacy experiment pipeline to execute every configured seed."
            ),
        },
        legacy=True,
        _legacy_dataset=dataset,
    )


def run_from_config(path: str | Path, *, legacy: bool = False) -> GeoFarmResult:
    """Load YAML and run only through canonical ``GeoFarmModel.fit``."""

    loaded = load_legacy_config(path) if legacy else load_config(path)
    return loaded.run()


def _load_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}.")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=_UniqueKeyLoader)
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Could not read YAML configuration {config_path}: {exc}"
        ) from exc
    if not isinstance(document, Mapping):
        raise ConfigurationError("The YAML document must contain a top-level mapping.")
    return config_path, dict(document)


def _parse_source(item: Any, index: int, base_path: Path) -> SourceConfiguration:
    location = f"sources[{index}]"
    if not isinstance(item, Mapping):
        raise ConfigurationError(f"{location} must be a mapping.")
    _reject_unknown(item, _SOURCE_KEYS, location)

    source_id = _required_string(item, "id", location)
    configured_path = _required_string(item, "path", location)
    path_value = Path(configured_path).expanduser()
    resolved_path = (
        path_value.resolve()
        if path_value.is_absolute()
        else (base_path / path_value).resolve()
    )
    if not resolved_path.is_file():
        raise ConfigurationError(
            f"Source {source_id!r} path does not exist: {resolved_path} "
            f"(configured as {configured_path!r})."
        )

    coordinates = _parse_coordinates(item.get("coordinates"), location)
    variables = _parse_variables(item.get("variables"), location)
    metadata = item.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigurationError(f"{location}.metadata must be a mapping.")
    return SourceConfiguration(
        source_id=source_id,
        configured_path=configured_path,
        resolved_path=resolved_path,
        coordinates=coordinates,
        variables=variables,
        metadata=dict(metadata),
    )


def _parse_coordinates(value: Any, location: str) -> CoordinateSpec:
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            f"{location}.coordinates must explicitly define x, y, and crs."
        )
    _reject_unknown(value, {"x", "y", "crs"}, f"{location}.coordinates")
    missing = [name for name in ("x", "y", "crs") if value.get(name) is None]
    if missing:
        raise ConfigurationError(
            f"{location}.coordinates is missing required fields: {missing}."
        )
    try:
        return CoordinateSpec(x=value["x"], y=value["y"], crs=value["crs"])
    except DataModelError as exc:
        raise ConfigurationError(f"Invalid {location}.coordinates: {exc}") from exc


def _parse_variables(value: Any, location: str) -> tuple[VariableSpec, ...]:
    if not isinstance(value, Mapping) or not value:
        raise ConfigurationError(f"{location}.variables must be a non-empty mapping.")
    variables: list[VariableSpec] = []
    for name, declaration in value.items():
        variable_location = f"{location}.variables.{name}"
        if not isinstance(declaration, Mapping):
            raise ConfigurationError(f"{variable_location} must be a mapping.")
        _reject_unknown(declaration, _VARIABLE_KEYS, variable_location)
        if "role" not in declaration:
            raise ConfigurationError(f"{variable_location} requires an explicit role.")
        try:
            variables.append(
                VariableSpec(
                    name=name,
                    role=declaration["role"],
                    domain=declaration.get("domain"),
                    units=declaration.get("units"),
                    description=declaration.get("description"),
                )
            )
        except DataModelError as exc:
            raise ConfigurationError(f"Invalid {variable_location}: {exc}") from exc
    return tuple(variables)


def _parse_analysis(
    value: Any,
    sources: tuple[SourceConfiguration, ...],
) -> AnalysisConfiguration:
    if not isinstance(value, Mapping):
        raise ConfigurationError("'analysis' must be a mapping.")
    _reject_unknown(value, _ANALYSIS_KEYS, "analysis")

    harmonization = _parse_harmonization(value.get("harmonization", {}))
    decomposition = _parse_decomposition(value.get("decomposition", {}))
    clustering = _parse_clustering(value.get("clustering", {}))
    selection, selection_outcome = _parse_selection(value.get("selection", _MISSING))
    random_state = value.get("random_state", 42)
    if not isinstance(random_state, Integral) or isinstance(random_state, bool):
        raise ConfigurationError("analysis.random_state must be an integer.")

    options = {
        **harmonization,
        **decomposition,
        **clustering,
        "selection": selection,
        "selection_outcome": selection_outcome,
        "random_state": int(random_state),
    }
    _validate_model_option_names(options)
    _validate_selection_roles(sources, selection, selection_outcome)
    return AnalysisConfiguration(options)


def _parse_harmonization(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis.harmonization must be a mapping.")
    _reject_unknown(value, _HARMONIZATION_KEYS, "analysis.harmonization")
    strategy = str(value.get("strategy", "auto")).lower().strip()
    if strategy not in _HARMONIZATION_STRATEGIES:
        raise ConfigurationError(
            f"Unsupported harmonization strategy {strategy!r}. Choose auto, direct, or grid."
        )
    method = str(value.get("method", "idw")).lower().strip()
    if method not in _RECONCILIATION_METHODS:
        raise ConfigurationError(
            f"Unsupported harmonization method {method!r}. Choose idw, nearest, or buffer_mean."
        )

    params: dict[str, Any] = {}
    resolution = value.get("resolution", "auto")
    if isinstance(resolution, str):
        if resolution.lower().strip() != "auto":
            raise ConfigurationError(
                "analysis.harmonization.resolution must be a positive number or 'auto'."
            )
    elif _positive_real(resolution, "analysis.harmonization.resolution"):
        params["cell_size"] = float(resolution)

    target_crs = value.get("target_crs")
    if target_crs is not None and str(target_crs).lower().strip() != "auto":
        try:
            parsed_target = CRS.from_user_input(target_crs)
        except (CRSError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid analysis.harmonization.target_crs {target_crs!r}."
            ) from exc
        if not parsed_target.is_projected or any(
            axis.unit_conversion_factor != 1.0 for axis in parsed_target.axis_info[:2]
        ):
            raise ConfigurationError(
                "analysis.harmonization.target_crs must be a projected CRS whose "
                "horizontal units are metres."
            )
        params["target_crs"] = target_crs

    numeric_parameters = {
        "location_tolerance": ("location_tolerance", False),
        "min_cell_size": ("min_cell_size", True),
        "max_cell_size": ("max_cell_size", True),
        "buffer_m": ("buffer_m", False),
    }
    for yaml_name, (model_name, positive) in numeric_parameters.items():
        if yaml_name not in value:
            continue
        number = value[yaml_name]
        _real(number, f"analysis.harmonization.{yaml_name}")
        if positive and number <= 0:
            raise ConfigurationError(
                f"analysis.harmonization.{yaml_name} must be positive."
            )
        if not positive and number < 0:
            raise ConfigurationError(
                f"analysis.harmonization.{yaml_name} must be non-negative."
            )
        params[model_name] = float(number)
    if (
        "min_cell_size" in params
        and "max_cell_size" in params
        and params["min_cell_size"] > params["max_cell_size"]
    ):
        raise ConfigurationError(
            "analysis.harmonization min_cell_size must not exceed max_cell_size."
        )
    return {
        "harmonization_strategy": strategy,
        "reconciliation_method": method,
        "harmonization_params": params,
    }


def _parse_decomposition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis.decomposition must be a mapping.")
    _reject_unknown(value, _DECOMPOSITION_KEYS, "analysis.decomposition")
    method = str(value.get("method", "pca")).lower().strip()
    engine = str(value.get("engine", "python")).lower().strip()
    n_components = value.get("n_components")
    if n_components is not None and (
        not isinstance(n_components, Integral)
        or isinstance(n_components, bool)
        or n_components < 1
    ):
        raise ConfigurationError(
            "analysis.decomposition.n_components must be a positive integer or null."
        )
    return {
        "decomposition": method,
        "multispati_engine": engine,
        "n_components": None if n_components is None else int(n_components),
    }


def _parse_clustering(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis.clustering must be a mapping.")
    _reject_unknown(value, _CLUSTERING_KEYS, "analysis.clustering")
    methods = value.get("methods", ["kmeans"])
    if isinstance(methods, str):
        normalized_methods: str | tuple[str, ...] = methods
    elif isinstance(methods, list) and methods:
        normalized_methods = tuple(methods)
    else:
        raise ConfigurationError(
            "analysis.clustering.methods must be a method name or non-empty list."
        )
    k = _parse_k(value.get("k", {"min": 2, "max": 5}))
    weights_k = value.get("weights_k", 8)
    if not isinstance(weights_k, Integral) or isinstance(weights_k, bool) or weights_k < 1:
        raise ConfigurationError("analysis.clustering.weights_k must be a positive integer.")
    sample_size = value.get("silhouette_sample_size")
    if sample_size is not None and (
        not isinstance(sample_size, Integral)
        or isinstance(sample_size, bool)
        or sample_size < 2
    ):
        raise ConfigurationError(
            "analysis.clustering.silhouette_sample_size must be at least 2 or null."
        )
    return {
        "clustering": normalized_methods,
        "k": k,
        "weights_k": int(weights_k),
        "silhouette_sample_size": (
            None if sample_size is None else int(sample_size)
        ),
    }


def _parse_k(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 2:
            raise ConfigurationError("analysis.clustering.k must be at least 2.")
        return int(value)
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            "analysis.clustering.k must be an integer or a {min, max} mapping."
        )
    _reject_unknown(value, {"min", "max"}, "analysis.clustering.k")
    minimum = value.get("min")
    maximum = value.get("max")
    if any(
        not isinstance(item, Integral) or isinstance(item, bool)
        for item in (minimum, maximum)
    ):
        raise ConfigurationError(
            "analysis.clustering.k range requires integer min and max values."
        )
    if minimum < 2 or maximum < minimum:
        raise ConfigurationError(
            "analysis.clustering.k range requires 2 <= min <= max."
        )
    return tuple(range(int(minimum), int(maximum) + 1))


def _parse_selection(value: Any) -> tuple[str | None, str | None]:
    if value is _MISSING:
        return "silhouette", None
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise ConfigurationError(
            "analysis.selection must be a {metric, outcome} mapping or null."
        )
    _reject_unknown(value, _SELECTION_KEYS, "analysis.selection")
    if "metric" not in value:
        raise ConfigurationError("analysis.selection requires a metric; use null for none.")
    metric = value["metric"]
    if not isinstance(metric, str) or not metric.strip():
        raise ConfigurationError("analysis.selection.metric must be a non-empty string.")
    outcome = value.get("outcome")
    if outcome is not None and (not isinstance(outcome, str) or not outcome.strip()):
        raise ConfigurationError("analysis.selection.outcome must be a non-empty string.")
    return metric.lower().strip(), None if outcome is None else outcome.strip()


def _validate_model_option_names(options: Mapping[str, Any]) -> None:
    model = GeoFarmModel(**dict(options))
    try:
        model._normalized_decomposition()
        model._normalized_algorithms()
        model._normalized_k_values()
        model._normalized_selection()
    except ModelConfigurationError as exc:
        raise ConfigurationError(str(exc)) from exc


def _validate_selection_roles(
    sources: tuple[SourceConfiguration, ...],
    selection: str | None,
    selection_outcome: str | None,
) -> None:
    scientific = [
        (source.source_id, spec)
        for source in sources
        for spec in source.variables
        if spec.role in {VariableRole.PREDICTOR, VariableRole.OUTCOME}
    ]
    outcomes = [item for item in scientific if item[1].role is VariableRole.OUTCOME]
    predictors = [item for item in scientific if item[1].role is VariableRole.PREDICTOR]
    if not predictors:
        raise ConfigurationError(
            "Configuration requires at least one explicitly declared predictor variable."
        )
    if selection != "variance_reduction":
        if selection_outcome is not None:
            raise ConfigurationError(
                "analysis.selection.outcome is only valid for variance_reduction."
            )
        return
    if not outcomes:
        raise ConfigurationError(
            "variance_reduction selection requires an explicitly declared outcome variable."
        )
    if selection_outcome is None:
        if len(outcomes) > 1:
            raise ConfigurationError(
                "variance_reduction selection with multiple outcomes requires "
                "analysis.selection.outcome."
            )
        return

    counts: dict[str, int] = {}
    for _, spec in scientific:
        counts[spec.name] = counts.get(spec.name, 0) + 1
    matches = [
        (source_id, spec)
        for source_id, spec in outcomes
        if selection_outcome
        in {
            spec.name,
            spec.name if counts[spec.name] == 1 else f"{source_id}:{spec.name}",
        }
    ]
    if len(matches) == 1:
        return
    predictor_matches = [spec for _, spec in predictors if spec.name == selection_outcome]
    if predictor_matches:
        raise ConfigurationError(
            f"Selection outcome {selection_outcome!r} is declared as a predictor, not an outcome."
        )
    if len(matches) > 1:
        raise ConfigurationError(
            f"Selection outcome {selection_outcome!r} is ambiguous; use source_id:variable_name."
        )
    raise ConfigurationError(
        f"Unknown selection outcome {selection_outcome!r}; it must name a declared outcome."
    )


def _legacy_model_options(
    document: Mapping[str, Any],
    dataset: FieldDataset,
) -> dict[str, Any]:
    grid = document.get("grid", {})
    spatial = document.get("spatial_pca", {})
    clustering = document.get("clustering", {})
    raster = document.get("raster", {})
    use_r = bool(spatial.get("use_r_multispati", False))
    if raster.get("enabled", False):
        if not raster.get("use_pca", True):
            decomposition = "none"
        elif use_r:
            decomposition = "multispati"
        elif spatial.get("engine", "multispaeti") == "multispaeti":
            decomposition = "multispati"
        else:
            decomposition = "pca"
    else:
        decomposition = "multispati" if use_r else "pca"

    cell_size = grid.get("cell_size_m")
    harmonization_params: dict[str, Any] = {
        "min_cell_size": float(grid.get("min_cell_size_m", 3.0)),
        "max_cell_size": float(grid.get("max_cell_size_m", 30.0)),
        "buffer_m": float(grid.get("buffer_m", 15.0)),
    }
    if cell_size:
        harmonization_params["cell_size"] = float(cell_size)
    seeds = clustering.get("seeds", [42])
    random_state = seeds[0] if seeds else 42
    outcome_names = dataset.outcome_names
    return {
        "decomposition": decomposition,
        "multispati_engine": "r" if use_r else "python",
        "n_components": int(spatial.get("n_components", 3)),
        "clustering": tuple(clustering.get("algorithms", ["kmeans"])),
        "k": tuple(clustering.get("k_values", [2, 3, 4, 5])),
        "weights_k": int(document.get("weights", {}).get("k", 8)),
        "random_state": int(random_state),
        "selection": "variance_reduction" if outcome_names else "silhouette",
        "selection_outcome": outcome_names[0] if outcome_names else None,
        "harmonization_strategy": "grid",
        "reconciliation_method": grid.get("method", "idw"),
        "harmonization_params": harmonization_params,
    }


def _build_legacy_dataset(
    document: Mapping[str, Any],
    legacy_root: Path,
) -> FieldDataset:
    """Translate only historical project-level outcome declarations.

    ``project.outcome`` was added to the old flat-table pipeline before the
    source-based schema existed. It remains supported here without teaching
    the canonical loader to infer roles or synthesize legacy identifiers.
    """

    project = document["project"]
    if "outcome" not in project:
        return FieldDataset.from_legacy_config(document, base_path=legacy_root)

    from geofarmai.outcome import resolve_pipeline_outcome

    predictor_document = deepcopy(document)
    predictor_document["project"].pop("yield", None)
    dataset = FieldDataset.from_legacy_config(
        predictor_document,
        base_path=legacy_root,
    )
    outcome = resolve_pipeline_outcome(project)
    if outcome is None:
        return dataset

    if outcome.uses_predictor_source:
        source = dataset.get_source("soil")
        data = source.data.copy()
        if outcome.source_column != outcome.name:
            if outcome.name in data.columns:
                raise ConfigurationError(
                    f"Legacy outcome name {outcome.name!r} conflicts with an existing "
                    "predictor-source column."
                )
            data = data.rename(columns={outcome.source_column: outcome.name})
        updated = DataSource(
            source_id=source.source_id,
            data=data,
            variables=tuple(source.variables)
            + (VariableSpec(outcome.name, VariableRole.OUTCOME),),
            coordinates=source.coordinates,
            metadata={
                **source.metadata,
                "legacy_outcome_source_column": outcome.source_column,
            },
            path=source.path,
        )
        return FieldDataset.from_sources([updated])

    declaration = dict(outcome.source_config or {})
    identifier = declaration.get("id_column")
    variables = [VariableSpec(outcome.source_column, VariableRole.OUTCOME)]
    if identifier:
        variables.append(VariableSpec(identifier, VariableRole.IDENTIFIER))
    external_path = Path(declaration["path"]).expanduser()
    if not external_path.is_absolute():
        external_path = legacy_root / external_path
    external = DataSource.from_csv(
        external_path,
        source_id=str(declaration.get("id", "outcome")),
        variables=variables,
        x=declaration["x"],
        y=declaration["y"],
        crs=project.get("crs_in"),
        metadata={
            "legacy_section": "outcome",
            "legacy_outcome_source_column": outcome.source_column,
        },
    )
    if outcome.source_column != outcome.name:
        data = external.data.rename(columns={outcome.source_column: outcome.name})
        renamed_variables = tuple(
            VariableSpec(
                outcome.name if spec.name == outcome.source_column else spec.name,
                spec.role,
                domain=spec.domain,
                units=spec.units,
                description=spec.description,
            )
            for spec in external.variables
        )
        external = DataSource(
            source_id=external.source_id,
            data=data,
            variables=renamed_variables,
            coordinates=external.coordinates,
            metadata=external.metadata,
            path=external.path,
        )
    return FieldDataset.from_sources([*dataset.sources, external])


def _legacy_root(config_path: Path, document: Mapping[str, Any]) -> Path:
    paths: list[Path] = []
    project = document.get("project", {})
    for section_name in ("soil", "yield"):
        section = project.get(section_name)
        if isinstance(section, Mapping) and section.get("path"):
            path = Path(section["path"])
            if not path.is_absolute():
                paths.append(path)
    candidates = (config_path.parent, config_path.parent.parent, Path.cwd())
    for candidate in candidates:
        if all((candidate / path).is_file() for path in paths):
            return candidate
    return config_path.parent


def _required_string(value: Mapping[str, Any], key: str, location: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ConfigurationError(f"{location}.{key} must be a non-empty string.")
    return item.strip()


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"Unsupported keys in {location}: {unknown}.")


def _real(value: Any, location: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ConfigurationError(f"{location} must be numeric.")
    return float(value)


def _positive_real(value: Any, location: str) -> bool:
    number = _real(value, location)
    if number <= 0:
        raise ConfigurationError(f"{location} must be positive.")
    return True


__all__ = [
    "AnalysisConfiguration",
    "GeoFarmConfiguration",
    "SourceConfiguration",
    "load_config",
    "load_legacy_config",
    "run_from_config",
]
