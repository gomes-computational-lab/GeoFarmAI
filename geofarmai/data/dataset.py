"""Field-level container for one or many canonical input sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from geofarmai.data.schema import CoordinateSpec, VariableRole, VariableSpec
from geofarmai.data.source import DataSource
from geofarmai.exceptions import DataModelError, SchemaValidationError


@dataclass(frozen=True, slots=True)
class FieldDataset:
    """Complete field input assembled from one or many unharmonized sources.

    This Phase 1 container does not interpolate, align, concatenate, or
    otherwise harmonize observations from different sources.
    """

    sources: tuple[DataSource, ...]

    def __post_init__(self) -> None:
        try:
            sources = tuple(self.sources)
        except TypeError as exc:
            raise DataModelError(
                "FieldDataset sources must be a sequence of DataSource objects."
            ) from exc
        if not sources:
            raise DataModelError("FieldDataset requires at least one DataSource.")
        if any(not isinstance(source, DataSource) for source in sources):
            raise DataModelError("FieldDataset sources must contain only DataSource objects.")

        source_ids = [source.source_id for source in sources]
        duplicates = sorted({source_id for source_id in source_ids if source_ids.count(source_id) > 1})
        if duplicates:
            raise DataModelError(f"FieldDataset contains duplicate source identifiers: {duplicates}.")

        object.__setattr__(self, "sources", sources)

    @classmethod
    def from_sources(cls, sources: Sequence[DataSource]) -> "FieldDataset":
        """Create a field dataset without harmonizing its sources."""

        try:
            normalized = tuple(sources)
        except TypeError as exc:
            raise DataModelError(
                "FieldDataset sources must be a sequence of DataSource objects."
            ) from exc
        return cls(normalized)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        source_id: str | None = None,
        variables: Sequence[VariableSpec] | None = None,
        predictors: Sequence[str] | None = None,
        outcome: str | Sequence[str] | None = None,
        coordinates: CoordinateSpec | tuple[str, str] | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
        read_csv_kwargs: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
        variables = _constructor_variables(variables, predictors, outcome)
        coordinates, x, y, crs = _constructor_coordinates(coordinates, x, y, crs)
        return cls.from_sources(
            [
                DataSource.from_csv(
                    path,
                    source_id=source_id,
                    variables=variables,
                    coordinates=coordinates,
                    x=x,
                    y=y,
                    crs=crs,
                    metadata=metadata,
                    read_csv_kwargs=read_csv_kwargs,
                )
            ]
        )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        *,
        source_id: str = "dataframe",
        variables: Sequence[VariableSpec] | None = None,
        predictors: Sequence[str] | None = None,
        outcome: str | Sequence[str] | None = None,
        coordinates: CoordinateSpec | tuple[str, str] | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
        variables = _constructor_variables(variables, predictors, outcome)
        coordinates, x, y, crs = _constructor_coordinates(coordinates, x, y, crs)
        return cls.from_sources(
            [
                DataSource.from_dataframe(
                    dataframe,
                    source_id=source_id,
                    variables=variables,
                    coordinates=coordinates,
                    x=x,
                    y=y,
                    crs=crs,
                    metadata=metadata,
                )
            ]
        )

    @classmethod
    def from_geodataframe(
        cls,
        geodataframe: gpd.GeoDataFrame,
        *,
        source_id: str = "geodataframe",
        variables: Sequence[VariableSpec] | None = None,
        predictors: Sequence[str] | None = None,
        outcome: str | Sequence[str] | None = None,
        coordinates: CoordinateSpec | tuple[str, str] | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
        variables = _constructor_variables(variables, predictors, outcome)
        coordinates, x, y, crs = _constructor_coordinates(coordinates, x, y, crs)
        return cls.from_sources(
            [
                DataSource.from_geodataframe(
                    geodataframe,
                    source_id=source_id,
                    variables=variables,
                    coordinates=coordinates,
                    x=x,
                    y=y,
                    crs=crs,
                    metadata=metadata,
                )
            ]
        )

    @classmethod
    def from_legacy_config(
        cls,
        config: Mapping[str, Any],
        *,
        base_path: str | Path | None = None,
    ) -> "FieldDataset":
        """Adapt the current soil/yield configuration without changing its pipeline.

        Roles come from the semantic legacy config sections, never from column
        names. The configured outcome keeps its original column name.
        """

        try:
            project = config["project"]
        except (KeyError, TypeError) as exc:
            raise SchemaValidationError("Legacy configuration requires a 'project' mapping.") from exc
        if not isinstance(project, Mapping):
            raise SchemaValidationError("Legacy configuration 'project' must be a mapping.")

        root = Path(base_path) if base_path is not None else Path.cwd()
        source_specs: list[DataSource] = []
        crs = project.get("crs_in")

        if "soil" in project:
            soil = project["soil"]
            _validate_legacy_section("soil", soil, ("path", "x", "y", "variables"))
            if isinstance(soil["variables"], (str, bytes)) or not isinstance(
                soil["variables"], Sequence
            ):
                raise SchemaValidationError(
                    "Legacy configuration 'soil.variables' must be a sequence of column names."
                )
            variables = [
                VariableSpec(name=name, role=VariableRole.PREDICTOR, domain="soil")
                for name in soil["variables"]
            ]
            identifier = soil.get("id_column")
            if identifier:
                variables.append(VariableSpec(identifier, VariableRole.IDENTIFIER))
            source_specs.append(
                DataSource.from_csv(
                    _legacy_path(root, soil["path"]),
                    source_id="soil",
                    variables=variables,
                    x=soil["x"],
                    y=soil["y"],
                    crs=crs,
                    metadata={
                        "legacy_section": "soil",
                        "required_variables": tuple(soil.get("required_variables", ())),
                    },
                )
            )

        if "yield" in project:
            outcome = project["yield"]
            _validate_legacy_section("yield", outcome, ("path", "x", "y", "column"))
            variables = [
                VariableSpec(name=outcome["column"], role=VariableRole.OUTCOME, domain="crop")
            ]
            identifier = outcome.get("id_column")
            if identifier:
                variables.append(VariableSpec(identifier, VariableRole.IDENTIFIER))
            source_specs.append(
                DataSource.from_csv(
                    _legacy_path(root, outcome["path"]),
                    source_id="yield",
                    variables=variables,
                    x=outcome["x"],
                    y=outcome["y"],
                    crs=crs,
                    metadata={"legacy_section": "yield"},
                )
            )

        if not source_specs:
            raise SchemaValidationError(
                "Legacy configuration must contain at least one 'soil' or 'yield' input section."
            )
        return cls.from_sources(source_specs)

    @property
    def source_list(self) -> tuple[DataSource, ...]:
        return self.sources

    @property
    def variable_specs(self) -> tuple[VariableSpec, ...]:
        """Return declarations in source order without erasing cross-source names.

        A variable is identified canonically by its source ID and name. Separate
        sources may therefore use the same name with different explicit roles.
        Duplicate declarations within one source remain invalid.
        """

        return tuple(spec for source in self.sources for spec in source.variables)

    @property
    def predictors(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variable_specs if spec.role is VariableRole.PREDICTOR)

    @property
    def outcomes(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variable_specs if spec.role is VariableRole.OUTCOME)

    @property
    def predictor_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.predictors)

    @property
    def outcome_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.outcomes)

    @property
    def has_outcomes(self) -> bool:
        return bool(self.outcomes)

    @property
    def crs_information(self) -> dict[str, CRS | None]:
        return {source.source_id: source.crs for source in self.sources}

    @property
    def common_crs(self) -> CRS | None:
        crs_values = {source.crs for source in self.sources}
        if len(crs_values) == 1 and None not in crs_values:
            return next(iter(crs_values))
        return None

    @property
    def variable_metadata(self) -> dict[str, dict[str, str | None]]:
        counts: dict[str, int] = {}
        for source in self.sources:
            for spec in source.variables:
                counts[spec.name] = counts.get(spec.name, 0) + 1

        metadata: dict[str, dict[str, str | None]] = {}
        for source in self.sources:
            for spec in source.variables:
                key = spec.name if counts[spec.name] == 1 else f"{source.source_id}:{spec.name}"
                metadata[key] = {
                    "role": spec.role.value,
                    "domain": spec.domain,
                    "units": spec.units,
                    "description": spec.description,
                }
        return metadata

    def get_source(self, source_id: str) -> DataSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise DataModelError(f"FieldDataset has no source with identifier {source_id!r}.")


def _legacy_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _constructor_variables(
    variables: Sequence[VariableSpec] | None,
    predictors: Sequence[str] | None,
    outcome: str | Sequence[str] | None,
) -> tuple[VariableSpec, ...]:
    """Normalize concise constructor arguments into explicit role declarations."""

    if variables is not None and (predictors is not None or outcome is not None):
        raise SchemaValidationError(
            "Pass either variables=[VariableSpec(...)] or predictors/outcome, not both."
        )
    if variables is not None:
        return tuple(variables)

    declarations: list[VariableSpec] = []
    if predictors is not None:
        if isinstance(predictors, (str, bytes)):
            raise SchemaValidationError("predictors must be a sequence of column names.")
        declarations.extend(VariableSpec(name, VariableRole.PREDICTOR) for name in predictors)

    if outcome is not None:
        outcome_names = (outcome,) if isinstance(outcome, str) else tuple(outcome)
        declarations.extend(VariableSpec(name, VariableRole.OUTCOME) for name in outcome_names)

    if not declarations:
        raise SchemaValidationError(
            "Declare variables explicitly, or provide predictors and an optional outcome."
        )
    return tuple(declarations)


def _constructor_coordinates(
    coordinates: CoordinateSpec | tuple[str, str] | None,
    x: str | None,
    y: str | None,
    crs: CRS | str | int | None,
) -> tuple[CoordinateSpec | None, str | None, str | None, CRS | str | int | None]:
    """Accept the concise ``coordinates=(x, y)`` public-API spelling."""

    if isinstance(coordinates, tuple):
        if len(coordinates) != 2:
            raise SchemaValidationError("coordinates must contain exactly two column names.")
        if x is not None or y is not None:
            raise SchemaValidationError(
                "Pass either coordinates=(x, y) or x/y arguments, not both."
            )
        return CoordinateSpec(coordinates[0], coordinates[1], crs), None, None, None
    return coordinates, x, y, crs


def _validate_legacy_section(
    name: str, section: Any, required: tuple[str, ...]
) -> None:
    if not isinstance(section, Mapping):
        raise SchemaValidationError(f"Legacy configuration section {name!r} must be a mapping.")
    missing = [key for key in required if key not in section]
    if missing:
        raise SchemaValidationError(
            f"Legacy configuration section {name!r} is missing keys: {missing}."
        )
