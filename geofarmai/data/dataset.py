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

        registry: dict[str, VariableSpec] = {}
        conflicts: list[str] = []
        for source in sources:
            for spec in source.variables:
                existing = registry.get(spec.name)
                if existing is None:
                    registry[spec.name] = spec
                elif existing != spec:
                    conflicts.append(spec.name)
        if conflicts:
            raise SchemaValidationError(
                "FieldDataset contains conflicting specifications for variables: "
                f"{sorted(set(conflicts))}."
            )
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
        variables: Sequence[VariableSpec],
        coordinates: CoordinateSpec | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
        read_csv_kwargs: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
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
        variables: Sequence[VariableSpec],
        coordinates: CoordinateSpec | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
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
        variables: Sequence[VariableSpec],
        coordinates: CoordinateSpec | None = None,
        x: str | None = None,
        y: str | None = None,
        crs: CRS | str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FieldDataset":
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
        by_name: dict[str, VariableSpec] = {}
        for source in self.sources:
            for spec in source.variables:
                by_name.setdefault(spec.name, spec)
        return tuple(by_name.values())

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
        return {
            spec.name: {
                "role": spec.role.value,
                "domain": spec.domain,
                "units": spec.units,
                "description": spec.description,
            }
            for spec in self.variable_specs
        }

    def get_source(self, source_id: str) -> DataSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise DataModelError(f"FieldDataset has no source with identifier {source_id!r}.")


def _legacy_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
