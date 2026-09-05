"""Canonical representation of one GeoFarmAI input source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pyproj import CRS

from geofarmai.data.schema import CoordinateSpec, VariableRole, VariableSpec
from geofarmai.exceptions import DataSourceError, SchemaValidationError


def _coordinate_spec(
    coordinates: CoordinateSpec | None,
    *,
    x: str | None,
    y: str | None,
    crs: CRS | str | int | None,
    geometry_backed: bool,
) -> CoordinateSpec:
    if coordinates is not None and any(value is not None for value in (x, y, crs)):
        raise SchemaValidationError(
            "Pass either coordinates=CoordinateSpec(...) or x/y/crs arguments, not both."
        )
    if coordinates is not None:
        return coordinates
    if geometry_backed and x is None and y is None:
        return CoordinateSpec(crs=crs)
    return CoordinateSpec(x=x, y=y, crs=crs)


@dataclass(slots=True)
class DataSource:
    """One tabular or geospatial source with explicit variable roles."""

    source_id: str
    data: pd.DataFrame
    variables: Sequence[VariableSpec]
    coordinates: CoordinateSpec
    metadata: Mapping[str, Any] | None = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise DataSourceError("DataSource source_id must be a non-empty string.")
        self.source_id = self.source_id.strip()

        if not isinstance(self.data, pd.DataFrame):
            raise DataSourceError(
                f"DataSource {self.source_id!r} requires a pandas DataFrame or GeoDataFrame."
            )
        if self.data.empty:
            raise DataSourceError(f"DataSource {self.source_id!r} contains no observations.")
        self.data = self.data.copy()
        duplicate_columns = sorted(
            {str(name) for name in self.data.columns[self.data.columns.duplicated()].tolist()}
        )
        if duplicate_columns:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} contains duplicate columns: {duplicate_columns}."
            )

        if not isinstance(self.coordinates, CoordinateSpec):
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} coordinates must be a CoordinateSpec."
            )
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} metadata must be a mapping or None."
            )
        self.metadata = dict(self.metadata or {})
        self.path = None if self.path is None else Path(self.path)

        try:
            specs = tuple(self.variables)
        except TypeError as exc:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} variables must be a sequence of VariableSpec objects."
            ) from exc
        if any(not isinstance(spec, VariableSpec) for spec in specs):
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} variables must contain only VariableSpec objects."
            )
        names = [spec.name for spec in specs]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} declares variables more than once: {duplicate_names}."
            )

        self.coordinates = self._validated_coordinates(self.coordinates)
        specs = self._with_coordinate_variables(specs)
        self._validate_variables(specs)
        self.variables = specs

    def _validated_coordinates(self, coordinates: CoordinateSpec) -> CoordinateSpec:
        is_geodataframe = isinstance(self.data, gpd.GeoDataFrame)

        if coordinates.uses_geometry:
            if not is_geodataframe:
                raise SchemaValidationError(
                    f"DataSource {self.source_id!r} must declare x and y columns unless its data is a GeoDataFrame."
                )
            geometry_name = getattr(self.data, "_geometry_column_name", None)
            if geometry_name is None or geometry_name not in self.data.columns:
                raise SchemaValidationError(
                    f"DataSource {self.source_id!r} has no active geometry column."
                )
            if self.data.geometry.isna().any() or self.data.geometry.is_empty.any():
                raise SchemaValidationError(
                    f"DataSource {self.source_id!r} contains missing or empty geometries."
                )
        else:
            missing = [name for name in coordinates.columns if name not in self.data.columns]
            if missing:
                raise SchemaValidationError(
                    f"DataSource {self.source_id!r} is missing coordinate columns: {missing}."
                )
            for name in coordinates.columns:
                if not is_numeric_dtype(self.data[name].dtype):
                    raise SchemaValidationError(
                        f"Coordinate column {name!r} in DataSource {self.source_id!r} must be numeric."
                    )
                if self.data[name].isna().any():
                    raise SchemaValidationError(
                        f"Coordinate column {name!r} in DataSource {self.source_id!r} contains missing values."
                    )

        frame_crs = CRS.from_user_input(self.data.crs) if is_geodataframe and self.data.crs is not None else None
        if coordinates.crs is not None and frame_crs is not None and coordinates.crs != frame_crs:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} has conflicting CRS declarations: "
                f"{coordinates.crs.to_string()} and {frame_crs.to_string()}."
            )
        if coordinates.crs is None and frame_crs is not None:
            return CoordinateSpec(x=coordinates.x, y=coordinates.y, crs=frame_crs)
        return coordinates

    def _with_coordinate_variables(
        self, specs: tuple[VariableSpec, ...]
    ) -> tuple[VariableSpec, ...]:
        by_name = {spec.name: spec for spec in specs}
        coordinate_columns = list(self.coordinates.columns)
        if self.coordinates.uses_geometry:
            geometry_name = getattr(self.data, "_geometry_column_name", None)
            if geometry_name is not None:
                coordinate_columns.append(geometry_name)

        for name in coordinate_columns:
            existing = by_name.get(name)
            if existing is not None and existing.role is not VariableRole.COORDINATE:
                raise SchemaValidationError(
                    f"Variable {name!r} is a declared coordinate in DataSource {self.source_id!r} "
                    f"but has role {existing.role.value!r}."
                )
            if existing is None:
                coordinate = VariableSpec(name=name, role=VariableRole.COORDINATE)
                specs += (coordinate,)
                by_name[name] = coordinate

        coordinate_names = set(coordinate_columns)
        unexpected = sorted(
            spec.name
            for spec in specs
            if spec.role is VariableRole.COORDINATE and spec.name not in coordinate_names
        )
        if unexpected:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} declares coordinate variables not present in its "
                f"CoordinateSpec: {unexpected}."
            )
        return specs

    def _validate_variables(self, specs: tuple[VariableSpec, ...]) -> None:
        missing = sorted(spec.name for spec in specs if spec.name not in self.data.columns)
        if missing:
            raise SchemaValidationError(
                f"DataSource {self.source_id!r} is missing declared variables: {missing}."
            )

        for spec in specs:
            if spec.role in (VariableRole.PREDICTOR, VariableRole.OUTCOME):
                if not is_numeric_dtype(self.data[spec.name].dtype):
                    raise SchemaValidationError(
                        f"{spec.role.value.capitalize()} variable {spec.name!r} in DataSource "
                        f"{self.source_id!r} must be numeric."
                    )

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
    ) -> "DataSource":
        """Load a CSV source without assigning scientific roles by name."""

        csv_path = Path(path)
        if not csv_path.is_file():
            raise DataSourceError(f"CSV input does not exist: {csv_path}.")
        try:
            frame = pd.read_csv(csv_path, **dict(read_csv_kwargs or {}))
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise DataSourceError(f"Could not read CSV input {csv_path}: {exc}") from exc
        return cls(
            source_id=source_id or csv_path.stem,
            data=frame,
            variables=variables,
            coordinates=_coordinate_spec(
                coordinates, x=x, y=y, crs=crs, geometry_backed=False
            ),
            metadata=metadata or {},
            path=csv_path,
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
    ) -> "DataSource":
        """Create a source from an in-memory pandas DataFrame."""

        return cls(
            source_id=source_id,
            data=dataframe,
            variables=variables,
            coordinates=_coordinate_spec(
                coordinates,
                x=x,
                y=y,
                crs=crs,
                geometry_backed=isinstance(dataframe, gpd.GeoDataFrame),
            ),
            metadata=metadata or {},
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
    ) -> "DataSource":
        """Create a source from a GeoDataFrame, defaulting to active geometry."""

        if not isinstance(geodataframe, gpd.GeoDataFrame):
            raise DataSourceError("from_geodataframe requires a geopandas GeoDataFrame.")
        return cls.from_dataframe(
            geodataframe,
            source_id=source_id,
            variables=variables,
            coordinates=coordinates,
            x=x,
            y=y,
            crs=crs,
            metadata=metadata,
        )

    @property
    def coordinate_spec(self) -> CoordinateSpec:
        """Alias exposing the source coordinate declaration."""

        return self.coordinates

    @property
    def variable_specs(self) -> tuple[VariableSpec, ...]:
        """Return the source's normalized variable declarations."""

        return tuple(self.variables)

    @property
    def crs(self) -> CRS | None:
        """Return this source's CRS, if declared or available from geometry."""

        return self.coordinates.crs

    @property
    def predictors(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variables if spec.role is VariableRole.PREDICTOR)

    @property
    def outcomes(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variables if spec.role is VariableRole.OUTCOME)

    @property
    def identifiers(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variables if spec.role is VariableRole.IDENTIFIER)

    @property
    def metadata_variables(self) -> tuple[VariableSpec, ...]:
        return tuple(spec for spec in self.variables if spec.role is VariableRole.METADATA)

    def variables_for_role(self, role: VariableRole | str) -> tuple[VariableSpec, ...]:
        """Return variables having an explicitly selected role."""

        try:
            selected_role = VariableRole(role)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Invalid variable role {role!r}.") from exc
        return tuple(spec for spec in self.variables if spec.role is selected_role)
