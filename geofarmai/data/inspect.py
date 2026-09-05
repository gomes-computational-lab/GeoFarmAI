"""Non-LLM schema inspection for prospective GeoFarmAI inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_numeric_dtype

from geofarmai.exceptions import DataSourceError


_X_NAMES = frozenset({"longitude", "lon", "lng", "x", "easting"})
_Y_NAMES = frozenset({"latitude", "lat", "y", "northing"})


def _normalized_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True, slots=True)
class CoordinateCandidates:
    """Possible x/y coordinate columns; no scientific role is assigned."""

    x: tuple[str, ...]
    y: tuple[str, ...]

    @property
    def is_unambiguous(self) -> bool:
        return len(self.x) == 1 and len(self.y) == 1 and self.x[0] != self.y[0]

    @property
    def unambiguous_pair(self) -> tuple[str, str] | None:
        return (self.x[0], self.y[0]) if self.is_unambiguous else None


@dataclass(frozen=True, slots=True)
class SchemaInspection:
    """Descriptive facts about a table, with coordinate candidates only."""

    columns: tuple[str, ...]
    dtypes: dict[str, str]
    numeric_columns: tuple[str, ...]
    missing_counts: dict[str, int]
    missing_fractions: dict[str, float]
    coordinate_candidates: CoordinateCandidates
    row_count: int
    geometry_column: str | None = None


def coordinate_candidates(dataframe: pd.DataFrame) -> CoordinateCandidates:
    """Find common coordinate-name candidates without assigning variable roles."""

    x_candidates: list[str] = []
    y_candidates: list[str] = []
    for column in dataframe.columns:
        name = str(column)
        normalized = _normalized_name(column)
        if normalized in _X_NAMES:
            x_candidates.append(name)
        if normalized in _Y_NAMES:
            y_candidates.append(name)
    return CoordinateCandidates(tuple(x_candidates), tuple(y_candidates))


def inspect_schema(
    data: pd.DataFrame | str | Path,
    *,
    read_csv_kwargs: Mapping[str, Any] | None = None,
) -> SchemaInspection:
    """Inspect a DataFrame, GeoDataFrame, or CSV without scientific inference."""

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise DataSourceError(f"CSV input does not exist: {path}.")
        try:
            dataframe = pd.read_csv(path, **dict(read_csv_kwargs or {}))
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            raise DataSourceError(f"Could not inspect CSV input {path}: {exc}") from exc
    elif isinstance(data, pd.DataFrame):
        dataframe = data
    else:
        raise DataSourceError("inspect_schema requires a DataFrame, GeoDataFrame, or CSV path.")

    columns = tuple(str(column) for column in dataframe.columns)
    dtypes = {str(column): str(dataframe[column].dtype) for column in dataframe.columns}
    numeric = tuple(
        str(column) for column in dataframe.columns if is_numeric_dtype(dataframe[column].dtype)
    )
    missing_counts = {
        str(column): int(dataframe[column].isna().sum()) for column in dataframe.columns
    }
    denominator = len(dataframe)
    missing_fractions = {
        name: (count / denominator if denominator else 0.0)
        for name, count in missing_counts.items()
    }
    geometry_column = (
        getattr(dataframe, "_geometry_column_name", None)
        if isinstance(dataframe, gpd.GeoDataFrame)
        else None
    )
    return SchemaInspection(
        columns=columns,
        dtypes=dtypes,
        numeric_columns=numeric,
        missing_counts=missing_counts,
        missing_fractions=missing_fractions,
        coordinate_candidates=coordinate_candidates(dataframe),
        row_count=denominator,
        geometry_column=geometry_column,
    )
