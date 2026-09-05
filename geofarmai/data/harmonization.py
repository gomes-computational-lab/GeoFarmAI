"""Spatial harmonization for canonical GeoFarmAI field datasets.

This module is an adapter over GeoFarmAI's established CRS, grid, and
reconciliation functions. It does not implement a second interpolation
algorithm and it does not connect harmonized data to decomposition or
clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from core.crs import to_utm_auto
from core.grid import build_field_grid, choose_cell_size
from core.reconcile import populate_grid
from geofarmai.data.dataset import FieldDataset
from geofarmai.data.schema import VariableRole, VariableSpec
from geofarmai.data.source import DataSource
from geofarmai.exceptions import HarmonizationError


_COLUMN_NAMES = ("source_id", "variable_name")
_SUPPORTED_STRATEGIES = {"auto", "direct", "grid"}
_SUPPORTED_METHODS = {"idw", "nearest", "buffer_mean"}


class HarmonizationWarning(UserWarning):
    """Warning for measurable spatial coverage concerns."""


@dataclass(frozen=True, slots=True, order=True)
class VariableIdentity:
    """Unambiguous identity of one variable within one source."""

    source_id: str
    variable_name: str

    @property
    def tuple(self) -> tuple[str, str]:
        return self.source_id, self.variable_name


@dataclass(frozen=True, slots=True)
class VariableProvenance:
    """Record how one harmonized scientific variable was constructed."""

    identity: VariableIdentity
    role: VariableRole
    original_crs: str
    target_crs: str
    alignment_method: str
    grid_resolution: float | None
    interpolation_parameters: Mapping[str, Any]
    source_observations: int
    source_non_missing: int


@dataclass(frozen=True, slots=True)
class HarmonizedFieldDataset:
    """One spatial support with role-separated analysis variables.

    ``predictor_matrix`` and ``outcome_matrix`` use two-level columns
    ``(source_id, variable_name)`` so equal names from different sources are
    never overwritten. Geometry is authoritative. ``coordinate_array()``
    derives point coordinates, using polygon centroids for grid support in the
    same manner as the existing reconciliation implementation.
    """

    geometry: gpd.GeoSeries
    predictor_matrix: pd.DataFrame
    outcome_matrix: pd.DataFrame
    crs: CRS
    variable_provenance: Mapping[VariableIdentity, VariableProvenance]
    source_provenance: Mapping[str, Mapping[str, Any]]
    coverage: pd.DataFrame
    harmonization_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.geometry) == 0:
            raise HarmonizationError("Harmonization produced an empty spatial support.")
        if self.geometry.crs is None:
            raise HarmonizationError("Harmonized geometry must have a CRS.")
        parsed_crs = CRS.from_user_input(self.crs)
        if CRS.from_user_input(self.geometry.crs) != parsed_crs:
            raise HarmonizationError("Harmonized geometry and dataset CRS do not match.")
        expected = len(self.geometry)
        if len(self.predictor_matrix) != expected or len(self.outcome_matrix) != expected:
            raise HarmonizationError(
                "Harmonized predictors, outcomes, and geometry must have equal row counts."
            )
        if self.predictor_matrix.shape[1] == 0:
            raise HarmonizationError("Harmonization requires at least one predictor variable.")

        predictor_ids = set(self.predictor_identities)
        outcome_ids = set(self.outcome_identities)
        if predictor_ids & outcome_ids:
            raise HarmonizationError("An outcome variable cannot also appear in predictors.")

    @property
    def predictors(self) -> pd.DataFrame:
        """Alias for the aligned predictor matrix."""

        return self.predictor_matrix

    @property
    def outcomes(self) -> pd.DataFrame:
        """Alias for the aligned outcome matrix, which may have zero columns."""

        return self.outcome_matrix

    @property
    def predictor_identities(self) -> tuple[VariableIdentity, ...]:
        return tuple(VariableIdentity(*column) for column in self.predictor_matrix.columns)

    @property
    def outcome_identities(self) -> tuple[VariableIdentity, ...]:
        return tuple(VariableIdentity(*column) for column in self.outcome_matrix.columns)

    @property
    def predictor_names(self) -> tuple[str, ...]:
        names = self.display_names
        return tuple(names[identity] for identity in self.predictor_identities)

    @property
    def outcome_names(self) -> tuple[str, ...]:
        names = self.display_names
        return tuple(names[identity] for identity in self.outcome_identities)

    @property
    def actual_strategy(self) -> str:
        return str(self.harmonization_metadata["actual_strategy"])

    @property
    def outcome_arrays(self) -> dict[VariableIdentity, np.ndarray]:
        return {
            identity: self.outcome_matrix[identity.tuple].to_numpy(copy=True)
            for identity in self.outcome_identities
        }

    @property
    def display_names(self) -> dict[VariableIdentity, str]:
        """Return short names only where they are unambiguous."""

        identities = self.predictor_identities + self.outcome_identities
        counts: dict[str, int] = {}
        for identity in identities:
            counts[identity.variable_name] = counts.get(identity.variable_name, 0) + 1
        return {
            identity: (
                identity.variable_name
                if counts[identity.variable_name] == 1
                else f"{identity.source_id}:{identity.variable_name}"
            )
            for identity in identities
        }

    def coordinate_array(self) -> np.ndarray:
        """Extract x/y coordinates from authoritative geometry.

        Direct supports contain points. Grid supports contain polygons, for
        which centroids match the existing IDW target-coordinate convention.
        No physical coordinate columns are materialized.
        """

        geometry = self.geometry
        if not (geometry.geom_type == "Point").all():
            geometry = geometry.centroid
        return np.column_stack((geometry.x.to_numpy(), geometry.y.to_numpy()))


def harmonize(
    field_dataset: FieldDataset,
    *,
    strategy: str = "auto",
    method: str = "idw",
    target_crs: CRS | str | int | None = None,
    location_tolerance: float = 1e-6,
    cell_size: float | None = None,
    min_cell_size: float = 3.0,
    max_cell_size: float = 30.0,
    buffer_m: float = 15.0,
) -> HarmonizedFieldDataset:
    """Align one or many canonical sources to a deterministic spatial support.

    Target-CRS policy is deterministic: an explicitly requested metric CRS is
    used; otherwise a shared projected CRS with metre units is preserved; all
    other cases use the existing mean-coordinate UTM selection. Consequently,
    Euclidean matching and interpolation never run in longitude/latitude
    degrees.

    ``strategy="auto"`` uses direct rows for one source, one-to-one alignment
    for colocated sources, a predictor reference support when only outcomes
    differ spatially, and the existing grid/reconciliation workflow when
    predictor sources differ. The selected strategy is recorded in the result.
    """

    if not isinstance(field_dataset, FieldDataset):
        raise HarmonizationError("harmonize() requires a FieldDataset.")
    if strategy not in _SUPPORTED_STRATEGIES:
        raise HarmonizationError(
            f"Unsupported harmonization strategy {strategy!r}. "
            f"Choose one of: {', '.join(sorted(_SUPPORTED_STRATEGIES))}."
        )
    if method == "kriging":
        raise HarmonizationError(
            "Vector harmonization method 'kriging' is not supported by the existing "
            "GeoFarmAI vector workflow. Use the raster kriging workflow or choose "
            "idw, nearest, or buffer_mean."
        )
    if method not in _SUPPORTED_METHODS:
        raise HarmonizationError(
            f"Unsupported harmonization method {method!r}. "
            f"Choose one of: {', '.join(sorted(_SUPPORTED_METHODS))}."
        )
    if not np.isfinite(location_tolerance) or location_tolerance < 0:
        raise HarmonizationError("location_tolerance must be a finite non-negative value.")
    if cell_size is not None and (not np.isfinite(cell_size) or cell_size <= 0):
        raise HarmonizationError("cell_size must be a finite positive value.")
    if min_cell_size <= 0 or max_cell_size <= 0 or min_cell_size > max_cell_size:
        raise HarmonizationError(
            "min_cell_size and max_cell_size must be positive, with min <= max."
        )
    if buffer_m < 0:
        raise HarmonizationError("buffer_m must be non-negative.")

    if not any(source.predictors for source in field_dataset.sources):
        raise HarmonizationError("Harmonization requires at least one predictor variable.")

    original = {source.source_id: _source_to_geodataframe(source) for source in field_dataset.sources}
    selected_crs, crs_policy = _select_target_crs(original, target_crs)
    projected = {
        source_id: frame.to_crs(selected_crs).reset_index(drop=True)
        for source_id, frame in original.items()
    }

    analysis_sources = [
        source for source in field_dataset.sources if source.predictors or source.outcomes
    ]
    predictor_sources = [source for source in analysis_sources if source.predictors]
    reference = predictor_sources[0]
    reference_frame = projected[reference.source_id]

    indexers = {
        source.source_id: _location_indexer(
            reference_frame, projected[source.source_id], location_tolerance
        )
        for source in analysis_sources
    }
    predictors_colocated = all(
        indexers[source.source_id] is not None for source in predictor_sources
    )
    all_colocated = all(indexers[source.source_id] is not None for source in analysis_sources)

    if strategy == "direct" and not all_colocated:
        raise HarmonizationError(
            "Direct harmonization requires all scientific sources to have equivalent "
            f"locations within {location_tolerance} metres."
        )

    if strategy == "grid":
        actual_strategy = "grid_reconciliation"
    elif strategy == "direct":
        actual_strategy = "direct_reuse" if len(analysis_sources) == 1 else "direct_alignment"
    elif len(analysis_sources) == 1:
        actual_strategy = "direct_reuse"
    elif all_colocated:
        actual_strategy = "direct_alignment"
    elif predictors_colocated:
        actual_strategy = "direct_alignment_with_reconciliation"
    else:
        actual_strategy = "grid_reconciliation"

    overlap_notes: list[str] = []
    resolved_cell_size: float | None = None
    if actual_strategy == "grid_reconciliation":
        overlap_notes = _warn_for_nonoverlap(predictor_sources, projected)
        combined_predictors = gpd.GeoDataFrame(
            geometry=pd.concat(
                [projected[source.source_id].geometry for source in predictor_sources],
                ignore_index=True,
            ),
            crs=selected_crs,
        )
        resolved_cell_size = (
            float(cell_size)
            if cell_size is not None
            else choose_cell_size(
                combined_predictors, min_cell=min_cell_size, max_cell=max_cell_size
            )
        )
        grid = build_field_grid(combined_predictors, combined_predictors, resolved_cell_size)
        if grid.empty:
            raise HarmonizationError(
                "Grid harmonization produced no cells; check source extent and cell_size."
            )
        support = gpd.GeoSeries(grid.geometry.array, crs=selected_crs).reset_index(drop=True)
    else:
        support = gpd.GeoSeries(reference_frame.geometry.array, crs=selected_crs).reset_index(
            drop=True
        )

    predictor_values: list[np.ndarray] = []
    outcome_values: list[np.ndarray] = []
    predictor_ids: list[VariableIdentity] = []
    outcome_ids: list[VariableIdentity] = []
    provenance: dict[VariableIdentity, VariableProvenance] = {}

    for source in analysis_sources:
        frame = projected[source.source_id]
        indexer = indexers[source.source_id]
        for spec in source.predictors:
            identity = VariableIdentity(source.source_id, spec.name)
            values, alignment_method, parameters = _aligned_values(
                source,
                frame,
                spec,
                support,
                actual_strategy,
                indexer,
                method,
                buffer_m,
                location_tolerance,
            )
            predictor_ids.append(identity)
            predictor_values.append(values)
            provenance[identity] = _variable_provenance(
                identity,
                spec,
                source,
                selected_crs,
                alignment_method,
                resolved_cell_size,
                parameters,
            )
        for spec in source.outcomes:
            identity = VariableIdentity(source.source_id, spec.name)
            values, alignment_method, parameters = _aligned_values(
                source,
                frame,
                spec,
                support,
                actual_strategy,
                indexer,
                method,
                buffer_m,
                location_tolerance,
            )
            outcome_ids.append(identity)
            outcome_values.append(values)
            provenance[identity] = _variable_provenance(
                identity,
                spec,
                source,
                selected_crs,
                alignment_method,
                resolved_cell_size,
                parameters,
            )

    row_index = pd.RangeIndex(len(support), name="analysis_row")
    predictors = _matrix(predictor_values, predictor_ids, row_index)
    outcomes = _matrix(outcome_values, outcome_ids, row_index)
    coverage = _coverage(predictors, outcomes, provenance)
    source_provenance = {
        source.source_id: {
            "source_id": source.source_id,
            "original_crs": source.crs.to_string() if source.crs is not None else None,
            "target_crs": selected_crs.to_string(),
            "transformed": source.crs != selected_crs,
            "observations": len(source.data),
            "path": None if source.path is None else str(source.path),
            "metadata": dict(source.metadata),
        }
        for source in field_dataset.sources
    }
    metadata = {
        "requested_strategy": strategy,
        "actual_strategy": actual_strategy,
        "reconciliation_method": method,
        "target_crs_policy": crs_policy,
        "location_tolerance_m": float(location_tolerance),
        "grid_resolution_m": resolved_cell_size,
        "buffer_m": float(buffer_m) if method == "buffer_mean" else None,
        "overlap_warnings": tuple(overlap_notes),
    }
    return HarmonizedFieldDataset(
        geometry=support,
        predictor_matrix=predictors,
        outcome_matrix=outcomes,
        crs=selected_crs,
        variable_provenance=provenance,
        source_provenance=source_provenance,
        coverage=coverage,
        harmonization_metadata=metadata,
    )


def _source_to_geodataframe(source: DataSource) -> gpd.GeoDataFrame:
    if source.crs is None:
        raise HarmonizationError(
            f"DataSource {source.source_id!r} requires a CRS for spatial harmonization."
        )

    data = source.data
    geometry_name = getattr(data, "_geometry_column_name", None)
    if isinstance(data, gpd.GeoDataFrame) and geometry_name in data.columns:
        frame = data.copy()
        if frame.crs is None:
            frame = frame.set_crs(source.crs)
    else:
        if source.coordinates.uses_geometry:
            raise HarmonizationError(
                f"DataSource {source.source_id!r} has no active geometry or x/y coordinates."
            )
        frame = gpd.GeoDataFrame(
            data.copy(),
            geometry=gpd.points_from_xy(
                data[source.coordinates.x], data[source.coordinates.y]
            ),
            crs=source.crs,
        )

    if not (frame.geometry.geom_type == "Point").all():
        raise HarmonizationError(
            f"DataSource {source.source_id!r} uses non-point geometry. The current "
            "GeoFarmAI reconciliation algorithms require point observations."
        )
    return frame


def _select_target_crs(
    frames: Mapping[str, gpd.GeoDataFrame],
    requested: CRS | str | int | None,
) -> tuple[CRS, str]:
    if requested is not None:
        try:
            parsed = CRS.from_user_input(requested)
        except (CRSError, TypeError, ValueError) as exc:
            raise HarmonizationError(f"Invalid target CRS {requested!r}.") from exc
        if not _is_metric_projected(parsed):
            raise HarmonizationError(
                "target_crs must be a projected CRS whose horizontal units are metres."
            )
        return parsed, "explicit_metric_crs"

    crs_values = [CRS.from_user_input(frame.crs) for frame in frames.values()]
    first = crs_values[0]
    if all(crs == first for crs in crs_values) and _is_metric_projected(first):
        return first, "preserved_shared_metric_crs"

    geographic_parts = [frame.geometry.to_crs("EPSG:4326") for frame in frames.values()]
    combined = gpd.GeoDataFrame(
        geometry=pd.concat(geographic_parts, ignore_index=True), crs="EPSG:4326"
    )
    _, epsg = to_utm_auto(combined)
    return CRS.from_user_input(epsg), "existing_auto_utm"


def _is_metric_projected(crs: CRS) -> bool:
    if not crs.is_projected or len(crs.axis_info) < 2:
        return False
    return all(
        axis.unit_name.lower() in {"metre", "meter"}
        or np.isclose(axis.unit_conversion_factor, 1.0)
        for axis in crs.axis_info[:2]
    )


def _location_indexer(
    reference: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    tolerance: float,
) -> np.ndarray | None:
    if len(reference) != len(candidate):
        return None
    reference_xy = np.column_stack((reference.geometry.x, reference.geometry.y))
    candidate_xy = np.column_stack((candidate.geometry.x, candidate.geometry.y))
    distances = cdist(reference_xy, candidate_xy)
    rows, columns = linear_sum_assignment(distances)
    if len(rows) != len(reference) or np.any(distances[rows, columns] > tolerance):
        return None
    indexer = np.empty(len(reference), dtype=int)
    indexer[rows] = columns
    return indexer


def _aligned_values(
    source: DataSource,
    frame: gpd.GeoDataFrame,
    spec: VariableSpec,
    support: gpd.GeoSeries,
    actual_strategy: str,
    indexer: np.ndarray | None,
    method: str,
    buffer_m: float,
    location_tolerance: float,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if actual_strategy in {"direct_reuse", "direct_alignment"}:
        if indexer is None:
            raise HarmonizationError(
                f"DataSource {source.source_id!r} is not colocated with the direct support."
            )
        alignment = "direct" if actual_strategy == "direct_reuse" else "direct_alignment"
        parameters = (
            {} if alignment == "direct" else {"location_tolerance_m": location_tolerance}
        )
        return frame[spec.name].to_numpy(dtype=float)[indexer], alignment, parameters

    if actual_strategy == "direct_alignment_with_reconciliation" and indexer is not None:
        return (
            frame[spec.name].to_numpy(dtype=float)[indexer],
            "direct_alignment",
            {"location_tolerance_m": location_tolerance},
        )

    values = _reconcile_variable(frame, spec.name, support, method, buffer_m)
    if method == "idw":
        parameters: dict[str, Any] = {"k": 8, "power": 2.0}
    elif method == "buffer_mean":
        parameters = {"buffer_m": float(buffer_m)}
    else:
        parameters = {}
    return values, method, parameters


def _reconcile_variable(
    source: gpd.GeoDataFrame,
    variable: str,
    support: gpd.GeoSeries,
    method: str,
    buffer_m: float,
) -> np.ndarray:
    value_column = "__geofarmai_value__"
    samples = source[[variable, "geometry"]].rename(columns={variable: value_column})
    targets = gpd.GeoDataFrame(
        {"cell_id": np.arange(len(support))},
        geometry=gpd.GeoSeries(support.array, crs=support.crs),
        crs=support.crs,
    )
    empty_outcome = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([], dtype="geometry", crs=support.crs), crs=support.crs
    )
    try:
        reconciled = populate_grid(
            samples,
            empty_outcome,
            targets,
            [value_column],
            method=method,
            buffer_m=buffer_m,
        )
    except ValueError as exc:
        raise HarmonizationError(
            f"Could not reconcile variable {variable!r} with method {method!r}: {exc}"
        ) from exc
    if len(reconciled) != len(targets):
        raise HarmonizationError(
            f"Reconciliation of variable {variable!r} returned ambiguous duplicate matches."
        )
    return reconciled[value_column].to_numpy(dtype=float)


def _variable_provenance(
    identity: VariableIdentity,
    spec: VariableSpec,
    source: DataSource,
    target_crs: CRS,
    alignment_method: str,
    grid_resolution: float | None,
    parameters: Mapping[str, Any],
) -> VariableProvenance:
    values = source.data[spec.name]
    return VariableProvenance(
        identity=identity,
        role=spec.role,
        original_crs=source.crs.to_string(),
        target_crs=target_crs.to_string(),
        alignment_method=alignment_method,
        grid_resolution=grid_resolution,
        interpolation_parameters=dict(parameters),
        source_observations=len(values),
        source_non_missing=int(values.notna().sum()),
    )


def _matrix(
    values: list[np.ndarray],
    identities: list[VariableIdentity],
    index: pd.Index,
) -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [identity.tuple for identity in identities], names=_COLUMN_NAMES
    )
    if not values:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    return pd.DataFrame(np.column_stack(values), index=index, columns=columns)


def _coverage(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    provenance: Mapping[VariableIdentity, VariableProvenance],
) -> pd.DataFrame:
    combined = pd.concat([predictors, outcomes], axis=1)
    rows: list[dict[str, float | int]] = []
    identities: list[tuple[str, str]] = []
    for column in combined.columns:
        identity = VariableIdentity(*column)
        values = combined[column]
        non_missing = int(values.notna().sum())
        total = len(values)
        source = provenance[identity]
        identities.append(identity.tuple)
        rows.append(
            {
                "source_observations": source.source_observations,
                "source_non_missing": source.source_non_missing,
                "harmonized_observations": total,
                "harmonized_non_missing": non_missing,
                "harmonized_missing": total - non_missing,
                "coverage_fraction": non_missing / total if total else np.nan,
            }
        )
    index = pd.MultiIndex.from_tuples(identities, names=_COLUMN_NAMES)
    return pd.DataFrame(rows, index=index)


def _warn_for_nonoverlap(
    sources: list[DataSource],
    frames: Mapping[str, gpd.GeoDataFrame],
) -> list[str]:
    messages: list[str] = []
    for left, right in combinations(sources, 2):
        left_hull = _geometry_union(frames[left.source_id].geometry).convex_hull
        right_hull = _geometry_union(frames[right.source_id].geometry).convex_hull
        if not left_hull.intersects(right_hull):
            message = (
                f"Predictor sources {left.source_id!r} and {right.source_id!r} have "
                "no intersecting spatial extent; reconciliation may extrapolate."
            )
            warnings.warn(message, HarmonizationWarning, stacklevel=3)
            messages.append(message)
    return messages


def _geometry_union(geometry: gpd.GeoSeries):
    """Use the non-deprecated API while retaining GeoPandas 0.14 support."""

    union_all = getattr(geometry, "union_all", None)
    return union_all() if union_all is not None else geometry.unary_union
