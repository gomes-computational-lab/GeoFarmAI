import numpy as np
import geopandas as gpd
from sklearn.neighbors import KDTree


def idw(points: gpd.GeoDataFrame, values: np.ndarray, targets: gpd.GeoDataFrame, k=8, power=2.0):
    # points/targets in meters CRS; values is 1D array
    tree = KDTree(np.c_[points.geometry.x, points.geometry.y])

    # IDW expects point targets; fall back to centroids for polygonal grids
    target_geom = targets.geometry
    if not (target_geom.geom_type == "Point").all():
        target_geom = target_geom.centroid

    coords = np.c_[target_geom.x, target_geom.y]
    dist, idx = tree.query(coords, k=min(k, len(points)))
    w = 1.0 / (dist ** power + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    est = (values[idx] * w).sum(axis=1)
    return est


def _buffer_mean(points: gpd.GeoDataFrame, value_column: str,
                 targets: gpd.GeoDataFrame, buffer_m: float):
    buffers = targets[["geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(buffer_m)
    joined = gpd.sjoin(
        points[[value_column, "geometry"]],
        buffers,
        how="inner",
        predicate="within",
    )
    return joined.groupby("index_right")[value_column].mean().reindex(targets.index)


def _validate_method(method: str) -> None:
    supported_methods = {"idw", "nearest", "buffer_mean"}
    if method not in supported_methods:
        if method == "kriging":
            raise ValueError(
                "Vector reconciliation method 'kriging' is not supported. "
                "Enable the raster workflow to use PyKrige ordinary kriging, "
                "or choose one of: idw, nearest, buffer_mean."
            )
        raise ValueError(
            f"Unsupported vector reconciliation method '{method}'. "
            "Choose one of: idw, nearest, buffer_mean."
        )

def reconcile_columns(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    variables: list[str],
    method="idw",
    buffer_m=15,
):
    """Apply the existing reconciliation algorithm to arbitrary variables."""

    _validate_method(method)
    out = grid.copy()
    for v in variables:
        if method == "nearest":
            joined = gpd.sjoin_nearest(out, points[[v, "geometry"]], how="left", distance_col=f"{v}_dist")
            out[v] = joined[v].values
        elif method == "buffer_mean":
            out[v] = _buffer_mean(points, v, out, buffer_m)
        else:  # IDW default
            out[v] = idw(points, points[v].values, out)
    return out


def populate_analysis_grid(
    predictor_points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    predictor_variables: list[str],
    *,
    outcome_points: gpd.GeoDataFrame | None = None,
    outcome_name: str | None = None,
    method="idw",
    buffer_m=15,
):
    """Reconcile predictors and an optional explicit outcome to one grid."""

    out = reconcile_columns(
        predictor_points,
        grid,
        predictor_variables,
        method=method,
        buffer_m=buffer_m,
    )
    if outcome_name is None:
        return out
    source = outcome_points if outcome_points is not None else predictor_points
    if outcome_name not in source.columns:
        raise ValueError(f"Configured outcome column {outcome_name!r} is missing.")
    reconciled_outcome = reconcile_columns(
        source,
        grid,
        [outcome_name],
        method=method,
        buffer_m=buffer_m,
    )
    out[outcome_name] = reconciled_outcome[outcome_name].values
    return out


def populate_grid(soil_utm: gpd.GeoDataFrame, yield_utm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                  soil_vars: list[str], method="idw", buffer_m=15):
    """Compatibility wrapper for the historical soil/yield API."""

    outcome = yield_utm if "yield" in yield_utm.columns else None
    return populate_analysis_grid(
        soil_utm,
        grid,
        soil_vars,
        outcome_points=outcome,
        outcome_name="yield" if outcome is not None else None,
        method=method,
        buffer_m=buffer_m,
    )
