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


def populate_grid(soil_utm: gpd.GeoDataFrame, yield_utm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                  soil_vars: list[str], method="idw", buffer_m=15):
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

    out = grid.copy()
    # soil variables
    for v in soil_vars:
        if method == "nearest":
            joined = gpd.sjoin_nearest(out, soil_utm[[v, "geometry"]], how="left", distance_col=f"{v}_dist")
            out[v] = joined[v].values
        elif method == "buffer_mean":
            out[v] = _buffer_mean(soil_utm, v, out, buffer_m)
        else:  # IDW default
            out[v] = idw(soil_utm, soil_utm[v].values, out)
    # yield
    if "yield" in yield_utm.columns:
        if method == "nearest":
            joined = gpd.sjoin_nearest(out, yield_utm[["yield","geometry"]], how="left", distance_col="yield_dist")
            out["yield"] = joined["yield"].values
        elif method == "buffer_mean":
            out["yield"] = _buffer_mean(yield_utm, "yield", out, buffer_m)
        else:
            out["yield"] = idw(yield_utm, yield_utm["yield"].values, out)
    return out
