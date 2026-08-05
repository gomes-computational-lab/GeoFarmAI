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

def populate_grid(soil_utm: gpd.GeoDataFrame, yield_utm: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                  soil_vars: list[str], method="idw", buffer_m=15):
    out = grid.copy()
    # soil variables
    for v in soil_vars:
        if method == "nearest":
            joined = gpd.sjoin_nearest(out, soil_utm[[v, "geometry"]], how="left", distance_col=f"{v}_dist")
            out[v] = joined[v].values
        elif method == "buffer_mean":
            buff = out.copy()
            buff["geometry"] = buff.geometry.buffer(buffer_m)
            joined = gpd.sjoin(soil_utm[[v, "geometry"]], buff, how="right", predicate="within")
            out[v] = joined.groupby("index_right")[v].mean()
        else:  # IDW default
            out[v] = idw(soil_utm, soil_utm[v].values, out)
    # yield
    if "yield" in yield_utm.columns:
        if method == "nearest":
            joined = gpd.sjoin_nearest(out, yield_utm[["yield","geometry"]], how="left", distance_col="yield_dist")
            out["yield"] = joined["yield"].values
        elif method == "buffer_mean":
            buff = out.copy()
            buff["geometry"] = buff.geometry.buffer(buffer_m)
            joined = gpd.sjoin(yield_utm[["yield","geometry"]], buff, how="right", predicate="within")
            out["yield"] = joined.groupby("index_right")["yield"].mean()
        else:
            out["yield"] = idw(yield_utm, yield_utm["yield"].values, out)
    return out
