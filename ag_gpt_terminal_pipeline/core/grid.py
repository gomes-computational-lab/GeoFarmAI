import numpy as np
import geopandas as gpd
from shapely.geometry import box

def median_nn_distance(gdf: gpd.GeoDataFrame) -> float:
    # fast approx via kdtree if desired; here brute-force for clarity
    coords = np.c_[gdf.geometry.x.values, gdf.geometry.y.values]
    if len(coords) < 2: return 5.0
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(coords)
    dists, _ = nn.kneighbors(coords)
    # dists[:,0] is 0 to self; take the neighbor at index 1
    return float(np.median(dists[:,1]))

def choose_cell_size(unioned_points: gpd.GeoDataFrame, min_cell=3, max_cell=30) -> float:
    d_med = median_nn_distance(unioned_points)
    cell = 0.5 * d_med
    return float(np.clip(cell, min_cell, max_cell))

def make_grid(bounds, cell):
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin, xmax + cell, cell)
    ys = np.arange(ymin, ymax + cell, cell)
    polys = []
    for x0 in xs[:-1]:
        for y0 in ys[:-1]:
            polys.append(box(x0, y0, x0 + cell, y0 + cell))
    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:32600")  # will overwrite crs from a template later

def build_field_grid(soil_utm: gpd.GeoDataFrame, yield_utm: gpd.GeoDataFrame, cell_m: float):
    field_bounds = soil_utm.total_bounds
    field_bounds = np.array([
        min(field_bounds[0], yield_utm.total_bounds[0]),
        min(field_bounds[1], yield_utm.total_bounds[1]),
        max(field_bounds[2], yield_utm.total_bounds[2]),
        max(field_bounds[3], yield_utm.total_bounds[3]),
    ])
    grid = make_grid(field_bounds, cell_m)
    grid.set_crs(soil_utm.crs, inplace=True, allow_override=True)
    grid["cell_id"] = np.arange(len(grid))
    return grid