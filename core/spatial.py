import numpy as np

from geofarmai.exceptions import OptionalDependencyError


def _spatial_dependencies():
    try:
        import libpysal
        from esda.moran import Moran
    except ImportError as exc:
        raise OptionalDependencyError(
            "Spatial-weight and Moran utilities require optional dependencies. "
            "Install GeoFarmAI with `pip install \"geofarmai[spatial]\"`."
        ) from exc
    return libpysal, Moran

def knn_weights(gdf, k=8):
    libpysal, _ = _spatial_dependencies()
    geom = gdf.geometry
    if not (geom.geom_type == "Point").all():
        geom = geom.centroid
    coords = np.column_stack((geom.x, geom.y))
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = 'r'
    return w

def morans_i(gdf, w, columns):
    _, Moran = _spatial_dependencies()
    out = {}
    for c in columns:
        out[c] = float(Moran(gdf[c].values, w).I)
    return out
