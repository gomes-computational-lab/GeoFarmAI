import numpy as np
import libpysal
from esda.moran import Moran

def knn_weights(gdf, k=8):
    geom = gdf.geometry
    if not (geom.geom_type == "Point").all():
        geom = geom.centroid
    coords = np.column_stack((geom.x, geom.y))
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = 'r'
    return w

def morans_i(gdf, w, columns):
    out = {}
    for c in columns:
        out[c] = float(Moran(gdf[c].values, w).I)
    return out
