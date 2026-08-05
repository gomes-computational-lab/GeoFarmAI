import os

import numpy as np
import pytest
from sklearn.neighbors import kneighbors_graph

from geofarmai.raster import raster_gridsearch


def test_parallel_raster_gridsearch_matches_configured_candidate():
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except (OSError, PermissionError):
        pytest.skip("The execution environment does not permit multiprocessing semaphores.")
    scores = np.array(
        [[-2.0, -2.0], [-1.8, -2.1], [-2.1, -1.9], [-1.9, -1.8], [-2.2, -2.0], [-1.7, -2.2],
         [2.0, 2.0], [1.8, 2.1], [2.1, 1.9], [1.9, 1.8], [2.2, 2.0], [1.7, 2.2]],
        dtype=float,
    )
    connectivity = kneighbors_graph(scores, n_neighbors=2, mode="connectivity", include_self=False)
    yield_array = np.arange(12, dtype=float).reshape(3, 4)
    valid_mask = np.ones(12, dtype=bool)
    cfg = {
        "clustering": {"algorithms": ["kmeans"], "k_values": [2], "seeds": [42]},
        "raster": {"parallel_gridsearch": True, "n_jobs": 2, "silhouette_sample_size": 12},
    }

    best, leaderboard, selections = raster_gridsearch(
        scores, yield_array, valid_mask, connectivity, cfg
    )

    assert len(leaderboard) == 1
    assert best["metrics"]["algo"] == "kmeans"
    assert selections["yield_variance"] is best
