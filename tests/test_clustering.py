import numpy as np
import pytest

from geofarmai.clustering import SUPPORTED_ALGORITHMS, cluster
from geofarmai.exceptions import ConfigurationError


X = np.array(
    [
        [-2.1, -2.0], [-2.0, -1.8], [-1.8, -2.1],
        [2.0, 2.1], [2.2, 1.9], [1.8, 2.0],
    ],
    dtype=float,
)


def partition(labels):
    labels = np.asarray(labels)
    return labels[:, None] == labels[None, :]


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
def test_supported_algorithms_are_reproducible(algorithm):
    first, first_metrics = cluster(X, algorithm, 2, random_state=42)
    second, second_metrics = cluster(X, algorithm, 2, random_state=42)
    assert np.array_equal(partition(first), partition(second))
    assert np.isfinite(first_metrics["asc"])
    assert np.isfinite(first_metrics["ch_score"])
    if algorithm == "fcm":
        assert "fpc" in first_metrics
    assert first_metrics == second_metrics


def test_unknown_algorithm_raises():
    with pytest.raises(ConfigurationError, match="Unsupported clustering"):
        cluster(X, "unknown", 2)


@pytest.mark.parametrize("k", [1, 6, 7])
def test_invalid_k_raises(k):
    with pytest.raises(ConfigurationError, match="k must"):
        cluster(X, "kmeans", k)
