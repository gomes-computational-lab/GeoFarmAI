from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import calinski_harabasz_score, silhouette_score

pytest.importorskip("skfuzzy")
from core.cluster import run_agglomerative, run_fcm, run_gmm, run_kmeans

from conftest import same_partition


@pytest.mark.parametrize(
    ("runner", "kwargs"),
    [
        (run_kmeans, {"random_state": 42}),
        (run_gmm, {"random_state": 42}),
        (run_fcm, {"random_state": 42}),
        (run_agglomerative, {}),
    ],
)
def test_supported_clusterers_find_three_separated_groups(separated_matrix, runner, kwargs):
    labels, metrics = runner(separated_matrix, 3, **kwargs)

    assert len(labels) == len(separated_matrix)
    assert len(np.unique(labels)) == 3
    assert metrics["asc"] == pytest.approx(silhouette_score(separated_matrix, labels), rel=1e-12)
    assert metrics["ch_score"] == pytest.approx(calinski_harabasz_score(separated_matrix, labels), rel=1e-12)
    assert metrics["asc"] > 0.9
    if runner is run_fcm:
        assert 0.0 <= metrics["fpc"] <= 1.0


@pytest.mark.parametrize(
    "runner",
    [run_kmeans, run_gmm, run_fcm],
)
def test_seeded_clusterers_are_partition_reproducible(separated_matrix, runner):
    first, first_metrics = runner(separated_matrix, 3, random_state=1337)
    second, second_metrics = runner(separated_matrix, 3, random_state=1337)

    assert same_partition(first, second)
    assert first_metrics == pytest.approx(second_metrics, rel=1e-12, abs=1e-12)


def test_silhouette_subsampling_is_deterministic():
    rng = np.random.default_rng(123)
    matrix = np.vstack([rng.normal(-2.0, 0.2, (100, 2)), rng.normal(2.0, 0.2, (100, 2))])

    _, first = run_kmeans(matrix, 2, random_state=42, sample_size=50)
    _, second = run_kmeans(matrix, 2, random_state=42, sample_size=50)

    assert first["asc"] == pytest.approx(second["asc"], rel=0.0, abs=0.0)
    assert first["ch_score"] == pytest.approx(second["ch_score"], rel=0.0, abs=0.0)
