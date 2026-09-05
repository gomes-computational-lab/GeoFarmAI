from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geofarmai import (
    FieldDataset,
    GeoFarmModel,
    GeoFarmResult,
    ModelConfigurationError,
    delineate_zones,
    harmonize,
)

from conftest import same_partition


CRS = "EPSG:32615"


def _field(*, outcome: str | None = None) -> FieldDataset:
    group = np.repeat([0.0, 5.0, 10.0], 4)
    frame = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.tile([0.0, 1.0, 0.0, 1.0], 3),
            "EC": group + np.tile([0.0, 0.1, -0.1, 0.05], 3),
            "moisture": group + np.tile([0.2, -0.1, 0.1, -0.2], 3),
        }
    )
    if outcome is not None:
        frame[outcome] = np.repeat([10.0, 50.0, 90.0], 4) + np.tile(
            [0.0, 1.0, -1.0, 0.5], 3
        )
    return FieldDataset.from_dataframe(
        frame,
        predictors=["EC", "moisture"],
        outcome=outcome,
        coordinates=("x", "y"),
        crs=CRS,
    )


def _assert_results_equivalent(direct: GeoFarmResult, convenient: GeoFarmResult):
    assert isinstance(convenient, GeoFarmResult)
    assert len(convenient.candidate_solutions) == len(direct.candidate_solutions)
    assert convenient.selection_metric == direct.selection_metric
    assert convenient.selection_outcome == direct.selection_outcome
    assert convenient.selection_direction == direct.selection_direction
    assert convenient.configuration == direct.configuration

    direct_selected = (
        None
        if direct.selected_solution is None
        else direct.selected_solution.solution_id
    )
    convenient_selected = (
        None
        if convenient.selected_solution is None
        else convenient.selected_solution.solution_id
    )
    assert convenient_selected == direct_selected

    for expected, actual in zip(
        direct.candidate_solutions, convenient.candidate_solutions
    ):
        assert actual.solution_id == expected.solution_id
        assert same_partition(actual.labels, expected.labels)
        assert actual.internal_metrics == pytest.approx(
            expected.internal_metrics, rel=1e-12, abs=1e-12
        )
        assert actual.outcome_metrics == expected.outcome_metrics

    if direct.zone_labels is None:
        assert convenient.zone_labels is None
    else:
        assert same_partition(convenient.zone_labels, direct.zone_labels)
    pd.testing.assert_frame_equal(convenient.leaderboard, direct.leaderboard)


@pytest.mark.parametrize(
    ("outcome", "options"),
    [
        (
            None,
            {
                "decomposition": "none",
                "clustering": ["kmeans", "gmm"],
                "k": 3,
                "selection": "silhouette",
                "random_state": 71,
            },
        ),
        (
            "yield",
            {
                "decomposition": "pca",
                "clustering": ["kmeans", "gmm"],
                "k": range(2, 4),
                "selection": "variance_reduction",
                "selection_outcome": "yield",
                "random_state": 71,
            },
        ),
        (
            "yield",
            {
                "decomposition": "none",
                "clustering": "kmeans",
                "k": range(2, 5),
                "selection": None,
                "random_state": 71,
            },
        ),
    ],
)
def test_delineate_zones_matches_model_fit_for_canonical_scenarios(
    outcome, options
):
    data = _field(outcome=outcome)

    direct = GeoFarmModel(**options).fit(data)
    convenient = delineate_zones(data, **options)

    _assert_results_equivalent(direct, convenient)


def test_delineate_zones_accepts_already_harmonized_canonical_input():
    data = harmonize(_field())
    options = {
        "decomposition": "none",
        "clustering": "kmeans",
        "k": 3,
        "selection": "calinski_harabasz",
        "random_state": 29,
    }

    direct = GeoFarmModel(**options).fit(data)
    convenient = delineate_zones(data, **options)

    _assert_results_equivalent(direct, convenient)


@pytest.mark.parametrize(
    ("outcome", "options"),
    [
        (
            None,
            {
                "decomposition": "none",
                "k": 2,
                "selection": "variance_reduction",
            },
        ),
        (
            None,
            {"decomposition": "none", "k": 2, "selection": "not-a-metric"},
        ),
        (
            None,
            {"decomposition": "none", "k": 2, "clustering": "dbscan"},
        ),
        (
            None,
            {"decomposition": "ica", "k": 2, "clustering": "kmeans"},
        ),
    ],
)
def test_delineate_zones_preserves_model_error_type_and_message(outcome, options):
    data = _field(outcome=outcome)

    with pytest.raises(ModelConfigurationError) as direct_error:
        GeoFarmModel(**options).fit(data)
    with pytest.raises(ModelConfigurationError) as convenient_error:
        delineate_zones(data, **options)

    assert str(convenient_error.value) == str(direct_error.value)


def test_delineate_zones_is_a_single_model_construction_and_fit(monkeypatch):
    import geofarmai.api as api_module

    calls = {}
    expected = object()
    data = _field()

    class RecordingModel:
        def __init__(self, **options):
            calls["options"] = options

        def fit(self, fitted_data):
            calls["data"] = fitted_data
            return expected

    monkeypatch.setattr(api_module, "GeoFarmModel", RecordingModel)

    actual = api_module.delineate_zones(
        data,
        decomposition="none",
        clustering="kmeans",
        k=3,
        selection=None,
    )

    assert actual is expected
    assert calls == {
        "data": data,
        "options": {
            "decomposition": "none",
            "clustering": "kmeans",
            "k": 3,
            "selection": None,
        },
    }
