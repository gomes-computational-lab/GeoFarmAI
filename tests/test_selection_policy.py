from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from geofarmai import FieldDataset, GeoFarmModel, ModelConfigurationError
import geofarmai.model as model_module
from jobs.flow_mzd import gridsearch as legacy_gridsearch


CRS = "EPSG:32615"


def _data(*, outcomes: tuple[str, ...] = ()) -> FieldDataset:
    group = np.repeat([0.0, 5.0, 10.0], 4)
    frame = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.tile([0.0, 1.0, 0.0, 1.0], 3),
            "EC": group + np.tile([0.0, 0.1, -0.1, 0.05], 3),
            "moisture": group + np.tile([0.2, -0.1, 0.1, -0.2], 3),
            "yield": np.repeat([10.0, 50.0, 90.0], 4),
            "nitrate": np.repeat([0.0, 10.0, 10.0], 4),
        }
    )
    variables = ["EC", "moisture"]
    return FieldDataset.from_dataframe(
        frame,
        predictors=variables,
        outcome=list(outcomes) if outcomes else None,
        coordinates=("x", "y"),
        crs=CRS,
    )


def _candidate_value(candidate, selection, outcome=None):
    if selection == "silhouette":
        return candidate.internal_metrics["asc"]
    if selection == "calinski_harabasz":
        return candidate.internal_metrics["ch_score"]
    return candidate.outcome_metrics[outcome]["variance_reduction"]


@pytest.mark.parametrize("outcome", ["yield", "nitrate"])
def test_variance_reduction_selection_maximizes_explicit_outcome(outcome):
    result = GeoFarmModel(
        decomposition="none",
        clustering="kmeans",
        k=[2, 3],
        selection="variance_reduction",
    ).fit(_data(outcomes=(outcome,)))

    expected = max(
        result.candidate_solutions,
        key=lambda candidate: _candidate_value(
            candidate, "variance_reduction", outcome
        ),
    )
    assert result.selected_solution is expected
    assert result.selection_metric == "variance_reduction"
    assert result.selection_outcome == outcome
    assert result.selection_direction == "maximize"


def test_variance_reduction_without_declared_outcome_fails_without_switching():
    # A column called yield is present in the source, but is deliberately not
    # declared as an outcome and must not be inferred as one.
    with pytest.raises(ModelConfigurationError, match="requires.*outcome"):
        GeoFarmModel(
            decomposition="none",
            k=[2, 3],
            selection="variance_reduction",
        ).fit(_data())


@pytest.mark.parametrize("selection", ["silhouette", "calinski_harabasz"])
@pytest.mark.parametrize("with_outcome", [False, True])
def test_internal_selection_maximizes_only_the_requested_metric(
    selection, with_outcome
):
    outcomes = ("yield",) if with_outcome else ()
    result = GeoFarmModel(
        decomposition="none",
        clustering=["kmeans", "gmm"],
        k=[2, 3],
        selection=selection,
    ).fit(_data(outcomes=outcomes))

    expected = max(
        result.candidate_solutions,
        key=lambda candidate: _candidate_value(candidate, selection),
    )
    assert result.selected_solution is expected
    assert result.selection_metric == selection
    assert result.selection_outcome is None
    assert result.selection_direction == "maximize"
    if with_outcome:
        assert all("yield" in candidate.outcome_metrics for candidate in result.candidate_solutions)


@pytest.mark.parametrize("with_outcome", [False, True])
def test_selection_none_retains_metrics_without_declaring_a_winner(with_outcome):
    outcomes = ("yield",) if with_outcome else ()
    model = GeoFarmModel(
        decomposition="none",
        clustering=["kmeans", "gmm"],
        k=[2, 3],
        selection=None,
    )

    result = model.fit(_data(outcomes=outcomes))

    assert len(result.candidate_solutions) == 4
    assert result.selected_solution is None
    assert result.selected_candidate is None
    assert result.zone_labels is None
    assert result.best_model is None
    assert result.best_k is None
    assert result.best_labels is None
    assert result.metrics == {}
    assert result.internal_metrics == {}
    assert result.outcome_validation_metrics == {}
    assert result.selection_metric is None
    assert result.selection_direction is None
    assert result.summary()["selected_solution"] is None
    assert not result.leaderboard["selected"].any()
    assert "zone" not in result.to_dataframe().columns
    assert model.selected_solution_ is None
    assert model.labels_ is None
    if with_outcome:
        assert all("yield" in candidate.outcome_metrics for candidate in result.candidate_solutions)


def test_selection_none_export_does_not_implicitly_choose_candidate(tmp_path):
    result = GeoFarmModel(
        decomposition="none",
        k=[2, 3],
        selection=None,
    ).fit(_data(outcomes=("yield",)))

    unselected = result.export(tmp_path / "unselected")
    samples = gpd.read_file(unselected["geopackage"], layer="samples")
    assert "zone" not in samples.columns

    selected = result.export(
        tmp_path / "explicit_candidate",
        solution=result.candidate_solutions[0],
    )
    assert "zone" in gpd.read_file(selected["geopackage"], layer="samples").columns
    assert not gpd.read_file(selected["geopackage"], layer="zones").empty


def test_multiple_outcomes_require_explicit_variance_reduction_outcome():
    with pytest.raises(ModelConfigurationError, match="multiple outcomes.*selection_outcome"):
        GeoFarmModel(
            decomposition="none",
            k=[2, 3],
            selection="variance_reduction",
        ).fit(_data(outcomes=("yield", "nitrate")))


def test_one_outcome_drives_selection_while_other_outcome_metrics_are_retained():
    result = GeoFarmModel(
        decomposition="none",
        clustering="kmeans",
        k=[2, 3],
        selection="variance_reduction",
        selection_outcome="yield",
    ).fit(_data(outcomes=("yield", "nitrate")))

    expected = max(
        result.candidate_solutions,
        key=lambda candidate: candidate.outcome_metrics["yield"]["variance_reduction"],
    )
    assert result.selected_solution is expected
    assert result.selection_outcome == "yield"
    assert all(
        set(candidate.outcome_metrics) == {"yield", "nitrate"}
        for candidate in result.candidate_solutions
    )


def test_fixed_k_evaluates_all_algorithms_without_forcing_selection():
    result = GeoFarmModel(
        decomposition="none",
        clustering=["kmeans", "gmm", "fcm", "agglomerative"],
        k=3,
        selection=None,
        weights_k=2,
    ).fit(_data())

    assert len(result.candidate_solutions) == 4
    assert {candidate.k for candidate in result.candidate_solutions} == {3}
    assert result.selected_solution is None


def test_candidate_k_range_retains_every_solution():
    result = GeoFarmModel(
        decomposition="none",
        clustering=["kmeans", "gmm"],
        k=range(2, 5),
        selection="silhouette",
    ).fit(_data())

    assert len(result.candidate_solutions) == 6
    assert {candidate.k for candidate in result.candidate_solutions} == {2, 3, 4}


@pytest.mark.parametrize("selection", ["silhouette", "calinski_harabasz"])
def test_exact_metric_ties_preserve_stable_candidate_generation_order(
    monkeypatch, selection
):
    def tied_runner(X, k, random_state, sample_size):
        return np.arange(len(X)) % k, {"asc": 0.5, "ch_score": 100.0}

    monkeypatch.setattr(model_module, "run_gmm", tied_runner)
    monkeypatch.setattr(model_module, "run_kmeans", tied_runner)

    result = GeoFarmModel(
        decomposition="none",
        clustering=["gmm", "kmeans"],
        k=[3, 2],
        selection=selection,
    ).fit(_data())

    assert result.selected_solution is result.candidate_solutions[0]
    assert result.selected_solution.algorithm == "gmm"
    assert result.selected_solution.k == 3


def test_unsupported_selection_metric_fails_explicitly():
    with pytest.raises(ModelConfigurationError, match="Unsupported selection metric"):
        GeoFarmModel(decomposition="none", k=2, selection="asc").fit(_data())


def test_selection_outcome_is_rejected_for_internal_selection():
    with pytest.raises(ModelConfigurationError, match="only valid"):
        GeoFarmModel(
            decomposition="none",
            k=2,
            selection="silhouette",
            selection_outcome="yield",
        ).fit(_data(outcomes=("yield",)))


def test_fit_predict_requires_an_explicit_selection_policy():
    model = GeoFarmModel(decomposition="none", k=2, selection=None)

    with pytest.raises(ModelConfigurationError, match="requires a selection policy"):
        model.fit_predict(_data())
    assert model.get_result().selected_solution is None


def test_legacy_gridsearch_retains_historical_outcome_first_selection():
    data = _data(outcomes=("yield",))
    frame = data.sources[0].data
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["x"], frame["y"]),
        crs=CRS,
    )
    components = frame[["EC", "moisture"]]
    config = {
        "project": {},
        "clustering": {
            "k_values": [2, 3],
            "algorithms": ["kmeans", "gmm"],
            "seeds": [42],
        },
    }

    best, leaderboard = legacy_gridsearch.fn(
        points,
        components,
        config,
        outcome_name="yield",
    )

    expected = max(leaderboard, key=lambda row: (row["vr"], row["asc"]))
    assert best["metrics"] == expected
    assert "selection" not in config["clustering"]
