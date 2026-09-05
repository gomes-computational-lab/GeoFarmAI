from __future__ import annotations

import builtins

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from core.multispati import python_multispati_components
from geofarmai import (
    CandidateSolution,
    FieldDataset,
    GeoFarmModel,
    GeoFarmResult,
    ModelConfigurationError,
    ModelNotFittedError,
    MultispatiUnavailableError,
    VariableIdentity,
    run_pipeline,
)
import geofarmai.model as model_module

from conftest import same_partition


CRS = "EPSG:32615"


def _field(*, outcome: str | None = "yield") -> FieldDataset:
    x = np.array([0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43], dtype=float)
    group = np.repeat([0.0, 5.0, 10.0], 4)
    frame = pd.DataFrame(
        {
            "x": x,
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
        source_id="field",
        predictors=["EC", "moisture"],
        outcome=outcome,
        coordinates=("x", "y"),
        crs=CRS,
    )


def test_ergonomic_csv_constructor_matches_documented_public_api(tmp_path):
    path = tmp_path / "field.csv"
    frame = _field().sources[0].data
    frame.to_csv(path, index=False)

    data = FieldDataset.from_csv(
        path,
        coordinates=("x", "y"),
        predictors=["EC", "moisture"],
        outcome="yield",
        crs=CRS,
    )

    assert data.predictor_names == ("EC", "moisture")
    assert data.outcome_names == ("yield",)


def test_complete_model_api_retains_every_algorithm_and_k_candidate():
    model = GeoFarmModel(
        decomposition="pca",
        clustering=["kmeans", "gmm", "fcm", "agglomerative"],
        k=range(2, 4),
        n_components=2,
        random_state=17,
        weights_k=2,
    )

    result = model.fit(_field())

    assert isinstance(result, GeoFarmResult)
    assert len(result.candidate_solutions) == 8
    assert {
        (candidate.algorithm, candidate.k)
        for candidate in result.candidate_solutions
    } == {
        (algorithm, k)
        for algorithm in ("kmeans", "gmm", "fcm", "agglomerative")
        for k in (2, 3)
    }
    assert any(
        result.selected_solution is candidate for candidate in result.candidate_solutions
    )
    assert result.zone_labels.shape == (12,)
    assert result.predictor_names == ("EC", "moisture")
    assert result.outcome_names == ("yield",)
    assert result.component_scores.shape == (12, 2)
    assert result.component_loadings.shape == (2, 2)
    assert "asc" in result.internal_metrics
    assert "ch_score" in result.internal_metrics
    assert set(result.outcome_validation_metrics["yield"]) == {
        "variance_reduction",
        "anova_p",
        "n_observations",
        "coverage_fraction",
    }
    assert model.result_ is result
    assert model.n_features_in_ == 2
    assert model.feature_names_in_.tolist() == ["EC", "moisture"]


def test_no_outcome_still_fits_and_uses_only_internal_metrics():
    model = GeoFarmModel(decomposition="none", clustering="kmeans", k=3)

    result = model.fit(_field(outcome=None))

    assert result.outcome_names == ()
    assert result.outcome_validation_metrics == {}
    assert all(candidate.outcome_metrics == {} for candidate in result.candidate_solutions)
    assert model.analysis_matrix_.columns.tolist() == [
        ("field", "EC"),
        ("field", "moisture"),
    ]
    np.testing.assert_allclose(model.analysis_matrix_.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.analysis_matrix_.std(ddof=0), 1.0, atol=1e-12
    )
    assert result.to_dataframe().columns.tolist() == [
        "geometry",
        "EC",
        "moisture",
        "zone",
    ]


def test_nitrate_is_a_generic_outcome_and_never_a_predictor():
    model = GeoFarmModel(decomposition="none", clustering="kmeans", k=[2, 3])

    result = model.fit(_field(outcome="nitrate"))

    assert result.outcome_names == ("nitrate",)
    assert "nitrate" in result.outcome_validation_metrics
    assert "nitrate" not in model.feature_names_in_
    assert ("field", "nitrate") not in model.analysis_matrix_.columns


def test_partial_outcome_coverage_is_reported_without_affecting_clustering():
    source = _field().sources[0]
    frame = source.data.copy()
    frame.loc[[0, 1], "yield"] = np.nan
    data = FieldDataset.from_dataframe(
        frame,
        predictors=["EC", "moisture"],
        outcome="yield",
        coordinates=("x", "y"),
        crs=CRS,
    )

    result = GeoFarmModel(decomposition="none", k=3).fit(data)

    metrics = result.outcome_validation_metrics["yield"]
    assert metrics["n_observations"] == 10
    assert metrics["coverage_fraction"] == pytest.approx(10 / 12)
    assert len(result.zone_labels) == 12


def test_seeded_model_fits_are_partition_reproducible():
    options = dict(
        decomposition="pca",
        clustering=["kmeans", "gmm", "fcm"],
        k=[2, 3],
        n_components=2,
        random_state=1337,
    )

    first = GeoFarmModel(**options).fit(_field())
    second = GeoFarmModel(**options).fit(_field())

    for left, right in zip(first.candidate_solutions, second.candidate_solutions):
        assert left.solution_id == right.solution_id
        assert same_partition(left.labels, right.labels)
        assert left.internal_metrics == pytest.approx(
            right.internal_metrics, rel=1e-12, abs=1e-12
        )


def test_multispati_orchestration_uses_existing_python_engine(monkeypatch):
    called = {}

    class _Fitted:
        components_ = np.array([[0.5, 0.5]])

    def fake_multispati(X, connectivity, n_components, random_state):
        called.update(
            rows=len(X),
            connectivity=connectivity.shape,
            n_components=n_components,
            random_state=random_state,
        )
        return (
            pd.DataFrame({"SPC1": np.linspace(-1.0, 1.0, len(X))}, index=X.index),
            _Fitted(),
            object(),
        )

    monkeypatch.setattr(model_module, "python_multispati_components", fake_multispati)

    result = GeoFarmModel(
        decomposition="multispati",
        multispati_engine="python",
        clustering="kmeans",
        k=2,
        n_components=1,
        random_state=91,
        weights_k=2,
    ).fit(_field())

    assert called == {
        "rows": 12,
        "connectivity": (12, 12),
        "n_components": 1,
        "random_state": 91,
    }
    assert result.decomposition_provenance == {
        "requested_method": "multispati",
        "actual_method": "multispati",
        "engine": "multispaeti",
        "used_r": False,
        "fallback_occurred": False,
        "standardized": True,
    }
    assert result.component_loadings.shape == (1, 2)


def test_unavailable_python_multispati_fails_without_pca_fallback(monkeypatch):
    original_import = builtins.__import__

    def reject_multispaeti(name, *args, **kwargs):
        if name == "multispaeti" or name.startswith("multispaeti."):
            raise ModuleNotFoundError("intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_multispaeti)
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    with pytest.raises(MultispatiUnavailableError, match="explicitly requested"):
        python_multispati_components(
            frame,
            np.eye(3),
            n_components=1,
            random_state=42,
        )


def test_multiple_outcomes_require_explicit_choice_only_for_external_selection():
    source = _field().sources[0]
    frame = source.data.copy()
    frame["protein"] = np.repeat([30.0, 10.0, 20.0], 4)
    data = FieldDataset.from_dataframe(
        frame,
        source_id="field",
        predictors=["EC", "moisture"],
        outcome=["yield", "protein"],
        coordinates=("x", "y"),
        crs=CRS,
    )

    internal = GeoFarmModel(decomposition="none", k=[2, 3]).fit(data)
    external = GeoFarmModel(
        decomposition="none",
        k=[2, 3],
        selection_outcome="protein",
    ).fit(data)

    assert internal.configuration["selection_outcome_used"] is None
    assert external.configuration["selection_outcome_used"] == "protein"
    assert set(external.outcome_validation_metrics) == {"yield", "protein"}


def test_result_summary_dataframe_leaderboard_and_export(tmp_path):
    result = GeoFarmModel(decomposition="pca", clustering="kmeans", k=[2, 3]).fit(
        _field()
    )

    summary = result.summary()
    table = result.to_dataframe()
    artifacts = result.export(tmp_path / "zones")

    assert summary["candidate_count"] == 2
    assert summary["selected_solution"] == result.selected_solution.solution_id
    assert result.metrics["algorithm"] == result.best_model
    assert result.best_labels.tolist() == result.zone_labels.tolist()
    assert table.crs.to_epsg() == 32615
    assert table["zone"].to_numpy().tolist() == result.zone_labels.tolist()
    assert result.leaderboard.shape[0] == 2
    assert all(path.is_file() for path in artifacts.values())
    assert result.artifacts == artifacts
    assert not gpd.read_file(artifacts["geopackage"], layer="zones").empty
    assert len(gpd.read_file(artifacts["geopackage"], layer="samples")) == len(table)


def test_run_pipeline_compatibility_wrapper_deprecates_to_model_api():
    with pytest.deprecated_call(match="compatibility API"):
        result = run_pipeline(
            _field(),
            decomposition="none",
            clustering="kmeans",
            k=3,
        )

    assert isinstance(result, GeoFarmResult)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"clustering": "dbscan", "k": 2}, "Unsupported clustering"),
        ({"clustering": "kmeans", "k": 1}, "at least 2"),
        ({"clustering": "kmeans", "k": 12}, "smaller than"),
        ({"decomposition": "ica", "k": 2}, "Unsupported decomposition"),
    ],
)
def test_invalid_model_configuration_fails_clearly(options, message):
    with pytest.raises(ModelConfigurationError, match=message):
        GeoFarmModel(**options).fit(_field())


def test_get_result_before_fit_fails_clearly():
    with pytest.raises(ModelNotFittedError, match="not been fitted"):
        GeoFarmModel().get_result()


def test_ambiguous_duplicate_outcome_name_requires_source_identity():
    first = _field(outcome="nitrate").sources[0]
    second_frame = first.data[["x", "y", "nitrate"]].copy()
    second = FieldDataset.from_dataframe(
        second_frame,
        source_id="lab2",
        predictors=["nitrate"],
        coordinates=("x", "y"),
        crs=CRS,
    ).sources[0]
    data = FieldDataset.from_sources([first, second])

    result = GeoFarmModel(
        decomposition="none",
        k=2,
        selection_outcome=VariableIdentity("field", "nitrate"),
    ).fit(data)

    assert result.configuration["selection_outcome_used"] == "field:nitrate"


def test_candidate_solution_can_be_selected_for_dataframe_output():
    result = GeoFarmModel(decomposition="none", k=[2, 3]).fit(_field())
    candidate = result.candidate_solutions[0]

    assert isinstance(candidate, CandidateSolution)
    assert result.to_dataframe(candidate)["zone"].tolist() == candidate.labels.tolist()
    assert result.to_dataframe(candidate.solution_id)["zone"].tolist() == candidate.labels.tolist()
