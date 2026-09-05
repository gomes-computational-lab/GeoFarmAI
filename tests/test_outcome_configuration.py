from __future__ import annotations

import pandas as pd
import pytest

from geofarmai import OutcomeConfigurationError
from geofarmai.outcome import resolve_pipeline_outcome

pytest.importorskip("prefect")
from jobs import flow_mzd


def test_generic_outcome_can_be_a_column_in_predictor_source(tmp_path):
    path = tmp_path / "combined.csv"
    pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "ec": [1.0, 2.0, 3.0],
            "protein": [10.0, 11.0, 12.0],
        }
    ).to_csv(path, index=False)
    cfg = {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {"path": str(path), "x": "x", "y": "y", "variables": ["ec"]},
            "outcome": "protein",
        }
    }

    predictors, outcome_points, outcome_name = flow_mzd.ingest_sources.fn(cfg)

    assert outcome_name == "protein"
    assert outcome_points is None
    assert "protein" in predictors.columns


def test_generic_outcome_can_use_a_separate_nitrate_source(tmp_path):
    predictor_path = tmp_path / "predictors.csv"
    outcome_path = tmp_path / "nitrate.csv"
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "ec": [1.0, 2.0]}
    ).to_csv(predictor_path, index=False)
    pd.DataFrame(
        {"x": [0.5, 1.5], "y": [0.5, 1.5], "n_mg": [8.0, 9.0]}
    ).to_csv(outcome_path, index=False)
    cfg = {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {
                "path": str(predictor_path),
                "x": "x",
                "y": "y",
                "variables": ["ec"],
            },
            "outcome": {
                "name": "nitrate",
                "column": "n_mg",
                "path": str(outcome_path),
                "x": "x",
                "y": "y",
            },
        }
    }

    predictors, outcome_points, outcome_name = flow_mzd.ingest_sources.fn(cfg)

    assert outcome_name == "nitrate"
    assert "nitrate" not in predictors.columns
    assert outcome_points is not None
    assert outcome_points["nitrate"].tolist() == [8.0, 9.0]


def test_explicit_null_outcome_is_unsupervised(tmp_path):
    path = tmp_path / "predictors.csv"
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "ec": [1.0, 2.0]}
    ).to_csv(path, index=False)
    cfg = {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {"path": str(path), "x": "x", "y": "y", "variables": ["ec"]},
            "outcome": None,
        }
    }

    predictors, outcome_points, outcome_name = flow_mzd.ingest_sources.fn(cfg)

    assert len(predictors) == 2
    assert outcome_points is None
    assert outcome_name is None


def test_legacy_yield_configuration_is_isolated_and_still_normalizes_column(tmp_path):
    predictor_path = tmp_path / "soil.csv"
    outcome_path = tmp_path / "harvest.csv"
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "ec": [1.0, 2.0]}
    ).to_csv(predictor_path, index=False)
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "harvest_mass": [20.0, 30.0]}
    ).to_csv(outcome_path, index=False)
    project = {
        "crs_in": "EPSG:32615",
        "soil": {
            "path": str(predictor_path),
            "x": "x",
            "y": "y",
            "variables": ["ec"],
        },
        "yield": {
            "path": str(outcome_path),
            "x": "x",
            "y": "y",
            "column": "harvest_mass",
        },
        "yield_column": "yield",
    }

    resolved = resolve_pipeline_outcome(project)
    _, outcome_points, outcome_name = flow_mzd.ingest_sources.fn({"project": project})

    assert resolved is not None and resolved.legacy_yield_adapter is True
    assert outcome_name == "yield"
    assert outcome_points["yield"].tolist() == [20.0, 30.0]


def test_outcome_cannot_also_be_a_predictor(tmp_path):
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "nitrate": [1.0, 2.0]}
    ).to_csv(path, index=False)
    cfg = {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {
                "path": str(path),
                "x": "x",
                "y": "y",
                "variables": ["nitrate"],
            },
            "outcome": "nitrate",
        }
    }

    with pytest.raises(ValueError, match="conflicts with a predictor name"):
        flow_mzd.ingest_sources.fn(cfg)


def test_external_outcome_cannot_overwrite_same_named_predictor(tmp_path):
    predictor_path = tmp_path / "predictors.csv"
    outcome_path = tmp_path / "outcome.csv"
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "nitrate": [1.0, 2.0]}
    ).to_csv(predictor_path, index=False)
    pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "lab_n": [8.0, 9.0]}
    ).to_csv(outcome_path, index=False)
    cfg = {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {
                "path": str(predictor_path),
                "x": "x",
                "y": "y",
                "variables": ["nitrate"],
            },
            "outcome": {
                "name": "nitrate",
                "column": "lab_n",
                "path": str(outcome_path),
                "x": "x",
                "y": "y",
            },
        }
    }

    with pytest.raises(ValueError, match="legacy flat-table pipeline"):
        flow_mzd.ingest_sources.fn(cfg)


def test_invalid_outcome_configuration_fails_clearly():
    with pytest.raises(OutcomeConfigurationError, match="requires 'column' or 'name'"):
        resolve_pipeline_outcome({"outcome": {"path": "values.csv", "x": "x", "y": "y"}})
