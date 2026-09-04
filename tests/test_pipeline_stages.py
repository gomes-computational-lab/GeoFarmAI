from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prefect")
pytest.importorskip("geopandas")
pytest.importorskip("libpysal")
pytest.importorskip("skfuzzy")
from jobs import flow_mzd


def test_current_ingestion_drops_bad_coordinates_deduplicates_and_normalizes_id(tmp_path):
    path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "x": [0.0, 0.0, "bad", 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "feature": [10.0, 99.0, 20.0, 30.0],
            "unused": [1, 2, 3, 4],
        }
    ).to_csv(path, index=False)

    result = flow_mzd._ingest_one(
        str(path),
        xcol="x",
        ycol="y",
        crs_in="EPSG:4326",
        keep_cols=["feature"],
    )

    assert len(result) == 2
    assert set(result.columns) == {"sample_id", "x", "y", "feature", "geometry"}
    assert result["sample_id"].tolist() == [0, 2]
    assert result["feature"].tolist() == [10.0, 30.0]


def test_predictor_matrix_uses_only_configured_soil_variables(monkeypatch):
    captured = {}
    table = pd.DataFrame(
        {
            "predictor_a": [1.0, 2.0, 3.0],
            "predictor_b": [3.0, 2.0, 1.0],
            "yield": [100.0, 200.0, 300.0],
            "metadata": [9, 9, 9],
        }
    )
    cfg = {
        "project": {
            "soil": {
                "variables": ["predictor_a", "predictor_b"],
                "required_variables": ["predictor_a"],
            }
        },
        "spatial_pca": {"n_components": 2, "use_r_multispati": False},
        "weights": {"k": 1},
    }

    monkeypatch.setattr(flow_mzd, "knn_weights", lambda frame, k: "weights")

    def capture_components(matrix, weights, n_components, use_r):
        captured["columns"] = matrix.columns.tolist()
        captured["weights"] = weights
        return pd.DataFrame(np.zeros((len(matrix), n_components))), False

    monkeypatch.setattr(flow_mzd, "multispati_components", capture_components)

    scores, weights, used_r = flow_mzd.components_from_grid.fn(table, cfg)

    assert captured["columns"] == ["predictor_a", "predictor_b"]
    assert "yield" not in captured["columns"]
    assert captured["weights"] == "weights"
    assert scores.shape == (3, 2)
    assert used_r is False


def test_candidate_k_evaluation_selects_current_vr_then_silhouette_maximum(separated_matrix):
    table = pd.DataFrame({"yield": np.repeat([1.0, 10.0, 20.0], 3)})
    scores = pd.DataFrame(separated_matrix)
    cfg = {
        "project": {"yield_column": "yield"},
        "clustering": {"algorithms": ["kmeans"], "k_values": [2, 3, 4], "seeds": [42]},
    }

    best, leaderboard = flow_mzd.gridsearch.fn(table, scores, cfg)

    assert len(leaderboard) == 3
    assert {row["k"] for row in leaderboard} == {2, 3, 4}
    assert all({"asc", "ch_score", "vr", "anova_p"}.issubset(row) for row in leaderboard)
    expected = max(leaderboard, key=lambda row: (row.get("vr", 0.0), row.get("asc", 0.0)))
    assert best["metrics"] == expected


def test_candidate_k_evaluation_without_outcome_uses_current_silhouette_fallback(separated_matrix):
    table = pd.DataFrame(index=np.arange(len(separated_matrix)))
    scores = pd.DataFrame(separated_matrix)
    cfg = {
        "project": {"yield_column": "yield"},
        "clustering": {"algorithms": ["kmeans"], "k_values": [2, 3], "seeds": [42]},
    }

    best, leaderboard = flow_mzd.gridsearch.fn(table, scores, cfg)

    assert all("vr" not in row and "anova_p" not in row for row in leaderboard)
    expected = max(leaderboard, key=lambda row: row["asc"])
    assert best["metrics"] == expected


def test_vector_flow_result_records_decomposition_provenance(monkeypatch):
    cfg = {
        "raster": {"enabled": False},
        "spatial_pca": {"use_r_multispati": False},
    }
    captured = {}
    monkeypatch.setattr(flow_mzd, "ingest_two", lambda config: ("soil", "outcome"))
    monkeypatch.setattr(flow_mzd, "reproject_to_meters", lambda soil, outcome, config: (soil, outcome))
    monkeypatch.setattr(flow_mzd, "make_density_grid", lambda soil, outcome, config: ("grid", 10.0))
    monkeypatch.setattr(flow_mzd, "reconcile_to_grid", lambda soil, outcome, grid, config: "table")
    monkeypatch.setattr(flow_mzd, "components_from_grid", lambda table, config: ("scores", "weights", False))
    monkeypatch.setattr(
        flow_mzd,
        "gridsearch",
        lambda table, scores, config: ({"labels": [0, 1], "metrics": {"asc": 0.75}}, []),
    )

    def capture_export(table, labels, metrics, leaderboard, config):
        captured["metrics"] = metrics
        return {"gpkg": "zones.gpkg", "pdf": "report.pdf", "images": [], "gridsearch_csv": None}

    monkeypatch.setattr(flow_mzd, "postprocess_and_export", capture_export)

    result = flow_mzd._run_mzd_flow(cfg)

    expected = {
        "requested_method": "pca",
        "actual_method": "pca",
        "used_r": False,
        "fallback_occurred": False,
    }
    assert result["decomposition"] == expected
    assert captured["metrics"]["requested_decomposition_method"] == "pca"
    assert captured["metrics"]["actual_decomposition_method"] == "pca"
    assert captured["metrics"]["used_r"] is False
    assert captured["metrics"]["decomposition_fallback_occurred"] is False
