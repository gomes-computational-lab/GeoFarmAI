from pathlib import Path

import pandas as pd

from geofarmai import GeoFarmPipeline, GeoFarmResult, run_pipeline


def test_minimal_vector_pipeline(synthetic_config: Path):
    result = run_pipeline(synthetic_config)
    assert isinstance(result, GeoFarmResult)
    assert result.best_model in {"kmeans", "agglomerative", "gmm", "fcm"}
    assert result.best_k in {2, 3}
    assert not result.leaderboard.empty
    assert {"vr", "anova_p", "asc", "ch_score"}.issubset(result.leaderboard.columns)
    for key in ("gpkg", "pdf", "gridsearch_csv", "artifact_manifest"):
        assert result.artifacts[key].is_file()
    assert pd.read_csv(result.artifacts["gridsearch_csv"]).shape[0] == 14


def test_pipeline_class_loads_same_configuration(synthetic_config: Path):
    pipeline = GeoFarmPipeline.from_yaml(synthetic_config)
    assert pipeline.config.source == synthetic_config.resolve()
