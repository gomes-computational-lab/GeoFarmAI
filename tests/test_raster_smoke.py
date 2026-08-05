from pathlib import Path

import pytest
import yaml

from geofarmai import run_pipeline


@pytest.mark.gis
def test_low_resolution_raster_pipeline(tmp_path: Path, synthetic_config: Path):
    config = yaml.safe_load(synthetic_config.read_text(encoding="utf-8"))
    config["raster"].update(
        {
            "enabled": True,
            "cache": False,
            "cell_size": 20.0,
            "use_pca": True,
            "parallel_gridsearch": False,
        }
    )
    config["spatial_pca"]["engine"] = "pca"
    config["clustering"] = {"algorithms": ["kmeans"], "k_values": [2], "seeds": [42]}
    raster_config = tmp_path / "raster.yaml"
    raster_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_pipeline(raster_config, force=True)

    assert result.best_model == "kmeans"
    assert result.best_k == 2
    assert result.artifacts["artifact"].is_file()
    assert result.artifacts["metrics_csv"].is_file()
    assert result.artifacts["gridsearch_csv"].is_file()
    assert result.artifacts["visual_manifest"].name == "artifact_manifest.json"
