from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prefect")
pytest.importorskip("geopandas")
pytest.importorskip("libpysal")
pytest.importorskip("skfuzzy")
from jobs import flow_mzd


@pytest.mark.integration
def test_current_vector_pipeline_smoke_with_distributed_sample_data(sample_csv_paths, tmp_path):
    soil_path, yield_path = sample_csv_paths
    soil_source = pd.read_csv(soil_path)
    yield_source = pd.read_csv(yield_path)
    soil_subset = soil_source.iloc[np.linspace(0, len(soil_source) - 1, 40, dtype=int)]
    yield_subset = yield_source.iloc[np.linspace(0, len(yield_source) - 1, 40, dtype=int)]
    test_soil = tmp_path / "soil.csv"
    test_yield = tmp_path / "yield.csv"
    soil_subset.to_csv(test_soil, index=False)
    yield_subset.to_csv(test_yield, index=False)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    cfg = {
        "project": {
            "name": "regression_smoke",
            "crs_in": "EPSG:4326",
            "auto_reproject_to_utm": True,
            "soil": {
                "path": str(test_soil),
                "x": "Longitude",
                "y": "Latitude",
                "id_column": None,
                "variables": ["Slope", "EC_DP", "Moisture"],
                "required_variables": ["Slope", "EC_DP"],
            },
            "yield": {
                "path": str(test_yield),
                "x": "Longitude",
                "y": "Latitude",
                "id_column": None,
                "column": "Yld_Mass_Dry_lb_ac",
            },
            "yield_column": "yield",
        },
        "grid": {"cell_size_m": 100.0, "method": "idw", "buffer_m": 15.0},
        "weights": {"k": 2},
        "spatial_pca": {"engine": "multispaeti", "use_r_multispati": False, "n_components": 2},
        "clustering": {"algorithms": ["kmeans"], "k_values": [2, 3], "seeds": [42]},
        "raster": {"enabled": False},
        "postprocess": {"min_area_m2": 0.0},
        "export": {"basemap": "none", "out_dir": str(output_dir)},
    }

    soil, outcome = flow_mzd.ingest_two.fn(cfg)
    soil, outcome = flow_mzd.reproject_to_meters.fn(soil, outcome, cfg)
    grid, cell = flow_mzd.make_density_grid.fn(soil, outcome, cfg)
    table = flow_mzd.reconcile_to_grid.fn(soil, outcome, grid, cfg)
    components, _, used_r = flow_mzd.components_from_grid.fn(table, cfg)
    best, leaderboard = flow_mzd.gridsearch.fn(table, components, cfg)
    metrics = {**best["metrics"], "used_r_multispati": used_r, "experiment": "baseline"}
    artifacts = flow_mzd.postprocess_and_export.fn(
        table,
        best["labels"],
        metrics,
        leaderboard,
        cfg,
    )

    assert cell == 100.0
    assert len(grid) > 3
    assert components.shape == (len(grid), 2)
    assert len(leaderboard) == 2
    assert {row["k"] for row in leaderboard} == {2, 3}
    assert {"asc", "ch_score", "vr", "anova_p"}.issubset(best["metrics"])
    assert used_r is False
    for key in ("gpkg", "pdf", "gridsearch_csv"):
        assert artifacts[key]
        assert __import__("pathlib").Path(artifacts[key]).is_file()
    assert artifacts["images"]
