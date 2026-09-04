from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyproj")
from shapely.geometry import Point

from core.crs import to_utm_auto, utm_zone_from_lon
from core.grid import build_field_grid, choose_cell_size, make_grid, median_nn_distance
from core.ingest import Ingestor


def test_ingestor_reads_distributed_csv_schema(sample_csv_paths, tmp_path):
    soil_path, _ = sample_csv_paths
    source = pd.read_csv(soil_path).head(8).copy()
    source.insert(0, "sample_id", np.arange(len(source)))
    path = tmp_path / "soil_subset.csv"
    source.to_csv(path, index=False)
    cfg = {
        "project": {
            "id_column": "sample_id",
            "x": "Longitude",
            "y": "Latitude",
            "variables": ["Slope", "EC_DP", "Moisture"],
            "crs": "EPSG:4326",
        }
    }

    result = Ingestor(cfg).read_csv(str(path))

    assert len(result) == len(source)
    assert result.crs.to_epsg() == 4326
    assert {"sample_id", "Slope", "EC_DP", "Moisture", "geometry"}.issubset(result.columns)
    np.testing.assert_allclose(result.geometry.x, source["Longitude"])
    np.testing.assert_allclose(result.geometry.y, source["Latitude"])


def test_ingestor_rejects_a_missing_declared_column(tmp_path):
    path = tmp_path / "incomplete.csv"
    pd.DataFrame({"id": [1], "x": [0.0], "y": [0.0]}).to_csv(path, index=False)
    cfg = {
        "project": {
            "id_column": "id",
            "x": "x",
            "y": "y",
            "variables": ["missing_predictor"],
            "crs": "EPSG:4326",
        }
    }

    with pytest.raises(ValueError, match="Missing required column: missing_predictor"):
        Ingestor(cfg).read_csv(str(path))


@pytest.mark.parametrize(
    ("longitude", "expected_zone"),
    [(-180.0, 1), (-111.7, 12), (0.0, 31), (179.9, 60)],
)
def test_utm_zone_detection(longitude, expected_zone):
    assert utm_zone_from_lon(longitude) == expected_zone


@pytest.mark.parametrize(
    ("latitude", "expected_epsg"),
    [(33.0, 32612), (-33.0, 32712)],
)
def test_automatic_utm_conversion_uses_hemisphere(latitude, expected_epsg):
    points = gpd.GeoDataFrame(
        geometry=[Point(-111.7, latitude), Point(-111.69, latitude + 0.01)],
        crs="EPSG:4326",
    )

    projected, epsg = to_utm_auto(points)

    assert epsg == f"EPSG:{expected_epsg}"
    assert projected.crs.to_epsg() == expected_epsg
    assert projected.crs.is_projected


def test_grid_creation_preserves_current_cell_geometry_and_count():
    grid = make_grid((0.0, 0.0, 20.0, 10.0), 10.0)

    assert len(grid) == 2
    np.testing.assert_allclose(grid.total_bounds, [0.0, 0.0, 20.0, 10.0])
    np.testing.assert_allclose(grid.area.to_numpy(), [100.0, 100.0])


def test_field_grid_uses_union_envelope_and_soil_crs():
    soil = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 10)], crs="EPSG:32612")
    outcome = gpd.GeoDataFrame(geometry=[Point(-5, -5), Point(15, 15)], crs="EPSG:32612")

    grid = build_field_grid(soil, outcome, 10.0)

    assert len(grid) == 4
    assert grid.crs == soil.crs
    assert grid["cell_id"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(grid.total_bounds, [-5.0, -5.0, 15.0, 15.0])


def test_density_based_cell_size_is_clipped_to_configured_limits():
    points = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(20, 0), Point(40, 0)],
        crs="EPSG:32612",
    )

    assert median_nn_distance(points) == pytest.approx(20.0)
    assert choose_cell_size(points, min_cell=3.0, max_cell=30.0) == pytest.approx(10.0)
    assert choose_cell_size(points, min_cell=12.0, max_cell=30.0) == pytest.approx(12.0)
