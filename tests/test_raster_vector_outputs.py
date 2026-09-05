from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from shapely import union_all
from shapely.geometry import Point

from core.export import save_package, zones_from_points
from core.raster_pipeline import (
    build_arcgis_grid,
    ordinary_krige,
    raster_gridsearch,
    spatial_pca_from_stack,
    write_raster,
    write_cluster_outputs,
    zones_from_label_raster,
)
from scipy.sparse import csr_matrix


def test_arcgis_grid_honors_explicit_bounds():
    grid_x, grid_y, cell = build_arcgis_grid(
        np.array([100.0, 200.0]),
        np.array([300.0, 400.0]),
        cell_size=10.0,
        bounds=(0.0, 30.0, 50.0, 90.0),
    )

    np.testing.assert_allclose(grid_x, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(grid_y, [50.0, 60.0, 70.0, 80.0])
    assert cell == 10.0


def test_raster_write_round_trip_preserves_values_crs_and_transform(tmp_path):
    values = np.array([[1.0, 2.0], [3.0, np.nan]], dtype="float64")
    path = tmp_path / "surface.tif"

    write_raster(values, path, "EPSG:32612", np.array([100.0, 110.0]), np.array([200.0, 210.0]), 10.0)

    with rasterio.open(path) as dataset:
        actual = dataset.read(1)
        assert dataset.crs.to_epsg() == 32612
        assert dataset.transform == from_origin(100.0, 210.0, 10.0, 10.0)
        assert np.isnan(dataset.nodata)
    np.testing.assert_allclose(actual, values, equal_nan=True)


def test_raster_zone_polygonization_is_label_permutation_invariant():
    labels = np.array([0, 0, 1, 1])
    permuted = np.array([8, 8, 3, 3])
    valid = np.ones(4, dtype=bool)
    profile = {"transform": from_origin(0.0, 20.0, 10.0, 10.0), "crs": "EPSG:32612"}

    first = zones_from_label_raster(labels, valid, (2, 2), profile, min_area=0.0)
    second = zones_from_label_raster(permuted, valid, (2, 2), profile, min_area=0.0)

    assert len(first) == len(second) == 2
    np.testing.assert_allclose(sorted(first.area), sorted(second.area))
    assert union_all(first.geometry.array).equals(union_all(second.geometry.array))


def test_vector_zone_and_geopackage_outputs(tmp_path):
    points = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        geometry=[Point(0, 0), Point(1, 0), Point(10, 0), Point(11, 0)],
        crs="EPSG:32612",
    )
    labels = np.array([0, 0, 1, 1])
    zones = zones_from_points(points, labels, min_area=0.0)
    output = tmp_path / "zones.gpkg"

    save_package(zones, points.assign(zone=labels), {"asc": 0.8}, str(output))

    assert output.is_file()
    assert (tmp_path / "zones_metrics.csv").is_file()
    assert len(gpd.read_file(output, layer="zones")) == 2
    assert len(gpd.read_file(output, layer="samples")) == 4


@pytest.mark.optional_scientific
def test_ordinary_kriging_is_deterministic_when_pykrige_is_available():
    pytest.importorskip("pykrige")
    x = np.array([0.0, 0.0, 10.0, 10.0, 5.0, 5.0])
    y = np.array([0.0, 10.0, 0.0, 10.0, 2.0, 8.0])
    values = 2.0 * x + y
    grid = np.array([0.0, 5.0, 10.0])
    cfg = {"variogram_model": "linear", "nlags": 4, "n_closest_points": 6, "backend": "loop"}

    first, first_parameters = ordinary_krige(x, y, values, grid, grid, cfg, "euclidean")
    second, second_parameters = ordinary_krige(x, y, values, grid, grid, cfg, "euclidean")

    assert first.shape == (3, 3)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(first_parameters, second_parameters, rtol=1e-10, atol=1e-10)


def test_raster_pca_path_preserves_pairwise_component_geometry(monkeypatch):
    import core.raster_pipeline as raster_pipeline

    monkeypatch.setattr(raster_pipeline, "MultispatiPCA", None)
    rows, cols = 3, 3
    base = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    stack = np.stack([base, base**2, np.flipud(base)], axis=-1)
    valid = np.ones(rows * cols, dtype=bool)
    profile = {"transform": from_origin(0.0, 30.0, 10.0, 10.0), "crs": "EPSG:32612"}
    cfg = {
        "weights": {"k": 2},
        "raster": {"use_pca": True},
        "spatial_pca": {"engine": "pca", "n_components": 2},
    }

    scores, summary, used_multispaeti, connectivity, names = spatial_pca_from_stack(
        stack, valid, profile, cfg, ["a", "b", "c"]
    )

    assert scores.shape == (9, 2)
    assert names == ["PC1", "PC2"]
    assert used_multispaeti is False
    assert "PCA fallback summary" in summary
    assert connectivity.shape == (9, 9)


def test_raster_gridsearch_and_cluster_map_work_without_outcome(
    separated_matrix, tmp_path
):
    valid = np.ones(9, dtype=bool)
    connectivity = csr_matrix((9, 9), dtype=float)
    cfg = {
        "clustering": {"algorithms": ["kmeans"], "k_values": [2, 3], "seeds": [42]},
        "raster": {"parallel_gridsearch": False, "silhouette_sample_size": 100},
    }

    best, leaderboard, selections = raster_gridsearch(
        separated_matrix,
        None,
        valid,
        connectivity,
        cfg,
    )

    assert best is not None
    assert all({"asc", "ch_score"}.issubset(row) for row in leaderboard)
    assert all("vr" not in row and "anova_p" not in row for row in leaderboard)
    assert "internal_metrics" in selections

    profile = {
        "driver": "GTiff",
        "height": 3,
        "width": 3,
        "count": 1,
        "dtype": "float64",
        "crs": "EPSG:32612",
        "transform": from_origin(0.0, 30.0, 10.0, 10.0),
        "nodata": np.nan,
    }
    paths = write_cluster_outputs(
        selections,
        leaderboard,
        valid,
        (3, 3),
        profile,
        tmp_path / "clusters",
        tmp_path / "preview",
    )

    assert paths
    assert (tmp_path / "clusters" / "best_clusters.tif").is_file()


def test_raster_gridsearch_uses_arbitrary_named_outcome(separated_matrix):
    valid = np.ones(9, dtype=bool)
    connectivity = csr_matrix((9, 9), dtype=float)
    nitrate = np.repeat([1.0, 10.0, 20.0], 3).reshape(3, 3)
    cfg = {
        "clustering": {"algorithms": ["kmeans"], "k_values": [2, 3], "seeds": [42]},
        "raster": {"parallel_gridsearch": False, "silhouette_sample_size": 100},
    }

    best, leaderboard, selections = raster_gridsearch(
        separated_matrix,
        nitrate,
        valid,
        connectivity,
        cfg,
        outcome_name="nitrate",
    )

    assert best is not None
    assert all(row["outcome_name"] == "nitrate" for row in leaderboard)
    assert all({"vr", "anova_p"}.issubset(row) for row in leaderboard)
    assert "outcome_variance" in selections


@pytest.mark.optional_scientific
def test_python_multispaeti_path_when_dependency_is_available():
    pytest.importorskip("multispaeti")
    rows, cols = 4, 4
    base = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    stack = np.stack([base, np.sin(base), np.cos(base)], axis=-1)
    valid = np.ones(rows * cols, dtype=bool)
    profile = {"transform": from_origin(0.0, 40.0, 10.0, 10.0), "crs": "EPSG:32612"}
    cfg = {
        "weights": {"k": 3},
        "raster": {"use_pca": True, "random_state": 42},
        "spatial_pca": {"engine": "multispaeti", "n_components": 2},
    }

    scores, summary, used_multispaeti, connectivity, names = spatial_pca_from_stack(
        stack, valid, profile, cfg, ["a", "b", "c"]
    )

    assert scores.shape == (16, 2)
    assert np.isfinite(scores).all()
    assert used_multispaeti is True
    assert names == ["PC1", "PC2"]
    assert "MULTISPATI-PCA summary" in summary
    assert connectivity.shape == (16, 16)
