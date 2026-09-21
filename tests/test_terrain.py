from __future__ import annotations

import numpy as np
import pytest

from geofarmai.exceptions import TerrainProcessingError
from geofarmai.terrain import terrain_derivatives


def _assert_nodata_border(values: np.ndarray) -> None:
    assert np.isnan(values[0, :]).all()
    assert np.isnan(values[-1, :]).all()
    assert np.isnan(values[:, 0]).all()
    assert np.isnan(values[:, -1]).all()


def test_flat_surface_has_zero_slope_and_curvature():
    elevation = np.full((7, 8), 125.0)

    result = terrain_derivatives(elevation, x_resolution_m=5.0)

    np.testing.assert_allclose(result.slope_degrees[1:-1, 1:-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.general_curvature[1:-1, 1:-1], 0.0, atol=1e-12)
    _assert_nodata_border(result.slope_degrees)
    _assert_nodata_border(result.general_curvature)


def test_planar_surface_has_known_horn_slope_and_zero_curvature():
    dx, dy = 4.0, 6.0
    x = np.arange(9) * dx
    y = np.arange(8) * dy
    xx, yy = np.meshgrid(x, y)
    a, b = 0.08, -0.03
    elevation = a * xx + b * yy + 100.0

    result = terrain_derivatives(
        elevation,
        x_resolution_m=dx,
        y_resolution_m=dy,
    )

    expected_slope = np.degrees(np.arctan(np.hypot(a, b)))
    np.testing.assert_allclose(
        result.slope_degrees[1:-1, 1:-1], expected_slope, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.general_curvature[1:-1, 1:-1], 0.0, atol=1e-12
    )


def test_quadratic_surface_matches_analytical_laplacian():
    dx, dy = 2.0, 3.0
    x = np.arange(-5, 6) * dx
    y = np.arange(-4, 5) * dy
    xx, yy = np.meshgrid(x, y)
    a, b = 0.004, -0.0015
    elevation = a * xx**2 + b * yy**2

    result = terrain_derivatives(
        elevation,
        x_resolution_m=dx,
        y_resolution_m=dy,
    )

    expected_curvature = 2.0 * a + 2.0 * b
    np.testing.assert_allclose(
        result.general_curvature[1:-1, 1:-1],
        expected_curvature,
        rtol=1e-11,
        atol=1e-12,
    )


def test_nodata_propagates_only_through_required_neighborhoods():
    elevation = np.arange(49, dtype=float).reshape(7, 7)
    elevation[3, 3] = np.nan

    result = terrain_derivatives(elevation, x_resolution_m=1.0)

    _assert_nodata_border(result.slope_degrees)
    _assert_nodata_border(result.general_curvature)
    assert np.isnan(result.slope_degrees[2:5, 2:5]).all()
    assert np.isnan(result.general_curvature[3, 3])
    assert np.isnan(result.general_curvature[2, 3])
    assert np.isnan(result.general_curvature[4, 3])
    assert np.isnan(result.general_curvature[3, 2])
    assert np.isnan(result.general_curvature[3, 4])
    assert np.isfinite(result.general_curvature[2, 2])


def test_invalid_grid_or_resolution_fails_clearly():
    with pytest.raises(TerrainProcessingError, match="at least 3 by 3"):
        terrain_derivatives(np.ones((2, 3)), x_resolution_m=1.0)
    with pytest.raises(TerrainProcessingError, match="finite positive"):
        terrain_derivatives(np.ones((3, 3)), x_resolution_m=0.0)


def test_raster_derivation_preserves_georeferencing_and_provenance(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from geofarmai.terrain import derive_terrain

    elevation_path = tmp_path / "elevation.tif"
    elevation = np.full((5, 6), 100.0)
    with rasterio.open(
        elevation_path,
        "w",
        driver="GTiff",
        height=5,
        width=6,
        count=1,
        dtype="float64",
        crs="EPSG:32612",
        transform=from_origin(400000.0, 3700000.0, 5.0, 5.0),
        nodata=np.nan,
    ) as dataset:
        dataset.write(elevation, 1)

    result = derive_terrain(
        elevation_path,
        output_directory=tmp_path / "terrain",
        provenance={"source_elevation_units": "feet"},
    )

    assert result.slope.is_file()
    assert result.curvature.is_file()
    assert result.provenance_path is not None and result.provenance_path.is_file()
    assert result.provenance["analysis_crs"] == "EPSG:32612"
    assert result.provenance["slope_units"] == "degrees"
    assert result.provenance["curvature_units"] == "m^-1"
    with rasterio.open(result.slope) as slope:
        assert slope.crs.to_epsg() == 32612
        assert slope.transform == from_origin(400000.0, 3700000.0, 5.0, 5.0)
        values = slope.read(1)
    np.testing.assert_allclose(values[1:-1, 1:-1], 0.0)
    _assert_nodata_border(values)


def test_geographic_raster_is_rejected_for_derivatives(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from geofarmai.terrain import derive_terrain

    elevation_path = tmp_path / "geographic.tif"
    with rasterio.open(
        elevation_path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=from_origin(-112.0, 34.0, 0.001, 0.001),
        nodata=np.nan,
    ) as dataset:
        dataset.write(np.ones((3, 3)), 1)

    with pytest.raises(TerrainProcessingError, match="projected CRS"):
        derive_terrain(elevation_path, output_directory=tmp_path / "terrain")


def test_point_workflow_reuses_kriging_and_converts_feet(monkeypatch, tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("matplotlib")
    pd = pytest.importorskip("pandas")
    import core.raster_pipeline as raster_pipeline

    from geofarmai.terrain import terrain_from_elevation_points

    observed = {}

    def fake_kriging(x, y, values, grid_x, grid_y, config, coordinates_type):
        observed["values"] = values.copy()
        observed["coordinates_type"] = coordinates_type
        observed["config"] = dict(config)
        return np.full((len(grid_y), len(grid_x)), values.mean()), np.array([1.0, 2.0])

    monkeypatch.setattr(raster_pipeline, "ordinary_krige", fake_kriging)
    points = pd.DataFrame(
        {
            "easting": [400000.0, 400015.0, 400000.0, 400015.0],
            "northing": [3700000.0, 3700000.0, 3700015.0, 3700015.0],
            "height": [100.0, 101.0, 102.0, 103.0],
        }
    )

    result = terrain_from_elevation_points(
        points,
        x_column="easting",
        y_column="northing",
        elevation_column="height",
        source_crs="EPSG:32612",
        elevation_units="feet",
        cell_size_m=5.0,
        output_directory=tmp_path / "terrain",
        interpolation_config={"variogram_model": "linear", "nlags": 4},
    )

    np.testing.assert_allclose(observed["values"], points["height"] * 0.3048)
    assert observed["coordinates_type"] == "euclidean"
    assert observed["config"] == {"variogram_model": "linear", "nlags": 4}
    assert result.provenance["elevation_conversion_factor_to_metres"] == pytest.approx(
        0.3048
    )
    assert result.provenance["interpolation_method"] == "ordinary kriging"
    assert result.provenance["interpolation_configuration"]["variogram_model"] == "linear"
    assert result.predictor_rasters["elevation"].is_file()
