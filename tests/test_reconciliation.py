from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box

from core.reconcile import idw, populate_analysis_grid, populate_grid


def test_idw_reproduces_samples_at_their_locations_and_interpolates_midpoint():
    samples = gpd.GeoDataFrame(
        geometry=[Point(0.0, 0.0), Point(10.0, 0.0)],
        crs="EPSG:32612",
    )
    targets = gpd.GeoDataFrame(
        geometry=[Point(0.0, 0.0), Point(5.0, 0.0), Point(10.0, 0.0)],
        crs=samples.crs,
    )

    estimates = idw(samples, np.array([0.0, 10.0]), targets, k=2, power=2.0)

    np.testing.assert_allclose(estimates, [0.0, 5.0, 10.0], atol=1e-8, rtol=1e-8)


def test_populate_grid_preserves_current_predictor_and_yield_interpolation():
    soil = gpd.GeoDataFrame(
        {"feature": [0.0, 10.0]},
        geometry=[Point(0.0, 5.0), Point(20.0, 5.0)],
        crs="EPSG:32612",
    )
    outcome = gpd.GeoDataFrame(
        {"yield": [100.0, 200.0]},
        geometry=[Point(0.0, 5.0), Point(20.0, 5.0)],
        crs=soil.crs,
    )
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs=soil.crs,
    )

    reconciled = populate_grid(soil, outcome, grid, ["feature"], method="idw")

    expected_feature = idw(soil, soil["feature"].to_numpy(), grid)
    expected_yield = idw(outcome, outcome["yield"].to_numpy(), grid)
    np.testing.assert_allclose(reconciled["feature"], expected_feature)
    np.testing.assert_allclose(reconciled["yield"], expected_yield)
    assert reconciled.geometry.equals(grid.geometry)


def test_populate_grid_nearest_neighbor_reconciliation():
    soil = gpd.GeoDataFrame(
        {"feature": [0.0, 10.0]},
        geometry=[Point(5.0, 5.0), Point(15.0, 5.0)],
        crs="EPSG:32612",
    )
    outcome = gpd.GeoDataFrame(
        {"yield": [100.0, 200.0]},
        geometry=[Point(5.0, 5.0), Point(15.0, 5.0)],
        crs=soil.crs,
    )
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs=soil.crs,
    )

    reconciled = populate_grid(soil, outcome, grid, ["feature"], method="nearest")

    np.testing.assert_allclose(reconciled["feature"], [0.0, 10.0])
    np.testing.assert_allclose(reconciled["yield"], [100.0, 200.0])


def test_generic_reconciliation_accepts_nitrate_as_optional_outcome():
    predictors = gpd.GeoDataFrame(
        {"feature": [0.0, 10.0]},
        geometry=[Point(0.0, 5.0), Point(20.0, 5.0)],
        crs="EPSG:32612",
    )
    outcomes = gpd.GeoDataFrame(
        {"nitrate": [2.0, 8.0]},
        geometry=[Point(0.0, 5.0), Point(20.0, 5.0)],
        crs=predictors.crs,
    )
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1]},
        geometry=[box(0.0, 0.0, 10.0, 10.0), box(10.0, 0.0, 20.0, 10.0)],
        crs=predictors.crs,
    )

    reconciled = populate_analysis_grid(
        predictors,
        grid,
        ["feature"],
        outcome_points=outcomes,
        outcome_name="nitrate",
    )

    np.testing.assert_allclose(
        reconciled["nitrate"], idw(outcomes, outcomes["nitrate"].to_numpy(), grid)
    )
    assert list(reconciled.columns).count("nitrate") == 1


def test_populate_grid_buffer_mean_handles_zero_one_multiple_and_duplicate_neighbors():
    soil = gpd.GeoDataFrame(
        {"feature": [10.0, 2.0, 6.0, 3.0, 9.0]},
        geometry=[
            Point(25.0, 5.0),
            Point(42.0, 5.0),
            Point(48.0, 5.0),
            Point(65.0, 5.0),
            Point(65.0, 5.0),
        ],
        crs="EPSG:32612",
    )
    outcome = gpd.GeoDataFrame(
        {"yield": [100.0, 20.0, 60.0, 30.0, 90.0]},
        geometry=soil.geometry.copy(),
        crs=soil.crs,
    )
    grid = gpd.GeoDataFrame(
        {"cell_id": [0, 1, 2, 3]},
        geometry=[
            box(0.0, 0.0, 10.0, 10.0),
            box(20.0, 0.0, 30.0, 10.0),
            box(40.0, 0.0, 50.0, 10.0),
            box(60.0, 0.0, 70.0, 10.0),
        ],
        crs=soil.crs,
    )

    reconciled = populate_grid(soil, outcome, grid, ["feature"], method="buffer_mean", buffer_m=0.0)

    assert np.isnan(reconciled.loc[0, "feature"])
    assert np.isnan(reconciled.loc[0, "yield"])
    np.testing.assert_allclose(reconciled.loc[1:, "feature"], [10.0, 4.0, 6.0])
    np.testing.assert_allclose(reconciled.loc[1:, "yield"], [100.0, 40.0, 60.0])


def test_vector_kriging_fails_loudly_without_calling_idw(monkeypatch):
    points = gpd.GeoDataFrame(
        {"feature": [1.0], "yield": [10.0]},
        geometry=[Point(5.0, 5.0)],
        crs="EPSG:32612",
    )
    grid = gpd.GeoDataFrame(
        {"cell_id": [0]},
        geometry=[box(0.0, 0.0, 10.0, 10.0)],
        crs=points.crs,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("IDW must not execute for method='kriging'")

    monkeypatch.setattr("core.reconcile.idw", fail_if_called)

    with pytest.raises(ValueError, match="Vector reconciliation method 'kriging' is not supported"):
        populate_grid(points, points, grid, ["feature"], method="kriging")
