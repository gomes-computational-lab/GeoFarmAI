from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from geofarmai import (
    DataSource,
    FieldDataset,
    HarmonizationError,
    HarmonizationWarning,
    VariableIdentity,
    VariableSpec,
    harmonize,
)


PROJECTED_CRS = "EPSG:32615"


def _source(
    source_id: str,
    coordinates: list[tuple[float, float]],
    variables: dict[str, tuple[str, list[float]]],
    *,
    crs: str = PROJECTED_CRS,
) -> DataSource:
    frame = pd.DataFrame(
        {
            "x": [coordinate[0] for coordinate in coordinates],
            "y": [coordinate[1] for coordinate in coordinates],
            **{name: values for name, (_, values) in variables.items()},
        }
    )
    return DataSource.from_dataframe(
        frame,
        source_id=source_id,
        variables=[VariableSpec(name, role) for name, (role, _) in variables.items()],
        x="x",
        y="y",
        crs=crs,
    )


def test_one_csv_preserves_colocated_predictors_and_yield_without_interpolation(tmp_path):
    path = tmp_path / "field.csv"
    pd.DataFrame(
        {
            "x": [0.0, 10.0, 0.0, 10.0],
            "y": [0.0, 0.0, 10.0, 10.0],
            "ec": [1.0, 2.0, 3.0, 4.0],
            "yield": [100.0, 110.0, 120.0, 130.0],
        }
    ).to_csv(path, index=False)
    dataset = FieldDataset.from_csv(
        path,
        source_id="combined",
        variables=[VariableSpec("ec", "predictor"), VariableSpec("yield", "outcome")],
        x="x",
        y="y",
        crs=PROJECTED_CRS,
    )

    result = harmonize(dataset)

    assert result.actual_strategy == "direct_reuse"
    np.testing.assert_allclose(result.predictors[("combined", "ec")], [1, 2, 3, 4])
    np.testing.assert_allclose(result.outcomes[("combined", "yield")], [100, 110, 120, 130])
    assert VariableIdentity("combined", "yield") not in result.predictor_identities
    assert result.variable_provenance[VariableIdentity("combined", "ec")].alignment_method == "direct"


def test_two_predictor_sources_at_identical_locations_align_without_interpolation():
    locations = [(0, 0), (10, 0), (0, 10), (10, 10)]
    first = _source("soil", locations, {"ec": ("predictor", [1, 2, 3, 4])})
    second = _source(
        "terrain",
        list(reversed(locations)),
        {"elevation": ("predictor", [40, 30, 20, 10])},
    )

    result = harmonize(FieldDataset.from_sources([first, second]))

    assert result.actual_strategy == "direct_alignment"
    np.testing.assert_allclose(result.predictors[("soil", "ec")], [1, 2, 3, 4])
    np.testing.assert_allclose(result.predictors[("terrain", "elevation")], [10, 20, 30, 40])
    assert {
        item.alignment_method for item in result.variable_provenance.values()
    } == {"direct_alignment"}


def test_different_predictor_locations_use_existing_grid_and_idw():
    first = _source(
        "soil",
        [(0, 0), (20, 0), (0, 20), (20, 20)],
        {"ec": ("predictor", [1, 2, 3, 4])},
    )
    second = _source(
        "sensor",
        [(5, 5), (15, 5), (5, 15), (15, 15)],
        {"moisture": ("predictor", [10, 20, 30, 40])},
    )

    result = harmonize(FieldDataset.from_sources([first, second]), cell_size=10.0)

    assert result.actual_strategy == "grid_reconciliation"
    assert len(result.geometry) == 4
    assert result.geometry.geom_type.eq("Polygon").all()
    assert result.predictors.shape == (4, 2)
    assert np.isfinite(result.predictors.to_numpy()).all()
    assert result.harmonization_metadata["grid_resolution_m"] == pytest.approx(10.0)
    assert {
        item.alignment_method for item in result.variable_provenance.values()
    } == {"idw"}


def test_sources_in_different_crs_are_transformed_to_existing_auto_utm():
    geographic = gpd.GeoDataFrame(
        {"ec": [1.0, 2.0, 3.0]},
        geometry=[Point(-91.5, 44.8), Point(-91.499, 44.8), Point(-91.5, 44.801)],
        crs="EPSG:4326",
    )
    projected = geographic.to_crs(PROJECTED_CRS)
    projected["elevation"] = [100.0, 101.0, 102.0]
    first = DataSource.from_geodataframe(
        geographic,
        source_id="soil",
        variables=[VariableSpec("ec", "predictor")],
    )
    second = DataSource.from_geodataframe(
        projected[["elevation", "geometry"]],
        source_id="terrain",
        variables=[VariableSpec("elevation", "predictor")],
    )

    result = harmonize(FieldDataset.from_sources([first, second]), location_tolerance=1e-4)

    assert result.crs.to_epsg() == 32615
    assert result.crs.is_projected
    assert result.actual_strategy == "direct_alignment"
    assert result.harmonization_metadata["target_crs_policy"] == "existing_auto_utm"
    assert result.source_provenance["soil"]["original_crs"] == "EPSG:4326"
    assert result.source_provenance["terrain"]["original_crs"] == PROJECTED_CRS


def test_differently_located_outcome_is_reconciled_to_predictor_support_only():
    predictors = _source(
        "soil",
        [(0, 0), (10, 0), (0, 10), (10, 10)],
        {"ec": ("predictor", [1, 2, 3, 4])},
    )
    outcome = _source(
        "harvest",
        [(1, 1), (9, 1), (1, 9), (9, 9)],
        {"yield": ("outcome", [100, 110, 120, 130])},
    )

    result = harmonize(FieldDataset.from_sources([predictors, outcome]))

    assert result.actual_strategy == "direct_alignment_with_reconciliation"
    np.testing.assert_allclose(result.predictors[("soil", "ec")], [1, 2, 3, 4])
    assert result.outcomes.shape == (4, 1)
    assert VariableIdentity("harvest", "yield") not in result.predictor_identities
    assert result.variable_provenance[
        VariableIdentity("harvest", "yield")
    ].alignment_method == "idw"


@pytest.mark.parametrize("role", ["predictor", "outcome"])
def test_nitrate_obeys_explicit_role_and_never_name_based_inference(role):
    source = _source(
        "lab",
        [(0, 0), (10, 0), (0, 10)],
        {
            "ec": ("predictor", [1, 2, 3]),
            "nitrate": (role, [5, 6, 7]),
        },
    )

    result = harmonize(FieldDataset.from_sources([source]))

    identity = VariableIdentity("lab", "nitrate")
    assert (identity in result.predictor_identities) is (role == "predictor")
    assert (identity in result.outcome_identities) is (role == "outcome")


def test_duplicate_variable_names_from_separate_sources_remain_distinct():
    locations = [(0, 0), (10, 0), (0, 10)]
    first = _source("probe_1", locations, {"moisture": ("predictor", [1, 2, 3])})
    second = _source("probe_2", locations, {"moisture": ("predictor", [11, 12, 13])})

    result = harmonize(FieldDataset.from_sources([first, second]))

    assert result.predictor_identities == (
        VariableIdentity("probe_1", "moisture"),
        VariableIdentity("probe_2", "moisture"),
    )
    np.testing.assert_allclose(result.predictors[("probe_1", "moisture")], [1, 2, 3])
    np.testing.assert_allclose(result.predictors[("probe_2", "moisture")], [11, 12, 13])
    assert result.display_names == {
        VariableIdentity("probe_1", "moisture"): "probe_1:moisture",
        VariableIdentity("probe_2", "moisture"): "probe_2:moisture",
    }


def test_predictors_only_produce_empty_outcome_matrix():
    source = _source(
        "soil", [(0, 0), (10, 0), (0, 10)], {"ec": ("predictor", [1, 2, 3])}
    )

    result = harmonize(FieldDataset.from_sources([source]))

    assert result.predictors.shape == (3, 1)
    assert result.outcomes.shape == (3, 0)
    assert result.outcome_identities == ()
    assert result.outcome_arrays == {}


def test_geometry_backed_source_uses_geometry_not_physical_xy_columns():
    frame = gpd.GeoDataFrame(
        {"x": [999.0, 999.0], "y": [999.0, 999.0], "ec": [1.0, 2.0]},
        geometry=[Point(0, 0), Point(10, 20)],
        crs=PROJECTED_CRS,
    )
    source = DataSource.from_geodataframe(
        frame,
        source_id="geometry_source",
        variables=[VariableSpec("ec", "predictor")],
        x="x",
        y="y",
        crs=PROJECTED_CRS,
    )

    result = harmonize(FieldDataset.from_sources([source]))

    np.testing.assert_allclose(result.coordinate_array(), [[0, 0], [10, 20]])


def test_buffer_mean_preserves_missing_coverage_without_filling():
    predictors = _source(
        "soil", [(0, 0), (10, 0)], {"ec": ("predictor", [1, 2])}
    )
    outcome = _source(
        "lab", [(0.5, 0)], {"nitrate": ("outcome", [8.0])}
    )

    result = harmonize(
        FieldDataset.from_sources([predictors, outcome]),
        method="buffer_mean",
        buffer_m=1.0,
    )

    aligned = result.outcomes[("lab", "nitrate")]
    assert aligned.iloc[0] == pytest.approx(8.0)
    assert np.isnan(aligned.iloc[1])
    assert result.coverage.loc[("lab", "nitrate"), "harmonized_missing"] == 1
    assert result.coverage.loc[("lab", "nitrate"), "coverage_fraction"] == pytest.approx(0.5)


def test_nonoverlapping_predictor_sources_warn_and_record_provenance():
    first = _source(
        "west", [(0, 0), (0, 10)], {"ec": ("predictor", [1, 2])}
    )
    second = _source(
        "east", [(100, 0), (100, 10)], {"slope": ("predictor", [3, 4])}
    )

    with pytest.warns(HarmonizationWarning, match="no intersecting spatial extent"):
        result = harmonize(
            FieldDataset.from_sources([first, second]), cell_size=10.0
        )

    assert result.harmonization_metadata["overlap_warnings"]


@pytest.mark.parametrize("method", ["kriging", "spline"])
def test_unsupported_vector_harmonization_methods_fail_loudly(method):
    source = _source(
        "soil", [(0, 0), (10, 0)], {"ec": ("predictor", [1, 2])}
    )

    with pytest.raises(HarmonizationError, match="not supported|Unsupported"):
        harmonize(FieldDataset.from_sources([source]), method=method)


def test_missing_crs_fails_when_harmonization_requires_spatial_meaning():
    source = _source(
        "soil", [(0, 0), (10, 0)], {"ec": ("predictor", [1, 2])}, crs=None
    )

    with pytest.raises(HarmonizationError, match="requires a CRS"):
        harmonize(FieldDataset.from_sources([source]))
