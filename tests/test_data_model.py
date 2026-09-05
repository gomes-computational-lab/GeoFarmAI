from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import Point

from geofarmai import (
    CoordinateSpec,
    DataModelError,
    DataSource,
    DataSourceError,
    FieldDataset,
    SchemaValidationError,
    VariableRole,
    VariableSpec,
)


def _frame(**columns) -> pd.DataFrame:
    defaults = {"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]}
    defaults.update(columns)
    return pd.DataFrame(defaults)


def _source(
    source_id: str,
    variable: str,
    role: str,
    *,
    domain: str | None = None,
    crs: str = "EPSG:32615",
) -> DataSource:
    return DataSource.from_dataframe(
        _frame(**{variable: [1.0, 2.0, 3.0]}),
        source_id=source_id,
        variables=[VariableSpec(variable, role, domain=domain)],
        x="x",
        y="y",
        crs=crs,
    )


def test_field_dataset_from_distributed_soil_csv(sample_csv_paths):
    soil_path, _ = sample_csv_paths

    dataset = FieldDataset.from_csv(
        soil_path,
        source_id="soil_observations",
        variables=[
            VariableSpec("Slope", "predictor", domain="soil", units="degrees"),
            VariableSpec("EC_DP", "predictor", domain="soil"),
        ],
        x="Longitude",
        y="Latitude",
        crs="EPSG:4326",
    )

    source = dataset.sources[0]
    assert source.path == soil_path
    assert len(source.data) > 100
    assert dataset.predictor_names == ("Slope", "EC_DP")
    assert dataset.outcomes == ()
    assert not dataset.has_outcomes
    assert source.coordinate_spec.is_geographic is True
    assert source.variables_for_role("coordinate") == (
        VariableSpec("Longitude", "coordinate"),
        VariableSpec("Latitude", "coordinate"),
    )


def test_field_dataset_from_dataframe_supports_explicit_roles_without_outcome():
    frame = _frame(
        nitrate=[4.1, 4.3, 4.5],
        field_id=["a", "b", "c"],
        season=["spring", "spring", "summer"],
    )

    dataset = FieldDataset.from_dataframe(
        frame,
        source_id="sensor",
        variables=[
            VariableSpec("nitrate", VariableRole.PREDICTOR, domain="soil", units="mg/kg"),
            VariableSpec("field_id", "identifier"),
            VariableSpec("season", "metadata"),
        ],
        coordinates=CoordinateSpec("x", "y", "EPSG:32615"),
        metadata={"instrument": "synthetic"},
    )

    assert dataset.predictor_names == ("nitrate",)
    assert dataset.outcome_names == ()
    assert dataset.sources[0].identifiers[0].name == "field_id"
    assert dataset.sources[0].metadata_variables[0].name == "season"
    assert dataset.sources[0].metadata == {"instrument": "synthetic"}
    assert dataset.variable_metadata["nitrate"]["domain"] == "soil"


def test_field_dataset_from_geodataframe_uses_geometry_and_crs():
    original = gpd.GeoDataFrame(
        {"biomass": [1.2, 2.4], "protein": [10.0, 11.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:32615",
    )

    dataset = FieldDataset.from_geodataframe(
        original,
        source_id="harvest_quality",
        variables=[
            VariableSpec("biomass", "predictor", domain="crop"),
            VariableSpec("protein", "outcome", domain="crop"),
        ],
    )

    source = dataset.sources[0]
    assert source.coordinate_spec.uses_geometry
    assert source.crs.to_epsg() == 32615
    assert source.variables_for_role("coordinate") == (
        VariableSpec("geometry", "coordinate"),
    )
    assert dataset.common_crs.to_epsg() == 32615
    assert dataset.predictor_names == ("biomass",)
    assert dataset.outcome_names == ("protein",)
    original.loc[0, "biomass"] = 999.0
    assert source.data.loc[0, "biomass"] == pytest.approx(1.2)


def test_field_dataset_preserves_multiple_unharmonized_sources():
    soil = _source("soil", "EC", "predictor", domain="soil")
    profitability = DataSource.from_dataframe(
        pd.DataFrame(
            {"east": [100.0, 120.0], "north": [200.0, 220.0], "profit": [25.0, 30.0]}
        ),
        source_id="economics",
        variables=[VariableSpec("profit", "outcome", domain="economics")],
        x="east",
        y="north",
        crs="EPSG:32615",
    )

    dataset = FieldDataset.from_sources([soil, profitability])

    assert [source.source_id for source in dataset.source_list] == ["soil", "economics"]
    assert dataset.predictor_names == ("EC",)
    assert dataset.outcome_names == ("profit",)
    assert len(dataset.get_source("soil").data) == 3
    assert len(dataset.get_source("economics").data) == 2


@pytest.mark.parametrize("role", ["predictor", "outcome"])
def test_nitrate_role_is_exactly_what_the_user_declares(role):
    dataset = FieldDataset.from_dataframe(
        _frame(nitrate=[1.0, 2.0, 3.0]),
        variables=[VariableSpec("nitrate", role, domain="environmental")],
        x="x",
        y="y",
    )

    assert dataset.variable_metadata["nitrate"]["role"] == role
    assert ("nitrate" in dataset.predictor_names) is (role == "predictor")
    assert ("nitrate" in dataset.outcome_names) is (role == "outcome")


def test_yield_is_an_outcome_only_when_explicitly_declared():
    dataset = FieldDataset.from_dataframe(
        _frame(**{"yield": [100.0, 110.0, 120.0]}),
        variables=[VariableSpec("yield", "outcome", domain="crop")],
        x="x",
        y="y",
    )

    assert dataset.predictors == ()
    assert dataset.outcome_names == ("yield",)


def test_numeric_predictor_and_outcome_missing_values_are_preserved():
    frame = _frame(predictor=[1.0, np.nan, 3.0], outcome=[np.nan, 5.0, 6.0])
    dataset = FieldDataset.from_dataframe(
        frame,
        variables=[
            VariableSpec("predictor", "predictor"),
            VariableSpec("outcome", "outcome"),
        ],
        x="x",
        y="y",
    )

    assert dataset.sources[0].data["predictor"].isna().sum() == 1
    assert dataset.sources[0].data["outcome"].isna().sum() == 1


def test_legacy_config_adapter_preserves_names_and_explicit_semantics(
    repository_root,
):
    with (repository_root / "configs" / "project.yaml").open() as stream:
        config = yaml.safe_load(stream)

    dataset = FieldDataset.from_legacy_config(config, base_path=repository_root)

    assert [source.source_id for source in dataset.sources] == ["soil", "yield"]
    assert "EC_DP" in dataset.predictor_names
    assert "Yld_Mass_Dry_lb_ac" in dataset.outcome_names
    assert "yield" not in dataset.outcome_names
    assert dataset.get_source("soil").metadata["legacy_section"] == "soil"


@pytest.mark.parametrize(
    ("role", "message"),
    [("target", "Invalid role"), (None, "Invalid role")],
)
def test_invalid_variable_roles_fail_clearly(role, message):
    with pytest.raises(SchemaValidationError, match=message):
        VariableSpec("value", role)


@pytest.mark.parametrize("role", ["predictor", "outcome"])
def test_nonnumeric_scientific_variables_are_rejected(role):
    with pytest.raises(SchemaValidationError, match="must be numeric"):
        FieldDataset.from_dataframe(
            _frame(value=["low", "medium", "high"]),
            variables=[VariableSpec("value", role)],
            x="x",
            y="y",
        )


def test_empty_input_is_rejected():
    with pytest.raises(DataSourceError, match="no observations"):
        FieldDataset.from_dataframe(
            pd.DataFrame(columns=["x", "y", "value"]),
            variables=[VariableSpec("value", "predictor")],
            x="x",
            y="y",
        )


def test_missing_csv_is_rejected(tmp_path):
    with pytest.raises(DataSourceError, match="does not exist"):
        FieldDataset.from_csv(
            tmp_path / "missing.csv",
            variables=[VariableSpec("value", "predictor")],
            x="x",
            y="y",
        )


def test_missing_coordinate_columns_are_rejected():
    with pytest.raises(SchemaValidationError, match="missing coordinate columns"):
        FieldDataset.from_dataframe(
            pd.DataFrame({"x": [1.0], "value": [2.0]}),
            variables=[VariableSpec("value", "predictor")],
            x="x",
            y="y",
        )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"x": ["west"], "y": [1.0], "value": [2.0]}),
        pd.DataFrame({"x": [np.nan], "y": [1.0], "value": [2.0]}),
    ],
)
def test_invalid_coordinate_values_are_rejected(frame):
    with pytest.raises(SchemaValidationError, match="Coordinate column"):
        FieldDataset.from_dataframe(
            frame,
            variables=[VariableSpec("value", "predictor")],
            x="x",
            y="y",
        )


def test_missing_declared_variable_is_rejected():
    with pytest.raises(SchemaValidationError, match="missing declared variables"):
        FieldDataset.from_dataframe(
            _frame(value=[1.0, 2.0, 3.0]),
            variables=[VariableSpec("not_present", "predictor")],
            x="x",
            y="y",
        )


def test_duplicate_variable_declarations_are_rejected():
    with pytest.raises(SchemaValidationError, match="more than once"):
        FieldDataset.from_dataframe(
            _frame(value=[1.0, 2.0, 3.0]),
            variables=[
                VariableSpec("value", "predictor"),
                VariableSpec("value", "predictor"),
            ],
            x="x",
            y="y",
        )


def test_duplicate_input_columns_are_rejected():
    frame = pd.DataFrame([[0.0, 1.0, 2.0, 3.0]], columns=["x", "y", "value", "value"])
    with pytest.raises(SchemaValidationError, match="duplicate columns"):
        FieldDataset.from_dataframe(
            frame,
            variables=[VariableSpec("value", "predictor")],
            x="x",
            y="y",
        )


def test_coordinate_role_conflict_is_rejected():
    with pytest.raises(SchemaValidationError, match="declared coordinate"):
        FieldDataset.from_dataframe(
            _frame(value=[1.0, 2.0, 3.0]),
            variables=[
                VariableSpec("x", "predictor"),
                VariableSpec("value", "predictor"),
            ],
            x="x",
            y="y",
        )


def test_conflicting_cross_source_variable_specifications_are_rejected():
    predictor = _source("sensor", "nitrate", "predictor")
    outcome = _source("lab", "nitrate", "outcome")

    with pytest.raises(SchemaValidationError, match="conflicting specifications"):
        FieldDataset.from_sources([predictor, outcome])


def test_duplicate_source_identifiers_are_rejected():
    first = _source("same", "EC", "predictor")
    second = _source("same", "biomass", "predictor")

    with pytest.raises(DataModelError, match="duplicate source identifiers"):
        FieldDataset.from_sources([first, second])


def test_invalid_crs_is_rejected():
    with pytest.raises(SchemaValidationError, match="Invalid CRS"):
        CoordinateSpec("x", "y", "definitely-not-a-crs")


def test_geodataframe_conflicting_crs_is_rejected():
    frame = gpd.GeoDataFrame(
        {"value": [1.0]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    with pytest.raises(SchemaValidationError, match="conflicting CRS"):
        FieldDataset.from_geodataframe(
            frame,
            variables=[VariableSpec("value", "predictor")],
            crs="EPSG:32615",
        )


def test_field_dataset_requires_at_least_one_source():
    with pytest.raises(DataModelError, match="at least one"):
        FieldDataset.from_sources([])


def test_unknown_source_identifier_fails_clearly():
    dataset = FieldDataset.from_sources([_source("soil", "EC", "predictor")])
    with pytest.raises(DataModelError, match="no source"):
        dataset.get_source("missing")
