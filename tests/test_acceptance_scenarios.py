"""Release-blocking end-to-end acceptance scenarios for GeoFarmAI.

These tests deliberately run the real public scientific pipeline. They use no
network, LLM, server, R, mock, private helper, or legacy job API.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from examples.heterogeneous_yield import run as run_heterogeneous_yield
from examples.nitrate_outcome import run as run_nitrate_outcome
from examples.predictors_only import run as run_predictors_only
from examples.single_csv_yield import run as run_single_csv_yield
from geofarmai import (
    CoordinateSpec,
    DataSource,
    FieldDataset,
    GeoFarmModel,
    GeoFarmResult,
    VariableSpec,
    load_config,
    run_from_config,
)


pytestmark = pytest.mark.acceptance

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"
DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
CRS = "EPSG:32615"
SCENARIOS = {
    "single_csv_yield": run_single_csv_yield,
    "heterogeneous_yield": run_heterogeneous_yield,
    "nitrate_outcome": run_nitrate_outcome,
    "predictors_only": run_predictors_only,
}


def _same_partition(left, right) -> bool:
    """Compare cluster membership without assigning meaning to label IDs."""

    left = np.asarray(left)
    right = np.asarray(right)
    return bool(
        np.array_equal(
            left[:, None] == left[None, :],
            right[:, None] == right[None, :],
        )
    )


def _assert_metric_maps_equivalent(expected, actual) -> None:
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], rel=1e-12, abs=1e-12)


def _assert_equivalent(expected: GeoFarmResult, actual: GeoFarmResult) -> None:
    """Compare deterministic scientific results without ordering label IDs."""

    assert actual.predictor_names == expected.predictor_names
    assert actual.outcome_names == expected.outcome_names
    assert actual.selection_metric == expected.selection_metric
    assert actual.selection_outcome == expected.selection_outcome
    assert actual.harmonized_data.actual_strategy == expected.harmonized_data.actual_strategy
    assert len(actual.candidate_solutions) == len(expected.candidate_solutions)

    for left, right in zip(expected.candidate_solutions, actual.candidate_solutions):
        assert right.solution_id == left.solution_id
        assert _same_partition(left.labels, right.labels)
        _assert_metric_maps_equivalent(left.internal_metrics, right.internal_metrics)
        assert set(right.outcome_metrics) == set(left.outcome_metrics)
        for outcome_name in left.outcome_metrics:
            _assert_metric_maps_equivalent(
                left.outcome_metrics[outcome_name], right.outcome_metrics[outcome_name]
            )

    if expected.selected_solution is None:
        assert actual.selected_solution is None
    else:
        assert actual.selected_solution is not None
        assert actual.selected_solution.solution_id == expected.selected_solution.solution_id
        assert _same_partition(
            expected.selected_solution.labels, actual.selected_solution.labels
        )


def _assert_common_provenance(
    result: GeoFarmResult,
    *,
    sources: set[str],
    predictor_identities: set[tuple[str, str]],
    outcome_identities: set[tuple[str, str]],
    strategy: str,
    selection: str | None,
) -> None:
    harmonization = result.harmonization_provenance
    assert set(harmonization["sources"]) == sources
    assert all(harmonization["sources"][source]["path"] for source in sources)
    assert {
        identity.tuple for identity in result.harmonized_data.predictor_identities
    } == predictor_identities
    assert {
        identity.tuple for identity in result.harmonized_data.outcome_identities
    } == outcome_identities
    assert set(harmonization["variables"]) == {
        *result.harmonized_data.predictor_identities,
        *result.harmonized_data.outcome_identities,
    }
    assert harmonization["metadata"]["actual_strategy"] == strategy
    assert result.decomposition_provenance["requested_method"] == "pca"
    assert result.decomposition_provenance["actual_method"] == "pca"
    assert result.configuration["clustering"] == ("kmeans",)
    assert result.configuration["k"] == (2, 3)
    assert result.configuration["random_state"] == 42
    assert result.configuration["selection"] == selection
    assert all(candidate.random_state == 42 for candidate in result.candidate_solutions)


def _assert_export(result: GeoFarmResult, output: Path, *, zones: bool) -> None:
    artifacts = result.export(output)

    assert artifacts == result.artifacts
    assert set(artifacts) == {"geopackage", "metrics", "candidates"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    leaderboard = pd.read_csv(artifacts["candidates"])
    assert len(leaderboard) == len(result.candidate_solutions)
    assert leaderboard["solution_id"].tolist() == result.leaderboard[
        "solution_id"
    ].tolist()
    metrics = pd.read_csv(artifacts["metrics"])
    assert len(metrics) == 1
    layers = set(gpd.io.file.fiona.listlayers(artifacts["geopackage"]))
    assert "samples" in layers
    assert ("zones" in layers) is zones
    assert not gpd.read_file(artifacts["geopackage"], layer="samples").empty
    if zones:
        assert not gpd.read_file(artifacts["geopackage"], layer="zones").empty


@pytest.mark.parametrize("name", tuple(SCENARIOS))
def test_acceptance_yaml_loads_runs_and_matches_python_api(name):
    """Every canonical YAML example executes equivalently to its Python twin."""

    config_path = CONFIG_DIR / f"{name}.yaml"
    configured = load_config(config_path)
    assert configured.configuration_path == config_path.resolve()
    assert all(source.resolved_path.is_file() for source in configured.sources)

    python_result = SCENARIOS[name]()
    yaml_result = run_from_config(config_path)

    _assert_equivalent(python_result, yaml_result)


def test_acceptance_1_single_csv_yield(tmp_path):
    result = run_single_csv_yield()

    assert isinstance(result, GeoFarmResult)
    assert len(result.harmonized_data.geometry) == 12
    assert result.predictor_names == ("EC", "moisture", "elevation")
    assert result.outcome_names == ("yield",)
    assert ("field_observations", "yield") not in set(
        result.harmonized_data.predictor_matrix.columns
    )
    assert result.component_loadings is not None
    assert "yield" not in result.component_loadings.columns
    assert len(result.candidate_solutions) == 2
    assert result.selected_solution is not None
    assert result.zone_labels is not None and len(result.zone_labels) == 12
    assert {candidate.k for candidate in result.candidate_solutions} == {2, 3}
    assert all(
        {"asc", "ch_score"} <= set(candidate.internal_metrics)
        for candidate in result.candidate_solutions
    )
    assert all(
        candidate.outcome_metrics["yield"]["variance_reduction"] is not None
        and candidate.outcome_metrics["yield"]["anova_p"] is not None
        for candidate in result.candidate_solutions
    )
    assert result.selected_solution is max(
        result.candidate_solutions,
        key=lambda candidate: candidate.outcome_metrics["yield"]["variance_reduction"],
    )
    _assert_common_provenance(
        result,
        sources={"field_observations"},
        predictor_identities={
            ("field_observations", "EC"),
            ("field_observations", "moisture"),
            ("field_observations", "elevation"),
        },
        outcome_identities={("field_observations", "yield")},
        strategy="direct_reuse",
        selection="variance_reduction",
    )
    _assert_export(result, tmp_path / "single_csv_yield.gpkg", zones=True)


def test_acceptance_2_heterogeneous_sources_with_yield(tmp_path):
    result = run_heterogeneous_yield()

    assert len(result.harmonized_data.geometry) == 18
    assert result.harmonized_data.actual_strategy == "grid_reconciliation"
    assert result.harmonized_data.harmonization_metadata["grid_resolution_m"] == 10.0
    assert result.predictor_names == ("EC", "moisture", "elevation", "slope")
    assert result.outcome_names == ("yield",)
    assert ("harvest", "yield") not in set(result.harmonized_data.predictor_matrix.columns)
    assert result.zone_labels is not None
    assert len(result.candidate_solutions) == 2
    assert result.selected_solution is not None
    assert "yield" in result.outcome_validation_metrics
    assert all(
        provenance.alignment_method == "idw"
        for provenance in result.harmonized_data.variable_provenance.values()
    )
    _assert_common_provenance(
        result,
        sources={"soil", "terrain", "harvest"},
        predictor_identities={
            ("soil", "EC"),
            ("soil", "moisture"),
            ("terrain", "elevation"),
            ("terrain", "slope"),
        },
        outcome_identities={("harvest", "yield")},
        strategy="grid_reconciliation",
        selection="variance_reduction",
    )
    _assert_export(result, tmp_path / "heterogeneous_yield.gpkg", zones=True)


def test_acceptance_3_nitrate_is_explicit_outcome(tmp_path):
    result = run_nitrate_outcome()

    predictor_columns = set(result.harmonized_data.predictor_matrix.columns)
    assert result.predictor_names == ("EC", "CEC", "organic_matter")
    assert result.outcome_names == ("nitrate",)
    assert ("nitrate_samples", "nitrate") in set(
        result.harmonized_data.outcome_matrix.columns
    )
    assert all(variable != "nitrate" for _, variable in predictor_columns)
    assert result.component_loadings is not None
    assert "nitrate" not in result.component_loadings.columns
    assert result.harmonized_data.actual_strategy == "direct_alignment_with_reconciliation"
    nitrate_provenance = next(
        provenance
        for identity, provenance in result.harmonized_data.variable_provenance.items()
        if identity.variable_name == "nitrate"
    )
    assert nitrate_provenance.alignment_method == "idw"
    assert result.selection_outcome == "nitrate"
    assert result.selected_solution is max(
        result.candidate_solutions,
        key=lambda candidate: candidate.outcome_metrics["nitrate"][
            "variance_reduction"
        ],
    )
    assert result.zone_labels is not None
    assert result.outcome_validation_metrics["nitrate"]["variance_reduction"] is not None
    assert result.outcome_validation_metrics["nitrate"]["anova_p"] is not None
    _assert_common_provenance(
        result,
        sources={"soil", "nitrate_samples"},
        predictor_identities={
            ("soil", "EC"),
            ("soil", "CEC"),
            ("soil", "organic_matter"),
        },
        outcome_identities={("nitrate_samples", "nitrate")},
        strategy="direct_alignment_with_reconciliation",
        selection="variance_reduction",
    )
    _assert_export(result, tmp_path / "nitrate_outcome.gpkg", zones=True)


def test_acceptance_3_companion_nitrate_role_controls_behavior():
    """The same name becomes a predictor only when its role says predictor."""

    coordinates = CoordinateSpec(x="x", y="y", crs=CRS)
    nitrate = DataSource.from_csv(
        DATA_DIR / "nitrate_outcome.csv",
        source_id="nitrate_samples",
        coordinates=coordinates,
        variables=[VariableSpec("nitrate", "predictor", domain="nutrient")],
    )
    data = FieldDataset.from_sources([nitrate])
    result = GeoFarmModel(
        decomposition="none",
        clustering="kmeans",
        k=2,
        selection="silhouette",
        random_state=42,
    ).fit(data)

    assert result.predictor_names == ("nitrate",)
    assert result.outcome_names == ()
    assert ("nitrate_samples", "nitrate") in set(
        result.harmonized_data.predictor_matrix.columns
    )
    assert result.outcome_validation_metrics == {}


def test_acceptance_4_predictors_only_selection_none(tmp_path):
    result = run_predictors_only()

    assert result.harmonized_data.actual_strategy == "grid_reconciliation"
    assert result.predictor_names == ("EC", "moisture", "elevation")
    assert result.outcome_names == ()
    assert result.harmonized_data.outcome_matrix.shape[1] == 0
    assert result.selected_solution is None
    assert result.selected_candidate is None
    assert result.zone_labels is None
    assert result.selection_metric is None
    assert result.internal_metrics == {}
    assert result.outcome_validation_metrics == {}
    assert len(result.candidate_solutions) == 2
    assert all(len(candidate.labels) == len(result.harmonized_data.geometry) for candidate in result.candidate_solutions)
    assert all(candidate.outcome_metrics == {} for candidate in result.candidate_solutions)
    assert all(
        {"asc", "ch_score"} <= set(candidate.internal_metrics)
        for candidate in result.candidate_solutions
    )
    assert not any(
        "variance_reduction" in candidate.internal_metrics
        or "anova_p" in candidate.internal_metrics
        for candidate in result.candidate_solutions
    )
    assert not result.leaderboard["selected"].any()
    assert "zone" not in result.to_dataframe().columns
    _assert_common_provenance(
        result,
        sources={"soil", "terrain"},
        predictor_identities={
            ("soil", "EC"),
            ("soil", "moisture"),
            ("terrain", "elevation"),
        },
        outcome_identities=set(),
        strategy="grid_reconciliation",
        selection=None,
    )
    _assert_export(result, tmp_path / "predictors_only.gpkg", zones=False)
    assert result.selected_solution is None
