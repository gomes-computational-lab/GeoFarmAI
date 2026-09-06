from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import same_partition
from geofarmai import (
    ConfigurationError,
    GeoFarmModel,
    VariableRole,
    load_config,
    load_legacy_config,
    run_from_config,
)


EXAMPLE_NAMES = (
    "example.yaml",
    "example_single_csv.yaml",
    "example_multiple_csv.yaml",
    "example_nitrate_outcome.yaml",
    "example_unsupervised.yaml",
    "example_multiple_outcomes.yaml",
)


def _base_document(path: str = "../data/field.csv") -> dict:
    return {
        "sources": [
            {
                "id": "field",
                "path": path,
                "coordinates": {"x": "x", "y": "y", "crs": "EPSG:32615"},
                "variables": {
                    "EC": {"role": "predictor", "domain": "soil"},
                    "yield": {"role": "outcome", "domain": "crop"},
                },
            }
        ],
        "analysis": {
            "harmonization": {"strategy": "auto", "resolution": "auto"},
            "decomposition": {"method": "none"},
            "clustering": {"methods": ["kmeans"], "k": 2},
            "selection": {"metric": "silhouette"},
            "random_state": 42,
        },
    }


def _write_project(tmp_path: Path, document: dict) -> Path:
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    data_dir.mkdir()
    config_dir.mkdir()
    frame = pd.DataFrame(
        {
            "x": np.arange(8, dtype=float),
            "y": np.tile([0.0, 1.0], 4),
            "EC": [0.1, 0.2, 0.15, 0.25, 4.8, 5.0, 5.1, 4.9],
            "yield": [10.0, 11.0, 9.0, 10.5, 50.0, 52.0, 49.0, 51.0],
            "nitrate": [12.0, 11.8, 12.1, 11.9, 7.0, 7.2, 6.9, 7.1],
            "sample_id": list(range(8)),
            "note": ["synthetic"] * 8,
        }
    )
    frame.to_csv(data_dir / "field.csv", index=False)
    path = config_dir / "project.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _legacy_document(outcome) -> dict:
    return {
        "project": {
            "crs_in": "EPSG:32615",
            "soil": {
                "path": "../data/field.csv",
                "x": "x",
                "y": "y",
                "variables": ["EC"],
            },
            "outcome": outcome,
        },
        "clustering": {"algorithms": ["kmeans"], "k_values": [2], "seeds": [42]},
    }


def _assert_result_equivalence(expected, actual):
    assert len(actual.candidate_solutions) == len(expected.candidate_solutions)
    assert actual.selection_metric == expected.selection_metric
    assert actual.selection_outcome == expected.selection_outcome
    assert (
        None if actual.selected_solution is None else actual.selected_solution.solution_id
    ) == (
        None
        if expected.selected_solution is None
        else expected.selected_solution.solution_id
    )
    for expected_candidate, actual_candidate in zip(
        expected.candidate_solutions, actual.candidate_solutions
    ):
        assert actual_candidate.solution_id == expected_candidate.solution_id
        assert same_partition(actual_candidate.labels, expected_candidate.labels)
        assert actual_candidate.internal_metrics == pytest.approx(
            expected_candidate.internal_metrics, rel=1e-12, abs=1e-12
        )
        assert actual_candidate.outcome_metrics == expected_candidate.outcome_metrics


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_distributed_examples_build_and_run(repository_root, name):
    configured = load_config(repository_root / "configs" / name)
    data, model = configured.build()

    result = model.fit(data)

    assert data.sources
    assert data.predictors
    assert result.candidate_solutions
    assert result.configuration["random_state"] == 42


def test_single_source_yaml_preserves_explicit_roles_and_metadata(repository_root):
    configured = load_config(repository_root / "configs" / "example_single_csv.yaml")
    data, model = configured.build()

    assert [source.source_id for source in data.sources] == ["observations"]
    assert data.predictor_names == ("EC", "moisture", "elevation")
    assert data.outcome_names == ("yield",)
    specs = {spec.name: spec for spec in data.sources[0].variables}
    assert specs["EC"].domain == "soil"
    assert model.selection == "variance_reduction"
    assert model.selection_outcome == "yield"


def test_multiple_sources_and_outcomes_map_to_canonical_objects(repository_root):
    configured = load_config(
        repository_root / "configs" / "example_multiple_outcomes.yaml"
    )
    data, model = configured.build()

    assert [source.source_id for source in data.sources] == ["soil", "outcomes"]
    assert data.predictor_names == ("EC", "moisture")
    assert data.outcome_names == ("yield", "nitrate")
    assert model.selection_outcome == "nitrate"
    assert tuple(model.k) == (2, 3)


def test_duplicate_variable_names_across_sources_preserve_source_identity(tmp_path):
    document = _base_document()
    document["sources"][0]["id"] = "probe_one"
    document["sources"].append(
        {
            "id": "probe_two",
            "path": "../data/field.csv",
            "coordinates": {"x": "x", "y": "y", "crs": "EPSG:32615"},
            "variables": {"EC": {"role": "predictor"}},
        }
    )
    configured = load_config(_write_project(tmp_path, document))

    result = configured.run()

    assert result.predictor_names == ("probe_one:EC", "probe_two:EC")
    assert set(result.harmonized_data.predictor_matrix.columns) == {
        ("probe_one", "EC"),
        ("probe_two", "EC"),
    }


def test_source_qualified_selection_outcome_disambiguates_duplicate_names(tmp_path):
    document = _base_document()
    document["sources"][0]["id"] = "harvest_one"
    document["sources"].append(
        {
            "id": "harvest_two",
            "path": "../data/field.csv",
            "coordinates": {"x": "x", "y": "y", "crs": "EPSG:32615"},
            "variables": {"yield": {"role": "outcome"}},
        }
    )
    document["analysis"]["selection"] = {
        "metric": "variance_reduction",
        "outcome": "harvest_two:yield",
    }
    configured = load_config(_write_project(tmp_path, document))

    result = configured.run()

    assert result.outcome_names == ("harvest_one:yield", "harvest_two:yield")
    assert result.selection_outcome == "harvest_two:yield"


def test_scientific_roles_are_never_inferred_from_names(tmp_path):
    document = _base_document()
    variables = document["sources"][0]["variables"]
    variables["yield"]["role"] = "predictor"
    variables["nitrate"] = {"role": "outcome"}
    document["analysis"]["selection"] = {
        "metric": "variance_reduction",
        "outcome": "nitrate",
    }
    configured = load_config(_write_project(tmp_path, document))
    data, _ = configured.build()

    assert data.predictor_names == ("EC", "yield")
    assert data.outcome_names == ("nitrate",)


def test_all_canonical_non_scientific_roles_are_supported(tmp_path):
    document = _base_document()
    document["sources"][0]["variables"].update(
        {
            "sample_id": {"role": "identifier"},
            "note": {"role": "metadata"},
        }
    )
    data, _ = load_config(_write_project(tmp_path, document)).build()

    by_name = {spec.name: spec.role for spec in data.sources[0].variables}
    assert by_name["sample_id"] is VariableRole.IDENTIFIER
    assert by_name["note"] is VariableRole.METADATA
    assert by_name["x"] is VariableRole.COORDINATE
    assert by_name["y"] is VariableRole.COORDINATE


def test_paths_resolve_relative_to_yaml_not_working_directory(tmp_path, monkeypatch):
    path = _write_project(tmp_path, _base_document())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    configured = load_config(path)
    source = configured.sources[0]

    assert source.configured_path == "../data/field.csv"
    assert source.resolved_path == (tmp_path / "data" / "field.csv").resolve()
    assert configured.build_dataset().sources[0].path == source.resolved_path


@pytest.mark.parametrize(
    ("configured_k", "expected"),
    [(3, 3), ({"min": 2, "max": 4}, (2, 3, 4))],
)
def test_fixed_and_inclusive_range_k_translate_without_changes(
    tmp_path, configured_k, expected
):
    document = _base_document()
    document["analysis"]["clustering"]["k"] = configured_k

    configured = load_config(_write_project(tmp_path, document))

    assert configured.model_options["k"] == expected


def test_selection_null_maps_to_no_automatic_selection(tmp_path):
    document = _base_document()
    document["analysis"]["selection"] = None
    configured = load_config(_write_project(tmp_path, document))

    result = configured.run()

    assert configured.model_options["selection"] is None
    assert result.selection_metric is None
    assert result.selected_solution is None


@pytest.mark.parametrize(
    "example_name",
    [
        "example_single_csv.yaml",
        "example_multiple_csv.yaml",
        "example_nitrate_outcome.yaml",
        "example_unsupervised.yaml",
        "example_multiple_outcomes.yaml",
    ],
)
def test_yaml_execution_matches_direct_python_model(repository_root, example_name):
    configured = load_config(repository_root / "configs" / example_name)
    data = configured.build_dataset()

    expected = GeoFarmModel(**configured.model_options).fit(data)
    actual = run_from_config(configured.configuration_path)

    _assert_result_equivalence(expected, actual)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda document: document["sources"].append(deepcopy(document["sources"][0])), "Duplicate source IDs"),
        (lambda document: document["sources"][0].pop("path"), "path must be a non-empty string"),
        (lambda document: document["sources"][0].pop("coordinates"), "coordinates must explicitly define"),
        (lambda document: document["sources"][0]["coordinates"].pop("crs"), "missing required fields"),
        (lambda document: document["sources"][0]["coordinates"].update(crs="EPSG:not-real"), "Invalid.*coordinates"),
        (lambda document: document["sources"][0]["variables"]["EC"].update(role="response"), "Invalid role"),
        (lambda document: document["analysis"]["decomposition"].update(method="ica"), "Unsupported decomposition"),
        (lambda document: document["analysis"]["clustering"].update(methods=["dbscan"]), "Unsupported clustering"),
        (lambda document: document["analysis"]["clustering"].update(k={"min": 4, "max": 2}), "2 <= min <= max"),
        (lambda document: document["analysis"]["harmonization"].update(strategy="magic"), "Unsupported harmonization strategy"),
    ],
)
def test_invalid_configuration_fails_early_with_configuration_error(
    tmp_path, mutate, message
):
    document = _base_document()
    mutate(document)
    path = _write_project(tmp_path, document)

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_missing_source_file_fails_with_resolved_context(tmp_path):
    document = _base_document("../data/missing.csv")
    path = _write_project(tmp_path, document)

    with pytest.raises(ConfigurationError, match="path does not exist.*missing.csv"):
        load_config(path)


def test_missing_csv_columns_fail_when_canonical_source_is_built(tmp_path):
    document = _base_document()
    document["sources"][0]["coordinates"]["x"] = "longitude"
    configured = load_config(_write_project(tmp_path, document))

    with pytest.raises(ConfigurationError, match="missing coordinate columns.*longitude"):
        configured.build_dataset()


def test_coordinate_cannot_be_conflicting_predictor_declaration(tmp_path):
    document = _base_document()
    document["sources"][0]["variables"]["x"] = {"role": "predictor"}
    configured = load_config(_write_project(tmp_path, document))

    with pytest.raises(ConfigurationError, match="coordinate.*role 'predictor'"):
        configured.build_dataset()


def test_duplicate_variable_yaml_key_is_rejected(tmp_path):
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    (data_dir / "field.csv").write_text("x,y,EC\n0,0,1\n", encoding="utf-8")
    path = config_dir / "duplicate.yaml"
    path.write_text(
        """sources:
  - id: field
    path: ../data/field.csv
    coordinates: {x: x, y: y, crs: EPSG:32615}
    variables:
      EC: {role: predictor}
      EC: {role: outcome}
analysis:
  decomposition: {method: none}
  clustering: {methods: [kmeans], k: 2}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="Duplicate YAML key 'EC'"):
        load_config(path)


def test_variance_reduction_requires_declared_outcome(tmp_path):
    document = _base_document()
    document["sources"][0]["variables"].pop("yield")
    document["analysis"]["selection"] = {"metric": "variance_reduction"}

    with pytest.raises(ConfigurationError, match="requires.*outcome"):
        load_config(_write_project(tmp_path, document))


def test_configuration_requires_an_explicit_predictor(tmp_path):
    document = _base_document()
    document["sources"][0]["variables"].pop("EC")

    with pytest.raises(ConfigurationError, match="at least one.*predictor"):
        load_config(_write_project(tmp_path, document))


@pytest.mark.parametrize(
    ("outcome", "message"),
    [("protein", "Unknown selection outcome"), ("EC", "declared as a predictor")],
)
def test_invalid_selection_outcome_fails_early(tmp_path, outcome, message):
    document = _base_document()
    document["analysis"]["selection"] = {
        "metric": "variance_reduction",
        "outcome": outcome,
    }

    with pytest.raises(ConfigurationError, match=message):
        load_config(_write_project(tmp_path, document))


def test_explicit_target_crs_must_be_valid_metric_projected_crs(tmp_path):
    document = _base_document()
    document["analysis"]["harmonization"]["target_crs"] = "EPSG:4326"

    with pytest.raises(ConfigurationError, match="projected CRS.*metres"):
        load_config(_write_project(tmp_path, document))


def test_legacy_loader_is_explicit_and_keeps_assumptions_isolated(repository_root):
    configured = load_legacy_config(repository_root / "configs" / "project.yaml")
    data, model = configured.build()

    assert configured.legacy is True
    assert [source.source_id for source in data.sources] == ["soil", "yield"]
    assert data.outcome_names == ("Yld_Mass_Dry_lb_ac",)
    assert model.selection == "variance_reduction"
    assert model.selection_outcome == "Yld_Mass_Dry_lb_ac"
    assert not any(
        spec.role is VariableRole.IDENTIFIER
        for source in data.sources
        for spec in source.variables
    )
    assert configured.metadata["legacy_adapter"] is True
    assert "Not applied" in configured.metadata["legacy_identifier_synthesis"]
    assert configured.metadata["legacy_configured_random_states"] == (42, 1337)
    assert model.random_state == 42


def test_legacy_loader_supports_existing_same_source_generic_outcome(tmp_path):
    configured = load_legacy_config(
        _write_project(tmp_path, _legacy_document("nitrate"))
    )
    data, model = configured.build()

    assert [source.source_id for source in data.sources] == ["soil"]
    assert data.predictor_names == ("EC",)
    assert data.outcome_names == ("nitrate",)
    assert model.selection == "variance_reduction"
    assert model.selection_outcome == "nitrate"


def test_legacy_loader_supports_existing_external_generic_outcome(tmp_path):
    declaration = {
        "id": "nutrient_lab",
        "name": "soil_nitrate",
        "column": "nitrate",
        "path": "../data/field.csv",
        "x": "x",
        "y": "y",
    }
    configured = load_legacy_config(
        _write_project(tmp_path, _legacy_document(declaration))
    )
    data, model = configured.build()

    assert [source.source_id for source in data.sources] == ["soil", "nutrient_lab"]
    assert data.outcome_names == ("soil_nitrate",)
    assert data.get_source("nutrient_lab").metadata[
        "legacy_outcome_source_column"
    ] == "nitrate"
    assert model.selection_outcome == "soil_nitrate"


def test_explicit_null_legacy_outcome_overrides_yield_section(tmp_path):
    document = _legacy_document(None)
    document["project"]["yield"] = {
        "path": "../data/field.csv",
        "x": "x",
        "y": "y",
        "column": "yield",
    }
    configured = load_legacy_config(_write_project(tmp_path, document))

    assert configured.build_dataset().outcome_names == ()
    assert configured.model_options["selection"] == "silhouette"


def test_new_loader_does_not_accept_legacy_schema(repository_root):
    with pytest.raises(ConfigurationError, match="Unsupported keys"):
        load_config(repository_root / "configs" / "project.yaml")
