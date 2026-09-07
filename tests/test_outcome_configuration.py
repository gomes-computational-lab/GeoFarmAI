from __future__ import annotations

import pytest

from geofarmai import OutcomeConfigurationError
from geofarmai.outcome import resolve_pipeline_outcome


def test_generic_outcome_can_use_the_predictor_source() -> None:
    resolved = resolve_pipeline_outcome({"outcome": "protein"})

    assert resolved is not None
    assert resolved.name == "protein"
    assert resolved.source_column == "protein"
    assert resolved.uses_predictor_source is True
    assert resolved.legacy_yield_adapter is False


def test_generic_outcome_can_use_a_separate_source() -> None:
    declaration = {
        "name": "nitrate",
        "column": "n_mg",
        "path": "nitrate.csv",
        "x": "x",
        "y": "y",
    }

    resolved = resolve_pipeline_outcome({"outcome": declaration})

    assert resolved is not None
    assert resolved.name == "nitrate"
    assert resolved.source_column == "n_mg"
    assert resolved.source_config == declaration
    assert resolved.uses_predictor_source is False


def test_explicit_null_outcome_is_unsupervised() -> None:
    assert resolve_pipeline_outcome({"outcome": None}) is None


def test_legacy_yield_configuration_remains_isolated() -> None:
    declaration = {
        "path": "harvest.csv",
        "x": "x",
        "y": "y",
        "column": "harvest_mass",
    }

    resolved = resolve_pipeline_outcome(
        {"yield": declaration, "yield_column": "yield"}
    )

    assert resolved is not None
    assert resolved.name == "yield"
    assert resolved.source_column == "harvest_mass"
    assert resolved.source_config == declaration
    assert resolved.legacy_yield_adapter is True


def test_invalid_outcome_configuration_fails_clearly() -> None:
    with pytest.raises(OutcomeConfigurationError, match="requires 'column' or 'name'"):
        resolve_pipeline_outcome(
            {"outcome": {"path": "values.csv", "x": "x", "y": "y"}}
        )
