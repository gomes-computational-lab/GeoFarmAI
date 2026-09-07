from __future__ import annotations

import json
import subprocess
import sys


def _run_cli(repository_root, *arguments):
    return subprocess.run(
        [sys.executable, "-m", "geofarmai.cli", *arguments],
        cwd=repository_root,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_scientific_cli_help_requires_no_application_service(repository_root):
    completed = _run_cli(repository_root, "--help")

    assert completed.returncode == 0
    assert "Validate and run GeoFarmAI scientific YAML analyses" in completed.stdout
    assert "validate" in completed.stdout
    assert "run" in completed.stdout


def test_scientific_cli_validates_source_based_example(repository_root):
    completed = _run_cli(
        repository_root,
        "validate",
        "--config",
        "configs/example.yaml",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["sources"] == ["field"]
    assert payload["predictors"] == ["EC", "moisture", "elevation"]
    assert payload["outcomes"] == ["yield"]


def test_scientific_cli_runs_through_canonical_model(repository_root):
    completed = _run_cli(
        repository_root,
        "run",
        "--config",
        "configs/example_single_csv.yaml",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["candidate_count"] == 1
    assert payload["selection_metric"] == "variance_reduction"
    assert payload["selection_outcome"] == "yield"
