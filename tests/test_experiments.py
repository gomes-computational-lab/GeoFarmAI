from pathlib import Path

from geofarmai.experiments import run_experiments


def test_configured_experiment_writes_summaries(synthetic_config: Path):
    results = run_experiments(synthetic_config)
    assert len(results) == 1
    output = results[0].output_directory
    assert (output / "synthetic_field_experiments_summary.csv").is_file()
    assert (output / "synthetic_field_experiments_gridsearch.csv").is_file()
