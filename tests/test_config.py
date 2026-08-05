from pathlib import Path

import pytest
import yaml

from geofarmai.config import GeoFarmConfig
from geofarmai.exceptions import ConfigurationError, InputDataError


def test_example_configuration_resolves_relative_paths(project_root: Path):
    config = GeoFarmConfig.from_yaml(project_root / "configs" / "example.yaml")
    assert Path(config.data["project"]["soil"]["path"]) == (
        project_root / "examples" / "data" / "soil_sample.csv"
    ).resolve()
    assert Path(config.data["export"]["out_dir"]) == (project_root / "outputs").resolve()


def test_missing_feature_column_has_helpful_error(tmp_path: Path, synthetic_config: Path):
    data = yaml.safe_load(synthetic_config.read_text(encoding="utf-8"))
    data["project"]["soil"]["variables"].append("NotAColumn")
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(InputDataError, match="NotAColumn"):
        GeoFarmConfig.from_yaml(broken)


def test_invalid_algorithm_has_helpful_error(tmp_path: Path, synthetic_config: Path):
    data = yaml.safe_load(synthetic_config.read_text(encoding="utf-8"))
    data["clustering"]["algorithms"] = ["not-a-method"]
    broken = tmp_path / "algorithm.yaml"
    broken.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsupported clustering"):
        GeoFarmConfig.from_yaml(broken)
