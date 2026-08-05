import os
from pathlib import Path

import pytest
import yaml


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/geofarmai-matplotlib-tests")


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_config(tmp_path: Path, project_root: Path) -> Path:
    source = yaml.safe_load((project_root / "configs" / "example.yaml").read_text(encoding="utf-8"))
    source["project"]["soil"]["path"] = str(project_root / "examples" / "data" / "soil_sample.csv")
    source["project"]["yield"]["path"] = str(project_root / "examples" / "data" / "yield_sample.csv")
    source["export"]["out_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return config_path
