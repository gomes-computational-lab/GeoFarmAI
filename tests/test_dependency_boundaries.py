from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import zipfile

import pytest


APPLICATION_PACKAGES = {
    "ollama",
    "langchain",
    "langchain-community",
    "langchain_community",
    "openai",
    "streamlit",
    "fastapi",
    "uvicorn",
}


def test_base_dependency_declarations_contain_only_canonical_science(repository_root):
    text = (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    base = text.split("[project.optional-dependencies]", 1)[0].lower()
    required = {
        "numpy",
        "pandas",
        "scipy",
        "geopandas",
        "pyproj",
        "shapely",
        "scikit-learn",
        "scikit-fuzzy",
        "statsmodels",
        "pyyaml",
    }
    non_base = APPLICATION_PACKAGES | {
        "rpy2",
        "multispaeti",
        "libpysal",
        "esda",
        "rasterio",
        "pykrige",
        "contextily",
        "matplotlib",
        "pydantic",
        "prefect",
        "scikit-gstat",
        "mapclassify",
        "tifffile",
    }

    assert all(name in base for name in required)
    assert all(name not in base for name in non_base)
    assert 'geofarmai = "geofarmai.cli:main"' in text
    assert '"jobs*"' not in text
    assert '"services*"' not in text
    assert "legacy-application" not in text


def test_repository_excludes_application_service_and_job_sources(repository_root):
    assert not (repository_root / "services").exists()
    assert not (repository_root / "jobs").exists()


def test_representative_canonical_workflow_rejects_optional_imports(repository_root):
    blocked = sorted(
        APPLICATION_PACKAGES
        | {
            "rpy2",
            "multispaeti",
            "libpysal",
            "esda",
            "rasterio",
            "pykrige",
            "contextily",
            "matplotlib",
            "pydantic",
            "prefect",
        }
    )
    script = f"""
import builtins
import json
import tempfile
from pathlib import Path

import geopandas
import numpy as np
import pandas as pd

blocked = {blocked!r}
original_import = builtins.__import__

def base_only_import(name, *args, **kwargs):
    root = name.split('.')[0]
    if root in blocked:
        raise ModuleNotFoundError(f'optional dependency {{root}} is unavailable')
    return original_import(name, *args, **kwargs)

builtins.__import__ = base_only_import

import geofarmai
import geofarmai.config
import geofarmai.data.harmonization
import geofarmai.model
import geofarmai.result
from geofarmai import FieldDataset, GeoFarmModel

frame = pd.DataFrame({{
    'x': np.arange(8, dtype=float),
    'y': np.tile([0.0, 1.0], 4),
    'EC': [0.1, 0.2, 0.15, 0.25, 4.8, 5.0, 5.1, 4.9],
    'yield': [10.0, 11.0, 9.0, 10.5, 50.0, 52.0, 49.0, 51.0],
}})
data = FieldDataset.from_dataframe(
    frame,
    source_id='clean_base',
    predictors=['EC'],
    outcome='yield',
    coordinates=('x', 'y'),
    crs='EPSG:32615',
)
result = GeoFarmModel(
    decomposition='pca',
    clustering='kmeans',
    k=2,
    selection='silhouette',
    random_state=42,
).fit(data)
with tempfile.TemporaryDirectory() as directory:
    artifacts = result.export(Path(directory) / 'zones.gpkg')
    exported = all(path.is_file() for path in artifacts.values())

loaded = sorted(
    name for name in __import__('sys').modules
    if any(name == item or name.startswith(item + '.') for item in blocked)
)
print(json.dumps({{
    'candidate_count': len(result.candidate_solutions),
    'best_k': result.best_k,
    'exported': exported,
    'loaded_optional': loaded,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload == {
        "candidate_count": 1,
        "best_k": 2,
        "exported": True,
        "loaded_optional": [],
    }


def test_legacy_mapping_pipeline_is_rejected_without_importing_jobs(repository_root):
    script = """
import sys
from geofarmai import ModelConfigurationError, run_pipeline

try:
    run_pipeline({})
except ModelConfigurationError as exc:
    assert 'run_from_config' in str(exc)
else:
    raise AssertionError('legacy mapping orchestration was not rejected')
assert 'jobs' not in sys.modules
assert 'services' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )


def test_built_wheel_excludes_application_services_and_jobs(repository_root, tmp_path):
    try:
        build_main = importlib.util.find_spec("build.__main__")
    except ModuleNotFoundError:
        build_main = None
    if build_main is None:
        pytest.skip("The wheel-boundary test requires the development 'build' extra.")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    wheel = next(tmp_path.glob("geofarmai-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()

    assert any(name.startswith("geofarmai/") for name in members)
    assert any(name.startswith("core/") for name in members)
    assert not any(name.startswith("services/") for name in members)
    assert not any(name.startswith("jobs/") for name in members)
