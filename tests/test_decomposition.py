from __future__ import annotations

import builtins
import json
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from core.multispati import multispati_components
import core.multispati as multispati_module
from geofarmai.exceptions import RMultispatiUnavailableError
from geofarmai.provenance import (
    decomposition_metric_fields,
    raster_decomposition_provenance,
    vector_decomposition_provenance,
)


class _UnusedWeights:
    def full(self):
        raise AssertionError("PCA fallback must not inspect spatial weights")


def test_pca_fallback_matches_current_standardized_pca():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 4.0, 8.0, 16.0],
            "b": [2.0, 1.0, 3.0, 7.0, 11.0],
            "c": [5.0, 4.0, 3.0, 2.0, 0.0],
        }
    )

    actual, used_r = multispati_components(frame, _UnusedWeights(), n_components=2, use_r=False)
    expected = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(frame))

    assert used_r is False
    assert actual.columns.tolist() == ["PC1", "PC2"]
    np.testing.assert_allclose(pairwise_distances(actual), pairwise_distances(expected), rtol=1e-10, atol=1e-10)


def test_explicit_r_request_raises_clear_error_when_rpy2_is_unavailable(monkeypatch):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    original_import = builtins.__import__

    def reject_rpy2(name, *args, **kwargs):
        if name == "rpy2" or name.startswith("rpy2."):
            raise ModuleNotFoundError("rpy2 intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_rpy2)
    monkeypatch.setattr(multispati_module, "_preflight_r_multispati", lambda: None)

    with pytest.raises(RMultispatiUnavailableError, match="explicitly requested"):
        multispati_components(frame, _UnusedWeights(), n_components=1, use_r=True)


def test_explicit_r_request_does_not_fall_back_when_r_package_loading_fails(monkeypatch):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    robjects = types.ModuleType("rpy2.robjects")
    robjects.pandas2ri = object()
    robjects.numpy2ri = object()
    robjects.r = lambda expression: (_ for _ in ()).throw(RuntimeError("adespatial unavailable"))
    conversion = types.ModuleType("rpy2.robjects.conversion")
    conversion.localconverter = object()
    package = types.ModuleType("rpy2")
    package.robjects = robjects
    monkeypatch.setitem(sys.modules, "rpy2", package)
    monkeypatch.setitem(sys.modules, "rpy2.robjects", robjects)
    monkeypatch.setitem(sys.modules, "rpy2.robjects.conversion", conversion)
    monkeypatch.setattr(multispati_module, "_preflight_r_multispati", lambda: None)

    with pytest.raises(RMultispatiUnavailableError, match="execution failed"):
        multispati_components(frame, _UnusedWeights(), n_components=1, use_r=True)


def test_r_preflight_converts_native_process_failure_to_geofarmai_error(monkeypatch):
    completed = types.SimpleNamespace(returncode=-11, stdout="", stderr="native crash")
    monkeypatch.setattr(multispati_module.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RMultispatiUnavailableError, match=r"preflight \(exit -11\)"):
        multispati_module._preflight_r_multispati()


def test_decomposition_provenance_records_requested_actual_r_and_fallback():
    vector = vector_decomposition_provenance(
        {"spatial_pca": {"use_r_multispati": False}},
        used_r=False,
    )
    raster = raster_decomposition_provenance(
        {"raster": {"use_pca": True}, "spatial_pca": {"engine": "multispaeti"}},
        used_multispaeti=False,
    )

    assert vector == {
        "requested_method": "pca",
        "actual_method": "pca",
        "used_r": False,
        "fallback_occurred": False,
    }
    assert raster == {
        "requested_method": "multispaeti",
        "actual_method": "pca",
        "used_r": False,
        "fallback_occurred": True,
    }
    assert decomposition_metric_fields(raster) == {
        "requested_decomposition_method": "multispaeti",
        "actual_decomposition_method": "pca",
        "used_r": False,
        "decomposition_fallback_occurred": True,
    }


@pytest.mark.r
@pytest.mark.optional_scientific
def test_r_multispati_when_r_and_spatial_packages_are_available(repository_root):
    # rpy2 can terminate the interpreter when its R runtime is misconfigured,
    # so optional R initialization must be isolated from the pytest process.
    script = """
import json
import numpy as np
import pandas as pd
import libpysal
from core.multispati import multispati_components

frame = pd.DataFrame({
    'a': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    'b': [1.0, 1.5, 3.0, 2.5, 5.0, 4.5],
    'c': [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
})
weights = libpysal.weights.KNN.from_array(
    np.column_stack([np.arange(6), np.zeros(6)]), k=2
)
weights.transform = 'r'
scores, used_r = multispati_components(frame, weights, n_components=2, use_r=True)
print(json.dumps({
    'used_r': used_r,
    'shape': list(scores.shape),
    'finite': bool(np.isfinite(scores.to_numpy()).all()),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        pytest.skip(f"R/rpy2 runtime is not usable (exit {completed.returncode})")
    try:
        result = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pytest.skip("R MULTISPATI did not produce a testable result")
    assert result["shape"] == [6, 2]
    assert result["finite"] is True
    assert result["used_r"] is True
