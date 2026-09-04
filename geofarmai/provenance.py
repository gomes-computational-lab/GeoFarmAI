"""Small helpers for recording decomposition method provenance."""

from __future__ import annotations

from typing import Any


def decomposition_provenance(
    requested_method: str,
    actual_method: str,
    *,
    used_r: bool,
) -> dict[str, Any]:
    """Build consistent decomposition provenance for results and manifests."""

    return {
        "requested_method": requested_method,
        "actual_method": actual_method,
        "used_r": bool(used_r),
        "fallback_occurred": requested_method != actual_method,
    }


def vector_decomposition_provenance(cfg: dict, used_r: bool) -> dict[str, Any]:
    requested = "multispati-r" if cfg["spatial_pca"].get("use_r_multispati", False) else "pca"
    actual = "multispati-r" if used_r else "pca"
    return decomposition_provenance(requested, actual, used_r=used_r)


def raster_decomposition_provenance(cfg: dict, used_multispaeti: bool) -> dict[str, Any]:
    if not cfg.get("raster", {}).get("use_pca", True):
        return decomposition_provenance("raw", "raw", used_r=False)
    requested = cfg.get("spatial_pca", {}).get("engine", "multispaeti")
    actual = "multispaeti" if used_multispaeti else "pca"
    return decomposition_provenance(requested, actual, used_r=False)


def decomposition_metric_fields(provenance: dict[str, Any]) -> dict[str, Any]:
    """Flatten decomposition provenance for metrics CSV records."""

    return {
        "requested_decomposition_method": provenance["requested_method"],
        "actual_decomposition_method": provenance["actual_method"],
        "used_r": provenance["used_r"],
        "decomposition_fallback_occurred": provenance["fallback_occurred"],
    }
