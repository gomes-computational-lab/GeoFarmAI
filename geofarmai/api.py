"""Compatibility entry points for the evolving GeoFarmAI public API."""

from __future__ import annotations

from typing import Any, Mapping
import warnings

from geofarmai.data import FieldDataset, HarmonizedFieldDataset
from geofarmai.exceptions import ModelConfigurationError
from geofarmai.model import GeoFarmModel
from geofarmai.result import GeoFarmResult


def delineate_zones(
    data: FieldDataset | HarmonizedFieldDataset,
    **model_options: Any,
) -> GeoFarmResult:
    """Delineate zones through the canonical :class:`GeoFarmModel` API.

    ``model_options`` are forwarded unchanged to ``GeoFarmModel``. Input roles,
    coordinates, and CRS must already be explicit in a ``FieldDataset`` (or in
    an already harmonized canonical dataset); this function performs no schema
    inference and implements no scientific workflow of its own.
    """

    return GeoFarmModel(**model_options).fit(data)


def run_pipeline(
    data_or_config: FieldDataset | HarmonizedFieldDataset | Mapping[str, Any],
    experiment: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **model_options: Any,
) -> GeoFarmResult:
    """Deprecated scientific wrapper around :class:`GeoFarmModel`.

    New code should construct :class:`GeoFarmModel` directly. Historical
    mapping-based workflow orchestration has moved to the ag-gpt application;
    YAML users should call :func:`geofarmai.run_from_config` instead.
    """

    warnings.warn(
        "run_pipeline() is a compatibility API; use GeoFarmModel(...).fit(data) "
        "for new scientific workflows.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(data_or_config, Mapping):
        raise ModelConfigurationError(
            "Mapping-based run_pipeline() workflow orchestration was removed from "
            "GeoFarmAI. Use run_from_config(path, legacy=True) for a historical YAML "
            "file, or GeoFarmModel.fit(FieldDataset) for canonical analysis."
        )

    if experiment is not None or metadata is not None:
        raise ModelConfigurationError(
            "experiment and metadata are legacy config arguments; pass only GeoFarmModel "
            "options with a FieldDataset."
        )
    return GeoFarmModel(**model_options).fit(data_or_config)


__all__ = ["delineate_zones", "run_pipeline"]
