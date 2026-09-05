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
) -> GeoFarmResult | tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compatibility wrapper around legacy and canonical analysis entry points.

    New code should construct :class:`GeoFarmModel` directly. A legacy config
    mapping retains the existing ``jobs.run_experiments.run_pipeline`` return
    contract through a lazy adapter.
    """

    warnings.warn(
        "run_pipeline() is a compatibility API; use GeoFarmModel(...).fit(data) "
        "for new scientific workflows.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(data_or_config, Mapping):
        if model_options:
            raise ModelConfigurationError(
                "GeoFarmModel options cannot be combined with a legacy config mapping."
            )
        from jobs.run_experiments import run_pipeline as legacy_run_pipeline

        return legacy_run_pipeline(
            dict(data_or_config),
            experiment,
            dict(metadata or {}),
        )

    if experiment is not None or metadata is not None:
        raise ModelConfigurationError(
            "experiment and metadata are legacy config arguments; pass only GeoFarmModel "
            "options with a FieldDataset."
        )
    return GeoFarmModel(**model_options).fit(data_or_config)


__all__ = ["delineate_zones", "run_pipeline"]
