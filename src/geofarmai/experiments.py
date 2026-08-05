"""Configured experiment expansion and execution."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import GeoFarmConfig
from .exceptions import ConfigurationError
from .pipeline import GeoFarmPipeline
from .result import GeoFarmResult


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _set_by_path(target: dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    keys = path.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    return str(value)


def _expand_experiment(definition: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], dict[str, Any]]]:
    name = definition.get("name", "experiment")
    overrides = definition.get("overrides", {})
    parameter_grid = definition.get("parameters", {})
    if not parameter_grid:
        yield name, deepcopy(overrides), {}
        return
    keys = list(parameter_grid)
    values = [items if isinstance(items, list) else [items] for items in parameter_grid.values()]
    for index, combination in enumerate(product(*values), start=1):
        combination_overrides = deepcopy(overrides)
        metadata = {}
        for key, value in zip(keys, combination):
            _set_by_path(combination_overrides, key, value)
            metadata[key] = value
        yield f"{name}_run{index}", combination_overrides, metadata


def run_experiments(config: str | Path) -> list[GeoFarmResult]:
    """Run the configured outer experiment grid and write compatibility summary tables."""
    base = GeoFarmConfig.from_yaml(config)
    definitions = base.data.get("experiments", [])
    if not definitions:
        raise ConfigurationError("No experiments are defined under the 'experiments' key.")
    planned = [item for definition in definitions for item in _expand_experiment(definition)]
    results = []
    summary_rows = []
    leaderboard_rows = []
    for index, (name, overrides, metadata) in enumerate(planned, start=1):
        print(f"[experiments] Run {index}/{len(planned)}: {name}")
        variant = base.copy()
        variant.pop("experiments", None)
        _deep_merge(variant, overrides)
        variant_config = GeoFarmConfig.from_mapping(variant, base.source)
        result = GeoFarmPipeline(variant_config).run(experiment=name, metadata=metadata)
        results.append(result)
        summary = dict(result.metrics)
        summary["artifact"] = str(result.artifacts.get("artifact", result.artifacts.get("gpkg", "")))
        for key, value in metadata.items():
            summary[f"param__{key}"] = _stringify(value)
        summary_rows.append(summary)
        frame = result.leaderboard.copy()
        if not frame.empty:
            frame["experiment"] = name
            for key, value in metadata.items():
                frame[f"param__{key}"] = _stringify(value)
            leaderboard_rows.extend(frame.to_dict("records"))
    output = Path(base.data["export"]["out_dir"])
    output.mkdir(parents=True, exist_ok=True)
    project = base.data["project"]["name"]
    pd.DataFrame(summary_rows).to_csv(output / f"{project}_experiments_summary.csv", index=False)
    if leaderboard_rows:
        pd.DataFrame(leaderboard_rows).to_csv(output / f"{project}_experiments_gridsearch.csv", index=False)
    return results
