"""Batch runner for exploring clustering permutations.

Usage:
    python -m jobs.run_experiments --cfg configs/project.yaml

The script reads the base project configuration and executes the pipeline for
each experiment defined under the `experiments` key. Each experiment may map to
a single override or to a grid of parameter combinations. Artifacts and
metrics are written to the standard export directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from core.logging_utils import run_log_context
from jobs.flow_mzd import (
    load_cfg,
    ingest_two,
    reproject_to_meters,
    make_density_grid,
    reconcile_to_grid,
    components_from_grid,
    gridsearch,
    postprocess_and_export,
)


def _deep_merge(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _set_by_path(target: Dict[str, Any], path: str, value: Any) -> None:
    cursor = target
    keys = path.split('.')
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return '-'.join(str(v) for v in value)
    return str(value)


def _expand_experiment(defn: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    name = defn.get('name', 'experiment')
    overrides = defn.get('overrides', {})
    grid = defn.get('parameters', {})

    if not grid:
        yield name, deepcopy(overrides), {}
        return

    grid_keys = list(grid.keys())
    grid_values = []
    for key in grid_keys:
        values = grid[key]
        if not isinstance(values, list):
            values = [values]
        grid_values.append(values)

    for idx, combo in enumerate(product(*grid_values), start=1):
        combo_overrides = deepcopy(overrides)
        metadata: Dict[str, Any] = {}
        for key, value in zip(grid_keys, combo):
            _set_by_path(combo_overrides, key, value)
            metadata[key] = value
        yield f"{name}_run{idx}", combo_overrides, metadata


def _augment_leaderboard(
    leaderboard: List[Dict[str, Any]],
    experiment: str,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    augmented: List[Dict[str, Any]] = []
    flat_meta = {f"param__{k}": _stringify(v) for k, v in metadata.items()}
    for entry in leaderboard:
        row = entry.copy()
        row['experiment'] = experiment
        row.update(flat_meta)
        augmented.append(row)
    return augmented


def run_pipeline(cfg: Dict[str, Any], experiment: str | None, metadata: Dict[str, Any]):
    label = experiment or 'baseline'
    if cfg.get("raster", {}).get("enabled", False):
        from core.raster_pipeline import run_raster_mzd_flow

        print(f"[experiments] {label}: running raster kriging, PCA, clustering, and export")
        result = run_raster_mzd_flow(
            cfg,
            experiment=label,
            metadata={key: _stringify(value) for key, value in metadata.items()},
        )
        leaderboard = []
        for row in result.get("leaderboard", []):
            augmented = row.copy()
            augmented["experiment"] = label
            for key, value in metadata.items():
                augmented[f"param__{key}"] = _stringify(value)
            leaderboard.append(augmented)

        metrics = {
            "experiment": label,
            "pipeline": "raster",
            "artifact": result.get("artifact"),
            "cell_m": result.get("cell_m"),
        }
        if leaderboard:
            best = max(leaderboard, key=lambda row: (row.get("vr", 0.0), row.get("asc", 0.0)))
            metrics.update(best)
        for key, value in metadata.items():
            metrics[f"param__{key}"] = _stringify(value)
        return metrics, leaderboard

    print(f"[experiments] {label}: loading soil and yield data")
    soil, yld = ingest_two.fn(cfg)

    print(f"[experiments] {label}: reprojecting inputs")
    soil, yld = reproject_to_meters.fn(soil, yld, cfg)

    print(f"[experiments] {label}: building adaptive grid")
    grid, cell = make_density_grid.fn(soil, yld, cfg)

    print(f"[experiments] {label}: reconciling soil/yield values to {len(grid)} grid cells")
    table = reconcile_to_grid.fn(soil, yld, grid, cfg)

    print(f"[experiments] {label}: computing spatial components")
    Z, W, used_r = components_from_grid.fn(table, cfg)

    print(f"[experiments] {label}: clustering and scoring candidate zones")
    best_payload, leaderboard = gridsearch.fn(table, Z, cfg)

    labels = best_payload['labels']
    metrics = best_payload['metrics'].copy()
    metrics['used_r_multispati'] = used_r
    metrics['experiment'] = experiment or 'baseline'
    for key, value in metadata.items():
        metrics[f"param__{key}"] = _stringify(value)

    augmented_leaderboard = _augment_leaderboard(leaderboard, metrics['experiment'], metadata)

    print(f"[experiments] {label}: exporting maps, metrics, and reports")
    artifact_bundle = postprocess_and_export.fn(
        table,
        labels,
        metrics,
        augmented_leaderboard,
        cfg,
        experiment=metrics['experiment'],
    )

    return {
        **metrics,
        'artifact': artifact_bundle['gpkg'],
        'cell_m': cell,
    }, augmented_leaderboard


def main():
    parser = argparse.ArgumentParser(description="Run permutation experiments for management zones")
    parser.add_argument("--cfg", default="configs/project.yaml", help="Path to base project YAML")
    args = parser.parse_args()

    base_cfg = load_cfg.fn(args.cfg)
    out_dir = base_cfg['export']['out_dir']
    with run_log_context(out_dir, "experiments"):
        _run_experiments(base_cfg)


def _run_experiments(base_cfg: Dict[str, Any]) -> None:
    experiments = base_cfg.get('experiments', [])
    if not experiments:
        raise SystemExit("No experiments defined in the configuration under 'experiments'.")

    out_dir = base_cfg['export']['out_dir']
    summary_rows: List[Dict[str, Any]] = []
    all_leaderboard_rows: List[Dict[str, Any]] = []

    planned = [
        (name, overrides, metadata)
        for exp in experiments
        for name, overrides, metadata in _expand_experiment(exp)
    ]
    total = len(planned)
    print(f"[experiments] Starting {total} configured experiment run(s).")

    for idx, (name, overrides, metadata) in enumerate(planned, start=1):
        print(f"[experiments] Run {idx}/{total}: {name}")
        if metadata:
            metadata_text = ", ".join(f"{key}={_stringify(value)}" for key, value in metadata.items())
            print(f"[experiments] Run {idx}/{total}: parameters: {metadata_text}")
        cfg_variant = deepcopy(base_cfg)
        cfg_variant.pop('experiments', None)
        _deep_merge(cfg_variant, overrides)
        metrics_row, leaderboard_rows = run_pipeline(cfg_variant, name, metadata)
        summary_rows.append(metrics_row)
        all_leaderboard_rows.extend(leaderboard_rows)
        vr = metrics_row.get('vr')
        vr_text = f"{vr:.4f}" if isinstance(vr, (int, float)) else "n/a"
        print(
            f"[experiments] Completed {idx}/{total}: {name} "
            f"(best k={metrics_row.get('k')}, algo={metrics_row.get('algo')}, "
            f"seed={metrics_row.get('seed')}, vr={vr_text})"
        )

    project_name = base_cfg['project']['name']

    summary_path = f"{out_dir}/{project_name}_experiments_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[experiments] Wrote summary: {summary_path}")

    if all_leaderboard_rows:
        leaderboard_path = f"{out_dir}/{project_name}_experiments_gridsearch.csv"
        pd.DataFrame(all_leaderboard_rows).to_csv(leaderboard_path, index=False)
        print(f"[experiments] Wrote leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
