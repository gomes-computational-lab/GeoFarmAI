"""Terminal-only runner for the Ag-GPT GIS management-zone pipeline.

This script is intended for GIS users who want to run the Python analysis
pipeline directly from two CSV files without using the Streamlit interface.

Example:
    python scripts/run_gis_pipeline.py \
        --soil-csv data/soil.csv \
        --yield-csv data/yield.csv \
        --project-name sacaton_field \
        --algorithms kmeans agglomerative gmm fcm \
        --k-values 2 3 4 5 6 \
        --seeds 42 1337 \
        --use-pca \
        --parallel-gridsearch \
        --n-jobs 4 \
        --force
"""

from __future__ import annotations

import argparse
import copy
import tempfile
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CFG = Path("configs/project.yaml")
MAX_WORKERS = 4


def _csv_columns(path: Path) -> list[str]:
    import pandas as pd

    return list(pd.read_csv(path, nrows=0).columns)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Ag-GPT GIS pipeline from soil and yield CSV files."
    )
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Base YAML config to modify for this run.")
    parser.add_argument("--project-name", default=None, help="Project name used for output folders.")
    parser.add_argument("--soil-csv", required=True, help="Path to the soil/terrain/spectral CSV.")
    parser.add_argument("--yield-csv", required=True, help="Path to the yield CSV.")
    parser.add_argument("--soil-x", default="Longitude", help="Soil CSV longitude/easting column.")
    parser.add_argument("--soil-y", default="Latitude", help="Soil CSV latitude/northing column.")
    parser.add_argument("--yield-x", default="Longitude", help="Yield CSV longitude/easting column.")
    parser.add_argument("--yield-y", default="Latitude", help="Yield CSV latitude/northing column.")
    parser.add_argument("--yield-column", default="Yld_Mass_Dry_lb_ac", help="Yield value column.")
    parser.add_argument(
        "--soil-variables",
        nargs="+",
        default=None,
        help="Predictor columns to use. Defaults to variables already listed in the config.",
    )
    parser.add_argument(
        "--required-soil-variables",
        nargs="+",
        default=None,
        help="Required predictor columns. Defaults to required variables listed in the config.",
    )
    parser.add_argument(
        "--pca-variables",
        nargs="+",
        default=None,
        help="Columns used for PCA/raw clustering. Defaults to --soil-variables or config raster.pca_variables.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=["kmeans", "agglomerative", "gmm", "fcm"],
        help="Clustering algorithms to compare.",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=None, help="Cluster counts to test.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds for seeded algorithms.")
    parser.add_argument("--pca-components", type=int, default=None, help="Number of PCA/MULTISPATI components.")
    parser.add_argument("--weights-k", type=int, default=None, help="K-nearest neighbors for spatial weights.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to config export.out_dir.")
    parser.add_argument("--target-crs", default=None, help="Target CRS, for example auto_utm or EPSG:32612.")
    parser.add_argument("--cell-size", type=float, default=None, help="Raster cell size. Defaults to auto.")
    parser.add_argument("--silhouette-sample-size", type=int, default=None, help="Sample size for silhouette scoring.")
    parser.add_argument("--use-pca", dest="use_pca", action="store_true", help="Use MULTISPATI-PCA/PCA features.")
    parser.add_argument("--raw-features", dest="use_pca", action="store_false", help="Cluster raw standardized rasters.")
    parser.set_defaults(use_pca=None)
    parser.add_argument("--force", action="store_true", help="Ignore cached outputs and recompute.")
    parser.add_argument("--parallel-gridsearch", action="store_true", help="Run grid search candidates in parallel.")
    parser.add_argument("--serial-gridsearch", action="store_true", help="Run grid search candidates serially.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=f"Number of multiprocessing workers. Values above {MAX_WORKERS} are capped at {MAX_WORKERS}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit.")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_columns(path: Path, required: list[str], label: str) -> None:
    columns = _csv_columns(path)
    missing = [name for name in required if name not in columns]
    if missing:
        raise SystemExit(f"{label} CSV is missing required column(s): {missing}")


def _apply_args_to_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    project = resolved.setdefault("project", {})
    soil = project.setdefault("soil", {})
    yld = project.setdefault("yield", {})
    raster = resolved.setdefault("raster", {})
    clustering = resolved.setdefault("clustering", {})
    spatial_pca = resolved.setdefault("spatial_pca", {})
    weights = resolved.setdefault("weights", {})
    export = resolved.setdefault("export", {})

    soil_csv = Path(args.soil_csv).expanduser().resolve()
    yield_csv = Path(args.yield_csv).expanduser().resolve()

    if args.project_name:
        project["name"] = args.project_name
    soil.update({"path": str(soil_csv), "x": args.soil_x, "y": args.soil_y})
    yld.update({"path": str(yield_csv), "x": args.yield_x, "y": args.yield_y, "column": args.yield_column})
    project["yield_column"] = "yield"

    if args.soil_variables:
        soil["variables"] = args.soil_variables
    if args.required_soil_variables:
        soil["required_variables"] = args.required_soil_variables
    if args.pca_variables:
        raster["pca_variables"] = args.pca_variables
    elif args.soil_variables:
        raster["pca_variables"] = args.soil_variables

    if args.algorithms:
        clustering["algorithms"] = args.algorithms
    if args.k_values:
        clustering["k_values"] = args.k_values
    if args.seeds:
        clustering["seeds"] = args.seeds
    if args.pca_components is not None:
        spatial_pca["n_components"] = args.pca_components
    if args.weights_k is not None:
        weights["k"] = args.weights_k
    if args.out_dir is not None:
        export["out_dir"] = args.out_dir
    if args.target_crs is not None:
        raster["target_crs"] = args.target_crs
    if args.cell_size is not None:
        raster["cell_size"] = args.cell_size
    if args.silhouette_sample_size is not None:
        raster["silhouette_sample_size"] = args.silhouette_sample_size
    if args.use_pca is not None:
        raster["use_pca"] = args.use_pca

    raster["enabled"] = True
    if args.parallel_gridsearch and args.serial_gridsearch:
        raise SystemExit("Use only one of --parallel-gridsearch or --serial-gridsearch.")
    if args.parallel_gridsearch:
        raster["parallel_gridsearch"] = True
    if args.serial_gridsearch:
        raster["parallel_gridsearch"] = False
    if args.n_jobs is not None:
        raster["n_jobs"] = max(1, min(args.n_jobs, MAX_WORKERS))

    _validate_columns(soil_csv, [soil["x"], soil["y"], *soil.get("variables", [])], "Soil")
    _validate_columns(yield_csv, [yld["x"], yld["y"], yld["column"]], "Yield")
    return resolved


def main() -> None:
    args = _parse_args()
    cfg = _apply_args_to_cfg(_load_cfg(Path(args.cfg)), args)

    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        temp_cfg_path = handle.name

    print(f"[gis-runner] Running project: {cfg['project']['name']}")
    print(f"[gis-runner] Soil CSV: {cfg['project']['soil']['path']}")
    print(f"[gis-runner] Yield CSV: {cfg['project']['yield']['path']}")
    print(f"[gis-runner] Algorithms: {cfg['clustering']['algorithms']}")
    print(f"[gis-runner] k values: {cfg['clustering']['k_values']}")
    print(f"[gis-runner] Outputs: {cfg.get('export', {}).get('out_dir', 'outputs')}")

    from jobs.flow_mzd import mzd_flow_two_csvs

    result = mzd_flow_two_csvs(temp_cfg_path, force=args.force)
    print("[gis-runner] Finished.")
    if isinstance(result, dict):
        for key in ["gridsearch_csv", "metrics_csv", "artifact", "pdf", "visual_manifest"]:
            value = result.get(key)
            if value:
                print(f"[gis-runner] {key}: {value}")


if __name__ == "__main__":
    main()
