"""Command-line interface for GeoFarmAI."""

from __future__ import annotations

import argparse
import os
import sys

from .config import GeoFarmConfig
from .exceptions import GeoFarmAIError
from .experiments import run_experiments
from .pipeline import GeoFarmPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geofarmai", description="Geospatial management-zone analysis framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a YAML configuration and its input schema")
    validate.add_argument("--config", required=True)
    run = subparsers.add_parser("run", help="Run a configured management-zone pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--force", action="store_true", help="Ignore reusable raster cache entries")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--serial-gridsearch", action="store_true")
    mode.add_argument("--parallel-gridsearch", action="store_true")
    run.add_argument("--n-jobs", type=int, default=None)
    experiment = subparsers.add_parser("experiment", help="Run configured experiment permutations")
    experiment.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = GeoFarmConfig.from_yaml(args.config)
            print(f"Configuration is valid: {config.source}")
            return 0
        if args.command == "experiment":
            results = run_experiments(args.config)
            print(f"Completed {len(results)} experiment run(s).")
            return 0
        parallel = True if args.parallel_gridsearch else False if args.serial_gridsearch else None
        if args.n_jobs is not None and args.n_jobs < 1:
            raise ValueError("--n-jobs must be at least 1.")
        result = GeoFarmPipeline.from_yaml(args.config).run(
            force=args.force, parallel_gridsearch=parallel, n_jobs=args.n_jobs,
        )
        print(f"Completed. Outputs: {result.output_directory}")
        if result.best_model is not None:
            print(f"Selected model: {result.best_model}, k={result.best_k}")
        return 0
    except (GeoFarmAIError, ValueError) as exc:
        print(f"GeoFarmAI error: {exc}", file=sys.stderr)
        if os.environ.get("GEOFARMAI_DEBUG"):
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
