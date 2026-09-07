"""Scientific command-line interface for GeoFarmAI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geofarmai",
        description="Validate and run GeoFarmAI scientific YAML analyses.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate YAML and construct its canonical data/model objects.",
    )
    _configuration_arguments(validate)

    run = subparsers.add_parser(
        "run",
        help="Run YAML through the canonical GeoFarmModel scientific API.",
    )
    _configuration_arguments(run)
    return parser


def _configuration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a source-based GeoFarmAI YAML file.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the isolated historical project-configuration adapter.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scientific CLI without importing application integrations."""

    parser = _parser()
    args = parser.parse_args(argv)

    from geofarmai.config import load_config, load_legacy_config
    from geofarmai.exceptions import GeoFarmAIError

    try:
        configuration = (
            load_legacy_config(args.config) if args.legacy else load_config(args.config)
        )
        if args.command == "validate":
            data, model = configuration.build()
            payload = {
                "configuration": str(configuration.configuration_path),
                "legacy": configuration.legacy,
                "sources": [source.source_id for source in data.sources],
                "predictors": list(data.predictor_names),
                "outcomes": list(data.outcome_names),
                "model": type(model).__name__,
                "status": "valid",
            }
        else:
            payload = configuration.run().summary()
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    except GeoFarmAIError as exc:
        print(f"geofarmai: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - console-script path is tested
    raise SystemExit(main())
