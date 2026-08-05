"""YAML loading, path resolution, and validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml
from pyproj import CRS

from .clustering import SUPPORTED_ALGORITHMS
from .exceptions import ConfigurationError, InputDataError


REQUIRED_SECTIONS = ("project", "grid", "weights", "spatial_pca", "clustering", "postprocess", "export")


@dataclass(frozen=True)
class GeoFarmConfig:
    """Validated pipeline configuration and the YAML file it came from."""

    data: dict[str, Any]
    source: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeoFarmConfig":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise InputDataError(f"Configuration file does not exist: {source}")
        try:
            raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("The configuration root must be a YAML mapping.")
        data = _resolve_paths(deepcopy(raw), source.parent)
        validate_config(data)
        return cls(data=data, source=source)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: str | Path) -> "GeoFarmConfig":
        source_path = Path(source).expanduser().resolve()
        resolved = _resolve_paths(deepcopy(dict(data)), source_path.parent)
        validate_config(resolved)
        return cls(data=resolved, source=source_path)

    def copy(self) -> dict[str, Any]:
        return deepcopy(self.data)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate YAML, returning the behavior-compatible configuration mapping."""
    return GeoFarmConfig.from_yaml(path).copy()


def _resolve_paths(cfg: dict[str, Any], base: Path) -> dict[str, Any]:
    project = cfg.get("project", {})
    for input_name in ("soil", "yield"):
        section = project.get(input_name, {})
        value = section.get("path")
        if value is not None:
            path = Path(value).expanduser()
            section["path"] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    export = cfg.get("export", {})
    value = export.get("out_dir", "outputs")
    path = Path(value).expanduser()
    export["out_dir"] = str(path.resolve() if path.is_absolute() else (base / path).resolve())
    return cfg


def _required(mapping: Mapping[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigurationError(f"Missing required {context} field(s): {', '.join(missing)}")


def _columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        raise InputDataError(f"Could not read CSV header from {path}: {exc}") from exc


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate fields needed by both raster-first and vector/grid-cell workflows."""
    missing_sections = [name for name in REQUIRED_SECTIONS if not isinstance(cfg.get(name), Mapping)]
    if missing_sections:
        raise ConfigurationError(f"Missing required configuration section(s): {', '.join(missing_sections)}")

    project = cfg["project"]
    _required(project, ("name", "soil", "yield"), "project")
    soil = project["soil"]
    yld = project["yield"]
    if not isinstance(soil, Mapping) or not isinstance(yld, Mapping):
        raise ConfigurationError("project.soil and project.yield must be mappings.")
    _required(soil, ("path", "x", "y", "variables"), "project.soil")
    _required(yld, ("path", "x", "y", "column"), "project.yield")
    if not isinstance(soil["variables"], list) or not soil["variables"]:
        raise ConfigurationError("project.soil.variables must be a non-empty list.")

    for label, section, features in (
        ("Soil", soil, list(soil["variables"])),
        ("Yield", yld, [yld["column"]]),
    ):
        path = Path(section["path"])
        if not path.is_file():
            raise InputDataError(f"{label} input file does not exist: {path}")
        columns = _columns(path)
        missing_coordinates = [name for name in (section["x"], section["y"]) if name not in columns]
        if missing_coordinates:
            raise InputDataError(
                f"{label} input is missing coordinate column(s): {', '.join(missing_coordinates)}"
            )
        missing_features = [name for name in features if name not in columns]
        if missing_features:
            raise InputDataError(f"{label} input is missing feature column(s): {', '.join(missing_features)}")

    algorithms = cfg["clustering"].get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ConfigurationError("clustering.algorithms must be a non-empty list.")
    unsupported = [name for name in algorithms if name not in SUPPORTED_ALGORITHMS]
    if unsupported:
        raise ConfigurationError(
            f"Unsupported clustering method(s): {', '.join(unsupported)}. "
            f"Supported methods: {', '.join(SUPPORTED_ALGORITHMS)}."
        )
    k_values = cfg["clustering"].get("k_values")
    if not isinstance(k_values, list) or not k_values or any(type(k) is not int or k < 2 for k in k_values):
        raise ConfigurationError("clustering.k_values must be a non-empty list of integers >= 2.")
    seeds = cfg["clustering"].get("seeds", [42])
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ConfigurationError("clustering.seeds must be a non-empty list of integers.")

    n_components = cfg["spatial_pca"].get("n_components")
    if type(n_components) is not int or n_components < 1:
        raise ConfigurationError("spatial_pca.n_components must be a positive integer.")
    if n_components > len(soil["variables"]):
        raise ConfigurationError("spatial_pca.n_components cannot exceed the number of soil variables.")
    weights_k = cfg["weights"].get("k")
    if type(weights_k) is not int or weights_k < 1:
        raise ConfigurationError("weights.k must be a positive integer.")
    min_area = cfg["postprocess"].get("min_area_m2")
    if not isinstance(min_area, (int, float)) or min_area < 0:
        raise ConfigurationError("postprocess.min_area_m2 must be zero or greater.")

    crs_value = project.get("crs_in", "EPSG:4326")
    try:
        CRS.from_user_input(crs_value)
    except Exception as exc:
        raise ConfigurationError(f"Invalid project.crs_in value '{crs_value}'.") from exc
    target = cfg.get("raster", {}).get("target_crs", "auto_utm")
    if str(target).lower() not in {"auto", "auto_utm", "utm"}:
        try:
            CRS.from_user_input(target)
        except Exception as exc:
            raise ConfigurationError(f"Invalid raster.target_crs value '{target}'.") from exc


def dump_config(cfg: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(cfg), sort_keys=False)
