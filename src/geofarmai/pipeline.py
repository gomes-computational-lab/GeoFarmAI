"""High-level orchestration for raster-first and vector/grid-cell workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import pandas as pd

from .clustering import cluster
from .config import GeoFarmConfig
from .crs import to_utm_auto
from .decomposition import multispati_components
from .evaluation import anova_p, variance_reduction
from .export import export_pdf_report, save_package, zones_from_points
from .grid import build_field_grid, choose_cell_size
from .logging_utils import run_log_context
from .postprocessing import write_artifact_manifest
from .reconcile import populate_grid
from .result import GeoFarmResult
from .spatial import knn_weights


def _ingest_one(path, xcol, ycol, crs_in, idcol=None, keep_cols=None):
    df = pd.read_csv(path)
    missing = [column for column in (xcol, ycol) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])
    if not idcol or idcol not in df.columns:
        idcol = idcol or "row_id"
        df[idcol] = range(len(df))
    base_cols = {idcol, xcol, ycol}
    if keep_cols:
        df = df[list(base_cols.union(keep_cols))]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xcol], df[ycol]), crs=crs_in)
    gdf = gdf.drop_duplicates(subset="geometry").reset_index(drop=True)
    gdf.rename(columns={idcol: "sample_id"}, inplace=True)
    return gdf


def ingest_two(cfg):
    project = cfg["project"]
    crs_in = project.get("crs_in", "EPSG:4326")
    soil_cfg = project["soil"]
    yield_cfg = project["yield"]
    soil = _ingest_one(
        soil_cfg["path"], soil_cfg["x"], soil_cfg["y"], crs_in,
        soil_cfg.get("id_column"), soil_cfg.get("variables"),
    )
    yld = _ingest_one(
        yield_cfg["path"], yield_cfg["x"], yield_cfg["y"], crs_in,
        yield_cfg.get("id_column"), [yield_cfg["column"]],
    )
    if yield_cfg["column"] != "yield":
        yld.rename(columns={yield_cfg["column"]: "yield"}, inplace=True)
    return soil, yld


def reproject_to_meters(soil, yld, cfg):
    if cfg["project"].get("auto_reproject_to_utm", True):
        soil, epsg = to_utm_auto(soil)
        yld = yld.to_crs(epsg)
    return soil, yld


def make_density_grid(soil, yld, cfg):
    cell = cfg["grid"].get("cell_size_m")
    if not cell:
        union = pd.concat([soil[["geometry"]], yld[["geometry"]]], ignore_index=True)
        union = gpd.GeoDataFrame(union, geometry="geometry", crs=soil.crs)
        cell = choose_cell_size(union, cfg["grid"]["min_cell_size_m"], cfg["grid"]["max_cell_size_m"])
    return build_field_grid(soil, yld, cell), cell


def reconcile_to_grid(soil, yld, grid, cfg):
    return populate_grid(
        soil, yld, grid, cfg["project"]["soil"]["variables"],
        method=cfg["grid"].get("method", "idw"), buffer_m=cfg["grid"].get("buffer_m", 15),
    )


def components_from_grid(table, cfg):
    variables = [name for name in cfg["project"]["soil"]["variables"] if name in table.columns]
    n_components = cfg["spatial_pca"]["n_components"]
    if len(variables) < n_components:
        raise ValueError(f"Need at least {n_components} soil features; found {len(variables)}: {variables}")
    weights = knn_weights(table, k=cfg["weights"]["k"])
    components, used_r = multispati_components(
        table[variables], weights, n_components=n_components,
        use_r=cfg["spatial_pca"].get("use_r_multispati", False),
    )
    return components, weights, used_r


def gridsearch(gdf, components, cfg):
    leaderboard = []
    best_score = None
    best_payload = None
    X = components.values
    yield_col = cfg["project"].get("yield_column", "yield")
    for k in cfg["clustering"]["k_values"]:
        for algorithm in cfg["clustering"]["algorithms"]:
            seeds = cfg["clustering"].get("seeds", [42]) if algorithm in {"kmeans", "gmm", "fcm"} else [None]
            for seed in seeds:
                labels, quality = cluster(X, algorithm, k, random_state=seed)
                metrics = {"k": k, "algo": algorithm, "seed": seed, **quality}
                if yield_col in gdf.columns:
                    metrics["vr"] = variance_reduction(gdf[yield_col], labels)
                    metrics["anova_p"] = anova_p(gdf[yield_col], labels)
                leaderboard.append(metrics.copy())
                score = (metrics.get("vr", 0.0), metrics.get("asc", 0.0))
                if best_score is None or score > best_score:
                    best_score = score
                    best_payload = {"labels": labels, "metrics": metrics.copy()}
    return best_payload, leaderboard


def postprocess_and_export(gdf, labels, metrics, leaderboard, cfg, experiment=None):
    zones = zones_from_points(gdf, labels, min_area=cfg["postprocess"]["min_area_m2"])
    suffix = f"_{experiment}" if experiment else ""
    base = Path(cfg["export"]["out_dir"]) / f"{cfg['project']['name']}{suffix}"
    base.parent.mkdir(parents=True, exist_ok=True)
    gpkg = base.with_suffix(".gpkg")
    labelled = gdf.assign(zone=labels)
    save_package(zones, labelled, metrics, str(gpkg))
    features = list(dict.fromkeys([*cfg["project"]["soil"]["variables"], "yield"]))
    pdf = base.with_suffix(".pdf")
    images = export_pdf_report(zones, labelled, features, str(pdf), basemap=cfg["export"].get("basemap", "web"))
    leaderboard_path = Path(f"{base}_gridsearch.csv")
    pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
    artifacts = {"gpkg": gpkg, "pdf": pdf, "gridsearch_csv": leaderboard_path}
    manifest = write_artifact_manifest(base.parent, {**artifacts, "images": images})
    artifacts["artifact_manifest"] = manifest
    return artifacts


class GeoFarmPipeline:
    def __init__(self, config: GeoFarmConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeoFarmPipeline":
        return cls(GeoFarmConfig.from_yaml(path))

    def run(
        self,
        force: bool = False,
        parallel_gridsearch: Optional[bool] = None,
        n_jobs: Optional[int] = None,
        experiment: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GeoFarmResult:
        cfg = self.config.copy()
        if parallel_gridsearch is not None:
            cfg.setdefault("raster", {})["parallel_gridsearch"] = parallel_gridsearch
        if n_jobs is not None:
            cfg.setdefault("raster", {})["n_jobs"] = n_jobs
        out_dir = Path(cfg["export"]["out_dir"])
        prefix = "experiments" if experiment else "baseline"
        with run_log_context(str(out_dir), prefix):
            if cfg.get("raster", {}).get("enabled", False):
                from .raster import run_raster_mzd_flow
                raw = run_raster_mzd_flow(cfg, experiment=experiment, metadata=metadata, force=force)
                return _result_from_raster(raw, cfg)
            return _run_vector(cfg, experiment=experiment, metadata=metadata)


def _run_vector(cfg, experiment=None, metadata=None):
    soil, yld = ingest_two(cfg)
    soil, yld = reproject_to_meters(soil, yld, cfg)
    grid, cell = make_density_grid(soil, yld, cfg)
    table = reconcile_to_grid(soil, yld, grid, cfg)
    components, _, used_r = components_from_grid(table, cfg)
    best, leaderboard = gridsearch(table, components, cfg)
    metrics = {**best["metrics"], "used_r_multispati": used_r, "experiment": experiment or "baseline", "cell_m": cell}
    for key, value in (metadata or {}).items():
        metrics[f"param__{key}"] = value
    rows = [dict(row, experiment=experiment or "baseline") for row in leaderboard]
    artifacts = postprocess_and_export(table, best["labels"], metrics, rows, cfg, experiment=experiment)
    return GeoFarmResult(
        output_directory=Path(cfg["export"]["out_dir"]), metrics=metrics,
        leaderboard=pd.DataFrame(rows), best_model=metrics.get("algo"), best_k=metrics.get("k"),
        best_labels=best["labels"], artifacts=artifacts, configuration=deepcopy(cfg),
    )


def _result_from_raster(raw, cfg):
    leaderboard = pd.DataFrame(raw.get("leaderboard", []))
    metrics = {}
    metrics_path = raw.get("metrics_csv")
    if metrics_path and Path(metrics_path).exists():
        rows = pd.read_csv(metrics_path).to_dict("records")
        metrics = rows[0] if rows else {}
    elif not leaderboard.empty:
        metrics = leaderboard.sort_values(["vr", "asc"], ascending=False).iloc[0].to_dict()
    artifact_keys = ("artifact", "pdf", "gridsearch_csv", "metrics_csv", "visual_manifest")
    artifacts = {key: Path(raw[key]) for key in artifact_keys if raw.get(key)}
    output = Path(raw.get("artifact", cfg["export"]["out_dir"])).parent
    best_k = metrics.get("k")
    return GeoFarmResult(
        output_directory=output, metrics=metrics, leaderboard=leaderboard,
        best_model=metrics.get("algo"), best_k=int(best_k) if pd.notna(best_k) else None,
        best_labels=raw.get("best_labels"), artifacts=artifacts,
        configuration=deepcopy(cfg), cache_hit=bool(raw.get("cache_hit", False)),
    )


def run_pipeline(config="configs/example.yaml", force=False, parallel_gridsearch=None, n_jobs=None):
    """Run GeoFarmAI from YAML and return a :class:`GeoFarmResult`."""
    return GeoFarmPipeline.from_yaml(config).run(
        force=force, parallel_gridsearch=parallel_gridsearch, n_jobs=n_jobs,
    )
