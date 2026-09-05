import argparse
import yaml
import pandas as pd
import geopandas as gpd
from prefect import flow, task
from core.logging_utils import run_log_context
from core.spatial import knn_weights
from core.multispati import multispati_components
from core.cluster import run_agglomerative, run_fcm, run_gmm, run_kmeans
from core.evaluate import variance_reduction, anova_p
from core.export import zones_from_points, save_package, export_pdf_report
from geofarmai.outcome import resolve_pipeline_outcome
from geofarmai.provenance import decomposition_metric_fields, vector_decomposition_provenance
from typing import Optional, Tuple, List

@task
def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _ingest_one(path: str, xcol: str, ycol: str, crs_in: str,
                idcol: Optional[str] = None,
                keep_cols: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)

    # Ensure required columns exist
    missing = [c for c in [xcol, ycol] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Coerce coordinates to numeric and drop invalid rows
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])

    # Handle optional ID: use provided column if present; else synthesize
    if idcol and idcol in df.columns:
        pass  # keep as is
    else:
        idcol = idcol or "row_id"
        df[idcol] = range(len(df))

    # If a keep list is given, subset (but always keep coords + id)
    base_cols = {idcol, xcol, ycol}
    if keep_cols:
        df = df[list(base_cols.union(keep_cols))]

    # Build GeoDataFrame with input CRS
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[xcol], df[ycol]),
        crs=crs_in
    )

    # Optional: drop exact duplicate points
    gdf = gdf.drop_duplicates(subset="geometry").reset_index(drop=True)
    # Add a canonical name for downstream steps if helpful
    gdf.rename(columns={idcol: "sample_id"}, inplace=True)

    return gdf

@task
def ingest_sources(
    cfg,
) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[str]]:
    """Ingest predictors and an optional explicitly configured outcome."""

    pj = cfg["project"]
    crs_in = pj.get("crs_in", "EPSG:4326")
    outcome = resolve_pipeline_outcome(pj)

    sj = pj["soil"]
    soil_keep = list(sj.get("variables", []))
    if outcome is not None:
        conflicting_name = outcome.name in soil_keep or (
            outcome.uses_predictor_source and outcome.source_column in soil_keep
        )
        if conflicting_name:
            raise ValueError(
                f"Configured outcome {outcome.name!r} conflicts with a predictor name in "
                "the legacy flat-table pipeline. Use a distinct outcome name."
            )
    if outcome is not None and outcome.uses_predictor_source:
        soil_keep.append(outcome.source_column)
    soil = _ingest_one(
        path=sj["path"],
        xcol=sj["x"],
        ycol=sj["y"],
        crs_in=crs_in,
        idcol=sj.get("id_column"),
        keep_cols=soil_keep or None,
    )

    if outcome is None:
        return soil, None, None

    if outcome.uses_predictor_source:
        if outcome.source_column != outcome.name:
            soil.rename(columns={outcome.source_column: outcome.name}, inplace=True)
        return soil, None, outcome.name

    outcome_cfg = outcome.source_config
    outcome_points = _ingest_one(
        path=outcome_cfg["path"],
        xcol=outcome_cfg["x"],
        ycol=outcome_cfg["y"],
        crs_in=crs_in,
        idcol=outcome_cfg.get("id_column"),
        keep_cols=[outcome.source_column],
    )
    if outcome.source_column != outcome.name:
        outcome_points.rename(columns={outcome.source_column: outcome.name}, inplace=True)
    return soil, outcome_points, outcome.name


@task
def ingest_two(cfg) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """Compatibility wrapper for the historical two-source task."""

    predictors, outcome, _ = ingest_sources.fn(cfg)
    return predictors, outcome

@task
def reproject_to_meters(predictors, outcome_points, cfg):
    if cfg["project"].get("auto_reproject_to_utm", True):
        from core.crs import to_utm_auto
        predictors, epsg = to_utm_auto(predictors)
        if outcome_points is not None:
            outcome_points = outcome_points.to_crs(epsg)
    return predictors, outcome_points

@task
def make_density_grid(predictors, outcome_points, cfg):
    from core.grid import build_analysis_grid, choose_cell_size
    cell = cfg["grid"].get("cell_size_m")
    if not cell:
        supports = [predictors[["geometry"]]]
        if outcome_points is not None:
            supports.append(outcome_points[["geometry"]])
        union = pd.concat(supports, ignore_index=True)
        union = gpd.GeoDataFrame(union, geometry="geometry", crs=predictors.crs)
        cell = choose_cell_size(union, cfg["grid"]["min_cell_size_m"], cfg["grid"]["max_cell_size_m"])
    grid = build_analysis_grid(predictors, cell, outcome_points=outcome_points)
    return grid, cell

@task
def reconcile_to_grid(
    predictors,
    outcome_points,
    grid,
    cfg,
    outcome_name: Optional[str] = None,
):
    from core.reconcile import populate_analysis_grid
    soil_vars = cfg["project"]["soil"]["variables"]
    if outcome_name is None:
        outcome = resolve_pipeline_outcome(cfg["project"])
        outcome_name = outcome.name if outcome is not None else None
    method = cfg["grid"].get("method", "idw")
    buffer_m = cfg["grid"].get("buffer_m", 15)
    table = populate_analysis_grid(
        predictors,
        grid,
        soil_vars,
        outcome_points=outcome_points,
        outcome_name=outcome_name,
        method=method,
        buffer_m=buffer_m,
    )
    return table

@task
def components_from_grid(table, cfg):
    required = cfg["project"]["soil"].get("required_variables", [])
    available = [c for c in required if c in table.columns]
    missing_required = [c for c in required if c not in available]
    if missing_required:
        print(f"[warn] Missing required soil variables: {missing_required}")
    else:
        print(f"[ok] All required soil variables present: {available}")

    soil_vars = cfg["project"]["soil"]["variables"]
    present_vars = [c for c in soil_vars if c in table.columns]
    n_components = cfg["spatial_pca"]["n_components"]
    if len(present_vars) < n_components:
        raise ValueError(
            f"Need at least {n_components} soil features; found {len(present_vars)}: {present_vars}"
        )
    else:
        print(f"[ok] Enough soil features for {n_components} components: {present_vars}")

    W = knn_weights(table, k=cfg["weights"]["k"])
    X = table[present_vars]
    Z, used_r = multispati_components(
        X,
        W,
        n_components=cfg["spatial_pca"]["n_components"],
        use_r=cfg["spatial_pca"]["use_r_multispati"]
    )
    return Z, W, used_r

@task
def spatial_weights(gdf, k):
    return knn_weights(gdf, k=k)

@task
def compute_components(gdf, W, cfg):
    X = gdf[cfg['project']['variables']]
    Z, used_r = multispati_components(
        X,
        W,
        n_components=cfg['spatial_pca']['n_components'],
        use_r=cfg['spatial_pca']['use_r_multispati']
    )
    return Z, used_r

@task
def gridsearch(gdf, Z, cfg, outcome_name: Optional[str] = None):
    seeds = cfg['clustering'].get('seeds', [42])
    best = None
    best_payload = None
    leaderboard = []
    X = Z.values
    if outcome_name is None:
        outcome = resolve_pipeline_outcome(cfg['project'])
        outcome_name = outcome.name if outcome is not None else None
    has_outcome = outcome_name is not None and outcome_name in gdf.columns
    for k in cfg['clustering']['k_values']:
        for algo in cfg['clustering']['algorithms']:
            seed_values = seeds if algo in {'gmm', 'fcm', 'kmeans'} else [None]
            for seed in seed_values:
                if algo == 'gmm':
                    labels, m = run_gmm(X, k, random_state=seed)
                elif algo == 'fcm':
                    labels, m = run_fcm(X, k, random_state=seed)
                elif algo == 'kmeans':
                    labels, m = run_kmeans(X, k, random_state=seed)
                elif algo == 'agglomerative':
                    labels, m = run_agglomerative(X, k)
                else:
                    raise ValueError(f"Unknown clustering algorithm: {algo}")
                metrics = {'k': k, 'algo': algo, 'seed': seed, **m}
                if has_outcome:
                    vr = variance_reduction(gdf[outcome_name], labels)
                    p = anova_p(gdf[outcome_name], labels)
                    metrics.update({'vr': vr, 'anova_p': p, 'outcome_name': outcome_name})
                leaderboard.append(metrics.copy())
                score = (
                    (metrics['vr'], metrics.get('asc', 0.0))
                    if has_outcome
                    else (metrics.get('asc', 0.0),)
                )
                if best is None or score > best:
                    best = score
                    best_payload = {'labels': labels, 'metrics': metrics.copy()}
    return best_payload, leaderboard

@task
def postprocess_and_export(
    gdf,
    labels,
    metrics,
    leaderboard,
    cfg,
    experiment: str | None = None,
    outcome_name: Optional[str] = None,
):
    zones = zones_from_points(gdf, labels, min_area=cfg['postprocess']['min_area_m2'])
    suffix = f"_{experiment}" if experiment else ""
    base_path = f"{cfg['export']['out_dir']}/{cfg['project']['name']}{suffix}"
    out = f"{base_path}.gpkg"
    labelled = gdf.assign(zone=labels)
    save_package(zones, labelled, metrics, out)

    feature_columns = list(cfg['project']['soil'].get('variables', []))
    if outcome_name is None:
        outcome = resolve_pipeline_outcome(cfg['project'])
        outcome_name = outcome.name if outcome is not None else None
    if outcome_name is not None and outcome_name in labelled.columns:
        feature_columns.append(outcome_name)
    feature_columns = list(dict.fromkeys(feature_columns))
    pdf_out = f"{base_path}.pdf"
    image_paths = export_pdf_report(
        zones,
        labelled,
        feature_columns,
        pdf_out,
        basemap=cfg['export'].get('basemap', 'web'),
    )
    leaderboard_path = None
    if leaderboard:
        leaderboard_path = f"{base_path}_gridsearch.csv"
        pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)
    return {
        "gpkg": out,
        "pdf": pdf_out,
        "images": image_paths,
        "gridsearch_csv": leaderboard_path,
    }

@flow(name="mzd_two_csvs")
def mzd_flow_two_csvs(
    cfg_path='configs/project.yaml',
    force: bool = False,
    parallel_gridsearch: Optional[bool] = None,
    n_jobs: Optional[int] = None,
):
    cfg = load_cfg(cfg_path)
    if parallel_gridsearch is not None or n_jobs is not None:
        cfg.setdefault("raster", {})
        if parallel_gridsearch is not None:
            cfg["raster"]["parallel_gridsearch"] = parallel_gridsearch
        if n_jobs is not None:
            cfg["raster"]["n_jobs"] = n_jobs
    out_dir = cfg.get('export', {}).get('out_dir', 'outputs')
    with run_log_context(out_dir, "baseline"):
        return _run_mzd_flow(cfg, force=force)


def _run_mzd_flow(cfg, force: bool = False):
    if cfg.get("raster", {}).get("enabled", False):
        from core.raster_pipeline import run_raster_mzd_flow
        return run_raster_mzd_flow(cfg, force=force)

    configured_outcome = resolve_pipeline_outcome(cfg.get("project", {}))
    outcome_name = configured_outcome.name if configured_outcome is not None else None
    soil, outcome_points = ingest_two(cfg)
    soil, outcome_points = reproject_to_meters(soil, outcome_points, cfg)
    grid, cell = make_density_grid(soil, outcome_points, cfg)
    if outcome_name is None:
        table = reconcile_to_grid(soil, outcome_points, grid, cfg)
    else:
        table = reconcile_to_grid(soil, outcome_points, grid, cfg, outcome_name)
    # proceed as before: clustering on Z from table
    Z, W, used_r = components_from_grid(table, cfg)
    if outcome_name is None:
        best_payload, leaderboard = gridsearch(table, Z, cfg)
    else:
        best_payload, leaderboard = gridsearch(table, Z, cfg, outcome_name)
    labels = best_payload['labels']
    decomposition = vector_decomposition_provenance(cfg, used_r)
    metrics = {
        **best_payload['metrics'],
        **decomposition_metric_fields(decomposition),
        "used_r_multispati": used_r,
        "experiment": "baseline",
    }
    leaderboard_aug = [dict(entry, experiment="baseline") for entry in leaderboard]
    if outcome_name is None:
        artifacts = postprocess_and_export(table, labels, metrics, leaderboard_aug, cfg)
    else:
        artifacts = postprocess_and_export(
            table, labels, metrics, leaderboard_aug, cfg, outcome_name=outcome_name
        )
    return {
        "artifact": artifacts["gpkg"],
        "pdf": artifacts["pdf"],
        "images": artifacts["images"],
        "gridsearch_csv": artifacts.get("gridsearch_csv"),
        "cell_m": cell,
        "used_r_multispati": used_r,
        "decomposition": decomposition,
        "leaderboard": leaderboard_aug,
        "outcome_name": outcome_name,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the management zone design pipeline")
    parser.add_argument("--cfg", default="configs/project.yaml", help="Path to project YAML")
    parser.add_argument("--force", action="store_true", help="Ignore cached raster outputs and recompute")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--parallel-gridsearch", action="store_true", help="Run raster clustering candidates with multiprocessing")
    group.add_argument("--serial-gridsearch", action="store_true", help="Run raster clustering candidates serially")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of multiprocessing workers for raster grid search")
    args = parser.parse_args()
    parallel_override = None
    if args.parallel_gridsearch:
        parallel_override = True
    elif args.serial_gridsearch:
        parallel_override = False
    mzd_flow_two_csvs(
        args.cfg,
        force=args.force,
        parallel_gridsearch=parallel_override,
        n_jobs=args.n_jobs,
    )
