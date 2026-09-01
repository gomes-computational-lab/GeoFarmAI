from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point, shape
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core.cluster import run_agglomerative, run_fcm, run_gmm, run_kmeans
from core.evaluate import anova_p, variance_reduction
from services.project_visuals import write_visual_manifest
from services.workspace_manifest import refresh_workspace_manifest

try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover - optional dependency guard
    OrdinaryKriging = None  # type: ignore

try:
    from multispaeti import MultispatiPCA
except Exception:  # pragma: no cover - optional dependency guard
    MultispatiPCA = None  # type: ignore


_GRIDSEARCH_CONTEXT: Dict[str, Any] = {}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|()\s%]+", "_", str(value)).strip("_")
    return re.sub(r"_+", "_", cleaned)


def build_run_fingerprint(
    cfg: Dict[str, Any],
    target_crs: str,
    experiment: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> str:
    project = cfg["project"]
    soil_cfg = project["soil"]
    yield_cfg = project["yield"]
    payload = {
        "version": 4,
        "experiment": experiment,
        "metadata": metadata or {},
        "target_crs": target_crs,
        "project": project,
        "weights": cfg.get("weights", {}),
        "spatial_pca": cfg.get("spatial_pca", {}),
        "clustering": cfg.get("clustering", {}),
        "postprocess": cfg.get("postprocess", {}),
        "raster": cfg.get("raster", {}),
        "inputs": {
            "soil": file_digest(soil_cfg["path"]),
            "yield": file_digest(yield_cfg["path"]),
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_digest(path: str) -> Dict[str, Any]:
    candidate = Path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_cached_raster_result(manifest_path: Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("fingerprint") != fingerprint:
        print("[raster][cache] Input/config fingerprint changed; recomputing.")
        return None

    result = manifest.get("result")
    if not isinstance(result, dict) or not cached_outputs_exist(result):
        print("[raster][cache] Manifest found but expected outputs are incomplete; recomputing.")
        return None

    leaderboard_path = result.get("gridsearch_csv")
    if leaderboard_path and Path(leaderboard_path).exists():
        result["leaderboard"] = pd.read_csv(leaderboard_path).to_dict(orient="records")
    result["images"] = sorted(str(path) for path in manifest_path.parent.glob("preview/*.png"))
    result["cache_hit"] = True
    return result


def cached_outputs_exist(result: Dict[str, Any]) -> bool:
    required = [result.get("artifact"), result.get("pdf"), result.get("metrics_csv"), result.get("gridsearch_csv")]
    return all(path and Path(path).exists() for path in required)


def write_run_manifest(manifest_path: Path, fingerprint: str, result: Dict[str, Any]) -> None:
    manifest = {
        "fingerprint": fingerprint,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "result": result,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[raster][cache] Wrote run manifest: {manifest_path}")


def run_raster_mzd_flow(
    cfg: Dict[str, Any],
    experiment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    if OrdinaryKriging is None:
        raise ImportError("PyKrige is required for raster kriging. Install pykrige in the ag-gpt environment.")

    project = cfg["project"]
    raster_cfg = cfg.get("raster", {})
    out_root = Path(cfg.get("export", {}).get("out_dir", "outputs"))
    suffix = f"_{safe_name(experiment)}" if experiment else ""
    run_dir = out_root / f"{project['name']}_raster{suffix}"
    raster_dir = run_dir / "rasters"
    preview_dir = run_dir / "preview"
    cluster_dir = run_dir / "clusters"
    for path in [raster_dir, preview_dir, cluster_dir]:
        path.mkdir(parents=True, exist_ok=True)

    crs_in = project.get("crs_in", "EPSG:4326")
    soil_cfg = project["soil"]
    yield_cfg = project["yield"]
    target_crs = resolve_target_crs(raster_cfg, soil_cfg["path"], soil_cfg["x"], soil_cfg["y"], crs_in)
    print(f"[raster] Using target CRS: {target_crs}")
    yield_column = yield_cfg.get("column", project.get("yield_column", "yield"))
    shared_bounds = choose_max_extent_bounds(
        soil_cfg,
        yield_cfg,
        crs_in,
        target_crs,
        buffer=float(raster_cfg.get("buffer", 0.0)),
    )
    manifest_path = run_dir / "run_manifest.json"
    fingerprint = build_run_fingerprint(cfg, target_crs, experiment, metadata)
    if raster_cfg.get("cache", True) and not force:
        cached = load_cached_raster_result(manifest_path, fingerprint)
        if cached is not None:
            print(f"[raster][cache] Inputs/config unchanged. Reusing outputs in {run_dir}")
            return cached
    elif force:
        print("[raster][cache] Force requested; recomputing raster pipeline.")

    print("[raster] Kriging soil variables")
    soil_rasters = krige_csv_variables(
        csv_path=soil_cfg["path"],
        x_col=soil_cfg["x"],
        y_col=soil_cfg["y"],
        variables=soil_cfg.get("variables", []),
        crs_in=crs_in,
        target_crs=target_crs,
        output_dir=raster_dir / "soil",
        raster_cfg=raster_cfg,
        preview_dir=preview_dir,
        bounds=shared_bounds,
    )

    print("[raster] Kriging yield variable")
    yield_rasters = krige_csv_variables(
        csv_path=yield_cfg["path"],
        x_col=yield_cfg["x"],
        y_col=yield_cfg["y"],
        variables=[yield_column],
        crs_in=crs_in,
        target_crs=target_crs,
        output_dir=raster_dir / "yield",
        raster_cfg=raster_cfg,
        preview_dir=preview_dir,
        bounds=shared_bounds,
    )

    pca_variables = raster_cfg.get("pca_variables") or soil_cfg.get("variables", [])
    required = soil_cfg.get("required_variables", [])
    missing_required = [name for name in required if name not in soil_rasters]
    if missing_required:
        raise ValueError(f"Missing required raster variables: {missing_required}")
    missing_pca = [name for name in pca_variables if name not in soil_rasters]
    if missing_pca:
        print(f"[warn] Skipping PCA variables without rasters: {missing_pca}")
    pca_variables = [name for name in pca_variables if name in soil_rasters]
    if len(pca_variables) < cfg["spatial_pca"]["n_components"]:
        raise ValueError(
            f"Need at least {cfg['spatial_pca']['n_components']} raster variables for PCA; found {pca_variables}"
        )

    print("[raster] Aligning soil raster stack")
    soil_paths = [soil_rasters[name]["tif"] for name in pca_variables]
    arrays, profile, valid_mask, bounds = stack_rasters(
        soil_paths,
        ref_index=raster_cfg.get("ref_index", 0),
        resampling=Resampling.bilinear,
    )

    yield_array = None
    yield_path = yield_rasters.get(yield_column, {}).get("tif")
    if yield_path:
        yield_array = align_single_raster(yield_path, profile, bounds, Resampling.bilinear)

    print("[raster] Building clustering feature matrix")
    scores, feature_summary, used_multispaeti, connectivity, score_names = spatial_pca_from_stack(
        arrays,
        valid_mask,
        profile,
        cfg,
        pca_variables,
    )
    representation = "pca" if raster_cfg.get("use_pca", True) else "raw"
    feature_dir = run_dir / ("pca" if representation == "pca" else "raw_features")
    feature_paths = write_component_rasters(scores, valid_mask, arrays.shape[:2], profile, feature_dir, preview_dir, score_names)
    feature_stats_path = feature_dir / ("pca_summary_stats.txt" if representation == "pca" else "raw_feature_summary_stats.txt")
    feature_stats_path.parent.mkdir(parents=True, exist_ok=True)
    feature_stats_path.write_text(feature_summary, encoding="utf-8")

    print("[raster] Running clustering grid search")
    best, leaderboard, selections = raster_gridsearch(scores, yield_array, valid_mask, connectivity, cfg)
    comparison_paths = write_metric_comparison_plots(leaderboard, preview_dir)
    cluster_paths = write_cluster_outputs(
        selections,
        leaderboard,
        valid_mask,
        arrays.shape[:2],
        profile,
        cluster_dir,
        preview_dir,
    )

    zones_path = run_dir / f"{project['name']}_zones.gpkg"
    zones = zones_from_label_raster(best["labels"], valid_mask, arrays.shape[:2], profile, cfg["postprocess"]["min_area_m2"])
    zones.to_file(zones_path, layer="zones", driver="GPKG")

    metrics = {
        **best["metrics"],
        "experiment": experiment or "baseline",
        "pipeline": "raster",
        "feature_representation": representation,
        "used_multispaeti": used_multispaeti,
        "pca_variables": ",".join(pca_variables),
    }
    for key, value in (metadata or {}).items():
        metrics[f"param__{key}"] = value
    metrics_path = run_dir / f"{project['name']}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    leaderboard_path = run_dir / f"{project['name']}_gridsearch.csv"
    pd.DataFrame(leaderboard).to_csv(leaderboard_path, index=False)

    pdf_path = run_dir / f"{project['name']}.pdf"
    image_paths = export_raster_pdf(pdf_path, preview_dir, project["name"])
    visual_manifest_path = write_visual_manifest(cfg)

    result = {
        "artifact": str(zones_path),
        "pdf": str(pdf_path),
        "images": image_paths,
        "gridsearch_csv": str(leaderboard_path),
        "metrics_csv": str(metrics_path),
        "visual_manifest": str(visual_manifest_path) if visual_manifest_path else None,
        "cell_m": None,
        "used_r_multispati": False,
        "used_multispaeti": used_multispaeti,
        "leaderboard": leaderboard,
        "rasters": {
            "soil": soil_rasters,
            "yield": yield_rasters,
            "features": feature_paths,
            "clusters": cluster_paths,
            "comparisons": comparison_paths,
        },
    }
    write_run_manifest(manifest_path, fingerprint, result)
    refresh_workspace_manifest(cfg)
    return result

def dataset_extent(csv_path: str, x_col: str, y_col: str, crs_in: str, target_crs: str, buffer: float = 0.0) -> Tuple[float, float, float, float]:
    _, x, y = load_points(csv_path, x_col, y_col, crs_in, target_crs)
    return x.min() - buffer, x.max() + buffer, y.min() - buffer, y.max() + buffer

def choose_max_extent_bounds(
    soil_cfg: Dict[str, Any],
    yield_cfg: Dict[str, Any],
    crs_in: str,
    target_crs: str,
    buffer: float = 0.0,
) -> Tuple[float, float, float, float]:
    soil_bounds = dataset_extent(soil_cfg["path"], soil_cfg["x"], soil_cfg["y"], crs_in, target_crs, buffer)
    yield_bounds = dataset_extent(yield_cfg["path"], yield_cfg["x"], yield_cfg["y"], crs_in, target_crs, buffer)
    soil_area = (soil_bounds[1] - soil_bounds[0]) * (soil_bounds[3] - soil_bounds[2])
    yield_area = (yield_bounds[1] - yield_bounds[0]) * (yield_bounds[3] - yield_bounds[2])
    if yield_area > soil_area:
        print(f"[raster] yield extent is larger ({yield_area:,.0f} m^2 vs {soil_area:,.0f} m^2); using yield bounds for kriging")
        return yield_bounds
    print(f"[raster] soil extent is larger ({soil_area:,.0f} m^2 vs {yield_area:,.0f} m^2); using soil bounds for kriging")
    return soil_bounds

def krige_csv_variables(
    csv_path: str,
    x_col: str,
    y_col: str,
    variables: Iterable[str],
    crs_in: str,
    target_crs: str,
    output_dir: Path,
    raster_cfg: Dict[str, Any],
    preview_dir: Path,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    gdf, x, y = load_points(csv_path, x_col, y_col, crs_in, target_crs)
    grid_x, grid_y, cell_size = build_arcgis_grid(
        x,
        y,
        buffer=float(raster_cfg.get("buffer", 0.0)),
        cell_size=raster_cfg.get("cell_size"),
    )
    coordinates_type = "geographic" if str(target_crs).upper() == "EPSG:4326" else "euclidean"
    results: Dict[str, Dict[str, Any]] = {}
    for variable in variables:
        if variable not in gdf.columns:
            print(f"[warn] Skipping '{variable}': column not found in {csv_path}")
            continue
        values = pd.to_numeric(gdf[variable], errors="coerce")
        valid = values.notna()
        if valid.sum() < 3:
            print(f"[warn] Skipping '{variable}': fewer than 3 numeric values")
            continue
        name = safe_name(variable)
        var_dir = output_dir / name
        var_dir.mkdir(parents=True, exist_ok=True)
        tif_path = var_dir / f"{name}_ok.tif"
        stats_path = var_dir / f"{name}_stats.txt"
        z_grid, params = ordinary_krige(
            x[valid.to_numpy()],
            y[valid.to_numpy()],
            values[valid].to_numpy(dtype=float),
            grid_x,
            grid_y,
            raster_cfg,
            coordinates_type,
        )
        write_raster(z_grid, tif_path, target_crs, grid_x, grid_y, cell_size)
        stats_path.write_text(build_kriging_stats(variable, z_grid, params), encoding="utf-8")
        preview_path = preview_dir / f"kriging_{name}.png"
        write_raster_preview(
            tif_path,
            preview_path,
            title=f"Kriging: {variable}",
            colorbar=True,
            colorbar_label=variable,
        )
        results[variable] = {
            "tif": str(tif_path),
            "stats": str(stats_path),
            "preview": str(preview_path),
            "variogram_params": [float(v) for v in np.asarray(params).ravel()],
        }
    return results


def resolve_target_crs(raster_cfg: Dict[str, Any], csv_path: str, x_col: str, y_col: str, crs_in: str) -> str:
    configured = raster_cfg.get("target_crs", "auto_utm")
    if configured and str(configured).lower() not in {"auto", "auto_utm", "utm"}:
        return str(configured)

    df = pd.read_csv(csv_path, usecols=[x_col, y_col])
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
    if df.empty:
        raise ValueError(f"Cannot auto-select UTM CRS because no valid coordinates were found in {csv_path}")

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs_in,
    )
    if points.crs is None:
        raise ValueError("Cannot auto-select UTM CRS because the input CRS is missing.")
    if not points.crs.is_geographic:
        return str(points.crs)

    lon = points.geometry.x.mean()
    lat = points.geometry.y.mean()
    zone = int(np.floor((lon + 180) / 6) + 1)
    epsg = f"EPSG:{326 if lat >= 0 else 327}{zone:02d}"
    return epsg


def load_points(csv_path: str, x_col: str, y_col: str, crs_in: str, target_crs: str) -> Tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing coordinate column '{col}' in {csv_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col]).reset_index(drop=True)
    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs_in).to_crs(target_crs)
    return gdf, gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()


def build_arcgis_grid(
    x: np.ndarray,
    y: np.ndarray,
    buffer: float = 0.0,
    cell_size: Optional[float] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
    else:
        x_min, x_max = x.min() - buffer, x.max() + buffer
        y_min, y_max = y.min() - buffer, y.max() + buffer
    width = x_max - x_min
    height = y_max - y_min
    if cell_size is None:
        cell_size = min(width, height) / 250
    grid_x = np.arange(x_min, x_max, cell_size)
    grid_y = np.arange(y_min, y_max, cell_size)
    return grid_x, grid_y, float(cell_size)


def ordinary_krige(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    raster_cfg: Dict[str, Any],
    coordinates_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model = OrdinaryKriging(
        x,
        y,
        values,
        variogram_model=raster_cfg.get("variogram_model", raster_cfg.get("kriging", {}).get("variogram_model", "spherical")),
        nlags=int(raster_cfg.get("nlags", raster_cfg.get("kriging", {}).get("nlags", 12))),
        weight=bool(raster_cfg.get("weight", raster_cfg.get("kriging", {}).get("weight", False))),
        enable_plotting=False,
        coordinates_type=coordinates_type,
    )
    z_grid, _ = model.execute(
        "grid",
        grid_x,
        grid_y[::-1],
        backend=raster_cfg.get("backend", "loop"),
        n_closest_points=int(raster_cfg.get("n_closest_points", raster_cfg.get("kriging", {}).get("n_closest_points", 24))),
    )
    return np.asarray(z_grid, dtype="float64"), np.asarray(model.variogram_model_parameters)


def write_raster(array: np.ndarray, out_path: Path, crs: str, grid_x: np.ndarray, grid_y: np.ndarray, cell_size: float) -> None:
    transform = from_origin(grid_x.min(), grid_y.max(), cell_size, cell_size)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(array, 1)


def build_kriging_stats(variable: str, z_grid: np.ndarray, params: np.ndarray, n_intervals: int = 10) -> str:
    lines = [
        f"Variable: {variable}",
        "",
        "Kriging Prediction Range",
        f"Min: {np.nanmin(z_grid):.5f}",
        f"Max: {np.nanmax(z_grid):.5f}",
        f"Mean: {np.nanmean(z_grid):.5f}",
        "",
        f"Fitted variogram parameters: {params}",
        "",
        "Raster Value Intervals",
    ]
    levels = np.linspace(np.nanmin(z_grid), np.nanmax(z_grid), n_intervals + 1)
    for low, high in zip(levels[:-1], levels[1:]):
        lines.append(f"{low:.5f} - {high:.5f}")
    return "\n".join(lines)


def stack_rasters(paths: List[str], ref_index: int = 0, resampling: Resampling = Resampling.bilinear):
    bounds = intersection_bounds(paths)
    arrays, profile = intersect_and_resample(paths, bounds, ref_index, resampling)
    stack = np.stack(arrays, axis=-1)
    valid_mask = ~np.any(np.isnan(stack), axis=-1)
    return stack, profile, valid_mask.reshape(-1), bounds


def intersection_bounds(paths: List[str]) -> Tuple[float, float, float, float]:
    lefts: List[float] = []
    bottoms: List[float] = []
    rights: List[float] = []
    tops: List[float] = []
    for path in paths:
        with rasterio.open(path) as src:
            b = src.bounds
            lefts.append(b.left)
            bottoms.append(b.bottom)
            rights.append(b.right)
            tops.append(b.top)
    left, bottom, right, top = max(lefts), max(bottoms), min(rights), min(tops)
    if left >= right or bottom >= top:
        raise ValueError("Rasters do not overlap.")
    return left, bottom, right, top


def intersect_and_resample(paths: List[str], bounds, ref_index: int, resampling: Resampling):
    left, bottom, right, top = bounds
    with rasterio.open(paths[ref_index]) as ref:
        ref_res = ref.res
        ref_crs = ref.crs
    width = int(round((right - left) / ref_res[0]))
    height = int(round((top - bottom) / ref_res[1]))
    transform = from_origin(left, top, ref_res[0], ref_res[1])
    arrays = [align_single_raster(path, {"height": height, "width": width, "crs": ref_crs, "transform": transform}, bounds, resampling) for path in paths]
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float64",
        "crs": ref_crs,
        "transform": transform,
        "nodata": np.nan,
    }
    return arrays, profile


def align_single_raster(path: str, profile: Dict[str, Any], bounds, resampling: Resampling) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), np.nan, dtype="float64")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=resampling,
        )
    return destination


def spatial_pca_from_stack(stack: np.ndarray, valid_mask: np.ndarray, profile: Dict[str, Any], cfg: Dict[str, Any], variables: List[str]):
    rows, cols, n_vars = stack.shape
    flat = stack.reshape(-1, n_vars)
    valid_pixels = flat[valid_mask]
    scaled = StandardScaler().fit_transform(valid_pixels)
    row_idx, col_idx = np.where(valid_mask.reshape(rows, cols))
    xs, ys = rasterio.transform.xy(profile["transform"], row_idx, col_idx)
    coords = np.column_stack([xs, ys])
    connectivity = kneighbors_graph(
        coords,
        n_neighbors=int(cfg.get("weights", {}).get("k", 8)),
        mode="connectivity",
        include_self=False,
    ).tocsr()
    if not cfg.get("raster", {}).get("use_pca", True):
        summary = build_raw_feature_summary(variables)
        score_names = [safe_name(name) for name in variables]
        return scaled, summary, False, connectivity, score_names

    n_components = int(cfg["spatial_pca"]["n_components"])
    if MultispatiPCA is not None and cfg["spatial_pca"].get("engine", "multispaeti") == "multispaeti":
        pca = MultispatiPCA(n_components=n_components, connectivity=connectivity, random_state=cfg.get("raster", {}).get("random_state", 42))
        scores = pca.fit_transform(scaled)
        summary = build_multispaeti_summary(pca, variables)
        return scores, summary, True, connectivity, [f"PC{i + 1}" for i in range(scores.shape[1])]
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)
    summary = build_pca_summary(pca, variables)
    return scores, summary, False, connectivity, [f"PC{i + 1}" for i in range(scores.shape[1])]


def build_raw_feature_summary(variables: List[str]) -> str:
    return "\n".join(
        [
            "Raw standardized raster feature summary",
            "",
            "Clustering used standardized kriged raster variables directly, without PCA/MULTISPATI dimensionality reduction.",
            "Variables: " + ", ".join(variables),
        ]
    )


def build_multispaeti_summary(pca: Any, variables: List[str]) -> str:
    lines = ["MULTISPATI-PCA summary", ""]
    for attr, label in [("eigenvalues_", "Eigenvalues"), ("variance_", "Variance"), ("moransI_", "Moran's I")]:
        values = getattr(pca, attr, None)
        if values is not None:
            lines.append(f"{label}: " + ", ".join(f"{float(v):.4f}" for v in values))
    components = getattr(pca, "components_", None)
    if components is not None:
        lines.append("")
        lines.append("Loadings:")
        for i, comp in enumerate(components, start=1):
            lines.append(f"PC{i}: " + ", ".join(f"{name}={weight:.3f}" for name, weight in zip(variables, comp)))
    return "\n".join(lines)


def build_pca_summary(pca: PCA, variables: List[str]) -> str:
    lines = ["PCA fallback summary", ""]
    lines.append("Explained variance ratio: " + ", ".join(f"{v:.4f}" for v in pca.explained_variance_ratio_))
    lines.append("")
    lines.append("Loadings:")
    for i, comp in enumerate(pca.components_, start=1):
        lines.append(f"PC{i}: " + ", ".join(f"{name}={weight:.3f}" for name, weight in zip(variables, comp)))
    return "\n".join(lines)


def write_component_rasters(scores, valid_mask, shape2d, profile, output_dir: Path, preview_dir: Path, names: List[str]) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, name in enumerate(names):
        safe = safe_name(name)
        path = output_dir / f"{safe}.tif"
        write_flat_raster(scores[:, index], valid_mask, shape2d, profile, path)
        write_raster_preview(
            path,
            preview_dir / f"feature_{safe}.png",
            title=name,
            colorbar=True,
            colorbar_label=name,
        )
        paths.append(str(path))
    return paths


def _set_gridsearch_context(scores, y_valid, connectivity_sym, sample_size) -> None:
    global _GRIDSEARCH_CONTEXT
    _GRIDSEARCH_CONTEXT = {
        "scores": scores,
        "y_valid": y_valid,
        "connectivity_sym": connectivity_sym,
        "sample_size": sample_size,
    }


def _run_gridsearch_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    scores = _GRIDSEARCH_CONTEXT["scores"]
    y_valid = _GRIDSEARCH_CONTEXT["y_valid"]
    connectivity_sym = _GRIDSEARCH_CONTEXT["connectivity_sym"]
    sample_size = _GRIDSEARCH_CONTEXT["sample_size"]

    algo = candidate["algo"]
    k = candidate["k"]
    seed = candidate["seed"]
    label = candidate["label"]
    start = time.perf_counter()

    if algo == "kmeans":
        labels, metrics = run_kmeans(scores, k, random_state=seed or 42, sample_size=sample_size)
    elif algo == "agglomerative":
        labels, metrics = run_agglomerative(scores, k, connectivity=connectivity_sym, sample_size=sample_size)
    elif algo == "gmm":
        labels, metrics = run_gmm(scores, k, random_state=seed or 42, sample_size=sample_size)
    elif algo == "fcm":
        labels, metrics = run_fcm(scores, k, random_state=seed, sample_size=sample_size)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algo}")

    row = {"k": k, "algo": algo, "seed": seed, **metrics}
    if y_valid is not None:
        row["vr"] = variance_reduction(pd.Series(y_valid), labels)
        row["anova_p"] = anova_p(pd.Series(y_valid), labels)

    return {
        "run_index": candidate["run_index"],
        "total": candidate["total"],
        "label": label,
        "labels": labels,
        "row": row,
        "elapsed": time.perf_counter() - start,
    }


def _build_gridsearch_candidates(k_values, algorithms, seeds) -> List[Dict[str, Any]]:
    planned = []
    for k in k_values:
        for algo in algorithms:
            seed_values = seeds if algo in {"kmeans", "gmm", "fcm"} else [None]
            for seed in seed_values:
                planned.append(
                    {
                        "k": k,
                        "algo": algo,
                        "seed": seed,
                        "label": f"algo={algo}, k={k}, seed={seed if seed is not None else 'n/a'}",
                    }
                )
    total = len(planned)
    for index, item in enumerate(planned, start=1):
        item["run_index"] = index
        item["total"] = total
    return planned


def raster_gridsearch(scores, yield_array, valid_mask, connectivity, cfg):
    algorithms = cfg["clustering"].get("algorithms", ["kmeans"])
    seeds = cfg["clustering"].get("seeds", [42])
    sample_size = cfg.get("raster", {}).get("silhouette_sample_size", 20000)
    y_valid = yield_array.reshape(-1)[valid_mask] if yield_array is not None else None
    best = None
    best_score = None
    best_ch = None
    best_ch_score = None
    leaderboard = []
    connectivity_sym = connectivity.maximum(connectivity.T)
    candidates = _build_gridsearch_candidates(cfg["clustering"]["k_values"], algorithms, seeds)
    total = len(candidates)
    raster_cfg = cfg.get("raster", {})
    parallel = bool(raster_cfg.get("parallel_gridsearch", False))
    n_jobs = int(raster_cfg.get("n_jobs", 1) or 1)
    if n_jobs < 1:
        n_jobs = 1
    if parallel and n_jobs == 1:
        n_jobs = min(os.cpu_count() or 1, total)
    print(
        "[raster][gridsearch] "
        f"{scores.shape[0]} valid pixels, {scores.shape[1]} PCA components, "
        f"{total} clustering candidate(s), silhouette sample size={sample_size}, "
        f"mode={'parallel' if parallel and n_jobs > 1 else 'serial'}, n_jobs={n_jobs}"
    )

    def handle_result(result: Dict[str, Any]) -> None:
        nonlocal best, best_score, best_ch, best_ch_score
        row = result["row"]
        labels = result["labels"]
        leaderboard.append(row.copy())
        score = (row.get("vr", 0.0), row.get("asc", 0.0))
        vr_text = _format_metric(row.get("vr"))
        asc_text = _format_metric(row.get("asc"))
        ch_text = _format_metric(row.get("ch_score"))
        print(
            f"[raster][gridsearch] {result['run_index']}/{total} finished {result['label']} "
            f"in {result['elapsed']:.1f}s | vr={vr_text}, asc={asc_text}, ch={ch_text}"
        )
        if best_score is None or score > best_score:
            best_score = score
            best = {"labels": labels, "metrics": row.copy()}
            print(
                f"[raster][gridsearch] new best: {result['label']} "
                f"with vr={vr_text}, asc={asc_text}, ch={ch_text}"
            )
        ch_value = row.get("ch_score")
        if isinstance(ch_value, (int, float, np.floating)) and not pd.isna(ch_value):
            if best_ch_score is None or float(ch_value) > best_ch_score:
                best_ch_score = float(ch_value)
                best_ch = {"labels": labels, "metrics": row.copy()}
                print(f"[raster][gridsearch] new CH best: {result['label']} with ch={ch_text}")

    _set_gridsearch_context(scores, y_valid, connectivity_sym, sample_size)
    if parallel and n_jobs > 1:
        for candidate in candidates:
            print(f"[raster][gridsearch] {candidate['run_index']}/{total} queued {candidate['label']}")
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_set_gridsearch_context,
            initargs=(scores, y_valid, connectivity_sym, sample_size),
        ) as executor:
            futures = [executor.submit(_run_gridsearch_candidate, candidate) for candidate in candidates]
            for future in as_completed(futures):
                handle_result(future.result())
    else:
        for candidate in candidates:
            print(f"[raster][gridsearch] {candidate['run_index']}/{total} starting {candidate['label']}")
            handle_result(_run_gridsearch_candidate(candidate))
    if best is not None:
        metrics = best["metrics"]
        print(
            "[raster][gridsearch] best by yield variance reduction: "
            f"algo={metrics.get('algo')}, k={metrics.get('k')}, seed={metrics.get('seed')} "
            f"| vr={_format_metric(metrics.get('vr'))}, asc={_format_metric(metrics.get('asc'))}, "
            f"ch={_format_metric(metrics.get('ch_score'))}"
        )
    if best_ch is not None:
        metrics = best_ch["metrics"]
        print(
            "[raster][gridsearch] best by Calinski-Harabasz pseudo-F: "
            f"algo={metrics.get('algo')}, k={metrics.get('k')}, seed={metrics.get('seed')} "
            f"| ch={_format_metric(metrics.get('ch_score'))}, asc={_format_metric(metrics.get('asc'))}, "
            f"vr={_format_metric(metrics.get('vr'))}"
        )
    selections = {"yield_variance": best}
    if best_ch is not None:
        selections["ch_score"] = best_ch
    return best, leaderboard, selections


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
        return f"{float(value):.4f}"
    return "n/a"


def write_metric_comparison_plots(leaderboard: List[Dict[str, Any]], preview_dir: Path) -> List[str]:
    if not leaderboard:
        return []

    df = pd.DataFrame(leaderboard)
    if df.empty or "algo" not in df.columns or "k" not in df.columns:
        return []

    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    metric_specs = [
        ("ch_score", "Calinski-Harabasz Pseudo-F Score", "Inter-cluster / intra-cluster dispersion, higher is better", False),
        ("vr", "Best Yield Variance Reduction", "Higher is better", False),
        ("asc", "Best Silhouette Score", "Higher is better", False),
        ("anova_p", "Best ANOVA Significance", "-log10(p), higher is better", True),
    ]

    for metric, title, ylabel, transform_p in metric_specs:
        if metric not in df.columns:
            continue

        plot_df = df[["algo", "k", metric]].copy()
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[metric])
        if plot_df.empty:
            continue

        value_col = metric
        if transform_p:
            value_col = "neg_log10_p"
            plot_df[value_col] = -np.log10(plot_df[metric].clip(lower=np.finfo(float).tiny))

        grouped = plot_df.groupby(["algo", "k"], as_index=False)[value_col].max()
        pivot = grouped.pivot(index="k", columns="algo", values=value_col).sort_index()
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(ax=ax, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Method", loc="best")
        fig.tight_layout()

        out_path = preview_dir / f"comparison_{metric}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_path))
        print(f"[raster][gridsearch] wrote comparison plot: {out_path}")

    return paths


def write_cluster_outputs(selections, leaderboard, valid_mask, shape2d, profile, output_dir: Path, preview_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[str] = []
    output_specs = [
        ("yield_variance", "best_clusters", "clusters_best", "Best Cluster Solution by Yield Variance Reduction"),
        ("ch_score", "best_ch_score_clusters", "clusters_best_ch_score", "Best Cluster Solution by Calinski-Harabasz Pseudo-F"),
    ]
    for key, stem, preview_stem, title in output_specs:
        selection = selections.get(key)
        if not selection:
            continue
        labels = selection["labels"]
        path = output_dir / f"{stem}.tif"
        write_flat_raster(labels.astype("float32"), valid_mask, shape2d, profile, path, dtype="float32", nodata=-9999)
        write_raster_preview(path, preview_dir / f"{preview_stem}.png", title=title, categorical=True)
        written_paths.append(str(path))

    for row in leaderboard:
        stats_path = output_dir / f"{row['algo']}_k{row['k']}_seed{row.get('seed', 'none')}_stats.txt"
        stats_path.write_text("\n".join(f"{key}: {value}" for key, value in row.items()), encoding="utf-8")
    return written_paths


def write_flat_raster(values, valid_mask, shape2d, profile, out_path: Path, dtype="float32", nodata=np.nan) -> None:
    rows, cols = shape2d
    full = np.full(rows * cols, nodata, dtype=dtype)
    full[valid_mask] = values
    full = full.reshape(rows, cols)
    out_profile = profile.copy()
    out_profile.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(full, 1)


def zones_from_label_raster(labels, valid_mask, shape2d, profile, min_area: float) -> gpd.GeoDataFrame:
    rows, cols = shape2d
    full = np.full(rows * cols, -9999, dtype="int32")
    full[valid_mask] = labels.astype("int32")
    full = full.reshape(rows, cols)
    mask = full != -9999
    records = []
    for geom, value in shapes(full, mask=mask, transform=profile["transform"]):
        records.append({"zone": int(value), "geometry": shape(geom)})
    if not records:
        return gpd.GeoDataFrame(columns=["zone", "zone_id", "geometry"], geometry="geometry", crs=profile["crs"])
    zones = gpd.GeoDataFrame(records, geometry="geometry", crs=profile["crs"])
    zones = zones.dissolve(by="zone").reset_index()
    area_source = zones
    if area_source.crs is not None and area_source.crs.is_geographic:
        area_source = area_source.to_crs(area_source.estimate_utm_crs())
    zones = zones.loc[area_source.area >= min_area].copy()
    zones["zone_id"] = zones["zone"]
    return zones


def write_raster_preview(
    tif_path: Path,
    png_path: Path,
    title: str,
    categorical: bool = False,
    colorbar: bool = False,
    colorbar_label: str = "Value",
) -> None:
    with rasterio.open(tif_path) as src:
        data = src.read(1, masked=True)
        bounds = src.bounds
    rows, cols = data.shape
    map_aspect = cols / rows if rows else 1.0
    fig_width = 10.0
    fig_height = max(2.0, min(6.0, fig_width / map_aspect + 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = "tab20" if categorical else "viridis"
    image = ax.imshow(
        data,
        cmap=cmap,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        aspect="equal",
    )
    if colorbar and ax.images:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(colorbar_label)
    ax.set_title(title, pad=8)
    ax.set_axis_off()
    fig.savefig(png_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def export_raster_pdf(pdf_path: Path, preview_dir: Path, project_name: str) -> List[str]:
    images = sorted(preview_dir.glob("*.png"))
    with PdfPages(pdf_path) as pdf:
        for image in images:
            array = plt.imread(image)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(array)
            ax.set_title(f"{project_name}: {image.stem}")
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return [str(path) for path in images]
