from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from services.workspace_manifest import ProjectWorkspaceManifest, WorkspaceArtifact, artifact_by_id, artifact_path


logger = logging.getLogger(__name__)
_MAP_CACHE: Dict[Tuple[str, float], Dict[str, Any]] = {}


def invalidate_map_cache() -> None:
    _MAP_CACHE.clear()


def best_vr_map_payload(cfg: dict, manifest: ProjectWorkspaceManifest) -> Dict[str, Any]:
    source_artifact, source_path, gdf = _load_best_vr_zones(cfg, manifest)
    mtime = source_path.stat().st_mtime
    cache_key = (source_artifact.id, mtime)
    cached = _MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if gdf.empty:
        raise FileNotFoundError("Best VR map exists, but no valid zone geometries were found.")
    gdf = _prepare_zone_gdf(gdf)
    bounds = gdf.total_bounds
    center = {"lat": float((bounds[1] + bounds[3]) / 2), "lng": float((bounds[0] + bounds[2]) / 2)}
    gridsearch_row = _best_vr_metadata(cfg, manifest)
    zones = sorted({int(value) for value in gdf["zone"].dropna().tolist()})
    payload = {
        "project_id": manifest.project_id,
        "selection_metric": "vr",
        "source_artifact_id": source_artifact.id,
        "bounds": {"south": float(bounds[1]), "west": float(bounds[0]), "north": float(bounds[3]), "east": float(bounds[2])},
        "center": center,
        "zone_count": len(zones),
        "geojson": json.loads(gdf.to_json(drop_id=True)),
        "legend": [{"zone": zone, "label": f"Zone {zone}"} for zone in zones],
        "metadata": gridsearch_row,
    }
    _MAP_CACHE[cache_key] = payload
    return payload


def _load_best_vr_zones(cfg: dict, manifest: ProjectWorkspaceManifest) -> Tuple[WorkspaceArtifact, Path, gpd.GeoDataFrame]:
    gpkg = _candidate_zone_gpkg(cfg, manifest)
    if gpkg is not None:
        artifact, path = gpkg
        try:
            gdf = _read_zone_gpkg(path)
            if not gdf.empty:
                return artifact, path, gdf
        except Exception:
            logger.exception("Failed to read best VR zones from GeoPackage", extra={"artifact_id": artifact.id})

    raster = artifact_by_id(manifest, "best_clusters")
    raster_path = artifact_path(cfg, raster)
    return raster, raster_path, _polygonize_cluster_raster(raster_path)


def _candidate_zone_gpkg(cfg: dict, manifest: ProjectWorkspaceManifest) -> Optional[Tuple[WorkspaceArtifact, Path]]:
    candidates = [artifact for artifact in manifest.artifacts if artifact.artifact_type == "geopackage"]
    candidates.sort(key=lambda item: (0 if "zone" in item.name.lower() else 1, item.name))
    for artifact in candidates:
        try:
            return artifact, artifact_path(cfg, artifact)
        except Exception:
            logger.exception("Skipping invalid GeoPackage artifact", extra={"artifact_id": artifact.id})
    return None


def _read_zone_gpkg(path: Path) -> gpd.GeoDataFrame:
    layer_names: List[str] = []
    if hasattr(gpd, "list_layers"):
        layers = gpd.list_layers(path)
        layer_names = [str(row["name"]) for _, row in layers.iterrows()] if hasattr(layers, "iterrows") else []
    if not layer_names:
        layer_names = ["zones"]
    preferred = sorted(layer_names, key=lambda name: (0 if name.lower() == "zones" else 1, name))
    for layer in preferred:
        try:
            gdf = gpd.read_file(path, layer=layer)
        except Exception:
            continue
        if not gdf.empty and gdf.geometry.name in gdf:
            return gdf
    return gpd.GeoDataFrame(columns=["zone", "geometry"], geometry="geometry", crs="EPSG:4326")


def _polygonize_cluster_raster(path: Path) -> gpd.GeoDataFrame:
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        mask = data != nodata if nodata is not None else ~pd.isna(data)
        records = []
        for geom, value in shapes(data.astype("int32"), mask=mask, transform=src.transform):
            if nodata is not None and int(value) == int(nodata):
                continue
            records.append({"zone": int(value), "geometry": shape(geom)})
        if not records:
            return gpd.GeoDataFrame(columns=["zone", "geometry"], geometry="geometry", crs=src.crs)
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=src.crs)
    return gdf.dissolve(by="zone").reset_index()


def _prepare_zone_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = gdf.copy()
    if "zone" not in work.columns:
        if "zone_id" in work.columns:
            work["zone"] = work["zone_id"]
        elif "cluster" in work.columns:
            work["zone"] = work["cluster"]
        else:
            work["zone"] = range(1, len(work) + 1)
    if work.crs is None:
        work = work.set_crs("EPSG:4326", allow_override=True)
    area_source = work
    if area_source.crs is not None and area_source.crs.is_geographic:
        try:
            area_source = area_source.to_crs(area_source.estimate_utm_crs())
        except Exception:
            area_source = area_source.to_crs("EPSG:3857")
    work["area_m2"] = area_source.geometry.area.astype(float)
    work = work.to_crs("EPSG:4326")
    work["zone"] = pd.to_numeric(work["zone"], errors="coerce").fillna(0).astype(int)
    keep = ["zone", "area_m2", "geometry"]
    if "zone_id" in work.columns:
        keep.insert(1, "zone_id")
    work = work[keep]
    try:
        work["geometry"] = work.geometry.simplify(0.000005, preserve_topology=True)
    except Exception:
        logger.debug("Geometry simplification failed for map payload", exc_info=True)
    return work


def _best_vr_metadata(cfg: dict, manifest: ProjectWorkspaceManifest) -> Dict[str, Any]:
    try:
        grid = artifact_by_id(manifest, "gridsearch")
        df = pd.read_csv(artifact_path(cfg, grid))
    except Exception:
        return {}
    if "vr" not in df.columns or df.empty:
        return {}
    row = df.sort_values("vr", ascending=False).iloc[0].to_dict()
    keys = ["algo", "k", "seed", "vr", "ch_score", "asc", "anova_p"]
    return {key: _json_safe(row.get(key)) for key in keys if key in row and pd.notna(row.get(key))}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
