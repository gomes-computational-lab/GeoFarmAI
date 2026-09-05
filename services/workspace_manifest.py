from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from services.project_results import dataframe_records
from services.project_visuals import (
    VisualCatalogItem,
    build_visual_catalog,
    normalize_visual_text,
    resolve_output_dir,
    resolve_project_raster_dir,
)

logger = logging.getLogger(__name__)


class WorkspaceArtifact(BaseModel):
    id: str
    project_id: str
    name: str
    relative_path: str
    artifact_type: Literal["table", "image", "raster", "geopackage", "pdf", "text", "log", "json", "other"]
    semantic_role: Optional[str] = None
    title: str
    description: str
    format: str
    columns: List[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    metric: Optional[str] = None
    variable: Optional[str] = None
    component: Optional[str] = None
    algorithm: Optional[str] = None
    k: Optional[int] = None
    seed: Optional[int] = None
    related_artifact_ids: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectWorkspaceManifest(BaseModel):
    project_id: str
    project_name: str
    generated_at: str
    output_root: str
    artifacts: List[WorkspaceArtifact]
    metric_definitions: Dict[str, Any] = Field(default_factory=dict)
    variables: List[str] = Field(default_factory=list)
    algorithms: List[str] = Field(default_factory=list)
    k_values: List[int] = Field(default_factory=list)


class InspectWorkspaceInput(BaseModel):
    artifact_types: List[str] = Field(default_factory=list)
    semantic_roles: List[str] = Field(default_factory=list)
    metric: Optional[str] = None
    variable: Optional[str] = None
    text_query: Optional[str] = None
    limit: int = 50


class ReadTableInput(BaseModel):
    artifact_id: str
    columns: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort_by: Optional[str] = None
    ascending: bool = True
    limit: int = 100


class ProfileTableInput(BaseModel):
    artifact_id: str


class TableAnalysisSpec(BaseModel):
    artifact_id: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    group_by: List[str] = Field(default_factory=list)
    aggregations: Dict[str, List[str]] = Field(default_factory=dict)
    sort_by: Optional[str] = None
    ascending: bool = True
    limit: Optional[int] = None
    output_name: Optional[str] = None


class PlotSpec(BaseModel):
    artifact_id: str
    plot_type: Literal["bar", "line", "scatter", "box", "histogram"]
    x: str
    y: Optional[str] = None
    group_by: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregation: Optional[str] = None
    title: Optional[str] = None
    output_name: Optional[str] = None


ARTIFACT_SUFFIX_TYPES = {
    ".csv": "table",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "raster",
    ".tiff": "raster",
    ".gpkg": "geopackage",
    ".pdf": "pdf",
    ".txt": "text",
    ".log": "log",
    ".json": "json",
}

METRIC_DEFINITIONS = {
    "vr": "outcome variance reduction; higher is better",
    "ch_score": "Calinski-Harabasz score; higher is better",
    "asc": "average silhouette coefficient; higher is better",
    "anova_p": "outcome ANOVA p-value; lower is stronger evidence of outcome differences",
}


def workspace_manifest_path(cfg: dict) -> Path:
    return resolve_project_raster_dir(cfg) / "workspace_manifest.json"


def load_or_build_workspace_manifest(cfg: dict, refresh: bool = False) -> ProjectWorkspaceManifest:
    path = workspace_manifest_path(cfg)
    existing: Optional[ProjectWorkspaceManifest] = None
    if path.exists():
        try:
            existing = ProjectWorkspaceManifest(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            existing = None
    if not refresh and path.exists() and not _manifest_is_stale(path, resolve_output_dir(cfg)):
        if existing is not None:
            return existing
    manifest = build_workspace_manifest(cfg)
    if existing is not None:
        _merge_manifest_metadata(manifest, existing)
    persist_workspace_manifest(cfg, manifest)
    return manifest


def build_workspace_manifest(cfg: dict) -> ProjectWorkspaceManifest:
    out_dir = resolve_output_dir(cfg)
    project_name = cfg.get("project", {}).get("name", "project")
    artifacts: List[WorkspaceArtifact] = []
    visual_by_path = {item.path: item for item in build_visual_catalog(cfg)}
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file():
                continue
            if path.name == "workspace_manifest.json":
                continue
            artifact = classify_workspace_artifact(path, out_dir, project_name, visual_by_path.get(path.relative_to(out_dir).as_posix()))
            artifacts.append(artifact)
    _link_related_artifacts(artifacts)
    return ProjectWorkspaceManifest(
        project_id=project_name,
        project_name=project_name,
        generated_at=datetime.now(timezone.utc).isoformat(),
        output_root=str(out_dir),
        artifacts=artifacts,
        metric_definitions=METRIC_DEFINITIONS,
        variables=sorted({item.variable for item in artifacts if item.variable}),
        algorithms=_table_unique_values(artifacts, out_dir, "algo"),
        k_values=[int(value) for value in _table_unique_values(artifacts, out_dir, "k") if _is_int_like(value)],
    )


def persist_workspace_manifest(cfg: dict, manifest: ProjectWorkspaceManifest) -> Path:
    path = workspace_manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = manifest.model_dump() if hasattr(manifest, "model_dump") else manifest.dict()
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _merge_manifest_metadata(rebuilt: ProjectWorkspaceManifest, existing: ProjectWorkspaceManifest) -> None:
    """Preserve durable derived-analysis metadata when a manifest is rebuilt from files."""

    existing_by_id = {artifact.id: artifact for artifact in existing.artifacts}
    existing_by_path = {artifact.relative_path: artifact for artifact in existing.artifacts}
    rebuilt_keys = {(artifact.id, artifact.relative_path) for artifact in rebuilt.artifacts}
    for artifact in rebuilt.artifacts:
        old = existing_by_id.get(artifact.id) or existing_by_path.get(artifact.relative_path)
        if old is None:
            continue
        artifact.id = old.id
        artifact.semantic_role = old.semantic_role or artifact.semantic_role
        artifact.title = old.title or artifact.title
        artifact.description = old.description or artifact.description
        artifact.related_artifact_ids = list(dict.fromkeys([*artifact.related_artifact_ids, *old.related_artifact_ids]))
        artifact.aliases = sorted(set([*artifact.aliases, *old.aliases]))
        artifact.metadata.update(old.metadata)
    for old in existing.artifacts:
        key = (old.id, old.relative_path)
        if key not in rebuilt_keys and old.semantic_role == "derived_analysis_bundle":
            rebuilt.artifacts.append(old)


def classify_workspace_artifact(path: Path, output_root: Path, project_id: str, visual: Optional[VisualCatalogItem] = None) -> WorkspaceArtifact:
    relative = path.relative_to(output_root).as_posix()
    suffix = path.suffix.lower()
    artifact_type = ARTIFACT_SUFFIX_TYPES.get(suffix, "other")
    artifact_id = _artifact_id_for_path(path, output_root)
    title = path.stem.replace("_", " ").title()
    description = f"Generated {artifact_type} artifact."
    semantic_role = None
    metric = variable = component = None
    aliases: List[str] = [path.stem, path.name, title]
    columns: List[str] = []
    row_count = None
    metadata: Dict[str, Any] = {}

    if visual is not None:
        artifact_id = visual.id
        title = visual.title
        description = visual.description
        semantic_role = _semantic_role_for_visual(visual)
        metric = visual.metric
        variable = visual.variable
        component = visual.component
        aliases.extend(visual.aliases)
        metadata.update({"category": visual.category, "visual_type": visual.visual_type, "role": visual.role})
    elif artifact_type == "table":
        table_meta = _table_metadata(path)
        columns = table_meta["columns"]
        row_count = table_meta["row_count"]
        if path.name.endswith("_gridsearch.csv"):
            artifact_id = "gridsearch"
            semantic_role = "gridsearch_table"
            title = "Grid-search results table"
            description = "Candidate clustering runs and evaluation metrics."
            aliases.extend(["gridsearch", "grid search", "model performance", "results table"])
        elif path.name.endswith("_metrics.csv"):
            artifact_id = "metrics"
            semantic_role = "metrics_table"
            title = "Selected-run metrics table"
            description = "Metrics for the selected management-zone result."
    elif artifact_type == "raster":
        lower = path.name.lower()
        if lower == "best_clusters.tif":
            artifact_id = "best_clusters"
            semantic_role = "best_vr_cluster_raster"
            title = "Best cluster raster by outcome variance reduction"
            description = "GeoTIFF cluster raster selected using outcome variance reduction."
            metric = "vr"
        elif lower == "best_ch_score_clusters.tif":
            artifact_id = "best_ch_score_clusters"
            semantic_role = "best_ch_cluster_raster"
            title = "Best cluster raster by Calinski-Harabasz score"
            description = "GeoTIFF cluster raster selected using Calinski-Harabasz score."
            metric = "ch_score"
        elif "/rasters/" in f"/{relative}":
            variable = path.stem
            semantic_role = "interpolation_raster"
            title = f"{path.stem.replace('_', ' ')} raster"
            description = "Interpolated raster surface."
    elif artifact_type == "geopackage":
        semantic_role = "zone_package"
        title = "Management-zone GeoPackage"
        description = "Vector zone package exported by the pipeline."
    elif artifact_type == "pdf":
        semantic_role = "project_report"
        title = "Project PDF report"
        description = "PDF report generated from project preview images."
    elif artifact_type == "log":
        semantic_role = "execution_log"
        title = "Execution log"
        description = "Pipeline or chat execution log."
    return WorkspaceArtifact(
        id=artifact_id,
        project_id=project_id,
        name=path.name,
        relative_path=relative,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        semantic_role=semantic_role,
        title=title,
        description=description,
        format=suffix.lstrip(".") or "file",
        columns=columns,
        row_count=row_count,
        metric=metric,
        variable=variable,
        component=component,
        aliases=sorted(set(aliases)),
        metadata=metadata,
    )


def inspect_workspace(manifest: ProjectWorkspaceManifest, query: InspectWorkspaceInput) -> Dict[str, Any]:
    matches = manifest.artifacts
    if query.artifact_types:
        matches = [item for item in matches if item.artifact_type in query.artifact_types]
    if query.semantic_roles:
        matches = [item for item in matches if item.semantic_role in query.semantic_roles]
    if query.metric:
        matches = [item for item in matches if item.metric == query.metric]
    if query.variable:
        normalized = normalize_visual_text(query.variable)
        matches = [item for item in matches if normalize_visual_text(item.variable or "") == normalized or any(normalized == normalize_visual_text(alias) for alias in item.aliases)]
    if query.text_query:
        terms = set(normalize_visual_text(query.text_query).split())
        matches = [item for item in matches if terms.intersection(set(normalize_visual_text(" ".join([item.title, item.description, item.name, *item.aliases])).split()))]
    return {"project_id": manifest.project_id, "matches": [_artifact_summary(item) for item in matches[: query.limit]]}


def artifact_by_id(manifest: ProjectWorkspaceManifest, artifact_id: str) -> WorkspaceArtifact:
    for artifact in manifest.artifacts:
        if artifact.id == artifact_id:
            return artifact
    raise KeyError(f"Unknown workspace artifact id: {artifact_id}")


def artifact_path(cfg: dict, artifact: WorkspaceArtifact) -> Path:
    out_dir = resolve_output_dir(cfg)
    candidate = (out_dir / artifact.relative_path).resolve()
    candidate.relative_to(out_dir)
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Workspace artifact is missing: {artifact.id}")
    return candidate


def read_table_artifact(cfg: dict, manifest: ProjectWorkspaceManifest, request: ReadTableInput) -> Dict[str, Any]:
    artifact = artifact_by_id(manifest, request.artifact_id)
    if artifact.artifact_type != "table":
        raise ValueError(f"Artifact '{artifact.id}' is not a table.")
    df = pd.read_csv(artifact_path(cfg, artifact))
    df = _apply_table_filters(df, request.filters)
    if request.columns:
        missing = [col for col in request.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Unknown table columns for '{artifact.id}': {missing}")
        df = df[request.columns]
    if request.sort_by:
        if request.sort_by not in df.columns:
            raise ValueError(f"Unknown sort column '{request.sort_by}'.")
        df = df.sort_values(request.sort_by, ascending=request.ascending)
    limit = max(1, min(int(request.limit or 100), 1000))
    return {"artifact_id": artifact.id, "columns": df.columns.tolist(), "row_count": int(len(df)), "rows": dataframe_records(df.head(limit))}


def profile_table_artifact(cfg: dict, manifest: ProjectWorkspaceManifest, request: ProfileTableInput) -> Dict[str, Any]:
    artifact = artifact_by_id(manifest, request.artifact_id)
    if artifact.artifact_type != "table":
        raise ValueError(f"Artifact '{artifact.id}' is not a table.")
    df = pd.read_csv(artifact_path(cfg, artifact))
    numeric = df.select_dtypes(include="number")
    low_cardinality = {
        col: dataframe_records(pd.DataFrame({"value": sorted(df[col].dropna().unique().tolist())[:20]}))
        for col in df.columns
        if df[col].nunique(dropna=True) <= 20
    }
    return {
        "artifact_id": artifact.id,
        "row_count": int(len(df)),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {col: int(value) for col, value in df.isna().sum().items()},
        "numeric_summary": dataframe_records(numeric.describe().reset_index()) if not numeric.empty else [],
        "low_cardinality_values": low_cardinality,
        "suggested_grouping_columns": [col for col in ["algo", "algorithm", "k", "seed"] if col in df.columns],
        "suggested_metric_columns": [col for col in ["vr", "asc", "ch_score", "anova_p", "fpc"] if col in df.columns],
    }


def run_table_analysis(cfg: dict, manifest: ProjectWorkspaceManifest, spec: TableAnalysisSpec) -> Dict[str, Any]:
    artifact = artifact_by_id(manifest, spec.artifact_id)
    if artifact.artifact_type != "table":
        raise ValueError(f"Artifact '{artifact.id}' is not a table.")
    df = pd.read_csv(artifact_path(cfg, artifact))
    df = _apply_table_filters(df, spec.filters)
    for col in [*spec.group_by, *spec.aggregations.keys()]:
        if col not in df.columns:
            raise ValueError(f"Unknown analysis column '{col}'.")
    allowed_aggs = {"mean", "median", "std", "min", "max", "count", "first", "last"}
    agg_spec: Dict[str, List[str]] = {}
    for col, aggs in spec.aggregations.items():
        bad = [agg for agg in aggs if agg not in allowed_aggs]
        if bad:
            raise ValueError(f"Unsupported aggregations for '{col}': {bad}")
        agg_spec[col] = aggs
    if spec.group_by:
        result = df.groupby(spec.group_by, dropna=False).agg(agg_spec)
        result.columns = ["_".join(str(part) for part in col if part) for col in result.columns]
        result = result.reset_index()
    else:
        result = df.agg(agg_spec)
        result = result.T.reset_index().rename(columns={"index": "column"})
    if spec.sort_by:
        if spec.sort_by not in result.columns:
            raise ValueError(f"Unknown sort column '{spec.sort_by}'.")
        result = result.sort_values(spec.sort_by, ascending=spec.ascending)
    if spec.limit:
        result = result.head(max(1, int(spec.limit)))
    derived_id = None
    derived_path = None
    if spec.output_name:
        derived_id, derived_path = register_derived_table(cfg, manifest, result, spec.output_name, source_artifact_ids=[artifact.id], operation="RunTableAnalysis")
    return {
        "artifact_id": artifact.id,
        "columns": result.columns.tolist(),
        "rows": dataframe_records(result),
        "derived_artifact_id": derived_id,
        "derived_path": derived_path,
        "provenance": [{"artifact_id": artifact.id, "operation": "RunTableAnalysis", "filters": spec.filters, "group_by": spec.group_by, "aggregations": spec.aggregations}],
    }


def register_derived_table(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    df: pd.DataFrame,
    output_name: str,
    source_artifact_ids: List[str],
    operation: str,
) -> tuple[str, str]:
    out_dir = resolve_output_dir(cfg)
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_name).strip("_") or "derived_table"
    path = derived_dir / f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(path, index=False)
    artifact = classify_workspace_artifact(path, out_dir, manifest.project_id, None)
    artifact.id = f"derived_{path.stem}"
    artifact.semantic_role = "derived_table"
    artifact.title = output_name.replace("_", " ").title()
    artifact.description = f"Derived table generated by {operation}."
    artifact.related_artifact_ids = source_artifact_ids
    artifact.metadata.update({"operation": operation, "created_at": datetime.now(timezone.utc).isoformat()})
    manifest.artifacts.append(artifact)
    persist_workspace_manifest(cfg, manifest)
    return artifact.id, artifact.relative_path


def create_plot(cfg: dict, manifest: ProjectWorkspaceManifest, spec: PlotSpec) -> Dict[str, Any]:
    artifact = artifact_by_id(manifest, spec.artifact_id)
    if artifact.artifact_type != "table":
        raise ValueError(f"Artifact '{artifact.id}' is not a table.")
    df = pd.read_csv(artifact_path(cfg, artifact))
    df = _apply_table_filters(df, spec.filters)
    if spec.x not in df.columns:
        raise ValueError(f"Unknown plot x column '{spec.x}'.")
    if spec.y and spec.y not in df.columns:
        raise ValueError(f"Unknown plot y column '{spec.y}'.")
    plot_df = df.copy()
    if spec.aggregation:
        if not spec.y:
            raise ValueError("An aggregated plot requires a y column.")
        allowed = {"mean", "median", "std", "min", "max", "count"}
        if spec.aggregation not in allowed:
            raise ValueError(f"Unsupported plot aggregation '{spec.aggregation}'.")
        plot_df = plot_df.groupby(spec.x, as_index=False)[spec.y].agg(spec.aggregation)

    fig, ax = plt.subplots(figsize=(8, 5))
    title = spec.title or spec.output_name or "Derived plot"
    if spec.plot_type == "bar":
        if not spec.y:
            raise ValueError("A bar plot requires a y column.")
        ax.bar(plot_df[spec.x].astype(str), plot_df[spec.y])
    elif spec.plot_type == "line":
        if not spec.y:
            raise ValueError("A line plot requires a y column.")
        ax.plot(plot_df[spec.x], plot_df[spec.y], marker="o")
    elif spec.plot_type == "scatter":
        if not spec.y:
            raise ValueError("A scatter plot requires a y column.")
        ax.scatter(plot_df[spec.x], plot_df[spec.y])
    elif spec.plot_type == "box":
        if not spec.y:
            raise ValueError("A box plot requires a y column.")
        plot_df.boxplot(column=spec.y, by=spec.x, ax=ax)
        fig.suptitle("")
    elif spec.plot_type == "histogram":
        ax.hist(plot_df[spec.x].dropna())
    ax.set_title(title)
    ax.set_xlabel(spec.x)
    if spec.y:
        ax.set_ylabel(f"{spec.y}_{spec.aggregation}" if spec.aggregation else spec.y)
    fig.tight_layout()

    derived_id, relative = register_derived_figure(cfg, manifest, fig, spec.output_name or title, [artifact.id], "CreatePlot")
    plt.close(fig)
    data_artifact_id = None
    bundle_id = None
    chart_provenance: Dict[str, Any] = {}
    try:
        from services.analysis_registry import ChartProvenance, chart_interpretation_facts, model_to_dict, register_analysis_bundle, register_derived_chart
        from services.analysis_registry import register_derived_table as register_analysis_derived_table

        data_artifact_id = register_analysis_derived_table(
            cfg,
            manifest,
            plot_df,
            output_name=(spec.output_name or title) + "_plotted_data",
            title=f"{title} plotted data",
            description="Exact data rows used to generate the derived chart.",
            source_artifact_ids=[artifact.id],
            analysis_type="structured_plot",
            metadata={
                "plot_spec": spec.dict() if hasattr(spec, "dict") else {},
                "transformation": f"{spec.plot_type} plot from {artifact.id}",
            },
        )
        y_columns = [spec.y] if spec.y else [spec.x]
        rows = dataframe_records(plot_df)
        provenance = ChartProvenance(
            chart_artifact_id=derived_id,
            data_artifact_id=data_artifact_id,
            source_artifact_ids=[artifact.id],
            title=title,
            description=f"Derived {spec.plot_type} chart generated from {artifact.title}.",
            chart_type=spec.plot_type,
            x_column=spec.x,
            y_columns=[col for col in y_columns if col],
            group_column=spec.group_by,
            filters=spec.filters,
            transformation=f"{spec.plot_type} plot from workspace artifact '{artifact.id}' with aggregation '{spec.aggregation}'.",
            plotted_rows=rows,
            interpretation_facts=chart_interpretation_facts(rows, spec.y or spec.x, spec.x, True),
        )
        register_derived_chart(cfg, manifest, Path(relative), provenance, "structured_plot")
        chart_provenance = model_to_dict(provenance)
        bundle = register_analysis_bundle(
            cfg,
            manifest,
            analysis_type="structured_plot",
            source_artifact_ids=[artifact.id],
            result_json={"plot_spec": spec.dict() if hasattr(spec, "dict") else {}, "rows": rows},
            table_artifact_ids=[data_artifact_id],
            chart_artifact_ids=[derived_id],
            provenance=provenance,
        )
        bundle_id = bundle.analysis_id
    except Exception:
        logger.exception(
            "Failed to register chart provenance for derived plot",
            extra={
                "operation": "CreatePlot.register_chart_provenance",
                "artifact_id": artifact.id,
                "derived_artifact_id": derived_id,
                "plot_type": spec.plot_type,
                "x": spec.x,
                "y": spec.y,
            },
        )
        data_artifact_id = None
        bundle_id = None
    return {
        "derived_artifact_id": derived_id,
        "derived_table_artifact_id": data_artifact_id,
        "analysis_bundle_id": bundle_id,
        "derived_path": relative,
        "provenance": [
            {
                "artifact_id": artifact.id,
                "operation": "CreatePlot",
                "plot_spec": spec.dict() if hasattr(spec, "dict") else {},
                "chart_provenance": chart_provenance,
                "derived_table_artifact_id": data_artifact_id,
                "analysis_bundle_id": bundle_id,
            }
        ],
    }


def register_derived_figure(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    fig,
    output_name: str,
    source_artifact_ids: List[str],
    operation: str,
) -> tuple[str, str]:
    out_dir = resolve_output_dir(cfg)
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_name).strip("_") or "derived_figure"
    path = derived_dir / f"{stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    artifact = classify_workspace_artifact(path, out_dir, manifest.project_id, None)
    artifact.id = f"derived_{path.stem}"
    artifact.semantic_role = "derived_figure"
    artifact.title = output_name.replace("_", " ").title()
    artifact.description = f"Derived figure generated by {operation}."
    artifact.related_artifact_ids = source_artifact_ids
    artifact.metadata.update({"operation": operation, "created_at": datetime.now(timezone.utc).isoformat()})
    manifest.artifacts.append(artifact)
    persist_workspace_manifest(cfg, manifest)
    return artifact.id, artifact.relative_path


def refresh_workspace_manifest(cfg: dict) -> ProjectWorkspaceManifest:
    manifest = build_workspace_manifest(cfg)
    persist_workspace_manifest(cfg, manifest)
    return manifest


def _apply_table_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    for col, value in filters.items():
        if col not in result.columns:
            raise ValueError(f"Unknown filter column '{col}'.")
        if isinstance(value, list):
            result = result[result[col].isin(value)]
        else:
            result = result[result[col] == value]
    return result


def _table_metadata(path: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path, nrows=1000)
    except Exception:
        return {"columns": [], "row_count": None}
    try:
        with path.open("r", encoding="utf-8") as handle:
            row_count = sum(1 for _ in handle) - 1
    except Exception:
        row_count = len(df)
    return {"columns": df.columns.tolist(), "row_count": max(row_count, 0)}


def _artifact_id_for_path(path: Path, output_root: Path) -> str:
    relative = path.relative_to(output_root)
    stem = "_".join(relative.with_suffix("").parts)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def _semantic_role_for_visual(visual: VisualCatalogItem) -> Optional[str]:
    if visual.id == "clusters_best":
        return "best_vr_cluster_map_preview"
    if visual.id == "clusters_best_ch_score":
        return "best_ch_cluster_map_preview"
    if visual.metric and visual.visual_type == "comparison_figure":
        return f"{visual.metric}_comparison_figure"
    if visual.category == "component_map":
        return "pca_component_preview"
    if visual.category == "interpolation":
        return "interpolation_preview"
    return visual.role


def _link_related_artifacts(artifacts: List[WorkspaceArtifact]) -> None:
    by_id = {item.id: item for item in artifacts}
    pairs = [
        ("clusters_best", "best_clusters"),
        ("clusters_best_ch_score", "best_ch_score_clusters"),
    ]
    for left, right in pairs:
        if left in by_id and right in by_id:
            by_id[left].related_artifact_ids.append(right)
            by_id[right].related_artifact_ids.append(left)


def _artifact_summary(item: WorkspaceArtifact) -> Dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "artifact_type": item.artifact_type,
        "semantic_role": item.semantic_role,
        "metric": item.metric,
        "variable": item.variable,
        "component": item.component,
        "description": item.description,
    }


def _table_unique_values(artifacts: List[WorkspaceArtifact], output_root: Path, column: str) -> List[Any]:
    values = set()
    for artifact in artifacts:
        if artifact.artifact_type != "table" or column not in artifact.columns:
            continue
        try:
            df = pd.read_csv(output_root / artifact.relative_path, usecols=[column])
        except Exception:
            continue
        values.update(df[column].dropna().unique().tolist())
    return sorted(values, key=lambda item: str(item))


def _is_int_like(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _manifest_is_stale(manifest_path: Path, output_root: Path) -> bool:
    try:
        manifest_mtime = manifest_path.stat().st_mtime
    except OSError:
        return True
    for path in output_root.rglob("*"):
        if path.is_file() and path != manifest_path and path.stat().st_mtime > manifest_mtime:
            return True
    return False
