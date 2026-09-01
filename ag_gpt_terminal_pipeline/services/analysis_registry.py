from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from services.project_results import dataframe_records
from services.project_visuals import resolve_project_raster_dir
from services.workspace_manifest import (
    METRIC_DEFINITIONS,
    ProjectWorkspaceManifest,
    WorkspaceArtifact,
    artifact_by_id,
    artifact_path,
    classify_workspace_artifact,
    persist_workspace_manifest,
)

logger = logging.getLogger(__name__)


class ChartProvenance(BaseModel):
    chart_artifact_id: str
    data_artifact_id: Optional[str] = None
    source_artifact_ids: List[str] = Field(default_factory=list)
    title: str
    description: str
    chart_type: str
    x_column: str
    y_columns: List[str]
    group_column: Optional[str] = None
    sort_column: Optional[str] = None
    sort_ascending: Optional[bool] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    transformation: str
    metric_definitions: Dict[str, str] = Field(default_factory=lambda: dict(METRIC_DEFINITIONS))
    plotted_rows: List[Dict[str, Any]] = Field(default_factory=list)
    interpretation_summary: Optional[str] = None
    caveats: List[str] = Field(default_factory=list)
    interpretation_facts: Dict[str, Any] = Field(default_factory=dict)


class DerivedAnalysisBundle(BaseModel):
    analysis_id: str
    analysis_type: str
    source_artifact_ids: List[str]
    result_json: Dict[str, Any] = Field(default_factory=dict)
    table_artifact_ids: List[str] = Field(default_factory=list)
    chart_artifact_ids: List[str] = Field(default_factory=list)
    provenance_artifact_id: Optional[str] = None
    created_at: str


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def safe_analysis_id(label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._").lower()
    return stem or "analysis"


def register_derived_table(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    df: pd.DataFrame,
    output_name: str,
    title: str,
    description: str,
    source_artifact_ids: List[str],
    analysis_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = derived_dir / f"{safe_analysis_id(output_name)}_{stamp}.csv"
    df.to_csv(path, index=False)
    artifact = classify_workspace_artifact(path, Path(manifest.output_root), manifest.project_id, None)
    artifact.id = f"derived_{path.stem}"
    artifact.semantic_role = "derived_analysis_table"
    artifact.title = title
    artifact.description = description
    artifact.related_artifact_ids = list(dict.fromkeys(source_artifact_ids))
    artifact.metadata.update(
        {
            "analysis_type": analysis_type,
            "generated_by": "analysis_registry.register_derived_table",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if metadata:
        artifact.metadata.update(metadata)
    manifest.artifacts.append(artifact)
    persist_workspace_manifest(cfg, manifest)
    return artifact.id


def register_derived_chart(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    chart_path: Path,
    provenance: ChartProvenance,
    analysis_type: str,
) -> str:
    artifact = artifact_by_id(manifest, provenance.chart_artifact_id)
    artifact.semantic_role = "derived_analysis_chart"
    artifact.title = provenance.title
    artifact.description = provenance.description
    related = [*provenance.source_artifact_ids]
    if provenance.data_artifact_id:
        related.append(provenance.data_artifact_id)
    artifact.related_artifact_ids = list(dict.fromkeys([*artifact.related_artifact_ids, *related]))
    artifact.metadata.update(
        {
            "analysis_type": analysis_type,
            "chart_provenance": model_to_dict(provenance),
            "chart_type": provenance.chart_type,
            "x_column": provenance.x_column,
            "y_columns": provenance.y_columns,
            "group_column": provenance.group_column,
            "sort_column": provenance.sort_column,
            "sort_ascending": provenance.sort_ascending,
            "transformation": provenance.transformation,
        }
    )
    if provenance.data_artifact_id:
        try:
            data_artifact = artifact_by_id(manifest, provenance.data_artifact_id)
            data_artifact.semantic_role = "derived_analysis_table"
            data_artifact.related_artifact_ids = list(dict.fromkeys([*data_artifact.related_artifact_ids, artifact.id]))
            data_artifact.metadata.update(
                {
                    "analysis_type": analysis_type,
                    "chart_artifact_id": artifact.id,
                    "chart_provenance": model_to_dict(provenance),
                }
            )
        except KeyError:
            logger.exception(
                "Chart provenance referenced a missing derived data artifact",
                extra={
                    "operation": "register_derived_chart",
                    "chart_artifact_id": provenance.chart_artifact_id,
                    "data_artifact_id": provenance.data_artifact_id,
                },
            )
    persist_workspace_manifest(cfg, manifest)
    return artifact.id


def register_analysis_bundle(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    analysis_type: str,
    source_artifact_ids: List[str],
    result_json: Dict[str, Any],
    table_artifact_ids: List[str],
    chart_artifact_ids: List[str],
    provenance: Optional[ChartProvenance] = None,
) -> DerivedAnalysisBundle:
    analysis_id = f"analysis_{safe_analysis_id(analysis_type)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    bundle = DerivedAnalysisBundle(
        analysis_id=analysis_id,
        analysis_type=analysis_type,
        source_artifact_ids=source_artifact_ids,
        result_json=result_json,
        table_artifact_ids=table_artifact_ids,
        chart_artifact_ids=chart_artifact_ids,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    path = derived_dir / f"{analysis_id}.provenance.json"
    artifact = classify_workspace_artifact(path, Path(manifest.output_root), manifest.project_id, None)
    artifact.id = f"derived_{path.stem}"
    bundle.provenance_artifact_id = artifact.id
    payload = model_to_dict(bundle)
    if provenance is not None:
        payload["chart_provenance"] = model_to_dict(provenance)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    artifact.semantic_role = "derived_analysis_bundle"
    artifact.title = analysis_type.replace("_", " ").title()
    artifact.description = "Analysis bundle provenance and result metadata."
    artifact.related_artifact_ids = list(dict.fromkeys([*source_artifact_ids, *table_artifact_ids, *chart_artifact_ids]))
    artifact.metadata.update({"analysis_type": analysis_type, "bundle": payload})
    manifest.artifacts.append(artifact)
    persist_workspace_manifest(cfg, manifest)
    return bundle


def get_analysis_bundle(manifest: ProjectWorkspaceManifest, analysis_id: str) -> Optional[DerivedAnalysisBundle]:
    for artifact in manifest.artifacts:
        bundle = artifact.metadata.get("bundle") if artifact.metadata else None
        if isinstance(bundle, dict) and bundle.get("analysis_id") == analysis_id:
            return DerivedAnalysisBundle(**bundle)
    return None


def get_chart_provenance(manifest: ProjectWorkspaceManifest, chart_artifact_id: str) -> Optional[ChartProvenance]:
    try:
        artifact = artifact_by_id(manifest, chart_artifact_id)
    except KeyError:
        return None
    raw = artifact.metadata.get("chart_provenance") if artifact.metadata else None
    if isinstance(raw, dict):
        return ChartProvenance(**raw)
    return None


def resolve_related_data_artifact(manifest: ProjectWorkspaceManifest, chart_artifact_id: str) -> Optional[str]:
    provenance = get_chart_provenance(manifest, chart_artifact_id)
    if provenance and provenance.data_artifact_id:
        return provenance.data_artifact_id
    try:
        chart = artifact_by_id(manifest, chart_artifact_id)
    except KeyError:
        return None
    for related_id in chart.related_artifact_ids:
        try:
            related = artifact_by_id(manifest, related_id)
        except KeyError:
            continue
        if related.artifact_type == "table":
            return related.id
    return None


def chart_interpretation_facts(rows: List[Dict[str, Any]], y_column: str, x_column: str, sort_ascending: bool = True) -> Dict[str, Any]:
    numeric = [row for row in rows if row.get(y_column) is not None]
    if not numeric:
        return {}
    ordered = sorted(numeric, key=lambda row: float(row[y_column]), reverse=not sort_ascending)
    values = [float(row[y_column]) for row in numeric]
    return {
        "leading_row": ordered[0],
        "trailing_row": ordered[-1],
        "value_range": {"minimum": min(values), "maximum": max(values)},
        "x_values": [row.get(x_column) for row in ordered],
    }


def explain_existing_chart(cfg: dict, manifest: ProjectWorkspaceManifest, chart_artifact_id: str) -> Dict[str, Any]:
    provenance = get_chart_provenance(manifest, chart_artifact_id)
    if provenance is None:
        raise ValueError(f"No chart provenance found for artifact '{chart_artifact_id}'.")
    data_artifact_id = resolve_related_data_artifact(manifest, chart_artifact_id)
    rows = list(provenance.plotted_rows)
    if data_artifact_id:
        data_artifact = artifact_by_id(manifest, data_artifact_id)
        table_path = artifact_path(cfg, data_artifact)
        df = pd.read_csv(table_path)
        rows = dataframe_records(df)
    y_column = provenance.y_columns[0] if provenance.y_columns else ""
    facts = chart_interpretation_facts(rows, y_column, provenance.x_column, provenance.sort_ascending if provenance.sort_ascending is not None else True)
    return {
        "chart_artifact_id": chart_artifact_id,
        "data_artifact_id": data_artifact_id,
        "provenance": model_to_dict(provenance),
        "rows": rows,
        "facts": facts,
    }
