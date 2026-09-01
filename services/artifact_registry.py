from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from services.workspace_manifest import (
    ProjectWorkspaceManifest,
    classify_workspace_artifact,
    persist_workspace_manifest,
)
from services.project_visuals import resolve_output_dir, resolve_project_raster_dir


def register_existing_derived_artifact(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    path: Path,
    title: str,
    description: str,
    source_artifact_ids: List[str],
    operation: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Register a validated derived artifact already written under the derived directory."""

    out_dir = resolve_output_dir(cfg)
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    resolved = path.resolve()
    resolved.relative_to(derived_dir.resolve())
    artifact = classify_workspace_artifact(resolved, out_dir, manifest.project_id, None)
    artifact.id = f"derived_{resolved.stem}"
    artifact.semantic_role = artifact.semantic_role or "derived_artifact"
    artifact.title = title
    artifact.description = description
    artifact.related_artifact_ids = source_artifact_ids
    artifact.metadata.update({"operation": operation})
    if metadata:
        artifact.metadata.update(metadata)
    manifest.artifacts.append(artifact)
    persist_workspace_manifest(cfg, manifest)
    return artifact.id


def register_dataframe_as_derived_csv(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    df: pd.DataFrame,
    filename: str,
    title: str,
    description: str,
    source_artifact_ids: List[str],
    operation: str,
) -> str:
    """Write and register a derived CSV under the project derived directory."""

    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    path = derived_dir / filename
    if path.exists():
        raise FileExistsError(f"Derived artifact already exists: {path.name}")
    df.to_csv(path, index=False)
    return register_existing_derived_artifact(cfg, manifest, path, title, description, source_artifact_ids, operation)
