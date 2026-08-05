"""Reusable output-manifest helpers independent of MZGPT services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _paths(value: Any):
    if isinstance(value, (str, Path)):
        yield Path(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _paths(child)


def write_artifact_manifest(run_directory: Path, result: Mapping[str, Any]) -> Path:
    """Write a compact manifest of files produced by a pipeline run."""
    run_directory = Path(run_directory)
    records = []
    seen = set()
    for candidate in _paths(result):
        if not candidate.exists() or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(run_directory.resolve()).as_posix()
        except ValueError:
            relative = str(resolved)
        records.append({"path": relative, "size_bytes": resolved.stat().st_size})
    manifest_path = run_directory / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": sorted(records, key=lambda item: item["path"])}, indent=2),
        encoding="utf-8",
    )
    return manifest_path
