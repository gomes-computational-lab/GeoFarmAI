"""Structured pipeline results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class GeoFarmResult:
    output_directory: Path
    metrics: dict[str, Any]
    leaderboard: pd.DataFrame
    best_model: str | None
    best_k: int | None
    best_labels: object | None
    artifacts: dict[str, Path] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a lightweight summary suitable for logs and callers."""
        return {
            "output_directory": str(self.output_directory),
            "metrics": self.metrics,
            "best_model": self.best_model,
            "best_k": self.best_k,
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
            "cache_hit": self.cache_hit,
        }
