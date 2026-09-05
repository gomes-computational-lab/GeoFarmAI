"""Structured fitted results for GeoFarmAI's public scientific API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from geofarmai.data import HarmonizedFieldDataset
from geofarmai.exceptions import ModelConfigurationError


@dataclass(frozen=True, slots=True, eq=False)
class CandidateSolution:
    """One clustering candidate retained from the model search."""

    algorithm: str
    k: int
    labels: np.ndarray
    internal_metrics: Mapping[str, float]
    outcome_metrics: Mapping[str, Mapping[str, float | int | None]] = field(
        default_factory=dict
    )
    random_state: int | None = None

    @property
    def solution_id(self) -> str:
        seed = "none" if self.random_state is None else str(self.random_state)
        return f"{self.algorithm}:k={self.k}:random_state={seed}"


@dataclass(slots=True)
class GeoFarmResult:
    """Complete analysis output returned by :meth:`GeoFarmModel.fit`."""

    harmonized_data: HarmonizedFieldDataset
    selected_solution: CandidateSolution | None
    candidate_solutions: tuple[CandidateSolution, ...]
    predictor_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    component_scores: pd.DataFrame | None
    component_loadings: pd.DataFrame | None
    internal_metrics: Mapping[str, float]
    outcome_validation_metrics: Mapping[str, Mapping[str, float | int | None]]
    selection_metric: str | None
    selection_outcome: str | None
    selection_direction: str | None
    spatial_metadata: Mapping[str, Any]
    harmonization_provenance: Mapping[str, Any]
    decomposition_provenance: Mapping[str, Any]
    configuration: Mapping[str, Any]
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def zone_labels(self) -> np.ndarray | None:
        """Labels for the selected candidate solution."""

        if self.selected_solution is None:
            return None
        return self.selected_solution.labels.copy()

    @property
    def selected_candidate(self) -> CandidateSolution | None:
        """Explicit alias distinguishing selection from candidate generation."""

        return self.selected_solution

    @property
    def best_model(self) -> str | None:
        return None if self.selected_solution is None else self.selected_solution.algorithm

    @property
    def best_k(self) -> int | None:
        return None if self.selected_solution is None else self.selected_solution.k

    @property
    def best_labels(self) -> np.ndarray | None:
        """Compatibility alias for labels of the selected solution."""

        return self.zone_labels

    @property
    def metrics(self) -> dict[str, Any]:
        """Selected-solution metrics in one compatibility-friendly mapping."""

        if self.selected_solution is None:
            return {}
        return {
            "algorithm": self.best_model,
            "k": self.best_k,
            **dict(self.internal_metrics),
            "outcomes": {
                name: dict(values)
                for name, values in self.outcome_validation_metrics.items()
            },
        }

    @property
    def leaderboard(self) -> pd.DataFrame:
        """Return one metrics row per retained candidate solution."""

        rows: list[dict[str, Any]] = []
        for candidate in self.candidate_solutions:
            row: dict[str, Any] = {
                "solution_id": candidate.solution_id,
                "algorithm": candidate.algorithm,
                "k": candidate.k,
                "random_state": candidate.random_state,
                "selected": candidate is self.selected_solution,
                **dict(candidate.internal_metrics),
            }
            for outcome_name, metrics in candidate.outcome_metrics.items():
                for metric_name, value in metrics.items():
                    row[f"outcome__{outcome_name}__{metric_name}"] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, Any]:
        """Return a compact, serialization-friendly fitted-analysis summary."""

        return {
            "selection_metric": self.selection_metric,
            "selection_outcome": self.selection_outcome,
            "selection_direction": self.selection_direction,
            "selected_solution": (
                None
                if self.selected_solution is None
                else self.selected_solution.solution_id
            ),
            "algorithm": self.best_model,
            "k": self.best_k,
            "candidate_count": len(self.candidate_solutions),
            "predictors": list(self.predictor_names),
            "outcomes": list(self.outcome_names),
            "internal_metrics": dict(self.internal_metrics),
            "outcome_validation_metrics": {
                name: dict(metrics)
                for name, metrics in self.outcome_validation_metrics.items()
            },
            "decomposition": dict(self.decomposition_provenance),
            "crs": self.spatial_metadata.get("crs"),
            "harmonization_strategy": self.harmonized_data.actual_strategy,
        }

    def to_dataframe(
        self,
        solution: CandidateSolution | str | None = None,
    ) -> gpd.GeoDataFrame:
        """Return aligned values, components, geometry, and one solution's zones."""

        candidate = self._resolve_solution(solution)
        frame = gpd.GeoDataFrame(
            index=self.harmonized_data.predictors.index.copy(),
            geometry=gpd.GeoSeries(
                self.harmonized_data.geometry.array,
                index=self.harmonized_data.predictors.index,
                crs=self.harmonized_data.crs,
            ),
            crs=self.harmonized_data.crs,
        )
        display_names = self.harmonized_data.display_names
        for identity in self.harmonized_data.predictor_identities:
            frame[display_names[identity]] = self.harmonized_data.predictors[
                identity.tuple
            ].to_numpy()
        for identity in self.harmonized_data.outcome_identities:
            frame[display_names[identity]] = self.harmonized_data.outcomes[
                identity.tuple
            ].to_numpy()
        if self.component_scores is not None:
            for column in self.component_scores.columns:
                output_name = str(column)
                if output_name in frame.columns:
                    output_name = f"component:{output_name}"
                frame[output_name] = self.component_scores[column].to_numpy()
        if candidate is not None:
            frame["zone"] = candidate.labels
        return frame

    def export(
        self,
        path: str | Path,
        *,
        min_area: float = 0.0,
        solution: CandidateSolution | str | None = None,
    ) -> dict[str, Path]:
        """Export analysis support, candidate metrics, and zones when selected.

        A result created with ``selection=None`` exports its unlabelled analysis
        support and full candidate leaderboard. Pass ``solution=...`` to export
        zones for a specific retained candidate without changing result state.
        """

        from core.export import save_package, zones_from_support

        output = Path(path)
        if output.suffix.lower() != ".gpkg":
            output = output.with_suffix(".gpkg")
        output.parent.mkdir(parents=True, exist_ok=True)

        candidate = self._resolve_solution(solution)
        samples = self.to_dataframe(candidate)
        metrics_path = output.with_name(f"{output.stem}_metrics.csv")
        candidates_path = output.with_name(f"{output.stem}_candidates.csv")
        if candidate is None:
            samples.to_file(output, layer="samples", driver="GPKG")
            pd.DataFrame(
                [
                    {
                        "selection_metric": self.selection_metric,
                        "selection_outcome": self.selection_outcome,
                        "selected_solution": None,
                    }
                ]
            ).to_csv(metrics_path, index=False)
        else:
            zones = zones_from_support(samples, candidate.labels, min_area=min_area)
            selected_metrics: dict[str, Any] = {
                "algorithm": candidate.algorithm,
                "k": candidate.k,
                "random_state": candidate.random_state,
                "selection_metric": self.selection_metric,
                "selection_outcome": self.selection_outcome,
                **dict(candidate.internal_metrics),
            }
            for outcome_name, metrics in candidate.outcome_metrics.items():
                for metric_name, value in metrics.items():
                    selected_metrics[f"outcome__{outcome_name}__{metric_name}"] = value
            save_package(zones, samples, selected_metrics, str(output))

        self.leaderboard.to_csv(candidates_path, index=False)
        artifacts = {
            "geopackage": output,
            "metrics": metrics_path,
            "candidates": candidates_path,
        }
        self.artifacts.update(artifacts)
        return artifacts

    def _resolve_solution(
        self,
        solution: CandidateSolution | str | None,
    ) -> CandidateSolution | None:
        if solution is None:
            return self.selected_solution
        if isinstance(solution, CandidateSolution):
            if not any(solution is candidate for candidate in self.candidate_solutions):
                raise ModelConfigurationError(
                    "The requested CandidateSolution does not belong to this result."
                )
            return solution
        matches = [
            candidate
            for candidate in self.candidate_solutions
            if candidate.solution_id == solution
        ]
        if len(matches) != 1:
            raise ModelConfigurationError(f"Unknown candidate solution {solution!r}.")
        return matches[0]


__all__ = ["CandidateSolution", "GeoFarmResult"]
