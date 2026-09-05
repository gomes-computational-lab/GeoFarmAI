"""Sklearn-inspired orchestration for GeoFarmAI scientific components."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

from core.cluster import run_agglomerative, run_fcm, run_gmm, run_kmeans
from core.evaluate import anova_p, variance_reduction
from core.multispati import (
    multispati_components,
    pca_components,
    python_multispati_components,
    standardized_predictors,
)
from core.spatial import knn_weights
from geofarmai.data import (
    FieldDataset,
    HarmonizedFieldDataset,
    VariableIdentity,
    harmonize,
)
from geofarmai.exceptions import ModelConfigurationError, ModelNotFittedError
from geofarmai.result import CandidateSolution, GeoFarmResult


_SUPPORTED_DECOMPOSITIONS = {"none", "pca", "multispati"}
_SUPPORTED_CLUSTERING = {"kmeans", "gmm", "fcm", "agglomerative"}
_CLUSTERING_ALIASES = {"hierarchical": "agglomerative"}
_SUPPORTED_MULTISPATI_ENGINES = {"python", "r"}
_SELECTION_DIRECTIONS = {
    "variance_reduction": "maximize",
    "silhouette": "maximize",
    "calinski_harabasz": "maximize",
}


class GeoFarmModel:
    """Fit management-zone candidates on a canonical spatial field dataset.

    The class is an orchestration layer. Harmonization, decomposition,
    clustering, and outcome validation remain implemented by GeoFarmAI's
    existing scientific modules. ``selection`` is explicitly one of
    ``"variance_reduction"``, ``"silhouette"``,
    ``"calinski_harabasz"``, or ``None``. All supported criteria maximize;
    exact ties retain stable candidate generation order.
    """

    def __init__(
        self,
        *,
        decomposition: str = "pca",
        clustering: str | Iterable[str] = ("kmeans",),
        k: int | Iterable[int] = range(2, 6),
        n_components: int | None = None,
        random_state: int = 42,
        multispati_engine: str = "python",
        weights_k: int = 8,
        silhouette_sample_size: int | None = None,
        selection: str | None = "silhouette",
        selection_outcome: str | VariableIdentity | None = None,
        harmonization_strategy: str = "auto",
        reconciliation_method: str = "idw",
        harmonization_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.decomposition = decomposition
        self.clustering = clustering
        self.k = k
        self.n_components = n_components
        self.random_state = random_state
        self.multispati_engine = multispati_engine
        self.weights_k = weights_k
        self.silhouette_sample_size = silhouette_sample_size
        self.selection = selection
        self.selection_outcome = selection_outcome
        self.harmonization_strategy = harmonization_strategy
        self.reconciliation_method = reconciliation_method
        self.harmonization_params = harmonization_params
        self.result_: GeoFarmResult | None = None

    def fit(self, data: FieldDataset | HarmonizedFieldDataset) -> GeoFarmResult:
        """Fit all configured candidates and return their structured result."""

        decomposition = self._normalized_decomposition()
        algorithms = self._normalized_algorithms()
        k_values = self._normalized_k_values()
        selection = self._normalized_selection()
        harmonized = self._harmonized(data)
        predictors = harmonized.predictors.astype(float)
        self._validate_analysis_data(predictors, k_values)

        connectivity = None
        if decomposition == "multispati" or "agglomerative" in algorithms:
            connectivity = self._connectivity(harmonized)

        scores, loadings, decomposition_metadata = self._decompose(
            predictors,
            harmonized,
            decomposition,
            connectivity,
        )
        analysis_matrix = predictors if scores is None else scores
        reported_scores = None if decomposition == "none" else scores
        selection_outcome = self._resolve_selection_outcome(harmonized, selection)

        candidates: list[CandidateSolution] = []
        for candidate_k in k_values:
            for algorithm in algorithms:
                labels, metrics, seed = self._cluster(
                    analysis_matrix.to_numpy(dtype=float),
                    algorithm,
                    candidate_k,
                    connectivity,
                )
                external = self._outcome_metrics(harmonized, labels)
                candidates.append(
                    CandidateSolution(
                        algorithm=algorithm,
                        k=candidate_k,
                        labels=np.asarray(labels, dtype=int).copy(),
                        internal_metrics=dict(metrics),
                        outcome_metrics=external,
                        random_state=seed,
                    )
                )

        selected = self._select_candidate(candidates, selection, selection_outcome)
        predictor_names = harmonized.predictor_names
        outcome_names = harmonized.outcome_names
        configuration = self.get_params()
        configuration["clustering"] = algorithms
        configuration["k"] = k_values
        configuration["selection"] = selection
        configuration["selection_outcome_used"] = selection_outcome

        result = GeoFarmResult(
            harmonized_data=harmonized,
            selected_solution=selected,
            candidate_solutions=tuple(candidates),
            predictor_names=predictor_names,
            outcome_names=outcome_names,
            component_scores=None if reported_scores is None else reported_scores.copy(),
            component_loadings=None if loadings is None else loadings.copy(),
            internal_metrics=(
                {} if selected is None else dict(selected.internal_metrics)
            ),
            outcome_validation_metrics=(
                {}
                if selected is None
                else {
                    name: dict(metrics)
                    for name, metrics in selected.outcome_metrics.items()
                }
            ),
            selection_metric=selection,
            selection_outcome=selection_outcome,
            selection_direction=(
                None if selection is None else _SELECTION_DIRECTIONS[selection]
            ),
            spatial_metadata={
                "crs": harmonized.crs.to_string(),
                "observation_count": len(harmonized.geometry),
                "geometry_types": tuple(sorted(harmonized.geometry.geom_type.unique())),
                "weights_k": (
                    min(int(self.weights_k), len(harmonized.geometry) - 1)
                    if connectivity is not None
                    else None
                ),
            },
            harmonization_provenance={
                "variables": dict(harmonized.variable_provenance),
                "sources": dict(harmonized.source_provenance),
                "coverage": harmonized.coverage.copy(),
                "metadata": dict(harmonized.harmonization_metadata),
            },
            decomposition_provenance=decomposition_metadata,
            configuration=configuration,
        )

        self.harmonized_data_ = harmonized
        self.analysis_matrix_ = analysis_matrix.copy()
        self.component_scores_ = (
            None if reported_scores is None else reported_scores.copy()
        )
        self.component_loadings_ = None if loadings is None else loadings.copy()
        self.candidate_solutions_ = tuple(candidates)
        self.selected_solution_ = selected
        self.labels_ = None if selected is None else selected.labels.copy()
        self.n_features_in_ = predictors.shape[1]
        self.feature_names_in_ = np.asarray(predictor_names, dtype=object)
        self.result_ = result
        return result

    def fit_predict(self, data: FieldDataset | HarmonizedFieldDataset) -> np.ndarray:
        """Fit the analysis and return selected zone labels."""

        labels = self.fit(data).zone_labels
        if labels is None:
            raise ModelConfigurationError(
                "fit_predict() requires a selection policy; selection=None retains "
                "candidates without selecting zone labels."
            )
        return labels

    def get_result(self) -> GeoFarmResult:
        """Return the most recent result with a clear pre-fit error."""

        if self.result_ is None:
            raise ModelNotFittedError("GeoFarmModel has not been fitted.")
        return self.result_

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters using sklearn's familiar convention."""

        del deep
        return {
            "decomposition": self.decomposition,
            "clustering": self.clustering,
            "k": self.k,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "multispati_engine": self.multispati_engine,
            "weights_k": self.weights_k,
            "silhouette_sample_size": self.silhouette_sample_size,
            "selection": self.selection,
            "selection_outcome": self.selection_outcome,
            "harmonization_strategy": self.harmonization_strategy,
            "reconciliation_method": self.reconciliation_method,
            "harmonization_params": dict(self.harmonization_params or {}),
        }

    def _harmonized(
        self,
        data: FieldDataset | HarmonizedFieldDataset,
    ) -> HarmonizedFieldDataset:
        if isinstance(data, HarmonizedFieldDataset):
            return data
        if not isinstance(data, FieldDataset):
            raise ModelConfigurationError(
                "GeoFarmModel.fit() requires a FieldDataset or HarmonizedFieldDataset."
            )
        options = dict(self.harmonization_params or {})
        conflicting = sorted(
            key for key in ("strategy", "method") if key in options
        )
        if conflicting:
            raise ModelConfigurationError(
                "Use harmonization_strategy/reconciliation_method instead of duplicate "
                f"harmonization_params keys: {conflicting}."
            )
        return harmonize(
            data,
            strategy=self.harmonization_strategy,
            method=self.reconciliation_method,
            **options,
        )

    def _normalized_decomposition(self) -> str:
        decomposition = str(self.decomposition).lower().strip()
        if decomposition not in _SUPPORTED_DECOMPOSITIONS:
            raise ModelConfigurationError(
                f"Unsupported decomposition {self.decomposition!r}. Choose one of: "
                f"{', '.join(sorted(_SUPPORTED_DECOMPOSITIONS))}."
            )
        engine = str(self.multispati_engine).lower().strip()
        if engine not in _SUPPORTED_MULTISPATI_ENGINES:
            raise ModelConfigurationError(
                f"Unsupported MULTISPATI engine {self.multispati_engine!r}. "
                "Choose 'python' or 'r'."
            )
        return decomposition

    def _normalized_algorithms(self) -> tuple[str, ...]:
        raw = (self.clustering,) if isinstance(self.clustering, str) else tuple(self.clustering)
        if not raw:
            raise ModelConfigurationError("At least one clustering method is required.")
        algorithms = tuple(
            _CLUSTERING_ALIASES.get(str(name).lower().strip(), str(name).lower().strip())
            for name in raw
        )
        unsupported = sorted(set(algorithms) - _SUPPORTED_CLUSTERING)
        if unsupported:
            raise ModelConfigurationError(
                f"Unsupported clustering methods: {unsupported}. Choose from: "
                f"{', '.join(sorted(_SUPPORTED_CLUSTERING))}."
            )
        if len(set(algorithms)) != len(algorithms):
            raise ModelConfigurationError("Clustering methods must not be duplicated.")
        return algorithms

    def _normalized_k_values(self) -> tuple[int, ...]:
        raw = (self.k,) if isinstance(self.k, Integral) else tuple(self.k)
        if not raw:
            raise ModelConfigurationError("k must contain at least one candidate value.")
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in raw):
            raise ModelConfigurationError("Every candidate k must be an integer.")
        values = tuple(int(value) for value in raw)
        if any(value < 2 for value in values):
            raise ModelConfigurationError("Every candidate k must be at least 2.")
        if len(set(values)) != len(values):
            raise ModelConfigurationError("Candidate k values must not be duplicated.")
        return values

    def _normalized_selection(self) -> str | None:
        if self.selection is None:
            return None
        selection = str(self.selection).lower().strip()
        if selection not in _SELECTION_DIRECTIONS:
            raise ModelConfigurationError(
                f"Unsupported selection metric {self.selection!r}. Choose one of: "
                "variance_reduction, silhouette, calinski_harabasz, or None."
            )
        return selection

    def _validate_analysis_data(
        self,
        predictors: pd.DataFrame,
        k_values: tuple[int, ...],
    ) -> None:
        values = predictors.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelConfigurationError(
                "The harmonized predictor matrix contains missing or non-finite values. "
                "GeoFarmModel does not silently impute or drop observations; adjust the "
                "harmonization coverage or provide complete predictors."
            )
        if max(k_values) >= len(predictors):
            raise ModelConfigurationError(
                f"Every candidate k must be smaller than the {len(predictors)} observations."
            )
        if not isinstance(self.random_state, Integral) or isinstance(self.random_state, bool):
            raise ModelConfigurationError("random_state must be an integer.")
        if not isinstance(self.weights_k, Integral) or isinstance(self.weights_k, bool):
            raise ModelConfigurationError("weights_k must be an integer.")
        if self.weights_k < 1:
            raise ModelConfigurationError("weights_k must be at least 1.")
        if self.silhouette_sample_size is not None and self.silhouette_sample_size < 2:
            raise ModelConfigurationError("silhouette_sample_size must be at least 2.")

    def _connectivity(self, harmonized: HarmonizedFieldDataset):
        neighbors = min(int(self.weights_k), len(harmonized.geometry) - 1)
        return kneighbors_graph(
            harmonized.coordinate_array(),
            n_neighbors=neighbors,
            mode="connectivity",
            include_self=False,
        ).tocsr()

    def _decompose(
        self,
        predictors: pd.DataFrame,
        harmonized: HarmonizedFieldDataset,
        decomposition: str,
        connectivity,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
        if decomposition == "none":
            standardized, _ = standardized_predictors(predictors)
            return standardized, None, {
                "requested_method": "none",
                "actual_method": "none",
                "engine": None,
                "used_r": False,
                "fallback_occurred": False,
                "standardized": True,
            }

        maximum = min(predictors.shape)
        components = min(3, maximum) if self.n_components is None else self.n_components
        if not isinstance(components, Integral) or isinstance(components, bool):
            raise ModelConfigurationError("n_components must be an integer or None.")
        components = int(components)
        if components < 1 or components > maximum:
            raise ModelConfigurationError(
                f"n_components must be between 1 and {maximum} for this dataset."
            )

        if decomposition == "pca":
            scores, fitted, _ = pca_components(predictors, n_components=components)
            loadings = self._loadings(fitted, scores.columns, harmonized.predictor_names)
            return scores, loadings, {
                "requested_method": "pca",
                "actual_method": "pca",
                "engine": "scikit-learn",
                "used_r": False,
                "fallback_occurred": False,
                "standardized": True,
                "explained_variance_ratio": tuple(
                    float(value) for value in fitted.explained_variance_ratio_
                ),
            }

        engine = str(self.multispati_engine).lower().strip()
        if engine == "python":
            scores, fitted, _ = python_multispati_components(
                predictors,
                connectivity,
                n_components=components,
                random_state=int(self.random_state),
            )
            loadings = self._loadings(fitted, scores.columns, harmonized.predictor_names)
            return scores, loadings, {
                "requested_method": "multispati",
                "actual_method": "multispati",
                "engine": "multispaeti",
                "used_r": False,
                "fallback_occurred": False,
                "standardized": True,
            }

        support = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(harmonized.geometry.array, crs=harmonized.crs),
            crs=harmonized.crs,
        )
        weights = knn_weights(
            support,
            k=min(int(self.weights_k), len(support) - 1),
        )
        scores, used_r = multispati_components(
            predictors,
            weights,
            n_components=components,
            use_r=True,
        )
        scores.index = predictors.index
        return scores, None, {
            "requested_method": "multispati",
            "actual_method": "multispati",
            "engine": "r",
            "used_r": bool(used_r),
            "fallback_occurred": False,
            "standardized": True,
        }

    @staticmethod
    def _loadings(fitted, component_names, predictor_names) -> pd.DataFrame | None:
        components = getattr(fitted, "components_", None)
        if components is None:
            return None
        return pd.DataFrame(components, index=component_names, columns=predictor_names)

    def _cluster(self, X, algorithm, candidate_k, connectivity):
        sample_size = self.silhouette_sample_size
        if algorithm == "kmeans":
            labels, metrics = run_kmeans(
                X,
                candidate_k,
                random_state=int(self.random_state),
                sample_size=sample_size,
            )
            return labels, metrics, int(self.random_state)
        if algorithm == "gmm":
            labels, metrics = run_gmm(
                X,
                candidate_k,
                random_state=int(self.random_state),
                sample_size=sample_size,
            )
            return labels, metrics, int(self.random_state)
        if algorithm == "fcm":
            labels, metrics = run_fcm(
                X,
                candidate_k,
                random_state=int(self.random_state),
                sample_size=sample_size,
            )
            return labels, metrics, int(self.random_state)
        labels, metrics = run_agglomerative(
            X,
            candidate_k,
            connectivity=connectivity.maximum(connectivity.T),
            sample_size=sample_size,
        )
        return labels, metrics, None

    @staticmethod
    def _outcome_metrics(
        harmonized: HarmonizedFieldDataset,
        labels: np.ndarray,
    ) -> dict[str, dict[str, float | int | None]]:
        display_names = harmonized.display_names
        metrics: dict[str, dict[str, float | int | None]] = {}
        for identity in harmonized.outcome_identities:
            values = harmonized.outcomes[identity.tuple].to_numpy(dtype=float)
            valid = np.isfinite(values)
            valid_labels = np.asarray(labels)[valid]
            group_count = np.unique(valid_labels).size
            if valid.sum() < 2 or group_count < 2 or valid.sum() <= group_count:
                metrics[display_names[identity]] = {
                    "variance_reduction": None,
                    "anova_p": None,
                    "n_observations": int(valid.sum()),
                    "coverage_fraction": float(valid.mean()),
                }
                continue
            variance = variance_reduction(values[valid], valid_labels)
            p_value = anova_p(values[valid], valid_labels)
            metrics[display_names[identity]] = {
                "variance_reduction": (
                    float(variance) if np.isfinite(variance) else None
                ),
                "anova_p": float(p_value) if np.isfinite(p_value) else None,
                "n_observations": int(valid.sum()),
                "coverage_fraction": float(valid.mean()),
            }
        return metrics

    def _resolve_selection_outcome(
        self,
        harmonized: HarmonizedFieldDataset,
        selection: str | None,
    ) -> str | None:
        identities = harmonized.outcome_identities
        if selection != "variance_reduction":
            if self.selection_outcome is not None:
                raise ModelConfigurationError(
                    "selection_outcome is only valid when "
                    "selection='variance_reduction'."
                )
            return None

        if not identities:
            raise ModelConfigurationError(
                "selection='variance_reduction' requires at least one explicitly "
                "declared outcome variable."
            )
        names = harmonized.display_names
        if self.selection_outcome is None:
            if len(identities) == 1:
                return names[identities[0]]
            raise ModelConfigurationError(
                "selection='variance_reduction' with multiple outcomes requires an "
                "explicit selection_outcome."
            )

        if isinstance(self.selection_outcome, VariableIdentity):
            matches = [identity for identity in identities if identity == self.selection_outcome]
        else:
            requested = str(self.selection_outcome)
            matches = [
                identity
                for identity in identities
                if requested in {names[identity], identity.variable_name}
            ]
        if len(matches) != 1:
            raise ModelConfigurationError(
                f"selection_outcome {self.selection_outcome!r} must identify exactly one "
                "harmonized outcome."
            )
        return names[matches[0]]

    @staticmethod
    def _select_candidate(
        candidates: list[CandidateSolution],
        selection: str | None,
        outcome_name: str | None,
    ) -> CandidateSolution | None:
        """Select by one requested metric, preserving generation order for ties."""

        if selection is None:
            return None
        direction = _SELECTION_DIRECTIONS[selection]
        selected: CandidateSolution | None = None
        selected_value: float | None = None
        for candidate in candidates:
            value = _candidate_metric(candidate, selection, outcome_name)
            if value is None:
                continue
            if selected is None:
                selected = candidate
                selected_value = value
                continue
            is_better = (
                value > selected_value
                if direction == "maximize"
                else value < selected_value
            )
            if is_better:
                selected = candidate
                selected_value = value
        if selected is None:
            detail = (
                f" for outcome {outcome_name!r}"
                if selection == "variance_reduction"
                else ""
            )
            raise ModelConfigurationError(
                f"Selection metric {selection!r}{detail} is unavailable for every "
                "candidate; no different metric was substituted."
            )
        return selected


def _candidate_metric(
    candidate: CandidateSolution,
    selection: str,
    outcome_name: str | None,
) -> float | None:
    if selection == "silhouette":
        value = candidate.internal_metrics.get("asc")
    elif selection == "calinski_harabasz":
        value = candidate.internal_metrics.get("ch_score")
    else:
        value = candidate.outcome_metrics.get(outcome_name or "", {}).get(
            "variance_reduction"
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


__all__ = ["GeoFarmModel"]
