from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


METRIC_DIRECTIONS: Dict[str, bool] = {
    "vr": False,
    "ch_score": False,
    "asc": False,
    "anova_p": True,
}

METRIC_OBJECTIVES: Dict[str, str] = {
    "vr": "Best yield variance reduction",
    "ch_score": "Best feature-space separation",
    "asc": "Best average silhouette coefficient",
    "anova_p": "Strongest yield-difference evidence",
}


class GridsearchResolutionError(RuntimeError):
    """Raised when a grid-search CSV cannot be resolved unambiguously."""


def resolve_gridsearch_csv(cfg: dict) -> Optional[Path]:
    """Resolve the project grid-search CSV using project-scoped candidates only.

    Preference order:
    1. Exact project experiment grid-search path.
    2. Exact project grid-search path.
    3. A single project-name-matching grid-search file.
    4. Ambiguous candidates raise GridsearchResolutionError.
    """

    out_dir = Path(cfg.get("export", {}).get("out_dir", "outputs"))
    project_name = cfg.get("project", {}).get("name", "project")
    exact_candidates = [
        out_dir / f"{project_name}_experiments_gridsearch.csv",
        out_dir / f"{project_name}_gridsearch.csv",
        out_dir / f"{project_name}_raster" / f"{project_name}_gridsearch.csv",
    ]
    for path in exact_candidates:
        if path.exists():
            return path

    matching = sorted(out_dir.rglob(f"{project_name}*_gridsearch.csv")) if out_dir.exists() else []
    matching = [path for path in matching if path.is_file()]
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        candidates = ", ".join(str(path) for path in matching)
        raise GridsearchResolutionError(f"Multiple project grid-search CSV files match '{project_name}': {candidates}")
    return None


def read_gridsearch(path: Path) -> pd.DataFrame:
    """Read a grid-search CSV and normalize common numeric columns."""

    df = pd.read_csv(path)
    for col in ["k", "seed", "vr", "asc", "ch_score", "anova_p", "fpc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def best_rows_by_metric(df: pd.DataFrame, source_filename: str = "") -> List[Dict[str, Any]]:
    """Return one best row for each available supported metric."""

    rows: List[Dict[str, Any]] = []
    for metric, ascending in METRIC_DIRECTIONS.items():
        if metric not in df.columns:
            continue
        top = top_rows_by_metric(df, metric, top_n=1, ascending=ascending)
        if top.empty:
            continue
        row = top.iloc[0]
        rows.append(_project_result_row(row, metric, source_filename))
    return rows


def top_rows_by_metric(
    df: pd.DataFrame,
    metric: str,
    top_n: int = 5,
    ascending: Optional[bool] = None,
) -> pd.DataFrame:
    """Return top rows by a metric using configured default direction."""

    if metric not in df.columns:
        return pd.DataFrame()
    direction = METRIC_DIRECTIONS.get(metric, False) if ascending is None else ascending
    ranked = df.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=[metric])
    if ranked.empty:
        return ranked
    return ranked.sort_values(metric, ascending=direction).head(top_n)


def compare_cluster_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize best metric values for each cluster count."""

    if "k" not in df.columns:
        return pd.DataFrame()
    grouped = df.copy()
    grouped["k"] = pd.to_numeric(grouped["k"], errors="coerce")
    grouped = grouped.dropna(subset=["k"])
    agg: Dict[str, str] = {}
    for metric, ascending in METRIC_DIRECTIONS.items():
        if metric in grouped.columns:
            agg[metric] = "min" if ascending else "max"
    if not agg:
        return pd.DataFrame()
    return grouped.groupby("k", as_index=False).agg(agg).sort_values("k")


def compare_algorithms(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize best metric values for each clustering algorithm."""

    algo_col = "algo" if "algo" in df.columns else "algorithm" if "algorithm" in df.columns else None
    if algo_col is None:
        return pd.DataFrame()
    agg: Dict[str, str] = {}
    for metric, ascending in METRIC_DIRECTIONS.items():
        if metric in df.columns:
            agg[metric] = "min" if ascending else "max"
    if not agg:
        return pd.DataFrame()
    return df.groupby(algo_col, as_index=False).agg(agg).sort_values(algo_col)


def filter_gridsearch(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Filter a grid-search DataFrame by exact column values."""

    result = df.copy()
    for key, value in filters.items():
        if key not in result.columns or value is None:
            continue
        if isinstance(value, list):
            result = result[result[key].isin(value)]
        else:
            result = result[result[key] == value]
    return result


def _project_result_row(row: pd.Series, metric: str, source_filename: str = "") -> Dict[str, Any]:
    value = row.get(metric)
    result = {
        "objective": METRIC_OBJECTIVES.get(metric, f"Best {metric}"),
        "metric": metric,
        "algorithm": row.get("algorithm", row.get("algo")),
        "algo": row.get("algo", row.get("algorithm")),
        "k": row.get("k"),
        "seed": row.get("seed"),
        "value": value,
        "vr": row.get("vr"),
        "asc": row.get("asc"),
        "ch_score": row.get("ch_score"),
        "anova_p": row.get("anova_p"),
        "source_filename": source_filename,
    }
    return {key: _json_safe(value) for key, value in result.items() if pd.notna(value)}


def dataframe_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return JSON-safe DataFrame records."""

    return [{key: _json_safe(value) for key, value in row.items() if pd.notna(value)} for row in df.to_dict("records")]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            logger.debug("Failed to convert scalar value to JSON-safe Python type", exc_info=True)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value
