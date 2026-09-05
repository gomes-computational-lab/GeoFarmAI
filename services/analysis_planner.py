from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field

from core.raster_pipeline import _build_gridsearch_candidates
from jobs.flow_mzd import mzd_flow_two_csvs
from jobs.run_experiments import _run_experiments
from services.project_visuals import resolve_output_dir
from services.workspace_manifest import refresh_workspace_manifest


SEEDED_ALGORITHMS = {"kmeans", "gmm", "fcm"}
SUPPORTED_ALGORITHMS = ["kmeans", "agglomerative", "gmm", "fcm"]
SUPPORTED_VALIDATION_METRICS = ["vr", "ch_score", "asc", "anova_p", "fragmentation"]
WORKLOAD_THRESHOLDS = {"quick": 10, "moderate": 40, "long": 100}
MAX_WORKER_COUNT = 4


class AnalysisSetupRequest(BaseModel):
    feature_methods: List[str]
    algorithms: List[str]
    k_values: List[int]
    seeds: List[int]
    algorithm_parameters: Dict[str, Any] = Field(default_factory=dict)
    validation_metrics: List[str] = Field(default_factory=list)
    force_recompute: bool = False
    parallel_gridsearch: bool = True
    n_jobs: Optional[int] = None
    generate_previews: bool = True
    generate_report: bool = True


class PlannedRun(BaseModel):
    feature_method: str
    algorithm: str
    k: int
    seed: Optional[int] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AnalysisPlan(BaseModel):
    planned_runs: List[PlannedRun]
    total_run_count: int
    run_counts_by_feature_method: Dict[str, int]
    run_counts_by_algorithm: Dict[str, int]
    workload_level: str
    estimated_duration_min_seconds: Optional[float] = None
    estimated_duration_max_seconds: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    configuration_summary: Dict[str, Any] = Field(default_factory=dict)


class AnalysisSetupOptions(BaseModel):
    project_name: str
    feature_methods: Dict[str, str]
    algorithms: List[str]
    k_values: List[int]
    seeds: List[int]
    validation_metrics: List[str]
    presets: Dict[str, AnalysisSetupRequest]


class AnalysisJobSubmitResponse(BaseModel):
    job_id: str
    status: str
    total_run_count: int
    resolved_config_path: str


class AnalysisJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    total_run_count: int
    completed_run_count: int = 0
    current_step: Optional[str] = None
    current_candidate: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    resolved_config_path: Optional[str] = None
    artifact_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class _JobRecord:
    def __init__(self, status: AnalysisJobStatus, request: AnalysisSetupRequest, plan: AnalysisPlan, cfg_path: str) -> None:
        self.status = status
        self.request = request
        self.plan = plan
        self.cfg_path = cfg_path
        self.logs: List[str] = []


_JOBS: Dict[str, _JobRecord] = {}
_LOCK = threading.Lock()
_PROJECT_LOCK = threading.Lock()


def setup_options(cfg: dict) -> AnalysisSetupOptions:
    clustering = cfg.get("clustering", {})
    algorithms = [algo for algo in clustering.get("algorithms", ["kmeans"]) if algo in SUPPORTED_ALGORITHMS]
    k_values = [int(k) for k in clustering.get("k_values", [2, 3, 4])]
    seeds = sorted({int(seed) for seed in clustering.get("seeds", [42])})
    feature_methods = discover_feature_methods(cfg)
    standard = AnalysisSetupRequest(
        feature_methods=["pca"] if "pca" in feature_methods else list(feature_methods)[:1],
        algorithms=algorithms,
        k_values=k_values,
        seeds=seeds,
        validation_metrics=["vr", "ch_score", "asc", "anova_p"],
        parallel_gridsearch=bool(cfg.get("raster", {}).get("parallel_gridsearch", False)),
        n_jobs=min(int(cfg.get("raster", {}).get("n_jobs", 1) or 1), MAX_WORKER_COUNT),
    )
    quick = AnalysisSetupRequest(
        feature_methods=standard.feature_methods[:1],
        algorithms=algorithms[:2] or ["kmeans"],
        k_values=k_values[:2] or [2],
        seeds=seeds[:1] or [42],
        validation_metrics=["vr", "asc"],
        parallel_gridsearch=False,
        n_jobs=1,
    )
    comprehensive = AnalysisSetupRequest(
        feature_methods=list(feature_methods),
        algorithms=algorithms,
        k_values=k_values,
        seeds=seeds,
        validation_metrics=SUPPORTED_VALIDATION_METRICS,
        parallel_gridsearch=True,
        n_jobs=min(os.cpu_count() or 1, MAX_WORKER_COUNT),
    )
    return AnalysisSetupOptions(
        project_name=str(cfg.get("project", {}).get("name", "project")),
        feature_methods=feature_methods,
        algorithms=algorithms,
        k_values=k_values,
        seeds=seeds,
        validation_metrics=SUPPORTED_VALIDATION_METRICS,
        presets={"quick": quick, "standard": standard, "comprehensive": comprehensive},
    )


def discover_feature_methods(cfg: dict) -> Dict[str, str]:
    methods = {}
    raster_cfg = cfg.get("raster", {})
    if cfg.get("raster", {}).get("enabled", False):
        methods["pca"] = "PCA/MULTISPATI feature stack using configured raster variables."
        methods["raw"] = "Raw interpolated feature stack without PCA reduction."
        experiments = cfg.get("experiments", [])
        seen_sets = {tuple(raster_cfg.get("pca_variables", []))}
        idx = 1
        for exp in experiments:
            params = exp.get("parameters", {})
            for variables in params.get("raster.pca_variables", []):
                key_tuple = tuple(variables) if isinstance(variables, list) else (str(variables),)
                if key_tuple in seen_sets:
                    continue
                seen_sets.add(key_tuple)
                idx += 1
                methods[f"pca_set_{idx}"] = f"PCA/MULTISPATI feature stack with variables: {', '.join(key_tuple)}."
    else:
        methods["spatial_pca"] = "Point-grid spatial PCA feature representation."
    return methods


def feature_method_labels(cfg: dict) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for key in discover_feature_methods(cfg):
        if key == "pca":
            labels[key] = "PCA / MULTISPATI components"
        elif key == "raw":
            labels[key] = "Original interpolated variables"
        elif key == "spatial_pca":
            labels[key] = "Spatial PCA components"
        elif key.startswith("pca_set_"):
            labels[key] = f"PCA / MULTISPATI variable set {key.removeprefix('pca_set_')}"
        else:
            labels[key] = key.replace("_", " ").title()
    return labels


def build_analysis_plan(request: AnalysisSetupRequest, cfg: dict) -> AnalysisPlan:
    _validate_request(request, cfg)
    planned_runs: List[PlannedRun] = []
    by_feature: Dict[str, int] = {method: 0 for method in request.feature_methods}
    by_algorithm: Dict[str, int] = {algo: 0 for algo in request.algorithms}
    candidates = _build_gridsearch_candidates(request.k_values, request.algorithms, sorted(set(request.seeds)))
    for feature_method in request.feature_methods:
        for candidate in candidates:
            params = request.algorithm_parameters.get(candidate["algo"], {})
            run = PlannedRun(
                feature_method=feature_method,
                algorithm=candidate["algo"],
                k=int(candidate["k"]),
                seed=candidate.get("seed"),
                parameters=params if isinstance(params, dict) else {"value": params},
            )
            planned_runs.append(run)
            by_feature[feature_method] += 1
            by_algorithm[run.algorithm] += 1

    warnings = _plan_warnings(request, planned_runs)
    workload = _workload_level(len(planned_runs), request)
    estimate = _duration_estimate(cfg, planned_runs, request)
    return AnalysisPlan(
        planned_runs=planned_runs,
        total_run_count=len(planned_runs),
        run_counts_by_feature_method=by_feature,
        run_counts_by_algorithm=by_algorithm,
        workload_level=workload,
        estimated_duration_min_seconds=estimate[0],
        estimated_duration_max_seconds=estimate[1],
        warnings=warnings,
        configuration_summary={
            "project": cfg.get("project", {}).get("name", "project"),
            "feature_methods": request.feature_methods,
            "feature_method_labels": feature_method_labels(cfg),
            "algorithms": request.algorithms,
            "k_values": request.k_values,
            "seeds": sorted(set(request.seeds)),
            "seed_independent_algorithms": [algo for algo in request.algorithms if algo not in SEEDED_ALGORITHMS],
            "validation_metrics": request.validation_metrics,
            "parallel_gridsearch": request.parallel_gridsearch,
            "n_jobs": request.n_jobs,
            "generate_previews": request.generate_previews,
            "generate_report": request.generate_report,
        },
    )


def submit_analysis_job(request: AnalysisSetupRequest, cfg: dict, cfg_path: str) -> AnalysisJobSubmitResponse:
    plan = build_analysis_plan(request, cfg)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    resolved_path = write_resolved_config(cfg, cfg_path, request, plan, run_id)
    status = AnalysisJobStatus(
        job_id=run_id,
        status="queued",
        total_run_count=plan.total_run_count,
        created_at=datetime.now(timezone.utc).isoformat(),
        resolved_config_path=str(resolved_path),
        warnings=plan.warnings,
    )
    record = _JobRecord(status, request, plan, str(resolved_path))
    with _LOCK:
        _JOBS[run_id] = record
    thread = threading.Thread(target=_run_job, args=(run_id,), daemon=True)
    thread.start()
    return AnalysisJobSubmitResponse(job_id=run_id, status="queued", total_run_count=plan.total_run_count, resolved_config_path=str(resolved_path))


def get_job_status(job_id: str) -> AnalysisJobStatus:
    with _LOCK:
        if job_id not in _JOBS:
            raise KeyError(job_id)
        record = _JOBS[job_id]
    _sync_status_from_pipeline_log(record)
    with _LOCK:
        return record.status


def get_job_logs(job_id: str) -> List[str]:
    with _LOCK:
        if job_id not in _JOBS:
            raise KeyError(job_id)
        record = _JOBS[job_id]
    _sync_status_from_pipeline_log(record)
    with _LOCK:
        return list(record.logs)


def write_resolved_config(cfg: dict, cfg_path: str, request: AnalysisSetupRequest, plan: AnalysisPlan, run_id: str) -> Path:
    out_dir = resolve_output_dir(cfg)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = copy.deepcopy(cfg)
    resolved.setdefault("clustering", {})
    resolved["clustering"]["algorithms"] = request.algorithms
    resolved["clustering"]["k_values"] = request.k_values
    resolved["clustering"]["seeds"] = sorted(set(request.seeds))
    resolved.setdefault("raster", {})
    resolved["raster"]["parallel_gridsearch"] = request.parallel_gridsearch
    if request.n_jobs is not None:
        resolved["raster"]["n_jobs"] = int(request.n_jobs)
    single_method = len(request.feature_methods) == 1
    if single_method:
        _apply_feature_method(resolved, request.feature_methods[0], cfg)
        resolved.pop("experiments", None)
    else:
        experiments = []
        for method in request.feature_methods:
            overrides = {
                "clustering": copy.deepcopy(resolved["clustering"]),
                "raster": {
                    "parallel_gridsearch": request.parallel_gridsearch,
                    "n_jobs": resolved["raster"].get("n_jobs", 1),
                },
            }
            _apply_feature_method(overrides, method, cfg)
            experiments.append({"name": f"analysis_setup_{method}", "overrides": overrides})
        resolved["experiments"] = experiments
    resolved["analysis_setup"] = {
        "run_id": run_id,
        "base_config": cfg_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "request": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
        "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
    }
    path = run_dir / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return path


def _run_job(job_id: str) -> None:
    with _PROJECT_LOCK:
        with _LOCK:
            record = _JOBS[job_id]
            record.status.status = "running"
            record.status.started_at = datetime.now(timezone.utc).isoformat()
            record.status.current_step = "preparing data"
            record.logs.append(f"{record.status.started_at} queued job started")
        start = time.monotonic()
        try:
            if len(record.request.feature_methods) > 1:
                _append_log(job_id, "running configured feature-method experiments")
                cfg = yaml.safe_load(Path(record.cfg_path).read_text(encoding="utf-8"))
                _run_experiments(cfg)
            else:
                _append_log(job_id, "running configured management-zone analysis")
                mzd_flow_two_csvs.fn(record.cfg_path, record.request.force_recompute, record.request.parallel_gridsearch, record.request.n_jobs)
            cfg = yaml.safe_load(Path(record.cfg_path).read_text(encoding="utf-8"))
            manifest = refresh_workspace_manifest(cfg)
            with _LOCK:
                record.status.status = "completed"
                record.status.completed_run_count = record.plan.total_run_count
                record.status.current_step = "completed"
                record.status.completed_at = datetime.now(timezone.utc).isoformat()
                record.status.artifact_count = len(manifest.artifacts)
                record.logs.append(f"{record.status.completed_at} completed in {time.monotonic() - start:.1f}s")
        except Exception as exc:
            with _LOCK:
                record.status.status = "failed"
                record.status.error = str(exc)
                record.status.current_step = "failed"
                record.status.completed_at = datetime.now(timezone.utc).isoformat()
                record.logs.append(f"{record.status.completed_at} failed: {exc}")


def _append_log(job_id: str, message: str) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].logs.append(f"{datetime.now(timezone.utc).isoformat()} {message}")


def _sync_status_from_pipeline_log(record: _JobRecord) -> None:
    """Infer progress from the latest pipeline log so the UI does not appear stuck."""

    if record.status.status not in {"queued", "running"}:
        return
    try:
        cfg = yaml.safe_load(Path(record.cfg_path).read_text(encoding="utf-8"))
        log_path = _latest_pipeline_log(cfg)
    except Exception:
        return
    if log_path is None:
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    completed = record.status.completed_run_count
    total = record.status.total_run_count
    current_step = record.status.current_step
    finished = False
    recent_pipeline_lines: List[str] = []
    for line in lines:
        if "[raster] Kriging" in line:
            current_step = "interpolation"
        elif "[raster] Building clustering feature matrix" in line:
            current_step = "feature preparation"
        elif "[raster] Running clustering grid search" in line:
            current_step = "clustering grid search"
        elif "[raster][gridsearch]" in line:
            current_step = "clustering grid search"
            match = re.search(r"\[raster\]\[gridsearch\]\s+(\d+)/(\d+)\s+finished", line)
            if match:
                completed = max(completed, int(match.group(1)))
                total = max(total, int(match.group(2)))
        elif "wrote comparison plot" in line or "Wrote run manifest" in line:
            current_step = "generating outputs"
        elif "[log] Finished recording run output" in line:
            finished = True
        if "[raster]" in line or "[log]" in line:
            recent_pipeline_lines.append(line)

    with _LOCK:
        record.status.completed_run_count = completed
        record.status.total_run_count = max(record.status.total_run_count, total)
        record.status.current_step = "completed" if finished and completed >= record.status.total_run_count else current_step
        existing = set(record.logs)
        for line in recent_pipeline_lines[-12:]:
            normalized = f"{log_path.name}: {line}"
            if normalized not in existing:
                record.logs.append(normalized)
        if finished and completed >= record.status.total_run_count:
            record.status.status = "completed"
            record.status.completed_run_count = record.status.total_run_count
            record.status.completed_at = record.status.completed_at or datetime.now(timezone.utc).isoformat()
            try:
                manifest = refresh_workspace_manifest(cfg)
                record.status.artifact_count = len(manifest.artifacts)
            except Exception:
                pass


def _latest_pipeline_log(cfg: dict) -> Optional[Path]:
    out_dir = resolve_output_dir(cfg)
    log_dir = out_dir / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def _validate_request(request: AnalysisSetupRequest, cfg: dict) -> None:
    options = setup_options(cfg)
    if not request.feature_methods:
        raise ValueError("Select at least one feature representation.")
    if not request.algorithms:
        raise ValueError("Select at least one clustering algorithm.")
    if not request.k_values:
        raise ValueError("Select at least one k value.")
    if not request.seeds:
        raise ValueError("Select at least one random seed.")
    bad_features = sorted(set(request.feature_methods) - set(options.feature_methods))
    bad_algorithms = sorted(set(request.algorithms) - set(options.algorithms))
    if bad_features:
        raise ValueError(f"Unsupported feature representation(s): {bad_features}")
    if bad_algorithms:
        raise ValueError(f"Unsupported clustering algorithm(s): {bad_algorithms}")
    if any(int(k) < 2 for k in request.k_values):
        raise ValueError("All k values must be at least 2.")
    if request.n_jobs is not None and int(request.n_jobs) > MAX_WORKER_COUNT:
        raise ValueError(f"Worker count cannot exceed {MAX_WORKER_COUNT}.")


def _plan_warnings(request: AnalysisSetupRequest, planned_runs: List[PlannedRun]) -> List[str]:
    warnings: List[str] = []
    if len(planned_runs) > WORKLOAD_THRESHOLDS["long"]:
        warnings.append(f"This plan includes {len(planned_runs)} candidate runs and may take a long time.")
    if request.force_recompute:
        warnings.append("Force recompute is selected, so cached intermediate results may be ignored.")
    if any(method.startswith("pca") for method in request.feature_methods):
        warnings.append("PCA/MULTISPATI feature preparation can add noticeable runtime.")
    if request.parallel_gridsearch and request.n_jobs and os.cpu_count() and request.n_jobs > os.cpu_count():
        warnings.append(f"Selected workers exceed detected CPU count ({os.cpu_count()}).")
    if "vr" not in request.validation_metrics and "anova_p" not in request.validation_metrics:
        warnings.append("No outcome-validation metric is selected; external comparison will be limited.")
    if len(set(request.seeds)) < 2 and any(algo in SEEDED_ALGORITHMS for algo in request.algorithms):
        warnings.append("Only one seed is selected; seed-stability analysis will be limited.")
    if request.generate_previews and request.generate_report and len(planned_runs) > WORKLOAD_THRESHOLDS["moderate"]:
        warnings.append("Preview and report generation are selected for a larger plan.")
    if os.cpu_count() is None:
        warnings.append("Available CPU count could not be detected.")
    return warnings


def _workload_level(count: int, request: AnalysisSetupRequest) -> str:
    adjusted = count + (10 if request.force_recompute else 0) + (5 * len(request.feature_methods))
    if adjusted <= WORKLOAD_THRESHOLDS["quick"]:
        return "Quick"
    if adjusted <= WORKLOAD_THRESHOLDS["moderate"]:
        return "Moderate"
    if adjusted <= WORKLOAD_THRESHOLDS["long"]:
        return "Long"
    return "Very Long"


def _duration_estimate(cfg: dict, planned_runs: List[PlannedRun], request: AnalysisSetupRequest) -> tuple[Optional[float], Optional[float]]:
    history_path = resolve_output_dir(cfg) / "runtime_history.jsonl"
    if not history_path.exists():
        return None, None
    values: List[float] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("runtime_seconds"):
            values.append(float(row["runtime_seconds"]))
    if not values:
        return None, None
    per_run = sum(values) / len(values)
    parallel_factor = max(1, int(request.n_jobs or 1)) if request.parallel_gridsearch else 1
    estimate = per_run * len(planned_runs) / parallel_factor
    return max(0.0, estimate * 0.75), estimate * 1.5


def _apply_feature_method(cfg: dict, method: str, base_cfg: dict) -> None:
    cfg.setdefault("raster", {})
    if method == "raw":
        cfg["raster"]["use_pca"] = False
    else:
        cfg["raster"]["use_pca"] = True
        if method.startswith("pca_set_"):
            variables = _pca_variables_for_method(method, base_cfg)
            if variables:
                cfg["raster"]["pca_variables"] = variables


def _pca_variables_for_method(method: str, base_cfg: dict) -> List[str]:
    if method == "pca":
        return list(base_cfg.get("raster", {}).get("pca_variables", []))
    options = discover_feature_methods(base_cfg)
    if method not in options:
        return []
    text = options[method].split("variables:", 1)
    if len(text) == 2:
        return [item.strip().strip(".") for item in text[1].split(",") if item.strip()]
    return []


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
