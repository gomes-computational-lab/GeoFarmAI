from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from services.artifact_registry import register_existing_derived_artifact
from services.project_visuals import resolve_project_raster_dir
from services.workspace_manifest import (
    ProjectWorkspaceManifest,
    artifact_by_id,
    artifact_path,
    load_or_build_workspace_manifest,
)


TRUTHY = {"1", "true", "yes", "on"}
APPROVED_IMPORTS = {
    "collections",
    "itertools",
    "json",
    "math",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "pathlib",
    "re",
    "scipy",
    "statistics",
}
DANGEROUS_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "ftplib",
    "http",
    "importlib",
    "marshal",
    "multiprocessing",
    "os",
    "paramiko",
    "pickle",
    "requests",
    "resource",
    "shelve",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "urllib",
    "webbrowser",
}
DANGEROUS_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
}
ALLOWED_OUTPUT_EXTENSIONS = {".csv", ".json", ".png", ".txt"}
ALLOWED_INPUT_ARTIFACT_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


class PythonAnalysisRequest(BaseModel):
    code: str
    input_artifact_ids: List[str]
    requested_outputs: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None
    question: Optional[str] = None


class PythonAnalysisResult(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    generated_files: List[str] = Field(default_factory=list)
    generated_artifact_ids: List[str] = Field(default_factory=list)
    result_json: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    disabled: bool = False
    timed_out: bool = False


class SandboxSettings(BaseModel):
    enabled: bool
    timeout_seconds: int
    max_output_mb: float
    max_files: int
    max_stdout_chars: int
    max_code_chars: int
    implemented: bool = True
    isolation_mode: str = "subprocess"
    network_isolation_guaranteed: bool = False


def sandbox_settings() -> SandboxSettings:
    return SandboxSettings(
        enabled=os.getenv("AGGPT_ENABLE_PYTHON_ANALYSIS", "false").lower() in TRUTHY,
        timeout_seconds=_env_int("AGGPT_ANALYSIS_TIMEOUT_SECONDS", 60, 1, 600),
        max_output_mb=_env_float("AGGPT_ANALYSIS_MAX_OUTPUT_MB", 25.0, 0.01, 1024.0),
        max_files=_env_int("AGGPT_ANALYSIS_MAX_FILES", 10, 1, 100),
        max_stdout_chars=_env_int("AGGPT_ANALYSIS_MAX_STDOUT_CHARS", 20_000, 100, 1_000_000),
        max_code_chars=_env_int("AGGPT_ANALYSIS_MAX_CODE_CHARS", 20_000, 100, 1_000_000),
    )


def run_python_analysis(
    request: PythonAnalysisRequest,
    cfg: Optional[dict] = None,
    manifest: Optional[ProjectWorkspaceManifest] = None,
    question: str = "",
) -> PythonAnalysisResult:
    """Run controlled Python analysis in a temporary subprocess sandbox."""

    settings = sandbox_settings()
    if not settings.enabled:
        return PythonAnalysisResult(
            success=False,
            disabled=True,
            stderr="RunPythonAnalysis is disabled. Set AGGPT_ENABLE_PYTHON_ANALYSIS=true to enable the subprocess sandbox.",
        )
    if cfg is None:
        return PythonAnalysisResult(success=False, stderr="RunPythonAnalysis requires a project configuration.")
    if len(request.code) > settings.max_code_chars:
        return PythonAnalysisResult(success=False, stderr=f"Submitted code exceeds AGGPT_ANALYSIS_MAX_CODE_CHARS={settings.max_code_chars}.")
    validation_error = validate_analysis_code(request.code)
    if validation_error:
        return PythonAnalysisResult(success=False, stderr=validation_error)
    try:
        manifest = manifest or load_or_build_workspace_manifest(cfg)
        input_mapping, source_paths = _prepare_input_mapping(cfg, manifest, request.input_artifact_ids)
    except Exception as exc:
        return PythonAnalysisResult(success=False, stderr=str(exc))

    timeout = _effective_timeout(request.timeout_seconds, settings.timeout_seconds)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aggpt_analysis_") as tmp:
        tmp_dir = Path(tmp).resolve()
        input_dir = tmp_dir / "inputs"
        output_dir = tmp_dir / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()
        local_inputs = _copy_inputs(input_mapping, source_paths, input_dir)
        (tmp_dir / "inputs.json").write_text(json.dumps(local_inputs, indent=2), encoding="utf-8")
        script_path = tmp_dir / "analysis_wrapper.py"
        script_path.write_text(_wrapper_source(request.code), encoding="utf-8")
        env = _sandbox_env(tmp_dir)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=str(tmp_dir),
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                shell=False,
                preexec_fn=_resource_limiter(settings) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            stdout = _truncate(_sanitize_output(exc.stdout or "", tmp_dir), settings.max_stdout_chars)
            stderr = _truncate(_sanitize_output(exc.stderr or "", tmp_dir), settings.max_stdout_chars)
            timeout_msg = f"RunPythonAnalysis timed out after {timeout} seconds."
            return PythonAnalysisResult(success=False, stdout=stdout, stderr=(stderr + "\n" + timeout_msg).strip(), duration_seconds=duration, timed_out=True)

        duration = time.monotonic() - start
        stdout = _truncate(_sanitize_output(completed.stdout, tmp_dir), settings.max_stdout_chars)
        stderr = _truncate(_sanitize_output(completed.stderr, tmp_dir), settings.max_stdout_chars)
        result_json = _read_result_json(tmp_dir / "result.json")
        if completed.returncode != 0:
            if completed.returncode < 0:
                signal_msg = f"Sandbox process was terminated by signal {-completed.returncode}; likely resource or timeout enforcement."
                return PythonAnalysisResult(
                    success=False,
                    stdout=stdout,
                    stderr=(stderr + "\n" + signal_msg).strip(),
                    result_json=result_json,
                    duration_seconds=duration,
                    timed_out=True,
                )
            return PythonAnalysisResult(success=False, stdout=stdout, stderr=stderr or f"Sandbox process exited with code {completed.returncode}.", result_json=result_json, duration_seconds=duration)

        try:
            generated_paths = _validate_generated_outputs(output_dir, settings)
            generated_files, generated_artifact_ids = _register_generated_outputs(
                cfg,
                manifest,
                generated_paths,
                request.input_artifact_ids,
                question or request.question or "",
            )
        except Exception as exc:
            return PythonAnalysisResult(success=False, stdout=stdout, stderr=str(exc), result_json=result_json, duration_seconds=duration)

        return PythonAnalysisResult(
            success=True,
            stdout=stdout,
            stderr=stderr,
            generated_files=generated_files,
            generated_artifact_ids=generated_artifact_ids,
            result_json=result_json,
            duration_seconds=duration,
        )


def validate_analysis_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Python syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            error = _validate_import_node(node)
            if error:
                return error
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DANGEROUS_CALLS:
                return f"Blocked dangerous call: {name}()."
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return f"Blocked dunder attribute access: {node.attr}."
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return f"Blocked dunder name access: {node.id}."
    return None


def _validate_import_node(node: ast.Import | ast.ImportFrom) -> Optional[str]:
    modules: List[str] = []
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    elif node.module:
        modules = [node.module]
    for module in modules:
        root = module.split(".", 1)[0]
        if root in DANGEROUS_IMPORT_ROOTS:
            return f"Blocked dangerous import: {module}."
        if module not in APPROVED_IMPORTS and root not in APPROVED_IMPORTS:
            return f"Import is not approved for RunPythonAnalysis: {module}."
    return None


def _call_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _prepare_input_mapping(cfg: dict, manifest: ProjectWorkspaceManifest, artifact_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, Path]]:
    if not artifact_ids:
        raise ValueError("RunPythonAnalysis requires at least one input artifact ID.")
    mapping: Dict[str, str] = {}
    sources: Dict[str, Path] = {}
    for artifact_id in artifact_ids:
        _validate_artifact_id(artifact_id)
        artifact = artifact_by_id(manifest, artifact_id)
        if artifact.artifact_type in {"other"}:
            raise ValueError(f"Artifact '{artifact_id}' is not approved as a sandbox input.")
        source = artifact_path(cfg, artifact)
        suffix = source.suffix.lower()
        if suffix in {".py", ".sh", ".zip"}:
            raise ValueError(f"Artifact '{artifact_id}' has a blocked input extension: {suffix}")
        safe_name = _safe_filename(artifact_id, suffix)
        mapping[artifact_id] = f"inputs/{safe_name}"
        sources[artifact_id] = source
    return mapping, sources


def _validate_artifact_id(artifact_id: str) -> None:
    if not artifact_id or not ALLOWED_INPUT_ARTIFACT_PATTERN.match(artifact_id):
        raise ValueError(f"Invalid artifact ID for sandbox input: {artifact_id!r}")
    if Path(artifact_id).is_absolute() or ".." in Path(artifact_id).parts:
        raise ValueError(f"Path-like artifact IDs are not allowed: {artifact_id!r}")


def _safe_filename(artifact_id: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact_id).strip("._") or "artifact"
    return f"{stem}{suffix}"


def _copy_inputs(mapping: Dict[str, str], sources: Dict[str, Path], input_dir: Path) -> Dict[str, str]:
    local: Dict[str, str] = {}
    for artifact_id, relative in mapping.items():
        target = input_dir / Path(relative).name
        shutil.copy2(sources[artifact_id], target)
        local[artifact_id] = relative
    return local


def _wrapper_source(user_code: str) -> str:
    indented = textwrap.indent(user_code, "    ")
    return f"""
import builtins
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

SANDBOX_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = str((SANDBOX_ROOT / "outputs").resolve())
RESULT_JSON_PATH = SANDBOX_ROOT / "result.json"
INPUTS = json.loads((SANDBOX_ROOT / "inputs.json").read_text(encoding="utf-8"))
INPUTS = {{key: str((SANDBOX_ROOT / value).resolve()) for key, value in INPUTS.items()}}

_real_open = builtins.open
_real_io_open = io.open

def _safe_path(path, mode="r"):
    resolved = Path(path).resolve()
    allowed_roots = [SANDBOX_ROOT / "inputs", SANDBOX_ROOT / "outputs", SANDBOX_ROOT / ".matplotlib"]
    write_mode = any(flag in str(mode) for flag in ["w", "a", "x", "+"])
    if not write_mode:
        allowed_roots.append(Path(sys.prefix).resolve())
    if not any(root.resolve() == resolved or root.resolve() in resolved.parents for root in allowed_roots):
        raise PermissionError(f"Sandbox file access denied: {{resolved}}")
    return resolved

def open(path, *args, **kwargs):  # noqa: A001
    if isinstance(path, int):
        return _real_open(path, *args, **kwargs)
    mode = args[0] if args else kwargs.get("mode", "r")
    return _real_open(_safe_path(path, mode), *args, **kwargs)

def _safe_io_open(path, *args, **kwargs):
    if isinstance(path, int):
        return _real_io_open(path, *args, **kwargs)
    mode = args[0] if args else kwargs.get("mode", "r")
    return _real_io_open(_safe_path(path, mode), *args, **kwargs)

builtins.open = open
io.open = _safe_io_open

RESULT = None

try:
{indented}
    if "RESULT" in globals() and RESULT is not None:
        with _real_open(RESULT_JSON_PATH, "w", encoding="utf-8") as _result_handle:
            _result_handle.write(json.dumps(RESULT, default=str))
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
"""


def _sandbox_env(tmp_dir: Path) -> Dict[str, str]:
    keep = {"CONDA_PREFIX", "PATH", "SYSTEMROOT", "WINDIR"}
    env = {key: value for key, value in os.environ.items() if key in keep and value}
    env.update(
        {
            "HOME": str(tmp_dir),
            "TMPDIR": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
            "MPLCONFIGDIR": str(tmp_dir / ".matplotlib"),
            "MPLBACKEND": "Agg",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _resource_limiter(settings: SandboxSettings):
    def _limit() -> None:
        try:
            import resource

            cpu_seconds = max(1, int(settings.timeout_seconds) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            max_bytes = int(settings.max_output_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))
        except Exception:
            pass

    return _limit


def _validate_generated_outputs(output_dir: Path, settings: SandboxSettings) -> List[Path]:
    if not output_dir.exists():
        return []
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() or path.is_symlink())
    if len(files) > settings.max_files:
        raise ValueError(f"Sandbox generated {len(files)} file(s), exceeding AGGPT_ANALYSIS_MAX_FILES={settings.max_files}.")
    max_bytes = int(settings.max_output_mb * 1024 * 1024)
    total = 0
    valid: List[Path] = []
    for path in files:
        resolved = path.resolve()
        try:
            resolved.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Sandbox output escaped the output directory: {path.name}") from exc
        if path.is_symlink():
            raise ValueError(f"Sandbox output symlinks are not allowed: {path.name}")
        if path.suffix.lower() not in ALLOWED_OUTPUT_EXTENSIONS:
            raise ValueError(f"Sandbox output extension is not allowed: {path.name}")
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"Sandbox output file exceeds size limit: {path.name}")
        total += size
        if total > max_bytes:
            raise ValueError("Sandbox generated files exceed the combined output size limit.")
        valid.append(path)
    return valid


def _register_generated_outputs(
    cfg: dict,
    manifest: ProjectWorkspaceManifest,
    generated_paths: List[Path],
    source_artifact_ids: List[str],
    question: str,
) -> Tuple[List[str], List[str]]:
    derived_dir = resolve_project_raster_dir(cfg) / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []
    generated_artifact_ids: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for index, source in enumerate(generated_paths, start=1):
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("_") or "python_analysis_output"
        target = derived_dir / f"{stem}_{timestamp}_{index}{source.suffix.lower()}"
        if target.exists():
            raise FileExistsError(f"Derived artifact already exists: {target.name}")
        shutil.copy2(source, target)
        artifact_id = register_existing_derived_artifact(
            cfg,
            manifest,
            target,
            title=stem.replace("_", " ").title(),
            description="Derived artifact generated by RunPythonAnalysis.",
            source_artifact_ids=source_artifact_ids,
            operation="RunPythonAnalysis",
            metadata={
                "generated_by": "RunPythonAnalysis",
                "source_artifact_ids": source_artifact_ids,
                "question": question,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "python_sandbox",
            },
        )
        generated_files.append(target.relative_to(resolve_project_raster_dir(cfg).parent).as_posix())
        generated_artifact_ids.append(artifact_id)
    return generated_files, generated_artifact_ids


def _read_result_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else {"value": data}


def _effective_timeout(request_timeout: Optional[int], default_timeout: int) -> int:
    if request_timeout is None:
        return default_timeout
    try:
        return max(1, min(int(request_timeout), default_timeout))
    except (TypeError, ValueError):
        return default_timeout


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(int(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(float(os.getenv(name, str(default))), maximum))
    except (TypeError, ValueError):
        return default


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]"


def _sanitize_output(value: str, tmp_dir: Path) -> str:
    return value.replace(str(tmp_dir), "<sandbox>")
