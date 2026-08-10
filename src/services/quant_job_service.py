# -*- coding: utf-8 -*-
"""Local quant workflow job runner.

This service exposes a constrained, auditable wrapper around the durable
``quant_workflow/scripts`` entry points.  It deliberately does not accept
arbitrary shell strings from the web UI.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import logging
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.services.quant_workflow_paths import (
    dsa_root,
    mlruns_root,
    projects_root,
    quant_data_root,
    reports_root,
    workflow_root,
)


logger = logging.getLogger(__name__)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DSA_ROOT = dsa_root()
PROJECTS_ROOT = projects_root(DSA_ROOT)
QUANT_WORKFLOW_ROOT = workflow_root(DSA_ROOT)
REPORTS_ROOT = reports_root(DSA_ROOT)
QUANT_DATA_ROOT = quant_data_root(DSA_ROOT)
MLRUNS_ROOT = mlruns_root(DSA_ROOT)
DEFAULT_JOBS_ROOT = REPORTS_ROOT / "jobs"
DSA_VENV_PYTHON = DSA_ROOT / ".venv" / "bin" / "python"
QUANT_WORKFLOW_PYTHON = _env_path("QUANT_WORKFLOW_PYTHON", DSA_VENV_PYTHON)
_LEGACY_QLIB_PYTHON_WARNING_EMITTED = False
_MISSING_QUANT_WORKFLOW_PYTHON_WARNING_EMITTED = False
CHILD_ENV_EXACT_NAMES = {
    "BAOSTOCK_LOGIN_TIMEOUT",
    "CHINA_ASTOCK_COMPARISON_TIMEOUT_SECONDS",
    "CHINA_ASTOCK_PYTHON",
    "CHINA_ASTOCK_QUANT_ROOT",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "NO_COLOR",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "QLIB_CLEANED_PROVIDER_URI",
    "QLIB_CLEANED_COMPATIBILITY_PROVIDER_URI",
    "QLIB_COMPATIBILITY_PROVIDER_URI",
    "QLIB_PROVIDER_ROOT",
    "QLIB_PROVIDER_URI",
    "QUANT_DATA_ROOT",
    "QUANT_STOCK_LIST_PATH",
    "QUANT_WORKFLOW_MLRUNS_DIR",
    "QUANT_WORKFLOW_PYTHON",
    "QUANT_WORKFLOW_REPORTS_ROOT",
    "QUANT_WORKFLOW_ROOT",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
    "WORKING_PROJECTS_ROOT",
}
CHILD_ENV_PREFIXES = ("BAOSTOCK_", "QLIB_")

JOB_TERMINATE_TIMEOUT_SECONDS = 5
RUNNER_IDENTITY_HANDSHAKE_TIMEOUT_SECONDS = 5.0
RUNNER_IDENTITY_HANDSHAKE_POLL_SECONDS = 0.02
RUNNER_IDENTITY_STARTUP_LEASE_SECONDS = 30.0
RECOVERED_RUNNER_POLL_INTERVAL_SECONDS = 1.0
RECOVERED_RUNNER_MAX_POLL_SECONDS = 48 * 60 * 60
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
EXIT_STATUS_FILENAME = "exit_status.json"
RUNNER_IDENTITY_FILENAME = "runner_identity.json"
START_PERMISSION_FILENAME = "start_permission.json"
ARTIFACT_BASELINE_FILENAME = "artifact_baseline.json"
PROGRESS_FILE_MAX_BYTES = 512_000
JQDATA_WEB_DRY_RUN_DEFAULT_LIMIT = 3
JQDATA_WEB_DRY_RUN_MAX_LIMIT = 20
GENERIC_WEB_LIMIT_MAX = 10000
QLIB_PREDICTION_WEB_TOPK_MAX = 500
QLIB_RUN_ID_MAX_LENGTH = 256
QLIB_RESEARCH_WEB_MAX_RUNS = 50
QLIB_RESEARCH_WEB_MAX_WINDOWS = 60
QLIB_RESEARCH_WEB_MAX_TRAIN_YEARS = 20
QLIB_RESEARCH_WEB_MAX_MONTHS = 120
CHINA_ASTOCK_INDICATOR_DEFAULT_LIMIT = 50
CHINA_ASTOCK_INDICATOR_MAX_LIMIT = 500
CHINA_ASTOCK_SYMBOL_MAX_COUNT = 100
CHINA_ASTOCK_SYMBOL_MAX_LENGTH = 20
BAOSTOCK_WEB_MAX_RETRIES = 20
BAOSTOCK_WEB_MAX_CONSECUTIVE_FAILURES = 1000
BAOSTOCK_WEB_MAX_OVERLAP_DAYS = 365
ASHARE_SOURCE_MIN_SLEEP_SECONDS = 2.0
ASHARE_SOURCE_REQUIRED_RETRIES = 1
ASHARE_SOURCE_MAX_CONSECUTIVE_FAILURES = 3
ASHARE_SOURCE_MAX_REQUESTS = 6000
ASHARE_SOURCE_MAX_OVERLAP_DAYS = 7
ASHARE_SOURCE_JOB_TYPES = {
    "download_baostock",
    "download_eastmoney",
    "download_akshare",
    "download_ashare_news",
    "run_daily_update",
}
QUANT_DATA_NETWORK_JOB_TYPES = ASHARE_SOURCE_JOB_TYPES | {
    "download_global_daily",
    "download_jqdata",
    "validate_latest_snapshot",
}
DATA_QUALITY_WEB_MAX_SAMPLE = 100000
DSA_BACKTEST_WEB_MAX_EVAL_WINDOW_DAYS = 3650
MARKET_REPORT_WEB_MAX_LOOKBACK_DAYS = 5000
MARKET_REPORT_WEB_MAX_MIN_SYMBOLS = 10000
OPTIMIZE_STRATEGY_WEB_MAX_PHASE = 4
FORMAL_TRAINING_JOB_TYPES = {"train_baseline_lgb", "run_candidate_pipeline"}
PROVIDER_PINNED_TRAINING_JOB_TYPES = FORMAL_TRAINING_JOB_TYPES | {
    "train_baseline_lgb_equivalence"
}
ROBUST_SIGNAL_DATA_PINNED_JOB_TYPES = {
    "run_robust_signal_research",
    "resume_robust_signal_research",
}
TRAINING_PROVIDER_DATASET_IDS = {
    "canonical": "ashare_daily_qfq",
    "cleaned": "ashare_daily_qfq",
}
PROVIDER_IDENTITY_FIELDS = (
    "provider_contract",
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
)
DIRECT_PROVIDER_IDENTITY_FIELDS = (
    "provider_contract",
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
    "descriptor_path",
    "descriptor_sha256",
    "cleaning_mode",
    "index_dataset_id",
    "index_snapshot_id",
    "index_content_sha256",
    "index_manifest_sha256",
)
MARKET_DATA_PINNED_JOB_TYPES = {
    "report_market_regime",
    "report_cash_control_backtest",
}
ASHARE_DAILY_QFQ_DATASET_ID = "ashare_daily_qfq"
ASHARE_DAILY_QFQ_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_qfq")
MARKET_DATA_IDENTITY_FIELDS = (
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
    "state_fingerprint",
)
CONFIG_FILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DURABLE_RUNNER_CODE = r"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback

command = json.loads(os.environ["QUANT_JOB_COMMAND_JSON"])
cwd = os.environ.get("QUANT_JOB_CWD") or None
exit_path = os.environ["QUANT_JOB_EXIT_STATUS_PATH"]
identity_path = os.environ["QUANT_JOB_RUNNER_IDENTITY_PATH"]
permission_path = os.environ.get("QUANT_JOB_START_PERMISSION_PATH") or ""
launch_token = os.environ["QUANT_JOB_LAUNCH_TOKEN"]
try:
    startup_deadline_epoch = float(os.environ["QUANT_JOB_STARTUP_DEADLINE_EPOCH"])
except (KeyError, TypeError, ValueError):
    startup_deadline_epoch = float("nan")

def write_exit_status(payload):
    tmp_path = exit_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, exit_path)

startup_expired_error = (
    "Durable runner startup lease expired; the real command was not started."
)
if not math.isfinite(startup_deadline_epoch) or time.time() >= startup_deadline_epoch:
    print(startup_expired_error, file=sys.stderr)
    write_exit_status({"return_code": 75, "error": startup_expired_error})
    sys.exit(75)

command_sha256 = hashlib.sha256(
    json.dumps(command, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()
identity_payload = {
    "schema_version": 1,
    "launch_token": launch_token,
    "pid": os.getpid(),
    "job_command_sha256": command_sha256,
}
identity_tmp_path = identity_path + ".tmp"
with open(identity_tmp_path, "w", encoding="utf-8") as handle:
    json.dump(identity_payload, handle, ensure_ascii=False, sort_keys=True)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(identity_tmp_path, identity_path)

permission_error = (
    "Durable runner start permission was not granted before the startup lease "
    "expired; the real command was not started."
)
permission_granted = False
while time.time() < startup_deadline_epoch:
    try:
        if permission_path and os.path.getsize(permission_path) <= 4096:
            with open(permission_path, "r", encoding="utf-8") as handle:
                permission = json.load(handle)
            permission_granted = (
                isinstance(permission, dict)
                and permission.get("schema_version") == 1
                and permission.get("launch_token") == launch_token
                and permission.get("job_command_sha256") == command_sha256
                and permission.get("pid") == os.getpid()
            )
            if permission_granted:
                break
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        pass
    time.sleep(0.01)

if not permission_granted:
    print(permission_error, file=sys.stderr)
    write_exit_status({"return_code": 75, "error": permission_error})
    sys.exit(75)

return_code = 1
error = None
try:
    if time.time() >= startup_deadline_epoch:
        raise RuntimeError(startup_expired_error)
    process = subprocess.Popen(command, cwd=cwd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    return_code = process.wait()
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    payload = {"return_code": return_code}
    if error:
        payload["error"] = error
    try:
        write_exit_status(payload)
    except BaseException:
        traceback.print_exc()
sys.exit(return_code if isinstance(return_code, int) and return_code >= 0 else 1)
"""
NUMERIC_PARAM_RULES = {
    "limit": ("integer", 1),
    "sleep": ("number", 0),
    "retries": ("integer", 0),
    "timeout": ("number", 1),
    "max_consecutive_failures": ("integer", 1),
    "max_requests": ("integer", 1),
    "overlap_days": ("integer", 0),
    "snapshot_limit": ("integer", 1),
    "news_limit": ("integer", 1),
    "overlap_sample": ("integer", 1),
    "merged_sample": ("integer", 1),
    "eval_window_days": ("integer", 1),
    "lookback_days": ("integer", 2),
    "min_symbols": ("integer", 1),
    "phase": ("integer", 0),
    "topk": ("integer", 1),
    "max_runs": ("integer", 1),
    "windows": ("integer", 1),
    "train_years": ("integer", 1),
    "valid_months": ("integer", 1),
    "test_months": ("integer", 1),
    "step_months": ("integer", 1),
}
NUMERIC_PARAM_MAXIMUMS = {
    "limit": GENERIC_WEB_LIMIT_MAX,
    "sleep": 60,
    "retries": BAOSTOCK_WEB_MAX_RETRIES,
    "timeout": 86400,
    "max_consecutive_failures": BAOSTOCK_WEB_MAX_CONSECUTIVE_FAILURES,
    "max_requests": ASHARE_SOURCE_MAX_REQUESTS,
    "overlap_days": BAOSTOCK_WEB_MAX_OVERLAP_DAYS,
    "snapshot_limit": GENERIC_WEB_LIMIT_MAX,
    "news_limit": GENERIC_WEB_LIMIT_MAX,
    "overlap_sample": DATA_QUALITY_WEB_MAX_SAMPLE,
    "merged_sample": DATA_QUALITY_WEB_MAX_SAMPLE,
    "eval_window_days": DSA_BACKTEST_WEB_MAX_EVAL_WINDOW_DAYS,
    "lookback_days": MARKET_REPORT_WEB_MAX_LOOKBACK_DAYS,
    "min_symbols": MARKET_REPORT_WEB_MAX_MIN_SYMBOLS,
    "phase": OPTIMIZE_STRATEGY_WEB_MAX_PHASE,
    "max_runs": QLIB_RESEARCH_WEB_MAX_RUNS,
    "windows": QLIB_RESEARCH_WEB_MAX_WINDOWS,
    "train_years": QLIB_RESEARCH_WEB_MAX_TRAIN_YEARS,
    "valid_months": QLIB_RESEARCH_WEB_MAX_MONTHS,
    "test_months": QLIB_RESEARCH_WEB_MAX_MONTHS,
    "step_months": QLIB_RESEARCH_WEB_MAX_MONTHS,
}
JOB_UPDATE_COLUMNS = {
    "status": "status",
    "pid": "pid",
    "return_code": "return_code",
    "started_at": "started_at",
    "completed_at": "completed_at",
    "updated_at": "updated_at",
    "error": "error",
    "artifact_paths_json": "artifact_paths_json",
    "runner_identity_json": "runner_identity_json",
}


@dataclass(frozen=True)
class QuantJob:
    """Persisted local quant workflow job."""

    job_id: str
    job_type: str
    label: str
    status: str
    params: Dict[str, Any] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    cwd: str = ""
    pid: Optional[int] = None
    return_code: Optional[int] = None
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = ""
    error: Optional[str] = None
    log_path: str = ""
    artifact_paths: List[str] = field(default_factory=list)
    runner_identity: Dict[str, Any] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)
    runner_alive: Optional[bool] = None
    can_cancel: bool = False
    log_updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "label": self.label,
            "status": self.status,
            "params": self.params,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "return_code": self.return_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "log_path": self.log_path,
            "artifact_paths": self.artifact_paths,
            "progress": self.progress,
            "runner_alive": self.runner_alive,
            "can_cancel": self.can_cancel,
            "log_updated_at": self.log_updated_at,
        }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _command_sha256(command: Iterable[str]) -> str:
    return hashlib.sha256(_json_dumps(list(command)).encode("utf-8")).hexdigest()


def _cancel_command_outcome(
    stdout: Any,
    contract: Mapping[str, Any],
) -> str:
    if contract.get("format") != "json":
        raise ValueError("cancel result contract format is not supported")
    if isinstance(stdout, bytes):
        if len(stdout) > 64_000:
            raise ValueError("cancel command output exceeded the trusted JSON limit")
        try:
            output_text = stdout.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("cancel command did not return valid UTF-8 JSON") from exc
    elif isinstance(stdout, str):
        if len(stdout.encode("utf-8")) > 64_000:
            raise ValueError("cancel command output exceeded the trusted JSON limit")
        output_text = stdout
    else:
        raise ValueError("cancel command did not return captured output")
    if not output_text.strip():
        raise ValueError("cancel command returned empty output")
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError("cancel command did not return valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("cancel command JSON must be an object")
    allowed = contract.get("allowed_outcomes")
    if not isinstance(allowed, Iterable) or isinstance(allowed, (str, bytes)):
        raise ValueError("cancel result contract has no allowed outcomes")
    allowed_outcomes = {str(value) for value in allowed}
    outcome = payload.get("cancel_outcome")
    if not isinstance(outcome, str) or outcome not in allowed_outcomes:
        raise ValueError("cancel command did not return a trusted cancel_outcome")
    return outcome


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _append_flag(
    command: List[str], params: Mapping[str, Any], key: str, flag: str
) -> None:
    if _as_bool(params.get(key)):
        command.append(flag)


def _append_value(
    command: List[str], params: Mapping[str, Any], key: str, flag: str
) -> None:
    value = params.get(key)
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def _numeric_kind_label(kind: str) -> str:
    return "an integer" if kind == "integer" else f"a {kind}"


def _validate_numeric_params(params: Mapping[str, Any]) -> None:
    for key, (kind, minimum) in NUMERIC_PARAM_RULES.items():
        value = params.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            raise ValueError(f"Param {key} must be {_numeric_kind_label(kind)}.")
        try:
            number = int(str(value)) if kind == "integer" else float(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Param {key} must be {_numeric_kind_label(kind)}."
            ) from exc
        if not math.isfinite(float(number)) or number < minimum:
            raise ValueError(f"Param {key} must be >= {minimum}.")
        maximum = NUMERIC_PARAM_MAXIMUMS.get(key)
        if maximum is not None and number > maximum:
            raise ValueError(f"Param {key} must be <= {maximum}.")


def _numeric_param_value(params: Mapping[str, Any], key: str) -> float | None:
    value = params.get(key)
    if value is None or value == "":
        return None
    return float(str(value))


def _validate_ashare_source_safety(job_type: str, params: Mapping[str, Any]) -> None:
    if job_type not in ASHARE_SOURCE_JOB_TYPES:
        return

    sleep_seconds = _numeric_param_value(params, "sleep")
    if sleep_seconds is not None and sleep_seconds < ASHARE_SOURCE_MIN_SLEEP_SECONDS:
        raise ValueError(
            f"Param sleep must be >= {ASHARE_SOURCE_MIN_SLEEP_SECONDS:g} for A-share source jobs."
        )

    retries = _numeric_param_value(params, "retries")
    if retries is not None and retries != ASHARE_SOURCE_REQUIRED_RETRIES:
        raise ValueError(
            f"Param retries must equal {ASHARE_SOURCE_REQUIRED_RETRIES} for A-share source jobs."
        )

    consecutive_failures = _numeric_param_value(params, "max_consecutive_failures")
    if consecutive_failures is not None and consecutive_failures > ASHARE_SOURCE_MAX_CONSECUTIVE_FAILURES:
        raise ValueError(
            f"Param max_consecutive_failures must be <= {ASHARE_SOURCE_MAX_CONSECUTIVE_FAILURES} "
            "for A-share source jobs."
        )

    request_budget = _numeric_param_value(params, "max_requests")
    if request_budget is not None and request_budget > ASHARE_SOURCE_MAX_REQUESTS:
        raise ValueError(
            f"Param max_requests must be <= {ASHARE_SOURCE_MAX_REQUESTS} for A-share source jobs."
        )

    overlap_days = _numeric_param_value(params, "overlap_days")
    if overlap_days is not None and overlap_days > ASHARE_SOURCE_MAX_OVERLAP_DAYS:
        raise ValueError(
            f"Param overlap_days must be <= {ASHARE_SOURCE_MAX_OVERLAP_DAYS} for A-share source jobs."
        )

    if job_type == "run_daily_update":
        source = str(params.get("ashare_source") or "none")
        allowed_sources = {"none", "baostock", "eastmoney", "akshare"}
        if source not in allowed_sources:
            raise ValueError(f"Param ashare_source must be one of: {', '.join(sorted(allowed_sources))}.")


def _int_param(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Param {key} must be an integer.")
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Param {key} must be an integer.") from exc
    if str(value).strip() not in {str(number), f"+{number}"}:
        raise ValueError(f"Param {key} must be an integer.")
    return number


def _safe_jqdata_web_limit(params: Mapping[str, Any]) -> int:
    raw = params.get("limit")
    if raw is None or raw == "":
        return JQDATA_WEB_DRY_RUN_DEFAULT_LIMIT
    value = _int_param(raw, "limit")
    if value < 1:
        raise ValueError("Param limit must be >= 1.")
    if value > JQDATA_WEB_DRY_RUN_MAX_LIMIT:
        raise ValueError(
            f"Param limit must be <= {JQDATA_WEB_DRY_RUN_MAX_LIMIT} for browser JQData dry-run jobs."
        )
    return value


def _ensure_jqdata_web_dry_run(params: Mapping[str, Any]) -> None:
    if "dry_run" in params and not _as_bool(params.get("dry_run")):
        raise ValueError(
            "download_jqdata web jobs only support dry_run=true. Use the script directly for real downloads."
        )


def _ensure_optimize_strategy_web_dry_run(params: Mapping[str, Any]) -> None:
    if "dry_run" in params and not _as_bool(params.get("dry_run")):
        raise ValueError(
            "optimize_strategy web jobs only support dry_run=true. Use the script directly for full experiments."
        )
    phase = _int_param(params.get("phase", 4), "phase")
    if phase < 0 or phase > 4:
        raise ValueError("phase must be between 0 and 4.")
    topk = _int_param(params.get("topk", 30), "topk")
    if topk < 1:
        raise ValueError("Param topk must be >= 1.")
    if topk > 500:
        raise ValueError("Param topk must be <= 500.")


def _safe_run_id(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("run_id must be a string.")
    raw = value.strip()
    if len(raw) > QLIB_RUN_ID_MAX_LENGTH:
        raise ValueError(f"run_id must be <= {QLIB_RUN_ID_MAX_LENGTH} characters.")
    if raw and not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError(
            "run_id may only contain letters, numbers, underscores, or hyphens."
        )
    return raw


def _approved_robust_signal_data_args(
    value: Mapping[str, Any],
) -> tuple[Path, str]:
    artifact_text = value.get("readiness_artifact_path")
    artifact_sha256 = value.get("readiness_artifact_sha256")
    if (
        not isinstance(artifact_text, str)
        or not artifact_text.strip()
        or not isinstance(artifact_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
    ):
        raise ValueError("Approved robust signal readiness identity is invalid.")
    artifact = Path(os.path.abspath(Path(artifact_text).expanduser()))
    readiness_root = (
        REPORTS_ROOT / "robust_signal" / "readiness_artifacts"
    ).resolve()
    expected_name = f"robust_signal_data_readiness_{artifact_sha256}.json"
    if (
        artifact.is_symlink()
        or not artifact.is_file()
        or artifact.parent.resolve(strict=False) != readiness_root
        or artifact.name != expected_name
        or artifact.stat().st_size > 1024 * 1024
    ):
        raise ValueError("Approved robust signal readiness artifact is invalid.")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != artifact_sha256:
        raise ValueError("Approved robust signal readiness artifact hash mismatch.")
    return artifact.resolve(), artifact_sha256


def _safe_iso_date(value: Any, key: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Param {key} must be an ISO date (YYYY-MM-DD).") from exc
    if parsed.isoformat() != raw:
        raise ValueError(f"Param {key} must be an ISO date (YYYY-MM-DD).")
    return raw


def _ensure_qlib_prediction_summary_params(params: Mapping[str, Any]) -> None:
    topk = _int_param(params.get("topk", 20), "topk")
    if topk < 1:
        raise ValueError("Param topk must be >= 1.")
    if topk > QLIB_PREDICTION_WEB_TOPK_MAX:
        raise ValueError(f"Param topk must be <= {QLIB_PREDICTION_WEB_TOPK_MAX}.")
    _safe_run_id(params.get("run_id"))


def _ensure_qlib_research_params(params: Mapping[str, Any]) -> None:
    if "topk" in params and params.get("topk") not in (None, ""):
        topk = _int_param(params.get("topk"), "topk")
        if topk < 1:
            raise ValueError("Param topk must be >= 1.")
        if topk > QLIB_PREDICTION_WEB_TOPK_MAX:
            raise ValueError(f"Param topk must be <= {QLIB_PREDICTION_WEB_TOPK_MAX}.")
    if "max_runs" in params and params.get("max_runs") not in (None, ""):
        max_runs = _int_param(params.get("max_runs"), "max_runs")
        if max_runs > QLIB_RESEARCH_WEB_MAX_RUNS:
            raise ValueError(f"Param max_runs must be <= {QLIB_RESEARCH_WEB_MAX_RUNS}.")
    if "windows" in params and params.get("windows") not in (None, ""):
        windows = _int_param(params.get("windows"), "windows")
        if windows > QLIB_RESEARCH_WEB_MAX_WINDOWS:
            raise ValueError(
                f"Param windows must be <= {QLIB_RESEARCH_WEB_MAX_WINDOWS}."
            )
    if "train_years" in params and params.get("train_years") not in (None, ""):
        train_years = _int_param(params.get("train_years"), "train_years")
        if train_years > QLIB_RESEARCH_WEB_MAX_TRAIN_YEARS:
            raise ValueError(
                f"Param train_years must be <= {QLIB_RESEARCH_WEB_MAX_TRAIN_YEARS}."
            )
    for key in ("valid_months", "test_months", "step_months"):
        if key in params and params.get(key) not in (None, ""):
            months = _int_param(params.get(key), key)
            if months > QLIB_RESEARCH_WEB_MAX_MONTHS:
                raise ValueError(
                    f"Param {key} must be <= {QLIB_RESEARCH_WEB_MAX_MONTHS}."
                )
    _safe_run_id(params.get("run_id"))
    if "config" in params and params.get("config") not in (None, ""):
        _safe_config_path(params.get("config"))


def _safe_symbols(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) > 1000:
        raise ValueError("symbols is too long.")
    if not re.fullmatch(r"[A-Za-z0-9,._\- ]+", raw):
        raise ValueError(
            "symbols may only contain letters, numbers, commas, dots, underscores, hyphens, and spaces."
        )
    symbols = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if any(re.search(r"\s", symbol) for symbol in symbols):
        raise ValueError(
            "symbols must be comma-separated stock codes without embedded spaces."
        )
    if len(symbols) > CHINA_ASTOCK_SYMBOL_MAX_COUNT:
        raise ValueError(
            f"symbols may include at most {CHINA_ASTOCK_SYMBOL_MAX_COUNT} entries."
        )
    too_long = next(
        (symbol for symbol in symbols if len(symbol) > CHINA_ASTOCK_SYMBOL_MAX_LENGTH),
        None,
    )
    if too_long is not None:
        raise ValueError(
            f"Each symbol must be <= {CHINA_ASTOCK_SYMBOL_MAX_LENGTH} characters."
        )
    return ",".join(symbols)


def _ensure_china_astock_indicator_params(params: Mapping[str, Any]) -> None:
    lookback = _int_param(params.get("lookback_days", 120), "lookback_days")
    if lookback < 20:
        raise ValueError(
            "lookback_days must be >= 20 for china-astock indicator reports."
        )
    if lookback > 2000:
        raise ValueError(
            "lookback_days must be <= 2000 for china-astock indicator reports."
        )
    _safe_china_astock_indicator_limit(params)
    _safe_symbols(params.get("symbols"))


def _safe_china_astock_indicator_limit(params: Mapping[str, Any]) -> int:
    raw = params.get("limit")
    limit = (
        CHINA_ASTOCK_INDICATOR_DEFAULT_LIMIT
        if raw in (None, "")
        else _int_param(raw, "limit")
    )
    if limit < 1:
        raise ValueError("Param limit must be >= 1.")
    if limit > CHINA_ASTOCK_INDICATOR_MAX_LIMIT:
        raise ValueError(
            f"Param limit must be <= {CHINA_ASTOCK_INDICATOR_MAX_LIMIT} for china-astock indicator reports."
        )
    return limit


def _safe_config_path(value: Any) -> str:
    raw = str(value or "baseline_lgb_alpha158.yaml").strip()
    if not raw:
        raw = "baseline_lgb_alpha158.yaml"
    if "%" in raw:
        raise ValueError("Config must be a file name under quant_workflow/configs.")
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("Config must be a file name under quant_workflow/configs.")
    if not CONFIG_FILE_NAME_RE.fullmatch(raw):
        raise ValueError(
            "Config file name may only contain letters, numbers, dots, underscores, and hyphens."
        )
    if Path(raw).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "Config must be a .yaml or .yml file under quant_workflow/configs."
        )
    configs_root = (QUANT_WORKFLOW_ROOT / "configs").resolve()
    config_path = (configs_root / raw).resolve()
    if not config_path.is_relative_to(configs_root):
        raise ValueError("Config must be a file name under quant_workflow/configs.")
    return str(config_path)


def _safe_provider(value: Any) -> str:
    raw = str(value or "canonical").strip().lower()
    allowed = {"canonical", "cleaned"}
    if raw not in allowed:
        raise ValueError("Provider must be one of: canonical, cleaned.")
    return raw


def _approved_provider_identity_json(
    provider: str,
    identity: Mapping[str, Any],
) -> str:
    expected_dataset_id = TRAINING_PROVIDER_DATASET_IDS[provider]
    payload = {field: identity.get(field) for field in PROVIDER_IDENTITY_FIELDS}
    if identity.get("provider_contract") == "direct_parquet_v1":
        payload = {
            field: identity.get(field) for field in DIRECT_PROVIDER_IDENTITY_FIELDS
        }
        descriptor_path = str(payload.get("descriptor_path") or "")
        if (
            payload.get("dataset_id") != expected_dataset_id
            or payload.get("cleaning_mode") != provider
            or not descriptor_path
            or Path(descriptor_path).is_absolute()
            or ".." in Path(descriptor_path).parts
            or payload.get("index_dataset_id") != "cn_index_daily"
            or not isinstance(payload.get("index_snapshot_id"), str)
            or Path(str(payload.get("index_snapshot_id"))).name
            != payload.get("index_snapshot_id")
            or any(
                not isinstance(payload.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field))) is None
                for field in (
                    "manifest_sha256",
                    "content_sha256",
                    "descriptor_sha256",
                    "index_content_sha256",
                    "index_manifest_sha256",
                )
            )
        ):
            raise ValueError("Approved direct Provider identity is invalid or mismatched.")
        return _json_dumps(payload)
    snapshot_id = payload.get("snapshot_id")
    if (
        payload.get("provider_contract") != "immutable_snapshot_v2"
        or payload.get("dataset_id") != expected_dataset_id
        or not isinstance(snapshot_id, str)
        or not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
        or any(
            not isinstance(payload.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field))) is None
            for field in ("manifest_sha256", "content_sha256")
        )
    ):
        raise ValueError("Approved provider identity is invalid or mismatched.")
    return _json_dumps(payload)


def _approved_market_data_bars_path(identity: Mapping[str, Any]) -> Path:
    payload = {field: identity.get(field) for field in MARKET_DATA_IDENTITY_FIELDS}
    snapshot_id = payload.get("snapshot_id")
    expected_state = payload.get("state_fingerprint")
    if (
        payload.get("dataset_id") != ASHARE_DAILY_QFQ_DATASET_ID
        or not isinstance(snapshot_id, str)
        or not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
        or any(
            not isinstance(payload.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(payload.get(field))) is None
            for field in ("manifest_sha256", "content_sha256")
        )
        or not isinstance(expected_state, dict)
    ):
        raise ValueError("Approved market data identity is invalid or mismatched.")

    root = QUANT_DATA_ROOT.expanduser().resolve()
    snapshots_root = (root / ASHARE_DAILY_QFQ_DATASET_ROOT / "snapshots").resolve()
    snapshot = (snapshots_root / snapshot_id).resolve()
    if snapshot.parent != snapshots_root:
        raise ValueError("Approved market data identity is invalid or mismatched.")
    relative_snapshot = snapshot.relative_to(root)
    current = root
    for part in relative_snapshot.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Approved market data identity is invalid or mismatched.")

    manifest_path = snapshot / "DATASET_MANIFEST.json"
    try:
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 1_048_576:
            raise ValueError("Approved market data identity is invalid or mismatched.")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Approved market data identity is invalid or mismatched."
        ) from exc
    if (
        not isinstance(manifest, dict)
        or hashlib.sha256(manifest_bytes).hexdigest()
        != payload.get("manifest_sha256")
        or manifest.get("dataset_id") != ASHARE_DAILY_QFQ_DATASET_ID
        or manifest.get("snapshot_id") != snapshot_id
        or manifest.get("content_sha256") != payload.get("content_sha256")
        or manifest.get("data_kind") != "daily_bars"
        or manifest.get("role") != "canonical"
        or manifest.get("status") != "ready"
        or manifest.get("price_basis") != "qfq"
    ):
        raise ValueError("Approved market data identity is invalid or mismatched.")
    bars_path = snapshot / "bars"
    if bars_path.is_symlink() or not bars_path.is_dir():
        raise ValueError("Approved market data identity is invalid or mismatched.")
    try:
        from quant_workflow.scripts.quant_data_contract import (
            DataContractError,
            filesystem_state_fingerprint,
        )

        current_state = filesystem_state_fingerprint(snapshot)
    except (DataContractError, OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "Approved market data snapshot state could not be verified."
        ) from exc
    if current_state != expected_state:
        raise ValueError("Approved market data snapshot changed after approval.")
    return bars_path.resolve()


def _python_bin() -> str:
    _warn_if_legacy_qlib_python_is_ignored()
    if QUANT_WORKFLOW_PYTHON.exists():
        return str(QUANT_WORKFLOW_PYTHON)
    _warn_if_missing_quant_workflow_python_is_ignored()
    return str(Path(sys.executable))


def _warn_if_legacy_qlib_python_is_ignored() -> None:
    global _LEGACY_QLIB_PYTHON_WARNING_EMITTED
    if _LEGACY_QLIB_PYTHON_WARNING_EMITTED:
        return
    if os.environ.get("QLIB_PYTHON") and not os.environ.get("QUANT_WORKFLOW_PYTHON"):
        logger.warning(
            "QLIB_PYTHON is ignored by the DSA web workflow. "
            "Set QUANT_WORKFLOW_PYTHON to override the Python used by web quant jobs."
        )
        _LEGACY_QLIB_PYTHON_WARNING_EMITTED = True


def _warn_if_missing_quant_workflow_python_is_ignored() -> None:
    global _MISSING_QUANT_WORKFLOW_PYTHON_WARNING_EMITTED
    if _MISSING_QUANT_WORKFLOW_PYTHON_WARNING_EMITTED:
        return
    if os.environ.get("QUANT_WORKFLOW_PYTHON"):
        logger.warning(
            "QUANT_WORKFLOW_PYTHON points to a missing file (%s); "
            "falling back to the running DSA Python interpreter (%s).",
            QUANT_WORKFLOW_PYTHON,
            Path(sys.executable),
        )
        _MISSING_QUANT_WORKFLOW_PYTHON_WARNING_EMITTED = True


def _quant_job_child_env(
    extra: Mapping[str, str],
    *,
    allowed_names: Iterable[str] = (),
    allowed_prefixes: Iterable[str] = (),
) -> Dict[str, str]:
    explicit_names = CHILD_ENV_EXACT_NAMES | {str(name) for name in allowed_names}
    explicit_prefixes = tuple(CHILD_ENV_PREFIXES) + tuple(
        str(prefix) for prefix in allowed_prefixes
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key in explicit_names
        or any(key.startswith(prefix) for prefix in explicit_prefixes)
    }
    env.update({key: str(value) for key, value in extra.items()})
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _return_code_indicates_cancel(return_code: Optional[int]) -> bool:
    return return_code in {
        -signal.SIGTERM,
        -signal.SIGKILL,
        128 + signal.SIGTERM,
        128 + signal.SIGKILL,
    }


def _artifact_patterns_may_overlap(left: str, right: str) -> bool:
    left_path = Path(os.path.abspath(os.path.expanduser(str(left))))
    right_path = Path(os.path.abspath(os.path.expanduser(str(right))))
    if left_path.parent != right_path.parent:
        return False

    left_name = left_path.name
    right_name = right_path.name
    left_magic = glob.has_magic(left_name)
    right_magic = glob.has_magic(right_name)
    if not left_magic and not right_magic:
        return left_name == right_name
    if left_magic and not right_magic:
        return fnmatch.fnmatchcase(right_name, left_name)
    if right_magic and not left_magic:
        return fnmatch.fnmatchcase(left_name, right_name)

    def literal_edges(pattern: str) -> tuple[str, str]:
        wildcard_positions = [
            index for index, character in enumerate(pattern) if character in "*?["
        ]
        first = min(wildcard_positions)
        last = max(wildcard_positions)
        return pattern[:first], pattern[last + 1 :]

    left_prefix, left_suffix = literal_edges(left_name)
    right_prefix, right_suffix = literal_edges(right_name)
    prefixes_overlap = left_prefix.startswith(right_prefix) or right_prefix.startswith(
        left_prefix
    )
    suffixes_overlap = left_suffix.endswith(right_suffix) or right_suffix.endswith(
        left_suffix
    )
    return prefixes_overlap and suffixes_overlap


def _artifact_pattern_sets_overlap(
    left_patterns: Iterable[str],
    right_patterns: Iterable[str],
) -> bool:
    return any(
        _artifact_patterns_may_overlap(str(left), str(right))
        for left in left_patterns
        for right in right_patterns
    )


def _script_command(script_name: str) -> List[str]:
    return [_python_bin(), str(QUANT_WORKFLOW_ROOT / "scripts" / script_name)]


def _append_reports_out_dir(command: List[str]) -> None:
    command.extend(["--out-dir", str(REPORTS_ROOT)])


def _build_default_command_specs() -> Dict[str, Dict[str, Any]]:
    data_quality_artifact_patterns = [
        str(REPORTS_ROOT / "data_quality_report_latest.json"),
        str(REPORTS_ROOT / "data_quality_report_latest.md"),
        str(REPORTS_ROOT / "data_quality_report_*.json"),
        str(REPORTS_ROOT / "data_quality_report_*.md"),
        str(REPORTS_ROOT / "data_quality_sources_*.csv"),
        str(REPORTS_ROOT / "data_quality_issues_*.csv"),
        str(REPORTS_ROOT / "data_quality_overlap_*.csv"),
        str(REPORTS_ROOT / "data_quality_ashare_qfq_summary_*.csv"),
        str(REPORTS_ROOT / "research_status_latest.json"),
        str(
            REPORTS_ROOT
            / "test_runs"
            / "data_quality"
            / "data_quality_report_sampled_latest.json"
        ),
        str(
            REPORTS_ROOT
            / "test_runs"
            / "data_quality"
            / "data_quality_report_sampled_latest.md"
        ),
        str(
            REPORTS_ROOT
            / "test_runs"
            / "data_quality"
            / "research_status_sampled_latest.json"
        ),
        str(REPORTS_ROOT / "test_runs" / "data_quality" / "data_quality_report_*.json"),
        str(REPORTS_ROOT / "test_runs" / "data_quality" / "data_quality_report_*.md"),
        str(REPORTS_ROOT / "test_runs" / "data_quality" / "data_quality_*.csv"),
    ]
    shared_training_artifact_patterns = [
        str(REPORTS_ROOT / ".training-runner-resource"),
        str(REPORTS_ROOT / "latest_training_result.json"),
        str(REPORTS_ROOT / "train_baseline_lgb_status_latest.json"),
        str(REPORTS_ROOT / "train_baseline_lgb_*.log"),
        str(REPORTS_ROOT / "experiment_runs.jsonl"),
        str(REPORTS_ROOT / "experiment_log.md"),
        str(REPORTS_ROOT / "full_experiment_report.md"),
        str(REPORTS_ROOT / "ic_ir_analysis.md"),
    ]

    def prepare_baostock_refresh(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("prepare_baostock_refresh.py")
        _append_reports_out_dir(command)
        _append_value(command, params, "target_end", "--target-end")
        _append_value(command, params, "limit", "--limit")
        return command

    def download_baostock(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("download_baostock.py")
        for key, flag in (
            ("start", "--start"),
            ("end", "--end"),
            ("sleep", "--sleep"),
            ("limit", "--limit"),
            ("retries", "--retries"),
            ("timeout", "--timeout"),
            ("max_consecutive_failures", "--max-consecutive-failures"),
            ("max_requests", "--max-requests"),
            ("overlap_days", "--overlap-days"),
        ):
            _append_value(command, params, key, flag)
        for key, flag in (
            ("retry_failed", "--retry-failed"),
            ("retry_empty", "--retry-empty"),
            ("retry_timeout", "--retry-timeout"),
            ("incremental", "--incremental"),
        ):
            _append_flag(command, params, key, flag)
        return command

    def download_public_daily(script_name: str, params: Mapping[str, Any]) -> List[str]:
        command = _script_command(script_name)
        for key, flag in (
            ("start", "--start"),
            ("end", "--end"),
            ("sleep", "--sleep"),
            ("limit", "--limit"),
            ("retries", "--retries"),
            ("timeout", "--timeout"),
            ("max_consecutive_failures", "--max-consecutive-failures"),
            ("max_requests", "--max-requests"),
            ("overlap_days", "--overlap-days"),
        ):
            _append_value(command, params, key, flag)
        for key, flag in (
            ("retry_failed", "--retry-failed"),
            ("incremental", "--incremental"),
            ("use_system_proxy", "--use-system-proxy"),
        ):
            _append_flag(command, params, key, flag)
        return command

    def download_etf_momentum(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("quant_data_cli.py")
        command.extend(["update", "cn_etf_daily"])
        _append_value(command, params, "end", "--through")
        return command

    def generate_etf_momentum_report(params: Mapping[str, Any]) -> List[str]:
        del params
        return _script_command("generate_etf_momentum_strategy_report.py")

    def validate_latest_snapshot(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("validate_latest_snapshot.py")
        _append_reports_out_dir(command)
        _append_value(command, params, "limit", "--limit")
        _append_value(command, params, "timeout", "--timeout")
        _append_flag(command, params, "use_system_proxy", "--use-system-proxy")
        return command

    def download_ashare_news(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("download_ashare_news.py")
        for key, flag in (
            ("limit", "--limit"),
            ("sleep", "--sleep"),
            ("retries", "--retries"),
            ("timeout", "--timeout"),
        ):
            _append_value(command, params, key, flag)
        _append_flag(command, params, "use_system_proxy", "--use-system-proxy")
        return command

    def download_jqdata(params: Mapping[str, Any]) -> List[str]:
        _ensure_jqdata_web_dry_run(params)
        command = _script_command("download_jqdata.py")
        for key, flag in (
            ("universe", "--universe"),
            ("start", "--start"),
            ("end", "--end"),
            ("sleep", "--sleep"),
        ):
            _append_value(command, params, key, flag)
        command.extend(["--limit", str(_safe_jqdata_web_limit(params)), "--dry-run"])
        return command

    def merge_to_qlib(params: Mapping[str, Any]) -> List[str]:
        if _as_bool(params.get("smoke")):
            command = _script_command("merge_to_qlib.py")
            command.extend(
                [
                    "--smoke-fixture",
                    "--smoke-dir",
                    str(REPORTS_ROOT / "smoke" / "merge_to_qlib"),
                ]
            )
            _append_flag(command, params, "no_overwrite", "--no-overwrite")
            return command
        if _as_bool(params.get("no_overwrite")):
            raise ValueError("no_overwrite is only supported by the isolated smoke fixture")
        return _script_command("build_qlib_direct_provider.py") + [
            "--cleaning-mode",
            "canonical",
        ]

    def apply_data_cleaning(params: Mapping[str, Any]) -> List[str]:
        if _as_bool(params.get("dry_run")):
            command = _script_command("apply_data_cleaning.py")
            _append_reports_out_dir(command)
            command.append("--dry-run")
            return command
        return _script_command("build_qlib_direct_provider.py") + [
            "--cleaning-mode",
            "cleaned",
        ]

    def repair_qlib_provider(_params: Mapping[str, Any]) -> List[str]:
        return _script_command("repair_qlib_provider.py") + ["--dry-run"]

    def generate_csi300(params: Mapping[str, Any]) -> List[str]:
        return _script_command("generate_csi300.py") + ["--dry-run"]

    def add_index_benchmark(params: Mapping[str, Any]) -> List[str]:
        return _script_command("add_index_benchmark.py") + ["--dry-run"]

    def report_data_quality(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_data_quality.py")
        _append_reports_out_dir(command)
        _append_value(command, params, "overlap_sample", "--overlap-sample")
        _append_value(command, params, "merged_sample", "--merged-sample")
        return command

    def report_data_anomalies(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_data_anomalies.py")
        _append_reports_out_dir(command)
        return command

    def report_data_hygiene(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_data_hygiene.py")
        _append_reports_out_dir(command)
        return command

    def report_workbench_capabilities(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_workbench_capabilities.py")
        _append_reports_out_dir(command)
        return command

    def report_market_regime(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_market_regime.py")
        _append_reports_out_dir(command)
        _append_value(command, params, "lookback_days", "--lookback-days")
        _append_value(command, params, "min_symbols", "--min-symbols")
        _append_value(command, params, "limit", "--limit")
        return command

    def report_cash_control_backtest(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("report_cash_control_backtest.py")
        _append_reports_out_dir(command)
        _append_value(command, params, "lookback_days", "--lookback-days")
        _append_value(command, params, "min_symbols", "--min-symbols")
        _append_value(command, params, "limit", "--limit")
        return command

    def report_qlib_predictions(params: Mapping[str, Any]) -> List[str]:
        _ensure_qlib_prediction_summary_params(params)
        command = _script_command("report_qlib_predictions.py")
        _append_reports_out_dir(command)
        run_id = _safe_run_id(params.get("run_id"))
        if run_id:
            command.extend(["--run-id", run_id])
        _append_value(command, params, "topk", "--topk")
        return command

    def report_qlib_research(section: str, params: Mapping[str, Any]) -> List[str]:
        _ensure_qlib_research_params(params)
        command = _script_command("report_qlib_research.py")
        _append_reports_out_dir(command)
        command.extend(["--section", section])
        run_id = _safe_run_id(params.get("run_id"))
        if run_id:
            command.extend(["--run-id", run_id])
        if "config" in params and params.get("config") not in (None, ""):
            command.extend(["--config", _safe_config_path(params.get("config"))])
        for key, flag in (
            ("topk", "--topk"),
            ("max_runs", "--max-runs"),
            ("windows", "--windows"),
            ("train_years", "--train-years"),
            ("valid_months", "--valid-months"),
            ("test_months", "--test-months"),
            ("step_months", "--step-months"),
        ):
            _append_value(command, params, key, flag)
        return command

    def report_china_astock_indicators(params: Mapping[str, Any]) -> List[str]:
        _ensure_china_astock_indicator_params(params)
        command = _script_command("report_china_astock_indicators.py")
        command.extend(["--output-dir", str(REPORTS_ROOT)])
        symbols = _safe_symbols(params.get("symbols"))
        if symbols:
            command.extend(["--symbols", symbols])
        _append_value(command, params, "lookback_days", "--lookback-days")
        command.extend(["--limit", str(_safe_china_astock_indicator_limit(params))])
        return command

    def optimize_strategy(params: Mapping[str, Any]) -> List[str]:
        _ensure_optimize_strategy_web_dry_run(params)
        command = _script_command("optimize_strategy.py")
        command.extend(
            ["--dry-run", "--skip-training", "--output-dir", str(REPORTS_ROOT)]
        )
        _append_value(command, params, "phase", "--phase")
        _append_value(command, params, "topk", "--topk")
        return command

    def train_baseline_lgb(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("train_baseline_lgb.py")
        command.extend(["--config", _safe_config_path(params.get("config"))])
        command.extend(["--provider", _safe_provider(params.get("provider"))])
        _append_value(command, params, "experiment_name", "--experiment-name")
        return command

    def train_baseline_lgb_equivalence(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("train_baseline_lgb.py")
        command.extend(
            [
                "--config",
                _safe_config_path("baseline_lgb_equivalence.yaml"),
                "--provider",
                _safe_provider(params.get("provider")),
                "--experiment-name",
                "dsa_training_equivalence",
                "--no-publish-latest",
                "--result-scope",
                "training_equivalence",
                "--equivalence-channel",
                "web",
            ]
        )
        return command

    def run_candidate_pipeline(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("run_candidate_pipeline.py")
        command.extend(["--output-dir", str(REPORTS_ROOT)])
        command.extend(["--provider", _safe_provider(params.get("provider"))])
        _append_flag(command, params, "dry_run", "--dry-run")
        return command

    def run_daily_update(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("run_daily_update.py")
        _append_value(command, params, "ashare_source", "--ashare-source")
        for key, flag in (
            ("skip_download", "--skip-download"),
            ("dry_run", "--dry-run"),
            ("continue_on_error", "--continue-on-error"),
        ):
            _append_flag(command, params, key, flag)
        for key, flag in (
            ("overlap_days", "--overlap-days"),
            ("sleep", "--sleep"),
            ("timeout", "--timeout"),
            ("retries", "--retries"),
            ("max_consecutive_failures", "--max-consecutive-failures"),
            ("max_requests", "--max-requests"),
            ("limit", "--limit"),
            ("end", "--end"),
            ("news_limit", "--news-limit"),
            ("overlap_sample", "--overlap-sample"),
            ("merged_sample", "--merged-sample"),
        ):
            _append_value(command, params, key, flag)
        for key, flag in (
            ("skip_baostock", "--skip-baostock"),
            ("skip_eastmoney", "--skip-eastmoney"),
            ("skip_akshare", "--skip-akshare"),
            ("skip_global_daily", "--skip-global-daily"),
            ("skip_ashare_news", "--skip-ashare-news"),
            ("skip_etf_momentum", "--skip-etf-momentum"),
            ("use_system_proxy", "--use-system-proxy"),
        ):
            _append_flag(command, params, key, flag)
        return command

    def run_dsa_backtest(params: Mapping[str, Any]) -> List[str]:
        command = [_python_bin(), str(DSA_ROOT / "main.py"), "--backtest"]
        _append_value(command, params, "code", "--backtest-code")
        _append_value(command, params, "eval_window_days", "--backtest-days")
        _append_flag(command, params, "force", "--backtest-force")
        return command

    def run_robust_strategy(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("launch_robust_strategy.py")
        command.extend(["--report-dir", str(REPORTS_ROOT / "robust_strategy")])
        command.extend(["--mlruns-dir", str(MLRUNS_ROOT)])
        as_of_date = _safe_iso_date(params.get("as_of_date"), "as_of_date")
        if as_of_date:
            command.extend(["--as-of-date", as_of_date])
        run_id = _safe_run_id(params.get("run_id"))
        if run_id:
            command.extend(["--run-id", run_id])
        _append_flag(command, params, "resume", "--resume")
        _append_flag(
            command, params, "skip_membership_build", "--skip-membership-build"
        )
        return command

    def _robust_signal_command(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("launch_robust_signal_research.py")
        command.extend(
            [
                "--config",
                str(QUANT_WORKFLOW_ROOT / "configs" / "robust_signal_research.yaml"),
                "--quant-data-dir",
                str(quant_data_root(DSA_ROOT)),
                "--report-dir",
                str(REPORTS_ROOT / "robust_signal"),
                "--mlruns-dir",
                str(MLRUNS_ROOT),
            ]
        )
        run_id = _safe_run_id(params.get("run_id"))
        if run_id:
            command.extend(["--run-id", run_id])
        _append_flag(command, params, "smoke", "--smoke")
        _append_flag(command, params, "plan", "--plan")
        return command

    def prepare_robust_signal_data(params: Mapping[str, Any]) -> List[str]:
        del params
        command = _script_command("prepare_robust_signal_data.py")
        command.extend(
            [
                "--quant-data-dir",
                str(quant_data_root(DSA_ROOT)),
                "--status-path",
                str(
                    REPORTS_ROOT
                    / "robust_signal"
                    / "robust_signal_data_readiness_latest.json"
                ),
                "--quality-pointer-path",
                str(REPORTS_ROOT / "data_quality_report_latest.json"),
            ]
        )
        return command

    def run_robust_signal_research(params: Mapping[str, Any]) -> List[str]:
        return _robust_signal_command(params)

    def preview_robust_signal_plan(params: Mapping[str, Any]) -> List[str]:
        del params
        return _robust_signal_command({"plan": True})

    def pause_robust_signal_research(params: Mapping[str, Any]) -> List[str]:
        run_id = _safe_run_id(params.get("run_id"))
        if not run_id:
            raise ValueError("run_id is required for robust signal pause")
        command = _robust_signal_command({"run_id": run_id})
        command.append("--request-pause")
        return command

    def resume_robust_signal_research(params: Mapping[str, Any]) -> List[str]:
        run_id = _safe_run_id(params.get("run_id"))
        if not run_id:
            raise ValueError("run_id is required for robust signal resume")
        command = _robust_signal_command(
            {"run_id": run_id, "smoke": _as_bool(params.get("smoke"))}
        )
        command.append("--resume")
        return command

    def cancel_robust_signal_research(params: Mapping[str, Any]) -> List[str]:
        run_id = _safe_run_id(params.get("run_id"))
        if not run_id:
            raise ValueError("run_id is required for robust signal cancellation")
        command = _robust_signal_command({"run_id": run_id})
        command.append("--cancel")
        return command

    def run_robust_confirmation(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("run_robust_confirmation.py")
        command.extend(["--report-dir", str(REPORTS_ROOT / "robust_strategy")])
        run_id = _safe_run_id(params.get("run_id"))
        if run_id:
            command.extend(["--run-id", run_id])
        _append_flag(command, params, "resume", "--resume")
        return command

    def run_robust_shadow_day(params: Mapping[str, Any]) -> List[str]:
        robust_dir = REPORTS_ROOT / "robust_strategy"
        command = _script_command("run_robust_shadow_day.py")
        command.extend(
            [
                "--report-dir",
                str(robust_dir),
                "--frozen",
                str(robust_dir / "frozen_strategy_latest.json"),
                "--confirmation-latest",
                str(robust_dir / "confirmation_latest.json"),
                "--provider-latest",
                str(robust_dir / "provider_latest.json"),
            ]
        )
        as_of_date = (
            _safe_iso_date(params.get("as_of_date"), "as_of_date")
            or date.today().isoformat()
        )
        command.extend(["--as-of-date", as_of_date])
        return command

    def qlib_research_artifacts(section: str) -> List[str]:
        return [
            str(REPORTS_ROOT / f"qlib_research_{section}_latest.json"),
            str(REPORTS_ROOT / f"qlib_research_{section}_latest.md"),
            str(REPORTS_ROOT / f"qlib_research_{section}_*.json"),
            str(REPORTS_ROOT / f"qlib_research_{section}_*.md"),
        ]

    return {
        "prepare_baostock_refresh": {
            "label": "准备 Baostock 增量清单",
            "command": prepare_baostock_refresh,
            "allowed_params": {"target_end", "limit"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "baostock_refresh_plan_latest.md"),
                str(REPORTS_ROOT / "baostock_refresh_symbols_latest.csv"),
                str(REPORTS_ROOT / "baostock_refresh_remaining_latest.txt"),
                str(REPORTS_ROOT / "baostock_refresh_remaining_plan_latest.md"),
            ],
        },
        "download_baostock": {
            "label": "下载 Baostock 长历史",
            "command": download_baostock,
            "allowed_params": {
                "start",
                "end",
                "sleep",
                "limit",
                "retries",
                "timeout",
                "max_consecutive_failures",
                "max_requests",
                "overlap_days",
                "retry_failed",
                "retry_empty",
                "retry_timeout",
                "incremental",
            },
            "artifact_patterns": [
                str(REPORTS_ROOT / "baostock_refresh_remaining_latest.txt"),
                str(REPORTS_ROOT / "baostock_refresh_stale_*_latest.txt"),
            ],
        },
        "run_daily_update": {
            "label": "运行每日数据更新与 V2 校验",
            "command": run_daily_update,
            "allowed_params": {
                "skip_download",
                "ashare_source",
                "dry_run",
                "continue_on_error",
                "overlap_days",
                "sleep",
                "timeout",
                "retries",
                "max_consecutive_failures",
                "max_requests",
                "limit",
                "end",
                "news_limit",
                "skip_baostock",
                "skip_eastmoney",
                "skip_akshare",
                "skip_global_daily",
                "skip_ashare_news",
                "skip_etf_momentum",
                "use_system_proxy",
                "overlap_sample",
                "merged_sample",
            },
            "artifact_patterns": [
                str(REPORTS_ROOT / "daily_update_summary_latest.json"),
                str(REPORTS_ROOT / "daily_update_summary_*.json"),
                *data_quality_artifact_patterns,
            ],
        },
        "download_eastmoney": {
            "label": "下载东方财富日线",
            "command": lambda params: download_public_daily(
                "download_eastmoney.py", params
            ),
            "allowed_params": {
                "start",
                "end",
                "sleep",
                "limit",
                "retries",
                "timeout",
                "max_consecutive_failures",
                "max_requests",
                "retry_failed",
                "incremental",
                "overlap_days",
                "use_system_proxy",
            },
        },
        "download_akshare": {
            "label": "下载 AkShare 日线",
            "command": lambda params: download_public_daily(
                "download_akshare.py", params
            ),
            "allowed_params": {
                "start",
                "end",
                "sleep",
                "limit",
                "retries",
                "timeout",
                "max_consecutive_failures",
                "max_requests",
                "retry_failed",
                "incremental",
                "overlap_days",
                "use_system_proxy",
            },
        },
        "download_global_daily": {
            "label": "下载全球日线与原始证据",
            "command": lambda params: download_public_daily("download_global_daily.py", params),
            "allowed_params": {
                "start",
                "end",
                "sleep",
                "limit",
                "retries",
                "timeout",
                "incremental",
                "overlap_days",
                "use_system_proxy",
            },
            "artifact_patterns": [
                str(REPORTS_ROOT / "download_state" / "global_daily_download_summary_latest.json"),
                str(QUANT_DATA_ROOT / "raw" / "global_1d" / "yahoo" / "**" / "*.json"),
            ],
        },
        "download_ashare_news": {
            "label": "下载 A 股新闻原始证据",
            "command": download_ashare_news,
            "allowed_params": {"limit", "sleep", "retries", "timeout", "use_system_proxy"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "download_state" / "ashare_news_download_summary_latest.json"),
                str(QUANT_DATA_ROOT / "raw" / "ashare_news" / "**" / "*.json"),
            ],
        },
        "download_etf_momentum": {
            "label": "更新统一 ETF 数据",
            "command": download_etf_momentum,
            "allowed_params": {"end"},
            "artifact_patterns": [
                str(
                    quant_data_root(DSA_ROOT)
                    / "parquet"
                    / "etf_1d"
                    / "cn_etf_daily"
                    / "LATEST.json"
                ),
            ],
        },
        "generate_etf_momentum_report": {
            "label": "生成 ETF 完整策略报告",
            "command": generate_etf_momentum_report,
            "allowed_params": set(),
            "artifact_patterns": [
                str(
                    REPORTS_ROOT
                    / "etf_momentum_rotation"
                    / "strategy_report_latest.json"
                ),
                str(
                    REPORTS_ROOT
                    / "etf_momentum_rotation"
                    / "strategy_report_latest.md"
                ),
                str(
                    REPORTS_ROOT
                    / "etf_momentum_rotation"
                    / "runs"
                    / "*"
                    / "strategy_report.json"
                ),
            ],
        },
        "validate_latest_snapshot": {
            "label": "校验新浪最新快照",
            "command": validate_latest_snapshot,
            "allowed_params": {"limit", "timeout", "use_system_proxy"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "latest_snapshot_validation_latest.json"),
                str(REPORTS_ROOT / "latest_snapshot_validation_latest.md"),
                str(REPORTS_ROOT / "latest_snapshot_validation_*.json"),
                str(REPORTS_ROOT / "latest_snapshot_validation_*.md"),
            ],
        },
        "download_jqdata": {
            "label": "验证 JQData 连接（dry-run）",
            "command": download_jqdata,
            "allowed_params": {"universe", "start", "end", "limit", "sleep", "dry_run"},
            "env_names": {"JQ_USERNAME", "JQ_PASSWORD"},
        },
        "merge_to_qlib": {
            "label": "生成正式 Qlib 直读 Provider",
            "command": merge_to_qlib,
            "allowed_params": {"no_overwrite", "smoke"},
        },
        "apply_data_cleaning": {
            "label": "生成正式 cleaned 直读 Qlib Provider",
            "command": apply_data_cleaning,
            "allowed_params": {"dry_run"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "data_cleaning_applied_latest.json"),
                str(REPORTS_ROOT / "data_cleaning_applied_latest.md"),
                str(REPORTS_ROOT / "data_cleaning_applied_*.json"),
                str(REPORTS_ROOT / "data_cleaning_applied_*.md"),
            ],
        },
        "repair_qlib_provider": {
            "label": "预演旧兼容 Qlib 修复",
            "command": repair_qlib_provider,
            "allowed_params": set(),
        },
        "check_qlib_data": {
            "label": "检查 Qlib 数据",
            "command": _script_command("check_qlib_data.py"),
            "allowed_params": set(),
        },
        "generate_csi300": {
            "label": "预演旧兼容 CSI300 股票池",
            "command": generate_csi300,
            "allowed_params": set(),
        },
        "add_index_benchmark": {
            "label": "预演旧兼容 benchmark",
            "command": add_index_benchmark,
            "allowed_params": set(),
        },
        "report_data_quality": {
            "label": "生成数据质量报告",
            "command": report_data_quality,
            "allowed_params": {"overlap_sample", "merged_sample"},
            "artifact_patterns": list(data_quality_artifact_patterns),
        },
        "report_data_anomalies": {
            "label": "生成异常股票和清洗策略",
            "command": report_data_anomalies,
            "allowed_params": set(),
            "artifact_patterns": [
                str(REPORTS_ROOT / "data_anomaly_stocks_latest.csv"),
                str(REPORTS_ROOT / "data_anomaly_events_latest.csv"),
                str(REPORTS_ROOT / "data_cleaning_strategy_latest.md"),
                str(REPORTS_ROOT / "data_anomaly_stocks_*.csv"),
                str(REPORTS_ROOT / "data_anomaly_events_*.csv"),
                str(REPORTS_ROOT / "data_cleaning_strategy_*.md"),
            ],
        },
        "report_data_hygiene": {
            "label": "生成数据治理盘点和清理预案",
            "command": report_data_hygiene,
            "allowed_params": set(),
            "artifact_patterns": [
                str(REPORTS_ROOT / "data_hygiene_reports_latest.json"),
                str(REPORTS_ROOT / "data_hygiene_inventory_latest.md"),
                str(REPORTS_ROOT / "data_hygiene_inventory_latest.json"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_latest.md"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_latest.json"),
                str(REPORTS_ROOT / "data_hygiene_inventory_*.md"),
                str(REPORTS_ROOT / "data_hygiene_inventory_*.json"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_*.md"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_*.json"),
            ],
        },
        "report_workbench_capabilities": {
            "label": "生成工作台能力盘点",
            "command": report_workbench_capabilities,
            "allowed_params": set(),
            "artifact_patterns": [
                str(REPORTS_ROOT / "workbench_capability_inventory_latest.md"),
                str(REPORTS_ROOT / "workbench_capability_inventory_latest.json"),
            ],
        },
        "report_market_regime": {
            "label": "生成市场状态和现金仓位信号",
            "command": report_market_regime,
            "allowed_params": {"lookback_days", "min_symbols", "limit"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "market_regime_latest.json"),
                str(REPORTS_ROOT / "market_regime_latest.md"),
                str(REPORTS_ROOT / "market_regime_*.json"),
                str(REPORTS_ROOT / "market_regime_*.md"),
            ],
        },
        "report_cash_control_backtest": {
            "label": "生成现金仓位叠加回测",
            "command": report_cash_control_backtest,
            "allowed_params": {"lookback_days", "min_symbols", "limit"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "cash_control_backtest_latest.json"),
                str(REPORTS_ROOT / "cash_control_backtest_latest.md"),
                str(REPORTS_ROOT / "cash_control_backtest_*.json"),
                str(REPORTS_ROOT / "cash_control_backtest_*.md"),
            ],
        },
        "report_qlib_predictions": {
            "label": "生成 Qlib 预测摘要",
            "command": report_qlib_predictions,
            "allowed_params": {"run_id", "topk"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "qlib_predictions_latest.json"),
                str(REPORTS_ROOT / "qlib_predictions_latest.md"),
                str(REPORTS_ROOT / "qlib_predictions_*.json"),
                str(REPORTS_ROOT / "qlib_predictions_*.md"),
            ],
        },
        "report_qlib_model_explainability": {
            "label": "生成 Qlib 选股模型解释报告",
            "command": lambda params: report_qlib_research("model_explainability", params),
            "allowed_params": {"run_id", "topk"},
            "artifact_patterns": qlib_research_artifacts("model_explainability"),
        },
        "report_qlib_backtest_detail": {
            "label": "生成 Qlib 回测明细报告",
            "command": lambda params: report_qlib_research("backtest_detail", params),
            "allowed_params": {"run_id"},
            "artifact_patterns": qlib_research_artifacts("backtest_detail"),
        },
        "report_qlib_model_comparison": {
            "label": "生成 Qlib 多选股模型/多实验对比",
            "command": lambda params: report_qlib_research("model_comparison", params),
            "allowed_params": {"max_runs"},
            "artifact_patterns": qlib_research_artifacts("model_comparison"),
        },
        "report_qlib_factor_analysis": {
            "label": "生成 Qlib 因子分析报告",
            "command": lambda params: report_qlib_research("factor_analysis", params),
            "allowed_params": {"run_id", "topk"},
            "artifact_patterns": qlib_research_artifacts("factor_analysis"),
        },
        "report_qlib_rolling_training": {
            "label": "生成 Qlib 滚动训练计划",
            "command": lambda params: report_qlib_research("rolling_training", params),
            "allowed_params": {
                "config",
                "windows",
                "train_years",
                "valid_months",
                "test_months",
                "step_months",
            },
            "artifact_patterns": qlib_research_artifacts("rolling_training"),
        },
        "report_qlib_risk_model": {
            "label": "生成 Qlib 组合风险诊断",
            "command": lambda params: report_qlib_research("risk_model", params),
            "allowed_params": {"run_id"},
            "artifact_patterns": qlib_research_artifacts("risk_model"),
        },
        "run_china_astock_comparison": {
            "label": "生成 china-astock 策略对比",
            "command": _script_command("run_china_astock_comparison.py"),
            "allowed_params": set(),
            "artifact_patterns": [
                str(REPORTS_ROOT / "china_astock_strategy_comparison_latest.csv"),
                str(REPORTS_ROOT / "china_astock_strategy_comparison_latest.md"),
                str(REPORTS_ROOT / "china_astock_strategy_comparison_latest.json"),
                str(REPORTS_ROOT / "china_astock_strategy_comparison_*.csv"),
                str(REPORTS_ROOT / "china_astock_strategy_comparison_*.md"),
            ],
        },
        "report_china_astock_indicators": {
            "label": "生成 china-astock 技术指标报告",
            "command": report_china_astock_indicators,
            "allowed_params": {"symbols", "lookback_days", "limit"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "china_astock_indicator_report_latest.json"),
                str(REPORTS_ROOT / "china_astock_indicator_report_latest.md"),
                str(REPORTS_ROOT / "china_astock_indicator_report_*.json"),
                str(REPORTS_ROOT / "china_astock_indicator_report_*.md"),
            ],
        },
        "optimize_strategy": {
            "label": "交易策略优化预演",
            "command": optimize_strategy,
            "allowed_params": {"dry_run", "phase", "topk"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "optimization_results_latest.json"),
                str(REPORTS_ROOT / "optimization_results_latest.md"),
                str(REPORTS_ROOT / "optimization_results_*.json"),
                str(REPORTS_ROOT / "optimization_results_*.md"),
            ],
        },
        "train_baseline_lgb": {
            "label": "训练 Qlib LightGBM 选股模型基线",
            "command": train_baseline_lgb,
            "allowed_params": {"config", "experiment_name", "provider"},
            "artifact_patterns": list(shared_training_artifact_patterns),
        },
        "train_baseline_lgb_equivalence": {
            "label": "运行真实 LightGBM 网页等价验收",
            "command": train_baseline_lgb_equivalence,
            "allowed_params": {"provider"},
            "artifact_patterns": [
                str(REPORTS_ROOT / ".training-runner-resource"),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "training_equivalence"
                    / "runs"
                    / "*"
                    / "training_result.json"
                ),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "training_equivalence"
                    / "latest_web.json"
                ),
            ],
        },
        "run_candidate_pipeline": {
            "label": "运行五个选股模型候选流程",
            "command": run_candidate_pipeline,
            "allowed_params": {"dry_run", "provider"},
            "artifact_patterns": [
                *shared_training_artifact_patterns,
                str(REPORTS_ROOT / "candidate_pipeline_latest.json"),
                str(REPORTS_ROOT / "candidate_pipeline_latest.md"),
                str(REPORTS_ROOT / "candidate_pipeline_progress_latest.json"),
                str(REPORTS_ROOT / "candidate_pipeline_*.json"),
                str(REPORTS_ROOT / "candidate_pipeline_*.md"),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_latest.json"
                ),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_latest.md"
                ),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_progress_latest.json"
                ),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_*.json"
                ),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_*.md"
                ),
            ],
            "progress_patterns": [
                str(REPORTS_ROOT / "candidate_pipeline_progress_latest.json"),
                str(
                    REPORTS_ROOT
                    / "test_runs"
                    / "candidate_pipeline"
                    / "candidate_pipeline_test_progress_latest.json"
                ),
            ],
        },
        "run_robust_strategy": {
            "label": "运行鲁棒策略开发流程",
            "command": run_robust_strategy,
            "allowed_params": {
                "as_of_date",
                "run_id",
                "resume",
                "skip_membership_build",
            },
            "artifact_patterns": [
                str(
                    REPORTS_ROOT
                    / "robust_strategy"
                    / "robust_strategy_status_latest.json"
                ),
                str(REPORTS_ROOT / "robust_strategy" / "preflight_latest.json"),
                str(REPORTS_ROOT / "robust_strategy" / "development_latest.json"),
                str(REPORTS_ROOT / "robust_strategy" / "development_latest.md"),
                str(REPORTS_ROOT / "robust_strategy" / "frozen_strategy_latest.json"),
            ],
        },
        "prepare_robust_signal_data": {
            "label": "验收并准备鲁棒 CSI800 数据",
            "command": prepare_robust_signal_data,
            "allowed_params": set(),
            "artifact_patterns": [
                str(
                    REPORTS_ROOT
                    / "robust_signal"
                    / "robust_signal_data_readiness_latest.json"
                ),
            ],
        },
        "run_robust_signal_research": {
            "label": "启动鲁棒选股信号研究",
            "command": run_robust_signal_research,
            "cancel_command": cancel_robust_signal_research,
            "cancel_result_contract": {
                "format": "json",
                "allowed_outcomes": ["cancelled", "already_terminal"],
            },
            "allowed_params": {"run_id", "smoke"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
                str(
                    REPORTS_ROOT
                    / "robust_signal"
                    / "robust_signal_smoke_status_latest.json"
                ),
                str(REPORTS_ROOT / "robust_signal" / "runs" / "*" / "result.json"),
            ],
            "progress_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
                str(
                    REPORTS_ROOT
                    / "robust_signal"
                    / "robust_signal_smoke_status_latest.json"
                ),
                str(REPORTS_ROOT / "robust_signal" / "runs" / "*" / "status.json"),
            ],
        },
        "preview_robust_signal_plan": {
            "label": "预览鲁棒选股信号研究计划",
            "command": preview_robust_signal_plan,
            "allowed_params": set(),
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_plan_latest.json"),
                str(REPORTS_ROOT / "robust_signal" / "plans" / "robust_signal_plan_*.json"),
            ],
        },
        "pause_robust_signal_research": {
            "label": "暂停鲁棒选股信号研究",
            "command": pause_robust_signal_research,
            "allowed_params": {"run_id"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
            ],
            "progress_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
            ],
        },
        "resume_robust_signal_research": {
            "label": "恢复鲁棒选股信号研究",
            "command": resume_robust_signal_research,
            "cancel_command": cancel_robust_signal_research,
            "cancel_result_contract": {
                "format": "json",
                "allowed_outcomes": ["cancelled", "already_terminal"],
            },
            "allowed_params": {"run_id", "smoke"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
            ],
            "progress_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
                str(REPORTS_ROOT / "robust_signal" / "runs" / "*" / "status.json"),
            ],
        },
        "run_robust_confirmation": {
            "label": "运行冻结策略 2026 确认期",
            "command": run_robust_confirmation,
            "allowed_params": {"run_id", "resume"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_strategy" / "confirmation_latest.json"),
                str(
                    REPORTS_ROOT / "robust_strategy" / "confirmation_state_latest.json"
                ),
            ],
        },
        "run_robust_shadow_day": {
            "label": "运行鲁棒策略当日纸面跟踪",
            "command": run_robust_shadow_day,
            "allowed_params": {"as_of_date"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_strategy" / "shadow_daily_latest.json"),
                str(REPORTS_ROOT / "robust_strategy" / "shadow_state_latest.json"),
            ],
        },
        "run_dsa_backtest": {
            "label": "运行 DSA 分析准确性回测",
            "command": run_dsa_backtest,
            "allowed_params": {"code", "eval_window_days", "force"},
            "cwd": str(DSA_ROOT),
        },
    }


class QuantJobService:
    """Create, persist, and supervise local quant workflow jobs."""

    def __init__(
        self,
        jobs_root: Path = DEFAULT_JOBS_ROOT,
        command_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        default_cwd: Path = DSA_ROOT,
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self.db_path = self.jobs_root / "index.sqlite"
        self.default_cwd = Path(default_cwd)
        self.command_specs = (
            command_specs
            if command_specs is not None
            else _build_default_command_specs()
        )
        self._processes: Dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._recover_running_jobs()

    def available_job_types(self) -> Dict[str, Dict[str, Any]]:
        return {
            job_type: {
                "label": str(spec.get("label") or job_type),
                "allowed_params": sorted(
                    str(item) for item in spec.get("allowed_params", [])
                ),
            }
            for job_type, spec in self.command_specs.items()
        }

    def submit_job(
        self,
        job_type: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        approved_provider_identity: Optional[Mapping[str, Any]] = None,
        approved_market_data_identity: Optional[Mapping[str, Any]] = None,
        approved_robust_signal_data: Optional[Mapping[str, Any]] = None,
    ) -> QuantJob:
        params_dict = dict(params or {})
        generated_robust_run_id: Optional[str] = None
        if job_type == "run_robust_signal_research" and not _safe_run_id(
            params_dict.get("run_id")
        ):
            generated_robust_run_id = (
                f"robust_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}"
            )
        spec = self._get_spec(job_type)
        provider_pinned_training = job_type in PROVIDER_PINNED_TRAINING_JOB_TYPES and not (
            job_type == "run_candidate_pipeline"
            and _as_bool(params_dict.get("dry_run"))
        )
        if provider_pinned_training and approved_provider_identity is None:
            raise ValueError(
                "Provider-pinned training requires an approved provider identity."
            )
        market_data_pinned = job_type in MARKET_DATA_PINNED_JOB_TYPES
        if market_data_pinned and approved_market_data_identity is None:
            raise ValueError(
                "Market report jobs require an approved market data identity."
            )
        robust_signal_data_pinned = job_type in ROBUST_SIGNAL_DATA_PINNED_JOB_TYPES
        if robust_signal_data_pinned and approved_robust_signal_data is None:
            raise ValueError(
                "Robust signal research requires an approved frozen readiness artifact."
            )
        command = self._build_command(job_type, params_dict)
        if generated_robust_run_id is not None:
            params_dict["run_id"] = generated_robust_run_id
            command.extend(["--run-id", generated_robust_run_id])
        if job_type == "run_candidate_pipeline":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_dict["batch_id"] = f"candidate-alpha158-{timestamp}"
            command.extend(["--timestamp", timestamp])
        if approved_provider_identity is not None:
            if job_type not in PROVIDER_PINNED_TRAINING_JOB_TYPES:
                raise ValueError(
                    "Approved provider identity is only valid for provider-pinned training jobs."
                )
            provider = _safe_provider(params_dict.get("provider"))
            command.extend(
                [
                    "--provider-identity-json",
                    _approved_provider_identity_json(
                        provider, approved_provider_identity
                    ),
                ]
            )
        if approved_market_data_identity is not None:
            if not market_data_pinned:
                raise ValueError(
                    "Approved market data identity is only valid for pinned market report jobs."
                )
            bars_path = _approved_market_data_bars_path(
                approved_market_data_identity
            )
            data_flag = (
                "--data-dir"
                if job_type == "report_market_regime"
                else "--market-data-dir"
            )
            command.extend([data_flag, str(bars_path)])
        if approved_robust_signal_data is not None:
            if not robust_signal_data_pinned:
                raise ValueError(
                    "Approved robust signal data is only valid for robust signal jobs."
                )
            readiness_artifact, readiness_sha256 = _approved_robust_signal_data_args(
                approved_robust_signal_data
            )
            command.extend(
                [
                    "--readiness-artifact",
                    str(readiness_artifact),
                    "--readiness-artifact-sha256",
                    readiness_sha256,
                ]
            )
            params_dict["readiness_artifact_sha256"] = readiness_sha256
        with self._lock:
            with self._connect() as conn:
                # Serialize the conflict check and pending-row claim across all
                # DSA processes sharing this SQLite job index.
                conn.execute("BEGIN IMMEDIATE")
                if job_type in QUANT_DATA_NETWORK_JOB_TYPES:
                    active = self._active_quant_data_network_job(conn=conn)
                    if active is not None:
                        raise ValueError(
                            "A quant_data network job is already active: "
                            f"{active.job_type} ({active.job_id}). Wait for it to finish or cancel it first."
                        )
                artifact_conflict = self._active_artifact_output_conflict(
                    job_type, conn=conn
                )
                if artifact_conflict is not None:
                    raise ValueError(
                        "Artifact output conflicts with an active quant job: "
                        f"{artifact_conflict.job_type} ({artifact_conflict.job_id}). "
                        "Wait for it to finish or cancel it first."
                    )

                job_id = f"qj_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
                job_dir = self.jobs_root / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / "job.log"
                timestamp = _now()
                job = QuantJob(
                    job_id=job_id,
                    job_type=job_type,
                    label=str(spec.get("label") or job_type),
                    status="pending",
                    params=params_dict,
                    command=command,
                    cwd=str(spec.get("cwd") or self.default_cwd),
                    created_at=timestamp,
                    updated_at=timestamp,
                    log_path=str(log_path),
                )
                self._insert_job(job, conn=conn)
            self._start_job(job)
            return self.get_job(job_id)

    def get_job(self, job_id: str) -> QuantJob:
        row = self._fetch_one("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if row is None:
            raise KeyError(job_id)
        return self._row_to_job(row)

    def list_jobs(
        self, limit: int = 50, job_type: Optional[str] = None
    ) -> List[QuantJob]:
        bounded_limit = max(1, min(int(limit), 200))
        if job_type:
            rows = self._fetch_all(
                "SELECT * FROM jobs WHERE job_type = ? "
                "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ?",
                (job_type, bounded_limit),
            )
        else:
            rows = self._fetch_all(
                "SELECT * FROM jobs "
                "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ?",
                (bounded_limit,),
            )
        return [self._row_to_job(row) for row in rows]

    def list_jobs_for_monitoring(self, history_limit: int = 20) -> List[QuantJob]:
        """Return every active job plus a bounded amount of recent history."""
        bounded_limit = max(1, min(int(history_limit), 200))
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN")
                rows = self._monitoring_job_rows(conn, bounded_limit)
        return [self._row_to_job(row) for row in rows]

    def _monitoring_job_rows(
        self, conn: sqlite3.Connection, history_limit: int
    ) -> List[sqlite3.Row]:
        active_rows = list(
            conn.execute(
                "SELECT rowid AS _rowid, * FROM jobs "
                "WHERE status IN ('pending', 'running') "
                "ORDER BY updated_at DESC, created_at DESC, rowid DESC",
                (),
            ).fetchall()
        )
        history_rows = list(
            conn.execute(
                "SELECT rowid AS _rowid, * FROM jobs "
                "WHERE status NOT IN ('pending', 'running') "
                "ORDER BY updated_at DESC, created_at DESC, rowid DESC LIMIT ?",
                (history_limit,),
            ).fetchall()
        )
        return sorted(
            [*active_rows, *history_rows],
            key=lambda row: (
                str(row["updated_at"]),
                str(row["created_at"]),
                int(row["_rowid"]),
            ),
            reverse=True,
        )

    def get_latest_jobs_by_type(self) -> Dict[str, QuantJob]:
        rows = self._fetch_all(
            """
            SELECT j.*
            FROM jobs j
            INNER JOIN (
                SELECT job_type, MAX(rowid) AS max_rowid
                FROM jobs
                GROUP BY job_type
            ) latest
            ON latest.max_rowid = j.rowid
            ORDER BY j.created_at DESC, j.rowid DESC
            """,
            (),
        )
        return {row["job_type"]: self._row_to_job(row) for row in rows}

    def active_count(self) -> int:
        row = self._fetch_one(
            "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('pending', 'running')",
            (),
        )
        return int(row["count"]) if row is not None else 0

    def get_status_snapshot(
        self, recent_limit: int = 10
    ) -> tuple[Dict[str, QuantJob], int, List[QuantJob]]:
        bounded_limit = max(1, min(int(recent_limit), 200))
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN")
                latest_rows = list(
                    conn.execute(
                        """
                        SELECT j.*
                        FROM jobs j
                        INNER JOIN (
                            SELECT job_type, MAX(rowid) AS max_rowid
                            FROM jobs
                            GROUP BY job_type
                        ) latest
                        ON latest.max_rowid = j.rowid
                        ORDER BY j.created_at DESC, j.rowid DESC
                        """,
                        (),
                    ).fetchall()
                )
                active_row = conn.execute(
                    "SELECT COUNT(*) AS count FROM jobs "
                    "WHERE status IN ('pending', 'running')",
                    (),
                ).fetchone()
                recent_rows = self._monitoring_job_rows(conn, bounded_limit)
        latest_jobs = {
            row["job_type"]: self._row_to_job(row) for row in latest_rows
        }
        active_count = int(active_row["count"]) if active_row is not None else 0
        recent_jobs = [self._row_to_job(row) for row in recent_rows]
        return latest_jobs, active_count, recent_jobs

    def _active_quant_data_network_job(
        self, *, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[QuantJob]:
        placeholders = ", ".join("?" for _item in QUANT_DATA_NETWORK_JOB_TYPES)
        query = f"""
            SELECT * FROM jobs
            WHERE status IN ('pending', 'running')
              AND job_type IN ({placeholders})
            ORDER BY rowid DESC
            LIMIT 1
            """
        params = tuple(sorted(QUANT_DATA_NETWORK_JOB_TYPES))
        row = (
            conn.execute(query, params).fetchone()
            if conn is not None
            else self._fetch_one(query, params)
        )
        return self._row_to_job(row) if row is not None else None

    def _active_artifact_output_conflict(
        self,
        job_type: str,
        *,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[QuantJob]:
        requested_patterns = self._get_spec(job_type).get("artifact_patterns", [])
        if not requested_patterns:
            return None
        query = (
            "SELECT * FROM jobs WHERE status IN ('pending', 'running') "
            "ORDER BY rowid DESC"
        )
        rows = (
            list(conn.execute(query, ()).fetchall())
            if conn is not None
            else self._fetch_all(query, ())
        )
        for row in rows:
            active = self._row_to_job(row)
            active_spec = self.command_specs.get(active.job_type, {})
            active_patterns = active_spec.get("artifact_patterns", [])
            if active_patterns and _artifact_pattern_sets_overlap(
                requested_patterns,
                active_patterns,
            ):
                return active
        return None

    def get_log(self, job_id: str, max_bytes: int = 20000) -> Dict[str, Any]:
        job = self.get_job(job_id)
        log_path = Path(job.log_path)
        if not log_path.exists():
            return {
                "job_id": job_id,
                "content": "",
                "truncated": False,
                "log_path": str(log_path),
            }

        size = log_path.stat().st_size
        read_size = max(1, min(int(max_bytes), 200000))
        with log_path.open("rb") as handle:
            if size > read_size:
                handle.seek(size - read_size)
            content = handle.read().decode("utf-8", errors="replace")
        return {
            "job_id": job_id,
            "content": content,
            "truncated": size > read_size,
            "log_path": str(log_path),
        }

    def cancel_job(self, job_id: str) -> QuantJob:
        job = self._claim_job_cancellation(job_id)
        if job.status in TERMINAL_STATUSES and not job.can_cancel:
            return job
        with self._lock:
            process = self._processes.get(job_id)

        durable_runner_stopped = False
        job_spec = self.command_specs.get(job.job_type, {})
        cancel_spec = job_spec.get("cancel_command")
        if cancel_spec is not None:
            try:
                if callable(cancel_spec):
                    cancel_command = [str(part) for part in cancel_spec(job.params)]
                elif isinstance(cancel_spec, Iterable) and not isinstance(
                    cancel_spec, (str, bytes)
                ):
                    cancel_command = [str(part) for part in cancel_spec]
                else:
                    raise ValueError("cancel_command is not a trusted command sequence")
                if not cancel_command:
                    raise ValueError("cancel_command produced an empty command")
                completed = subprocess.run(
                    cancel_command,
                    cwd=job.cwd or None,
                    env=_quant_job_child_env(
                        {},
                        allowed_names=job_spec.get("env_names", ()),
                        allowed_prefixes=job_spec.get("env_prefixes", ()),
                    ),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=max(30, JOB_TERMINATE_TIMEOUT_SECONDS * 4),
                )
                if int(completed.returncode) != 0:
                    raise RuntimeError(
                        "durable runner cancellation command did not confirm shutdown"
                    )
                cancel_result_contract = job_spec.get("cancel_result_contract")
                if cancel_result_contract is not None:
                    if not isinstance(cancel_result_contract, Mapping):
                        raise ValueError("cancel_result_contract is not a mapping")
                    cancel_outcome = _cancel_command_outcome(
                        completed.stdout,
                        cancel_result_contract,
                    )
                    if cancel_outcome == "already_terminal":
                        return self.get_job(job_id)
                    if cancel_outcome != "cancelled":
                        raise ValueError("cancel command outcome did not confirm cancellation")
                durable_runner_stopped = True
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                timestamp = _now()
                self._update_job(
                    job_id,
                    updated_at=timestamp,
                    error=(
                        "Cancellation was not confirmed; the durable runner may still "
                        f"be active ({type(exc).__name__})."
                    ),
                )
                return self.get_job(job_id)

        if process is not None and process.poll() is None:
            process_pid = getattr(process, "pid", job.pid)
            if hasattr(process, "terminate"):
                process.terminate()
            self._terminate_process_group(process_pid, signal.SIGTERM)
            deadline = time.time() + JOB_TERMINATE_TIMEOUT_SECONDS
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            if process.poll() is None:
                self._terminate_process_group(process_pid, signal.SIGKILL)
                if hasattr(process, "kill"):
                    process.kill()
        elif job.pid:
            if not self._runner_identity_matches(job):
                timestamp = _now()
                self._update_job(
                    job_id,
                    status="failed",
                    completed_at=timestamp,
                    updated_at=timestamp,
                    error=(
                        "Recovered runner identity no longer matches; no process was "
                        "signaled during cancellation."
                    ),
                )
                return self.get_job(job_id)
            self._terminate_process_group(job.pid, signal.SIGTERM)
            deadline = time.time() + JOB_TERMINATE_TIMEOUT_SECONDS
            while time.time() < deadline and self._pid_is_alive(job.pid):
                time.sleep(0.1)
            if self._pid_is_alive(job.pid):
                self._terminate_process_group(job.pid, signal.SIGKILL)

        with self._lock:
            current = self.get_job(job_id)
            if current.status == "failed" and _return_code_indicates_cancel(
                current.return_code
            ):
                timestamp = _now()
                self._update_job(
                    job_id,
                    status="cancelled",
                    completed_at=current.completed_at or timestamp,
                    updated_at=timestamp,
                    error="Cancelled by user from the web UI.",
                )
                return self.get_job(job_id)
            if current.status in TERMINAL_STATUSES and not (
                durable_runner_stopped and current.status != "cancelled"
            ):
                return current
            timestamp = _now()
            self._update_job(
                job_id,
                status="cancelled",
                completed_at=timestamp,
                updated_at=timestamp,
                error="Cancelled by user from the web UI.",
            )
            return self.get_job(job_id)

    def _get_spec(self, job_type: str) -> Dict[str, Any]:
        spec = self.command_specs.get(job_type)
        if spec is None:
            raise ValueError(f"Unsupported quant job type: {job_type}")
        return spec

    def _build_command(self, job_type: str, params: Mapping[str, Any]) -> List[str]:
        spec = self._get_spec(job_type)
        allowed_params = spec.get("allowed_params")
        if allowed_params is not None:
            unknown = sorted(set(params) - set(allowed_params))
            if unknown:
                raise ValueError(
                    f"Unsupported params for {job_type}: {', '.join(unknown)}"
                )
        _validate_numeric_params(params)
        _validate_ashare_source_safety(job_type, params)

        command_spec = spec.get("command")
        if callable(command_spec):
            command = list(command_spec(params))
        elif isinstance(command_spec, Iterable) and not isinstance(
            command_spec, (str, bytes)
        ):
            command = [str(part) for part in command_spec]
        else:
            raise ValueError(f"Job type {job_type} has no runnable command.")
        if not command:
            raise ValueError(f"Job type {job_type} produced an empty command.")
        return command

    @staticmethod
    def _row_runner_identity(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            payload = json.loads(str(row["runner_identity_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _claim_job_cancellation(self, job_id: str) -> QuantJob:
        timestamp = _now()
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(job_id)
                current = self._row_to_job(row)
                if current.status == "pending":
                    conn.execute(
                        "UPDATE jobs SET status = ?, completed_at = ?, updated_at = ?, "
                        "error = ? WHERE job_id = ? AND status = 'pending'",
                        (
                            "cancelled",
                            timestamp,
                            timestamp,
                            "Cancelled by user from the web UI.",
                            job_id,
                        ),
                    )
                elif current.status == "running":
                    runner_identity = self._row_runner_identity(row)
                    runner_identity.update(
                        {
                            "cancel_requested": True,
                            "cancel_requested_at": timestamp,
                        }
                    )
                    conn.execute(
                        "UPDATE jobs SET runner_identity_json = ?, updated_at = ? "
                        "WHERE job_id = ? AND status = 'running'",
                        (_json_dumps(runner_identity), timestamp, job_id),
                    )
                conn.commit()
                claimed_row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
        if claimed_row is None:  # pragma: no cover - protected by the transaction.
            raise KeyError(job_id)
        return self._row_to_job(claimed_row)

    def _claim_pending_job_start(
        self,
        job_id: str,
        *,
        started_at: str,
        starting_identity: Mapping[str, Any],
    ) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    "UPDATE jobs SET status = 'running', started_at = ?, updated_at = ?, "
                    "runner_identity_json = ? WHERE job_id = ? AND status = 'pending'",
                    (
                        started_at,
                        started_at,
                        _json_dumps(dict(starting_identity)),
                        job_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1

    def _record_starting_runner_pid(
        self,
        job_id: str,
        *,
        launch_token: str,
        runner_identity: Mapping[str, Any],
    ) -> bool:
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status, runner_identity_json FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                current_identity = (
                    self._row_runner_identity(row) if row is not None else {}
                )
                if (
                    row is None
                    or row["status"] != "running"
                    or current_identity.get("launch_token") != launch_token
                    or current_identity.get("handshake_status") != "starting"
                    or current_identity.get("cancel_requested") is True
                ):
                    conn.rollback()
                    return False
                cursor = conn.execute(
                    "UPDATE jobs SET pid = ?, runner_identity_json = ?, updated_at = ? "
                    "WHERE job_id = ? AND status = 'running'",
                    (
                        int(runner_identity["pid"]),
                        _json_dumps(dict(runner_identity)),
                        _now(),
                        job_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1

    def _authorize_runner_start(
        self,
        job: QuantJob,
        *,
        launch_token: str,
        confirmed_identity: Mapping[str, Any],
    ) -> bool:
        permission_path = self._start_permission_path(job)
        temporary = permission_path.with_name(
            f".{permission_path.name}.{launch_token}.tmp"
        )
        permission_payload = {
            "schema_version": 1,
            "job_id": job.job_id,
            "launch_token": launch_token,
            "job_command_sha256": _command_sha256(job.command),
            "pid": int(confirmed_identity["pid"]),
        }
        authorized = False
        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT status, pid, runner_identity_json FROM jobs "
                        "WHERE job_id = ?",
                        (job.job_id,),
                    ).fetchone()
                    current_identity = (
                        self._row_runner_identity(row) if row is not None else {}
                    )
                    if (
                        row is None
                        or row["status"] != "running"
                        or row["pid"] != int(confirmed_identity["pid"])
                        or current_identity.get("launch_token") != launch_token
                        or current_identity.get("handshake_status") != "starting"
                        or current_identity.get("cancel_requested") is True
                    ):
                        conn.rollback()
                        return False
                    final_identity = {
                        **dict(confirmed_identity),
                        "handshake_status": "confirmed",
                        "start_authorized_at": _now(),
                    }
                    with temporary.open("w", encoding="utf-8") as handle:
                        json.dump(
                            permission_payload,
                            handle,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    conn.execute(
                        "UPDATE jobs SET pid = ?, runner_identity_json = ?, "
                        "updated_at = ? WHERE job_id = ? AND status = 'running'",
                        (
                            int(confirmed_identity["pid"]),
                            _json_dumps(final_identity),
                            _now(),
                            job.job_id,
                        ),
                    )
                    os.replace(temporary, permission_path)
                    conn.commit()
                    authorized = True
            return True
        finally:
            temporary.unlink(missing_ok=True)
            if not authorized:
                permission_path.unlink(missing_ok=True)

    def _mark_start_failed_if_current(
        self,
        job_id: str,
        launch_token: str,
        error: str,
    ) -> bool:
        timestamp = _now()
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT status, runner_identity_json FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                current_identity = (
                    self._row_runner_identity(row) if row is not None else {}
                )
                if (
                    row is None
                    or row["status"] != "running"
                    or current_identity.get("launch_token") != launch_token
                    or current_identity.get("cancel_requested") is True
                ):
                    conn.rollback()
                    return False
                failed_identity = {
                    **current_identity,
                    "handshake_status": "failed",
                }
                cursor = conn.execute(
                    "UPDATE jobs SET status = 'failed', completed_at = ?, "
                    "updated_at = ?, error = ?, runner_identity_json = ? "
                    "WHERE job_id = ? AND status = 'running'",
                    (
                        timestamp,
                        timestamp,
                        error,
                        _json_dumps(failed_identity),
                        job_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1

    def _start_job(self, job: QuantJob) -> None:
        started_at = _now()
        launch_token = uuid.uuid4().hex
        startup_deadline_epoch = time.time() + RUNNER_IDENTITY_STARTUP_LEASE_SECONDS
        starting_identity = {
            "schema_version": 1,
            "handshake_status": "starting",
            "launch_token": launch_token,
            "pid": None,
            "job_command_sha256": _command_sha256(job.command),
            "startup_deadline_epoch": startup_deadline_epoch,
        }
        if not self._claim_pending_job_start(
            job.job_id,
            started_at=started_at,
            starting_identity=starting_identity,
        ):
            return
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        runner_identity_path = self._runner_identity_path(job)
        start_permission_path = self._start_permission_path(job)
        runner_identity_path.unlink(missing_ok=True)
        start_permission_path.unlink(missing_ok=True)
        handle: Any = None
        try:
            self._write_artifact_baseline(job)
            handle = log_path.open("ab")
            header = (
                f"[job] {job.job_id} {job.job_type}\n"
                f"[cwd] {job.cwd}\n"
                # Keep secrets in the child environment only; command arguments
                # are intentionally persisted to this user-visible audit log.
                f"[command] {' '.join(job.command)}\n\n"
            )
            handle.write(header.encode("utf-8"))
            handle.flush()
            spec = self._get_spec(job.job_type)
            env = _quant_job_child_env(
                {
                    "QUANT_JOB_COMMAND_JSON": _json_dumps(job.command),
                    "QUANT_JOB_CWD": job.cwd,
                    "QUANT_JOB_EXIT_STATUS_PATH": str(self._exit_status_path(job)),
                    "QUANT_JOB_RUNNER_IDENTITY_PATH": str(runner_identity_path),
                    "QUANT_JOB_START_PERMISSION_PATH": str(start_permission_path),
                    "QUANT_JOB_LAUNCH_TOKEN": launch_token,
                    "QUANT_JOB_STARTUP_DEADLINE_EPOCH": str(startup_deadline_epoch),
                },
                allowed_names=spec.get("env_names", ()),
                allowed_prefixes=spec.get("env_prefixes", ()),
            )
            process = subprocess.Popen(
                [str(Path(sys.executable)), "-c", _DURABLE_RUNNER_CODE],
                cwd=job.cwd or None,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface launch failures to UI.
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                f"Failed to start job: {exc}",
            )
            return

        process_identity = self._process_runtime_identity(process.pid) or {}
        runner_identity = {
            **starting_identity,
            "pid": process.pid,
            "process_start": process_identity.get("process_start"),
            "process_command_sha256": process_identity.get(
                "process_command_sha256"
            ),
        }
        try:
            pid_recorded = self._record_starting_runner_pid(
                job.job_id,
                launch_token=launch_token,
                runner_identity=runner_identity,
            )
        except Exception as exc:  # noqa: BLE001 - convert startup faults to job state.
            self._stop_unconfirmed_runner(process)
            try:
                handle.close()
            except Exception:
                pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                f"Durable runner PID registration failed ({type(exc).__name__}).",
            )
            return
        if not pid_recorded:
            self._stop_unconfirmed_runner(process)
            try:
                handle.close()
            except Exception:
                pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                "Durable runner PID registration no longer matched the active start.",
            )
            return
        started_job = replace(
            job,
            status="running",
            pid=process.pid,
            started_at=started_at,
            updated_at=_now(),
            runner_identity=runner_identity,
        )
        confirmed_identity = self._await_runner_identity(started_job, process)
        if confirmed_identity is None:
            self._stop_unconfirmed_runner(process)
            try:
                handle.close()
            except Exception:
                pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                (
                    "Durable runner identity handshake was not confirmed; the exact "
                    "newly started runner was stopped."
                ),
            )
            return

        try:
            start_authorized = self._authorize_runner_start(
                job,
                launch_token=launch_token,
                confirmed_identity=confirmed_identity,
            )
        except Exception as exc:  # noqa: BLE001 - convert startup faults to job state.
            self._stop_unconfirmed_runner(process)
            try:
                handle.close()
            except Exception:
                pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                f"Durable runner start authorization failed ({type(exc).__name__}).",
            )
            return
        if not start_authorized:
            self._stop_unconfirmed_runner(process)
            try:
                handle.close()
            except Exception:
                pass
            self._mark_start_failed_if_current(
                job.job_id,
                launch_token,
                "Durable runner start authorization no longer matched the active start.",
            )
            return
        with self._lock:
            self._processes[job.job_id] = process
        thread = threading.Thread(
            target=self._wait_for_process,
            args=(job.job_id, process, handle),
            name=f"quant-job-{job.job_id}",
            daemon=True,
        )
        thread.start()

    def _exit_status_path(self, job: QuantJob) -> Path:
        return Path(job.log_path).parent / EXIT_STATUS_FILENAME

    def _runner_identity_path(self, job: QuantJob) -> Path:
        return Path(job.log_path).parent / RUNNER_IDENTITY_FILENAME

    def _start_permission_path(self, job: QuantJob) -> Path:
        return Path(job.log_path).parent / START_PERMISSION_FILENAME

    @staticmethod
    def _process_runtime_identity(pid: Optional[int]) -> Optional[Dict[str, str]]:
        if not pid:
            return None
        try:
            completed = subprocess.run(
                ["ps", "-ww", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
                text=True,
                capture_output=True,
                check=False,
                timeout=2,
            )
        except (OSError, TypeError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        parts = completed.stdout.strip().split(None, 5)
        if len(parts) != 6:
            return None
        process_start = " ".join(parts[:5])
        process_command = parts[5].strip()
        if not process_command:
            return None
        return {
            "process_start": process_start,
            "process_command_sha256": hashlib.sha256(
                process_command.encode("utf-8")
            ).hexdigest(),
        }

    def _read_runner_identity(self, job: QuantJob) -> Optional[Dict[str, Any]]:
        path = self._runner_identity_path(job)
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _confirmed_runner_identity(
        self,
        job: QuantJob,
        *,
        direct_process: Optional[subprocess.Popen[bytes]] = None,
    ) -> Optional[Dict[str, Any]]:
        expected = job.runner_identity
        if (
            not isinstance(expected, dict)
            or expected.get("schema_version") != 1
            or not re.fullmatch(r"[0-9a-f]{32}", str(expected.get("launch_token") or ""))
            or expected.get("job_command_sha256") != _command_sha256(job.command)
        ):
            return None
        started = self._read_runner_identity(job)
        started_pid = started.get("pid") if isinstance(started, dict) else None
        if (
            not isinstance(started, dict)
            or started.get("schema_version") != 1
            or started.get("launch_token") != expected.get("launch_token")
            or started.get("job_command_sha256") != expected.get("job_command_sha256")
            or isinstance(started_pid, bool)
            or not isinstance(started_pid, int)
            or started_pid <= 0
            or (job.pid is not None and started_pid != job.pid)
            or (
                expected.get("pid") is not None
                and started_pid != expected.get("pid")
            )
            or (
                direct_process is not None
                and started_pid != getattr(direct_process, "pid", None)
            )
        ):
            return None

        current = self._process_runtime_identity(started_pid)
        if current is None:
            if direct_process is None or direct_process.poll() is None:
                return None
            current = {
                "process_start": expected.get("process_start"),
                "process_command_sha256": expected.get("process_command_sha256"),
            }
        for key in ("process_start", "process_command_sha256"):
            expected_value = expected.get(key)
            if expected_value and current.get(key) != expected_value:
                return None
        return {
            **expected,
            "handshake_status": "confirmed",
            "pid": started_pid,
            "process_start": current.get("process_start"),
            "process_command_sha256": current.get("process_command_sha256"),
        }

    def _await_runner_identity(
        self,
        job: QuantJob,
        process: subprocess.Popen[bytes],
        *,
        timeout_seconds: float = RUNNER_IDENTITY_HANDSHAKE_TIMEOUT_SECONDS,
        poll_interval_seconds: float = RUNNER_IDENTITY_HANDSHAKE_POLL_SECONDS,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            confirmed = self._confirmed_runner_identity(
                job, direct_process=process
            )
            if confirmed is not None:
                return confirmed
            if process.poll() is not None:
                # The runner writes identity atomically before it exits. A fast
                # command can finish between the first read and poll(), so do
                # one final strict read before declaring the handshake failed.
                return self._confirmed_runner_identity(
                    job, direct_process=process
                )
            if time.monotonic() >= deadline:
                return None
            if poll_interval_seconds > 0:
                time.sleep(poll_interval_seconds)

    def _stop_unconfirmed_runner(self, process: subprocess.Popen[bytes]) -> None:
        process_pid = getattr(process, "pid", None)
        try:
            process.terminate()
        except (AttributeError, OSError):
            pass
        self._terminate_process_group(process_pid, signal.SIGTERM)
        try:
            process.wait(timeout=JOB_TERMINATE_TIMEOUT_SECONDS)
            return
        except (AttributeError, OSError, subprocess.TimeoutExpired, TypeError):
            pass
        self._terminate_process_group(process_pid, signal.SIGKILL)
        try:
            process.kill()
        except (AttributeError, OSError):
            pass

    def _ownerless_startup_lease_expired(self, job: QuantJob) -> bool:
        expected = job.runner_identity
        if (
            job.status not in {"pending", "running"}
            or job.pid is not None
            or not isinstance(expected, dict)
            or expected.get("handshake_status") != "starting"
            or self._read_runner_identity(job) is not None
        ):
            return False
        raw_deadline = expected.get("startup_deadline_epoch")
        if isinstance(raw_deadline, bool):
            return True
        try:
            deadline = float(raw_deadline)
        except (TypeError, ValueError):
            return True
        return not math.isfinite(deadline) or time.time() >= deadline

    def _mark_ownerless_startup_lease_failed(self, job: QuantJob) -> None:
        timestamp = _now()
        self._update_job(
            job.job_id,
            status="failed",
            completed_at=timestamp,
            updated_at=timestamp,
            error=(
                "Durable runner startup lease expired before a PID or signed identity "
                "was recorded; the real command was not started and output ownership "
                "was released."
            ),
        )

    def _runner_identity_startup_pending(self, job: QuantJob) -> bool:
        expected = job.runner_identity
        if (
            job.status not in {"pending", "running"}
            or not isinstance(expected, dict)
            or expected.get("schema_version") != 1
            or expected.get("handshake_status") != "starting"
            or not re.fullmatch(r"[0-9a-f]{32}", str(expected.get("launch_token") or ""))
            or expected.get("job_command_sha256") != _command_sha256(job.command)
        ):
            return False
        identity_path = self._runner_identity_path(job)
        expected_pid = expected.get("pid")
        if expected_pid is not None and expected_pid != job.pid:
            return False
        if not identity_path.exists() and not identity_path.is_symlink():
            if job.pid is None:
                return not self._ownerless_startup_lease_expired(job)
            return self._pid_is_alive(job.pid)

        started = self._read_runner_identity(job)
        started_pid = started.get("pid") if isinstance(started, dict) else None
        if (
            not isinstance(started, dict)
            or started.get("schema_version") != 1
            or started.get("launch_token") != expected.get("launch_token")
            or started.get("job_command_sha256") != expected.get("job_command_sha256")
            or isinstance(started_pid, bool)
            or not isinstance(started_pid, int)
            or started_pid <= 0
            or (job.pid is not None and started_pid != job.pid)
            or (expected_pid is not None and started_pid != expected_pid)
            or not self._pid_is_alive(started_pid)
        ):
            return False
        current = self._process_runtime_identity(started_pid)
        if current is None:
            return True
        for key in ("process_start", "process_command_sha256"):
            expected_value = expected.get(key)
            if expected_value and current.get(key) != expected_value:
                return False
        return False

    def _confirm_starting_runner_identity(self, job: QuantJob) -> Optional[QuantJob]:
        if job.runner_identity.get("handshake_status") != "starting":
            return None
        confirmed = self._confirmed_runner_identity(job)
        if confirmed is None:
            return None
        self._update_job(
            job.job_id,
            pid=int(confirmed["pid"]),
            runner_identity_json=_json_dumps(confirmed),
            updated_at=_now(),
        )
        return self.get_job(job.job_id)

    def _runner_identity_matches(self, job: QuantJob) -> bool:
        confirmed = self._confirmed_runner_identity(job)
        return bool(
            confirmed
            and job.pid is not None
            and confirmed.get("pid") == job.pid
            and confirmed.get("process_start")
            and confirmed.get("process_command_sha256")
        )

    def _read_exit_status(self, job: QuantJob) -> Optional[Dict[str, Any]]:
        path = self._exit_status_path(job)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(
            payload.get("return_code"), int
        ):
            return None
        return payload

    def _pid_is_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _terminate_process_group(self, pid: Optional[int], sig: int) -> None:
        if not pid:
            return
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                pass

    def _complete_job_from_return_code(
        self,
        job: QuantJob,
        return_code: int,
        completed_at: str,
        runner_error: Optional[str] = None,
    ) -> None:
        with self._lock:
            current = self.get_job(job.job_id)
            artifact_paths = (
                self._collect_artifact_paths(current)
                if return_code == 0
                else current.artifact_paths
            )
            if current.status == "cancelled":
                updates: Dict[str, Any] = {
                    "return_code": return_code,
                    "updated_at": completed_at,
                }
                if artifact_paths:
                    updates["artifact_paths_json"] = _json_dumps(artifact_paths)
                self._update_job(job.job_id, **updates)
                return

            status = "completed" if return_code == 0 else "failed"
            error = (
                runner_error
                if runner_error
                else None
                if return_code == 0
                else f"Process exited with return code {return_code}."
            )
            updates = {
                "status": status,
                "return_code": return_code,
                "completed_at": completed_at,
                "updated_at": completed_at,
                "error": error,
            }
            if artifact_paths:
                updates["artifact_paths_json"] = _json_dumps(artifact_paths)
            self._update_job(job.job_id, **updates)

    def _wait_for_process(
        self, job_id: str, process: subprocess.Popen[bytes], handle: Any
    ) -> None:
        try:
            return_code = process.wait()
            completed_at = _now()
            current = self.get_job(job_id)
            payload = self._read_exit_status(current) or {}
            runner_error = (
                payload.get("error") if isinstance(payload.get("error"), str) else None
            )
            self._complete_job_from_return_code(
                current, return_code, completed_at, runner_error
            )
        finally:
            try:
                handle.close()
            finally:
                with self._lock:
                    self._processes.pop(job_id, None)

    def _artifact_baseline_path(self, job: QuantJob) -> Path:
        return Path(job.log_path).parent / ARTIFACT_BASELINE_FILENAME

    def _matching_artifact_states(self, job_type: str) -> Dict[str, List[int]]:
        patterns = self._get_spec(job_type).get("artifact_patterns", [])
        states: Dict[str, List[int]] = {}
        for pattern in patterns:
            for match in glob.glob(str(pattern)):
                path = Path(match)
                try:
                    metadata = path.stat()
                except OSError:
                    continue
                if path.is_file():
                    states[str(path)] = [
                        int(metadata.st_dev),
                        int(metadata.st_ino),
                        int(metadata.st_size),
                        int(metadata.st_mtime_ns),
                        int(metadata.st_ctime_ns),
                    ]
        return states

    def _write_artifact_baseline(self, job: QuantJob) -> None:
        path = self._artifact_baseline_path(job)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "schema_version": 1,
            "job_id": job.job_id,
            "artifacts": self._matching_artifact_states(job.job_type),
        }
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_artifact_baseline(self, job: QuantJob) -> Optional[Dict[str, List[int]]]:
        try:
            payload = json.loads(
                self._artifact_baseline_path(job).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        artifacts = payload.get("artifacts")
        if (
            payload.get("schema_version") != 1
            or payload.get("job_id") != job.job_id
            or not isinstance(artifacts, dict)
        ):
            return None
        normalized: Dict[str, List[int]] = {}
        for raw_path, raw_state in artifacts.items():
            if (
                isinstance(raw_path, str)
                and isinstance(raw_state, list)
                and len(raw_state) == 5
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_state
                )
            ):
                normalized[raw_path] = raw_state
            else:
                return None
        return normalized

    def _collect_artifact_paths(self, job: QuantJob) -> List[str]:
        baseline = self._read_artifact_baseline(job)
        if baseline is None:
            return []
        current = self._matching_artifact_states(job.job_type)
        paths = [
            Path(path)
            for path, state in current.items()
            if baseline.get(path) != state
        ]

        def sort_key(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        unique_paths: Dict[str, None] = {}
        for path in sorted(paths, key=sort_key, reverse=True):
            unique_paths.setdefault(str(path), None)
        return list(unique_paths)[:12]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    pid INTEGER,
                    return_code INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    log_path TEXT NOT NULL,
                    artifact_paths_json TEXT NOT NULL,
                    runner_identity_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "runner_identity_json" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN runner_identity_json TEXT NOT NULL DEFAULT '{}'"
                )
            conn.commit()

    def _recover_running_jobs(self) -> None:
        rows = self._fetch_all(
            "SELECT * FROM jobs WHERE status IN ('pending', 'running') ORDER BY created_at ASC, rowid ASC",
            (),
        )
        for row in rows:
            job = self._row_to_job(row)
            payload = self._read_exit_status(job)
            if payload is not None:
                runner_error = (
                    payload.get("error")
                    if isinstance(payload.get("error"), str)
                    else None
                )
                self._complete_job_from_return_code(
                    job, int(payload["return_code"]), _now(), runner_error
                )
                continue
            confirmed_job = self._confirm_starting_runner_identity(job)
            if confirmed_job is not None:
                job = confirmed_job
            if self._ownerless_startup_lease_expired(job):
                self._mark_ownerless_startup_lease_failed(job)
                continue
            if job.runner_alive is True:
                timestamp = _now()
                self._update_job(
                    job.job_id,
                    status="running",
                    updated_at=timestamp,
                    error="API service restarted while this durable runner was still running; monitoring exit status.",
                )
                self._start_recovery_monitor(job.job_id, job.pid)
                continue
            if self._runner_identity_startup_pending(job):
                timestamp = _now()
                self._update_job(
                    job.job_id,
                    status="running",
                    updated_at=timestamp,
                    error=(
                        "API service restarted while the durable runner identity "
                        "handshake was still starting; waiting for signed identity."
                    ),
                )
                self._start_recovery_monitor(job.job_id, job.pid)
                continue
            self._mark_job_failed_after_restart(job)

    def _start_recovery_monitor(self, job_id: str, pid: Optional[int]) -> None:
        thread = threading.Thread(
            target=self._poll_recovered_process,
            args=(job_id, pid),
            name=f"quant-job-recover-{job_id}",
            daemon=True,
        )
        thread.start()

    def _poll_recovered_process(
        self,
        job_id: str,
        pid: Optional[int],
        *,
        max_poll_seconds: float = RECOVERED_RUNNER_MAX_POLL_SECONDS,
        poll_interval_seconds: float = RECOVERED_RUNNER_POLL_INTERVAL_SECONDS,
    ) -> None:
        started_at = time.monotonic()
        handshake_started_at = started_at
        while True:
            try:
                job = self.get_job(job_id)
            except KeyError:
                return
            if job.status in TERMINAL_STATUSES:
                return
            payload = self._read_exit_status(job)
            if payload is not None:
                runner_error = (
                    payload.get("error")
                    if isinstance(payload.get("error"), str)
                    else None
                )
                self._complete_job_from_return_code(
                    job, int(payload["return_code"]), _now(), runner_error
                )
                return
            if job.runner_identity.get("handshake_status") == "starting":
                confirmed_job = self._confirm_starting_runner_identity(job)
                if confirmed_job is not None:
                    job = confirmed_job
                elif self._ownerless_startup_lease_expired(job):
                    self._mark_ownerless_startup_lease_failed(job)
                    return
                elif self._runner_identity_startup_pending(job):
                    if (
                        time.monotonic() - handshake_started_at
                        >= RUNNER_IDENTITY_HANDSHAKE_TIMEOUT_SECONDS
                    ):
                        timestamp = _now()
                        self._update_job(
                            job_id,
                            status="running",
                            completed_at=None,
                            updated_at=timestamp,
                            error=(
                                "Recovered durable runner did not complete its signed "
                                "identity handshake within the startup window; output "
                                "ownership remains locked and monitoring will continue."
                            ),
                        )
                        self._start_recovery_monitor(job_id, pid)
                        return
                    if poll_interval_seconds > 0:
                        time.sleep(poll_interval_seconds)
                    continue
                else:
                    timestamp = _now()
                    self._update_job(
                        job_id,
                        status="failed",
                        completed_at=timestamp,
                        updated_at=timestamp,
                        error=(
                            "Recovered durable runner identity handshake did not match; "
                            "no process was signaled."
                        ),
                    )
                    return
            if job.runner_alive is not True:
                payload = self._read_exit_status(job)
                if payload is not None:
                    runner_error = (
                        payload.get("error")
                        if isinstance(payload.get("error"), str)
                        else None
                    )
                    self._complete_job_from_return_code(
                        job, int(payload["return_code"]), _now(), runner_error
                    )
                else:
                    timestamp = _now()
                    self._update_job(
                        job_id,
                        status="failed",
                        completed_at=timestamp,
                        updated_at=timestamp,
                        error="Recovered durable runner exited without writing an exit status.",
                    )
                return
            if time.monotonic() - started_at >= max_poll_seconds:
                timestamp = _now()
                self._update_job(
                    job_id,
                    status="running",
                    completed_at=None,
                    updated_at=timestamp,
                    error=(
                        "Recovered durable runner exceeded the recovery monitoring "
                        "window without writing an exit status; output ownership "
                        "remains locked and monitoring will continue."
                    ),
                )
                self._start_recovery_monitor(job_id, pid)
                return
            if poll_interval_seconds > 0:
                time.sleep(poll_interval_seconds)

    def _mark_job_failed_after_restart(self, job: QuantJob) -> None:
        payload = self._read_exit_status(job)
        if payload is not None:
            runner_error = (
                payload.get("error") if isinstance(payload.get("error"), str) else None
            )
            self._complete_job_from_return_code(
                job, int(payload["return_code"]), _now(), runner_error
            )
            return
        timestamp = _now()
        self._update_job(
            job.job_id,
            status="failed",
            completed_at=timestamp,
            updated_at=timestamp,
            error="Job was marked failed because the API service restarted before it finished and no durable runner was found.",
        )

    def _insert_job(
        self, job: QuantJob, *, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        query = """
                    INSERT INTO jobs (
                        job_id, job_type, label, status, params_json, command_json, cwd,
                        pid, return_code, created_at, started_at, completed_at, updated_at,
                        error, log_path, artifact_paths_json, runner_identity_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
        values = (
            job.job_id,
            job.job_type,
            job.label,
            job.status,
            _json_dumps(job.params),
            _json_dumps(job.command),
            job.cwd,
            job.pid,
            job.return_code,
            job.created_at,
            job.started_at,
            job.completed_at,
            job.updated_at,
            job.error,
            job.log_path,
            _json_dumps(job.artifact_paths),
            _json_dumps(job.runner_identity),
        )
        if conn is not None:
            conn.execute(query, values)
            return
        with self._lock:
            with self._connect() as local_conn:
                local_conn.execute(query, values)
                local_conn.commit()

    def _update_job(self, job_id: str, **updates: Any) -> None:
        if not updates:
            return
        invalid = sorted(set(updates) - set(JOB_UPDATE_COLUMNS))
        if invalid:
            raise ValueError(f"Unsupported job update fields: {', '.join(invalid)}")
        assignments = ", ".join(f"{JOB_UPDATE_COLUMNS[key]} = ?" for key in updates)
        values = list(updates.values())
        values.append(job_id)
        with self._lock:
            with self._connect() as conn:
                conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", values)
                conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_one(self, query: str, params: tuple[Any, ...]) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(query, params).fetchone()

    def _fetch_all(self, query: str, params: tuple[Any, ...]) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return list(conn.execute(query, params).fetchall())

    @staticmethod
    def _timestamp(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.timestamp()

    def _progress_payload(self, job: QuantJob) -> Dict[str, Any]:
        patterns = self.command_specs.get(job.job_type, {}).get("progress_patterns", [])
        if not patterns:
            return {}
        started_timestamp = self._timestamp(job.started_at or job.created_at)
        candidates: List[Path] = []
        for pattern in patterns:
            candidates.extend(Path(match) for match in glob.glob(str(pattern)))
        valid_paths: List[Path] = []
        for path in candidates:
            try:
                metadata = path.stat()
            except OSError:
                continue
            if not path.is_file() or metadata.st_size > PROGRESS_FILE_MAX_BYTES:
                continue
            if started_timestamp is not None and metadata.st_mtime + 1 < started_timestamp:
                continue
            valid_paths.append(path)
        if not valid_paths:
            return {}
        raw: Optional[Dict[str, Any]] = None
        for progress_path in sorted(
            valid_paths,
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        ):
            try:
                candidate = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict) and self._progress_matches_job(
                job, candidate
            ):
                raw = candidate
                break
        if raw is None:
            return {}

        completed_raw = raw.get("completed", raw.get("completed_count", 0))
        total_raw = raw.get("total", raw.get("expected_count", 0))
        try:
            completed = max(0, int(completed_raw or 0))
            total = max(0, int(total_raw or 0))
        except (TypeError, ValueError):
            completed, total = 0, 0
        if total:
            completed = min(completed, total)
        percent = round((completed / total) * 100, 2) if total else 0.0

        started = self._timestamp(job.started_at)
        finished = self._timestamp(job.completed_at) if job.completed_at else time.time()
        elapsed_seconds = (
            max(0, int((finished or time.time()) - started))
            if started is not None
            else 0
        )
        eta_seconds: Optional[int] = None
        if job.status in {"pending", "running"} and completed > 0 and total > completed:
            eta_seconds = max(0, int((elapsed_seconds / completed) * (total - completed)))

        progress: Dict[str, Any] = {
            "stage": str(raw.get("stage") or raw.get("current_stage") or ""),
            "completed": completed,
            "total": total,
            "percent": percent,
            "current": str(
                raw.get("current")
                or raw.get("current_model")
                or raw.get("signal_id")
                or ""
            ),
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
        }
        for source, target in (
            ("updated_at", "updated_at"),
            ("pause_state", "pause_state"),
            ("lifecycle_status", "lifecycle_status"),
            ("status", "status"),
            ("run_id", "run_id"),
            ("batch_id", "batch_id"),
            ("reason_code", "reason_code"),
            ("next_gate", "next_gate"),
        ):
            value = raw.get(source)
            if isinstance(value, (str, int, float, bool)) and str(value):
                progress[target] = value
        artifact = raw.get("current_artifact")
        if isinstance(artifact, str) and artifact.strip():
            progress["current_artifact"] = Path(artifact).name
        return progress

    @staticmethod
    def _progress_matches_job(job: QuantJob, raw: Mapping[str, Any]) -> bool:
        for key in ("run_id", "batch_id"):
            expected = job.params.get(key)
            if expected not in (None, "") and str(raw.get(key) or "") != str(expected):
                return False

        if "smoke" in job.params:
            expected_mode = "smoke" if _as_bool(job.params.get("smoke")) else "formal"
            actual_mode = raw.get("run_mode")
            if actual_mode is None and isinstance(raw.get("smoke"), bool):
                actual_mode = "smoke" if raw["smoke"] else "formal"
            if str(actual_mode or "") != expected_mode:
                return False

        if job.job_type == "run_candidate_pipeline" or "dry_run" in job.params:
            if not isinstance(raw.get("dry_run"), bool):
                return False
            if bool(raw["dry_run"]) != _as_bool(job.params.get("dry_run")):
                return False

        for key, expected in job.params.items():
            if key in {"run_id", "batch_id", "smoke", "dry_run"} or key not in raw:
                continue
            actual = raw.get(key)
            if isinstance(expected, (str, int, float, bool)) and actual != expected:
                return False
        return True

    def _job_can_cancel(
        self, job: QuantJob, progress: Mapping[str, Any]
    ) -> bool:
        if job.status in {"pending", "running"}:
            return True
        if job.status != "completed":
            return False
        spec = self.command_specs.get(job.job_type, {})
        if spec.get("cancel_command") is None:
            return False
        lifecycle = str(progress.get("lifecycle_status") or "").strip().lower()
        pause_state = str(progress.get("pause_state") or "").strip().lower()
        return lifecycle in {"starting", "running", "pause_requested"} or pause_state in {
            "starting",
            "running",
            "pause_requested",
        }

    @staticmethod
    def _log_updated_at(job: QuantJob) -> Optional[str]:
        try:
            modified = Path(job.log_path).stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(modified).astimezone().isoformat(timespec="microseconds")

    def _row_to_job(self, row: sqlite3.Row) -> QuantJob:
        job = QuantJob(
            job_id=row["job_id"],
            job_type=row["job_type"],
            label=row["label"],
            status=row["status"],
            params=json.loads(row["params_json"] or "{}"),
            command=json.loads(row["command_json"] or "[]"),
            cwd=row["cwd"],
            pid=row["pid"],
            return_code=row["return_code"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=row["updated_at"],
            error=row["error"],
            log_path=row["log_path"],
            artifact_paths=json.loads(row["artifact_paths_json"] or "[]"),
            runner_identity=json.loads(row["runner_identity_json"] or "{}"),
        )
        progress = self._progress_payload(job)
        runner_alive = None
        if job.pid:
            if job.status in TERMINAL_STATUSES:
                runner_alive = False
            elif self._runner_identity_startup_pending(job):
                runner_alive = None
            else:
                runner_alive = self._runner_identity_matches(job)
        return replace(
            job,
            progress=progress,
            runner_alive=runner_alive,
            can_cancel=self._job_can_cancel(job, progress),
            log_updated_at=self._log_updated_at(job),
        )
