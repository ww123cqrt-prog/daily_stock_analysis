# -*- coding: utf-8 -*-
"""Local quant workflow job runner.

This service exposes a constrained, auditable wrapper around the durable
``quant_workflow/scripts`` entry points.  It deliberately does not accept
arbitrary shell strings from the web UI.
"""

from __future__ import annotations

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
import glob
from dataclasses import dataclass, field
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
RECOVERED_RUNNER_POLL_INTERVAL_SECONDS = 1.0
RECOVERED_RUNNER_MAX_POLL_SECONDS = 48 * 60 * 60
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
EXIT_STATUS_FILENAME = "exit_status.json"
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
CONFIG_FILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DURABLE_RUNNER_CODE = r"""
import json
import os
import subprocess
import sys
import traceback

command = json.loads(os.environ["QUANT_JOB_COMMAND_JSON"])
cwd = os.environ.get("QUANT_JOB_CWD") or None
exit_path = os.environ["QUANT_JOB_EXIT_STATUS_PATH"]
return_code = 1
error = None
try:
    process = subprocess.Popen(command, cwd=cwd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    return_code = process.wait()
except BaseException as exc:
    error = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    payload = {"return_code": return_code}
    if error:
        payload["error"] = error
    tmp_path = exit_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_path, exit_path)
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
        }


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def _script_command(script_name: str) -> List[str]:
    return [_python_bin(), str(QUANT_WORKFLOW_ROOT / "scripts" / script_name)]


def _append_reports_out_dir(command: List[str]) -> None:
    command.extend(["--out-dir", str(REPORTS_ROOT)])


def _build_default_command_specs() -> Dict[str, Dict[str, Any]]:
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
        command.extend(["update", "etf_momentum_daily_qfq"])
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
        command = _script_command("merge_to_qlib.py")
        if _as_bool(params.get("smoke")):
            command.extend(
                [
                    "--smoke-fixture",
                    "--smoke-dir",
                    str(REPORTS_ROOT / "smoke" / "merge_to_qlib"),
                ]
            )
        _append_flag(command, params, "no_overwrite", "--no-overwrite")
        return command

    def apply_data_cleaning(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("apply_data_cleaning.py")
        _append_reports_out_dir(command)
        _append_flag(command, params, "dry_run", "--dry-run")
        _append_flag(command, params, "no_overwrite", "--no-overwrite")
        _append_flag(command, params, "skip_qlib_convert", "--skip-qlib-convert")
        return command

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
        return command

    def run_robust_signal_research(params: Mapping[str, Any]) -> List[str]:
        return _robust_signal_command(params)

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
        command = _robust_signal_command({"run_id": run_id})
        command.append("--resume")
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
            "label": "更新 ETF 动量数据",
            "command": download_etf_momentum,
            "allowed_params": {"end"},
            "artifact_patterns": [
                str(
                    quant_data_root(DSA_ROOT)
                    / "parquet"
                    / "etf_1d"
                    / "etf_momentum_daily_raw"
                    / "LATEST.json"
                ),
                str(
                    quant_data_root(DSA_ROOT)
                    / "parquet"
                    / "etf_1d"
                    / "etf_momentum_daily_qfq"
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
            "label": "合并并转换为 Qlib",
            "command": merge_to_qlib,
            "allowed_params": {"no_overwrite", "smoke"},
        },
        "apply_data_cleaning": {
            "label": "应用清洗策略生成 cleaned Qlib",
            "command": apply_data_cleaning,
            "allowed_params": {"dry_run", "no_overwrite", "skip_qlib_convert"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "data_cleaning_applied_latest.json"),
                str(REPORTS_ROOT / "data_cleaning_applied_latest.md"),
                str(REPORTS_ROOT / "data_cleaning_applied_*.json"),
                str(REPORTS_ROOT / "data_cleaning_applied_*.md"),
            ],
        },
        "repair_qlib_provider": {
            "label": "预演修复 Qlib Provider",
            "command": repair_qlib_provider,
            "allowed_params": set(),
        },
        "check_qlib_data": {
            "label": "检查 Qlib 数据",
            "command": _script_command("check_qlib_data.py"),
            "allowed_params": set(),
        },
        "generate_csi300": {
            "label": "预演生成 CSI300 股票池",
            "command": generate_csi300,
            "allowed_params": set(),
        },
        "add_index_benchmark": {
            "label": "预演添加沪深300 benchmark",
            "command": add_index_benchmark,
            "allowed_params": set(),
        },
        "report_data_quality": {
            "label": "生成数据质量报告",
            "command": report_data_quality,
            "allowed_params": {"overlap_sample", "merged_sample"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "data_quality_report_latest.md"),
                str(REPORTS_ROOT / "data_quality_report_*.md"),
                str(REPORTS_ROOT / "data_quality_sources_*.csv"),
                str(REPORTS_ROOT / "data_quality_issues_*.csv"),
                str(REPORTS_ROOT / "data_quality_overlap_*.csv"),
                str(REPORTS_ROOT / "data_quality_merged_summary_*.csv"),
                str(REPORTS_ROOT / "research_status_latest.json"),
            ],
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
                str(REPORTS_ROOT / "data_hygiene_inventory_latest.md"),
                str(REPORTS_ROOT / "data_hygiene_inventory_latest.json"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_latest.md"),
                str(REPORTS_ROOT / "data_hygiene_cleanup_plan_latest.json"),
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
            "artifact_patterns": [
                str(REPORTS_ROOT / "latest_training_result.json"),
                str(REPORTS_ROOT / "train_baseline_lgb_status_latest.json"),
                str(REPORTS_ROOT / "train_baseline_lgb_*.log"),
                str(REPORTS_ROOT / "experiment_runs.jsonl"),
                str(REPORTS_ROOT / "experiment_log.md"),
                str(REPORTS_ROOT / "full_experiment_report.md"),
                str(REPORTS_ROOT / "ic_ir_analysis.md"),
            ],
        },
        "run_candidate_pipeline": {
            "label": "运行五个选股模型候选流程",
            "command": run_candidate_pipeline,
            "allowed_params": {"dry_run", "provider"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "candidate_pipeline_latest.json"),
                str(REPORTS_ROOT / "candidate_pipeline_latest.md"),
                str(REPORTS_ROOT / "candidate_pipeline_progress_latest.json"),
                str(REPORTS_ROOT / "candidate_pipeline_*.json"),
                str(REPORTS_ROOT / "candidate_pipeline_*.md"),
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
        "run_robust_signal_research": {
            "label": "启动鲁棒选股信号研究",
            "command": run_robust_signal_research,
            "allowed_params": {"run_id"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
                str(REPORTS_ROOT / "robust_signal" / "runs" / "*" / "result.json"),
            ],
        },
        "pause_robust_signal_research": {
            "label": "暂停鲁棒选股信号研究",
            "command": pause_robust_signal_research,
            "allowed_params": {"run_id"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
            ],
        },
        "resume_robust_signal_research": {
            "label": "恢复鲁棒选股信号研究",
            "command": resume_robust_signal_research,
            "allowed_params": {"run_id"},
            "artifact_patterns": [
                str(REPORTS_ROOT / "robust_signal" / "robust_signal_status_latest.json"),
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
        self, job_type: str, params: Optional[Mapping[str, Any]] = None
    ) -> QuantJob:
        params_dict = dict(params or {})
        spec = self._get_spec(job_type)
        command = self._build_command(job_type, params_dict)
        with self._lock:
            if job_type in QUANT_DATA_NETWORK_JOB_TYPES:
                active = self._active_quant_data_network_job()
                if active is not None:
                    raise ValueError(
                        "A quant_data network job is already active: "
                        f"{active.job_type} ({active.job_id}). Wait for it to finish or cancel it first."
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
            self._insert_job(job)
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
                "SELECT * FROM jobs WHERE job_type = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (job_type, bounded_limit),
            )
        else:
            rows = self._fetch_all(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (bounded_limit,),
            )
        return [self._row_to_job(row) for row in rows]

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

    def _active_quant_data_network_job(self) -> Optional[QuantJob]:
        placeholders = ", ".join("?" for _item in QUANT_DATA_NETWORK_JOB_TYPES)
        row = self._fetch_one(
            f"""
            SELECT * FROM jobs
            WHERE status IN ('pending', 'running')
              AND job_type IN ({placeholders})
            ORDER BY rowid DESC
            LIMIT 1
            """,
            tuple(sorted(QUANT_DATA_NETWORK_JOB_TYPES)),
        )
        return self._row_to_job(row) if row is not None else None

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
        with self._lock:
            job = self.get_job(job_id)
            if job.status in TERMINAL_STATUSES:
                return job
            process = self._processes.get(job_id)

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
            if current.status in TERMINAL_STATUSES:
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

    def _start_job(self, job: QuantJob) -> None:
        started_at = _now()
        self._update_job(
            job.job_id, status="running", started_at=started_at, updated_at=started_at
        )
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle: Any = None
        try:
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
            self._update_job(
                job.job_id,
                status="failed",
                completed_at=_now(),
                updated_at=_now(),
                error=f"Failed to start job: {exc}",
            )
            return

        with self._lock:
            self._processes[job.job_id] = process
            self._update_job(job.job_id, pid=process.pid, updated_at=_now())
        thread = threading.Thread(
            target=self._wait_for_process,
            args=(job.job_id, process, handle),
            name=f"quant-job-{job.job_id}",
            daemon=True,
        )
        thread.start()

    def _exit_status_path(self, job: QuantJob) -> Path:
        return Path(job.log_path).parent / EXIT_STATUS_FILENAME

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
                self._collect_artifact_paths(current.job_type)
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

    def _collect_artifact_paths(self, job_type: str) -> List[str]:
        patterns = self._get_spec(job_type).get("artifact_patterns", [])
        paths: List[Path] = []
        for pattern in patterns:
            for match in glob.glob(str(pattern)):
                path = Path(match)
                if path.is_file():
                    paths.append(path)

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
                    artifact_paths_json TEXT NOT NULL
                )
                """
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
            if self._pid_is_alive(job.pid):
                timestamp = _now()
                self._update_job(
                    job.job_id,
                    status="running",
                    updated_at=timestamp,
                    error="API service restarted while this durable runner was still running; monitoring exit status.",
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
            if not self._pid_is_alive(pid):
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
                    status="failed",
                    completed_at=timestamp,
                    updated_at=timestamp,
                    error=(
                        "Recovered durable runner exceeded the recovery monitoring "
                        "window without writing an exit status."
                    ),
                )
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

    def _insert_job(self, job: QuantJob) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, label, status, params_json, command_json, cwd,
                        pid, return_code, created_at, started_at, completed_at, updated_at,
                        error, log_path, artifact_paths_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                    ),
                )
                conn.commit()

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

    def _row_to_job(self, row: sqlite3.Row) -> QuantJob:
        return QuantJob(
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
        )
