# -*- coding: utf-8 -*-
"""Local quant workflow job runner.

This service exposes a constrained, auditable wrapper around the durable
``quant_workflow/scripts`` entry points.  It deliberately does not accept
arbitrary shell strings from the web UI.
"""

from __future__ import annotations

import json
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DSA_ROOT = _env_path("DSA_ROOT", Path(__file__).resolve().parents[2])
PROJECTS_ROOT = _env_path("WORKING_PROJECTS_ROOT", DSA_ROOT.parent)
QUANT_WORKFLOW_ROOT = _env_path("QUANT_WORKFLOW_ROOT", PROJECTS_ROOT / "quant_workflow")
REPORTS_ROOT = _env_path("QUANT_WORKFLOW_REPORTS_ROOT", QUANT_WORKFLOW_ROOT / "reports")
DEFAULT_JOBS_ROOT = REPORTS_ROOT / "jobs"
QLIB_PYTHON = _env_path("QLIB_PYTHON", Path(sys.executable))

JOB_TERMINATE_TIMEOUT_SECONDS = 5
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
EXIT_STATUS_FILENAME = "exit_status.json"
JQDATA_WEB_DRY_RUN_DEFAULT_LIMIT = 3
JQDATA_WEB_DRY_RUN_MAX_LIMIT = 20
QLIB_PREDICTION_WEB_TOPK_MAX = 500
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
    "overlap_sample": ("integer", 1),
    "merged_sample": ("integer", 1),
    "eval_window_days": ("integer", 1),
    "lookback_days": ("integer", 2),
    "min_symbols": ("integer", 1),
    "phase": ("integer", 0),
    "topk": ("integer", 1),
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


def _append_flag(command: List[str], params: Mapping[str, Any], key: str, flag: str) -> None:
    if _as_bool(params.get(key)):
        command.append(flag)


def _append_value(command: List[str], params: Mapping[str, Any], key: str, flag: str) -> None:
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
            raise ValueError(f"Param {key} must be {_numeric_kind_label(kind)}.") from exc
        if not math.isfinite(float(number)) or number < minimum:
            raise ValueError(f"Param {key} must be >= {minimum}.")


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
        raise ValueError(f"Param limit must be <= {JQDATA_WEB_DRY_RUN_MAX_LIMIT} for browser JQData dry-run jobs.")
    return value


def _ensure_jqdata_web_dry_run(params: Mapping[str, Any]) -> None:
    if "dry_run" in params and not _as_bool(params.get("dry_run")):
        raise ValueError("download_jqdata web jobs only support dry_run=true. Use the script directly for real downloads.")


def _ensure_optimize_strategy_web_dry_run(params: Mapping[str, Any]) -> None:
    if "dry_run" in params and not _as_bool(params.get("dry_run")):
        raise ValueError("optimize_strategy web jobs only support dry_run=true. Use the script directly for full experiments.")
    phase = _int_param(params.get("phase", 4), "phase")
    if phase < 0 or phase > 4:
        raise ValueError("phase must be between 0 and 4.")
    topk = _int_param(params.get("topk", 30), "topk")
    if topk < 1:
        raise ValueError("Param topk must be >= 1.")
    if topk > 500:
        raise ValueError("Param topk must be <= 500.")


def _safe_run_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw and not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError("run_id may only contain letters, numbers, underscores, or hyphens.")
    return raw


def _ensure_qlib_prediction_summary_params(params: Mapping[str, Any]) -> None:
    topk = _int_param(params.get("topk", 20), "topk")
    if topk < 1:
        raise ValueError("Param topk must be >= 1.")
    if topk > QLIB_PREDICTION_WEB_TOPK_MAX:
        raise ValueError(f"Param topk must be <= {QLIB_PREDICTION_WEB_TOPK_MAX}.")
    _safe_run_id(params.get("run_id"))


def _safe_config_path(value: Any) -> str:
    raw = str(value or "baseline_lgb_alpha158.yaml").strip()
    if not raw:
        raw = "baseline_lgb_alpha158.yaml"
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("Config must be a file name under quant_workflow/configs.")
    return str(QUANT_WORKFLOW_ROOT / "configs" / raw)


def _safe_provider(value: Any) -> str:
    raw = str(value or "canonical").strip().lower()
    allowed = {"canonical", "cleaned"}
    if raw not in allowed:
        raise ValueError("Provider must be one of: canonical, cleaned.")
    return raw


def _python_bin() -> str:
    return str(QLIB_PYTHON if QLIB_PYTHON.exists() else Path(sys.executable))


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
            command.extend(["--smoke-fixture", "--smoke-dir", str(REPORTS_ROOT / "smoke" / "merge_to_qlib")])
        _append_flag(command, params, "no_overwrite", "--no-overwrite")
        return command

    def apply_data_cleaning(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("apply_data_cleaning.py")
        _append_reports_out_dir(command)
        _append_flag(command, params, "dry_run", "--dry-run")
        _append_flag(command, params, "no_overwrite", "--no-overwrite")
        _append_flag(command, params, "skip_qlib_convert", "--skip-qlib-convert")
        return command

    def repair_qlib_provider(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("repair_qlib_provider.py")
        _append_flag(command, params, "dry_run", "--dry-run")
        return command

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

    def optimize_strategy(params: Mapping[str, Any]) -> List[str]:
        _ensure_optimize_strategy_web_dry_run(params)
        command = _script_command("optimize_strategy.py")
        command.extend(["--dry-run", "--skip-training", "--output-dir", str(REPORTS_ROOT)])
        _append_value(command, params, "phase", "--phase")
        _append_value(command, params, "topk", "--topk")
        return command

    def train_baseline_lgb(params: Mapping[str, Any]) -> List[str]:
        command = _script_command("train_baseline_lgb.py")
        command.extend(["--config", _safe_config_path(params.get("config"))])
        command.extend(["--provider", _safe_provider(params.get("provider"))])
        _append_value(command, params, "experiment_name", "--experiment-name")
        return command

    def run_dsa_backtest(params: Mapping[str, Any]) -> List[str]:
        command = [_python_bin(), str(DSA_ROOT / "main.py"), "--backtest"]
        _append_value(command, params, "code", "--backtest-code")
        _append_value(command, params, "eval_window_days", "--backtest-days")
        _append_flag(command, params, "force", "--backtest-force")
        return command

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
        "download_jqdata": {
            "label": "验证 JQData 连接（dry-run）",
            "command": download_jqdata,
            "allowed_params": {"universe", "start", "end", "limit", "sleep", "dry_run"},
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
            "label": "修复 Qlib Provider",
            "command": repair_qlib_provider,
            "allowed_params": {"dry_run"},
        },
        "check_qlib_data": {
            "label": "检查 Qlib 数据",
            "command": _script_command("check_qlib_data.py"),
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
        "optimize_strategy": {
            "label": "策略优化预演",
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
            "label": "训练 Qlib LightGBM 基线",
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
        default_cwd: Path = PROJECTS_ROOT,
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self.db_path = self.jobs_root / "index.sqlite"
        self.default_cwd = Path(default_cwd)
        self.command_specs = command_specs if command_specs is not None else _build_default_command_specs()
        self._processes: Dict[str, subprocess.Popen[bytes]] = {}
        self._lock = threading.RLock()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._recover_running_jobs()

    def available_job_types(self) -> Dict[str, Dict[str, Any]]:
        return {
            job_type: {
                "label": str(spec.get("label") or job_type),
                "allowed_params": sorted(str(item) for item in spec.get("allowed_params", [])),
            }
            for job_type, spec in self.command_specs.items()
        }

    def submit_job(self, job_type: str, params: Optional[Mapping[str, Any]] = None) -> QuantJob:
        params_dict = dict(params or {})
        spec = self._get_spec(job_type)
        command = self._build_command(job_type, params_dict)
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

    def list_jobs(self, limit: int = 50, job_type: Optional[str] = None) -> List[QuantJob]:
        bounded_limit = max(1, min(int(limit), 200))
        if job_type:
            rows = self._fetch_all(
                "SELECT * FROM jobs WHERE job_type = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (job_type, bounded_limit),
            )
        else:
            rows = self._fetch_all("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (bounded_limit,))
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

    def get_log(self, job_id: str, max_bytes: int = 20000) -> Dict[str, Any]:
        job = self.get_job(job_id)
        log_path = Path(job.log_path)
        if not log_path.exists():
            return {"job_id": job_id, "content": "", "truncated": False, "log_path": str(log_path)}

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
                raise ValueError(f"Unsupported params for {job_type}: {', '.join(unknown)}")
        _validate_numeric_params(params)

        command_spec = spec.get("command")
        if callable(command_spec):
            command = list(command_spec(params))
        elif isinstance(command_spec, Iterable) and not isinstance(command_spec, (str, bytes)):
            command = [str(part) for part in command_spec]
        else:
            raise ValueError(f"Job type {job_type} has no runnable command.")
        if not command:
            raise ValueError(f"Job type {job_type} produced an empty command.")
        return command

    def _start_job(self, job: QuantJob) -> None:
        started_at = _now()
        self._update_job(job.job_id, status="running", started_at=started_at, updated_at=started_at)
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle: Any = None
        try:
            handle = log_path.open("ab")
            header = (
                f"[job] {job.job_id} {job.job_type}\n"
                f"[cwd] {job.cwd}\n"
                f"[command] {' '.join(job.command)}\n\n"
            )
            handle.write(header.encode("utf-8"))
            handle.flush()
            env = os.environ.copy()
            env.update(
                {
                    "QUANT_JOB_COMMAND_JSON": _json_dumps(job.command),
                    "QUANT_JOB_CWD": job.cwd,
                    "QUANT_JOB_EXIT_STATUS_PATH": str(self._exit_status_path(job)),
                }
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
        if not isinstance(payload, dict) or not isinstance(payload.get("return_code"), int):
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
            artifact_paths = self._collect_artifact_paths(current.job_type) if return_code == 0 else current.artifact_paths
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
            error = runner_error if runner_error else None if return_code == 0 else f"Process exited with return code {return_code}."
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

    def _wait_for_process(self, job_id: str, process: subprocess.Popen[bytes], handle: Any) -> None:
        try:
            return_code = process.wait()
            completed_at = _now()
            current = self.get_job(job_id)
            payload = self._read_exit_status(current) or {}
            runner_error = payload.get("error") if isinstance(payload.get("error"), str) else None
            self._complete_job_from_return_code(current, return_code, completed_at, runner_error)
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
                runner_error = payload.get("error") if isinstance(payload.get("error"), str) else None
                self._complete_job_from_return_code(job, int(payload["return_code"]), _now(), runner_error)
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
            self._mark_job_failed_after_restart(job.job_id)

    def _start_recovery_monitor(self, job_id: str, pid: Optional[int]) -> None:
        thread = threading.Thread(
            target=self._poll_recovered_process,
            args=(job_id, pid),
            name=f"quant-job-recover-{job_id}",
            daemon=True,
        )
        thread.start()

    def _poll_recovered_process(self, job_id: str, pid: Optional[int]) -> None:
        while True:
            time.sleep(1.0)
            try:
                job = self.get_job(job_id)
            except KeyError:
                return
            if job.status in TERMINAL_STATUSES:
                return
            payload = self._read_exit_status(job)
            if payload is not None:
                runner_error = payload.get("error") if isinstance(payload.get("error"), str) else None
                self._complete_job_from_return_code(job, int(payload["return_code"]), _now(), runner_error)
                return
            if not self._pid_is_alive(pid):
                payload = self._read_exit_status(job)
                if payload is not None:
                    runner_error = payload.get("error") if isinstance(payload.get("error"), str) else None
                    self._complete_job_from_return_code(job, int(payload["return_code"]), _now(), runner_error)
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

    def _mark_job_failed_after_restart(self, job_id: str) -> None:
        timestamp = _now()
        self._update_job(
            job_id,
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
