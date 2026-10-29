# -*- coding: utf-8 -*-
"""Research workspace status endpoint."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import sys
import threading
import time
import yaml
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException, Query

from quant_workflow.scripts.quant_data_contract import (
    AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS,
    AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
    DataContractError,
    filesystem_state_fingerprint,
    resolve_dataset,
    sha256_file,
    validate_latest_pointer,
)
from quant_workflow.scripts._file_lock import lock_file, unlock_file
from quant_workflow.scripts._mlflow_run_provenance import (
    RunProvenanceError,
    normalize_provider_lineage,
    provider_identity,
    provider_lineage_is_complete,
    provider_lineages_match,
    resolve_immutable_provider_identity,
)
from quant_workflow.scripts._paths import (
    provider_calendar_days,
    provider_has_instrument_pool,
    provider_instrument_ranges,
    qlib_workflow_provider_uri,
)
from quant_workflow.scripts.quant_data_service_client import (
    QuantDataClientError,
    QuantDataIdentity,
)
from quant_workflow.scripts.prepare_robust_provider import RuntimeProviderIdentityCache
from quant_workflow.scripts.robust_signal_data_contract import (
    RobustSignalDataBlocked,
    consume_accepted_robust_signal_data,
    public_readiness_payload,
    resolve_robust_signal_data_readiness,
)
from src.safe_errors import sanitize_public_error
from src.safe_urls import public_http_url
from src.services.quant_workflow_paths import (
    dsa_root,
    projects_root,
    quant_data_root,
    reports_root,
    workflow_root,
)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DSA_ROOT = dsa_root()
PROJECTS_ROOT = projects_root(DSA_ROOT)
QUANT_WORKFLOW_ROOT = workflow_root(DSA_ROOT)
QUANT_WORKFLOW_REPORTS_ROOT = reports_root(DSA_ROOT)
QUANT_DATA_ROOT = quant_data_root(DSA_ROOT)
QLIB_PROVIDER_ROOT = _env_path("QLIB_PROVIDER_ROOT", QUANT_DATA_ROOT / "qlib")
RESEARCH_STATUS_PATH = QUANT_WORKFLOW_REPORTS_ROOT / "research_status_latest.json"
_ROBUST_PROVIDER_IDENTITY = RuntimeProviderIdentityCache()
EXPERIMENT_LOG_FILENAME = "experiment_log.md"
EXPERIMENT_RUNS_FILENAME = "experiment_runs.jsonl"
EXPERIMENT_DELETED_RUNS_FILENAME = "experiment_runs_deleted.jsonl"
EXPERIMENT_LOCK_FILENAME = ".experiment.lock"
LATEST_TRAINING_RESULT_FILENAME = "latest_training_result.json"
CASH_CONTROL_BACKTEST_FILENAME = "cash_control_backtest_latest.json"
QLIB_PREDICTIONS_FILENAME = "qlib_predictions_latest.json"
CANDIDATE_PIPELINE_FILENAME = "candidate_pipeline_latest.json"
DATA_MANIFEST_FILENAME = "DATA_MANIFEST.json"
ASHARE_DAILY_QFQ_DATASET_ID = "ashare_daily_qfq"
ASHARE_DAILY_QFQ_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_qfq")
QLIB_PROVIDER_NAMES = ("canonical", "cleaned")
QLIB_TRAINING_REQUIRED_DATASETS = (
    ASHARE_DAILY_QFQ_DATASET_ID,
    "cn_index_daily",
    "pit_csi800",
)
QLIB_CLEANED_REQUIRED_DATASETS = ("qlib_cleaning_plan",)
ROBUST_STRATEGY_DIRNAME = "robust_strategy"
ROBUST_SIGNAL_DIRNAME = "robust_signal"
QLIB_RESEARCH_REPORTS = {
    "model_explainability": (
        "模型效果深度解释",
        "qlib_research_model_explainability_latest.json",
    ),
    "backtest_detail": ("回测详细分解", "qlib_research_backtest_detail_latest.json"),
    "model_comparison": ("多选股模型/多实验对比", "qlib_research_model_comparison_latest.json"),
    "factor_analysis": ("因子挖掘辅助", "qlib_research_factor_analysis_latest.json"),
    "rolling_training": ("滚动训练计划", "qlib_research_rolling_training_latest.json"),
    "risk_model": ("风险模型集成诊断", "qlib_research_risk_model_latest.json"),
}
EXPERIMENT_FILE_LOCK = threading.RLock()
DEFAULT_MLFLOW_URL = "http://127.0.0.1:6006/"
DIRECTORY_SIZE_CACHE_TTL_SECONDS = 60.0
DATA_HYGIENE_STATE_SCAN_MAX_ENTRIES = 200_000
DATA_HYGIENE_STATE_SCAN_MAX_SECONDS = 5.0
DATA_HYGIENE_POINTER_MAX_BYTES = 64 * 1024
DATA_HYGIENE_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
DATA_QUALITY_POINTER_MAX_BYTES = 64 * 1024
DATA_QUALITY_EVIDENCE_MAX_BYTES = 4 * 1024 * 1024
DATA_QUALITY_MARKDOWN_MAX_BYTES = 4 * 1024 * 1024
DIRECTORY_SIZE_SCAN_MAX_ENTRIES = 50000
DIRECTORY_SIZE_SCAN_MAX_SECONDS = 0.5
CLEANUP_ACTION_PREVIEW_LIMIT = 50
ROBUST_REPORT_MAX_BYTES = 8 * 1024 * 1024
ROBUST_LIST_PREVIEW_LIMIT = 100
ROBUST_MAPPING_PREVIEW_LIMIT = 100
ROBUST_TREE_MAX_DEPTH = 8
ROBUST_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
ROBUST_STAGE_PROGRESS_REPORTS = {
    "build_pit_csi800": "build_status_latest.json",
}
TONGHUASHUN_CANDIDATE_PATHS = (
    Path("/Applications/同花顺.app"),
    Path("/Applications/同花顺远航版.app"),
    Path("/Applications/同花顺iFinD.app"),
    Path.home() / "Applications" / "同花顺.app",
    Path.home() / "Applications" / "同花顺远航版.app",
    Path.home() / "Applications" / "同花顺iFinD.app",
)


@dataclass(frozen=True)
class _NativeQlibProviderResolution:
    authoritative: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    lineage: Optional[Dict[str, str]] = None
    provider_name: str = "canonical"
    dataset_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    manifest_sha256: Optional[str] = None
    content_sha256: Optional[str] = None
    calendar_end: Optional[str] = None
    instrument_count: int = 0
    benchmark_complete: bool = False


@dataclass(frozen=True)
class _DataQualityPublication:
    payload: Dict[str, Any]
    markdown_bytes: bytes
    report_id: str
    evidence_sha256: Optional[str]
    training_authoritative: bool


logger = logging.getLogger(__name__)

router = APIRouter()
_DIRECTORY_SIZE_CACHE_LOCK = threading.Lock()
_DIRECTORY_SIZE_CACHE: Dict[str, Tuple[float, float]] = {}
_DATA_HYGIENE_STATE_SCAN_LOCK = threading.Lock()


def _filesystem_state_fingerprint_live(path: Path) -> Dict[str, Any]:
    with _DATA_HYGIENE_STATE_SCAN_LOCK:
        return filesystem_state_fingerprint(
            path,
            max_entries=DATA_HYGIENE_STATE_SCAN_MAX_ENTRIES,
            max_seconds=DATA_HYGIENE_STATE_SCAN_MAX_SECONDS,
        )


def _valid_filesystem_state_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("algorithm")
        == "relative-path-size-mtime-ctime-device-inode-v2"
        and isinstance(value.get("file_count"), int)
        and not isinstance(value.get("file_count"), bool)
        and value["file_count"] >= 0
        and isinstance(value.get("total_bytes"), int)
        and not isinstance(value.get("total_bytes"), bool)
        and value["total_bytes"] >= 0
        and isinstance(value.get("scanned_entries"), int)
        and not isinstance(value.get("scanned_entries"), bool)
        and value["scanned_entries"] >= value["file_count"]
        and isinstance(value.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
    )


def _lock_experiment_handle(lock_handle) -> None:
    lock_file(lock_handle, exclusive=True)


def _unlock_experiment_handle(lock_handle) -> None:
    unlock_file(lock_handle)


@contextmanager
def _experiment_file_guard(reports_dir: Path) -> Iterator[None]:
    """Guard experiment notebook mutations across threads and local processes."""
    with EXPERIMENT_FILE_LOCK:
        reports_dir.mkdir(parents=True, exist_ok=True)
        lock_path = reports_dir / EXPERIMENT_LOCK_FILENAME
        # Keep this small advisory lock file around deliberately. It only
        # coordinates writers that also take this lock; quant_workflow writers
        # must use the same path before mutating experiment notebook files.
        with lock_path.open("a+b") as lock_handle:
            _lock_experiment_handle(lock_handle)
            try:
                yield
            finally:
                _unlock_experiment_handle(lock_handle)


def _read_json_object_bytes(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[bytes]]:
    """Read and decode one JSON object from the exact bytes used by callers."""

    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, raw
    return payload, raw


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    payload, _raw = _read_json_object_bytes(path)
    return payload


def _trusted_root_relative_path(root: Path, raw_path: Any) -> Optional[Path]:
    """Keep descendant components lexical while allowing a trusted root alias."""

    if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
        return None
    root_input = Path(os.path.abspath(str(Path(root).expanduser())))
    root_resolved = _resolve_for_boundary(root_input)
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        candidate = Path(os.path.abspath(str(candidate)))
        relative = None
        for boundary in (root_input, root_resolved):
            try:
                relative = candidate.relative_to(boundary)
                break
            except ValueError:
                continue
        if relative is None:
            return None
    else:
        relative = candidate
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative


def _read_bounded_regular_file_at(
    root: Path,
    relative_path: Path,
    *,
    max_bytes: int,
) -> Optional[bytes]:
    """Read one regular file without following descendant symlinks."""

    relative = Path(relative_path)
    if (
        max_bytes < 0
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None

    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        return _read_bounded_regular_file_fallback(
            root,
            relative,
            max_bytes=max_bytes,
        )

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        file_flags |= os.O_NONBLOCK

    descriptors: list[int] = []
    file_fd: Optional[int] = None
    try:
        root_fd = os.open(str(_resolve_for_boundary(root)), directory_flags)
        descriptors.append(root_fd)
        current_fd = root_fd
        for part in relative.parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            descriptors.append(current_fd)
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=current_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            return None

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                return None
        after = os.fstat(file_fd)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_state != after_state or total != after.st_size:
            return None
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_bounded_regular_file_fallback(
    root: Path,
    relative: Path,
    *,
    max_bytes: int,
) -> Optional[bytes]:
    """Read a bounded file where directory-fd flags are unavailable.

    Each path component is checked with ``lstat`` and the opened file is
    compared with the pre-open and post-read metadata. This is less race-proof
    than the descriptor-walk above, but it rejects symlinks and keeps reads
    inside the resolved root on Windows and similar platforms.
    """

    root = Path(root).expanduser().absolute()
    try:
        root_before = root.lstat()
    except OSError:
        return None
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        return None
    current = root
    component_states: list[tuple[Path, os.stat_result]] = []
    try:
        for index, part in enumerate(relative.parts):
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                return None
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                return None
            component_states.append((current, metadata))
        resolved_root = root.resolve(strict=True)
        resolved_file = current.resolve(strict=True)
        resolved_file.relative_to(resolved_root)
        with current.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                return None
            content = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        if (
            len(content) > max_bytes
            or not stat.S_ISREG(after.st_mode)
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(content) != after.st_size
        ):
            return None
        for component, expected in component_states:
            current_state = component.lstat()
            if (
                stat.S_ISLNK(current_state.st_mode)
                or current_state.st_dev != expected.st_dev
                or current_state.st_ino != expected.st_ino
                or current_state.st_mtime_ns != expected.st_mtime_ns
                or current_state.st_ctime_ns != expected.st_ctime_ns
            ):
                return None
        root_after = root.lstat()
        if (
            stat.S_ISLNK(root_after.st_mode)
            or root_after.st_dev != root_before.st_dev
            or root_after.st_ino != root_before.st_ino
        ):
            return None
        return content
    except (OSError, ValueError):
        return None


def _read_json_object_bytes_at(
    root: Path,
    relative_path: Path,
    *,
    max_bytes: int,
) -> tuple[Optional[Dict[str, Any]], Optional[bytes]]:
    raw = _read_bounded_regular_file_at(
        root,
        relative_path,
        max_bytes=max_bytes,
    )
    if raw is None:
        return None, None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None, raw
    return (payload if isinstance(payload, dict) else None), raw


def _public_path_label(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw
    resolved = _resolve_for_boundary(path)
    for root in (PROJECTS_ROOT, DSA_ROOT, QUANT_WORKFLOW_ROOT, QUANT_DATA_ROOT):
        root_resolved = _resolve_for_boundary(root)
        try:
            relative = resolved.relative_to(root_resolved)
        except ValueError:
            continue
        return str(Path(root_resolved.name) / relative)
    return resolved.name or path.name or raw


def _file_mtime_iso(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return None


def _read_robust_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.stat().st_size > ROBUST_REPORT_MAX_BYTES:
            return {
                "status": "invalid",
                "reason_code": "report_too_large",
                "generated_at": _file_mtime_iso(path),
            }
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return _read_json_object(path)


def _robust_report_timestamp(path: Path, payload: Dict[str, Any]) -> float:
    for key in ("updated_at", "generated_at", "created_at"):
        raw = payload.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _latest_training_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_object(reports_dir / LATEST_TRAINING_RESULT_FILENAME)


def _cash_control_backtest_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_object(reports_dir / CASH_CONTROL_BACKTEST_FILENAME)


def _qlib_predictions_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_object(reports_dir / QLIB_PREDICTIONS_FILENAME)


def _public_artifacts(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    artifacts: Dict[str, str] = {}
    for key, value in raw.items():
        label = _public_path_label(value)
        if label:
            artifacts[str(key)] = label
    return artifacts


def _sanitize_public_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_public_tree(item) for key, item in value.items()
                if key not in {"input_dir", "qmt_data_dir"}}
    if isinstance(value, list):
        return [_sanitize_public_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_public_tree(item) for item in value)
    if isinstance(value, str):
        return sanitize_public_error(value)
    return value


def _candidate_pipeline_payload(
    reports_dir: Path,
    native_resolution: Optional[_NativeQlibProviderResolution] = None,
) -> Optional[Dict[str, Any]]:
    payload = _read_json_object(reports_dir / CANDIDATE_PIPELINE_FILENAME)
    if payload is None:
        return None
    # Older builds published synthetic dry-runs through the formal latest alias.
    # Treat those files as test artifacts, never as trained candidate results.
    if payload.get("dry_run") is True:
        return None
    _annotate_provider_payload(payload, native_resolution)
    if payload.get("provider_uri"):
        payload["provider_uri"] = _public_path_label(payload["provider_uri"])
    payload["artifacts"] = _public_artifacts(payload.get("artifacts"))
    return _sanitize_public_tree(payload)


def _robust_signal_report_row(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    result = {
        key: raw[key]
        for key in (
            "phase",
            "signal_id",
            "model_id",
            "passed",
            "failed_gates",
            "values",
            "quarter_rank_ic",
            "window_stability",
            "topk_sensitivity",
            "prediction_dispersion",
            "monotonicity",
            "returns_after_20bps",
            "board_attribution",
            "market_state_attribution",
            "factor_quality",
        )
        if key in raw
    }
    return _sanitize_robust_tree(result)


def _robust_signal_report_matrix(raw: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        row
        for item in raw[:ROBUST_LIST_PREVIEW_LIMIT]
        if (row := _robust_signal_report_row(item)) is not None
    ]


def approved_robust_signal_data_publication(
    *,
    reports_dir: Path = QUANT_WORKFLOW_REPORTS_ROOT,
    artifact_path: Path | None = None,
    artifact_sha256: str | None = None,
) -> Dict[str, Any]:
    pointer_path = (
        reports_dir / ROBUST_SIGNAL_DIRNAME / "robust_signal_data_readiness_latest.json"
    )
    if artifact_path is None and artifact_sha256 is None:
        artifact_path, artifact_sha256, _payload = resolve_robust_signal_data_readiness(
            pointer_path
        )
    elif artifact_path is None or artifact_sha256 is None:
        raise ValueError("Both readiness artifact path and SHA-256 are required.")
    payload = consume_accepted_robust_signal_data(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        reports_dir=reports_dir,
        provider_resolver=qlib_workflow_provider_uri,
    )
    return {**payload, "readiness_artifact_path": str(artifact_path)}


def _robust_signal_data_readiness_payload(reports_dir: Path) -> Dict[str, Any]:
    try:
        payload = approved_robust_signal_data_publication(
            reports_dir=reports_dir,
        )
    except RobustSignalDataBlocked as exc:
        payload = exc.to_payload()
    return public_readiness_payload(payload)


ROBUST_SIGNAL_STATUS_FIELDS = (
    "status",
    "lifecycle_status",
    "reason_code",
    "run_id",
    "current_stage",
    "pause_state",
    "next_gate",
    "completed_count",
    "expected_count",
    "updated_at",
    "config_hash",
    "git_commit",
    "data_manifest_hash",
    "context_hash",
)


def _robust_signal_status_source(
    signal_dir: Path, alias_name: str
) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    alias = _read_robust_json_object(signal_dir / alias_name)
    if not isinstance(alias, dict):
        return None, None
    run_id = alias.get("run_id")
    run_dir = (
        signal_dir / "runs" / run_id
        if isinstance(run_id, str) and ROBUST_RUN_ID_RE.fullmatch(run_id)
        else None
    )
    if run_dir is not None:
        live_status = _read_robust_json_object(run_dir / "status.json")
        if (
            isinstance(live_status, dict)
            and live_status.get("run_id") == run_id
        ):
            return live_status, run_dir
    return alias, run_dir


def _robust_signal_progress_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _robust_signal_progress_payload(
    status: Dict[str, Any], run_dir: Optional[Path]
) -> Dict[str, Any]:
    progress = {
        key: status[key]
        for key in (
            "current_stage",
            "pause_state",
            "completed_count",
            "expected_count",
        )
        if key in status
    }
    if status.get("lifecycle_status") != "running" or run_dir is None:
        return progress
    stage = status.get("current_stage")
    if stage not in {"screening", "validation"}:
        return progress
    phase_progress = _read_robust_json_object(run_dir / stage / "progress.json")
    if not isinstance(phase_progress, dict):
        return progress
    current_job = phase_progress.get("current_job")
    if isinstance(current_job, str):
        progress["current_job"] = current_job[:240]
    phase_completed = _robust_signal_progress_count(
        phase_progress.get("completed_count")
    )
    if stage == "validation":
        phase_completed += _robust_signal_progress_count(status.get("completed_count"))
    progress["completed_count"] = phase_completed
    progress["expected_count"] = _robust_signal_progress_count(
        status.get("expected_count")
    )
    return progress


def _robust_signal_run_summary(
    status: Dict[str, Any],
    run_dir: Optional[Path],
    *,
    test_only: bool = False,
) -> Dict[str, Any]:
    payload = {
        key: status[key]
        for key in ROBUST_SIGNAL_STATUS_FIELDS
        if key in status
    }
    payload["progress"] = _robust_signal_progress_payload(status, run_dir)
    if test_only:
        payload["test_only"] = True
    return payload


def _robust_signal_payload(reports_dir: Path) -> Dict[str, Any]:
    signal_dir = reports_dir / ROBUST_SIGNAL_DIRNAME
    data_readiness = _robust_signal_data_readiness_payload(reports_dir)
    status, run_dir = _robust_signal_status_source(
        signal_dir, "robust_signal_status_latest.json"
    )
    smoke_status, smoke_run_dir = _robust_signal_status_source(
        signal_dir, "robust_signal_smoke_status_latest.json"
    )
    if not isinstance(status, dict):
        payload: Dict[str, Any] = {
            "status": "missing",
            "lifecycle_status": "not_started",
            "screening_reports": [],
            "validation_reports": [],
        }
        result = None
        contract = None
    else:
        payload = _robust_signal_run_summary(status, run_dir)
        result = _read_robust_json_object(run_dir / "result.json") if run_dir else None
        contract = (
            _read_robust_json_object(run_dir / "factor_contract.json")
            if run_dir
            else None
        )

    def read_reports(phase: str) -> list[Dict[str, Any]]:
        if run_dir is None:
            return []
        report = _read_robust_json_object(
            run_dir / phase / f"{phase}_report.json"
        )
        return _robust_signal_report_matrix(report.get("reports") if report else None)

    payload["smoke_run"] = (
        _robust_signal_run_summary(smoke_status, smoke_run_dir, test_only=True)
        if isinstance(smoke_status, dict)
        else None
    )
    payload["result"] = _sanitize_robust_tree(
        {
            key: result[key]
            for key in ("status", "reason_code", "screening_candidates", "selected_signal_id", "selected_model_id", "challengers")
            if isinstance(result, dict) and key in result
        }
    )
    payload["factor_contract"] = _sanitize_robust_tree(
        {
            key: contract[key]
            for key in (
                "status",
                "signal_id",
                "model_id",
                "feature_family",
                "feature_prefixes",
                "horizon_days",
                "label_expression",
                "signal_cutoff",
                "execution_start",
                "normalization_scope",
                "target",
                "contract_hash",
                "offline_daily_parity_required",
            )
            if isinstance(contract, dict) and key in contract
        }
    )
    payload["screening_reports"] = read_reports("screening")
    payload["validation_reports"] = read_reports("validation")
    payload["data_readiness"] = data_readiness
    return _sanitize_robust_tree(payload)


ROBUST_PRIVATE_KEYS = {
    "argv",
    "command",
    "commands",
    "observations",
    "artifacts",
    "model_artifacts",
}
ROBUST_METRIC_KEYS = {
    "annual_return",
    "benchmark_annual_return",
    "annual_excess_return",
    "information_ratio",
    "sharpe",
    "max_drawdown",
    "maximum_drawdown",
    "average_turnover",
    "rank_ic",
    "rank_ic_mean",
    "days",
    "trading_days",
    "portfolio_return",
    "benchmark_return",
    "net_excess_return",
    "task_success_ratio",
    "first_date",
    "last_date",
    "quarter",
    "market_regime",
    "regime",
}


def _sanitize_robust_tree(value: Any, *, _depth: int = 0) -> Any:
    if _depth >= ROBUST_TREE_MAX_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        result = {}
        items = list(value.items())
        for key, item in items[:ROBUST_MAPPING_PREVIEW_LIMIT]:
            normalized = str(key).lower()
            if (
                normalized in ROBUST_PRIVATE_KEYS
                or normalized.endswith("_path")
                or normalized.endswith("_uri")
                or normalized in {"path", "uri"}
            ):
                continue
            result[key] = _sanitize_robust_tree(item, _depth=_depth + 1)
        if len(items) > ROBUST_MAPPING_PREVIEW_LIMIT:
            result["_truncated"] = True
            result["_total"] = len(items)
        return result
    if isinstance(value, list):
        result = [
            _sanitize_robust_tree(item, _depth=_depth + 1)
            for item in value[:ROBUST_LIST_PREVIEW_LIMIT]
        ]
        if len(value) > ROBUST_LIST_PREVIEW_LIMIT:
            result.append({"_truncated": True, "_total": len(value)})
        return result
    return _sanitize_public_tree(value)


def _missing_robust_section() -> Dict[str, str]:
    return {"status": "missing"}


def _robust_metrics(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    payload = {
        key: _sanitize_robust_tree(value)
        for key, value in raw.items()
        if str(key) in ROBUST_METRIC_KEYS
    }
    return payload


def _robust_verdict(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    result = {
        key: _sanitize_robust_tree(raw[key])
        for key in ("status", "passed", "remaining_days")
        if key in raw
    }
    failed_gates = raw.get("failed_gates")
    if isinstance(failed_gates, list):
        result["failed_gates"] = _sanitize_robust_tree(
            failed_gates[:ROBUST_LIST_PREVIEW_LIMIT]
        )
        result["failed_gates_total"] = len(failed_gates)
        result["failed_gates_truncated"] = len(failed_gates) > ROBUST_LIST_PREVIEW_LIMIT
    gates = raw.get("gates")
    if isinstance(gates, dict):
        gate_items = list(gates.items())[:ROBUST_MAPPING_PREVIEW_LIMIT]
        result["gates"] = {
            str(key): value for key, value in gate_items if isinstance(value, bool)
        }
        invalid_gates = [
            str(key) for key, value in gate_items if not isinstance(value, bool)
        ]
        if invalid_gates:
            result["invalid_gates"] = invalid_gates
        result["gates_total"] = len(gates)
        result["gates_truncated"] = len(gates) > ROBUST_MAPPING_PREVIEW_LIMIT
    return result


def _robust_model_stability(raw: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:ROBUST_LIST_PREVIEW_LIMIT]:
        if not isinstance(item, dict):
            continue
        stability = (
            item.get("stability") if isinstance(item.get("stability"), dict) else {}
        )
        result.append(
            {
                key: value
                for key, value in {
                    "model_id": item.get("model_id"),
                    "eligible": item.get("eligible"),
                    "development_score": item.get("development_score"),
                    "stability": {
                        name: stability[name]
                        for name in ("rank_correlation_median", "topk_overlap_mean")
                        if name in stability
                    },
                }.items()
                if value not in (None, {})
            }
        )
    return _sanitize_robust_tree(result)


def _robust_strategy_row(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    result: Dict[str, Any] = {}
    if raw.get("strategy_id") is not None:
        result["strategy_id"] = str(raw["strategy_id"])
    if isinstance(raw.get("eligible_for_final_selection"), bool):
        result["eligible_for_final_selection"] = raw[
            "eligible_for_final_selection"
        ]
    acceptance = raw.get("acceptance")
    if isinstance(acceptance, dict):
        result["acceptance"] = {
            key: _sanitize_robust_tree(acceptance[key])
            for key in ("passed", "failed_gates", "values", "thresholds")
            if key in acceptance
        }
    for scenario in ("baseline", "stress", "delayed"):
        section = raw.get(scenario)
        if isinstance(section, dict):
            result[scenario] = {"metrics": _robust_metrics(section.get("metrics"))}
    quarter_metrics = raw.get("quarter_metrics")
    if isinstance(quarter_metrics, list):
        result["quarter_metrics"] = [
            _robust_metrics(item)
            for item in quarter_metrics[:ROBUST_LIST_PREVIEW_LIMIT]
            if isinstance(item, dict)
        ]
        result["quarter_metrics_total"] = len(quarter_metrics)
        result["quarter_metrics_truncated"] = (
            len(quarter_metrics) > ROBUST_LIST_PREVIEW_LIMIT
        )
    regime_metrics = raw.get("regime_metrics")
    if isinstance(regime_metrics, dict):
        regime_items = list(regime_metrics.items())
        result["regime_metrics"] = {
            str(name): _robust_metrics(metrics)
            for name, metrics in regime_items[:ROBUST_MAPPING_PREVIEW_LIMIT]
            if isinstance(metrics, dict)
        }
        result["regime_metrics_total"] = len(regime_items)
        result["regime_metrics_truncated"] = (
            len(regime_items) > ROBUST_MAPPING_PREVIEW_LIMIT
        )
    for key in ("board_attribution", "attribution_reconciliation"):
        if isinstance(raw.get(key), dict):
            result[key] = _sanitize_robust_tree(raw[key])
    return _sanitize_robust_tree(result)


def _robust_strategy_matrix(raw: Any) -> list[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [
        row
        for item in raw[:ROBUST_LIST_PREVIEW_LIMIT]
        if (row := _robust_strategy_row(item)) is not None
    ]


def _robust_shadow_daily_payload(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return _missing_robust_section()
    result = {
        key: _sanitize_robust_tree(raw[key])
        for key in (
            "status",
            "reason_code",
            "as_of",
            "outcome",
            "inference_success",
            "appended_days",
            "generated_at",
            "state_status",
            "error_type",
        )
        if key in raw
    }
    result["state_metrics"] = _robust_metrics(raw.get("state_metrics"))
    return result


def _robust_stage_progress(
    robust_dir: Path, orchestration: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not isinstance(orchestration, dict):
        return None
    run_id = orchestration.get("run_id")
    stage = orchestration.get("current_stage")
    if (
        not isinstance(run_id, str)
        or not ROBUST_RUN_ID_RE.fullmatch(run_id)
        or not isinstance(stage, str)
        or stage not in ROBUST_STAGE_PROGRESS_REPORTS
    ):
        return None
    report = _read_robust_json_object(
        robust_dir
        / "runs"
        / run_id
        / "stage_reports"
        / stage
        / ROBUST_STAGE_PROGRESS_REPORTS[stage]
    )
    if not isinstance(report, dict):
        return None
    completed = report.get("completed_date_count")
    expected = report.get("expected_date_count")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(expected, bool)
        or not isinstance(expected, int)
        or completed < 0
        or expected <= 0
        or completed > expected
    ):
        return None
    return {
        key: _sanitize_robust_tree(value)
        for key, value in {
            "stage": stage,
            "status": report.get("status"),
            "completed_date_count": completed,
            "expected_date_count": expected,
            "generated_at": report.get("generated_at"),
        }.items()
        if value is not None
    }


def _robust_strategy_payload(reports_dir: Path) -> Dict[str, Any]:
    robust_dir = reports_dir / ROBUST_STRATEGY_DIRNAME
    orchestration_path = robust_dir / "robust_strategy_status_latest.json"
    preflight_path = robust_dir / "preflight_latest.json"
    development_path = robust_dir / "development_latest.json"
    confirmation_path = robust_dir / "confirmation_latest.json"
    shadow_path = robust_dir / "shadow_state_latest.json"
    shadow_daily_path = robust_dir / "shadow_daily_latest.json"
    orchestration = _read_robust_json_object(orchestration_path)
    preflight_raw = _read_robust_json_object(preflight_path)
    development_raw = _read_robust_json_object(development_path)
    confirmation_raw = _read_robust_json_object(confirmation_path)
    shadow_raw = _read_robust_json_object(shadow_path)
    shadow_daily_raw = _read_robust_json_object(shadow_daily_path)
    shadow_daily = _robust_shadow_daily_payload(shadow_daily_raw)
    progress = _robust_stage_progress(robust_dir, orchestration)
    lifecycle_reports = [
        (orchestration_path, orchestration),
        (preflight_path, preflight_raw),
        (development_path, development_raw),
        (confirmation_path, confirmation_raw),
        (shadow_path, shadow_raw),
    ]
    available_reports = [
        (index, path, payload)
        for index, (path, payload) in enumerate(lifecycle_reports)
        if payload is not None
    ]
    primary_entry = max(
        available_reports,
        key=lambda item: (_robust_report_timestamp(item[1], item[2]), item[0]),
        default=None,
    )
    primary_path = primary_entry[1] if primary_entry is not None else None
    primary = primary_entry[2] if primary_entry is not None else None
    if primary is None:
        return {
            "status": "missing",
            "lifecycle_status": "not_started",
            "stages": [],
            "preflight": _missing_robust_section(),
            "development": _missing_robust_section(),
            "confirmation": _missing_robust_section(),
            "shadow": _missing_robust_section(),
            "shadow_daily": shadow_daily,
            "mlflow_experiments": [
                "robust_csi800_development",
                "robust_csi800_confirmation",
            ],
        }

    stages = []
    stages_raw = (orchestration or {}).get("stages")
    stages_status = (
        "available"
        if isinstance(stages_raw, dict)
        else "invalid"
        if stages_raw is not None
        else "missing"
    )
    stage_items = list(stages_raw.items()) if isinstance(stages_raw, dict) else []
    for name, raw in stage_items[:ROBUST_LIST_PREVIEW_LIMIT]:
        if not isinstance(raw, dict):
            continue
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
        stages.append(
            {
                key: value
                for key, value in {
                    "name": str(name),
                    "status": raw.get("status"),
                    "started_at": raw.get("started_at"),
                    "finished_at": raw.get("finished_at"),
                    "return_code": raw.get("return_code"),
                    "reason_code": evidence.get("reason_code"),
                }.items()
                if value is not None
            }
        )

    if preflight_raw is None:
        preflight: Dict[str, Any] = _missing_robust_section()
    else:
        preflight = {
            key: _sanitize_robust_tree(preflight_raw[key])
            for key in ("status", "reason_code", "generated_at", "as_of_date")
            if key in preflight_raw
        }
        for field in ("summary", "checks"):
            value = preflight_raw.get(field)
            if isinstance(value, list):
                preflight[field] = _sanitize_robust_tree(
                    value[:ROBUST_LIST_PREVIEW_LIMIT]
                )
                preflight[f"{field}_total"] = len(value)
                preflight[f"{field}_truncated"] = len(value) > ROBUST_LIST_PREVIEW_LIMIT
            elif isinstance(value, dict):
                items = list(value.items())
                preflight[field] = _sanitize_robust_tree(
                    dict(items[:ROBUST_MAPPING_PREVIEW_LIMIT])
                )
                preflight[f"{field}_total"] = len(items)
                preflight[f"{field}_truncated"] = (
                    len(items) > ROBUST_MAPPING_PREVIEW_LIMIT
                )
            elif value is not None:
                preflight[field] = _sanitize_robust_tree(value)
    if development_raw is None:
        development: Dict[str, Any] = _missing_robust_section()
    else:
        development = {
            key: _sanitize_robust_tree(development_raw[key])
            for key in (
                "status",
                "reason_code",
                "generated_at",
                "as_of_date",
                "run_id",
                "training_job_count",
                "selected_models",
                "selected_strategy",
            )
            if key in development_raw
        }
        evaluation = (
            development_raw.get("evaluation")
            if isinstance(development_raw.get("evaluation"), dict)
            else {}
        )
        model_reports = development_raw.get("model_reports")
        strategies = evaluation.get("strategies")
        development["model_stability"] = _robust_model_stability(model_reports)
        development["model_stability_total"] = (
            len(model_reports) if isinstance(model_reports, list) else 0
        )
        development["model_stability_truncated"] = (
            isinstance(model_reports, list)
            and len(model_reports) > ROBUST_LIST_PREVIEW_LIMIT
        )
        development["strategy_matrix"] = _robust_strategy_matrix(strategies)
        development["strategy_matrix_total"] = (
            len(strategies) if isinstance(strategies, list) else 0
        )
        development["strategy_matrix_truncated"] = (
            isinstance(strategies, list) and len(strategies) > ROBUST_LIST_PREVIEW_LIMIT
        )
    if confirmation_raw is None:
        confirmation: Dict[str, Any] = _missing_robust_section()
    else:
        confirmation = {
            key: _sanitize_robust_tree(confirmation_raw[key])
            for key in (
                "status",
                "reason_code",
                "generated_at",
                "run_id",
                "confirmation_start",
                "latest_signal_day",
                "latest_market_day",
                "selected_models",
                "selected_strategy",
            )
            if key in confirmation_raw
        }
        confirmation["metrics"] = _robust_metrics(confirmation_raw.get("metrics"))
        confirmation["verdict"] = _robust_verdict(confirmation_raw.get("verdict"))
    if shadow_raw is None:
        shadow: Dict[str, Any] = _missing_robust_section()
    else:
        shadow = {
            key: _sanitize_robust_tree(shadow_raw[key])
            for key in (
                "status",
                "confirmation_run_id",
                "start_after",
                "created_at",
                "updated_at",
            )
            if key in shadow_raw
        }
        shadow["metrics"] = _robust_metrics(shadow_raw.get("metrics"))
        shadow["verdict"] = _robust_verdict(shadow_raw.get("verdict"))
    primary_updated_at = (
        primary.get("updated_at")
        or primary.get("generated_at")
        or primary.get("created_at")
        or (_file_mtime_iso(primary_path) if primary_path is not None else None)
    )
    result = {
        key: value
        for key, value in {
            "status": primary.get("status") or "missing",
            "lifecycle_status": primary.get("lifecycle_status")
            or primary.get("status"),
            "reason_code": primary.get("reason_code"),
            "run_id": primary.get("run_id") or primary.get("confirmation_run_id"),
            "current_stage": primary.get("current_stage"),
            "failed_stage": primary.get("failed_stage"),
            "updated_at": primary_updated_at,
            "stages": stages,
            "stages_status": stages_status,
            "stages_total": len(stage_items),
            "stages_truncated": len(stage_items) > ROBUST_LIST_PREVIEW_LIMIT,
            "preflight": preflight,
            "development": development,
            "confirmation": confirmation,
            "shadow": shadow,
            "shadow_daily": shadow_daily,
            "progress": progress,
            "mlflow_experiments": [
                "robust_csi800_development",
                "robust_csi800_confirmation",
            ],
        }.items()
        if value is not None
    }
    return _sanitize_robust_tree(result)


def _qlib_research_payload(reports_dir: Path) -> Dict[str, Any]:
    sections: Dict[str, Any] = {}
    available_count = 0
    for section, (title, filename) in QLIB_RESEARCH_REPORTS.items():
        path = reports_dir / filename
        payload = _read_json_object(path)
        if payload:
            item = dict(payload)
            item["artifacts"] = _public_artifacts(item.get("artifacts"))
            item["path"] = _public_path_label(path)
            sections[section] = item
            available_count += 1
        else:
            sections[section] = {
                "section": section,
                "title": title,
                "status": "missing",
                "path": _public_path_label(path) if path.exists() else None,
            }
    return {
        "status": "available" if available_count else "missing",
        "available_sections": available_count,
        "total_sections": len(QLIB_RESEARCH_REPORTS),
        "sections": sections,
    }


def _mlflow_payload() -> Dict[str, Any]:
    raw_url = (
        os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL).strip() or DEFAULT_MLFLOW_URL
    )
    payload = {
        "url": public_http_url(raw_url, DEFAULT_MLFLOW_URL),
        "status": "linked",
        "role": "experiment_raw_records",
        "checked_at": None,
        "health_checked": False,
        "error": "Link only; live health is checked on the workflow page for local MLflow URLs, and remote MLflow URLs are not probed by design.",
    }
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL must be an http(s) URL."
    elif parsed.username or parsed.password:
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL must not include username or password."
    else:
        decoded_path = unquote(parsed.path or "")
        if any(part == ".." for part in decoded_path.replace("\\", "/").split("/")):
            payload["status"] = "invalid"
            payload["error"] = "MLFLOW_URL must not contain path traversal."
    return payload


def _data_quality_publication_from_json(
    path: Path,
) -> Optional[_DataQualityPublication]:
    payload, _pointer_bytes = _read_json_object_bytes_at(
        path.parent,
        Path(path.name),
        max_bytes=DATA_QUALITY_POINTER_MAX_BYTES,
    )
    if not payload:
        return None
    if (
        payload.get("schema_version") == 3
        and payload.get("publication_contract")
        == "atomic_json_pointer_to_immutable_quality_evidence_v2"
    ):
        report_id = payload.get("report_id")
        evidence_sha256 = payload.get("evidence_sha256")
        evidence_artifact = payload.get("evidence_artifact")
        if (
            not isinstance(report_id, str)
            or not re.fullmatch(r"data-quality-[0-9a-f]{16}", report_id)
            or not isinstance(evidence_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256)
            or report_id != f"data-quality-{evidence_sha256[:16]}"
            or not isinstance(evidence_artifact, str)
            or not re.fullmatch(
                r"data_quality_report_\d{8}_\d{6}_[0-9a-f]{16}\.json",
                evidence_artifact,
            )
            or not evidence_artifact.endswith(f"_{evidence_sha256[:16]}.json")
        ):
            return None
        evidence_bytes = _read_bounded_regular_file_at(
            path.parent,
            Path(evidence_artifact),
            max_bytes=DATA_QUALITY_EVIDENCE_MAX_BYTES,
        )
        if (
            evidence_bytes is None
            or hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256
        ):
            return None
        try:
            evidence = json.loads(evidence_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema_version") != 3
            or evidence.get("evidence_contract")
            != "immutable_data_quality_evidence_v2"
        ):
            return None
        publication = _validated_quality_markdown(
            path.parent,
            evidence,
            report_id=report_id,
            evidence_sha256=evidence_sha256,
            training_authoritative=True,
        )
        return publication

    if (
        payload.get("schema_version") != 2
        or payload.get("publication_contract")
        != "atomic_json_pointer_to_immutable_markdown_v1"
    ):
        return None
    report_id = payload.get("report_id")
    markdown_sha256 = payload.get("markdown_sha256")
    markdown_artifact = payload.get("markdown_artifact")
    if (
        not isinstance(report_id, str)
        or not re.fullmatch(r"data-quality-[0-9a-f]{16}", report_id)
        or not isinstance(markdown_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", markdown_sha256)
        or report_id != f"data-quality-{markdown_sha256[:16]}"
        or not isinstance(markdown_artifact, str)
        or not re.fullmatch(
            r"data_quality_report_\d{8}_\d{6}_[0-9a-f]{16}\.md",
            markdown_artifact,
        )
    ):
        return None
    return _validated_quality_markdown(
        path.parent,
        payload,
        report_id=report_id,
        evidence_sha256=None,
        training_authoritative=False,
    )


def _validated_quality_markdown(
    reports_dir: Path,
    payload: Dict[str, Any],
    *,
    report_id: str,
    evidence_sha256: Optional[str],
    training_authoritative: bool,
) -> Optional[_DataQualityPublication]:
    markdown_sha256 = payload.get("markdown_sha256")
    markdown_artifact = payload.get("markdown_artifact")
    if (
        not isinstance(markdown_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", markdown_sha256)
        or not isinstance(markdown_artifact, str)
        or not re.fullmatch(
            r"data_quality_report_\d{8}_\d{6}_[0-9a-f]{16}\.md",
            markdown_artifact,
        )
        or not markdown_artifact.endswith(f"_{markdown_sha256[:16]}.md")
    ):
        return None
    markdown_bytes = _read_bounded_regular_file_at(
        reports_dir,
        Path(markdown_artifact),
        max_bytes=DATA_QUALITY_MARKDOWN_MAX_BYTES,
    )
    if (
        markdown_bytes is None
        or hashlib.sha256(markdown_bytes).hexdigest() != markdown_sha256
    ):
        return None
    return _DataQualityPublication(
        payload=dict(payload),
        markdown_bytes=markdown_bytes,
        report_id=report_id,
        evidence_sha256=evidence_sha256,
        training_authoritative=training_authoritative,
    )


def _directory_size_mb(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    cache_key = str(path.expanduser().absolute())
    now = time.monotonic()
    with _DIRECTORY_SIZE_CACHE_LOCK:
        cached = _DIRECTORY_SIZE_CACHE.get(cache_key)
        if cached and now - cached[0] <= DIRECTORY_SIZE_CACHE_TTL_SECONDS:
            return cached[1]

    total = 0
    scanned_entries = 0
    started_at = time.monotonic()
    try:
        for item in path.rglob("*"):
            scanned_entries += 1
            if scanned_entries > DIRECTORY_SIZE_SCAN_MAX_ENTRIES:
                logger.warning(
                    "Directory size scan exceeded %s entries under %s; returning unknown size.",
                    DIRECTORY_SIZE_SCAN_MAX_ENTRIES,
                    path,
                )
                return None
            if (
                scanned_entries % 100 == 0
                and time.monotonic() - started_at > DIRECTORY_SIZE_SCAN_MAX_SECONDS
            ):
                logger.warning(
                    "Directory size scan exceeded %.2f seconds under %s; returning unknown size.",
                    DIRECTORY_SIZE_SCAN_MAX_SECONDS,
                    path,
                )
                return None
            if item.is_symlink():
                continue
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return None
    size_mb = round(total / (1024 * 1024), 2)
    with _DIRECTORY_SIZE_CACHE_LOCK:
        _DIRECTORY_SIZE_CACHE[cache_key] = (time.monotonic(), size_mb)
    return size_mb


def _clear_directory_size_cache() -> None:
    with _DIRECTORY_SIZE_CACHE_LOCK:
        _DIRECTORY_SIZE_CACHE.clear()


def _resolve_for_boundary(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return path.expanduser().absolute()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_manifest_path(root: Path, raw_path: Any) -> Optional[Path]:
    """Resolve a manifest path without allowing it to escape quant_data."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return None
    resolved_root = _resolve_for_boundary(root)
    resolved_candidate = _resolve_for_boundary(root / candidate)
    if not _is_relative_to(resolved_candidate, resolved_root):
        return None
    return resolved_candidate


def _manifest_path_contains_symlink(root: Path, relative_path: Path) -> bool:
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def approved_ashare_qfq_snapshot_identity() -> Dict[str, Any]:
    """Return the current market snapshot only when full quality evidence still matches."""

    pointer_path = QUANT_WORKFLOW_REPORTS_ROOT / "data_quality_report_latest.json"
    publication = _data_quality_publication_from_json(pointer_path)
    if publication is None or publication.training_authoritative is not True:
        raise DataContractError("approved A-share data quality evidence is unavailable")

    quality_payload = publication.payload
    thresholds = quality_payload.get("training_thresholds")
    minimum_instruments = (
        thresholds.get("min_instruments") if isinstance(thresholds, dict) else None
    )
    scanned_instruments = quality_payload.get("scanned_instruments")
    if (
        quality_payload.get("trainable") is not True
        or quality_payload.get("scan_mode") != "full"
        or quality_payload.get("coverage_status") != "full"
        or quality_payload.get("verification_status") != "verified"
        or quality_payload.get("requested_sample") != 0
        or not isinstance(thresholds, dict)
        or thresholds.get("authoritative") is not True
        or not isinstance(minimum_instruments, int)
        or isinstance(minimum_instruments, bool)
        or minimum_instruments < AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS
        or not isinstance(scanned_instruments, int)
        or isinstance(scanned_instruments, bool)
        or scanned_instruments < minimum_instruments
    ):
        raise DataContractError("A-share data quality evidence is not a verified full scan")

    data_contract = quality_payload.get("data_contract")
    identity = (
        data_contract.get(ASHARE_DAILY_QFQ_DATASET_ID)
        if isinstance(data_contract, dict)
        else None
    )
    snapshot_state = (
        data_contract.get("snapshot_state")
        if isinstance(data_contract, dict)
        else None
    )
    identity_fields = (
        "dataset_id",
        "snapshot_id",
        "manifest_sha256",
        "content_sha256",
    )
    if (
        not isinstance(data_contract, dict)
        or data_contract.get("status") != "verified"
        or not isinstance(identity, dict)
        or identity.get("dataset_id") != ASHARE_DAILY_QFQ_DATASET_ID
        or not isinstance(identity.get("snapshot_id"), str)
        or not identity.get("snapshot_id")
        or Path(str(identity.get("snapshot_id"))).name != identity.get("snapshot_id")
        or identity.get("snapshot_id") in {".", ".."}
        or any(
            not isinstance(identity.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(identity.get(field))) is None
            for field in ("manifest_sha256", "content_sha256")
        )
    ):
        raise DataContractError("A-share data quality evidence has an invalid identity")

    snapshot_stages = ("before", "after_scan", "after_revalidation", "after")
    stage_states = [
        snapshot_state.get(stage, {}).get(ASHARE_DAILY_QFQ_DATASET_ID)
        if isinstance(snapshot_state, dict)
        and isinstance(snapshot_state.get(stage), dict)
        else None
        for stage in snapshot_stages
    ]
    expected_state = stage_states[0]
    if (
        not isinstance(snapshot_state, dict)
        or snapshot_state.get("stable") is not True
        or snapshot_state.get("content_revalidated") is not True
        or not all(_valid_filesystem_state_fingerprint(state) for state in stage_states)
        or not all(state == expected_state for state in stage_states[1:])
    ):
        raise DataContractError("A-share snapshot was not stable during the quality scan")

    root = _resolve_for_boundary(QUANT_DATA_ROOT)
    catalog_path = root / DATA_MANIFEST_FILENAME
    if catalog_path.is_symlink():
        raise DataContractError("A-share root catalog path is unsafe")
    catalog = _read_json_object(catalog_path)
    datasets = catalog.get("datasets") if isinstance(catalog, dict) else None
    catalog_entry = (
        datasets.get(ASHARE_DAILY_QFQ_DATASET_ID)
        if isinstance(datasets, dict)
        else None
    )
    expected_pointer = ASHARE_DAILY_QFQ_DATASET_ROOT / "LATEST.json"
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 2
        or catalog.get("path_format") != "relative_to_quant_data_root"
        or not isinstance(catalog_entry, dict)
        or catalog_entry.get("dataset_id") != ASHARE_DAILY_QFQ_DATASET_ID
        or catalog_entry.get("dataset_root")
        != ASHARE_DAILY_QFQ_DATASET_ROOT.as_posix()
        or catalog_entry.get("role") != "canonical"
        or catalog_entry.get("status") != "ready"
        or catalog_entry.get("price_basis") != "qfq"
        or catalog_entry.get("latest_pointer") != expected_pointer.as_posix()
        or catalog_entry.get("snapshot_id") != identity.get("snapshot_id")
        or catalog_entry.get("snapshot_manifest_sha256")
        != identity.get("manifest_sha256")
        or catalog_entry.get("content_sha256") != identity.get("content_sha256")
    ):
        raise DataContractError("A-share root catalog does not match quality evidence")

    live_pointer_path = root / expected_pointer
    if _manifest_path_contains_symlink(root, expected_pointer):
        raise DataContractError("A-share latest pointer path is unsafe")
    try:
        latest_pointer, snapshot = validate_latest_pointer(
            live_pointer_path,
            quant_data_root=root,
            expected_dataset_id=ASHARE_DAILY_QFQ_DATASET_ID,
        )
    except (DataContractError, OSError, UnicodeError, TypeError, ValueError) as exc:
        raise DataContractError("A-share latest pointer is invalid") from exc
    expected_snapshot_relative = (
        ASHARE_DAILY_QFQ_DATASET_ROOT
        / "snapshots"
        / str(identity["snapshot_id"])
    )
    if (
        latest_pointer.get("snapshot_id") != identity.get("snapshot_id")
        or latest_pointer.get("manifest_sha256") != identity.get("manifest_sha256")
        or latest_pointer.get("snapshot_path")
        != expected_snapshot_relative.as_posix()
        or _manifest_path_contains_symlink(root, expected_snapshot_relative)
    ):
        raise DataContractError("A-share latest pointer does not match quality evidence")

    manifest, manifest_bytes = _read_json_object_bytes(
        snapshot / "DATASET_MANIFEST.json"
    )
    actual_manifest_sha256 = (
        hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_bytes is not None
        else None
    )
    if (
        not isinstance(manifest, dict)
        or actual_manifest_sha256 != identity.get("manifest_sha256")
        or any(manifest.get(field) != identity.get(field) for field in identity_fields[:2])
        or manifest.get("content_sha256") != identity.get("content_sha256")
        or manifest.get("data_kind") != "daily_bars"
        or manifest.get("role") != "canonical"
        or manifest.get("status") != "ready"
        or manifest.get("price_basis") != "qfq"
    ):
        raise DataContractError("A-share snapshot manifest does not match quality evidence")
    bars_path = snapshot / "bars"
    if bars_path.is_symlink() or not bars_path.is_dir():
        raise DataContractError("A-share snapshot bars directory is unavailable")

    try:
        current_state = _filesystem_state_fingerprint_live(snapshot)
    except (DataContractError, OSError) as exc:
        raise DataContractError("A-share snapshot state cannot be verified") from exc
    if current_state != expected_state:
        raise DataContractError("A-share snapshot changed after the quality scan")

    return {
        **{field: str(identity[field]) for field in identity_fields},
        "state_fingerprint": dict(expected_state),
    }


def _provider_name_for_request(dataset_id: str, provider: str) -> str:
    requested = str(provider or "canonical").strip().lower()
    if requested == "canonical" and str(dataset_id or "").endswith("_cleaned_v2"):
        requested = "cleaned"
    if requested not in QLIB_PROVIDER_NAMES:
        raise ValueError("Qlib Provider must be one of: canonical, cleaned.")
    return requested


def _required_training_datasets(provider_name: str) -> Tuple[str, ...]:
    required = list(QLIB_TRAINING_REQUIRED_DATASETS)
    if provider_name == "cleaned":
        required.extend(QLIB_CLEANED_REQUIRED_DATASETS)
    return tuple(required)


def _blocked_training_preflight(
    payload: Dict[str, Any],
    *,
    summary: str,
    provider_name: str,
    required_datasets: Tuple[str, ...],
) -> None:
    checked_at = datetime.now(timezone.utc).isoformat()
    payload["trainable"] = False
    payload["training_readiness"] = {
        "status": "blocked",
        "summary": summary,
    }
    payload["training_preflight"] = {
        "status": "blocked",
        "provider": provider_name,
        "required_datasets": list(required_datasets),
        "checked_at": checked_at,
        "summary": summary,
    }


def _training_preflight_payload(
    native_resolution: _NativeQlibProviderResolution,
    *,
    identities: Optional[Mapping[str, QuantDataIdentity]] = None,
    service_error: Optional[str] = None,
) -> Dict[str, Any]:
    provider_name = native_resolution.provider_name or "canonical"
    required_datasets = _required_training_datasets(provider_name)
    checked_at = datetime.now(timezone.utc).isoformat()
    if native_resolution.path is None:
        return {
            "status": "blocked",
            "provider": provider_name,
            "required_datasets": list(required_datasets),
            "checked_at": checked_at,
            "summary": native_resolution.error or "当前 Qlib Provider 不可用。",
        }

    if service_error is not None or identities is None:
        return {
            "status": "blocked",
            "provider": provider_name,
            "required_datasets": list(required_datasets),
            "checked_at": checked_at,
            "summary": service_error or "quant_data 数据后端暂不可用，正式训练已阻止。",
        }

    missing = [dataset_id for dataset_id in required_datasets if dataset_id not in identities]
    if missing:
        return {
            "status": "blocked",
            "provider": provider_name,
            "required_datasets": list(required_datasets),
            "available_datasets": sorted(identities),
            "checked_at": checked_at,
            "summary": (
                "quant_data 缺少正式训练所需的 canonical 数据集："
                f"{', '.join(missing)}。"
            ),
        }

    provider_identity_payload = _resolution_lineage(native_resolution)
    if provider_identity_payload is None:
        return {
            "status": "blocked",
            "provider": provider_name,
            "required_datasets": list(required_datasets),
            "available_datasets": sorted(identities),
            "checked_at": checked_at,
            "summary": "当前 Qlib Provider 缺少完整的不可变数据血缘，正式训练已阻止。",
        }

    if provider_identity_payload.get("provider_contract") == "direct_parquet_v1":
        for prefix in ("", "index_"):
            dataset_id = provider_identity_payload.get(f"{prefix}dataset_id")
            identity = identities.get(dataset_id)
            if identity is None or any(
                getattr(identity, field) != provider_identity_payload.get(f"{prefix}{field}")
                for field in ("snapshot_id", "manifest_sha256", "content_sha256")
            ):
                return {
                    "status": "blocked", "provider": provider_name,
                    "required_datasets": list(required_datasets),
                    "available_datasets": sorted(identities), "checked_at": checked_at,
                    "summary": "本地快照与当前 Qlib Provider 的冻结身份不一致，请刷新 Provider 后再训练。",
                }

    summary = "本地快照已具备所需数据集，当前 Qlib Provider 轻量预检通过。"
    return {
        "status": "passed",
        "provider": provider_name,
        "required_datasets": list(required_datasets),
        "available_datasets": sorted(identities),
        "dataset_identities": {
            dataset_id: identities[dataset_id].as_dict()
            for dataset_id in required_datasets
        },
        "provider_identity": provider_identity_payload,
        "checked_at": checked_at,
        "summary": summary,
    }


def _load_training_dataset_identities(*, provider_name: str = "canonical") -> Tuple[
    Optional[Dict[str, QuantDataIdentity]], Optional[str]
]:
    try:
        # Explicit historical replays use DSA's existing contract. The QMT
        # experiment path never enters this branch or imports the retired package.
        identities = {}
        for dataset_id in _required_training_datasets(provider_name):
            resolved = resolve_dataset(dataset_id, quant_data_root=QUANT_DATA_ROOT)
            identities[dataset_id] = QuantDataIdentity.from_mapping({
                **resolved.manifest,
                "manifest_sha256": sha256_file(resolved.snapshot_path / "DATASET_MANIFEST.json"),
            })
    except (DataContractError, OSError, QuantDataClientError, ValueError):
        return None, "历史实验依赖的本地快照缺失或身份校验失败。"
    return (
        identities,
        None,
    )


def _apply_training_preflight(
    payload: Dict[str, Any],
    native_resolution: _NativeQlibProviderResolution,
    *,
    identities: Optional[Mapping[str, QuantDataIdentity]] = None,
    service_error: Optional[str] = None,
) -> None:
    if identities is None and service_error is None and native_resolution.path is not None:
        identities, service_error = _load_training_dataset_identities(provider_name=native_resolution.provider_name)
    preflight = _training_preflight_payload(
        native_resolution,
        identities=identities,
        service_error=service_error,
    )
    payload["training_preflight"] = preflight
    payload["trainable"] = preflight["status"] == "passed"
    payload["training_readiness"] = {
        "status": "trainable" if payload["trainable"] else "blocked",
        "summary": preflight["summary"],
    }


def _resolution_lineage(
    resolution: _NativeQlibProviderResolution,
) -> Optional[Dict[str, str]]:
    if not isinstance(resolution.lineage, Mapping):
        return None
    try:
        return normalize_provider_lineage(resolution.lineage)
    except RunProvenanceError:
        return None


def _resolution_common_identity(
    resolution: _NativeQlibProviderResolution,
) -> Dict[str, str]:
    lineage = _resolution_lineage(resolution)
    if lineage is not None:
        return {
            field: lineage[field]
            for field in (
                "dataset_id",
                "snapshot_id",
                "manifest_sha256",
                "content_sha256",
            )
        }
    values = {
        "dataset_id": resolution.dataset_id,
        "snapshot_id": resolution.snapshot_id,
        "manifest_sha256": resolution.manifest_sha256,
        "content_sha256": resolution.content_sha256,
    }
    return {
        field: str(value)
        for field, value in values.items()
        if isinstance(value, str) and value
    }


def _resolution_from_provider_uri(
    provider_uri: Path,
    *,
    provider_name: str,
) -> _NativeQlibProviderResolution:
    provider_uri = Path(provider_uri).expanduser()
    lineage = normalize_provider_lineage(
        provider_identity(provider_uri, quant_data_root=QUANT_DATA_ROOT)
    )
    calendar = provider_calendar_days(provider_uri)
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("Qlib Provider calendar is invalid.")
    instruments = provider_instrument_ranges(provider_uri)
    if not instruments:
        raise ValueError("Qlib Provider instruments are unavailable.")
    benchmark_complete = provider_has_instrument_pool(provider_uri, "csi300")
    return _NativeQlibProviderResolution(
        authoritative=True,
        path=provider_uri,
        lineage=lineage,
        provider_name=provider_name,
        dataset_id=lineage["dataset_id"],
        snapshot_id=lineage["snapshot_id"],
        manifest_sha256=lineage["manifest_sha256"],
        content_sha256=lineage["content_sha256"],
        calendar_end=calendar[-1],
        instrument_count=len(instruments),
        benchmark_complete=benchmark_complete,
    )


def _resolve_native_qlib_provider(
    dataset_id: str = ASHARE_DAILY_QFQ_DATASET_ID,
    *,
    provider: str = "canonical",
) -> _NativeQlibProviderResolution:
    if _resolve_for_boundary(RESEARCH_STATUS_PATH.parent) != _resolve_for_boundary(
        QUANT_WORKFLOW_REPORTS_ROOT
    ):
        return _NativeQlibProviderResolution(authoritative=False)

    try:
        provider_name = _provider_name_for_request(dataset_id, provider)
        provider_uri = qlib_workflow_provider_uri(provider_name)
        return _resolution_from_provider_uri(
            provider_uri,
            provider_name=provider_name,
        )
    except (
        FileNotFoundError,
        OSError,
        RunProvenanceError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        provider_name = (
            str(provider or "canonical").strip().lower()
            if str(provider or "canonical").strip().lower() in QLIB_PROVIDER_NAMES
            else "canonical"
        )
        logger.warning("Qlib Provider resolution failed: %s", exc)
        return _NativeQlibProviderResolution(
            authoritative=True,
            provider_name=provider_name,
            error="Qlib Provider 不可用：quant_data 数据后端未通过完整性校验。",
        )


def _current_native_provider_resolutions(
    *,
    requested_dataset_id: str,
    requested_resolution: _NativeQlibProviderResolution,
    requested_provider: str = "canonical",
) -> Dict[str, _NativeQlibProviderResolution]:
    requested_name = _provider_name_for_request(
        requested_dataset_id,
        requested_provider,
    )
    return {
        provider_name: (
            requested_resolution
            if provider_name == requested_name
            else _resolve_native_qlib_provider(provider=provider_name)
        )
        for provider_name in QLIB_PROVIDER_NAMES
    }


def _current_native_provider_identities(
    resolutions: Mapping[str, _NativeQlibProviderResolution],
) -> Dict[str, Dict[str, Any]]:
    identities: Dict[str, Dict[str, Any]] = {}
    for provider_name, resolution in resolutions.items():
        identity: Dict[str, Any] = {
            "provider": provider_name,
            "status": "ready" if resolution.path is not None else "blocked",
            "calendar_end": resolution.calendar_end,
            "instruments": resolution.instrument_count,
            "benchmark_complete": resolution.benchmark_complete,
        }
        identity.update(_resolution_common_identity(resolution))
        lineage = _resolution_lineage(resolution)
        if lineage is not None:
            identity["lineage_sha256"] = hashlib.sha256(
                json.dumps(
                    lineage,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        if resolution.error:
            identity["error"] = resolution.error
        identities[provider_name] = {
            key: value for key, value in identity.items() if value is not None
        }
    return identities


def _attach_training_preflights(
    payload: Dict[str, Any],
    *,
    requested_dataset_id: str,
    requested_resolution: _NativeQlibProviderResolution,
    requested_provider: str,
) -> None:
    resolutions = _current_native_provider_resolutions(
        requested_dataset_id=requested_dataset_id,
        requested_resolution=requested_resolution,
        requested_provider=requested_provider,
    )
    payload["current_provider_identities"] = _current_native_provider_identities(
        resolutions
    )
    payload["training_preflights"] = {}
    for provider_name, resolution in resolutions.items():
        dataset_identities, service_error = (
            _load_training_dataset_identities(provider_name=provider_name)
            if resolution.path is not None else (None, None)
        )
        payload["training_preflights"][provider_name] = _training_preflight_payload(
            resolution,
            identities=dataset_identities,
            service_error=service_error,
        )
    requested_name = _provider_name_for_request(
        requested_dataset_id,
        requested_provider,
    )
    selected_preflight = payload["training_preflights"][requested_name]
    payload["training_preflight"] = selected_preflight
    payload["trainable"] = selected_preflight["status"] == "passed"
    payload["training_readiness"] = {
        "status": "trainable" if payload["trainable"] else "blocked",
        "summary": selected_preflight["summary"],
    }


def _deprecate_legacy_quality_status(payload: Dict[str, Any]) -> None:
    """Keep old response keys inert while callers migrate away from them."""

    payload["quality_report_id"] = None
    payload["quality_evidence_sha256"] = None
    payload["quality_scan"] = {}
    payload["provider_quality"] = {}
    payload["data_sources"] = []


def _saved_provider_lineage(payload: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    raw_identity = payload.get("provider_identity")
    if not isinstance(raw_identity, Mapping):
        return None
    try:
        return normalize_provider_lineage(raw_identity)
    except RunProvenanceError:
        return None


def _lineage_matches_resolution(
    lineage: Mapping[str, Any] | None,
    resolution: _NativeQlibProviderResolution,
) -> bool:
    current = _resolution_lineage(resolution)
    if current is not None:
        return provider_lineages_match(lineage, current)
    if not isinstance(lineage, Mapping):
        return False
    expected = _resolution_common_identity(resolution)
    return len(expected) == 4 and all(
        lineage.get(field) == value for field, value in expected.items()
    )


def _annotate_saved_provider(
    payload: Dict[str, Any],
    resolution: _NativeQlibProviderResolution,
    *,
    artifact_label: str,
) -> None:
    if not resolution.authoritative:
        return

    target_resolution = _current_provider_resolution_for_payload(payload, resolution)
    if target_resolution.path is None:
        payload["provider_current"] = False
        payload["provider_status"] = "catalog_invalid"
        payload["provider_warning"] = (
            target_resolution.error or "当前 Qlib Provider 无法验证。"
        )
        return

    lineage = _saved_provider_lineage(payload)
    if lineage is not None:
        current = _lineage_matches_resolution(lineage, target_resolution)
        payload["provider_current"] = current
        payload["provider_status"] = "current" if current else "stale"
        if not current:
            payload["provider_warning"] = (
                f"{artifact_label}记录的 Qlib Provider 血缘与当前数据后端不一致，"
                "只能作为历史结果。"
            )
        else:
            payload.pop("provider_warning", None)
        return

    provider_uri = payload.get("provider_uri") or payload.get("qlib_provider")
    path_points_to_current = bool(
        isinstance(provider_uri, str)
        and provider_uri.strip()
        and _resolve_for_boundary(Path(provider_uri).expanduser())
        == _resolve_for_boundary(target_resolution.path)
    )
    payload["provider_current"] = False
    payload["provider_status"] = "unknown" if path_points_to_current else "stale"
    payload["provider_warning"] = (
        f"{artifact_label}没有记录完整的 Qlib Provider 血缘，"
        "无法证明它使用了当前 quant_data 数据后端。"
    )


def _current_provider_resolution_for_payload(
    payload: Mapping[str, Any],
    fallback: _NativeQlibProviderResolution,
) -> _NativeQlibProviderResolution:
    raw_provider = str(payload.get("provider") or fallback.provider_name or "canonical")
    provider_name = raw_provider.strip().lower()
    if provider_name not in QLIB_PROVIDER_NAMES:
        provider_name = fallback.provider_name
    if provider_name == fallback.provider_name:
        return fallback
    return _resolve_native_qlib_provider(provider=provider_name)


def _resolve_saved_provider(
    lineage: Mapping[str, Any],
    *,
    provider_name: str,
) -> _NativeQlibProviderResolution:
    try:
        normalized = normalize_provider_lineage(lineage)
        provider_uri = resolve_immutable_provider_identity(
            normalized,
            quant_data_root=QUANT_DATA_ROOT,
        )
        resolution = _resolution_from_provider_uri(
            provider_uri,
            provider_name=provider_name,
        )
        if not provider_lineages_match(normalized, resolution.lineage):
            raise RunProvenanceError("Qlib Provider lineage changed.")
        return resolution
    except (
        FileNotFoundError,
        OSError,
        RunProvenanceError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            provider_name=provider_name,
            lineage=(
                dict(lineage)
                if provider_lineage_is_complete(lineage)
                else None
            ),
            error="历史 Qlib Provider 血缘无法验证。",
        )


def _saved_provider_resolution_for_payload(
    payload: Dict[str, Any],
    fallback: _NativeQlibProviderResolution,
) -> _NativeQlibProviderResolution:
    current = _current_provider_resolution_for_payload(payload, fallback)
    lineage = _saved_provider_lineage(payload)
    if lineage is None:
        return _NativeQlibProviderResolution(
            authoritative=True,
            provider_name=current.provider_name,
            error="历史结果没有完整的 Qlib Provider 血缘。",
        )
    if current.path is not None and _lineage_matches_resolution(lineage, current):
        return current
    return _resolve_saved_provider(
        lineage,
        provider_name=current.provider_name,
    )


def _annotate_provider_payload(
    payload: Dict[str, Any],
    native_resolution: Optional[_NativeQlibProviderResolution] = None,
) -> None:
    resolution = native_resolution or _resolve_native_qlib_provider(
        provider="canonical"
    )
    _annotate_saved_provider(payload, resolution, artifact_label="候选报告")


def _qlib_provider_from_path(path: Path) -> Dict[str, Any]:
    provider_uri = Path(path).expanduser()
    provider: Dict[str, Any] = {"path": str(provider_uri)}
    try:
        calendar = provider_calendar_days(provider_uri)
        instruments = provider_instrument_ranges(provider_uri)
        lineage = normalize_provider_lineage(
            provider_identity(provider_uri, quant_data_root=QUANT_DATA_ROOT)
        )
        provider.update(
            {
                "dataset_id": lineage["dataset_id"],
                "snapshot_id": lineage["snapshot_id"],
                "manifest_sha256": lineage["manifest_sha256"],
                "content_sha256": lineage["content_sha256"],
                "calendar_days": len(calendar),
                "instruments": len(instruments),
                "feature_dirs": len(instruments),
                "benchmark_complete": provider_has_instrument_pool(
                    provider_uri,
                    "csi300",
                ),
                "calendar_range": (
                    f"{calendar[0]} ~ {calendar[-1]}" if calendar else ""
                ),
            }
        )
    except (
        FileNotFoundError,
        OSError,
        RunProvenanceError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        provider["status"] = "blocked"
        provider["provider_error"] = (
            sanitize_public_error(str(exc)) or "Qlib Provider 无法验证。"
        )
    return provider


def _annotate_latest_training_provider(
    latest_training: Dict[str, Any],
    native_resolution: _NativeQlibProviderResolution,
) -> None:
    _annotate_saved_provider(
        latest_training,
        native_resolution,
        artifact_label="训练结果",
    )


def _sync_qlib_provider_with_latest_training(
    payload: Dict[str, Any],
    latest_training: Dict[str, Any],
    *,
    native_resolution: _NativeQlibProviderResolution,
) -> None:
    existing = payload.get("qlib_provider")
    provider = dict(existing) if isinstance(existing, dict) else {}
    target_resolution = _saved_provider_resolution_for_payload(
        latest_training,
        native_resolution,
    )
    saved_lineage = _saved_provider_lineage(latest_training)

    if target_resolution.path is None:
        warning = latest_training.get("provider_warning") or (
            target_resolution.error
            or "latest_training 没有记录完整的 Qlib Provider 血缘，无法安全复现。"
        )
        payload["latest_training_provider"] = {
            "status": "blocked",
            "provider_source": "Qlib Provider",
            "provider_error": target_resolution.error,
            "provider_current": False,
            "provider_trusted": False,
            "provider_warning": warning,
        }
        provider["latest_training_provider_trusted"] = False
        provider["latest_training_provider_current"] = False
        provider["provider_uri_warning"] = warning
        payload["qlib_provider"] = provider
        return

    training_provider = _qlib_provider_from_path(target_resolution.path)
    identity_trusted = bool(
        saved_lineage is not None
        and _lineage_matches_resolution(saved_lineage, target_resolution)
    )
    provider_current = bool(
        latest_training.get("provider_current") is True
        and _lineage_matches_resolution(saved_lineage, target_resolution)
    )
    training_provider.update(
        {
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "provider_source": (
                "Qlib Provider"
                if provider_current
                else "historical Qlib Provider"
            ),
            "provider_current": provider_current,
            "provider_trusted": identity_trusted,
        }
    )
    provider["latest_training_provider_trusted"] = identity_trusted
    provider["latest_training_provider_current"] = provider_current

    if provider_current:
        provider.pop("latest_training_provider_uri", None)
        provider.pop("provider_uri_warning", None)
    elif identity_trusted:
        warning = latest_training.get("provider_warning") or (
            "latest_training 使用的是已验证的历史 Qlib Provider 血缘；"
            "它不是当前训练入口，但可用于复现当时结果。"
        )
        provider["provider_uri_warning"] = warning
        training_provider["provider_warning"] = warning
    else:
        warning = latest_training.get("provider_warning") or (
            "latest_training 无法证明使用了当前 Qlib Provider，"
            "请重新运行训练后再把该结果作为当前结果使用。"
        )
        provider["provider_uri_warning"] = warning
        training_provider["provider_warning"] = warning

    payload["latest_training_provider"] = training_provider
    payload["qlib_provider"] = provider


def _experiment_notebook_payload_unlocked(reports_dir: Path) -> Dict[str, Any]:
    log_path = reports_dir / EXPERIMENT_LOG_FILENAME
    runs_path = reports_dir / EXPERIMENT_RUNS_FILENAME
    latest_note = ""
    runs = []
    invalid_lines = 0

    if log_path.exists():
        try:
            latest_note = log_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            latest_note = f"Experiment log could not be read: {exc}"

    if runs_path.exists():
        try:
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if isinstance(item, dict):
                    runs.append(item)
                else:
                    invalid_lines += 1
        except OSError:
            invalid_lines += 1

    status = "available" if latest_note or runs else "missing"
    return {
        "status": status,
        "latest_note": latest_note,
        "runs": runs[-20:][::-1],
        "total_runs": len(runs),
        "invalid_lines": invalid_lines,
        "log_path": _public_path_label(log_path),
        "runs_path": _public_path_label(runs_path),
    }


def _experiment_notebook_payload(reports_dir: Path) -> Dict[str, Any]:
    with _experiment_file_guard(reports_dir):
        return _experiment_notebook_payload_unlocked(reports_dir)


def _tonghuashun_candidate_paths() -> list[Path]:
    paths: list[Path] = []
    configured = os.environ.get("TONGHUASHUN_APP_PATH")
    if configured:
        paths.append(Path(configured).expanduser())
    paths.extend(Path(path).expanduser() for path in TONGHUASHUN_CANDIDATE_PATHS)
    return paths


def _tonghuashun_status_payload() -> Dict[str, Any]:
    detected_path = next(
        (path for path in _tonghuashun_candidate_paths() if path.exists()), None
    )
    installed = detected_path is not None
    return {
        "name": "同花顺桌面端",
        "status": "available" if installed else "missing",
        "installed": installed,
        "mode": "read_only",
        "path": str(detected_path) if detected_path else None,
        "summary": (
            "已检测到本机同花顺，可作为后续只读观察层。"
            if installed
            else "未在默认路径检测到同花顺；如已安装到自定义位置，可设置 TONGHUASHUN_APP_PATH。"
        ),
        "capabilities": [
            "本机应用存在检测",
            "后续可接入只读窗口观察或截图/OCR",
        ],
        "restrictions": [
            "不读取账号密码",
            "不自动下单",
            "不连接券商交易接口",
        ],
    }


def _read_data_hygiene_bundle(
    reports_dir: Path,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Path,
    Path,
    str,
]:
    pointer_path = reports_dir / "data_hygiene_reports_latest.json"
    legacy_inventory_path = reports_dir / "data_hygiene_inventory_latest.json"
    legacy_cleanup_path = reports_dir / "data_hygiene_cleanup_plan_latest.json"
    if not pointer_path.exists():
        inventory, _inventory_bytes = _read_json_object_bytes_at(
            reports_dir,
            Path(legacy_inventory_path.name),
            max_bytes=DATA_HYGIENE_ARTIFACT_MAX_BYTES,
        )
        cleanup_plan, _cleanup_bytes = _read_json_object_bytes_at(
            reports_dir,
            Path(legacy_cleanup_path.name),
            max_bytes=DATA_HYGIENE_ARTIFACT_MAX_BYTES,
        )
        catalog = inventory.get("catalog") if isinstance(inventory, dict) else None
        if (
            isinstance(catalog, dict)
            and catalog.get("validation_contract_version") == 2
        ):
            return None, None, pointer_path, pointer_path, "stale"
        return (
            inventory,
            cleanup_plan,
            legacy_inventory_path,
            legacy_cleanup_path,
            "legacy" if inventory or cleanup_plan else "missing",
        )

    pointer, _pointer_bytes = _read_json_object_bytes_at(
        reports_dir,
        Path(pointer_path.name),
        max_bytes=DATA_HYGIENE_POINTER_MAX_BYTES,
    )
    if not pointer or (
        pointer.get("schema_version") != 1
        or pointer.get("publication_contract")
        != "atomic_json_pointer_to_immutable_hygiene_bundle_v1"
    ):
        return None, None, pointer_path, pointer_path, "invalid"

    bundle_id = pointer.get("bundle_id")
    inventory_name = pointer.get("inventory_artifact")
    cleanup_name = pointer.get("cleanup_plan_artifact")
    inventory_sha256 = pointer.get("inventory_sha256")
    cleanup_sha256 = pointer.get("cleanup_plan_sha256")
    inventory_match = (
        re.fullmatch(
            r"data_hygiene_inventory_(\d{8}_\d{6}_[0-9a-f]{16})\.json",
            inventory_name,
        )
        if isinstance(inventory_name, str)
        else None
    )
    cleanup_match = (
        re.fullmatch(
            r"data_hygiene_cleanup_plan_(\d{8}_\d{6}_[0-9a-f]{16})\.json",
            cleanup_name,
        )
        if isinstance(cleanup_name, str)
        else None
    )
    if (
        not isinstance(bundle_id, str)
        or not re.fullmatch(r"data-hygiene-[0-9a-f]{16}", bundle_id)
        or inventory_match is None
        or cleanup_match is None
        or inventory_match.group(1) != cleanup_match.group(1)
        or not isinstance(inventory_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", inventory_sha256)
        or not isinstance(cleanup_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", cleanup_sha256)
    ):
        return None, None, pointer_path, pointer_path, "invalid"

    inventory_path = reports_dir / str(inventory_name)
    cleanup_path = reports_dir / str(cleanup_name)
    inventory, inventory_bytes = _read_json_object_bytes_at(
        reports_dir,
        Path(str(inventory_name)),
        max_bytes=DATA_HYGIENE_ARTIFACT_MAX_BYTES,
    )
    cleanup_plan, cleanup_bytes = _read_json_object_bytes_at(
        reports_dir,
        Path(str(cleanup_name)),
        max_bytes=DATA_HYGIENE_ARTIFACT_MAX_BYTES,
    )
    if (
        inventory is None
        or inventory_bytes is None
        or cleanup_plan is None
        or cleanup_bytes is None
        or hashlib.sha256(inventory_bytes).hexdigest() != inventory_sha256
        or hashlib.sha256(cleanup_bytes).hexdigest() != cleanup_sha256
    ):
        return None, None, pointer_path, pointer_path, "invalid"
    bundle_digest = hashlib.sha256(
        inventory_bytes + b"\0" + cleanup_bytes
    ).hexdigest()
    if bundle_id != f"data-hygiene-{bundle_digest[:16]}":
        return None, None, pointer_path, pointer_path, "invalid"
    return inventory, cleanup_plan, inventory_path, cleanup_path, "available"


def _data_hygiene_payload(reports_dir: Path) -> Dict[str, Any]:
    (
        inventory,
        cleanup_plan,
        inventory_path,
        cleanup_plan_path,
        hygiene_publication_status,
    ) = _read_data_hygiene_bundle(reports_dir)
    cleaning_strategy_path = reports_dir / "data_cleaning_strategy_latest.md"
    manifest_path = QUANT_DATA_ROOT / "DATA_MANIFEST.json"
    readme_path = QUANT_DATA_ROOT / "README.md"
    maintenance_lock_path = QUANT_DATA_ROOT / ".maintenance.lock"

    manifest, manifest_bytes = _read_json_object_bytes_at(
        QUANT_DATA_ROOT,
        Path(DATA_MANIFEST_FILENAME),
        max_bytes=DATA_HYGIENE_ARTIFACT_MAX_BYTES,
    )
    manifest_sha256 = (
        hashlib.sha256(manifest_bytes).hexdigest()
        if manifest is not None and manifest_bytes is not None
        else None
    )
    maintenance_locked = maintenance_lock_path.exists()

    # Extract cleanup plan actions
    cleanup_actions = []
    cleanup_plan_actions = cleanup_plan.get("actions", []) if cleanup_plan else []
    if not isinstance(cleanup_plan_actions, list):
        cleanup_plan_actions = []
    if cleanup_plan_actions:
        for action in cleanup_plan_actions[:CLEANUP_ACTION_PREVIEW_LIMIT]:
            if not isinstance(action, dict):
                continue
            cleanup_actions.append(
                {
                    "action": str(action.get("action") or ""),
                    "source": _public_path_label(action.get("source") or ""),
                    "destination": _public_path_label(action.get("destination") or "")
                    or None,
                    "reason": str(action.get("reason") or ""),
                    "risk": str(action.get("risk") or "") or None,
                    "requires_confirmation": action.get(
                        "requires_user_confirmation", False
                    ),
                }
            )

    inventory_counts = _data_hygiene_inventory_counts(inventory)
    manifest_evidence = _manifest_official_entry_evidence(
        manifest,
        inventory=inventory,
        current_manifest_sha256=manifest_sha256,
    )
    manifest_is_v2 = bool(manifest and manifest.get("schema_version") == 2)
    validation_status = str(manifest_evidence["validation_status"])
    inventory_catalog = inventory.get("catalog") if isinstance(inventory, dict) else None
    inventory_requires_manifest = bool(
        isinstance(inventory_catalog, dict)
        and inventory_catalog.get("validation_contract_version") == 2
    )
    validation_required = (
        inventory_requires_manifest
        or manifest_is_v2
        or (manifest_path.exists() and manifest is None)
    )
    if validation_required and validation_status == "missing":
        validation_status = "stale" if inventory_requires_manifest else "unverified"
    inventory_status = "missing"
    if inventory:
        inventory_status = (
            "available"
            if not validation_required or validation_status == "verified"
            else validation_status
        )
    elif hygiene_publication_status in {"invalid", "stale"}:
        inventory_status = hygiene_publication_status
    manifest_status = "invalid" if manifest_path.exists() and manifest is None else "missing"
    if manifest is not None:
        manifest_status = (
            "available"
            if not manifest_is_v2 or validation_status == "verified"
            else validation_status
        )
    cleanup_plan_status = "missing"
    if cleanup_plan:
        cleanup_plan_status = (
            "available"
            if not validation_required or validation_status == "verified"
            else validation_status
        )
    elif hygiene_publication_status in {"invalid", "stale"}:
        cleanup_plan_status = hygiene_publication_status
    visible_cleanup_actions = (
        cleanup_actions if cleanup_plan_status == "available" else []
    )

    return {
        "inventory": {
            "status": inventory_status,
            "validation_status": validation_status,
            "path": _public_path_label(inventory_path)
            if inventory_path.exists()
            else None,
            "generated_at": inventory.get("generated_at") if inventory else None,
            "official_entries": (
                manifest_evidence["official_entries"]
                if validation_required
                else inventory_counts["official_entries"]
            ),
            "candidate_entries": inventory_counts["candidate_entries"],
            "legacy_entries": inventory_counts["legacy_entries"],
            "archive_candidates": inventory_counts["archive_candidates"],
        },
        "cleanup_plan": {
            "status": cleanup_plan_status,
            "path": _public_path_label(cleanup_plan_path)
            if cleanup_plan_path.exists()
            else None,
            "generated_at": cleanup_plan.get("generated_at") if cleanup_plan else None,
            "dry_run_only": cleanup_plan.get("dry_run_only", True)
            if cleanup_plan
            else True,
            "total_actions": (
                len(cleanup_plan_actions)
                if cleanup_plan_status == "available"
                else 0
            ),
            "preview_limit": CLEANUP_ACTION_PREVIEW_LIMIT,
            "truncated": (
                len(cleanup_plan_actions) > len(visible_cleanup_actions)
                if cleanup_plan_status == "available"
                else False
            ),
            "actions": visible_cleanup_actions,
        },
        "quality_report": {
            "status": "deprecated",
            "path": None,
            "summary": None,
        },
        "cleaning_strategy": {
            "status": "available" if cleaning_strategy_path.exists() else "missing",
            "path": _public_path_label(cleaning_strategy_path)
            if cleaning_strategy_path.exists()
            else None,
        },
        "manifest": {
            "status": manifest_status,
            "path": _public_path_label(manifest_path)
            if manifest_path.exists()
            else None,
            "generated_at": manifest.get("generated_at") if manifest else None,
            "source_report": _public_path_label(manifest.get("source_report"))
            if manifest
            else None,
            "official_entries": manifest_evidence["official_entries"],
            "validation_status": validation_status,
        },
        "readme": {
            "status": "available" if readme_path.exists() else "missing",
            "path": _public_path_label(readme_path) if readme_path.exists() else None,
        },
        "maintenance_lock": {
            "locked": maintenance_locked,
            "path": _public_path_label(maintenance_lock_path),
            "updated_at": _file_mtime_iso(maintenance_lock_path)
            if maintenance_locked
            else None,
        },
    }


def _collection_count(value: Any) -> int:
    if isinstance(value, (dict, list, tuple)):
        return len(value)
    return 0


def _manifest_official_entry_evidence(
    manifest: Optional[Dict[str, Any]],
    *,
    inventory: Optional[Dict[str, Any]],
    current_manifest_sha256: Optional[str],
) -> Dict[str, Any]:
    if not manifest:
        return {"official_entries": 0, "validation_status": "missing"}
    datasets = manifest.get("datasets")
    if manifest.get("schema_version") == 2 and isinstance(datasets, dict):
        catalog = inventory.get("catalog") if isinstance(inventory, dict) else None
        if not isinstance(catalog, dict):
            return {"official_entries": 0, "validation_status": "unverified"}
        audited_sha256 = catalog.get("manifest_sha256")
        if (
            not isinstance(current_manifest_sha256, str)
            or not isinstance(audited_sha256, str)
            or audited_sha256 != current_manifest_sha256
        ):
            return {"official_entries": 0, "validation_status": "stale"}
        catalog_errors = catalog.get("errors")
        if (
            catalog.get("status") != "available"
            or not isinstance(catalog_errors, list)
            or catalog_errors
        ):
            return {"official_entries": 0, "validation_status": "unverified"}
        records = catalog.get("datasets")
        recorded_ids = catalog.get("validated_dataset_ids")
        if not isinstance(records, list) or not isinstance(recorded_ids, list):
            return {"official_entries": 0, "validation_status": "unverified"}
        dataset_count = catalog.get("dataset_count")
        if (
            isinstance(dataset_count, bool)
            or not isinstance(dataset_count, int)
            or dataset_count != len(records)
            or dataset_count != len(datasets)
        ):
            return {"official_entries": 0, "validation_status": "unverified"}
        verified_ids = {
            str(item.get("dataset_id"))
            for item in records
            if isinstance(item, dict)
            and item.get("canonical_valid") is True
            and isinstance(item.get("dataset_id"), str)
            and item.get("dataset_id")
        }
        declared_ids = {
            value for value in recorded_ids if isinstance(value, str) and value
        }
        if verified_ids != declared_ids or len(declared_ids) != len(recorded_ids):
            return {"official_entries": 0, "validation_status": "unverified"}
        if catalog.get("validation_contract_version") != 2:
            return {"official_entries": 0, "validation_status": "unverified"}
        for record in records:
            if not isinstance(record, dict) or record.get("canonical_valid") is not True:
                continue
            pointer_status = _inventory_latest_pointer_status(record)
            if pointer_status != "verified":
                return {"official_entries": 0, "validation_status": pointer_status}
            state_status = _inventory_snapshot_state_status(
                record,
                catalog_entry=datasets.get(record.get("dataset_id")),
            )
            if state_status != "verified":
                return {"official_entries": 0, "validation_status": state_status}
        current_ids = {
            key
            for key, entry in datasets.items()
            if isinstance(key, str)
            and isinstance(entry, dict)
            and entry.get("dataset_id") == key
            and entry.get("role") == "canonical"
            and entry.get("status") == "ready"
        }
        if current_ids != verified_ids:
            return {"official_entries": 0, "validation_status": "unverified"}
        return {
            "official_entries": len(current_ids),
            "validation_status": "verified",
        }
    return {
        "official_entries": _collection_count(manifest.get("official_entries")),
        "validation_status": "legacy_unverified",
    }


def _inventory_snapshot_state_status(
    record: Dict[str, Any],
    *,
    catalog_entry: Optional[Dict[str, Any]] = None,
) -> str:
    raw_path = record.get("snapshot_path")
    expected = record.get("state_fingerprint")
    if not isinstance(raw_path, str) or not raw_path or not isinstance(expected, dict):
        return "unverified"
    normalized_expected = {
        "algorithm": expected.get("algorithm"),
        "file_count": expected.get("file_count"),
        "total_bytes": expected.get("total_bytes"),
        "scanned_entries": expected.get("scanned_entries"),
        "sha256": expected.get("sha256"),
    }
    if (
        normalized_expected["algorithm"]
        != "relative-path-size-mtime-ctime-device-inode-v2"
        or isinstance(normalized_expected["file_count"], bool)
        or not isinstance(normalized_expected["file_count"], int)
        or normalized_expected["file_count"] < 0
        or isinstance(normalized_expected["total_bytes"], bool)
        or not isinstance(normalized_expected["total_bytes"], int)
        or normalized_expected["total_bytes"] < 0
        or isinstance(normalized_expected["scanned_entries"], bool)
        or not isinstance(normalized_expected["scanned_entries"], int)
        or normalized_expected["scanned_entries"] < normalized_expected["file_count"]
        or not isinstance(normalized_expected["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", normalized_expected["sha256"])
    ):
        return "unverified"

    root = _resolve_for_boundary(QUANT_DATA_ROOT)
    relative = _trusted_root_relative_path(QUANT_DATA_ROOT, raw_path)
    if relative is None:
        return "unverified"
    if _manifest_path_contains_symlink(root, relative):
        return "stale"
    snapshot = root / relative
    try:
        current = _filesystem_state_fingerprint_live(snapshot)
    except (DataContractError, OSError):
        return "stale"
    if current != normalized_expected:
        return "stale"
    if sys.platform == "win32":
        # Windows path ctime is creation time, so restoring mtime can hide a
        # same-size rewrite from metadata. Reuse the existing content proofs.
        try:
            dataset_id = str(record.get("dataset_id") or "")
            if record.get("storage_contract") == "immutable_snapshot_v2":
                resolved = resolve_dataset(dataset_id, quant_data_root=root)
                if (
                    resolved.snapshot_path.resolve() != snapshot.resolve()
                    or resolved.manifest.get("snapshot_id") != record.get("snapshot_id")
                ):
                    return "stale"
            elif (
                record.get("storage_contract") == "catalog_registered_fixed_path"
                and isinstance(catalog_entry, dict)
            ):
                from quant_workflow.scripts.report_data_hygiene import (
                    _validate_fixed_path_dataset,
                )

                snapshot_hash = _validate_fixed_path_dataset(
                    dataset_id, snapshot, catalog_entry
                )
                if snapshot_hash != record.get("snapshot_id"):
                    return "stale"
            else:
                return "unverified"
        except (DataContractError, OSError, TypeError, ValueError):
            return "stale"
    return "verified"


def _inventory_latest_pointer_status(record: Dict[str, Any]) -> str:
    if record.get("storage_contract") != "immutable_snapshot_v2":
        return "verified"
    raw_path = record.get("latest_pointer_path")
    expected_sha256 = record.get("latest_pointer_sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        return "unverified"
    root = _resolve_for_boundary(QUANT_DATA_ROOT)
    relative = _trusted_root_relative_path(QUANT_DATA_ROOT, raw_path)
    if relative is None:
        return "unverified"
    if _manifest_path_contains_symlink(root, relative):
        return "stale"
    pointer_bytes = _read_bounded_regular_file_at(
        root,
        relative,
        max_bytes=DATA_HYGIENE_POINTER_MAX_BYTES,
    )
    if pointer_bytes is None:
        return "stale"
    return (
        "verified"
        if hashlib.sha256(pointer_bytes).hexdigest() == expected_sha256
        else "stale"
    )


def _data_hygiene_inventory_counts(
    inventory: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    if not inventory:
        return {
            "official_entries": 0,
            "candidate_entries": 0,
            "legacy_entries": 0,
            "archive_candidates": 0,
        }

    direct_counts = {
        "official_entries": inventory.get("official_entries"),
        "candidate_entries": inventory.get("candidate_entries"),
        "legacy_entries": inventory.get("legacy_entries"),
        "archive_candidates": inventory.get("archive_candidates"),
    }
    if all(isinstance(value, int) for value in direct_counts.values()):
        return {key: int(value) for key, value in direct_counts.items()}

    items = inventory.get("items")
    if not isinstance(items, list):
        return {
            "official_entries": _safe_int(direct_counts["official_entries"]),
            "candidate_entries": _safe_int(direct_counts["candidate_entries"]),
            "legacy_entries": _safe_int(direct_counts["legacy_entries"]),
            "archive_candidates": _safe_int(direct_counts["archive_candidates"]),
        }

    return {
        "official_entries": _count_inventory_category(items, "official"),
        "candidate_entries": _count_inventory_category(items, "candidate"),
        "legacy_entries": _count_inventory_category(items, "legacy"),
        "archive_candidates": _count_inventory_category(items, "archive_candidate"),
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _count_inventory_category(items: list[Any], category: str) -> int:
    total = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_category = item.get("category")
        categories = item.get("categories")
        if item_category == category:
            total += 1
            continue
        if isinstance(categories, list) and category in categories:
            total += 1
    return total


def _external_tools_payload() -> Dict[str, Any]:
    return {
        "tonghuashun": _tonghuashun_status_payload(),
        "iphone_mirroring": {
            "name": "iPhone 镜像",
            "status": "deferred",
            "installed": False,
            "mode": "deferred",
            "path": None,
            "summary": "手机连接暂时不做；未来也默认只做只读观察，不读取密码、不自动交易。",
            "capabilities": [],
            "restrictions": [
                "本阶段不接入",
                "不读取账号密码",
                "不自动下单",
            ],
        },
    }


def _string_items(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _capability_matrix_payload(external_tools: Dict[str, Any]) -> list[Dict[str, Any]]:
    tonghuashun = (
        external_tools.get("tonghuashun") if isinstance(external_tools, dict) else None
    )
    tonghuashun = tonghuashun if isinstance(tonghuashun, dict) else {}
    tonghuashun_capabilities = _string_items(tonghuashun.get("capabilities"))
    tonghuashun_restrictions = _string_items(tonghuashun.get("restrictions"))
    tonghuashun_summary = str(
        tonghuashun.get("summary") or "同花顺仅作为后续只读观察方向。"
    )
    tonghuashun_status = (
        "read_only"
        if tonghuashun.get("status") == "available" or tonghuashun.get("installed")
        else "deferred"
    )
    tonghuashun_browser = tonghuashun_summary
    if tonghuashun_capabilities:
        tonghuashun_browser = (
            f"{tonghuashun_browser} 能力：{'、'.join(tonghuashun_capabilities)}"
        )

    return [
        {
            "id": "dsa_web_core",
            "module": "daily_stock_analysis WebUI",
            "status": "available",
            "entry": "首页 / 研究 / 工作流 / 回测 / 问股 / 持仓 / 告警 / 选股",
            "route": "/workflow",
            "tests": "后端 pytest + 前端 vitest/lint/build",
            "browser": "local-workbench smoke 覆盖主页面和关键交互",
            "review": "opencode复审通过",
            "next": "继续把重任务体验做顺，并补齐更多结果预览。",
            "restrictions": [],
        },
        {
            "id": "quant_workflow_pipeline",
            "module": "quant_workflow 流水线",
            "status": "available",
            "entry": "工作流任务中心",
            "route": "/workflow",
            "tests": "quant_workflow unittest + DSA quant API 测试",
            "browser": "可从网页查看实时数据状态，并运行异常报告、Qlib 转换和训练入口",
            "review": "opencode复审通过",
            "next": "补更多产物预览、失败排查和每日研究向导。",
            "restrictions": ["只允许白名单脚本", "网页不能提交任意 shell 命令"],
        },
        {
            "id": "qlib_integration",
            "module": "Qlib 机器学习量化",
            "status": "available",
            "entry": "回测 / 工作流 / MLflow",
            "route": "/backtest",
            "tests": "train_baseline_lgb + research_status API + BacktestPage 测试",
            "browser": "网页可查看 Qlib 训练摘要、关键指标、MLflow 链接和回测入口",
            "review": "opencode复审通过",
            "next": "继续补模型质量、清仓信号和组合回测对比。",
            "restrictions": [
                "不直接改造 Qlib 官方源码为主 UI",
                "MLflow 只作为实验原始记录页",
            ],
        },
        {
            "id": "china_astock_quant_reuse",
            "module": "china-astock-quant",
            "status": "available",
            "entry": "工作流技术指标报告 / 研究页项目关系",
            "route": "/workflow",
            "tests": "report_china_astock_indicators + run_china_astock_comparison + DSA quant API 测试",
            "browser": "可从网页生成只读技术指标报告和策略对比报告",
            "review": "待本阶段 opencode 复审",
            "next": "继续评估是否把事件驱动回测包装为只读对比报告。",
            "restrictions": ["不改造成主网页", "不接入交易执行"],
        },
        {
            "id": "external_tools_tonghuashun",
            "module": "同花顺桌面端",
            "status": tonghuashun_status,
            "entry": "研究页 / 本机只读检测",
            "route": "/research",
            "tests": "research_status API + ResearchPage 测试",
            "browser": tonghuashun_browser,
            "review": "opencode复审通过",
            "next": "只读观察，暂不接入交易。",
            "restrictions": tonghuashun_restrictions,
        },
        {
            "id": "trading_execution",
            "module": "交易执行",
            "status": "not_planned",
            "entry": "无网页入口",
            "tests": "不提供交易 API",
            "browser": "页面只展示研究、回测和观察能力",
            "review": "安全边界需持续复审",
            "next": "保持人工交易，不做自动下单。",
            "restrictions": ["不自动下单", "不读取券商密码", "不连接券商交易接口"],
        },
    ]


def _function_inventory_payload() -> list[Dict[str, Any]]:
    return [
        {
            "id": "dsa_home_analysis",
            "area": "DSA 原有分析",
            "name": "首页股票分析",
            "status": "available",
            "entry": "首页分析入口",
            "route": "/",
            "owner": "daily_stock_analysis",
            "coverage": "已有 Web 页面、分析 API、历史记录和通知链路。",
            "gaps": "基本面、资金面、宏观和情绪数据仍需继续补强。",
            "next": "把数据质量和实验结论继续回填到分析上下文。",
            "verification": "HomePage/analysis API 测试 + 浏览器 smoke。",
        },
        {
            "id": "dsa_chat_research",
            "area": "DSA 原有分析",
            "name": "问股/研究对话",
            "status": "available",
            "entry": "问股页面",
            "route": "/chat",
            "owner": "daily_stock_analysis",
            "coverage": "已有聊天式研究入口，可调用 DSA 分析能力。",
            "gaps": "还未把 Qlib 实验结论自动整理为问股上下文。",
            "next": "把最新训练摘要、数据质量摘要作为只读上下文注入。",
            "verification": "ChatPage 测试 + 浏览器 smoke。",
        },
        {
            "id": "dsa_backtest_accuracy",
            "area": "回测/历史验证",
            "name": "DSA 历史分析准确性回测",
            "status": "available",
            "entry": "回测页面",
            "route": "/backtest",
            "owner": "daily_stock_analysis",
            "coverage": "已有 DSA 历史建议与后续走势一致性展示。",
            "gaps": "需要与 Qlib 选股模型和交易策略组合的回测结果继续统一解释口径。",
            "next": "在回测页继续拆分 DSA 准确性和 Qlib 交易策略回测结果。",
            "verification": "BacktestPage/API 测试 + 浏览器 smoke。",
        },
        {
            "id": "quant_workflow_jobs",
            "area": "量化流水线",
            "name": "网页任务中心",
            "status": "available",
            "entry": "量化工作流",
            "route": "/workflow",
            "owner": "daily_stock_analysis + quant_workflow",
            "coverage": "可从网页创建白名单任务、看状态、日志、产物和失败原因。",
            "gaps": "长任务取消、失败诊断和更多产物预览仍可继续增强。",
            "next": "补齐每日研究向导和更多报告预览。",
            "verification": "quant_jobs API 测试 + WorkflowPage 测试 + 浏览器 smoke。",
        },
        {
            "id": "qlib_data_pipeline",
            "area": "Qlib 接入",
            "name": "数据检查/转换/训练/回测",
            "status": "available",
            "entry": "工作流 + 回测 + MLflow",
            "route": "/workflow",
            "owner": "quant_workflow + Qlib",
            "coverage": "网页可触发数据检查、Qlib 转换、LightGBM baseline 训练，并查看训练摘要和 MLflow 链接。",
            "gaps": "模型质量、市场择时、清仓信号和策略对比仍需继续实验。",
            "next": "增加 Qlib 模型效果解释、现金仓位叠加和失败排查视图。",
            "verification": "train_baseline_lgb 最小测试 + quant/status + 浏览器验证。",
        },
        {
            "id": "quant_workflow_script_coverage",
            "area": "量化流水线",
            "name": "quant_workflow 脚本覆盖盘点",
            "status": "available",
            "entry": "研究页功能入口盘点",
            "route": "/research",
            "owner": "quant_workflow + daily_stock_analysis",
            "coverage": (
                "网页白名单覆盖 Qlib 合并、清洗、实时数据检查、异常报告、"
                "数据治理盘点和 dry-run 清理预案、市场状态、现金仓位回测、china-astock 指标报告、"
                "china-astock 对比、交易策略优化预演和选股模型基线训练；"
                "report_data_hygiene.py 可从网页生成 inventory 与 dry-run cleanup plan；"
                "add_index_benchmark.py 与 generate_csi300.py 已作为 provider 维护预演接入网页任务中心，强制 dry-run。"
            ),
            "gaps": (
                "supervise_baostock.py 与网页任务中心的持久任务能力重叠，暂不单独暴露；"
                "provider 维护任务当前只开放 dry-run 预演，不允许网页写入 provider。"
            ),
            "next": "先保持 provider 维护任务为 dry-run；若未来需要真实写入，必须补二次确认、目标 provider 参数和产物 diff 预览。",
            "verification": "research_status API 测试 + ResearchPage 功能入口盘点展示。",
        },
        {
            "id": "experiment_notebook",
            "area": "实验记录",
            "name": "实验记录本与删除",
            "status": "available",
            "entry": "研究页实验记录本",
            "route": "/research",
            "owner": "quant_workflow reports + daily_stock_analysis",
            "coverage": "网页可查看实验摘要、归档记录，并对带标记段落做永久删除。",
            "gaps": "还需要更友好的记录编辑和批量筛选。",
            "next": "补新增记录表单和按数据版本/模型筛选。",
            "verification": "research_status API 删除测试 + ResearchPage 测试。",
        },
        {
            "id": "china_astock_reference",
            "area": "参考项目复用",
            "name": "china-astock-quant 可复用能力",
            "status": "available",
            "entry": "研究页清单 + 工作流指标报告 + 工作流对比报告",
            "route": "/workflow",
            "owner": "china-astock-quant + quant_workflow",
            "coverage": "已复用指标库生成只读技术指标报告，并有只读策略对比报告入口。",
            "gaps": "事件驱动回测和策略搜索尚未作为网页任务直接运行。",
            "next": "继续评估是否包装事件驱动回测；交易相关模块保持禁止接入。",
            "verification": "report_china_astock_indicators 测试 + run_china_astock_comparison 测试 + WorkflowPage 测试。",
        },
        {
            "id": "external_observation",
            "area": "外部工具",
            "name": "同花顺只读观察",
            "status": "read_only",
            "entry": "研究页检测",
            "route": "/research",
            "owner": "local desktop",
            "coverage": "仅检测本机应用存在和列出后续只读观察方向。",
            "gaps": "不接入账号、密码、交易接口和手机镜像。",
            "next": "未来只做 read-only 观察或截图/OCR 参考。",
            "verification": "research_status API + ResearchPage 测试 + 安全复审。",
        },
    ]


def _china_astock_reuse_payload() -> list[Dict[str, Any]]:
    return [
        {
            "id": "indicators",
            "capability": "技术指标库",
            "source": "strategies/_indicators.py",
            "status": "available",
            "priority": "P1",
            "decision": "优先复用",
            "target": "工作流指标报告 report_china_astock_indicators",
            "next": "继续把指标报告结果接入研究页摘要和后续技术分析解释。",
            "restrictions": [],
        },
        {
            "id": "sample_strategies",
            "capability": "示例策略",
            "source": "strategies/dual_ma.py, mean_reversion.py, momentum_rotation.py",
            "status": "partial",
            "priority": "P1",
            "decision": "选择性接入",
            "target": "quant_workflow 白名单策略实验",
            "next": "先接入只读策略对比和小样本回测，再评估是否进入正式工作流。",
            "restrictions": ["不接入交易执行"],
        },
        {
            "id": "event_backtest",
            "capability": "A 股事件驱动回测",
            "source": "backtest/engine.py, portfolio.py, metrics.py",
            "status": "partial",
            "priority": "P2",
            "decision": "作为 Qlib 外的补充回测引擎",
            "target": "quant_workflow 对比报告",
            "next": "用于验证 T+1、100 股、费用和滑点等 A 股交易约束。",
            "restrictions": ["只做历史回测", "不连接券商"],
        },
        {
            "id": "grid_walk_forward",
            "capability": "参数搜索和滚动验证",
            "source": "backtest/grid_search.py, walk_forward.py",
            "status": "partial",
            "priority": "P2",
            "decision": "后续接入",
            "target": "策略稳健性报告",
            "next": "把赢家评估和 walk-forward 输出整理成 DSA 可读报告。",
            "restrictions": ["避免过拟合", "必须记录样本外结果"],
        },
        {
            "id": "akshare_cache",
            "capability": "AKShare Parquet 缓存经验",
            "source": "data_fetcher/akshare_fetcher.py",
            "status": "deferred",
            "priority": "P3",
            "decision": "仅参考缓存模式",
            "target": "quant_data 数据层",
            "next": "当前主数据链路优先 Baostock/JQData，本模块只参考缓存设计。",
            "restrictions": ["不替换现有数据主链路"],
        },
        {
            "id": "paper_trading",
            "capability": "纸面交易/券商适配",
            "source": "trading/*, scripts/paper_live.py, scripts/paper_replay.py",
            "status": "not_planned",
            "priority": "禁止项",
            "decision": "本阶段不接入",
            "target": "无网页入口",
            "next": "保持为参考代码，不进入 DSA 主工作台。",
            "restrictions": ["不自动下单", "不读取券商密码", "不连接券商交易接口"],
        },
    ]


def _acceptance_checklist_payload() -> list[Dict[str, Any]]:
    return [
        {
            "id": "project_hygiene",
            "title": "项目整洁",
            "status": "partial",
            "route": "/research",
            "evidence": "DSA 无散落未跟踪源码；当前仍有未提交的正式改动，提交或明确保留后才算 Git 工作区完全干净。",
            "next": "完成最终提交或明确保留当前正式改动。",
        },
        {
            "id": "core_pages_browser",
            "title": "核心页面浏览器验收",
            "status": "verified",
            "route": "/research",
            "evidence": "/research、/workflow、/backtest 和原有页面由 local-workbench smoke 覆盖。",
            "next": "后续新增页面继续补浏览器 smoke。",
        },
        {
            "id": "automated_tests",
            "title": "自动测试和构建",
            "status": "verified",
            "route": "/research",
            "evidence": "后端 pytest、前端 vitest、lint、build 和脚本最小测试是阶段验收命令。",
            "next": "每次新增行为继续先写测试再实现。",
        },
        {
            "id": "qlib_web_flow",
            "title": "Qlib 网页流程",
            "status": "available",
            "route": "/workflow",
            "evidence": "网页可触发数据检查、Qlib 转换、训练任务、日志、产物、回测摘要和 MLflow 链接。",
            "next": "继续补模型效果解释、清仓信号和组合对比。",
        },
        {
            "id": "quant_workflow_wrapped",
            "title": "quant_workflow 稳定包装",
            "status": "available",
            "route": "/workflow",
            "evidence": "核心脚本通过白名单任务中心运行；provider 维护脚本已作为强制 dry-run 预演接入，未暴露脚本已在脚本覆盖盘点中说明原因。",
            "next": "若未来开放 provider 真实写入，必须先补二次确认、目标 provider 参数和产物 diff 预览。",
        },
        {
            "id": "china_astock_reuse",
            "title": "china-astock-quant 复用",
            "status": "partial",
            "route": "/research",
            "evidence": "研究页已列出指标、策略、事件回测、参数搜索和交易禁区，并有只读策略对比报告入口。",
            "next": "优先把指标库和小样本策略回测包装成可复验报告。",
        },
        {
            "id": "external_tools_read_only",
            "title": "同花顺只读观察",
            "status": "read_only",
            "route": "/research",
            "evidence": "页面只做本机应用存在检测和后续只读方向说明；手机连接暂不做。",
            "next": "未来只做截图/OCR 或界面参考，不读取密码、不自动下单。",
        },
        {
            "id": "opencode_review",
            "title": "opencode 只读复审",
            "status": "verified",
            "route": "/research",
            "evidence": "阶段改动已经过 opencode 子 agent 只读 review；问题需修复或记录后续项。",
            "next": "每个阶段继续复审 DSA diff 和 quant_workflow 边界。",
        },
        {
            "id": "self_service_docs",
            "title": "自助文档和页面说明",
            "status": "available",
            "route": "/research",
            "evidence": "local-quant-workbench 手册和研究页说明可支撑用户不依赖 Codex 自己操作。",
            "next": "继续把常见失败原因和产物解释补进页面。",
        },
    ]


def _provider_status_payload(
    resolution: _NativeQlibProviderResolution,
) -> Dict[str, Any]:
    if resolution.path is None:
        return {
            "provider": resolution.provider_name,
            "role": "canonical",
            "status": "blocked",
            "price_basis": "qfq",
            "provider_source": "Qlib Provider",
            "provider_error": resolution.error,
        }
    payload = {
        **_qlib_provider_from_path(resolution.path),
        "provider": resolution.provider_name,
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "provider_source": "Qlib Provider",
    }
    payload.update(_resolution_common_identity(resolution))
    return payload


def _missing_status_payload(
    provider_dataset_id: str = ASHARE_DAILY_QFQ_DATASET_ID,
    provider: str = "canonical",
) -> Dict[str, Any]:
    reports_dir = RESEARCH_STATUS_PATH.parent
    external_tools = _external_tools_payload()
    native_resolution = _resolve_native_qlib_provider(
        provider_dataset_id,
        provider=provider,
    )
    latest_training = _latest_training_payload(reports_dir)
    if latest_training is not None:
        _annotate_latest_training_provider(latest_training, native_resolution)
    qlib_provider = None
    if native_resolution.authoritative:
        qlib_provider = _provider_status_payload(native_resolution)
    training_readiness = {
        "status": "missing",
        "summary": "正在实时检查 quant_data 与当前 Qlib Provider。",
    }
    if native_resolution.authoritative and native_resolution.path is None:
        training_readiness = {
            "status": "blocked",
            "summary": native_resolution.error,
        }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trainable": False,
        "training_readiness": training_readiness,
        "quality_report_id": None,
        "quality_evidence_sha256": None,
        "quality_scan": {},
        "provider_quality": {},
        "data_sources": [],
        "qlib_provider": qlib_provider,
        "latest_training": latest_training,
        "cash_control_backtest": _cash_control_backtest_payload(reports_dir),
        "qlib_predictions": _qlib_predictions_payload(reports_dir),
        "candidate_pipeline": _candidate_pipeline_payload(
            reports_dir, native_resolution
        ),
        "robust_strategy": _robust_strategy_payload(reports_dir),
        "experiment_notebook": _experiment_notebook_payload(reports_dir),
        "mlflow": _mlflow_payload(),
        "data_hygiene": _data_hygiene_payload(reports_dir),
        "external_tools": external_tools,
        "capability_matrix": _capability_matrix_payload(external_tools),
        "function_inventory": _function_inventory_payload(),
        "china_astock_reuse": _china_astock_reuse_payload(),
        "acceptance_checklist": _acceptance_checklist_payload(),
    }
    _attach_training_preflights(
        payload,
        requested_dataset_id=provider_dataset_id,
        requested_resolution=native_resolution,
        requested_provider=provider,
    )
    _deprecate_legacy_quality_status(payload)
    if latest_training is not None:
        _sync_qlib_provider_with_latest_training(
            payload,
            latest_training,
            native_resolution=native_resolution,
        )
    return payload


def _remove_active_experiment_run(reports_dir: Path, run_id: str) -> Dict[str, Any]:
    runs_path = reports_dir / EXPERIMENT_RUNS_FILENAME

    if not runs_path.exists():
        raise HTTPException(
            status_code=404, detail="Experiment runs file is not available."
        )

    remaining = []
    removed = None
    invalid_lines = []
    try:
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_lines.append(stripped)
                continue
            if not isinstance(item, dict):
                invalid_lines.append(stripped)
                continue
            if item.get("run_id") == run_id:
                removed = item
            else:
                remaining.append(item)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Experiment runs file could not be read: {exc}"
        ) from exc

    if removed is None:
        raise HTTPException(
            status_code=404, detail=f"Experiment run {run_id} was not found."
        )

    tmp_path = runs_path.with_suffix(runs_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for item in remaining:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
            for line in invalid_lines:
                handle.write(line + "\n")
        tmp_path.replace(runs_path)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Experiment run could not be removed: {exc}"
        ) from exc

    return removed


def _archive_experiment_run(reports_dir: Path, run_id: str) -> Dict[str, Any]:
    deleted_runs_path = reports_dir / EXPERIMENT_DELETED_RUNS_FILENAME
    runs_path = reports_dir / EXPERIMENT_RUNS_FILENAME
    runs_snapshot = _snapshot_text_file(runs_path)
    deleted_runs_snapshot = _snapshot_text_file(deleted_runs_path)
    try:
        removed = _remove_active_experiment_run(reports_dir, run_id)
        archived = {
            **removed,
            "deleted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "deleted_from": EXPERIMENT_RUNS_FILENAME,
        }
        with deleted_runs_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(archived, ensure_ascii=False, sort_keys=True) + "\n"
            )
    except HTTPException:
        _restore_text_file(runs_path, *runs_snapshot)
        _restore_text_file(deleted_runs_path, *deleted_runs_snapshot)
        raise
    except Exception as exc:
        _restore_text_file(runs_path, *runs_snapshot)
        _restore_text_file(deleted_runs_path, *deleted_runs_snapshot)
        raise HTTPException(
            status_code=500, detail=f"Experiment run could not be archived: {exc}"
        ) from exc
    return archived


def _find_marked_experiment_note_block(
    reports_dir: Path, run_id: str
) -> Optional[Tuple[Path, str, int, int]]:
    log_path = reports_dir / EXPERIMENT_LOG_FILENAME
    if not log_path.exists():
        return None

    start_marker = f"<!-- experiment-run:{run_id} -->"
    end_marker = f"<!-- /experiment-run:{run_id} -->"

    try:
        content = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Experiment log could not be read: {exc}"
        ) from exc

    start = content.find(start_marker)
    if start == -1:
        return None

    end = content.find(end_marker, start + len(start_marker))
    if end == -1:
        raise HTTPException(
            status_code=409, detail=f"Experiment note block for {run_id} is incomplete."
        )
    end += len(end_marker)
    return log_path, content, start, end


def _ensure_marked_experiment_note_removable(reports_dir: Path, run_id: str) -> None:
    _find_marked_experiment_note_block(reports_dir, run_id)


def _remove_marked_experiment_note(reports_dir: Path, run_id: str) -> bool:
    note_block = _find_marked_experiment_note_block(reports_dir, run_id)
    if note_block is None:
        return False
    log_path, content, start, end = note_block

    updated = content[:start].rstrip() + "\n\n" + content[end:].lstrip()
    updated = updated.rstrip() + "\n" if updated.strip() else ""
    try:
        _write_text_file_atomic(log_path, updated)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Experiment log could not be updated: {exc}"
        ) from exc
    return True


def _snapshot_text_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, ""
    try:
        return True, path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{path.name} could not be read before mutation: {exc}",
        ) from exc


def _write_text_file_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _restore_text_file(path: Path, existed: bool, content: str) -> None:
    try:
        if existed:
            _write_text_file_atomic(path, content)
        elif path.exists():
            path.unlink()
    except OSError as exc:
        # Preserve the original failure for the caller; this is best-effort rollback.
        logger.warning("Experiment rollback restore failed for %s: %s", path, exc)


def build_research_status_payload(
    provider_dataset_id: str = ASHARE_DAILY_QFQ_DATASET_ID,
    provider: str = "qmt_native",
    training_config: Path | None = None,
) -> Dict[str, Any]:
    """Build the latest quant workflow status used by web pages."""
    if provider == "qmt_native":
        return _qmt_native_research_status(training_config=training_config)
    if not RESEARCH_STATUS_PATH.exists():
        return _missing_status_payload(provider_dataset_id, provider)

    try:
        payload = json.loads(RESEARCH_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fallback = _missing_status_payload(provider_dataset_id, provider)
        fallback["training_readiness"] = {
            "status": "invalid",
            "summary": f"Research status artifact could not be read: {exc}",
        }
        return fallback

    if not isinstance(payload, dict):
        fallback = _missing_status_payload(provider_dataset_id, provider)
        fallback["training_readiness"] = {
            "status": "invalid",
            "summary": "Research status artifact is not a JSON object.",
        }
        return fallback

    native_resolution = _resolve_native_qlib_provider(
        provider_dataset_id,
        provider=provider,
    )
    reports_dir = RESEARCH_STATUS_PATH.parent
    if native_resolution.path is not None:
        payload["qlib_provider"] = _provider_status_payload(native_resolution)
    elif native_resolution.authoritative:
        payload["qlib_provider"] = _provider_status_payload(native_resolution)
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": native_resolution.error,
        }
    latest_training = _latest_training_payload(reports_dir)
    if latest_training is not None:
        _annotate_latest_training_provider(latest_training, native_resolution)
        payload["latest_training"] = latest_training
        _sync_qlib_provider_with_latest_training(
            payload,
            latest_training,
            native_resolution=native_resolution,
        )
    cash_control_backtest = _cash_control_backtest_payload(reports_dir)
    if cash_control_backtest is not None:
        payload["cash_control_backtest"] = cash_control_backtest
    payload["qlib_predictions"] = _qlib_predictions_payload(reports_dir)
    payload["candidate_pipeline"] = _candidate_pipeline_payload(
        reports_dir, native_resolution
    )
    payload["robust_strategy"] = _robust_strategy_payload(reports_dir)
    payload["robust_signal"] = _robust_signal_payload(reports_dir)
    payload["qlib_research"] = _qlib_research_payload(reports_dir)
    payload["experiment_notebook"] = _experiment_notebook_payload(reports_dir)
    payload["mlflow"] = _mlflow_payload()
    payload["data_hygiene"] = _data_hygiene_payload(reports_dir)
    external_tools = _external_tools_payload()
    payload["external_tools"] = external_tools
    payload["capability_matrix"] = _capability_matrix_payload(external_tools)
    payload["function_inventory"] = _function_inventory_payload()
    payload["china_astock_reuse"] = _china_astock_reuse_payload()
    payload["acceptance_checklist"] = _acceptance_checklist_payload()
    _deprecate_legacy_quality_status(payload)
    _attach_training_preflights(
        payload,
        requested_dataset_id=provider_dataset_id,
        requested_resolution=native_resolution,
        requested_provider=provider,
    )
    return payload


def _qmt_native_research_status(*, training_config: Path | None = None) -> Dict[str, Any]:
    """The new input path has no catalog, legacy readiness or live SDK gate."""
    from api.v1.endpoints.data_status import inspect_qmt_input

    config_path = training_config or QUANT_WORKFLOW_ROOT / "configs" / "baseline_qmt_native.yaml"
    status, identity = inspect_qmt_input(training_config=config_path)
    experiment = status["experiment"]
    checked = status.get("training_preflight", {})
    trainable = checked.get("status") == "passed"
    summary = checked.get("summary", status["summary"])
    public_identity = {key: value for key, value in (identity or {}).items() if key != "input_dir"}
    preflight = {
        "status": "passed" if trainable else "blocked", "provider": "qmt_native",
        "provider_identity": public_identity, "checked_at": status["checked_at"],
        "summary": summary, "time_semantics": checked.get("time_semantics"),
    }
    current = {
        **public_identity, "provider": "qmt_native",
        "status": "ready" if identity is not None else "blocked",
        "calendar_end": experiment["calendar_end"] if experiment else None,
        "error": None if identity is not None else status["summary"],
    }
    reports_dir = RESEARCH_STATUS_PATH.parent
    latest_training = _latest_training_payload(reports_dir)
    if latest_training is not None:
        latest_training["provider_current"] = identity is not None and provider_lineages_match(
            latest_training.get("provider_identity"), public_identity,
        )
        latest_training["provider_status"] = "current" if latest_training["provider_current"] else "historical"
    external_tools = _external_tools_payload()
    return _sanitize_public_tree({
        "generated_at": status["checked_at"], "trainable": trainable,
        "training_readiness": {"status": "trainable" if trainable else "blocked", "summary": summary},
        "training_preflight": preflight, "training_preflights": {"qmt_native": preflight},
        "current_provider_identities": {"qmt_native": current}, "provider_quality": {},
        "qlib_provider": {
            **public_identity, "provider_source": "qmt_native", "status": current["status"],
            "calendar_days": experiment["calendar_days"] if experiment else 0,
            "calendar_range": f"{experiment['calendar_start']} ~ {experiment['calendar_end']}" if experiment else "",
            "instruments": experiment["symbol_count"] if experiment else 0,
        },
        "data_sources": [{
            "name": "qmt_native", "role": "experiment", "status": "available",
            "files": 1, "rows": experiment["rows"],
            "date_range": f"{experiment['actual_start']} ~ {experiment['actual_end']}",
        }] if experiment else [],
        "latest_training": _sanitize_public_tree(latest_training),
        "cash_control_backtest": _sanitize_public_tree(_cash_control_backtest_payload(reports_dir)),
        "qlib_predictions": _sanitize_public_tree(_qlib_predictions_payload(reports_dir)),
        "qlib_research": _qlib_research_payload(reports_dir),
        "candidate_pipeline": _sanitize_public_tree(_read_json_object(reports_dir / CANDIDATE_PIPELINE_FILENAME)),
        "robust_strategy": _robust_strategy_payload(reports_dir),
        "robust_signal": _robust_signal_payload(reports_dir),
        "experiment_notebook": _experiment_notebook_payload(reports_dir),
        "mlflow": _mlflow_payload(),
        "external_tools": external_tools,
        "capability_matrix": _capability_matrix_payload(external_tools),
        "function_inventory": _function_inventory_payload(),
        "china_astock_reuse": _china_astock_reuse_payload(),
        "acceptance_checklist": _acceptance_checklist_payload(),
    })


@router.get("/status")
def get_research_status() -> Dict[str, Any]:
    """Return the latest quant workflow status used by the research page."""
    return build_research_status_payload()


@router.get("/experiments")
def get_experiment_notebook() -> Dict[str, Any]:
    """Return experiment records without scanning the Qlib provider."""
    return _sanitize_public_tree(_experiment_notebook_payload(RESEARCH_STATUS_PATH.parent))


@router.post("/experiments/{run_id}/archive")
async def archive_experiment_run(run_id: str) -> Dict[str, Any]:
    """Archive one experiment run from the active notebook list."""
    with _experiment_file_guard(RESEARCH_STATUS_PATH.parent):
        removed = _archive_experiment_run(RESEARCH_STATUS_PATH.parent, run_id)
        return _sanitize_public_tree({
            "removed_run": removed,
            "experiment_notebook": _experiment_notebook_payload_unlocked(
                RESEARCH_STATUS_PATH.parent
            ),
        })


@router.delete("/experiments/{run_id}")
async def delete_experiment_run(run_id: str) -> Dict[str, Any]:
    """Backward-compatible archive route for older local frontends."""
    return await archive_experiment_run(run_id)


@router.delete("/experiments/{run_id}/permanent")
async def permanently_delete_experiment_run(
    run_id: str,
    confirm: str = Query(
        "", description="Must be 'permanent' to confirm irreversible deletion."
    ),
) -> Dict[str, Any]:
    """Permanently remove one experiment run and its marked Markdown note block."""
    if confirm != "permanent":
        raise HTTPException(
            status_code=400, detail="Permanent deletion requires confirm=permanent."
        )

    reports_dir = RESEARCH_STATUS_PATH.parent
    with _experiment_file_guard(reports_dir):
        runs_path = reports_dir / EXPERIMENT_RUNS_FILENAME
        log_path = reports_dir / EXPERIMENT_LOG_FILENAME

        _ensure_marked_experiment_note_removable(reports_dir, run_id)
        runs_snapshot = _snapshot_text_file(runs_path)
        log_snapshot = _snapshot_text_file(log_path)
        try:
            removed = _remove_active_experiment_run(reports_dir, run_id)
            note_removed = _remove_marked_experiment_note(reports_dir, run_id)
        except Exception:
            _restore_text_file(runs_path, *runs_snapshot)
            _restore_text_file(log_path, *log_snapshot)
            raise
        return _sanitize_public_tree({
            "removed_run": removed,
            "note_removed": note_removed,
            "experiment_notebook": _experiment_notebook_payload_unlocked(reports_dir),
        })
