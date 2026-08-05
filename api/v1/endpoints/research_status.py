# -*- coding: utf-8 -*-
"""Research workspace status endpoint."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException, Query

from quant_workflow.scripts.quant_data_contract import (
    AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS,
    AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
    DataContractError,
    filesystem_state_fingerprint,
    validate_latest_pointer,
    validate_qlib_snapshot,
)
from quant_workflow.scripts._file_lock import lock_file, unlock_file
from quant_workflow.scripts.robust_signal_data_contract import (
    RobustSignalDataBlocked,
    RuntimeProviderIdentityCache,
    evaluate_robust_signal_data,
    public_readiness_payload,
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
NATIVE_QLIB_DATASET_ID = "qlib_ashare_v2"
NATIVE_QLIB_DATASET_ROOT = Path("qlib/cn_data_ashare_v2")
NATIVE_QLIB_LATEST_POINTER = NATIVE_QLIB_DATASET_ROOT / "LATEST.json"
NATIVE_QLIB_DATASET_ROOTS = {
    NATIVE_QLIB_DATASET_ID: NATIVE_QLIB_DATASET_ROOT,
    "qlib_ashare_cleaned_v2": Path("qlib/cn_data_ashare_cleaned_v2"),
}
NATIVE_QLIB_DATA_CONTRACT_KEYS = {
    "qlib_ashare_v2": "canonical",
    "qlib_ashare_cleaned_v2": "cleaned",
}
NATIVE_QLIB_REQUIRED_FEATURE_MIN_COVERAGE = 0.50
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
NATIVE_QLIB_VALIDATION_CACHE_TTL_SECONDS = 60.0
NATIVE_QLIB_VALIDATION_CACHE_MAX_ENTRIES = 8
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
PROVIDER_FRESHNESS_STALE_AFTER_DAYS = 4


@dataclass(frozen=True)
class _NativeQlibProviderResolution:
    authoritative: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    dataset_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    manifest_sha256: Optional[str] = None
    content_sha256: Optional[str] = None


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
_NATIVE_QLIB_VALIDATION_CACHE_LOCK = threading.Lock()
_NATIVE_QLIB_VALIDATION_CACHE: Dict[
    Tuple[str, str, str], Tuple[float, Dict[str, Any]]
] = {}
_DATA_HYGIENE_STATE_SCAN_LOCK = threading.Lock()


def _clear_native_qlib_validation_cache() -> None:
    with _NATIVE_QLIB_VALIDATION_CACHE_LOCK:
        _NATIVE_QLIB_VALIDATION_CACHE.clear()


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


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _annotate_provider_quality_freshness(
    provider_quality: Dict[str, Any],
    *,
    as_of_date: Optional[date] = None,
) -> None:
    """Add freshness metadata without changing structural training eligibility."""

    effective_as_of = as_of_date or datetime.now().astimezone().date()
    for record in provider_quality.values():
        if not isinstance(record, dict):
            continue
        record["structural_status"] = (
            "accepted" if record.get("trainable") is True else "blocked"
        )
        raw_calendar_end = record.get("calendar_end")
        calendar_end = None
        if isinstance(raw_calendar_end, str) and raw_calendar_end:
            try:
                calendar_end = date.fromisoformat(raw_calendar_end)
            except ValueError:
                calendar_end = None

        calendar_age_days = (
            (effective_as_of - calendar_end).days
            if calendar_end is not None
            else None
        )
        if calendar_age_days is None:
            status = "unknown"
        elif calendar_age_days < 0:
            status = "future"
        elif calendar_age_days > PROVIDER_FRESHNESS_STALE_AFTER_DAYS:
            status = "stale"
        else:
            status = "current"
        record["freshness"] = {
            "status": status,
            "calendar_end": (
                calendar_end.isoformat() if calendar_end is not None else None
            ),
            "as_of_date": effective_as_of.isoformat(),
            "calendar_age_days": calendar_age_days,
            "blocks_training": False,
        }


def _validate_native_qlib_snapshot_cached(
    provider_path: Path,
    *,
    manifest_sha256: str,
    evidence_root: Path,
) -> Dict[str, Any]:
    """Reuse expensive file hashes briefly; catalog and manifest checks stay live."""

    key = (
        str(provider_path.resolve()),
        manifest_sha256,
        str(evidence_root.resolve()),
    )
    now = time.monotonic()
    with _NATIVE_QLIB_VALIDATION_CACHE_LOCK:
        expired_keys = [
            cached_key
            for cached_key, (cached_at, _) in _NATIVE_QLIB_VALIDATION_CACHE.items()
            if now - cached_at >= NATIVE_QLIB_VALIDATION_CACHE_TTL_SECONDS
        ]
        for expired_key in expired_keys:
            _NATIVE_QLIB_VALIDATION_CACHE.pop(expired_key, None)
        cached = _NATIVE_QLIB_VALIDATION_CACHE.get(key)
        if cached is not None:
            return dict(cached[1])
        validated = validate_qlib_snapshot(
            provider_path,
            verify_hashes=True,
            evidence_root=evidence_root,
        )
        if len(_NATIVE_QLIB_VALIDATION_CACHE) >= NATIVE_QLIB_VALIDATION_CACHE_MAX_ENTRIES:
            oldest_key = min(
                _NATIVE_QLIB_VALIDATION_CACHE,
                key=lambda cached_key: _NATIVE_QLIB_VALIDATION_CACHE[cached_key][0],
            )
            _NATIVE_QLIB_VALIDATION_CACHE.pop(oldest_key, None)
        _NATIVE_QLIB_VALIDATION_CACHE[key] = (time.monotonic(), dict(validated))
        return dict(validated)


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
        return {key: _sanitize_public_tree(item) for key, item in value.items()}
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


def _robust_signal_data_readiness_payload(reports_dir: Path) -> Dict[str, Any]:
    status_path = (
        reports_dir
        / ROBUST_SIGNAL_DIRNAME
        / "robust_signal_data_readiness_latest.json"
    )
    try:
        payload = evaluate_robust_signal_data(
            status_path=status_path,
            data_manifest_path=QUANT_DATA_ROOT / DATA_MANIFEST_FILENAME,
            provider_root=(
                QUANT_DATA_ROOT / "runtime" / "robust_strategy" / "providers"
            ),
            quality_pointer_path=reports_dir / "data_quality_report_latest.json",
            provider_identity=_ROBUST_PROVIDER_IDENTITY,
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


def _data_quality_summary_from_json(
    path: Path,
) -> Optional[Dict[str, Any]]:
    publication = _data_quality_publication_from_json(path)
    if publication is None:
        return None
    return _data_quality_summary_from_markdown_bytes(publication.markdown_bytes)


def _apply_data_quality_training_gate(
    payload: Dict[str, Any],
    reports_dir: Path,
    native_resolution: _NativeQlibProviderResolution,
) -> None:
    raw_trainable = payload.get("trainable")
    if not isinstance(raw_trainable, bool):
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": (
                "Research status has an invalid trainable flag; rerun the full "
                "data quality report before training."
            ),
        }
        return
    provider_specific_gate = (
        native_resolution.authoritative and native_resolution.path is not None
    )
    if raw_trainable is False and not provider_specific_gate:
        payload["trainable"] = False
        return

    pointer_path = reports_dir / "data_quality_report_latest.json"
    status_report_id = payload.get("quality_report_id")
    publication = _data_quality_publication_from_json(pointer_path)
    if publication is None:
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": (
                "Latest data quality publication is missing or invalid; "
                "rerun the full data quality report before training."
            ),
        }
        return

    quality_payload = publication.payload
    current_report_id = publication.report_id
    payload["quality_scan"] = {
        "mode": quality_payload.get("scan_mode"),
        "coverage": quality_payload.get("coverage_status"),
        "verification_status": quality_payload.get("verification_status"),
        "requested_sample": quality_payload.get("requested_sample", 0),
        "scanned_instruments": quality_payload.get("scanned_instruments"),
    }
    if status_report_id != current_report_id:
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": (
                "Research status does not match the latest data quality report; "
                "rerun the full data quality report before training."
            ),
        }
        return

    status_evidence_sha256 = payload.get("quality_evidence_sha256")
    if (
        not publication.training_authoritative
        or not isinstance(status_evidence_sha256, str)
        or status_evidence_sha256 != publication.evidence_sha256
    ):
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": (
                "Latest data quality report is not fully verified and trainable; "
                "rerun the full data quality report before training."
            ),
        }
        return

    thresholds = quality_payload.get("training_thresholds")
    thresholds_verified = (
        isinstance(thresholds, dict)
        and thresholds.get("authoritative") is True
        and all(
            isinstance(thresholds.get(field), int)
            and not isinstance(thresholds.get(field), bool)
            and thresholds[field] >= minimum
            for field, minimum in (
                (
                    "min_calendar_days",
                    AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS,
                ),
                ("min_instruments", AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS),
                (
                    "min_merged_instruments",
                    AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
                ),
            )
        )
    )
    fully_verified = (
        quality_payload.get("scan_mode") == "full"
        and quality_payload.get("coverage_status") == "full"
        and quality_payload.get("verification_status") == "verified"
        and thresholds_verified
        and (
            provider_specific_gate
            or quality_payload.get("trainable") is True
        )
    )
    if not fully_verified:
        payload["trainable"] = False
        payload["training_readiness"] = {
            "status": "blocked",
            "summary": (
                "Latest data quality report is not fully verified and trainable; "
                "rerun the full data quality report before training."
            ),
        }
        return

    if native_resolution.authoritative and native_resolution.path is not None:
        data_contract = quality_payload.get("data_contract")
        provider_key = NATIVE_QLIB_DATA_CONTRACT_KEYS.get(
            str(native_resolution.dataset_id or "")
        )
        provider_identities = (
            data_contract.get("qlib_providers")
            if isinstance(data_contract, dict)
            else None
        )
        qlib_identity = (
            provider_identities.get(provider_key)
            if isinstance(provider_identities, dict) and provider_key is not None
            else None
        )
        if qlib_identity is None and provider_key == "canonical":
            qlib_identity = (
                data_contract.get("qlib_provider")
                if isinstance(data_contract, dict)
                else None
            )
        expected_identity = {
            "dataset_id": native_resolution.dataset_id,
            "snapshot_id": native_resolution.snapshot_id,
            "manifest_sha256": native_resolution.manifest_sha256,
            "content_sha256": native_resolution.content_sha256,
        }
        identity_matches = (
            isinstance(data_contract, dict)
            and data_contract.get("status") == "verified"
            and isinstance(qlib_identity, dict)
            and all(
                isinstance(expected, str)
                and expected
                and qlib_identity.get(field) == expected
                for field, expected in expected_identity.items()
            )
        )
        if not identity_matches:
            payload["trainable"] = False
            payload["training_readiness"] = {
                "status": "blocked",
                "summary": (
                    "Latest data quality evidence does not match the current "
                    "native V2 snapshot; rerun the full data quality report before training."
                ),
            }
            return

        provider_quality = quality_payload.get("provider_quality")
        quality_record = (
            provider_quality.get(provider_key)
            if isinstance(provider_quality, dict) and provider_key is not None
            else None
        )
        payload["provider_quality"] = (
            provider_quality if isinstance(provider_quality, dict) else {}
        )
        record_identity_matches = isinstance(quality_record, dict) and all(
            quality_record.get(field) == expected
            for field, expected in expected_identity.items()
        )
        record_metrics_valid = (
            isinstance(quality_record, dict)
            and isinstance(quality_record.get("calendar_valid"), bool)
            and _is_nonnegative_int(
                quality_record.get("invalid_calendar_entries")
            )
            and isinstance(quality_record.get("instrument_records_valid"), bool)
            and _is_nonnegative_int(
                quality_record.get("invalid_instrument_records")
            )
            and isinstance(quality_record.get("calendar_days"), int)
            and not isinstance(quality_record.get("calendar_days"), bool)
            and isinstance(quality_record.get("instruments"), int)
            and not isinstance(quality_record.get("instruments"), bool)
            and isinstance(quality_record.get("instrument_feature_dirs"), int)
            and not isinstance(
                quality_record.get("instrument_feature_dirs"), bool
            )
            and isinstance(
                quality_record.get("missing_instrument_feature_dirs"), int
            )
            and not isinstance(
                quality_record.get("missing_instrument_feature_dirs"), bool
            )
            and isinstance(
                quality_record.get("complete_instrument_feature_dirs"), int
            )
            and not isinstance(
                quality_record.get("complete_instrument_feature_dirs"), bool
            )
            and quality_record.get("complete_instrument_feature_dirs") >= 0
            and isinstance(
                quality_record.get("incomplete_instrument_feature_dirs"), int
            )
            and not isinstance(
                quality_record.get("incomplete_instrument_feature_dirs"), bool
            )
            and quality_record.get("incomplete_instrument_feature_dirs") >= 0
            and isinstance(
                quality_record.get("missing_instrument_feature_files"), int
            )
            and not isinstance(
                quality_record.get("missing_instrument_feature_files"), bool
            )
            and quality_record.get("missing_instrument_feature_files") >= 0
            and isinstance(
                quality_record.get("invalid_instrument_feature_files"), int
            )
            and not isinstance(
                quality_record.get("invalid_instrument_feature_files"), bool
            )
            and quality_record.get("invalid_instrument_feature_files") >= 0
            and _is_finite_number(
                quality_record.get("required_feature_min_coverage")
            )
            and float(quality_record.get("required_feature_min_coverage"))
            == NATIVE_QLIB_REQUIRED_FEATURE_MIN_COVERAGE
            and _is_finite_number(
                quality_record.get("minimum_valid_instrument_feature_coverage")
            )
            and 0.0
            <= float(
                quality_record.get("minimum_valid_instrument_feature_coverage")
            )
            <= 1.0
            and _is_nonnegative_int(
                quality_record.get("misaligned_instrument_feature_dirs")
            )
            and quality_record.get("complete_instrument_feature_dirs")
            + quality_record.get("incomplete_instrument_feature_dirs")
            == quality_record.get("instruments")
            and isinstance(quality_record.get("benchmark_complete"), bool)
            and isinstance(quality_record.get("failure_reasons"), list)
            and all(
                isinstance(reason, str)
                for reason in quality_record.get("failure_reasons", [])
            )
            and isinstance(quality_record.get("trainable"), bool)
            and (
                quality_record.get("trainable") is not True
                or (
                    quality_record.get("complete_instrument_feature_dirs")
                    >= thresholds["min_instruments"]
                    and quality_record.get("incomplete_instrument_feature_dirs")
                    == 0
                    and quality_record.get("missing_instrument_feature_files")
                    == 0
                    and quality_record.get("invalid_instrument_feature_files")
                    == 0
                    and quality_record.get("calendar_valid") is True
                    and quality_record.get("invalid_calendar_entries") == 0
                    and quality_record.get("instrument_records_valid") is True
                    and quality_record.get("invalid_instrument_records") == 0
                    and float(
                        quality_record.get(
                            "minimum_valid_instrument_feature_coverage"
                        )
                    )
                    >= NATIVE_QLIB_REQUIRED_FEATURE_MIN_COVERAGE
                    and quality_record.get("misaligned_instrument_feature_dirs")
                    == 0
                )
            )
        )
        if not record_identity_matches or not record_metrics_valid:
            payload["trainable"] = False
            payload["training_readiness"] = {
                "status": "blocked",
                "summary": (
                    "Latest data quality evidence does not contain valid "
                    "provider-specific quality for the requested native V2 snapshot; "
                    "rerun the full data quality report before training."
                ),
            }
            return

        if quality_record.get("trainable") is not True:
            readiness = quality_record.get("training_readiness")
            payload["trainable"] = False
            if (
                isinstance(readiness, dict)
                and isinstance(readiness.get("status"), str)
                and readiness.get("status") != "trainable"
                and isinstance(readiness.get("summary"), str)
            ):
                payload["training_readiness"] = dict(readiness)
            else:
                payload["training_readiness"] = {
                    "status": "blocked",
                    "summary": (
                        "The requested native V2 provider did not pass its own "
                        "quality checks; rerun the full data quality report after repair."
                    ),
                }
            return

        snapshot_state = data_contract.get("snapshot_state")
        snapshot_stages = (
            "before",
            "after_scan",
            "after_revalidation",
            "after",
        )
        provider_states = []
        for stage in snapshot_stages:
            stage_states = (
                snapshot_state.get(stage)
                if isinstance(snapshot_state, dict)
                else None
            )
            provider_states.append(
                stage_states.get(native_resolution.dataset_id)
                if isinstance(stage_states, dict)
                else None
            )
        expected_before = provider_states[0]
        expected_after = provider_states[-1]
        stable_state_is_valid = (
            isinstance(snapshot_state, dict)
            and snapshot_state.get("stable") is True
            and snapshot_state.get("content_revalidated") is True
            and all(
                _valid_filesystem_state_fingerprint(state)
                for state in provider_states
            )
            and all(state == expected_before for state in provider_states[1:])
        )
        if not stable_state_is_valid:
            payload["trainable"] = False
            payload["training_readiness"] = {
                "status": "blocked",
                "summary": (
                    "Latest data quality evidence does not contain a stable provider "
                    "snapshot state; rerun the full data quality report before training."
                ),
            }
            return
        try:
            current_state = _filesystem_state_fingerprint_live(
                native_resolution.path
            )
        except (DataContractError, OSError):
            current_state = None
        if current_state != expected_after:
            payload["trainable"] = False
            payload["training_readiness"] = {
                "status": "blocked",
                "summary": (
                    "The native V2 provider changed after the latest data quality "
                    "report; rerun the full report before training."
                ),
            }
            return

        provider_thresholds_pass = (
            quality_record["calendar_days"] >= thresholds["min_calendar_days"]
            and quality_record["instruments"] >= thresholds["min_instruments"]
            and quality_record["instrument_feature_dirs"]
            >= thresholds["min_instruments"]
            and quality_record["missing_instrument_feature_dirs"] == 0
            and quality_record["benchmark_complete"] is True
            and quality_record["invalid_instrument_feature_files"] == 0
            and quality_record["calendar_valid"] is True
            and quality_record["invalid_calendar_entries"] == 0
            and quality_record["instrument_records_valid"] is True
            and quality_record["invalid_instrument_records"] == 0
            and quality_record["minimum_valid_instrument_feature_coverage"]
            >= NATIVE_QLIB_REQUIRED_FEATURE_MIN_COVERAGE
            and quality_record["misaligned_instrument_feature_dirs"] == 0
            and not quality_record["failure_reasons"]
        )
        readiness = quality_record.get("training_readiness")
        if (
            not provider_thresholds_pass
            or not isinstance(readiness, dict)
            or readiness.get("status") != "trainable"
        ):
            payload["trainable"] = False
            payload["training_readiness"] = {
                "status": "blocked",
                "summary": (
                    "The requested native V2 provider has inconsistent quality "
                    "evidence; rerun the full data quality report before training."
                ),
            }
            return

        payload["trainable"] = True
        payload["training_readiness"] = dict(readiness)


def _data_quality_summary_from_markdown(path: Path) -> Optional[Dict[str, Any]]:
    markdown_bytes = _read_bounded_regular_file_at(
        path.parent,
        Path(path.name),
        max_bytes=DATA_QUALITY_MARKDOWN_MAX_BYTES,
    )
    if markdown_bytes is None:
        return None
    return _data_quality_summary_from_markdown_bytes(markdown_bytes)


def _data_quality_summary_from_markdown_bytes(
    markdown_bytes: bytes,
) -> Optional[Dict[str, Any]]:
    try:
        lines = markdown_bytes.decode("utf-8").splitlines()
    except UnicodeError:
        return None

    summary: Dict[str, Any] = {}
    prefixes = {
        "Stock universe": "stock_universe",
        "Canonical qfq usable instruments": "usable_instruments",
        "Merged usable instruments": "usable_instruments",
        "Canonical qfq rows": "total_rows",
        "Merged rows": "total_rows",
        "Canonical qfq date range": "date_range",
        "Calendar range": "date_range",
    }
    for line in lines[:30]:
        clean = line.strip()
        if clean.startswith("- "):
            clean = clean[2:].strip()
        for prefix, key in prefixes.items():
            marker = f"{prefix}:"
            if clean.startswith(marker):
                summary[key] = clean[len(marker) :].strip()
                break
    return summary or None


def _data_quality_summary_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _data_quality_summary_from_json(
        reports_dir / "data_quality_report_latest.json",
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


def _is_trusted_qlib_provider_path(path: Path) -> bool:
    provider_path = _resolve_for_boundary(path)
    provider_root = _resolve_for_boundary(QLIB_PROVIDER_ROOT)
    if provider_path == provider_root or not _is_relative_to(
        provider_path, provider_root
    ):
        return False
    required_children = ("calendars", "instruments", "features")
    return all((provider_path / child).is_dir() for child in required_children)


def _resolve_native_qlib_provider(
    dataset_id: str = NATIVE_QLIB_DATASET_ID,
) -> _NativeQlibProviderResolution:
    """Resolve the provider selected by the live v2 catalog.

    A saved training report is historical evidence. It must not decide which
    provider the current status page calls canonical. The catalog and its
    immutable LATEST pointer are the source of truth for that decision.
    """

    # Tests and callers may inspect an isolated report artifact. Do not mix it
    # with the process-wide live catalog unless it is the configured report
    # location; this also prevents an arbitrary report path from selecting a
    # real local provider during diagnostics.
    if _resolve_for_boundary(RESEARCH_STATUS_PATH.parent) != _resolve_for_boundary(
        QUANT_WORKFLOW_REPORTS_ROOT
    ):
        return _NativeQlibProviderResolution(authoritative=False)

    dataset_root = NATIVE_QLIB_DATASET_ROOTS.get(dataset_id)
    if dataset_root is None:
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="训练结果记录了不受支持的原生 V2 数据集。",
        )
    latest_pointer = dataset_root / "LATEST.json"
    root = _resolve_for_boundary(QUANT_DATA_ROOT)
    catalog_path = root / DATA_MANIFEST_FILENAME
    if catalog_path.is_symlink():
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 根清单路径不安全，已阻止回退到旧兼容数据。",
        )
    catalog = _read_json_object(catalog_path)
    if catalog is None:
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 根清单缺失或损坏，已阻止回退到旧兼容数据。",
        )
    if (
        catalog.get("schema_version") != 2
        or catalog.get("path_format") != "relative_to_quant_data_root"
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 根清单契约无效，已阻止回退到旧兼容数据。",
        )
    datasets = catalog.get("datasets") if isinstance(catalog, dict) else None
    entry = (
        datasets.get(dataset_id)
        if isinstance(datasets, dict)
        else None
    )
    if not isinstance(entry, dict):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 数据集未在根清单中登记，已阻止回退到旧兼容数据。",
        )
    if (
        entry.get("dataset_id") != dataset_id
        or entry.get("dataset_root") != dataset_root.as_posix()
        or entry.get("role") != "canonical"
        or entry.get("status") != "ready"
        or entry.get("price_basis") != "qfq"
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 根清单口径无效，已阻止回退到旧兼容数据。",
        )

    if entry.get("latest_pointer") != latest_pointer.as_posix():
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 的 LATEST 指针位置无效，已阻止回退到旧兼容数据。",
        )
    pointer_path = _safe_manifest_path(root, latest_pointer.as_posix())
    if pointer_path is None or _manifest_path_contains_symlink(
        root, latest_pointer
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 的 LATEST 指针不安全，已阻止回退到旧兼容数据。",
        )
    pointer = _read_json_object(pointer_path)
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_version") != 2
        or pointer.get("dataset_id") != dataset_id
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 的 LATEST 指针缺失或损坏，已阻止回退到旧兼容数据。",
        )

    snapshot_id = entry.get("snapshot_id")
    manifest_sha256 = entry.get("snapshot_manifest_sha256")
    if (
        not isinstance(snapshot_id, str)
        or not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
        or not isinstance(manifest_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
        or snapshot_id != pointer.get("snapshot_id")
        or manifest_sha256 != pointer.get("manifest_sha256")
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照标识或清单哈希不一致，已阻止回退到旧兼容数据。",
        )

    expected_snapshot_path = dataset_root / "snapshots" / snapshot_id
    if pointer.get("snapshot_path") != expected_snapshot_path.as_posix():
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照指针指向了错误的数据集目录，已阻止使用。",
        )
    provider_path = _safe_manifest_path(root, expected_snapshot_path.as_posix())
    if provider_path is None or _manifest_path_contains_symlink(
        root, expected_snapshot_path
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照路径不安全，已阻止使用。",
        )
    if not _is_trusted_qlib_provider_path(provider_path):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照目录不完整，已阻止使用。",
        )

    dataset_manifest_path = provider_path / "DATASET_MANIFEST.json"
    if dataset_manifest_path.is_symlink() or not dataset_manifest_path.is_file():
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照缺少可信的数据集清单，已阻止使用。",
        )
    dataset_manifest, dataset_manifest_bytes = _read_json_object_bytes(
        dataset_manifest_path
    )
    actual_manifest_sha256 = (
        hashlib.sha256(dataset_manifest_bytes).hexdigest()
        if dataset_manifest_bytes is not None
        else None
    )
    if actual_manifest_sha256 != manifest_sha256:
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照清单校验失败，已阻止回退到旧兼容数据。",
        )
    try:
        validated_manifest = _validate_native_qlib_snapshot_cached(
            provider_path,
            manifest_sha256=manifest_sha256,
            evidence_root=root,
        )
    except (DataContractError, OSError, UnicodeError, TypeError, ValueError):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照完整性校验失败，已阻止回退到旧兼容数据。",
        )
    if entry.get("content_sha256") != dataset_manifest.get("content_sha256"):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 根清单与快照的内容哈希不一致，已阻止使用。",
        )
    if (
        not isinstance(dataset_manifest, dict)
        or dataset_manifest != validated_manifest
        or dataset_manifest.get("dataset_id") != dataset_id
        or dataset_manifest.get("snapshot_id") != snapshot_id
        or dataset_manifest.get("role") != "canonical"
        or dataset_manifest.get("status") != "ready"
        or dataset_manifest.get("price_basis") != "qfq"
        or entry.get("upstream_snapshots", [])
        != dataset_manifest.get("upstream_snapshots", [])
        or entry.get("dependency_snapshots", [])
        != dataset_manifest.get("dependency_snapshots", [])
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            error="原生 V2 快照清单校验失败，已阻止回退到旧兼容数据。",
        )
    lineage = [
        *dataset_manifest.get("upstream_snapshots", []),
        *dataset_manifest.get("dependency_snapshots", []),
    ]
    for upstream in lineage:
        upstream_entry = (
            datasets.get(upstream.get("dataset_id"))
            if isinstance(upstream, dict) and isinstance(datasets, dict)
            else None
        )
        if not isinstance(upstream_entry, dict) or upstream_entry.get(
            "status"
        ) != "ready":
            return _NativeQlibProviderResolution(
                authoritative=True,
                error="原生 V2 上游数据已失效，当前训练入口已阻断。",
            )
        if any(
            upstream.get(field) != upstream_entry.get(field)
            for field in ("snapshot_id", "content_sha256", "price_basis")
        ):
            return _NativeQlibProviderResolution(
                authoritative=True,
                error="原生 V2 上游快照已变化，当前训练入口已阻断。",
            )
    return _NativeQlibProviderResolution(
        authoritative=True,
        path=provider_path,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        content_sha256=str(dataset_manifest.get("content_sha256") or ""),
    )


def _saved_provider_identity(
    payload: Dict[str, Any],
) -> Optional[tuple[str, str, str, str]]:
    raw_identity = payload.get("provider_identity")
    if not isinstance(raw_identity, dict):
        return None
    identity = raw_identity
    provider_contract = identity.get("provider_contract")
    dataset_id = identity.get("dataset_id")
    snapshot_id = identity.get("snapshot_id")
    content_sha256 = identity.get("content_sha256")
    manifest_sha256 = identity.get("manifest_sha256") or identity.get(
        "snapshot_manifest_sha256"
    )
    if (
        provider_contract != "immutable_snapshot_v2"
        or not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(snapshot_id, str)
        or not snapshot_id
        or not isinstance(manifest_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
        or not isinstance(content_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", content_sha256)
    ):
        return None
    return dataset_id, snapshot_id, content_sha256, manifest_sha256


def _annotate_saved_provider(
    payload: Dict[str, Any],
    resolution: _NativeQlibProviderResolution,
    *,
    artifact_label: str,
) -> None:
    if not resolution.authoritative:
        return

    identity = _saved_provider_identity(payload)
    target_resolution = _current_provider_resolution_for_payload(payload, resolution)

    if target_resolution.path is None:
        payload["provider_current"] = False
        payload["provider_status"] = "catalog_invalid"
        payload["provider_warning"] = target_resolution.error
        return

    expected_identity = (
        target_resolution.dataset_id,
        target_resolution.snapshot_id,
        target_resolution.content_sha256,
        target_resolution.manifest_sha256,
    )
    if identity is not None:
        current = identity == expected_identity
        payload["provider_current"] = current
        payload["provider_status"] = "current_native_v2" if current else "stale"
        if not current:
            payload["provider_warning"] = (
                f"{artifact_label}记录的不可变快照身份与当前原生 V2 不一致，"
                "只能作为历史结果。"
            )
        return

    provider_uri = payload.get("provider_uri") or payload.get("qlib_provider")
    path_points_to_current = False
    if isinstance(provider_uri, str) and provider_uri.strip():
        path_points_to_current = (
            _resolve_for_boundary(Path(provider_uri).expanduser())
            == _resolve_for_boundary(target_resolution.path)
        )
    payload["provider_current"] = False
    payload["provider_status"] = "unknown" if path_points_to_current else "stale"
    if path_points_to_current:
        payload["provider_warning"] = (
            f"{artifact_label}只记录了可变路径，没有记录数据集、快照和清单哈希，"
            "无法证明它使用了当前原生 V2 数据。"
        )
    else:
        payload["provider_warning"] = (
            f"{artifact_label}使用的 provider 不是当前原生 V2 数据，"
            "只能作为历史结果。"
        )


def _current_provider_resolution_for_payload(
    payload: Dict[str, Any],
    fallback: _NativeQlibProviderResolution,
) -> _NativeQlibProviderResolution:
    identity = _saved_provider_identity(payload)
    if identity is None or identity[0] not in NATIVE_QLIB_DATASET_ROOTS:
        return fallback
    if identity[0] == fallback.dataset_id:
        return fallback
    return _resolve_native_qlib_provider(identity[0])


def _resolve_saved_native_qlib_snapshot(
    identity: tuple[str, str, str, str],
) -> _NativeQlibProviderResolution:
    dataset_id, snapshot_id, content_sha256, manifest_sha256 = identity
    dataset_root = NATIVE_QLIB_DATASET_ROOTS.get(dataset_id)
    if dataset_root is None:
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练结果记录了不受支持的原生 V2 数据集。",
        )
    root = _resolve_for_boundary(QUANT_DATA_ROOT)
    relative_snapshot = dataset_root / "snapshots" / snapshot_id
    provider_path = _safe_manifest_path(root, relative_snapshot.as_posix())
    if provider_path is None or _manifest_path_contains_symlink(
        root, relative_snapshot
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练结果记录的原生 V2 历史快照路径不安全。",
        )
    if not _is_trusted_qlib_provider_path(provider_path):
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练使用的原生 V2 历史快照已不存在或目录不完整。",
        )
    manifest_path = provider_path / "DATASET_MANIFEST.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练使用的原生 V2 历史快照缺少可信清单。",
        )
    manifest, manifest_bytes = _read_json_object_bytes(manifest_path)
    actual_manifest_sha256 = (
        hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_bytes is not None
        else None
    )
    if actual_manifest_sha256 != manifest_sha256:
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练使用的原生 V2 历史快照清单哈希不一致。",
        )
    try:
        validated_manifest = _validate_native_qlib_snapshot_cached(
            provider_path,
            manifest_sha256=manifest_sha256,
            evidence_root=root,
        )
    except (DataContractError, OSError, UnicodeError, TypeError, ValueError):
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练使用的原生 V2 历史快照完整性校验失败。",
        )
    if (
        not isinstance(manifest, dict)
        or manifest != validated_manifest
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("snapshot_id") != snapshot_id
        or manifest.get("content_sha256") != content_sha256
        or manifest.get("role") != "canonical"
        or manifest.get("status") != "ready"
        or manifest.get("price_basis") != "qfq"
    ):
        return _NativeQlibProviderResolution(
            authoritative=True,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            error="训练使用的原生 V2 历史快照身份校验失败。",
        )
    return _NativeQlibProviderResolution(
        authoritative=True,
        path=provider_path,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        content_sha256=content_sha256,
        manifest_sha256=manifest_sha256,
    )


def _saved_provider_resolution_for_payload(
    payload: Dict[str, Any],
    fallback: _NativeQlibProviderResolution,
) -> _NativeQlibProviderResolution:
    identity = _saved_provider_identity(payload)
    if identity is None:
        return _current_provider_resolution_for_payload(payload, fallback)
    current = _current_provider_resolution_for_payload(payload, fallback)
    current_identity = (
        current.dataset_id,
        current.snapshot_id,
        current.content_sha256,
        current.manifest_sha256,
    )
    if current.path is not None and identity == current_identity:
        return current
    return _resolve_saved_native_qlib_snapshot(identity)


def _annotate_provider_payload(
    payload: Dict[str, Any],
    native_resolution: Optional[_NativeQlibProviderResolution] = None,
) -> None:
    """Describe whether a saved report used the current native provider."""

    resolution = native_resolution or _resolve_native_qlib_provider()
    _annotate_saved_provider(payload, resolution, artifact_label="候选报告")


def _qlib_provider_from_path(path: Path) -> Dict[str, Any]:
    provider: Dict[str, Any] = {"path": str(path)}
    calendar_path = path / "calendars" / "day.txt"
    instruments_path = path / "instruments" / "all.txt"
    features_path = path / "features"

    try:
        if calendar_path.exists():
            calendar_days = [
                line.strip()
                for line in calendar_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            provider["calendar_days"] = len(calendar_days)
            if calendar_days:
                provider["calendar_range"] = f"{calendar_days[0]} ~ {calendar_days[-1]}"
        if instruments_path.exists():
            provider["instruments"] = sum(
                1
                for line in instruments_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if features_path.exists():
            provider["feature_dirs"] = sum(
                1 for item in features_path.iterdir() if item.is_dir()
            )
    except OSError:
        return provider

    size_mb = _directory_size_mb(path)
    if size_mb is not None:
        provider["size_mb"] = size_mb
    return provider


def _annotate_latest_training_provider(
    latest_training: Dict[str, Any],
    native_resolution: _NativeQlibProviderResolution,
) -> None:
    """Mark historical training evidence relative to the live native provider."""

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
    provider_uri = latest_training.get("provider_uri") or latest_training.get("qlib_provider")
    existing = payload.get("qlib_provider")
    provider = dict(existing) if isinstance(existing, dict) else {}
    target_resolution = _saved_provider_resolution_for_payload(
        latest_training,
        native_resolution,
    )

    if target_resolution.authoritative:
        dataset_id = target_resolution.dataset_id or NATIVE_QLIB_DATASET_ID
        if target_resolution.path is None:
            payload["latest_training_provider"] = {
                "dataset_id": dataset_id,
                "role": "canonical",
                "status": "blocked",
                "price_basis": "qfq",
                "provider_source": "native_v2_catalog",
                "provider_error": target_resolution.error,
                "provider_current": False,
                "provider_trusted": False,
            }
            provider["latest_training_provider_trusted"] = False
            provider["latest_training_provider_current"] = False
            payload["qlib_provider"] = provider
            return

        native_provider_path = target_resolution.path
        saved_identity = _saved_provider_identity(latest_training)
        resolved_identity = (
            target_resolution.dataset_id,
            target_resolution.snapshot_id,
            target_resolution.content_sha256,
            target_resolution.manifest_sha256,
        )
        identity_trusted = saved_identity is not None and saved_identity == resolved_identity
        training_provider = _qlib_provider_from_path(native_provider_path)
        training_provider.update(
            {
                "dataset_id": dataset_id,
                "role": "canonical",
                "status": "ready",
                "price_basis": "qfq",
                "provider_source": (
                    "native_v2_catalog"
                    if latest_training.get("provider_current") is True
                    else "saved_immutable_v2_identity"
                ),
                "snapshot_id": target_resolution.snapshot_id,
                "manifest_sha256": target_resolution.manifest_sha256,
                "content_sha256": target_resolution.content_sha256,
                "provider_current": latest_training.get("provider_current") is True,
                "provider_trusted": identity_trusted,
            }
        )
        if latest_training.get("provider_current") is True:
            provider["latest_training_provider_trusted"] = True
            provider["latest_training_provider_current"] = True
            provider.pop("latest_training_provider_uri", None)
            provider.pop("provider_uri_warning", None)
        elif identity_trusted:
            provider["latest_training_provider_trusted"] = True
            provider["latest_training_provider_current"] = False
            warning = latest_training.get("provider_warning") or (
                "latest_training 使用的是已验证的原生 V2 历史快照；"
                "它不是当前训练入口，但可用于复现当时结果。"
            )
            provider["provider_uri_warning"] = warning
            training_provider["provider_warning"] = warning
        else:
            if isinstance(provider_uri, str) and provider_uri.strip():
                provider["latest_training_provider_uri"] = str(
                    Path(provider_uri).expanduser()
                )
            provider["latest_training_provider_trusted"] = False
            provider["latest_training_provider_current"] = False
            warning = latest_training.get(
                "provider_warning"
            ) or (
                f"latest_training 无法证明使用了当前原生 {dataset_id}，"
                "请重新运行训练后再把该结果作为当前结果使用。"
            )
            provider["provider_uri_warning"] = warning
            training_provider["provider_warning"] = warning
        payload["latest_training_provider"] = training_provider
        payload["qlib_provider"] = provider
        return

    if not isinstance(provider_uri, str) or not provider_uri.strip():
        return

    provider_path = Path(provider_uri).expanduser()
    if not _is_trusted_qlib_provider_path(provider_path):
        provider["latest_training_provider_uri"] = str(provider_path)
        provider["latest_training_provider_trusted"] = False
        provider["provider_uri_warning"] = (
            "latest_training provider_uri is outside configured Qlib provider root "
            "or is not a concrete Qlib provider; skipped provider scan."
        )
        payload["latest_training_provider"] = {
            "path": str(provider_path),
            "provider_current": False,
            "provider_trusted": False,
            "provider_warning": provider["provider_uri_warning"],
        }
        payload["qlib_provider"] = provider
        return

    training_provider = _qlib_provider_from_path(provider_path)
    training_provider["provider_current"] = False
    training_provider["provider_trusted"] = True
    provider["latest_training_provider_trusted"] = True
    provider["latest_training_provider_current"] = False
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
    quality_report_path = reports_dir / "data_quality_report_latest.json"
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

    quality_summary = _data_quality_summary_payload(reports_dir)
    quality_report_status = (
        "available"
        if quality_summary is not None
        else "invalid"
        if quality_report_path.exists()
        else "missing"
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
            "status": quality_report_status,
            "path": _public_path_label(quality_report_path)
            if quality_report_path.exists()
            else None,
            "summary": quality_summary,
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
            state_status = _inventory_snapshot_state_status(record)
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


def _inventory_snapshot_state_status(record: Dict[str, Any]) -> str:
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
    return "verified" if current == normalized_expected else "stale"


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
            "browser": "可从网页运行数据检查、质量报告、异常报告、Qlib 转换和训练入口",
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
                "网页白名单覆盖下载、JQData dry-run、Qlib 合并、清洗、检查、质量报告、异常报告、"
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


def _missing_status_payload() -> Dict[str, Any]:
    reports_dir = RESEARCH_STATUS_PATH.parent
    external_tools = _external_tools_payload()
    native_resolution = _resolve_native_qlib_provider()
    latest_training = _latest_training_payload(reports_dir)
    if latest_training is not None:
        _annotate_latest_training_provider(latest_training, native_resolution)
    qlib_provider = None
    if native_resolution.authoritative:
        if native_resolution.path is not None:
            qlib_provider = {
                **_qlib_provider_from_path(native_resolution.path),
                "dataset_id": NATIVE_QLIB_DATASET_ID,
                "role": "canonical",
                "status": "ready",
                "price_basis": "qfq",
                "provider_source": "native_v2_catalog",
                "snapshot_id": native_resolution.snapshot_id,
                "manifest_sha256": native_resolution.manifest_sha256,
                "content_sha256": native_resolution.content_sha256,
            }
        else:
            qlib_provider = {
                "dataset_id": NATIVE_QLIB_DATASET_ID,
                "role": "canonical",
                "status": "blocked",
                "price_basis": "qfq",
                "provider_source": "native_v2_catalog",
                "provider_error": native_resolution.error,
            }
    training_readiness = {
        "status": "missing",
        "summary": "Research status artifact is not available yet. Run quant_workflow/scripts/report_data_quality.py first.",
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
    provider_dataset_id: str = NATIVE_QLIB_DATASET_ID,
) -> Dict[str, Any]:
    """Build the latest quant workflow status used by web pages."""
    if not RESEARCH_STATUS_PATH.exists():
        return _missing_status_payload()

    try:
        payload = json.loads(RESEARCH_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fallback = _missing_status_payload()
        fallback["training_readiness"] = {
            "status": "invalid",
            "summary": f"Research status artifact could not be read: {exc}",
        }
        return fallback

    if not isinstance(payload, dict):
        fallback = _missing_status_payload()
        fallback["training_readiness"] = {
            "status": "invalid",
            "summary": "Research status artifact is not a JSON object.",
        }
        return fallback

    native_resolution = _resolve_native_qlib_provider(provider_dataset_id)
    reports_dir = RESEARCH_STATUS_PATH.parent
    _apply_data_quality_training_gate(payload, reports_dir, native_resolution)
    if native_resolution.path is not None:
        payload["qlib_provider"] = {
            **_qlib_provider_from_path(native_resolution.path),
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": native_resolution.dataset_id or provider_dataset_id,
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "provider_source": "native_v2_catalog",
            "snapshot_id": native_resolution.snapshot_id,
            "manifest_sha256": native_resolution.manifest_sha256,
            "content_sha256": native_resolution.content_sha256,
        }
    elif native_resolution.authoritative:
        payload["qlib_provider"] = {
            "dataset_id": provider_dataset_id,
            "role": "canonical",
            "status": "blocked",
            "price_basis": "qfq",
            "provider_source": "native_v2_catalog",
            "provider_error": native_resolution.error,
        }
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
    provider_quality = payload.get("provider_quality")
    if isinstance(provider_quality, dict):
        _annotate_provider_quality_freshness(provider_quality)
    return payload


@router.get("/status")
def get_research_status() -> Dict[str, Any]:
    """Return the latest quant workflow status used by the research page."""
    return build_research_status_payload()


@router.get("/experiments")
def get_experiment_notebook() -> Dict[str, Any]:
    """Return experiment records without scanning the Qlib provider."""
    return _experiment_notebook_payload(RESEARCH_STATUS_PATH.parent)


@router.post("/experiments/{run_id}/archive")
async def archive_experiment_run(run_id: str) -> Dict[str, Any]:
    """Archive one experiment run from the active notebook list."""
    with _experiment_file_guard(RESEARCH_STATUS_PATH.parent):
        removed = _archive_experiment_run(RESEARCH_STATUS_PATH.parent, run_id)
        return {
            "removed_run": removed,
            "experiment_notebook": _experiment_notebook_payload_unlocked(
                RESEARCH_STATUS_PATH.parent
            ),
        }


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
        return {
            "removed_run": removed,
            "note_removed": note_removed,
            "experiment_notebook": _experiment_notebook_payload_unlocked(reports_dir),
        }
