# -*- coding: utf-8 -*-
"""Research workspace status endpoint."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from urllib.parse import unquote, urlparse

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is only available on Windows.
    msvcrt = None

from fastapi import APIRouter, HTTPException, Query

from src.safe_urls import public_http_url


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DSA_ROOT = _env_path("DSA_ROOT", Path(__file__).resolve().parents[3])
PROJECTS_ROOT = _env_path("WORKING_PROJECTS_ROOT", DSA_ROOT.parent)
QUANT_WORKFLOW_ROOT = _env_path("QUANT_WORKFLOW_ROOT", DSA_ROOT / "quant_workflow")
QUANT_WORKFLOW_REPORTS_ROOT = _env_path("QUANT_WORKFLOW_REPORTS_ROOT", QUANT_WORKFLOW_ROOT / "reports")
QUANT_DATA_ROOT = _env_path("QUANT_DATA_ROOT", DSA_ROOT / "quant_data")
QLIB_PROVIDER_ROOT = _env_path("QLIB_PROVIDER_ROOT", QUANT_DATA_ROOT / "qlib")
RESEARCH_STATUS_PATH = QUANT_WORKFLOW_REPORTS_ROOT / "research_status_latest.json"
EXPERIMENT_LOG_FILENAME = "experiment_log.md"
EXPERIMENT_RUNS_FILENAME = "experiment_runs.jsonl"
EXPERIMENT_DELETED_RUNS_FILENAME = "experiment_runs_deleted.jsonl"
EXPERIMENT_LOCK_FILENAME = ".experiment.lock"
LATEST_TRAINING_RESULT_FILENAME = "latest_training_result.json"
CASH_CONTROL_BACKTEST_FILENAME = "cash_control_backtest_latest.json"
QLIB_PREDICTIONS_FILENAME = "qlib_predictions_latest.json"
QLIB_RESEARCH_REPORTS = {
    "model_explainability": ("模型效果深度解释", "qlib_research_model_explainability_latest.json"),
    "backtest_detail": ("回测详细分解", "qlib_research_backtest_detail_latest.json"),
    "model_comparison": ("多模型/多实验对比", "qlib_research_model_comparison_latest.json"),
    "factor_analysis": ("因子挖掘辅助", "qlib_research_factor_analysis_latest.json"),
    "rolling_training": ("滚动训练计划", "qlib_research_rolling_training_latest.json"),
    "risk_model": ("风险模型集成诊断", "qlib_research_risk_model_latest.json"),
}
EXPERIMENT_FILE_LOCK = threading.RLock()
DEFAULT_MLFLOW_URL = "http://127.0.0.1:6006/"
DIRECTORY_SIZE_CACHE_TTL_SECONDS = 60.0
DIRECTORY_SIZE_SCAN_MAX_ENTRIES = 50000
DIRECTORY_SIZE_SCAN_MAX_SECONDS = 0.5
CLEANUP_ACTION_PREVIEW_LIMIT = 50
TONGHUASHUN_CANDIDATE_PATHS = (
    Path("/Applications/同花顺.app"),
    Path("/Applications/同花顺远航版.app"),
    Path("/Applications/同花顺iFinD.app"),
    Path.home() / "Applications" / "同花顺.app",
    Path.home() / "Applications" / "同花顺远航版.app",
    Path.home() / "Applications" / "同花顺iFinD.app",
)
logger = logging.getLogger(__name__)

router = APIRouter()
_DIRECTORY_SIZE_CACHE_LOCK = threading.Lock()
_DIRECTORY_SIZE_CACHE: Dict[str, Tuple[float, float]] = {}


def _lock_experiment_handle(lock_handle) -> None:
    if fcntl is not None:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is not None:  # pragma: no cover - covered through patched fallback tests on non-Windows.
        lock_handle.seek(0, os.SEEK_END)
        if lock_handle.tell() == 0:
            lock_handle.write("0")
            lock_handle.flush()
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)


def _unlock_experiment_handle(lock_handle) -> None:
    if fcntl is not None:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover - covered through patched fallback tests on non-Windows.
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _experiment_file_guard(reports_dir: Path) -> Iterator[None]:
    """Guard experiment notebook mutations across threads and local processes."""
    with EXPERIMENT_FILE_LOCK:
        reports_dir.mkdir(parents=True, exist_ok=True)
        lock_path = reports_dir / EXPERIMENT_LOCK_FILENAME
        # Keep this small advisory lock file around deliberately. It only
        # coordinates writers that also take this lock; quant_workflow writers
        # must use the same path before mutating experiment notebook files.
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            _lock_experiment_handle(lock_handle)
            try:
                yield
            finally:
                _unlock_experiment_handle(lock_handle)


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
    raw_url = os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL).strip() or DEFAULT_MLFLOW_URL
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


def _data_quality_summary_from_json(path: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json_object(path)
    if not payload:
        return None
    raw_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    summary: Dict[str, Any] = {}
    for source_key, target_key in (
        ("stock_universe", "stock_universe"),
        ("usable_instruments", "usable_instruments"),
        ("merged_usable_instruments", "usable_instruments"),
        ("total_rows", "total_rows"),
        ("merged_rows", "total_rows"),
        ("date_range", "date_range"),
        ("calendar_range", "date_range"),
    ):
        value = raw_summary.get(source_key)
        if value not in (None, ""):
            summary[target_key] = str(value)
    return summary or None


def _data_quality_summary_from_markdown(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    summary: Dict[str, Any] = {}
    prefixes = {
        "Stock universe": "stock_universe",
        "Merged usable instruments": "usable_instruments",
        "Merged rows": "total_rows",
        "Calendar range": "date_range",
    }
    for line in lines[:30]:
        clean = line.strip()
        if clean.startswith("- "):
            clean = clean[2:].strip()
        for prefix, key in prefixes.items():
            marker = f"{prefix}:"
            if clean.startswith(marker):
                summary[key] = clean[len(marker):].strip()
                break
    return summary or None


def _data_quality_summary_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return (
        _data_quality_summary_from_json(reports_dir / "data_quality_report_latest.json")
        or _data_quality_summary_from_markdown(reports_dir / "data_quality_report_latest.md")
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
            if scanned_entries % 100 == 0 and time.monotonic() - started_at > DIRECTORY_SIZE_SCAN_MAX_SECONDS:
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


def _is_trusted_qlib_provider_path(path: Path) -> bool:
    provider_path = _resolve_for_boundary(path)
    provider_root = _resolve_for_boundary(QLIB_PROVIDER_ROOT)
    if provider_path == provider_root or not _is_relative_to(provider_path, provider_root):
        return False
    required_children = ("calendars", "instruments", "features")
    return all((provider_path / child).is_dir() for child in required_children)


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
            provider["feature_dirs"] = sum(1 for item in features_path.iterdir() if item.is_dir())
    except OSError:
        return provider

    size_mb = _directory_size_mb(path)
    if size_mb is not None:
        provider["size_mb"] = size_mb
    return provider


def _sync_qlib_provider_with_latest_training(payload: Dict[str, Any], latest_training: Dict[str, Any]) -> None:
    provider_uri = latest_training.get("provider_uri") or latest_training.get("qlib_provider")
    if not isinstance(provider_uri, str) or not provider_uri.strip():
        return

    provider_path = Path(provider_uri).expanduser()
    existing = payload.get("qlib_provider")
    provider = dict(existing) if isinstance(existing, dict) else {}
    if not _is_trusted_qlib_provider_path(provider_path):
        provider["latest_training_provider_uri"] = str(provider_path)
        provider["latest_training_provider_trusted"] = False
        provider["provider_uri_warning"] = (
            "latest_training provider_uri is outside configured Qlib provider root "
            "or is not a concrete Qlib provider; skipped provider scan."
        )
        payload["qlib_provider"] = provider
        return

    provider.update(_qlib_provider_from_path(provider_path))
    provider["latest_training_provider_trusted"] = True
    payload["qlib_provider"] = provider


def _experiment_notebook_payload(reports_dir: Path) -> Dict[str, Any]:
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


def _tonghuashun_candidate_paths() -> list[Path]:
    paths: list[Path] = []
    configured = os.environ.get("TONGHUASHUN_APP_PATH")
    if configured:
        paths.append(Path(configured).expanduser())
    paths.extend(Path(path).expanduser() for path in TONGHUASHUN_CANDIDATE_PATHS)
    return paths


def _tonghuashun_status_payload() -> Dict[str, Any]:
    detected_path = next((path for path in _tonghuashun_candidate_paths() if path.exists()), None)
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


def _data_hygiene_payload(reports_dir: Path) -> Dict[str, Any]:
    inventory_path = reports_dir / "data_hygiene_inventory_latest.json"
    cleanup_plan_path = reports_dir / "data_hygiene_cleanup_plan_latest.json"
    quality_report_path = reports_dir / "data_quality_report_latest.md"
    cleaning_strategy_path = reports_dir / "data_cleaning_strategy_latest.md"
    manifest_path = QUANT_DATA_ROOT / "DATA_MANIFEST.json"
    readme_path = QUANT_DATA_ROOT / "README.md"
    maintenance_lock_path = QUANT_DATA_ROOT / ".maintenance.lock"

    inventory = _read_json_object(inventory_path)
    cleanup_plan = _read_json_object(cleanup_plan_path)
    manifest = _read_json_object(manifest_path)
    official_entries = manifest.get("official_entries") if manifest else None

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
            cleanup_actions.append({
                "action": str(action.get("action") or ""),
                "source": _public_path_label(action.get("source") or ""),
                "destination": _public_path_label(action.get("destination") or "") or None,
                "reason": str(action.get("reason") or ""),
                "risk": str(action.get("risk") or "") or None,
                "requires_confirmation": action.get("requires_user_confirmation", False),
            })

    quality_summary = _data_quality_summary_payload(reports_dir)
    inventory_counts = _data_hygiene_inventory_counts(inventory)

    return {
        "inventory": {
            "status": "available" if inventory else "missing",
            "path": _public_path_label(inventory_path) if inventory_path.exists() else None,
            "generated_at": inventory.get("generated_at") if inventory else None,
            "official_entries": inventory_counts["official_entries"],
            "candidate_entries": inventory_counts["candidate_entries"],
            "legacy_entries": inventory_counts["legacy_entries"],
            "archive_candidates": inventory_counts["archive_candidates"],
        },
        "cleanup_plan": {
            "status": "available" if cleanup_plan else "missing",
            "path": _public_path_label(cleanup_plan_path) if cleanup_plan_path.exists() else None,
            "generated_at": cleanup_plan.get("generated_at") if cleanup_plan else None,
            "dry_run_only": cleanup_plan.get("dry_run_only", True) if cleanup_plan else True,
            "total_actions": len(cleanup_plan_actions),
            "preview_limit": CLEANUP_ACTION_PREVIEW_LIMIT,
            "truncated": len(cleanup_plan_actions) > len(cleanup_actions),
            "actions": cleanup_actions,
        },
        "quality_report": {
            "status": "available" if quality_report_path.exists() else "missing",
            "path": _public_path_label(quality_report_path) if quality_report_path.exists() else None,
            "summary": quality_summary,
        },
        "cleaning_strategy": {
            "status": "available" if cleaning_strategy_path.exists() else "missing",
            "path": _public_path_label(cleaning_strategy_path) if cleaning_strategy_path.exists() else None,
        },
        "manifest": {
            "status": "available" if manifest else "missing",
            "path": _public_path_label(manifest_path) if manifest_path.exists() else None,
            "generated_at": manifest.get("generated_at") if manifest else None,
            "source_report": _public_path_label(manifest.get("source_report")) if manifest else None,
            "official_entries": _collection_count(official_entries),
        },
        "readme": {
            "status": "available" if readme_path.exists() else "missing",
            "path": _public_path_label(readme_path) if readme_path.exists() else None,
        },
        "maintenance_lock": {
            "locked": maintenance_locked,
            "path": _public_path_label(maintenance_lock_path),
            "updated_at": _file_mtime_iso(maintenance_lock_path) if maintenance_locked else None,
        },
    }


def _collection_count(value: Any) -> int:
    if isinstance(value, (dict, list, tuple)):
        return len(value)
    return 0


def _data_hygiene_inventory_counts(inventory: Optional[Dict[str, Any]]) -> Dict[str, int]:
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
    tonghuashun = external_tools.get("tonghuashun") if isinstance(external_tools, dict) else None
    tonghuashun = tonghuashun if isinstance(tonghuashun, dict) else {}
    tonghuashun_capabilities = _string_items(tonghuashun.get("capabilities"))
    tonghuashun_restrictions = _string_items(tonghuashun.get("restrictions"))
    tonghuashun_summary = str(tonghuashun.get("summary") or "同花顺仅作为后续只读观察方向。")
    tonghuashun_status = "read_only" if tonghuashun.get("status") == "available" or tonghuashun.get("installed") else "deferred"
    tonghuashun_browser = tonghuashun_summary
    if tonghuashun_capabilities:
        tonghuashun_browser = f"{tonghuashun_browser} 能力：{'、'.join(tonghuashun_capabilities)}"

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
            "restrictions": ["不直接改造 Qlib 官方源码为主 UI", "MLflow 只作为实验原始记录页"],
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
            "gaps": "需要与 Qlib 模型回测结果继续统一解释口径。",
            "next": "在回测页继续拆分 DSA 准确性和 Qlib 策略结果。",
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
                "china-astock 对比、策略优化预演和 baseline 训练；"
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
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trainable": False,
        "training_readiness": {
            "status": "missing",
            "summary": "Research status artifact is not available yet. Run quant_workflow/scripts/report_data_quality.py first.",
        },
        "data_sources": [],
        "qlib_provider": None,
        "latest_training": _latest_training_payload(reports_dir),
        "cash_control_backtest": _cash_control_backtest_payload(reports_dir),
        "qlib_predictions": _qlib_predictions_payload(reports_dir),
        "experiment_notebook": _experiment_notebook_payload(reports_dir),
        "mlflow": _mlflow_payload(),
        "data_hygiene": _data_hygiene_payload(reports_dir),
        "external_tools": external_tools,
        "capability_matrix": _capability_matrix_payload(external_tools),
        "function_inventory": _function_inventory_payload(),
        "china_astock_reuse": _china_astock_reuse_payload(),
        "acceptance_checklist": _acceptance_checklist_payload(),
    }


def _remove_active_experiment_run(reports_dir: Path, run_id: str) -> Dict[str, Any]:
    runs_path = reports_dir / EXPERIMENT_RUNS_FILENAME

    if not runs_path.exists():
        raise HTTPException(status_code=404, detail="Experiment runs file is not available.")

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
        raise HTTPException(status_code=500, detail=f"Experiment runs file could not be read: {exc}") from exc

    if removed is None:
        raise HTTPException(status_code=404, detail=f"Experiment run {run_id} was not found.")

    tmp_path = runs_path.with_suffix(runs_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for item in remaining:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            for line in invalid_lines:
                handle.write(line + "\n")
        tmp_path.replace(runs_path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Experiment run could not be removed: {exc}") from exc

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
            handle.write(json.dumps(archived, ensure_ascii=False, sort_keys=True) + "\n")
    except HTTPException:
        _restore_text_file(runs_path, *runs_snapshot)
        _restore_text_file(deleted_runs_path, *deleted_runs_snapshot)
        raise
    except Exception as exc:
        _restore_text_file(runs_path, *runs_snapshot)
        _restore_text_file(deleted_runs_path, *deleted_runs_snapshot)
        raise HTTPException(status_code=500, detail=f"Experiment run could not be archived: {exc}") from exc
    return archived


def _find_marked_experiment_note_block(reports_dir: Path, run_id: str) -> Optional[Tuple[Path, str, int, int]]:
    log_path = reports_dir / EXPERIMENT_LOG_FILENAME
    if not log_path.exists():
        return None

    start_marker = f"<!-- experiment-run:{run_id} -->"
    end_marker = f"<!-- /experiment-run:{run_id} -->"

    try:
        content = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Experiment log could not be read: {exc}") from exc

    start = content.find(start_marker)
    if start == -1:
        return None

    end = content.find(end_marker, start + len(start_marker))
    if end == -1:
        raise HTTPException(status_code=409, detail=f"Experiment note block for {run_id} is incomplete.")
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
        raise HTTPException(status_code=500, detail=f"Experiment log could not be updated: {exc}") from exc
    return True


def _snapshot_text_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, ""
    try:
        return True, path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"{path.name} could not be read before mutation: {exc}") from exc


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


def build_research_status_payload() -> Dict[str, Any]:
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

    reports_dir = RESEARCH_STATUS_PATH.parent
    latest_training = _latest_training_payload(reports_dir)
    if latest_training is not None:
        payload["latest_training"] = latest_training
        _sync_qlib_provider_with_latest_training(payload, latest_training)
    cash_control_backtest = _cash_control_backtest_payload(reports_dir)
    if cash_control_backtest is not None:
        payload["cash_control_backtest"] = cash_control_backtest
    payload["qlib_predictions"] = _qlib_predictions_payload(reports_dir)
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
    return payload


@router.get("/status")
async def get_research_status() -> Dict[str, Any]:
    """Return the latest quant workflow status used by the research page."""
    return build_research_status_payload()


@router.post("/experiments/{run_id}/archive")
async def archive_experiment_run(run_id: str) -> Dict[str, Any]:
    """Archive one experiment run from the active notebook list."""
    with _experiment_file_guard(RESEARCH_STATUS_PATH.parent):
        removed = _archive_experiment_run(RESEARCH_STATUS_PATH.parent, run_id)
        return {
            "removed_run": removed,
            "experiment_notebook": _experiment_notebook_payload(RESEARCH_STATUS_PATH.parent),
        }


@router.delete("/experiments/{run_id}")
async def delete_experiment_run(run_id: str) -> Dict[str, Any]:
    """Backward-compatible archive route for older local frontends."""
    return await archive_experiment_run(run_id)


@router.delete("/experiments/{run_id}/permanent")
async def permanently_delete_experiment_run(
    run_id: str,
    confirm: str = Query("", description="Must be 'permanent' to confirm irreversible deletion."),
) -> Dict[str, Any]:
    """Permanently remove one experiment run and its marked Markdown note block."""
    if confirm != "permanent":
        raise HTTPException(status_code=400, detail="Permanent deletion requires confirm=permanent.")

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
            "experiment_notebook": _experiment_notebook_payload(reports_dir),
        }
