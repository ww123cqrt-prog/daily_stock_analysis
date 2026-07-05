# -*- coding: utf-8 -*-
"""Research workspace status endpoint."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DSA_ROOT = _env_path("DSA_ROOT", Path(__file__).resolve().parents[3])
PROJECTS_ROOT = _env_path("WORKING_PROJECTS_ROOT", DSA_ROOT.parent)
QUANT_WORKFLOW_ROOT = _env_path("QUANT_WORKFLOW_ROOT", PROJECTS_ROOT / "quant_workflow")
QUANT_WORKFLOW_REPORTS_ROOT = _env_path("QUANT_WORKFLOW_REPORTS_ROOT", QUANT_WORKFLOW_ROOT / "reports")
RESEARCH_STATUS_PATH = QUANT_WORKFLOW_REPORTS_ROOT / "research_status_latest.json"
EXPERIMENT_LOG_FILENAME = "experiment_log.md"
EXPERIMENT_RUNS_FILENAME = "experiment_runs.jsonl"
EXPERIMENT_DELETED_RUNS_FILENAME = "experiment_runs_deleted.jsonl"
LATEST_TRAINING_RESULT_FILENAME = "latest_training_result.json"
CASH_CONTROL_BACKTEST_FILENAME = "cash_control_backtest_latest.json"
EXPERIMENT_FILE_LOCK = threading.RLock()
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


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _latest_training_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_object(reports_dir / LATEST_TRAINING_RESULT_FILENAME)


def _cash_control_backtest_payload(reports_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json_object(reports_dir / CASH_CONTROL_BACKTEST_FILENAME)


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
        "log_path": str(log_path),
        "runs_path": str(runs_path),
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


def _missing_status_payload() -> Dict[str, Any]:
    reports_dir = RESEARCH_STATUS_PATH.parent
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
        "experiment_notebook": _experiment_notebook_payload(reports_dir),
        "external_tools": _external_tools_payload(),
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
    removed = _remove_active_experiment_run(reports_dir, run_id)
    archived = {
        **removed,
        "deleted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deleted_from": EXPERIMENT_RUNS_FILENAME,
    }
    try:
        with deleted_runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(archived, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Experiment run could not be archived: {exc}") from exc
    return archived


def _find_marked_experiment_note_block(reports_dir: Path, run_id: str) -> Optional[tuple[Path, str, int, int]]:
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
        log_path.write_text(updated, encoding="utf-8")
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


def _restore_text_file(path: Path, existed: bool, content: str) -> None:
    try:
        if existed:
            path.write_text(content, encoding="utf-8")
        elif path.exists():
            path.unlink()
    except OSError as exc:
        # Preserve the original failure for the caller; this is best-effort rollback.
        logger.warning("Experiment rollback restore failed for %s: %s", path, exc)


@router.get("/status")
async def get_research_status() -> Dict[str, Any]:
    """Return the latest quant workflow status used by the research page."""
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
    cash_control_backtest = _cash_control_backtest_payload(reports_dir)
    if cash_control_backtest is not None:
        payload["cash_control_backtest"] = cash_control_backtest
    payload["experiment_notebook"] = _experiment_notebook_payload(reports_dir)
    payload["external_tools"] = _external_tools_payload()
    return payload


@router.post("/experiments/{run_id}/archive")
async def archive_experiment_run(run_id: str) -> Dict[str, Any]:
    """Archive one experiment run from the active notebook list."""
    with EXPERIMENT_FILE_LOCK:
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
async def permanently_delete_experiment_run(run_id: str) -> Dict[str, Any]:
    """Permanently remove one experiment run and its marked Markdown note block."""
    with EXPERIMENT_FILE_LOCK:
        reports_dir = RESEARCH_STATUS_PATH.parent
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
