# -*- coding: utf-8 -*-
"""
Strategy asset repository.

SQLite-backed persistence for two first-class quant assets:

- ``strategy_assets``: named, versioned Python trading strategies authored in
  the browser (script + content hash + script API version).
- ``backtest_runs``: durable backtest executions. Every run keeps its data
  snapshot identity, parameters, metrics, and the full report, so two runs can
  be listed and compared later. Runs may reference a saved strategy or remain
  ad-hoc (``strategy_id`` NULL), in which case ``script_sha256`` still groups
  repeated executions of the same source.

The database file defaults to ``quant_workflow/reports/strategy_assets.db``
and can be overridden with ``DSA_STRATEGY_ASSETS_DB``. This repository is
independent of the quant research CRUD store (``quant_research.db``); the two
stores are merged only when the workbench asset model is finalised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "2026-10-18-strategy-archive"

_local = threading.local()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return str(uuid.uuid4())


class StrategyAssetsRepository:
    """Persistence for scripted strategies and their backtest runs."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            configured = os.environ.get("DSA_STRATEGY_ASSETS_DB")
            if configured:
                db_path = Path(configured).expanduser()
            else:
                from src.services.quant_workflow_paths import reports_root

                dsa_root = Path(__file__).resolve().parents[2]
                db_path = reports_root(dsa_root) / "strategy_assets.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._close_stale_connection()
        self._ensure_schema()

    def _close_stale_connection(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn is None:
            return
        try:
            row = conn.execute("PRAGMA database_list").fetchone()
            if row is not None and len(row) > 2 and row[2] != str(self._db_path):
                conn.close()
                _local.conn = None
        except Exception:
            _local.conn = None

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            _local.conn = conn
        return conn

    @contextmanager
    def _transaction(self):
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _ensure_schema(self) -> None:
        with self._transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS strategy_assets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    script TEXT NOT NULL,
                    script_sha256 TEXT NOT NULL,
                    api_version TEXT NOT NULL DEFAULT 'dsa-script-v1',
                    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_strategy_assets_sha
                    ON strategy_assets(script_sha256);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    strategy_name TEXT NOT NULL DEFAULT '',
                    script_sha256 TEXT NOT NULL,
                    script_json TEXT NOT NULL DEFAULT '',
                    symbols_json TEXT NOT NULL DEFAULT '[]',
                    params_json TEXT NOT NULL DEFAULT '{}',
                    data_quality_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    report_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (strategy_id) REFERENCES strategy_assets(id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_backtest_runs_created
                    ON backtest_runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_sha
                    ON backtest_runs(script_sha256);
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy
                    ON backtest_runs(strategy_id);
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, _utc_now_iso()),
            )
            # Migration for databases created before script freezing: add the
            # column so historical runs can be replayed after an upgrade.
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(backtest_runs)")
            }
            if "script_json" not in existing_columns:
                conn.execute(
                    "ALTER TABLE backtest_runs ADD COLUMN script_json TEXT NOT NULL DEFAULT ''"
                )
            asset_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(strategy_assets)")
            }
            if "archived" not in asset_columns:
                conn.execute(
                    "ALTER TABLE strategy_assets ADD COLUMN archived INTEGER NOT NULL "
                    "DEFAULT 0 CHECK (archived IN (0, 1))"
                )

    # === Strategy asset CRUD ===

    def create_strategy(
        self,
        *,
        name: str,
        script: str,
        script_sha256: str,
        description: str = "",
        api_version: str = "dsa-script-v1",
    ) -> Dict[str, Any]:
        actual_script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        if script_sha256 != actual_script_sha256:
            raise ValueError("script_sha256 does not match script contents")
        strategy_id = _generate_id()
        now = _utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO strategy_assets
                    (id, name, description, script, script_sha256,
                     api_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (strategy_id, name, description, script, script_sha256, api_version, now, now),
            )
        return self.get_strategy(strategy_id)  # type: ignore[return-value]

    def list_strategies(
        self, limit: int = 100, *, archived: bool = False
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, name, description, script_sha256, api_version, archived,
                       created_at, updated_at
                FROM strategy_assets
                WHERE archived = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(archived), limit),
            ).fetchall()
        return [_strategy_summary_row(row) for row in rows]

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_assets WHERE id = ?", (strategy_id,)
            ).fetchone()
        return _strategy_detail_row(row) if row is not None else None

    def update_strategy(
        self,
        strategy_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        script: Optional[str] = None,
        script_sha256: Optional[str] = None,
        api_version: Optional[str] = None,
        archived: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = self.get_strategy(strategy_id)
        if existing is None:
            return None
        if script is not None:
            actual_script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
            if script_sha256 is None:
                script_sha256 = actual_script_sha256
            elif script_sha256 != actual_script_sha256:
                raise ValueError("script_sha256 does not match script contents")
        elif script_sha256 is not None and script_sha256 != existing["script_sha256"]:
            raise ValueError("script_sha256 does not match existing script contents")
        updates: List[str] = []
        values: List[Any] = []
        if name is not None:
            updates.append("name = ?")
            values.append(name)
        if description is not None:
            updates.append("description = ?")
            values.append(description)
        if script is not None:
            updates.append("script = ?")
            values.append(script)
        if script_sha256 is not None:
            updates.append("script_sha256 = ?")
            values.append(script_sha256)
        if api_version is not None:
            updates.append("api_version = ?")
            values.append(api_version)
        if archived is not None:
            updates.append("archived = ?")
            values.append(int(archived))
        if not updates:
            return existing
        updates.append("updated_at = ?")
        values.append(_utc_now_iso())
        values.append(strategy_id)
        with self._transaction() as conn:
            conn.execute(
                f"UPDATE strategy_assets SET {', '.join(updates)} WHERE id = ?",
                tuple(values),
            )
        return self.get_strategy(strategy_id)

    def delete_strategy(self, strategy_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM strategy_assets WHERE id = ?", (strategy_id,))
        return cursor.rowcount > 0

    # === Backtest run persistence ===

    def create_run(
        self,
        *,
        strategy_id: Optional[str],
        strategy_name: str,
        script_sha256: str,
        script: str = "",
        symbols: List[str],
        params: Dict[str, Any],
        data_quality: Dict[str, Any],
        metrics: Dict[str, Any],
        report: Dict[str, Any],
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        actual_script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        if script_sha256 != actual_script_sha256:
            raise ValueError("script_sha256 does not match script contents")

        run_id = _generate_id()
        created_at = _utc_now_iso()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO backtest_runs
                    (id, strategy_id, strategy_name, script_sha256, script_json,
                     symbols_json, params_json, data_quality_json,
                     metrics_json, report_json, status, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy_id,
                    strategy_name,
                    script_sha256,
                    script,
                    json.dumps(symbols, ensure_ascii=False),
                    json.dumps(params, ensure_ascii=False, default=str),
                    json.dumps(data_quality, ensure_ascii=False, default=str),
                    json.dumps(metrics, ensure_ascii=False, default=str),
                    json.dumps(report, ensure_ascii=False, default=str),
                    status,
                    error_message,
                    created_at,
                ),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def list_runs(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        strategy_id: Optional[str] = None,
        script_sha256: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        clauses: List[str] = []
        values: List[Any] = []
        if strategy_id is not None:
            clauses.append("strategy_id = ?")
            values.append(strategy_id)
        if script_sha256 is not None:
            clauses.append("script_sha256 = ?")
            values.append(script_sha256)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        # Deliberately exclude report_json from list queries: it can be large.
        with self._transaction() as conn:
            rows = conn.execute(
                f"""
                SELECT id, strategy_id, strategy_name, script_sha256,
                       symbols_json, params_json, data_quality_json,
                       metrics_json, status, error_message, created_at
                FROM backtest_runs
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (*values, limit, offset),
            ).fetchall()
        return [_run_summary_row(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._transaction() as conn:
            row = conn.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_detail_row(row) if row is not None else None

    def delete_run(self, run_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0


# === Shared process-level default repository ===

_default_repo: Optional[StrategyAssetsRepository] = None


def get_strategy_assets_repo() -> StrategyAssetsRepository:
    """Return the process-level default repository (lazy singleton)."""
    global _default_repo
    if _default_repo is None:
        _default_repo = StrategyAssetsRepository()
    return _default_repo


# === Row converters ===


def _json_loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _strategy_summary_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "script_sha256": row["script_sha256"],
        "api_version": row["api_version"],
        "archived": bool(row["archived"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _strategy_detail_row(row: sqlite3.Row) -> Dict[str, Any]:
    payload = _strategy_summary_row(row)
    payload["script"] = row["script"]
    return payload


def _run_summary_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "strategy_id": row["strategy_id"],
        "strategy_name": row["strategy_name"],
        "script_sha256": row["script_sha256"],
        "symbols": _json_loads(row["symbols_json"], []),
        "params": _json_loads(row["params_json"], {}),
        "data_quality": _json_loads(row["data_quality_json"], {}),
        "metrics": _json_loads(row["metrics_json"], {}),
        "status": row["status"],
        "error_message": row["error_message"],
        "created_at": row["created_at"],
    }


def _run_detail_row(row: sqlite3.Row) -> Dict[str, Any]:
    payload = _run_summary_row(row)
    payload["report"] = _json_loads(row["report_json"], {})
    payload["script"] = row["script_json"]
    return payload
