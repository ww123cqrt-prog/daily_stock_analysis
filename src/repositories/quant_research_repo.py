# -*- coding: utf-8 -*-
"""
Quant Research domain repository.

Provides SQLite-backed persistence for Research, Experiment, and Run entities.
Uses a dedicated database file to avoid invasive changes to the main storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

AlgorithmExistsFn = Callable[[str], bool]

logger = logging.getLogger(__name__)

# Schema version for migrations
_SCHEMA_VERSION = "2026-08-12-initial"

# Thread-local storage for connections
_local = threading.local()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    return str(uuid.uuid4())


def _compute_research_definition_hash(
    experiment_type: str,
    method_kind: str,
    algorithm_id: Optional[str],
    protocol: Dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "experiment_type": experiment_type,
            "method_kind": method_kind,
            "algorithm_id": algorithm_id,
            "protocol": protocol,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_experiment_definition(
    method_kind: str,
    algorithm_id: Optional[str],
    algorithm_exists: Optional[AlgorithmExistsFn] = None,
) -> None:
    """Validate method_kind/algorithm_id combination. Raises ValueError on invalid."""
    if method_kind == "algorithm":
        if not algorithm_id:
            raise ValueError("algorithm_id is required when method_kind is 'algorithm'")
        if algorithm_exists and not algorithm_exists(algorithm_id):
            raise ValueError(f"Unknown algorithm: {algorithm_id}")
    elif method_kind == "rule":
        if algorithm_id:
            raise ValueError("algorithm_id must be empty when method_kind is 'rule'")


_FROZEN_STATUSES = {"submitted", "running", "completed", "failed", "cancelled"}
_IMMUTABLE_FIELDS = {"experiment_type", "method_kind", "algorithm_id", "protocol"}


class QuantResearchRepository:
    """
    SQLite-backed repository for quant research domain.

    Uses a dedicated database file (default: quant_workflow/reports/quant_research.db)
    to avoid invasive changes to the main DSA storage.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from src.services.quant_workflow_paths import reports_root
            dsa_root = Path(__file__).resolve().parent.parent.parent
            db_path = reports_root(dsa_root) / "quant_research.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._close_stale_connection()
        self._ensure_schema()

    def _close_stale_connection(self) -> None:
        conn = getattr(_local, "conn", None)
        if conn is not None:
            try:
                conn_path = conn.execute("PRAGMA database_list").fetchone()
                if conn_path and len(conn_path) > 2 and conn_path[2] != str(self._db_path):
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
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _ensure_schema(self) -> None:
        with self._transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    research_type TEXT NOT NULL DEFAULT 'stock_selection',
                    config_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment (
                    id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    experiment_type TEXT NOT NULL DEFAULT 'stock_selection',
                    method_kind TEXT NOT NULL DEFAULT 'algorithm',
                    algorithm_id TEXT,
                    protocol_json TEXT,
                    protocol_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (research_id) REFERENCES research(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS run (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    run_label TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    metrics_json TEXT,
                    artifacts_json TEXT,
                    error_message TEXT,
                    is_legacy INTEGER NOT NULL DEFAULT 0,
                    mlflow_run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiment(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_experiment_research
                    ON experiment(research_id);
                CREATE INDEX IF NOT EXISTS idx_run_experiment
                    ON run(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_run_status
                    ON run(status);
            """)
            # Record schema version
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, _utc_now_iso()),
            )
            try:
                conn.execute("SELECT protocol_hash FROM experiment LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE experiment ADD COLUMN protocol_hash TEXT")

    # === Research CRUD ===

    def create_research(
        self,
        name: str,
        description: str = "",
        research_type: str = "stock_selection",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        research_id = _generate_id()
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO research (id, name, description, status, research_type, config_json, created_at, updated_at)
                   VALUES (?, ?, ?, 'draft', ?, ?, ?, ?)""",
                (research_id, name, description, research_type, json.dumps(config or {}), now, now),
            )
        return self.get_research(research_id)

    def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM research WHERE id = ?", (research_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_research(row)

    def list_research(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM research ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_research(r) for r in rows]

    def update_research(self, research_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        allowed = {"name", "description", "status", "research_type", "config"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_research(research_id)
        set_clauses = []
        params = []
        for key, value in updates.items():
            if key == "config":
                set_clauses.append("config_json = ?")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        set_clauses.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(research_id)
        with self._transaction() as conn:
            conn.execute(
                f"UPDATE research SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
        return self.get_research(research_id)

    def delete_research(self, research_id: str) -> None:
        with self._transaction() as conn:
            # Serialize the protection check with deletion and concurrent runs.
            conn.execute("BEGIN IMMEDIATE")
            research = conn.execute("SELECT name FROM research WHERE id = ?", (research_id,)).fetchone()
            if research is None:
                raise ValueError(f"Research not found: {research_id}")
            protected = conn.execute(
                """SELECT e.name, EXISTS(SELECT 1 FROM run r WHERE r.experiment_id = e.id) AS has_runs
                   FROM experiment e WHERE e.research_id = ?
                   AND (e.status != 'draft' OR EXISTS(SELECT 1 FROM run r WHERE r.experiment_id = e.id))
                   LIMIT 1""", (research_id,),
            ).fetchone()
            if protected is not None:
                reason = "has run(s)" if protected["has_runs"] else "is frozen"
                raise ValueError(f"Cannot delete research '{research['name']}': experiment '{protected['name']}' {reason}")
            conn.execute("DELETE FROM research WHERE id = ?", (research_id,))

    def _row_to_research(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "status": row["status"],
            "research_type": row["research_type"],
            "config": json.loads(row["config_json"]) if row["config_json"] else {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # === Experiment CRUD ===

    def create_experiment(
        self,
        research_id: str,
        name: str,
        description: str = "",
        experiment_type: str = "stock_selection",
        method_kind: str = "algorithm",
        algorithm_id: Optional[str] = None,
        protocol: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        experiment_id = _generate_id()
        protocol_data = protocol or {}
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO experiment
                   (id, research_id, name, description, experiment_type, method_kind, algorithm_id, protocol_json, protocol_hash, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'draft', ?, ?)""",
                (experiment_id, research_id, name, description, experiment_type, method_kind,
                 algorithm_id, json.dumps(protocol_data), now, now),
            )
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM experiment WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def get_experiment_checked(self, experiment_id: str, algorithm_exists: Optional[AlgorithmExistsFn] = None) -> Optional[Dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is not None:
            self._check_integrity_or_raise(experiment, algorithm_exists)
        return experiment

    def list_experiments(self, research_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM experiment WHERE research_id = ? ORDER BY created_at DESC LIMIT ?",
            (research_id, limit),
        ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def list_experiments_checked(self, research_id: str, algorithm_exists: Optional[AlgorithmExistsFn] = None, limit: int = 50) -> List[Dict[str, Any]]:
        experiments = self.list_experiments(research_id, limit)
        for exp in experiments:
            self._check_integrity_or_raise(exp, algorithm_exists)
        return experiments

    def update_experiment(self, experiment_id: str, algorithm_exists: Optional[AlgorithmExistsFn] = None, **kwargs) -> Optional[Dict[str, Any]]:
        current = self.get_experiment(experiment_id)
        if current is None:
            return None
        is_frozen = current["status"] in _FROZEN_STATUSES or current.get("protocol_hash") is not None
        if is_frozen:
            changed_immutable = _IMMUTABLE_FIELDS & set(kwargs.keys())
            for field in changed_immutable:
                if kwargs[field] != current[field]:
                    raise ValueError(f"Cannot modify {field} of a frozen experiment")
            new_status = kwargs.get("status")
            if new_status == "draft" and current["status"] in _FROZEN_STATUSES:
                raise ValueError("Cannot revert a frozen experiment back to draft")
        allowed = {"name", "description", "experiment_type", "method_kind", "algorithm_id", "protocol", "status"}
        updates = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "algorithm_id":
                updates[k] = v
            elif v is not None:
                updates[k] = v
        if not updates:
            return current
        new_status = updates.get("status", current["status"])
        new_experiment_type = updates.get("experiment_type", current["experiment_type"])
        new_method_kind = updates.get("method_kind", current["method_kind"])
        new_algorithm_id = updates.get("algorithm_id", current["algorithm_id"])
        new_protocol = updates.get("protocol", current["protocol"])
        changing_method_or_algo = "method_kind" in updates or "algorithm_id" in updates
        freezing = new_status in _FROZEN_STATUSES and current["status"] not in _FROZEN_STATUSES
        if changing_method_or_algo or freezing:
            validate_experiment_definition(new_method_kind, new_algorithm_id, algorithm_exists)
        protocol_hash = current.get("protocol_hash")
        if new_status in _FROZEN_STATUSES and protocol_hash is None:
            protocol_hash = _compute_research_definition_hash(
                new_experiment_type, new_method_kind, new_algorithm_id, new_protocol,
            )
        set_clauses = []
        params = []
        for key, value in updates.items():
            if key == "protocol":
                set_clauses.append("protocol_json = ?")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        set_clauses.append("protocol_hash = ?")
        params.append(protocol_hash)
        set_clauses.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(experiment_id)
        with self._transaction() as conn:
            conn.execute(
                f"UPDATE experiment SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
        return self.get_experiment(experiment_id)

    def _row_to_experiment(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "research_id": row["research_id"],
            "name": row["name"],
            "description": row["description"],
            "experiment_type": row["experiment_type"],
            "method_kind": row["method_kind"],
            "algorithm_id": row["algorithm_id"],
            "protocol": json.loads(row["protocol_json"]) if row["protocol_json"] else {},
            "protocol_hash": row["protocol_hash"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete_experiment(self, experiment_id: str) -> None:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_id}")
        runs = self.list_runs(experiment_id)
        if runs:
            raise ValueError(f"Cannot delete experiment '{experiment['name']}': it has {len(runs)} run(s)")
        if experiment["status"] != "draft":
            raise ValueError(f"Cannot delete frozen experiment '{experiment['name']}': only draft experiments can be deleted")
        with self._transaction() as conn:
            conn.execute("DELETE FROM experiment WHERE id = ?", (experiment_id,))

    # === Run CRUD ===

    def verify_experiment_integrity(self, experiment_id: str) -> bool:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_id}")
        stored_hash = experiment.get("protocol_hash")
        is_frozen = experiment["status"] in _FROZEN_STATUSES
        if is_frozen and stored_hash is None:
            return False
        if stored_hash is None:
            return True
        expected = _compute_research_definition_hash(
            experiment["experiment_type"],
            experiment["method_kind"],
            experiment["algorithm_id"],
            experiment["protocol"],
        )
        return stored_hash == expected

    def _check_integrity_or_raise(self, experiment: Dict[str, Any], algorithm_exists: Optional[AlgorithmExistsFn] = None) -> None:
        stored_hash = experiment.get("protocol_hash")
        is_frozen = experiment["status"] in _FROZEN_STATUSES
        if is_frozen:
            validate_experiment_definition(
                experiment["method_kind"],
                experiment["algorithm_id"],
                algorithm_exists,
            )
            if stored_hash is None:
                raise ValueError(f"Experiment {experiment['id']} is frozen but has no research definition hash")
        if stored_hash is None:
            return
        expected = _compute_research_definition_hash(
            experiment["experiment_type"],
            experiment["method_kind"],
            experiment["algorithm_id"],
            experiment["protocol"],
        )
        if stored_hash != expected:
            raise ValueError(f"Experiment {experiment['id']} integrity check failed: stored hash does not match research definition")

    def create_run(
        self,
        experiment_id: str,
        run_label: str = "",
        is_legacy: bool = False,
        mlflow_run_id: Optional[str] = None,
        algorithm_exists: Optional[AlgorithmExistsFn] = None,
    ) -> Dict[str, Any]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_id}")
        if experiment["status"] not in _FROZEN_STATUSES:
            raise ValueError("Run can only be created from a frozen experiment (submitted/running/completed/failed/cancelled)")
        if not experiment.get("protocol_hash"):
            raise ValueError("Run requires a frozen experiment with a valid research definition hash")
        validate_experiment_definition(experiment["method_kind"], experiment["algorithm_id"], algorithm_exists)
        if not self.verify_experiment_integrity(experiment_id):
            raise ValueError("Experiment integrity check failed: stored hash does not match research definition")
        now = _utc_now_iso()
        run_id = _generate_id()
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO run
                   (id, experiment_id, run_label, status, is_legacy, mlflow_run_id, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)""",
                (run_id, experiment_id, run_label, 1 if is_legacy else 0, mlflow_run_id, now, now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(self, experiment_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM run WHERE experiment_id = ? ORDER BY created_at DESC LIMIT ?",
            (experiment_id, limit),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def update_run(self, run_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        allowed = {"run_label", "status", "started_at", "completed_at", "metrics", "artifacts", "error_message", "mlflow_run_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        if not updates:
            return self.get_run(run_id)
        set_clauses = []
        params = []
        for key, value in updates.items():
            if key in ("metrics", "artifacts"):
                set_clauses.append(f"{key}_json = ?")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        set_clauses.append("updated_at = ?")
        params.append(_utc_now_iso())
        params.append(run_id)
        with self._transaction() as conn:
            conn.execute(
                f"UPDATE run SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
        return self.get_run(run_id)

    def _row_to_run(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "experiment_id": row["experiment_id"],
            "run_label": row["run_label"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            "artifacts": json.loads(row["artifacts_json"]) if row["artifacts_json"] else [],
            "error_message": row["error_message"],
            "is_legacy": bool(row["is_legacy"]),
            "mlflow_run_id": row["mlflow_run_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # === Aggregate queries ===

    def get_research_with_experiments(self, research_id: str, algorithm_exists: Optional[AlgorithmExistsFn] = None) -> Optional[Dict[str, Any]]:
        research = self.get_research(research_id)
        if research is None:
            return None
        research["experiments"] = self.list_experiments(research_id)
        for exp in research["experiments"]:
            self._check_integrity_or_raise(exp, algorithm_exists)
            exp["runs"] = self.list_runs(exp["id"])
        return research

    def get_overview(self) -> Dict[str, Any]:
        conn = self._get_connection()
        research_count = conn.execute("SELECT COUNT(*) FROM research").fetchone()[0]
        experiment_count = conn.execute("SELECT COUNT(*) FROM experiment").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM run").fetchone()[0]
        recent_research = self.list_research(limit=5)
        return {
            "total_research": research_count,
            "total_experiments": experiment_count,
            "total_runs": run_count,
            "recent_research": recent_research,
        }
