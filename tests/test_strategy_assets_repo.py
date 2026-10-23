# -*- coding: utf-8 -*-
"""Repository tests for strategy assets and durable backtest runs."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from src.repositories.strategy_assets_repo import StrategyAssetsRepository


@pytest.fixture
def repo(tmp_path):
    return StrategyAssetsRepository(db_path=tmp_path / "strategy_assets.db")


SCRIPT_A = "def initialize(context):\n    pass\n"
SCRIPT_B = "def initialize(context):\n    context.x = 1\n"


def _script_sha256(script: str) -> str:
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


def _make_strategy(repo, name="My strategy", script=SCRIPT_A):
    return repo.create_strategy(
        name=name,
        script=script,
        script_sha256=_script_sha256(script),
    )


def _make_run(repo, strategy_id=None, script=SCRIPT_A, script_sha256=None):
    script_sha256 = script_sha256 or _script_sha256(script)
    return repo.create_run(
        strategy_id=strategy_id,
        strategy_name="python_script",
        script_sha256=script_sha256,
        script=script,
        symbols=["SZ159915"],
        params={"initial_capital": 100000.0},
        data_quality={"dataset_id": "cn_etf_daily"},
        metrics={"total_return_pct": 10.5},
        report={"status": "completed", "metrics": {"total_return_pct": 10.5}},
        status="completed",
    )


class TestStrategyAssetCRUD:
    def test_create_and_get_strategy(self, repo):
        saved = _make_strategy(repo)
        assert saved["name"] == "My strategy"
        assert saved["script"] == SCRIPT_A
        assert saved["script_sha256"] == _script_sha256(SCRIPT_A)

        fetched = repo.get_strategy(saved["id"])
        assert fetched is not None
        assert fetched["id"] == saved["id"]
        assert fetched["script"] == SCRIPT_A

    def test_list_strategies_excludes_script_bodies(self, repo):
        _make_strategy(repo, name="A")
        _make_strategy(repo, name="B")
        items = repo.list_strategies()
        assert len(items) == 2
        for item in items:
            assert "script" not in item
            assert "script_sha256" in item

    def test_update_strategy_script_and_name(self, repo):
        saved = _make_strategy(repo)
        updated = repo.update_strategy(
            saved["id"], name="Renamed", script=SCRIPT_B,
            script_sha256=_script_sha256(SCRIPT_B)
        )
        assert updated is not None
        assert updated["name"] == "Renamed"
        assert updated["script"] == SCRIPT_B
        assert updated["script_sha256"] == _script_sha256(SCRIPT_B)

    def test_update_missing_strategy_returns_none(self, repo):
        assert repo.update_strategy("missing", name="x") is None

    def test_archive_and_restore_preserves_script_and_linked_run(self, repo):
        current = _make_strategy(repo, name="Current")
        historical = _make_strategy(repo, name="Historical")
        run = _make_run(repo, strategy_id=historical["id"])

        archived = repo.update_strategy(historical["id"], archived=True)

        assert archived["archived"] is True
        assert archived["script"] == historical["script"]
        assert archived["script_sha256"] == historical["script_sha256"]
        assert [item["id"] for item in repo.list_strategies()] == [current["id"]]
        assert [item["id"] for item in repo.list_strategies(archived=True)] == [historical["id"]]
        assert repo.get_run(run["id"]) == run

        restored = repo.update_strategy(historical["id"], archived=False)

        assert restored["archived"] is False
        assert len(repo.list_strategies()) == 2
        assert repo.list_strategies(archived=True) == []
        assert repo.get_run(run["id"]) == run

    def test_archive_filter_applies_before_limit(self, repo):
        current = _make_strategy(repo, name="Current")
        historical = _make_strategy(repo, name="Historical")
        repo.update_strategy(historical["id"], archived=True)

        assert [item["id"] for item in repo.list_strategies(limit=1)] == [current["id"]]

    def test_existing_database_gains_archive_state_without_rewriting_asset(self, tmp_path):
        path = tmp_path / "legacy.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE strategy_assets ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL, "
                "script TEXT NOT NULL, script_sha256 TEXT NOT NULL, api_version TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO strategy_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("legacy", "Historical", "Keep me", SCRIPT_A, _script_sha256(SCRIPT_A),
                 "dsa-script-v1", "created", "updated"),
            )

        migrated = StrategyAssetsRepository(db_path=path)
        asset = migrated.get_strategy("legacy")
        assert asset["archived"] is False
        assert asset["script"] == SCRIPT_A
        assert asset["description"] == "Keep me"
        assert asset["created_at"] == "created"
        assert asset["updated_at"] == "updated"

        migrated.update_strategy("legacy", archived=True)
        reopened = StrategyAssetsRepository(db_path=path)
        assert reopened.get_strategy("legacy")["archived"] is True
        assert reopened.list_strategies() == []

    def test_delete_strategy(self, repo):
        saved = _make_strategy(repo)
        assert repo.delete_strategy(saved["id"]) is True
        assert repo.get_strategy(saved["id"]) is None
        assert repo.delete_strategy(saved["id"]) is False

    def test_delete_strategy_unlinks_runs(self, repo):
        saved = _make_strategy(repo)
        run = _make_run(repo, strategy_id=saved["id"])
        assert repo.get_run(run["id"])["strategy_id"] == saved["id"]

        repo.delete_strategy(saved["id"])
        detached = repo.get_run(run["id"])
        assert detached is not None
        assert detached["strategy_id"] is None


class TestBacktestRuns:
    def test_create_run_rejects_script_sha256_mismatch(self, repo):
        with pytest.raises(ValueError, match="script_sha256 does not match script contents"):
            _make_run(repo, script=SCRIPT_A, script_sha256=_script_sha256(SCRIPT_B))

    def test_create_and_get_run_roundtrip(self, repo):
        run = _make_run(repo)
        fetched = repo.get_run(run["id"])
        assert fetched is not None
        assert fetched["id"] == run["id"]
        assert fetched["status"] == "completed"
        assert fetched["metrics"]["total_return_pct"] == 10.5
        assert fetched["symbols"] == ["SZ159915"]
        # Full report only in the detail payload
        assert fetched["report"]["status"] == "completed"

    def test_list_runs_excludes_report(self, repo):
        _make_run(repo)
        items = repo.list_runs()
        assert len(items) == 1
        assert "report" not in items[0]
        assert items[0]["metrics"]["total_return_pct"] == 10.5

    def test_list_runs_orders_newest_first(self, repo, monkeypatch):
        timestamps = iter((
            "2026-10-23T00:00:00+00:00",
            "2026-10-23T00:00:01+00:00",
        ))
        monkeypatch.setattr(
            "src.repositories.strategy_assets_repo._utc_now_iso",
            lambda: next(timestamps),
        )
        first = _make_run(repo)
        second = _make_run(repo)
        assert first["created_at"] < second["created_at"]
        items = repo.list_runs()
        assert [item["id"] for item in items] == [second["id"], first["id"]]

    def test_list_runs_filters_by_strategy_and_sha(self, repo):
        saved = _make_strategy(repo)
        _make_run(repo, strategy_id=saved["id"], script=SCRIPT_A)
        _make_run(repo, strategy_id=None, script=SCRIPT_B)

        by_strategy = repo.list_runs(strategy_id=saved["id"])
        assert len(by_strategy) == 1
        assert by_strategy[0]["strategy_id"] == saved["id"]

        by_sha = repo.list_runs(script_sha256=_script_sha256(SCRIPT_B))
        assert len(by_sha) == 1
        assert by_sha[0]["script_sha256"] == _script_sha256(SCRIPT_B)

    def test_get_missing_run_returns_none(self, repo):
        assert repo.get_run("missing") is None

    def test_delete_run(self, repo):
        run = _make_run(repo)
        assert repo.delete_run(run["id"]) is True
        assert repo.get_run(run["id"]) is None
        assert repo.delete_run(run["id"]) is False

    def test_list_runs_limit_and_offset(self, repo):
        for _ in range(5):
            _make_run(repo)
        page = repo.list_runs(limit=2, offset=1)
        assert len(page) == 2
        all_items = repo.list_runs(limit=100)
        assert page[0]["id"] == all_items[1]["id"]
