# -*- coding: utf-8 -*-
"""API contract tests for strategy assets and durable backtest run history."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from src.repositories.strategy_assets_repo import StrategyAssetsRepository

VALID_SCRIPT = (
    "def initialize(context):\n"
    "    pass\n"
    "\n"
    "\n"
    "def on_bar(context, data):\n"
    "    context.target_cash()\n"
)
INVALID_SCRIPT = "def initialize(context):\n    import os\n    pass\n"


def _report() -> dict:
    return {
        "status": "completed",
        "strategy_name": "python_script",
        "symbols": ["SZ159915"],
        "period": {"start_date": "2026-01-02", "end_date": "2026-01-04"},
        "strategy": {
            "strategy_id": "python_script",
            "name": "DSA Python strategy",
            "script_sha256": hashlib.sha256(VALID_SCRIPT.encode("utf-8")).hexdigest(),
        },
        "data_quality": {
            "dataset_id": "cn_etf_daily",
            "snapshot_id": "snap-1",
            "valid_signals": 2,
            "price_bars": 3,
        },
        "metrics": {
            "initial_capital": 100000.0,
            "final_equity": 110000.0,
            "total_return_pct": 10.0,
            "annualized_return_pct": 12.0,
            "benchmark_return_pct": 6.0,
            "excess_return_pct": 4.0,
            "max_drawdown_pct": -5.0,
            "benchmark_max_drawdown_pct": -8.0,
            "trade_count": 2,
            "exposure_pct": 66.67,
            "total_cost": 0.0,
            "turnover_pct": 180.0,
            "signal_count": 2,
            "executed_signal_count": 2,
            "skipped_signal_count": 0,
            "trading_days": 3,
        },
        "benchmark": {"kind": "equal_weight_buy_and_hold", "symbols": ["SZ159915"]},
        "assumptions": {
            "execution": "next_common_trading_day_open",
            "initial_capital": 100000.0,
            "side_cost_pct": 0.0,
            "lot_size": 100,
        },
        "equity_curve": [],
        "trades": [],
        "warnings": [],
        "logs": [],
        "script_check": {"status": "passed", "api_version": "dsa-script-v1"},
    }


@pytest.fixture
def repo(tmp_path):
    return StrategyAssetsRepository(db_path=tmp_path / "strategy_assets.db")


@pytest.fixture
def client(repo, monkeypatch):
    monkeypatch.setattr(
        "api.v1.endpoints.backtest_assets.get_strategy_assets_repo", lambda: repo
    )
    monkeypatch.setattr(
        "api.v1.endpoints.backtest.get_strategy_assets_repo", lambda: repo
    )
    with tempfile.TemporaryDirectory() as temp_name:
        static_dir = Path(temp_name) / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        test_client = TestClient(create_app(static_dir=static_dir))
        test_client.headers.update({"Origin": "http://127.0.0.1:8010"})
        yield test_client


# === Strategy asset endpoints ===


class TestStrategyAssetEndpoints:
    def test_save_and_get_strategy(self, client):
        created = client.post(
            "/api/v1/backtest/strategies",
            json={"name": "Alpha strategy", "script": VALID_SCRIPT},
        )
        assert created.status_code == 201
        body = created.json()
        assert body["name"] == "Alpha strategy"
        assert body["script"] == VALID_SCRIPT
        assert len(body["script_sha256"]) == 64

        fetched = client.get(f"/api/v1/backtest/strategies/{body['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["script"] == VALID_SCRIPT

    def test_save_rejects_invalid_script(self, client):
        response = client.post(
            "/api/v1/backtest/strategies",
            json={"name": "Bad", "script": INVALID_SCRIPT},
        )
        assert response.status_code == 400

    def test_list_strategies_omits_script(self, client):
        client.post("/api/v1/backtest/strategies", json={"name": "A", "script": VALID_SCRIPT})
        client.post("/api/v1/backtest/strategies", json={"name": "B", "script": VALID_SCRIPT})
        response = client.get("/api/v1/backtest/strategies")
        assert response.status_code == 200
        items = response.json()
        assert len(items) == 2
        assert all("script" not in item for item in items)

    def test_update_strategy(self, client):
        created = client.post(
            "/api/v1/backtest/strategies", json={"name": "A", "script": VALID_SCRIPT}
        ).json()
        updated = client.patch(
            f"/api/v1/backtest/strategies/{created['id']}",
            json={"name": "A2"},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "A2"
        assert updated.json()["script"] == VALID_SCRIPT

    def test_archive_and_restore_keeps_detail_and_run_link(self, client, repo):
        created = client.post(
            "/api/v1/backtest/strategies", json={"name": "Historical", "script": VALID_SCRIPT}
        ).json()
        run = repo.create_run(
            strategy_id=created["id"], strategy_name=created["name"],
            script_sha256=created["script_sha256"], script=VALID_SCRIPT,
            symbols=["SZ159915"], params={}, data_quality={}, metrics={},
            report={"status": "completed"},
        )

        archived = client.patch(
            f"/api/v1/backtest/strategies/{created['id']}", json={"archived": True}
        )
        assert archived.status_code == 200
        assert archived.json()["archived"] is True
        assert client.get("/api/v1/backtest/strategies").json() == []
        history = client.get("/api/v1/backtest/strategies", params={"archived": True}).json()
        assert [item["id"] for item in history] == [created["id"]]
        assert "script" not in history[0]
        detail = client.get(f"/api/v1/backtest/strategies/{created['id']}").json()
        assert detail["script"] == VALID_SCRIPT
        assert detail["script_sha256"] == created["script_sha256"]
        saved_run = client.get(f"/api/v1/backtest/runs/{run['id']}").json()
        assert saved_run["strategy_id"] == created["id"]
        assert saved_run["script"] == VALID_SCRIPT

        restored = client.patch(
            f"/api/v1/backtest/strategies/{created['id']}", json={"archived": False}
        )
        assert restored.status_code == 200
        assert restored.json()["archived"] is False
        assert len(client.get("/api/v1/backtest/strategies").json()) == 1
        assert client.get("/api/v1/backtest/strategies", params={"archived": True}).json() == []

    def test_update_rejects_invalid_script(self, client):
        created = client.post(
            "/api/v1/backtest/strategies", json={"name": "A", "script": VALID_SCRIPT}
        ).json()
        response = client.patch(
            f"/api/v1/backtest/strategies/{created['id']}",
            json={"script": INVALID_SCRIPT},
        )
        assert response.status_code == 400

    def test_get_missing_strategy_404(self, client):
        assert client.get("/api/v1/backtest/strategies/missing").status_code == 404

    def test_delete_strategy(self, client):
        created = client.post(
            "/api/v1/backtest/strategies", json={"name": "A", "script": VALID_SCRIPT}
        ).json()
        assert client.delete(f"/api/v1/backtest/strategies/{created['id']}").status_code == 204
        assert client.get(f"/api/v1/backtest/strategies/{created['id']}").status_code == 404
        assert client.delete(f"/api/v1/backtest/strategies/{created['id']}").status_code == 404


# === Run persistence on the script backtest endpoint ===


class TestBacktestRunPersistence:
    def test_script_backtest_persists_run_and_returns_run_id(self, client):
        with patch(
            "src.services.backtest_service.BacktestService.run_script_strategy",
            return_value=_report(),
        ):
            response = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": VALID_SCRIPT,
                    "symbols": ["SZ159915"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"]

        runs = client.get("/api/v1/backtest/runs").json()
        assert len(runs) == 1
        assert runs[0]["id"] == body["run_id"]
        assert runs[0]["metrics"]["total_return_pct"] == 10.0
        assert "report" not in runs[0]

        detail = client.get(f"/api/v1/backtest/runs/{body['run_id']}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["report"]["status"] == "completed"
        assert detail_body["report"]["metrics"]["total_return_pct"] == 10.0
        assert detail_body["data_quality"]["snapshot_id"] == "snap-1"

    def test_script_backtest_links_saved_strategy(self, client):
        created = client.post(
            "/api/v1/backtest/strategies", json={"name": "A", "script": VALID_SCRIPT}
        ).json()
        with patch(
            "src.services.backtest_service.BacktestService.run_script_strategy",
            return_value=_report(),
        ):
            response = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": VALID_SCRIPT,
                    "symbols": ["SZ159915"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                    "strategy_asset_id": created["id"],
                },
            )
        assert response.status_code == 200
        run_id = response.json()["run_id"]

        runs = client.get("/api/v1/backtest/runs", params={"strategy_id": created["id"]}).json()
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert runs[0]["strategy_id"] == created["id"]

    def test_unknown_strategy_asset_id_degrades_to_ad_hoc_run(self, client):
        with patch(
            "src.services.backtest_service.BacktestService.run_script_strategy",
            return_value=_report(),
        ):
            response = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": VALID_SCRIPT,
                    "symbols": ["SZ159915"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                    "strategy_asset_id": "missing",
                },
            )
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        detail = client.get(f"/api/v1/backtest/runs/{run_id}").json()
        assert detail["strategy_id"] is None

    def test_persistence_failure_does_not_break_backtest(self, client):
        with patch(
            "src.services.backtest_service.BacktestService.run_script_strategy",
            return_value=_report(),
        ), patch.object(
            StrategyAssetsRepository, "create_run", side_effect=RuntimeError("disk full")
        ):
            response = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": VALID_SCRIPT,
                    "symbols": ["SZ159915"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                },
            )
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert response.json()["run_id"] is None

    def test_run_history_endpoints_404(self, client):
        assert client.get("/api/v1/backtest/runs/missing").status_code == 404
        assert client.get("/api/v1/backtest/runs").status_code == 200
        assert client.get("/api/v1/backtest/runs").json() == []

    def test_delete_run_endpoint(self, client):
        with patch(
            "src.services.backtest_service.BacktestService.run_script_strategy",
            return_value=_report(),
        ):
            created = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": VALID_SCRIPT,
                    "symbols": ["SZ159915"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                },
            ).json()
        run_id = created["run_id"]
        assert client.delete(f"/api/v1/backtest/runs/{run_id}").status_code == 204
        assert client.get(f"/api/v1/backtest/runs/{run_id}").status_code == 404
        assert client.delete(f"/api/v1/backtest/runs/{run_id}").status_code == 404
        assert client.get("/api/v1/backtest/runs").json() == []
