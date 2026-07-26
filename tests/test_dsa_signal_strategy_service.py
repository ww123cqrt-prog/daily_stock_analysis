# -*- coding: utf-8 -*-
"""Integration tests for the DSA-history signal strategy adapter."""

import json
from datetime import date, datetime

import pytest

from src.services.backtest_service import BacktestService
from src.storage import AnalysisHistory, DatabaseManager, StockDaily


@pytest.fixture()
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "strategy.db"))
    DatabaseManager.reset_instance()
    db = DatabaseManager.get_instance()
    yield db
    DatabaseManager.reset_instance()


def _analysis(
    *,
    code: str,
    signal_date: date,
    created_at: datetime,
    advice: str,
    query_id: str,
) -> AnalysisHistory:
    return AnalysisHistory(
        query_id=query_id,
        code=code,
        name="测试标的",
        report_type="simple",
        operation_advice=advice,
        trend_prediction="测试",
        created_at=created_at,
        context_snapshot=json.dumps(
            {"enhanced_context": {"date": signal_date.isoformat()}}
        ),
    )


def test_dsa_adapter_builds_continuous_next_open_strategy_and_skips_backfill(database) -> None:
    with database.get_session() as session:
        session.add_all(
            [
                _analysis(
                    code="600519",
                    signal_date=date(2026, 1, 1),
                    created_at=datetime(2026, 1, 1, 15, 30),
                    advice="买入",
                    query_id="buy",
                ),
                _analysis(
                    code="600519",
                    signal_date=date(2026, 1, 3),
                    created_at=datetime(2026, 1, 3, 15, 30),
                    advice="观望",
                    query_id="cash",
                ),
                _analysis(
                    code="600519",
                    signal_date=date(2026, 1, 2),
                    created_at=datetime(2026, 1, 5, 12, 0),
                    advice="卖出",
                    query_id="backfilled",
                ),
            ]
        )
        session.add_all(
            [
                StockDaily(code="600519", date=date(2026, 1, 1), open=10, close=10),
                StockDaily(code="600519", date=date(2026, 1, 2), open=10, close=10),
                StockDaily(code="600519", date=date(2026, 1, 3), open=9, close=8),
                StockDaily(code="600519", date=date(2026, 1, 4), open=8, close=9),
            ]
        )
        session.commit()

    report = BacktestService(database).run_dsa_signal_strategy(
        code="600519",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 4),
        initial_capital=100_000,
        side_cost_pct=0,
        lot_size=100,
    )

    assert report["status"] == "completed"
    assert [trade["action"] for trade in report["trades"]] == ["buy", "sell"]
    assert report["metrics"]["total_return_pct"] == pytest.approx(-20.0)
    assert report["strategy"]["asset_scope"] == "any_supported_instrument"
    assert report["data_quality"]["analysis_records"] == 3
    assert report["data_quality"]["valid_signals"] == 2
    assert report["data_quality"]["backfilled_signals_skipped"] == 1


def test_dsa_adapter_reports_missing_history_without_faking_a_backtest(database) -> None:
    report = BacktestService(database).run_dsa_signal_strategy(
        code="159915",
        initial_capital=100_000,
        side_cost_pct=0,
        lot_size=100,
    )

    assert report["status"] == "no_valid_signals"
    assert report["equity_curve"] == []
    assert report["trades"] == []
    assert report["data_quality"]["valid_signals"] == 0
