# -*- coding: utf-8 -*-
"""API contract tests for the real DSA signal strategy backtest."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app


def _client() -> tuple[tempfile.TemporaryDirectory, TestClient]:
    temp_dir = tempfile.TemporaryDirectory()
    static_dir = Path(temp_dir.name) / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir, TestClient(create_app(static_dir=static_dir))


def _report() -> dict:
    return {
        "status": "completed",
        "strategy_name": "dsa_history_signal",
        "symbols": ["600519"],
        "period": {"start_date": "2026-01-02", "end_date": "2026-01-04"},
        "strategy": {"asset_scope": "any_supported_instrument"},
        "data_quality": {"valid_signals": 2, "price_bars": 3},
        "metrics": {
            "initial_capital": 100000.0,
            "final_equity": 80000.0,
            "total_return_pct": -20.0,
            "annualized_return_pct": -99.9,
            "benchmark_return_pct": -10.0,
            "excess_return_pct": -10.0,
            "max_drawdown_pct": -20.0,
            "benchmark_max_drawdown_pct": -20.0,
            "trade_count": 2,
            "exposure_pct": 66.67,
            "total_cost": 0.0,
            "turnover_pct": 180.0,
            "signal_count": 2,
            "executed_signal_count": 2,
            "skipped_signal_count": 0,
            "trading_days": 3,
        },
        "benchmark": {"kind": "single_asset_buy_and_hold", "symbols": ["600519"]},
        "assumptions": {
            "execution": "next_common_trading_day_open",
            "initial_capital": 100000.0,
            "side_cost_pct": 0.0,
            "lot_size": 100,
        },
        "equity_curve": [
            {
                "date": "2026-01-02",
                "strategy_equity": 100000.0,
                "benchmark_equity": 100000.0,
                "strategy_return_pct": 0.0,
                "benchmark_return_pct": 0.0,
                "drawdown_pct": 0.0,
                "benchmark_drawdown_pct": 0.0,
                "position": "600519",
            }
        ],
        "trades": [
            {
                "source_id": "1",
                "signal_date": "2026-01-01",
                "execution_date": "2026-01-02",
                "symbol": "600519",
                "action": "buy",
                "shares": 10000,
                "price": 10.0,
                "notional": 100000.0,
                "fee": 0.0,
                "reason": "买入",
            }
        ],
        "warnings": [],
    }


def _trained_report() -> dict:
    return {
        "status": "completed",
        "generated_at": "2026-07-26T10:00:00+08:00",
        "run_id": "a" * 32,
        "model_id": "xgboost",
        "model_label": "XGBoost",
        "strategy_id": "low_turnover_weekly",
        "strategy_label": "每周低换手",
        "provider": "canonical",
        "period": {"start_date": "2024-01-02", "end_date": "2025-12-31"},
        "parameters": {
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "topk": 50,
            "cost_bps": 15.0,
        },
        "metrics": {
            "days": 485,
            "total_return": 0.5,
            "benchmark_total_return": 0.25,
            "annual_return": 0.24,
            "benchmark_annual_return": 0.12,
            "annual_excess_return": 0.12,
            "information_ratio": 1.2,
            "sharpe": 1.1,
            "max_drawdown": -0.16,
            "avg_turnover": 0.04,
            "avg_position": 1.0,
        },
        "equity_curve": [
            {
                "date": "2024-01-02",
                "strategy_cumulative_return": 0.01,
                "benchmark_cumulative_return": 0.005,
                "drawdown": 0.0,
                "benchmark_drawdown": 0.0,
                "daily_return": 0.01,
                "benchmark_return": 0.005,
                "turnover": 0.2,
                "position_pct": 1.0,
            }
        ],
        "warnings": [],
    }


def test_run_dsa_strategy_backtest_passes_explicit_execution_inputs() -> None:
    temp_dir, client = _client()
    try:
        with patch("api.v1.endpoints.backtest.BacktestService") as service_cls:
            service_cls.return_value.run_dsa_signal_strategy.return_value = _report()
            response = client.post(
                "/api/v1/backtest/strategy/dsa-signal",
                json={
                    "code": "600519",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-04",
                    "initial_capital": 100000,
                    "side_cost_pct": 0.0,
                    "lot_size": 100,
                },
            )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 200
    assert response.json()["metrics"]["total_return_pct"] == -20.0
    service_cls.return_value.run_dsa_signal_strategy.assert_called_once()
    kwargs = service_cls.return_value.run_dsa_signal_strategy.call_args.kwargs
    assert kwargs["code"] == "600519"
    assert kwargs["side_cost_pct"] == 0.0
    assert kwargs["lot_size"] == 100


def test_run_generic_strategy_backtest_passes_strategy_and_symbols() -> None:
    temp_dir, client = _client()
    try:
        with patch("api.v1.endpoints.backtest.BacktestService") as service_cls:
            service_cls.return_value.run_strategy.return_value = {
                **_report(),
                "strategy_name": "momentum_rotation",
                "symbols": ["SZ159915", "SH513100"],
            }
            response = client.post(
                "/api/v1/backtest/strategy/run",
                json={
                    "strategy_id": "momentum_rotation",
                    "symbols": ["159915", "513100"],
                    "lookback_days": 20,
                    "moving_average_days": 20,
                    "top_n": 1,
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                },
            )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 200
    kwargs = service_cls.return_value.run_strategy.call_args.kwargs
    assert kwargs["strategy_id"] == "momentum_rotation"
    assert kwargs["symbols"] == ["159915", "513100"]
    assert kwargs["lookback_days"] == 20
    assert kwargs["moving_average_days"] == 20
    assert kwargs["top_n"] == 1


def test_run_script_strategy_backtest_passes_source_and_execution_inputs() -> None:
    temp_dir, client = _client()
    try:
        with patch("api.v1.endpoints.backtest.BacktestService") as service_cls:
            service_cls.return_value.run_script_strategy.return_value = {
                **_report(),
                "strategy_name": "python_script",
                "symbols": ["SZ159915", "SH513100"],
                "logs": ["script started"],
                "script_check": {"status": "passed"},
            }
            response = client.post(
                "/api/v1/backtest/strategy/script",
                json={
                    "script": "def on_bar(context, data):\n    context.target_cash()",
                    "symbols": ["159915", "513100"],
                    "initial_capital": 100000,
                    "side_cost_pct": 0,
                    "lot_size": 100,
                },
            )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 200
    kwargs = service_cls.return_value.run_script_strategy.call_args.kwargs
    assert kwargs["symbols"] == ["159915", "513100"]
    assert "def on_bar" in kwargs["script"]


def test_run_trained_strategy_backtest_passes_the_human_selected_run_and_parameters() -> None:
    temp_dir, client = _client()
    try:
        with patch("api.v1.endpoints.backtest.TrainedStrategyBacktestService") as service_cls:
            service_cls.return_value.run.return_value = _trained_report()
            response = client.post(
                "/api/v1/backtest/strategy/trained",
                json={
                    "run_id": "a" * 32,
                    "model_id": "xgboost",
                    "strategy_id": "low_turnover_weekly",
                    "start_date": "2024-01-01",
                    "end_date": "2025-12-31",
                    "topk": 50,
                    "cost_bps": 15,
                },
            )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 200
    assert response.json()["metrics"]["annual_excess_return"] == 0.12
    service_cls.return_value.run.assert_called_once_with(
        run_id="a" * 32,
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        start_date="2024-01-01",
        end_date="2025-12-31",
        topk=50,
        cost_bps=15.0,
    )


def test_run_dsa_strategy_backtest_rejects_reversed_date_range() -> None:
    temp_dir, client = _client()
    try:
        response = client.post(
            "/api/v1/backtest/strategy/dsa-signal",
            json={
                "code": "600519",
                "start_date": "2026-01-04",
                "end_date": "2026-01-01",
            },
        )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_params"


def test_run_dsa_strategy_backtest_requires_an_instrument_code() -> None:
    temp_dir, client = _client()
    try:
        response = client.post(
            "/api/v1/backtest/strategy/dsa-signal",
            json={"code": ""},
        )
    finally:
        temp_dir.cleanup()

    assert response.status_code == 422
