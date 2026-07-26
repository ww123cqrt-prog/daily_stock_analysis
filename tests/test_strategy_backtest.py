# -*- coding: utf-8 -*-
"""Tests for the asset-agnostic target-allocation backtest engine."""

from datetime import date

import pytest

from src.core.strategy_backtest import (
    PriceBar,
    StrategyBacktestConfig,
    TargetAllocation,
    run_portfolio_backtest,
)


def _bar(
    day: int,
    *,
    open_price: float,
    close: float,
    symbol: str = "600519",
) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        date=date(2026, 1, day),
        open=open_price,
        close=close,
    )


def _allocation(
    day: int,
    weights: dict[str, float],
    source_id: int,
) -> TargetAllocation:
    return TargetAllocation(
        source_id=str(source_id),
        signal_date=date(2026, 1, day),
        target_weights=weights,
        reason="测试信号",
    )


def test_executes_each_signal_at_the_next_open_and_keeps_position_between_signals() -> None:
    report = run_portfolio_backtest(
        strategy_name="dsa_signal",
        allocations=[
            _allocation(1, {"600519": 1.0}, 1),
            _allocation(3, {}, 2),
        ],
        bars=[
            _bar(1, open_price=10.0, close=10.0),
            _bar(2, open_price=10.0, close=10.0),
            _bar(3, open_price=9.0, close=8.0),
            _bar(4, open_price=8.0, close=9.0),
        ],
        config=StrategyBacktestConfig(
            initial_capital=100_000.0,
            side_cost_pct=0.0,
            lot_size=100,
        ),
    )

    assert report["status"] == "completed"
    assert [trade["execution_date"] for trade in report["trades"]] == [
        "2026-01-02",
        "2026-01-04",
    ]
    assert [trade["action"] for trade in report["trades"]] == ["buy", "sell"]
    assert [point["position"] for point in report["equity_curve"]] == [
        "600519",
        "600519",
        "CASH",
    ]
    assert report["metrics"]["final_equity"] == pytest.approx(80_000.0)
    assert report["metrics"]["total_return_pct"] == pytest.approx(-20.0)
    assert report["metrics"]["max_drawdown_pct"] == pytest.approx(-20.0)
    assert report["metrics"]["benchmark_return_pct"] == pytest.approx(-10.0)


def test_deducts_the_user_configured_cost_without_hidden_default() -> None:
    report = run_portfolio_backtest(
        strategy_name="dsa_signal",
        allocations=[_allocation(1, {"600519": 1.0}, 1)],
        bars=[
            _bar(1, open_price=10.0, close=10.0),
            _bar(2, open_price=10.0, close=10.0),
        ],
        config=StrategyBacktestConfig(
            initial_capital=100_000.0,
            side_cost_pct=0.1,
            lot_size=100,
        ),
    )

    assert report["trades"][0]["shares"] == 9_900
    assert report["trades"][0]["fee"] == pytest.approx(99.0)
    assert report["metrics"]["final_equity"] == pytest.approx(99_901.0)
    assert report["metrics"]["total_cost"] == pytest.approx(99.0)
    assert report["assumptions"]["side_cost_pct"] == pytest.approx(0.1)


def test_repeated_same_target_does_not_create_fake_turnover() -> None:
    report = run_portfolio_backtest(
        strategy_name="dsa_signal",
        allocations=[
            _allocation(1, {"600519": 1.0}, 1),
            _allocation(2, {"600519": 1.0}, 2),
        ],
        bars=[
            _bar(1, open_price=10.0, close=10.0),
            _bar(2, open_price=10.0, close=10.0),
            _bar(3, open_price=11.0, close=11.0),
        ],
        config=StrategyBacktestConfig(initial_capital=100_000.0),
    )

    assert len(report["trades"]) == 1
    assert report["metrics"]["trade_count"] == 1
    assert report["metrics"]["exposure_pct"] == pytest.approx(100.0)


def test_partial_target_change_trades_only_the_position_delta() -> None:
    symbols = ("600519", "159915")
    bars = [
        _bar(day, open_price=price, close=price, symbol=symbol)
        for day in (1, 2, 3)
        for symbol, price in zip(symbols, (10.0, 20.0), strict=True)
    ]

    report = run_portfolio_backtest(
        strategy_name="mixed_instrument_allocation",
        allocations=[
            _allocation(1, {"600519": 1.0}, 1),
            _allocation(2, {"600519": 0.5, "159915": 0.5}, 2),
        ],
        bars=bars,
        config=StrategyBacktestConfig(
            initial_capital=100_000.0,
            side_cost_pct=0.0,
            lot_size=100,
        ),
    )

    assert [
        (trade["action"], trade["symbol"], trade["shares"])
        for trade in report["trades"]
    ] == [
        ("buy", "600519", 10_000),
        ("sell", "600519", 5_000),
        ("buy", "159915", 2_500),
    ]


def test_drawdown_includes_first_execution_day_loss_from_initial_capital() -> None:
    report = run_portfolio_backtest(
        strategy_name="first_day_loss",
        allocations=[_allocation(1, {"600519": 1.0}, 1)],
        bars=[
            _bar(1, open_price=10.0, close=10.0),
            _bar(2, open_price=10.0, close=8.0),
        ],
        config=StrategyBacktestConfig(initial_capital=100_000.0),
    )

    assert report["equity_curve"][0]["drawdown_pct"] == pytest.approx(-20.0)
    assert report["metrics"]["max_drawdown_pct"] == pytest.approx(-20.0)


def test_same_engine_rotates_between_two_etfs() -> None:
    symbols = ("159915", "513100")
    bars = [
        _bar(day, open_price=price, close=price, symbol=symbol)
        for day, first, second in ((1, 1.0, 2.0), (2, 1.1, 2.0), (3, 1.0, 2.2))
        for symbol, price in zip(symbols, (first, second), strict=True)
    ]

    report = run_portfolio_backtest(
        strategy_name="etf_momentum_rotation",
        allocations=[
            _allocation(1, {"159915": 1.0}, 1),
            _allocation(2, {"513100": 1.0}, 2),
        ],
        bars=bars,
        config=StrategyBacktestConfig(
            initial_capital=100_000.0,
            side_cost_pct=0.0,
            lot_size=100,
        ),
    )

    assert [(trade["action"], trade["symbol"]) for trade in report["trades"]] == [
        ("buy", "159915"),
        ("sell", "159915"),
        ("buy", "513100"),
    ]
    assert report["equity_curve"][-1]["position"] == "513100"
    assert report["benchmark"]["kind"] == "equal_weight_buy_and_hold"


def test_requires_valid_prices_and_weights() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        StrategyBacktestConfig(lot_size=100.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="positive open and close"):
        run_portfolio_backtest(
            strategy_name="invalid",
            allocations=[_allocation(1, {"600519": 1.0}, 1)],
            bars=[_bar(2, open_price=0.0, close=10.0)],
            config=StrategyBacktestConfig(initial_capital=100_000.0),
        )

    with pytest.raises(ValueError, match="sum to at most 1"):
        run_portfolio_backtest(
            strategy_name="invalid",
            allocations=[_allocation(1, {"600519": 1.1}, 1)],
            bars=[_bar(2, open_price=10.0, close=10.0)],
            config=StrategyBacktestConfig(initial_capital=100_000.0),
        )
