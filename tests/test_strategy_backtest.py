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


# === Vectorized view-engine tests (run_portfolio_backtest_view) ===

import numpy as np

from src.core.script_market import MarketView
from src.core.strategy_backtest import (
    _align_strategy_comparisons,
    run_portfolio_backtest_view,
)


def _make_view(
    dates: list[date],
    symbols: list[str],
    open_matrix: list[list[float | None]],
    close_matrix: list[list[float | None]],
    members: list[list[bool]] | None = None,
) -> MarketView:
    if members is None:
        members = [[not (value is None) for value in row] for row in close_matrix]
    return MarketView(
        dates=np.array([day.toordinal() for day in dates], dtype=np.int32),
        symbols=np.array(symbols, dtype=object),
        open_matrix=np.array(open_matrix, dtype=np.float64),
        close_matrix=np.array(close_matrix, dtype=np.float64),
        members=np.array(members, dtype=bool),
        meta={},
    )


def _view_config(initial_capital: float = 100_000.0) -> StrategyBacktestConfig:
    return StrategyBacktestConfig(
        initial_capital=initial_capital,
        side_cost_pct=0.0,
        lot_size=100,
    )


def test_suspended_open_keeps_target_and_retries_next_signal() -> None:
    """C-2: a suspension-blocked rebalance must not silently drop the target;
    the pending target is retried daily once the symbol resumes trading."""
    days = [date(2026, 1, day) for day in range(1, 6)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[[None], [None], [10.0], [10.0], [10.0]],
        close_matrix=[[10.0], [10.0], [10.0], [10.0], [10.0]],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[
            _allocation(1, {"SZ000001": 1.0}, 1),
            _allocation(4, {"SZ000001": 1.0}, 2),
        ],
        view=view,
        config=_view_config(),
    )
    # Signal 1 executes at day 2 open (suspended) -> blocked, target kept.
    # Day 3 resumes trading: the pending target executes without any new
    # signal (retry), so the buy lands on day 3.
    buys = [trade for trade in report["trades"] if trade["action"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["execution_date"] == days[2].isoformat()
    assert report["metrics"]["blocked_signal_count"] == 1


def test_suspended_rebalance_retries_until_executed_without_new_signal() -> None:
    """P1: clear-out signal blocked by suspension executes after resumption
    even when no further signal ever arrives."""
    days = [date(2026, 1, day) for day in range(1, 8)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[
            [10.0], [10.0], [10.0], [None], [10.0], [10.0], [10.0],
        ],
        close_matrix=[
            [10.0], [10.0], [10.0], [10.0], [10.0], [10.0], [10.0],
        ],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[
            _allocation(1, {"SZ000001": 1.0}, 1),  # buy at day 2 open
            _allocation(3, {}, 2),  # cash-out signal; executes day 4 (suspended)
        ],
        view=view,
        config=_view_config(),
    )
    # Day 4 is suspended, no further signals: the sell must still happen on
    # day 5 (the first tradable open after the block).
    sells = [trade for trade in report["trades"] if trade["action"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["execution_date"] == days[4].isoformat()
    assert report["metrics"]["blocked_signal_count"] == 1
    # The position ends flat.
    assert report["metrics"]["final_equity"] == pytest.approx(
        report["metrics"]["initial_capital"]
    )


@pytest.mark.parametrize("cancel", [True, False])
def test_new_hold_signal_cancels_pending_liquidation(cancel) -> None:
    days = [date(2026, 1, day) for day in range(1, 8)]
    view = _make_view(
        days, ["SZ000001"],
        open_matrix=[[10.], [10.], [10.], [None], [None], [10.], [12.]],
        close_matrix=[[10.], [10.], [10.], [10.], [10.], [10.], [12.]],
    )
    allocations = [_allocation(1, {"SZ000001": 1.0}, 1), _allocation(3, {}, 2)]
    if cancel:
        allocations.append(_allocation(4, {"SZ000001": 1.0}, 3))
    report = run_portfolio_backtest_view(
        strategy_name="cancel liquidation", allocations=allocations,
        view=view, config=_view_config(),
    )
    sells = [t for t in report["trades"] if t["action"] == "sell"]
    if cancel:
        assert sells == []
        assert report["equity_curve"][-1]["position"] == "SZ000001"
        assert report["metrics"]["final_equity"] == pytest.approx(120_000.)
    else:
        assert len(sells) == 1
        assert sells[0]["execution_date"] == "2026-01-06"
        assert report["metrics"]["final_equity"] == pytest.approx(100_000.)


def test_sub_period_return_uses_opening_equity_not_original_capital() -> None:
    """P1: a sub-period that opens with carried-in profit must report its own
    return against the opening equity, not the original capital."""
    days = [date(2026, 1, day) for day in range(1, 8)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[[10.0]] * 7,
        close_matrix=[
            [10.0], [11.0], [12.0], [13.0], [14.0], [15.0], [16.0],
        ],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[
            _allocation(1, {"SZ000001": 1.0}, 1),  # buy at day 2 open = 10
        ],
        view=view,
        config=_view_config(100_000.0),
        period_start=days[3],  # window opens day 4, close 13 -> 130k
    )
    # Opening equity is the value right before day 4's open execution:
    # 10k shares marked at the day-4 open (10) = 100k. Final is 160k:
    # sub-period return = 60%, and the carry-in float (bought at 10, closed
    # 12 before the window) is excluded. The first day's intraday move
    # (open 10 -> close 13) IS included.
    assert report["metrics"]["opening_equity"] == pytest.approx(100_000.0)
    assert report["metrics"]["total_return_pct"] == pytest.approx(
        60_000.0 / 100_000.0 * 100.0
    )
    assert report["equity_curve"][0]["strategy_return_pct"] == pytest.approx(
        30_000.0 / 100_000.0 * 100.0
    )
    assert report["equity_curve"][-1]["strategy_return_pct"] == pytest.approx(
        60_000.0 / 100_000.0 * 100.0
    )


def test_risk_metrics_floor_total_loss_days_instead_of_zero() -> None:
    """P2: a total-loss day must not be recorded as a zero-return day."""
    from src.core.strategy_backtest import _log_return, _risk_metrics

    assert _log_return(0.0, 100.0) == pytest.approx(-27.631)
    assert _log_return(0.0, 0.0) == 0.0
    metrics = _risk_metrics(
        [100.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],
        opening_equity=100.0,
        benchmark_opening_equity=100.0,
    )
    assert metrics["annualized_volatility_pct"] > 0.0


def test_first_day_intraday_return_is_not_lost() -> None:
    """P1: buy at open 10, close 20 on the same day must report +100%,
    not 0% — the opening equity is pre-execution, not the first close."""
    days = [date(2026, 1, day) for day in range(1, 3)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[[10.0], [10.0]],
        close_matrix=[[20.0], [20.0]],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[_allocation(1, {"SZ000001": 1.0}, 1)],
        view=view,
        config=_view_config(100_000.0),
    )
    assert report["metrics"]["opening_equity"] == pytest.approx(100_000.0)
    # 10k shares bought at 10, close 20 -> 200k -> +100%.
    assert report["metrics"]["total_return_pct"] == pytest.approx(100.0)
    assert report["equity_curve"][0]["strategy_return_pct"] == pytest.approx(100.0)


def test_suspended_close_valuation_carries_last_close_forward() -> None:
    """C-2: a suspended close must not wipe the position's mark to zero."""
    days = [date(2026, 1, day) for day in range(1, 4)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[[10.0], [10.0], [10.0]],
        close_matrix=[[10.0], [None], [11.0]],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[_allocation(1, {"SZ000001": 1.0}, 1)],
        view=view,
        config=_view_config(),
    )
    curve = report["equity_curve"]
    # window opens on the buy day: bought at 10 open, close suspended ->
    # carry the buy price, equity stays 100k (the old code dropped it to 0).
    assert curve[0]["strategy_equity"] == pytest.approx(100_000.0)
    # day 3: close 11 -> equity recovers without a phantom jump.
    assert curve[1]["strategy_equity"] == pytest.approx(110_000.0)


def test_benchmark_uses_fixed_day_one_weights_not_survivor_renormalization() -> None:
    """M-2: a delisted member must keep its 1/N weight, not be erased."""
    days = [date(2026, 1, day) for day in range(1, 5)]
    # A rises 100->110->121; B stays 100 then delists (NaN close, NaN open).
    # The first signal executes at day 2 open: the benchmark's day-one is
    # day 2, when both members are still valid.
    view = _make_view(
        days,
        ["A", "B"],
        open_matrix=[
            [100.0, 100.0],
            [100.0, 100.0],
            [100.0, None],
            [100.0, None],
        ],
        close_matrix=[
            [100.0, 100.0],
            [100.0, 100.0],
            [110.0, None],
            [121.0, None],
        ],
        members=[
            [True, True],
            [True, True],
            [True, True],
            [True, False],
        ],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[_allocation(1, {}, 1)],  # stay all-cash; executes day 2
        view=view,
        config=_view_config(),
    )
    curve = report["equity_curve"]
    # Fixed weights from day 2: day3 return = 0.5*10% + 0.5*0% = 5%;
    # day4 = 0.5*21% + 0.5*0% = 10.5%.
    assert curve[1]["benchmark_equity"] == pytest.approx(105_000.0)
    assert curve[2]["benchmark_equity"] == pytest.approx(110_500.0)


def test_period_start_truncates_window_and_preserves_opening_position() -> None:
    """C-3/M-5: period_start is the real first day; pre-window signals
    pre-execute so the sub-period opens with the held position."""
    days = [date(2026, 1, day) for day in range(1, 7)]
    view = _make_view(
        days,
        ["SZ000001"],
        open_matrix=[[10.0]] * 6,
        close_matrix=[[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
    )
    report = run_portfolio_backtest_view(
        strategy_name="t",
        allocations=[
            _allocation(1, {"SZ000001": 1.0}, 1),  # executes day 1 (pre-window)
            _allocation(5, {}, 2),  # cash out at day 5
        ],
        view=view,
        config=_view_config(),
        period_start=days[2],  # day 3
        period_end=days[5],  # day 6
    )
    assert report["period"]["start_date"] == days[2].isoformat()
    curve = report["equity_curve"]
    # Opens with 10k shares bought at 10; day 3 close 12 -> 120k.
    assert curve[0]["strategy_equity"] == pytest.approx(120_000.0)
    # The buy happened before the window (trades list) and the sell on day 6.
    actions = [
        (trade["action"], trade["execution_date"]) for trade in report["trades"]
    ]
    assert ("buy", days[1].isoformat()) in actions
    assert ("sell", days[5].isoformat()) in actions


def test_align_strategy_comparisons_return_matches_normalized_curve() -> None:
    """C-8b: headline return must agree with the normalized plotted curve."""
    dates = [date(2026, 1, day) for day in range(1, 6)]
    simulation_indices = [2, 3, 4]
    comparisons = _align_strategy_comparisons(
        comparisons_reports=[
            {
                "name": "cmp",
                "report": {
                    "equity_curve": [
                        {"date": dates[i].isoformat(), "strategy_equity": value}
                        for i, value in enumerate([100.0, 200.0, 300.0, 400.0, 500.0])
                    ],
                    "metrics": {"final_equity": 500.0},
                },
            }
        ],
        dates=dates,
        simulation_indices=simulation_indices,
        initial_capital=100_000.0,
    )
    entry = comparisons["strategy:cmp"]
    # Window starts at 300 (first valid in-window value); curve ends at 500.
    # Return = 500/300 - 1 = 66.67%, NOT the full-range final_equity 400%.
    assert entry["total_return_pct"] == pytest.approx(66.6667, abs=1e-3)
    assert entry["equity"][0] == pytest.approx(100_000.0)
    assert entry["equity"][-1] == pytest.approx(166_666.67, abs=1.0)
