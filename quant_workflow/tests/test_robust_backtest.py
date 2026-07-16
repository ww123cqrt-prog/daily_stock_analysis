from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robust_backtest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("robust_backtest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def market_fixture(
    *, limit_up_on_execution: bool = False, flat: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-06", periods=7)
    rows = []
    for symbol, base in [("SH600000", 10.0), ("SZ000001", 20.0)]:
        previous = base
        for index, day in enumerate(dates):
            if flat:
                open_price = base
            elif symbol == "SH600000" and index == 6:
                open_price = 12.0
            else:
                open_price = base
            if limit_up_on_execution and symbol == "SH600000" and day == dates[5]:
                previous = 10.0
                open_price = 11.0
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": open_price,
                    "previous_close": previous,
                    "industry": "bank" if symbol == "SH600000" else "consumer",
                    "board": "main",
                    "is_st": False,
                    "suspended": False,
                }
            )
            previous = open_price
    market = pd.DataFrame(rows)
    benchmark = pd.DataFrame({"date": dates, "open": [100.0] * len(dates)})
    predictions = pd.DataFrame(
        {
            "datetime": [dates[4], dates[4]],
            "instrument": ["SH600000", "SZ000001"],
            "score": [2.0, 1.0],
        }
    )
    return market, benchmark, predictions


def test_weekly_schedule_uses_last_trading_day_signal_and_next_open_execution() -> None:
    module = load_module()
    calendar = pd.bdate_range("2025-01-06", periods=10)

    schedule = module.weekly_signal_execution_pairs(calendar)

    assert schedule[0] == (pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-13"))
    assert all(execution > signal for signal, execution in schedule)


def test_backtest_buys_at_next_open_and_only_earns_later_open_to_open_move() -> None:
    module = load_module()
    market, benchmark, predictions = market_fixture()

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="weekly_top50",
        initial_cash=100000.0,
        topk=1,
        fixed_position=0.9,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
    )

    daily = result["daily"]
    execution_day = pd.Timestamp("2025-01-13")
    after_move = pd.Timestamp("2025-01-14")
    assert daily.loc[daily["date"] < execution_day, "portfolio_return"].eq(0.0).all()
    assert daily.loc[daily["date"] == execution_day, "trade_count"].iloc[0] == 1
    assert daily.loc[daily["date"] == after_move, "portfolio_return"].iloc[0] > 0.15


def test_limit_up_open_blocks_buy_and_keeps_cash() -> None:
    module = load_module()
    market, benchmark, predictions = market_fixture(limit_up_on_execution=True)

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="weekly_top50",
        initial_cash=100000.0,
        topk=1,
        fixed_position=0.9,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
    )

    assert result["daily"]["trade_count"].sum() == 0
    assert result["daily"]["portfolio_value"].eq(100000.0).all()


def test_flat_market_loses_exactly_recorded_transaction_costs() -> None:
    module = load_module()
    market, benchmark, predictions = market_fixture(flat=True)

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="weekly_top50",
        initial_cash=100000.0,
        topk=1,
        fixed_position=0.9,
        commission_rate=0.0003,
        minimum_commission=5.0,
        slippage_bps=10.0,
    )

    total_cost = result["daily"]["transaction_cost"].sum()
    assert total_cost > 0
    assert np.isclose(
        result["daily"]["portfolio_value"].iloc[-1], 100000.0 - total_cost
    )
    assert result["metrics"]["annual_excess_return"] < 0


def test_low_turnover_keeps_existing_stock_within_dropout_buffer() -> None:
    module = load_module()
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "score": [4.0, 3.0, 2.0, 1.0],
            "industry": ["i1", "i2", "i3", "i4"],
            "board": ["main"] * 4,
            "tradable": [True] * 4,
        }
    )

    selected = module.select_symbols(
        candidates,
        strategy_id="low_turnover",
        topk=2,
        current_symbols={"C"},
        dropout_buffer=1,
    )

    assert selected == ["A", "C"]


def test_full_risk_control_applies_regime_and_sector_caps() -> None:
    module = load_module()
    candidates = pd.DataFrame(
        {
            "symbol": [f"SZ300{i:03d}" for i in range(10)]
            + [f"SH600{i:03d}" for i in range(20)],
            "score": list(range(30, 0, -1)),
            "industry": ["tech"] * 15 + ["other"] * 15,
            "board": ["growth"] * 10 + ["main"] * 20,
            "tradable": [True] * 30,
        }
    )

    target = module.build_strategy_target(
        candidates,
        strategy_id="full_risk_control",
        current_symbols=set(),
        topk=30,
        target_position=0.5,
        single_stock_max=0.03,
        industry_max=0.25,
        growth_star_max=0.30,
    )

    assert target["weight"].sum() <= 0.5 + 1e-12
    assert target["weight"].max() <= 0.03 + 1e-12
    assert target.groupby("industry")["weight"].sum().max() <= 0.25 + 1e-12


def test_full_risk_control_can_reduce_next_day_without_daily_reranking() -> None:
    module = load_module()
    market, benchmark, predictions = market_fixture(flat=True)
    dates = pd.DatetimeIndex(benchmark["date"])
    regimes = pd.Series("risk_on", index=dates)
    regimes.loc[pd.Timestamp("2025-01-13")] = "risk_off"

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="full_risk_control",
        initial_cash=100000.0,
        topk=1,
        fixed_position=0.9,
        regimes=regimes,
        regime_positions={
            "risk_on": 0.9,
            "neutral": 0.7,
            "defensive": 0.5,
            "risk_off": 0.2,
        },
        single_stock_max=0.9,
        industry_max=0.9,
        growth_star_max=0.9,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
    )

    daily = result["daily"].set_index("date")
    assert daily.loc[pd.Timestamp("2025-01-13"), "gross_position"] > 0.85
    assert 0.15 <= daily.loc[pd.Timestamp("2025-01-14"), "gross_position"] <= 0.25
    assert daily.loc[pd.Timestamp("2025-01-14"), "trade_count"] == 1


def test_additional_execution_delay_moves_trade_from_t_plus_1_to_t_plus_2() -> None:
    module = load_module()
    market, benchmark, predictions = market_fixture(flat=True)

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="weekly_top50",
        initial_cash=100000.0,
        topk=1,
        fixed_position=0.9,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
        additional_execution_delay_days=1,
    )

    daily = result["daily"].set_index("date")
    assert daily.loc[pd.Timestamp("2025-01-13"), "trade_count"] == 0
    assert daily.loc[pd.Timestamp("2025-01-14"), "trade_count"] == 1
    assert daily.loc[pd.Timestamp("2025-01-14"), "signal_date"] == pd.Timestamp(
        "2025-01-10"
    )


def test_40bp_round_trip_cost_adds_regulatory_cost_and_slippage_to_40bp() -> None:
    module = load_module()
    buy = pd.DataFrame({"side": ["buy"], "notional": [100000.0]})
    sell = pd.DataFrame({"side": ["sell"], "notional": [100000.0]})

    buy_cost = module.calculate_all_in_order_costs(
        buy,
        trade_date="2025-01-02",
        commission_rate=0.0003,
        minimum_commission=0.0,
        round_trip_total_bps=40.0,
    )
    sell_cost = module.calculate_all_in_order_costs(
        sell,
        trade_date="2025-01-03",
        commission_rate=0.0003,
        minimum_commission=0.0,
        round_trip_total_bps=40.0,
    )

    assert np.isclose(buy_cost["total"] + sell_cost["total"], 400.0)
    assert buy_cost["slippage"] > sell_cost["slippage"]
