from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
                    "list_date": pd.Timestamp("1990-01-01"),
                    "listing_trading_day": 100 + index,
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
    assert daily.loc[daily["date"] == after_move, "portfolio_return"].iloc[0] > 0.005
    assert (
        daily.loc[daily["date"] == execution_day, "gross_position"].iloc[0]
        <= 0.03 + 1e-12
    )


def test_limit_up_open_defers_buy_until_next_tradable_open() -> None:
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

    daily = result["daily"]
    blocked_day = pd.Timestamp("2025-01-13")
    retry_day = pd.Timestamp("2025-01-14")
    assert daily.loc[daily["date"] == blocked_day, "trade_count"].iloc[0] == 0
    assert daily.loc[daily["date"] == blocked_day, "pending_order_count"].iloc[0] == 1
    assert daily.loc[daily["date"] == retry_day, "trade_count"].iloc[0] == 1
    assert daily.loc[daily["date"] == retry_day, "pending_order_count"].iloc[0] == 0
    assert daily["trade_count"].sum() == 1


def test_blocked_sell_prevents_same_industry_buy_from_breaching_cap() -> None:
    module = load_module()
    dates = pd.bdate_range("2025-01-06", periods=12)
    rows = []
    for symbol in ("SH600000", "SH600001"):
        for index, day in enumerate(dates):
            open_price = 10.0
            previous_close = 10.0
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": open_price,
                    "previous_close": previous_close,
                    "list_date": pd.Timestamp("1990-01-01"),
                    "listing_trading_day": 100 + index,
                    "industry": "technology",
                    "board": "main",
                    "is_st": False,
                    "suspended": symbol == "SH600000"
                    and day == pd.Timestamp("2025-01-20"),
                }
            )
    market = pd.DataFrame(rows)
    benchmark = pd.DataFrame({"date": dates, "open": [100.0] * len(dates)})
    predictions = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-17")],
            "instrument": ["SH600000", "SH600001"],
            "score": [2.0, 2.0],
        }
    )

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="full_risk_control",
        initial_cash=100_000.0,
        topk=1,
        fixed_position=0.9,
        single_stock_max=0.25,
        industry_max=0.25,
        growth_star_max=0.30,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
    )

    daily = result["daily"].set_index("date")
    blocked_rebalance = daily.loc[pd.Timestamp("2025-01-20")]
    assert blocked_rebalance["trade_count"] == 0
    assert blocked_rebalance["pending_order_count"] == 2
    assert blocked_rebalance["gross_position"] <= 0.25 + 1e-12


def test_blocked_sell_prevents_buy_above_current_regime_position_limit() -> None:
    module = load_module()
    dates = pd.bdate_range("2025-01-06", periods=12)
    rows = []
    for symbol, industry in (("SH600000", "bank"), ("SH600001", "technology")):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "open": 10.0,
                    "previous_close": 10.0,
                    "list_date": pd.Timestamp("1990-01-01"),
                    "listing_trading_day": 100 + index,
                    "industry": industry,
                    "board": "main",
                    "is_st": False,
                    "suspended": symbol == "SH600000"
                    and day == pd.Timestamp("2025-01-20"),
                }
            )
    predictions = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-17")],
            "instrument": ["SH600000", "SH600001"],
            "score": [2.0, 2.0],
        }
    )
    benchmark = pd.DataFrame({"date": dates, "open": [100.0] * len(dates)})
    regimes = pd.Series("risk_on", index=dates)
    regimes.loc[pd.Timestamp("2025-01-17")] = "risk_off"

    result = module.simulate_strategy(
        predictions=predictions,
        market=pd.DataFrame(rows),
        benchmark=benchmark,
        strategy_id="full_risk_control",
        initial_cash=100_000.0,
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

    blocked_rebalance = result["daily"].set_index("date").loc["2025-01-20"]
    assert blocked_rebalance["trade_count"] == 0
    assert 0.89 <= blocked_rebalance["gross_position"] <= 0.91


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


def test_board_return_contribution_uses_carried_position_open_to_open_pnl() -> None:
    module = load_module()
    dates = pd.bdate_range("2025-01-06", periods=7)
    market = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["SZ300001"] * len(dates),
            "open": [10.0] * 6 + [11.0],
            "previous_close": [10.0] * len(dates),
            "list_date": [pd.Timestamp("2020-01-01")] * len(dates),
            "listing_trading_day": list(range(100, 100 + len(dates))),
            "industry": ["technology"] * len(dates),
            "board": ["growth"] * len(dates),
            "is_st": [False] * len(dates),
            "suspended": [False] * len(dates),
        }
    )
    benchmark = pd.DataFrame({"date": dates, "open": [100.0] * len(dates)})
    predictions = pd.DataFrame(
        {
            "datetime": [dates[4]],
            "instrument": ["SZ300001"],
            "score": [1.0],
        }
    )

    result = module.simulate_strategy(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id="weekly_top50",
        initial_cash=100_000.0,
        topk=1,
        fixed_position=0.9,
        single_stock_max=0.9,
        industry_max=0.9,
        growth_star_max=0.9,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=0.0,
    )

    move_day = result["daily"].set_index("date").loc[dates[6]]
    assert move_day["board_return_contribution_growth"] > 0.08
    assert np.isclose(
        move_day["board_excess_contribution_growth"],
        move_day["board_return_contribution_growth"],
    )
    assert move_day["board_return_contribution_main"] == 0.0


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


def test_sector_and_full_control_have_distinct_selection_rules() -> None:
    module = load_module()
    candidates = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "score": list(range(8, 0, -1)),
            "industry": ["tech"] * 4 + ["bank", "consumer", "health", "energy"],
            "board": ["growth"] * 4 + ["main"] * 4,
            "tradable": [True] * 8,
        }
    )
    common = {
        "candidates": candidates,
        "current_symbols": {"F"},
        "topk": 4,
        "target_position": 0.8,
        "single_stock_max": 0.25,
        "industry_max": 0.25,
        "growth_star_max": 0.25,
        "dropout_buffer": 4,
    }

    weekly = module.build_strategy_target(strategy_id="weekly_top50", **common)
    low_turnover = module.build_strategy_target(strategy_id="low_turnover", **common)
    sector = module.build_strategy_target(strategy_id="sector_constrained", **common)
    full = module.build_strategy_target(strategy_id="full_risk_control", **common)

    assert weekly["symbol"].tolist() == ["A", "B", "C", "D"]
    assert "F" in set(low_turnover["symbol"])
    assert set(sector["symbol"]) != set(weekly["symbol"])
    assert sector["industry"].nunique() >= 3
    assert "F" in set(full["symbol"])
    assert set(full["symbol"]) != set(low_turnover["symbol"])


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


@pytest.mark.parametrize(
    "strategy_id",
    [
        "weekly_top50",
        "low_turnover",
        "regime_scaled",
        "sector_constrained",
        "full_risk_control",
    ],
)
def test_every_candidate_strategy_obeys_portfolio_caps(strategy_id: str) -> None:
    module = load_module()
    candidates = pd.DataFrame(
        {
            "symbol": [f"SZ300{i:03d}" for i in range(20)]
            + [f"SH600{i:03d}" for i in range(20)],
            "score": list(range(40, 0, -1)),
            "industry": ["tech"] * 25 + ["other"] * 15,
            "board": ["growth"] * 20 + ["main"] * 20,
            "tradable": [True] * 40,
        }
    )

    target = module.build_strategy_target(
        candidates,
        strategy_id=strategy_id,
        current_symbols=set(),
        topk=40,
        target_position=0.9,
        single_stock_max=0.03,
        industry_max=0.25,
        growth_star_max=0.30,
    )

    assert target["weight"].max() <= 0.03 + 1e-12
    assert target.groupby("industry")["weight"].sum().max() <= 0.25 + 1e-12
    assert (
        target.loc[target["board"].isin(["growth", "star"]), "weight"].sum()
        <= 0.30 + 1e-12
    )


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
