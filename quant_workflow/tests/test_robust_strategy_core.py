from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "robust_strategy_core.py"
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
ACCEPTANCE = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["acceptance"]


def load_module():
    spec = importlib.util.spec_from_file_location("robust_strategy_core", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rolling_split_uses_fixed_window_validation_and_label_purge() -> None:
    module = load_module()
    calendar = pd.bdate_range("2019-01-01", "2025-12-31")

    split = module.make_rolling_split(
        calendar,
        "2023Q1",
        train_days=504,
        validation_days=126,
        label_horizon_days=5,
    )

    assert split.test_start == pd.Timestamp("2023-01-02")
    assert (
        len(calendar[(calendar >= split.train_start) & (calendar <= split.train_end)])
        == 504
    )
    assert (
        len(
            calendar[
                (calendar >= split.validation_start)
                & (calendar <= split.validation_end)
            ]
        )
        == 126
    )
    train_end_index = calendar.get_loc(split.train_end)
    validation_start_index = calendar.get_loc(split.validation_start)
    validation_end_index = calendar.get_loc(split.validation_end)
    test_start_index = calendar.get_loc(split.test_start)
    assert validation_start_index - train_end_index - 1 == 5
    assert test_start_index - validation_end_index - 1 == 5
    assert split.test_end == pd.Timestamp("2023-03-31")


def test_five_day_open_label_uses_t_plus_1_to_t_plus_6_without_crossing_symbols() -> (
    None
):
    module = load_module()
    dates = pd.bdate_range("2026-01-05", periods=8)
    frame = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["SH600000"] * len(dates) + ["SZ000001"] * len(dates),
            "open": list(range(10, 18)) + list(range(20, 28)),
        }
    )

    labeled = module.add_forward_open_label(frame)

    sh_first = labeled[
        (labeled["symbol"] == "SH600000") & (labeled["date"] == dates[0])
    ].iloc[0]
    sz_first = labeled[
        (labeled["symbol"] == "SZ000001") & (labeled["date"] == dates[0])
    ].iloc[0]
    assert np.isclose(sh_first["label_5d_open"], 16 / 11 - 1)
    assert np.isclose(sz_first["label_5d_open"], 26 / 21 - 1)
    assert (
        labeled.groupby("symbol")["label_5d_open"]
        .apply(lambda values: values.tail(6).isna().all())
        .all()
    )


def test_market_regime_position_takes_effect_one_trading_day_later() -> None:
    module = load_module()
    regimes = pd.Series(
        ["risk_on", "risk_off", "neutral"],
        index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
    )

    positions = module.execution_positions_from_regime(
        regimes,
        {"risk_on": 0.9, "neutral": 0.7, "defensive": 0.5, "risk_off": 0.2},
    )

    assert np.isnan(positions.iloc[0])
    assert positions.iloc[1] == 0.9
    assert positions.iloc[2] == 0.2


def test_portfolio_weights_obey_stock_industry_and_growth_star_caps() -> None:
    module = load_module()
    candidates = pd.DataFrame(
        {
            "symbol": [f"SZ300{i:03d}" for i in range(15)]
            + [f"SH600{i:03d}" for i in range(35)],
            "score": list(range(50, 0, -1)),
            "industry": ["tech"] * 20 + ["industry_b"] * 15 + ["industry_c"] * 15,
            "board": ["growth"] * 15 + ["main"] * 35,
            "tradable": [True] * 50,
        }
    )

    portfolio = module.build_constrained_portfolio(
        candidates,
        target_position=0.9,
        topk=50,
        single_stock_max=0.03,
        industry_max=0.25,
        growth_star_max=0.30,
    )

    assert portfolio["weight"].max() <= 0.03 + 1e-12
    assert portfolio.groupby("industry")["weight"].sum().max() <= 0.25 + 1e-12
    assert (
        portfolio.loc[portfolio["board"].isin(["growth", "star"]), "weight"].sum()
        <= 0.30 + 1e-12
    )
    assert portfolio["weight"].sum() <= 0.9 + 1e-12
    assert portfolio["weight"].sum() > 0.70


def test_a_share_price_limits_change_by_board_date_and_st_status() -> None:
    module = load_module()

    assert module.price_limit_ratio("SZ300001", "2020-08-21", is_st=False) == 0.10
    assert module.price_limit_ratio("SZ300001", "2020-08-24", is_st=False) == 0.20
    assert module.price_limit_ratio("SH688001", "2020-01-02", is_st=False) == 0.20
    assert module.price_limit_ratio("SH600000", "2026-01-02", is_st=False) == 0.10
    assert module.price_limit_ratio("SH600000", "2026-01-02", is_st=True) == 0.05
    assert (
        module.price_limit_ratio(
            "SH688001",
            "2025-01-02",
            is_st=False,
            list_date="2025-01-02",
            listing_trading_day=1,
        )
        is None
    )
    assert (
        module.price_limit_ratio(
            "SH600001",
            "2020-01-02",
            is_st=False,
            list_date="2020-01-02",
            listing_trading_day=1,
            side="buy",
        )
        == 0.44
    )
    assert (
        module.price_limit_ratio(
            "SH600001",
            "2020-01-02",
            is_st=False,
            list_date="2020-01-02",
            listing_trading_day=1,
            side="sell",
        )
        == 0.36
    )
    assert (
        module.price_limit_ratio(
            "SH688001",
            "2025-01-09",
            is_st=True,
            list_date="2025-01-02",
            listing_trading_day=6,
        )
        == 0.20
    )
    assert (
        module.price_limit_ratio(
            "SZ300001",
            "2025-01-09",
            is_st=True,
            list_date="2025-01-02",
            listing_trading_day=6,
        )
        == 0.20
    )
    assert (
        module.price_limit_ratio(
            "SZ300001",
            "2025-01-08",
            is_st=False,
            list_date="2025-01-02",
            listing_trading_day=5,
        )
        is None
    )
    assert (
        module.price_limit_ratio(
            "SZ300001",
            "2025-01-09",
            is_st=False,
            list_date="2025-01-02",
            listing_trading_day=6,
        )
        == 0.20
    )
    assert (
        module.price_limit_ratio(
            "SH600001",
            "2025-01-02",
            is_st=False,
            list_date="2025-01-02",
            listing_trading_day=1,
        )
        is None
    )


def test_open_execution_is_fail_closed_for_suspension_and_blocked_limits() -> None:
    module = load_module()

    assert not module.can_execute_open(
        side="buy",
        previous_close=10.0,
        open_price=11.0,
        limit_ratio=0.10,
        suspended=False,
    )
    assert not module.can_execute_open(
        side="sell",
        previous_close=10.0,
        open_price=9.0,
        limit_ratio=0.10,
        suspended=False,
    )
    assert not module.can_execute_open(
        side="buy",
        previous_close=10.0,
        open_price=10.2,
        limit_ratio=0.10,
        suspended=True,
    )
    assert module.can_execute_open(
        side="buy",
        previous_close=10.0,
        open_price=10.5,
        limit_ratio=0.10,
        suspended=False,
    )
    assert module.can_execute_open(
        side="buy",
        previous_close=float("nan"),
        open_price=25.0,
        limit_ratio=None,
        suspended=False,
    )
    assert not module.can_execute_open(
        side="buy",
        previous_close=10.01,
        open_price=11.01,
        limit_ratio=0.10,
        suspended=False,
    )
    assert not module.can_execute_open(
        side="sell",
        previous_close=10.01,
        open_price=9.01,
        limit_ratio=0.10,
        suspended=False,
    )


def test_order_costs_apply_minimum_commission_and_date_specific_stamp_tax() -> None:
    module = load_module()
    orders = pd.DataFrame(
        {
            "side": ["buy", "sell"],
            "notional": [1000.0, 10000.0],
        }
    )

    before = module.calculate_order_costs(
        orders,
        trade_date="2023-08-25",
        commission_rate=0.0003,
        minimum_commission=5.0,
        slippage_bps=0.0,
    )
    after = module.calculate_order_costs(
        orders,
        trade_date="2023-08-28",
        commission_rate=0.0003,
        minimum_commission=5.0,
        slippage_bps=0.0,
    )

    assert before["commission"] == 10.0
    assert before["stamp_tax"] == 10.0
    assert after["stamp_tax"] == 5.0
    assert after["total"] == 15.0


def test_acceptance_requires_every_development_gate() -> None:
    module = load_module()
    quarters = [
        {
            "annual_excess_return": 0.08,
            "information_ratio": 0.8,
            "rank_ic": 0.03,
            "max_drawdown": -0.10,
        }
        for _ in range(12)
    ]
    regimes = {
        "risk_on": {"days": 100, "annual_excess_return": 0.08},
        "neutral": {"days": 100, "annual_excess_return": 0.04},
        "defensive": {"days": 80, "annual_excess_return": 0.01},
        "risk_off": {"days": 80, "annual_excess_return": -0.03},
    }

    verdict = module.evaluate_development_acceptance(
        quarter_metrics=quarters,
        aggregate_annual_excess_return=0.07,
        aggregate_max_drawdown=-0.15,
        window_rank_correlations=[0.75, 0.8],
        top50_overlaps=[0.65, 0.7],
        stress_excess_return=0.01,
        delayed_excess_return=0.005,
        regime_metrics=regimes,
        thresholds=ACCEPTANCE,
    )

    assert verdict["passed"] is True
    failed = module.evaluate_development_acceptance(
        quarter_metrics=quarters,
        aggregate_annual_excess_return=0.07,
        aggregate_max_drawdown=-0.15,
        window_rank_correlations=[0.75, 0.8],
        top50_overlaps=[0.65, 0.7],
        stress_excess_return=-0.001,
        delayed_excess_return=0.005,
        regime_metrics=regimes,
        thresholds=ACCEPTANCE,
    )
    assert failed["passed"] is False
    assert "stress_cost_positive_excess" in failed["failed_gates"]

    cross_quarter_drawdown = module.evaluate_development_acceptance(
        quarter_metrics=quarters,
        aggregate_annual_excess_return=0.07,
        aggregate_max_drawdown=-0.25,
        window_rank_correlations=[0.75, 0.8],
        top50_overlaps=[0.65, 0.7],
        stress_excess_return=0.01,
        delayed_excess_return=0.005,
        regime_metrics=regimes,
        thresholds=ACCEPTANCE,
    )
    assert cross_quarter_drawdown["passed"] is False
    assert "maximum_drawdown" in cross_quarter_drawdown["failed_gates"]
    assert cross_quarter_drawdown["values"]["aggregate_max_drawdown_abs"] == 0.25

    stricter = dict(ACCEPTANCE)
    stricter["annual_excess_return_min"] = 0.08
    strict_verdict = module.evaluate_development_acceptance(
        quarter_metrics=quarters,
        aggregate_annual_excess_return=0.07,
        aggregate_max_drawdown=-0.15,
        window_rank_correlations=[0.75, 0.8],
        top50_overlaps=[0.65, 0.7],
        stress_excess_return=0.01,
        delayed_excess_return=0.005,
        regime_metrics=regimes,
        thresholds=stricter,
    )
    assert strict_verdict["passed"] is False
    assert "aggregate_annual_excess" in strict_verdict["failed_gates"]


def test_model_selection_uses_development_metrics_only() -> None:
    module = load_module()
    candidates = [
        {"id": "stable", "development_score": 1.2, "confirmation_score": -99.0},
        {"id": "holdout_lucky", "development_score": 0.8, "confirmation_score": 99.0},
        {"id": "third", "development_score": 0.7, "confirmation_score": 0.5},
    ]

    selected = module.select_ensemble_models(candidates, count=2)

    assert [item["id"] for item in selected] == ["stable", "holdout_lucky"]


def test_confirmation_split_keeps_2026_out_of_train_and_validation_labels() -> None:
    module = load_module()
    calendar = pd.bdate_range("2019-01-01", "2026-07-15")

    split = module.make_confirmation_split(
        calendar,
        train_days=1008,
        validation_days=126,
        label_horizon_days=5,
        test_start="2026-01-01",
        test_end="2026-07-15",
    )

    assert split.test_start == pd.Timestamp("2026-01-01")
    assert split.test_end == pd.Timestamp("2026-07-15")
    assert (
        len(calendar[(calendar >= split.train_start) & (calendar <= split.train_end)])
        == 1008
    )
    assert (
        len(
            calendar[
                (calendar >= split.validation_start)
                & (calendar <= split.validation_end)
            ]
        )
        == 126
    )
    assert (
        calendar.get_loc(split.validation_start) - calendar.get_loc(split.train_end) - 1
        == 5
    )
    assert (
        calendar.get_loc(split.test_start) - calendar.get_loc(split.validation_end) - 1
        == 5
    )
