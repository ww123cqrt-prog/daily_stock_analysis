from __future__ import annotations

import pandas as pd
import pytest

from _optimized_strategy_core import OptimizedStrategy


def _frames(
    scores_by_day: list[list[float]],
    labels_by_day: list[list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-02", periods=len(scores_by_day))
    symbols = ["SH600000", "SH600001", "SH600002", "SH600003"]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["datetime", "instrument"]
    )
    scores = [value for day in scores_by_day for value in day]
    labels = [value for day in labels_by_day for value in day]
    return (
        pd.DataFrame({"score": scores}, index=index),
        pd.DataFrame({"label": labels}, index=index),
        dates,
    )


def _market(dates: pd.DatetimeIndex, trend: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ma200": [1.0] * len(dates), "trend_200": trend},
        index=dates,
    )


def test_position_reduction_and_reentry_charge_weight_turnover() -> None:
    pred, label, dates = _frames(
        [[4, 3, 2, 1]] * 3,
        [[0, 0, 0, 0]] * 3,
    )
    strategy = OptimizedStrategy(
        topk=2,
        cost_bps=100,
        rebalance_freq=3,
        turnover_cap=1.0,
        trend_filter=True,
        trend_reduce_pct=0.5,
        max_drawdown_stop=0.0,
    )

    result = strategy.run(pred, label, _market(dates, [1.0, 0.0, 1.0]))

    assert result.iloc[1]["position_pct"] == pytest.approx(0.5)
    assert result.iloc[1]["turnover"] == pytest.approx(0.25)
    assert result.iloc[1]["return"] == pytest.approx(-0.0025)
    assert result.iloc[2]["position_pct"] == pytest.approx(1.0)
    assert result.iloc[2]["turnover"] == pytest.approx(0.25)
    assert result.iloc[2]["return"] == pytest.approx(-0.0025)


def test_drawdown_liquidation_and_cash_reentry_charge_turnover() -> None:
    pred, label, dates = _frames(
        [[4, 3, 2, 1]] * 3,
        [[-0.20, -0.20, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    )
    strategy = OptimizedStrategy(
        topk=2,
        cost_bps=100,
        turnover_cap=1.0,
        max_drawdown_stop=-0.10,
        max_dd_cooldown_days=1,
    )

    result = strategy.run(pred, label)

    assert result.iloc[1]["position_pct"] == pytest.approx(0.0)
    assert result.iloc[1]["turnover"] == pytest.approx(0.5)
    assert result.iloc[1]["return"] == pytest.approx(-0.005)
    assert result.iloc[2]["position_pct"] == pytest.approx(1.0)
    assert result.iloc[2]["turnover"] == pytest.approx(0.5)
    assert result.iloc[2]["return"] == pytest.approx(-0.005)


def test_holding_and_position_changes_use_actual_target_weights() -> None:
    pred, label, dates = _frames(
        [[4, 3, 2, 1], [1, 2, 4, 3]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
    )
    strategy = OptimizedStrategy(
        topk=2,
        cost_bps=100,
        turnover_cap=1.0,
        trend_filter=True,
        trend_reduce_pct=0.5,
        max_drawdown_stop=0.0,
    )

    result = strategy.run(pred, label, _market(dates, [1.0, 0.0]))

    assert result.iloc[1]["position_pct"] == pytest.approx(0.5)
    assert result.iloc[1]["turnover"] == pytest.approx(0.75)
    assert result.iloc[1]["return"] == pytest.approx(-0.0075)
