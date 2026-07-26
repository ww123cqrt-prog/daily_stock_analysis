# -*- coding: utf-8 -*-
"""Tests for explicit, asset-agnostic strategy signal generators."""

from datetime import date

import pytest

from src.core.strategy_backtest import PriceBar
from src.core.strategy_signals import (
    build_buy_and_hold_allocations,
    build_dual_moving_average_allocations,
    build_momentum_rotation_allocations,
)


def _bars(symbol: str, closes: list[float]) -> list[PriceBar]:
    return [
        PriceBar(
            symbol=symbol,
            date=date(2026, 1, index),
            open=close,
            close=close,
        )
        for index, close in enumerate(closes, start=1)
    ]


def test_momentum_rotation_selects_the_strongest_asset_above_its_average() -> None:
    allocations = build_momentum_rotation_allocations(
        bars=[
            *_bars("SZ159915", [10, 11, 12, 13, 12, 11]),
            *_bars("SH513100", [10, 10, 10, 12, 14, 16]),
        ],
        symbol_order=("SZ159915", "SH513100"),
        lookback_days=2,
        moving_average_days=2,
        top_n=1,
    )

    assert allocations[0].signal_date == date(2026, 1, 3)
    assert allocations[0].target_weights == {"SZ159915": 1.0}
    assert allocations[-1].target_weights == {"SH513100": 1.0}
    assert "2-day momentum" in str(allocations[-1].reason)


def test_momentum_rotation_moves_to_cash_when_no_asset_is_above_its_average() -> None:
    allocations = build_momentum_rotation_allocations(
        bars=[
            *_bars("600519", [10, 9, 8, 7]),
            *_bars("000001", [10, 9, 8, 7]),
        ],
        symbol_order=("600519", "000001"),
        lookback_days=2,
        moving_average_days=2,
        top_n=1,
    )

    assert len(allocations) == 1
    assert allocations[0].target_weights == {}


def test_momentum_rotation_does_not_replace_a_failed_winner_with_second_place() -> None:
    allocations = build_momentum_rotation_allocations(
        bars=[
            *_bars("SZ159915", [10, 100, 20]),
            *_bars("SH513100", [10, 11, 12]),
        ],
        symbol_order=("SZ159915", "SH513100"),
        lookback_days=2,
        moving_average_days=3,
        top_n=1,
    )

    assert len(allocations) == 1
    assert allocations[0].target_weights == {}


def test_dual_moving_average_is_a_single_asset_long_or_cash_rule() -> None:
    allocations = build_dual_moving_average_allocations(
        bars=_bars("600519", [10, 10, 11, 12, 8, 7]),
        symbol="600519",
        fast_days=2,
        slow_days=3,
    )

    assert allocations[0].target_weights == {"600519": 1.0}
    assert allocations[-1].target_weights == {}
    with pytest.raises(ValueError, match="fast_days must be less than slow_days"):
        build_dual_moving_average_allocations(
            bars=_bars("600519", [10, 11, 12]),
            symbol="600519",
            fast_days=3,
            slow_days=3,
        )


def test_buy_and_hold_enters_on_the_first_requested_trading_day() -> None:
    allocations = build_buy_and_hold_allocations(
        bars=[
            *_bars("600519", [10, 11, 12]),
            *_bars("000001", [20, 21, 22]),
        ],
        symbols=("600519", "000001"),
        start_date=date(2026, 1, 2),
    )

    assert len(allocations) == 1
    assert allocations[0].signal_date == date(2026, 1, 1)
    assert allocations[0].target_weights == {"600519": 0.5, "000001": 0.5}
