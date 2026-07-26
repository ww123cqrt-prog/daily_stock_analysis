# -*- coding: utf-8 -*-
"""Explicit strategy rules that produce target allocations for the shared engine."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from src.core.strategy_backtest import PriceBar, TargetAllocation


def _common_history(
    bars: Sequence[PriceBar],
    symbols: Sequence[str],
) -> tuple[list[date], dict[str, dict[date, float]]]:
    ordered_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not ordered_symbols or any(not symbol for symbol in ordered_symbols):
        raise ValueError("at least one symbol is required")
    if len(set(ordered_symbols)) != len(ordered_symbols):
        raise ValueError("symbols must not contain duplicates")

    closes: dict[str, dict[date, float]] = {symbol: {} for symbol in ordered_symbols}
    for bar in bars:
        symbol = str(bar.symbol).strip().upper()
        if symbol not in closes:
            continue
        close = float(bar.close)
        if close <= 0:
            raise ValueError("close prices must be positive")
        if bar.date in closes[symbol]:
            raise ValueError(f"duplicate price bar: {symbol} {bar.date.isoformat()}")
        closes[symbol][bar.date] = close

    missing = [symbol for symbol, values in closes.items() if not values]
    if missing:
        raise ValueError(f"symbols have no price bars: {', '.join(missing)}")
    common_dates = set(closes[ordered_symbols[0]])
    for symbol in ordered_symbols[1:]:
        common_dates &= set(closes[symbol])
    calendar = sorted(common_dates)
    if not calendar:
        raise ValueError("symbols have no common trading dates")
    return calendar, closes


def _in_signal_period(
    signal_date: date,
    *,
    start_date: date | None,
    end_date: date | None,
) -> bool:
    return not (
        (start_date is not None and signal_date < start_date)
        or (end_date is not None and signal_date > end_date)
    )


def build_momentum_rotation_allocations(
    *,
    bars: Sequence[PriceBar],
    symbol_order: Sequence[str],
    lookback_days: int,
    moving_average_days: int,
    top_n: int = 1,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TargetAllocation]:
    """Select the strongest eligible symbols at T close for T+1 open execution."""

    symbols = tuple(str(symbol).strip().upper() for symbol in symbol_order)
    if len(symbols) < 2:
        raise ValueError("momentum_rotation requires at least two symbols")
    if lookback_days <= 0 or moving_average_days <= 0:
        raise ValueError("lookback_days and moving_average_days must be positive")
    if top_n <= 0 or top_n > len(symbols):
        raise ValueError("top_n must be between 1 and the number of symbols")

    calendar, closes = _common_history(bars, symbols)
    first_index = max(lookback_days, moving_average_days - 1)
    allocations: list[TargetAllocation] = []
    previous_target: dict[str, float] | None = None
    order_index = {symbol: index for index, symbol in enumerate(symbols)}

    for index in range(first_index, len(calendar)):
        signal_date = calendar[index]
        if not _in_signal_period(signal_date, start_date=start_date, end_date=end_date):
            continue
        ranked: list[tuple[str, float, bool]] = []
        for symbol in symbols:
            close = closes[symbol][signal_date]
            previous_close = closes[symbol][calendar[index - lookback_days]]
            moving_average = sum(
                closes[symbol][day]
                for day in calendar[index - moving_average_days + 1 : index + 1]
            ) / moving_average_days
            ranked.append(
                (symbol, close / previous_close - 1.0, close > moving_average)
            )
        ranked.sort(key=lambda item: (-item[1], order_index[item[0]]))
        selected = [item for item in ranked[:top_n] if item[2]]
        target = (
            {symbol: 1.0 / len(selected) for symbol, _, _ in selected}
            if selected
            else {}
        )
        if target == previous_target:
            continue
        detail = ", ".join(
            f"{symbol}={momentum:.4f}" for symbol, momentum, _ in selected
        )
        allocations.append(
            TargetAllocation(
                source_id=f"momentum_rotation:{signal_date.isoformat()}",
                signal_date=signal_date,
                target_weights=target,
                reason=(
                    f"Top {top_n} by {lookback_days}-day momentum and close above "
                    f"{moving_average_days}-day moving average"
                    + (f": {detail}" if detail else "; otherwise CASH")
                ),
            )
        )
        previous_target = target
    return allocations


def build_dual_moving_average_allocations(
    *,
    bars: Sequence[PriceBar],
    symbol: str,
    fast_days: int,
    slow_days: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TargetAllocation]:
    """Hold one asset when its fast close average is above its slow average."""

    normalized_symbol = str(symbol).strip().upper()
    if fast_days <= 0 or slow_days <= 0:
        raise ValueError("moving-average windows must be positive")
    if fast_days >= slow_days:
        raise ValueError("fast_days must be less than slow_days")
    calendar, closes = _common_history(bars, (normalized_symbol,))
    allocations: list[TargetAllocation] = []
    previous_target: dict[str, float] | None = None

    for index in range(slow_days - 1, len(calendar)):
        signal_date = calendar[index]
        if not _in_signal_period(signal_date, start_date=start_date, end_date=end_date):
            continue
        fast_average = sum(
            closes[normalized_symbol][day]
            for day in calendar[index - fast_days + 1 : index + 1]
        ) / fast_days
        slow_average = sum(
            closes[normalized_symbol][day]
            for day in calendar[index - slow_days + 1 : index + 1]
        ) / slow_days
        target = {normalized_symbol: 1.0} if fast_average > slow_average else {}
        if target == previous_target:
            continue
        allocations.append(
            TargetAllocation(
                source_id=f"dual_moving_average:{signal_date.isoformat()}",
                signal_date=signal_date,
                target_weights=target,
                reason=(
                    f"{fast_days}-day moving average {fast_average:.6f} "
                    f"{'above' if target else 'not above'} {slow_days}-day moving average "
                    f"{slow_average:.6f}"
                ),
            )
        )
        previous_target = target
    return allocations


def build_buy_and_hold_allocations(
    *,
    bars: Sequence[PriceBar],
    symbols: Sequence[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[TargetAllocation]:
    """Create one equal-weight allocation executed at the first requested open."""

    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    calendar, _ = _common_history(bars, normalized_symbols)
    execution_dates = [
        day
        for day in calendar
        if (start_date is None or day >= start_date) and (end_date is None or day <= end_date)
    ]
    if not execution_dates:
        return []
    execution_date = execution_dates[0]
    execution_index = calendar.index(execution_date)
    signal_date = (
        calendar[execution_index - 1]
        if execution_index > 0
        else execution_date - timedelta(days=1)
    )
    weight = 1.0 / len(normalized_symbols)
    return [
        TargetAllocation(
            source_id=f"buy_and_hold:{execution_date.isoformat()}",
            signal_date=signal_date,
            target_weights={symbol: weight for symbol in normalized_symbols},
            reason="Equal-weight buy and hold from the first requested trading-day open",
        )
    ]
