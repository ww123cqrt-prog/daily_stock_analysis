"""Only tolerate 21–24 closes; preserve the original BVR decision rules."""
from types import SimpleNamespace
from datetime import date, timedelta

import numpy as np
import pytest

from src.core.script_market import MarketView
from src.core.script_strategy import _StrategyData
from quant_workflow.scripts import bvr20_fill_feedback as bvr


def scenario(missing=0, touch_ago=1):
    requests = []
    context = SimpleNamespace(current_weights={}, target_cash=lambda: requests.append({}),
                              target_percent=lambda symbol, weight: requests.append({symbol: weight}))
    bvr.initialize(context)
    context.symbols = context.stock_pool
    close = np.full((57, len(context.symbols)), np.nan)
    close[:, 0] = 10
    close[49 - touch_ago, 0] = 12
    close[49, 0] = 10.5
    close[50:, 0] = 12.5
    volume = np.full_like(close, np.nan)
    volume[:, 0] = 100
    for ago in range(6, 6 + missing):
        close[49 - ago, 0] = volume[49 - ago, 0] = np.nan
    view = MarketView(
        dates=np.array([(date(2024, 1, 2) + timedelta(days=i)).toordinal() for i in range(57)]),
        symbols=np.array(context.symbols), open_matrix=close.copy(), close_matrix=close,
        members=np.isfinite(close), volume_matrix=volume,
        index_close=np.arange(100, 157, dtype=float), meta={},
    )
    data = _StrategyData(view)
    data._index = 49
    return context, data, requests


def legacy_pullback(data, symbol, bands):
    if bands is None or bvr._strong_streak(data) < 3 or bands["close"] < bands["middle"] or bands["close"] >= bands["upper"]:
        return None
    closes = data.history(symbol, 25, "close")
    if len(closes) < 25:
        return None
    touches = []
    for index in range(19, 24):
        window = closes[index - 19:index + 1]
        deviation = bvr._std(window)
        if deviation is not None and closes[index] >= bvr._mean(window) + 2.0 * deviation:
            touches.append(closes[index])
    return touches[-1] if touches else None


@pytest.mark.parametrize('missing', [0, 1, 2, 3, 4, 5])
def test_only_short_history_gate_changes(missing):
    context, data, requests = scenario()
    for ago in [21, 22, 23, 24, 20][:missing]:
        data._view.close[49 - ago, 0] = np.nan
        data._view.volume[49 - ago, 0] = np.nan
    symbol = context.symbols[0]
    assert len(data.history(symbol, 25)) == 25 - missing
    bvr.on_bar(context, data)
    assert bool(context.observations) is (missing <= 4)
    if missing <= 4:
        assert context.observations[symbol] == {"touch_close": 12, "days_left": 3}
    assert "market_bar" not in vars(context)
    data._index = 50
    bvr.on_bar(context, data)
    assert requests == ([{}, {symbol: 0.5}] if missing <= 4 else [])


def test_recent_missing_prices_only_change_setup_not_confirmation_or_exit():
    context, data, requests = scenario(missing=4)
    symbol = context.symbols[0]
    assert bvr._bands(data, symbol) is None
    bvr.on_bar(context, data)
    assert context.observations[symbol] == {"touch_close": 12, "days_left": 3}
    data._index = 50
    bvr.on_bar(context, data)
    assert requests == []  # Original confirmation still requires its original bands.
    assert bvr._bands(data, symbol) is None


def test_original_waiting_allows_confirmation_after_fifth_touch_day():
    context, data, requests = scenario(touch_ago=5)
    symbol = context.symbols[0]
    bvr.on_bar(context, data)
    assert context.observations[symbol] == {"touch_close": 12, "days_left": 3}
    data._index = 50
    bvr.on_bar(context, data)
    assert requests == [{}, {symbol: 0.5}]
    assert "market_bar" not in vars(context)


def test_complete_history_matches_original_pullback_for_varied_prices():
    context, data, _ = scenario()
    symbol = context.symbols[0]
    random = np.random.default_rng(20261031)
    for _ in range(300):
        data._view.close[:50, 0] = 10 + random.normal(0, 1, 50)
        bands = bvr._bands(data, symbol)
        assert bvr._mode_b_pullback(data, symbol, bands) == legacy_pullback(data, symbol, bands)


def test_complete_closes_do_not_bypass_original_volume_requirement():
    context, data, requests = scenario()
    data._view.volume[48, 0] = np.nan
    assert len(data.history(context.symbols[0], 25)) == 25
    bvr.on_bar(context, data)
    assert not requests
    assert not context.observations


def test_short_history_uses_twenty_sample_boll_not_shorter_windows():
    context, data, _ = scenario(missing=4)
    prices = data.history(context.symbols[0], 25)
    assert len(prices) == 21
    expected = 12 if prices[-2] >= np.mean(prices[:-1]) + 2 * np.std(prices[:-1], ddof=1) else None
    assert bvr._mode_b_pullback(data, context.symbols[0], None) == expected
    data._view.close[50:, 0] = 999999
    assert bvr._mode_b_pullback(data, context.symbols[0], None) == expected
