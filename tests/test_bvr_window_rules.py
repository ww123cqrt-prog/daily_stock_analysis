"""BVR's bounded short-suspension proposal uses the existing market-data API."""
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


@pytest.mark.parametrize('missing', [0, 1, 4, 5])
def test_only_21_or_more_valid_prices_in_25_market_dates_can_setup(missing):
    context, data, requests = scenario(missing)
    symbol = context.symbols[0]
    assert len(data.history(symbol, 25)) == 25 - missing
    bvr.on_bar(context, data)
    assert bool(context.observations) is (missing <= 4)
    data._index = 50
    bvr.on_bar(context, data)
    assert requests == ([{}, {symbol: 0.5}] if missing <= 4 else [])


def test_touch_outside_five_market_dates_is_not_reused():
    context, data, requests = scenario(touch_ago=6)
    bvr.on_bar(context, data)
    assert not context.observations
    assert not requests


def test_observed_touch_expires_before_late_confirmation():
    context, data, requests = scenario(touch_ago=5)
    bvr.on_bar(context, data)
    assert context.observations
    data._index = 50
    bvr.on_bar(context, data)
    assert not requests
    assert not context.observations


def test_suspension_cannot_issue_entry_or_extend_observation():
    context, data, requests = scenario(missing=1)
    bvr.on_bar(context, data)
    assert context.observations
    data._view.close[50:56, 0] = np.nan
    data._view.volume[50:56, 0] = np.nan
    for index in range(50, 57):
        data._index = index
        bvr.on_bar(context, data)
    assert not requests
    assert not context.observations


def test_boll_keeps_twenty_valid_samples_and_does_not_look_ahead():
    context, data, _ = scenario(missing=4)
    symbol = context.symbols[0]
    prices = data.history(symbol, 25)[-20:]
    bands = bvr._bands(data, symbol, b_mode=True)
    assert bands['middle'] == pytest.approx(sum(prices) / 20)
    assert bands['upper'] == pytest.approx(np.mean(prices) + 2 * np.std(prices, ddof=1))
    assert bvr._bands(data, symbol) is None  # Mode A's original strict window.
    data._view.close[50:, 0] = 999999
    assert bvr._bands(data, symbol, b_mode=True) == bands


def test_historical_touch_needs_twenty_valid_prices_at_that_time():
    context, data, requests = scenario(missing=4, touch_ago=5)
    data._view.close[:25, 0] = np.nan
    bvr.on_bar(context, data)
    assert not context.observations
    assert not requests


def test_touch_boll_can_use_earlier_prices_but_not_future_prices():
    context, data, _ = scenario(missing=4, touch_ago=5)
    symbol = context.symbols[0]
    window = bvr._b_values(data, symbol, ago=5)
    assert len(window) == 20
    assert window[-1] == 12
    assert bvr._mode_b_pullback(data, symbol, bvr._bands(data, symbol, b_mode=True)) == (12, 5)
    assert len(data.history(symbol, 25)) == 21  # The shared API is unchanged.
