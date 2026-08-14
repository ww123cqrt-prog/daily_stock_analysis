# -*- coding: utf-8 -*-
"""Tests for the bounded DSA Python strategy API."""

from datetime import date

import pytest

from src.core.script_strategy import (
    ScriptExecutionError,
    ScriptValidationError,
    _StrategyContext,
    execute_strategy_script,
    validate_strategy_script,
)
from src.core.strategy_backtest import PriceBar


SCRIPT = '''
def initialize(context):
    context.lookback = 2
    context.ma_days = 2

def on_bar(context, data):
    ranked = data.rank_by_return(context.lookback)
    if not ranked:
        context.target_cash()
        return
    best = ranked[0]
    average = data.sma(best, context.ma_days)
    if average is not None and data.close(best) > average:
        context.target_equal([best])
    else:
        context.target_cash()
'''


def _bars(symbol: str, closes: list[float]) -> list[PriceBar]:
    return [
        PriceBar(symbol=symbol, date=date(2026, 1, index), open=value, close=value)
        for index, value in enumerate(closes, start=1)
    ]


def test_script_generates_target_changes_without_future_access() -> None:
    result = execute_strategy_script(
        SCRIPT,
        bars=[
            *_bars("SZ159915", [10, 11, 12, 11, 10]),
            *_bars("SH513100", [10, 10, 10, 12, 14]),
        ],
        symbols=("SZ159915", "SH513100"),
    )

    assert result.allocations[0].target_weights == {"SZ159915": 1.0}
    assert result.allocations[-1].target_weights == {"SH513100": 1.0}
    assert result.check["status"] == "passed"


@pytest.mark.parametrize(
    "source, message",
    [
        ("import os\ndef on_bar(context, data):\n    pass", "imports are not allowed"),
        ("def on_bar(context, data):\n    open('/tmp/x', 'w')", "open is not allowed"),
        ("def on_bar(context, data):\n    return data.__class__", "private attributes are not allowed"),
    ],
)
def test_static_check_rejects_unsafe_python(source: str, message: str) -> None:
    with pytest.raises(ScriptValidationError, match=message):
        validate_strategy_script(source)


def test_script_cannot_request_future_history() -> None:
    source = '''
def on_bar(context, data):
    data.close(context.symbols[0], ago=-1)
'''
    with pytest.raises(ValueError, match="ago must be non-negative"):
        execute_strategy_script(
            source,
            bars=_bars("600519", [10, 11, 12]),
            symbols=("600519",),
        )


import numpy as np

from src.core.script_market import MarketView
from src.core.script_strategy import _StrategyData


def _data_view(closes: list[float | None]) -> _StrategyData:
    close_matrix = np.array(
        [[value if value is not None else np.nan] for value in closes],
        dtype=np.float64,
    )
    view = MarketView(
        dates=np.array(
            [date(2026, 1, day).toordinal() for day in range(1, len(closes) + 1)],
            dtype=np.int32,
        ),
        symbols=np.array(["SZ000001"], dtype=object),
        open_matrix=close_matrix.copy(),
        close_matrix=close_matrix,
        members=~np.isnan(close_matrix),
        meta={},
    )
    data = _StrategyData(view)
    data._index = len(closes) - 1
    return data


def test_sma_requires_a_full_window_of_valid_closes() -> None:
    """C-5: 2 valid closes must not fabricate a "3-day average"."""
    data = _data_view([10.0, None, 12.0])
    assert data.sma("SZ000001", 3) is None
    full = _data_view([10.0, 11.0, 12.0])
    assert full.sma("SZ000001", 3) == pytest.approx(11.0)


def test_sma_returns_none_when_history_shorter_than_window() -> None:
    data = _data_view([10.0, 11.0])
    assert data.sma("SZ000001", 3) is None


def _pit_view() -> MarketView:
    close_matrix = np.array(
        [[10.0, 10.0], [11.0, 11.0]],
        dtype=np.float64,
    )
    return MarketView(
        dates=np.array(
            [date(2026, 1, day).toordinal() for day in range(1, 3)],
            dtype=np.int32,
        ),
        symbols=np.array(["OLD", "FUTURE"], dtype=object),
        open_matrix=close_matrix.copy(),
        close_matrix=close_matrix,
        members=np.array(
            [[True, False], [True, True]], dtype=bool
        ),  # FUTURE enters the pool on day 2
        meta={"pool": "csi800"},
    )


def test_pit_pool_blocks_access_to_non_member_on_current_date() -> None:
    """P1: a script must not read or target a symbol before it enters the
    PIT pool (look-ahead bias guard)."""
    view = _pit_view()
    data = _StrategyData(view)
    data._index = 0  # day 1: FUTURE not yet a member
    with pytest.raises(ValueError, match="not a tradable member"):
        data.close("FUTURE")
    context = _StrategyContext(("OLD", "FUTURE"), [])
    context._begin_bar(date(2026, 1, 1), {}, {"OLD"})
    with pytest.raises(ValueError, match="not a tradable member"):
        context.target_equal(["FUTURE"])
    with pytest.raises(ValueError, match="not a tradable member"):
        context.target_percent("FUTURE", 1.0)
    # Day 2: FUTURE is a member — access works.
    data._index = 1
    assert data.close("FUTURE") == pytest.approx(11.0)


def test_pit_pool_guards_only_apply_to_pit_pools() -> None:
    """ETF/custom pools keep their permissive behavior (suspended symbols
    stay targetable; the engine defers execution instead)."""
    view = _pit_view()
    view.meta = {"pool": "etf"}
    data = _StrategyData(view)
    data._index = 0
    assert data.close("FUTURE") == pytest.approx(10.0)


def test_statistics_facade_does_not_expose_sys_module() -> None:
    """P1: statistics.sys must not be a route to builtins.open."""
    view = _data_view([10.0, 11.0])._view
    with pytest.raises(ScriptExecutionError, match="AttributeError"):
        execute_strategy_script(
            "def initialize(context):\n"
            "    pass\n"
            "\n"
            "def on_bar(context, data):\n"
            "    context.log(str(statistics.sys))\n",
            view=view,
        )
