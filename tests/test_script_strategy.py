# -*- coding: utf-8 -*-
"""Tests for the bounded DSA Python strategy API."""

from datetime import date

import pytest

from src.core.script_strategy import (
    ScriptValidationError,
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
