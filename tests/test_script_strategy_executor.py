# -*- coding: utf-8 -*-
"""Process-isolation tests for user supplied strategy scripts."""

from datetime import date

import pytest

from src.core.script_strategy import ScriptExecutionError
from src.core.strategy_backtest import PriceBar
from src.services.script_strategy_executor import run_script_in_sandbox


_BARS = (
    PriceBar(
        symbol="SZ159915",
        date=date(2026, 7, 24),
        open=100.0,
        close=101.0,
    ),
)


def test_sandbox_returns_checked_allocations() -> None:
    allocations, logs, check = run_script_in_sandbox(
        "def on_bar(context, data):\n    context.target_equal(data.symbols)",
        bars=_BARS,
        symbols=("SZ159915",),
        start_date=None,
        end_date=None,
        timeout_seconds=2,
    )

    assert len(allocations) == 1
    assert allocations[0].target_weights == {"SZ159915": 1.0}
    assert logs == []
    assert check["status"] == "passed"


def test_sandbox_terminates_an_infinite_strategy() -> None:
    with pytest.raises(ScriptExecutionError, match="exceeded"):
        run_script_in_sandbox(
            "def on_bar(context, data):\n    while True:\n        pass",
            bars=_BARS,
            symbols=("SZ159915",),
            start_date=None,
            end_date=None,
            timeout_seconds=0.2,
        )
