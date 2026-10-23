# -*- coding: utf-8 -*-
"""Process-isolation tests for user supplied strategy scripts."""

from datetime import date
import platform

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


@pytest.mark.skipif(platform.system() != "Darwin", reason="Requires the native macOS sandbox-exec")
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


@pytest.mark.skipif(platform.system() != "Darwin", reason="Requires the native macOS sandbox-exec")
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


@pytest.mark.parametrize("host", ["Linux", "Windows"])
def test_refuses_execution_outside_macos_sandbox(monkeypatch, host: str) -> None:
    """B-1: non-Darwin hosts must never silently run bare subprocesses."""
    import platform as platform_module

    from src.core.script_strategy import ScriptExecutionError
    from src.services.script_strategy_executor import run_script_in_sandbox

    monkeypatch.setattr(platform_module, "system", lambda: host)

    def reject_process_launch(*_args, **_kwargs):
        pytest.fail("unsupported platforms must not launch a strategy process")

    monkeypatch.setattr("src.services.script_strategy_executor.subprocess.run", reject_process_launch)
    script = (
        "def initialize(context):\n"
        "    pass\n"
        "\n"
        "def on_bar(context, data):\n"
        "    pass\n"
    )
    with pytest.raises(ScriptExecutionError, match="not supported"):
        run_script_in_sandbox(
            script,
            bundle_dir=None,
            bars=list(_BARS),
            symbols=["SZ159915"],
            start_date=date(2026, 7, 24),
            end_date=date(2026, 7, 24),
        )
