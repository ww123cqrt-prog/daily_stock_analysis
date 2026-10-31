"""Actual-fill feedback must precede the next close signal, never a target."""
from datetime import date, timedelta
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.script_market import MarketView
from src.core.script_strategy import execute_strategy_script
from src.core.strategy_backtest import StrategyBacktestConfig


SCRIPT = '''
def initialize(context):
    context.entry_price = 0
    context.last_buy_price = 0
    context.add_count = 0

def on_fill(context, trade):
    if trade["action"] == "buy":
        if context.entry_price == 0:
            context.entry_price = trade["price"]
        else:
            context.add_count += 1
        context.last_buy_price = trade["price"]
        context.log(("fill", context.add_count, context.last_buy_price, context.positions["A"]))

def on_bar(context, data):
    context.log(("bar", context.add_count, context.last_buy_price))
    if not context.positions or data.close("A") <= context.last_buy_price * 0.9:
        context.target_percent("A", 0.75)
'''


def view(opens, closes):
    op = np.array(opens, dtype=float)[:, None]
    cl = np.array(closes, dtype=float)[:, None]
    return MarketView(
        dates=np.array([(date(2026, 1, 5)+timedelta(days=i)).toordinal() for i in range(len(op))]),
        symbols=np.array(["A"], dtype=object), open_matrix=op, close_matrix=cl,
        members=np.isfinite(cl), meta={},
    )


def run(source=SCRIPT, *, opens=(10, 10, 8, 8, 4), closes=(10, 10, 8, 6, 4)):
    return execute_strategy_script(source, view=view(opens, closes),
        start_date=date(2026, 1, 5), end_date=date(2026, 1, 9),
        execution_config=StrategyBacktestConfig(initial_capital=3000, lot_size=100, side_cost_pct=0))


def test_zero_lot_does_not_count_as_add_and_later_fill_uses_actual_price():
    result = run()
    fills = [ast.literal_eval(log.split(' ', 1)[1]) for log in result.logs if "('fill'" in log]
    assert fills == [("fill", 0, 10.0, 200), ("fill", 1, 4.0, 300)]
    trades = result.backtest_report["trades"]
    assert [(t["execution_date"], t["shares"], t["price"]) for t in trades] == [
        ("2026-01-06", 200, 10.0), ("2026-01-09", 100, 4.0)]
    assert result.backtest_report["metrics"]["final_equity"] == pytest.approx(1800)


def test_suspended_order_only_emits_fill_after_real_execution():
    result = run(opens=(10, np.nan, 8, 8, 8), closes=(10, 10, 8, 8, 8))
    fills = [ast.literal_eval(log.split(' ', 1)[1]) for log in result.logs if "('fill'" in log]
    assert fills == [("fill", 0, 8.0, 200)]
    assert result.backtest_report["trades"][0]["execution_date"] == "2026-01-07"


def test_no_initial_fill_does_not_create_entry_state():
    result = run(opens=(10000,)*5, closes=(10000,)*5)
    assert not result.backtest_report["trades"]
    assert not any("('fill'" in log for log in result.logs)


def test_fill_record_and_positions_are_read_only():
    source = SCRIPT.replace('if trade["action"] == "buy":', 'trade["price"] = 1\n    if trade["action"] == "buy":')
    with pytest.raises(ValueError, match="does not support item assignment"):
        run(source)


def test_actual_positions_are_read_only():
    source = SCRIPT.replace('if trade["action"] == "buy":', 'context.positions["A"] = 0\n    if trade["action"] == "buy":')
    with pytest.raises(ValueError, match="does not support item assignment"):
        run(source)


def test_fill_aware_script_refuses_signal_only_execution():
    with pytest.raises(ValueError, match="requires execution_config"):
        execute_strategy_script(SCRIPT, view=view((10,)*5, (10,)*5))


def test_only_last_day_signal_is_reported_as_unexecuted():
    source = '''
def on_fill(context, trade):
    pass
def on_bar(context, data):
    if context.current_date.day == 9:
        context.target_percent("A", 0.75)
'''
    result = run(source)
    assert result.backtest_report["status"] == "no_executable_signals"
    assert result.backtest_report["metrics"]["skipped_signal_count"] == 1


def test_fill_aware_subwindow_retries_suspension_before_window():
    from src.core.strategy_backtest import run_portfolio_backtest_view, TargetAllocation
    v = view((10, np.nan, 8, 10, 10), (10, 10, 8, 10, 10))
    report = run_portfolio_backtest_view(strategy_name="feedback", view=v,
        allocations=[TargetAllocation("entry", date(2026,1,5), {"A":0.75})],
        config=StrategyBacktestConfig(initial_capital=3000,lot_size=100,side_cost_pct=0),
        period_start=date(2026,1,9),reapply_targets=True)
    assert report["trades"][0]["execution_date"] == "2026-01-07"
    assert report["metrics"]["trade_count"] == 0


def test_partial_sell_preserves_entry_until_position_is_flat():
    source = '''
def initialize(context):
    context.started = False
def on_fill(context, trade):
    context.log((trade["action"], context.positions.get("A", 0)))
def on_bar(context, data):
    if context.current_date.day == 5:
        context.target_percent("A", 0.75)
    elif context.current_date.day == 6:
        context.target_percent("A", 0.34)
    elif context.current_date.day == 7:
        context.target_cash()
'''
    result = run(source, opens=(10,)*5, closes=(10,)*5)
    events = [ast.literal_eval(log.split(' ', 1)[1]) for log in result.logs]
    assert events == [("buy", 200), ("sell", 100), ("sell", 0)]


def test_bvr_updates_anchor_only_for_confirmed_add_and_keeps_partial_sale():
    from quant_workflow.scripts.bvr20_fill_feedback import on_fill
    context = SimpleNamespace(states={"A": dict(mode="A", first_entry_price=10, last_buy_price=10, add_count=0)},
        pending_adds={"A": "2026-01-07"}, pending_entries={}, pending_exits=set(),
        positions={"A": 300}, log=lambda value: None)
    on_fill(context, dict(symbol="A", action="buy", signal_date="2026-01-07", price=8, shares=100))
    assert context.states["A"]["add_count"] == 1
    assert context.states["A"]["last_buy_price"] == 8
    assert context.states["A"]["first_entry_price"] == 10
    context.positions = {"A": 200}
    on_fill(context, dict(symbol="A", action="sell", signal_date="2026-01-08", price=7, shares=100))
    assert context.states["A"]["add_count"] == 1
    context.positions = {}
    on_fill(context, dict(symbol="A", action="sell", signal_date="2026-01-09", price=7, shares=200))
    assert "A" not in context.states


def test_real_sandbox_and_service_use_actual_fill_feedback(tmp_path, monkeypatch):
    from src.core.script_market import MarketBundle
    from src.services.script_strategy_executor import run_script_backtest_in_sandbox
    from src.services.backtest_service import BacktestService
    v = view((10, 10, 8, 8, 4), (10, 10, 8, 6, 4))
    bundle = tmp_path / "bundle"
    MarketBundle(dates=v.dates, symbols=v.symbols, open=v.open, close=v.close, members=v.members, meta={}).save(bundle)
    config = StrategyBacktestConfig(initial_capital=3000, side_cost_pct=0, lot_size=100)
    report, allocations, logs, check = run_script_backtest_in_sandbox(SCRIPT, bundle_dir=bundle,
        start_date=date(2026,1,5), end_date=date(2026,1,9), config=config)
    assert report["metrics"]["final_equity"] == 1800
    assert len(report["trades"]) == 2 and check["fill_feedback"]
    monkeypatch.setattr("src.services.backtest_service.load_market_view", lambda *a, **k: (v, {"bundle_dir":str(bundle)}))
    monkeypatch.setattr("src.services.backtest_service.load_benchmark_series", lambda *a, **k: {})
    monkeypatch.setattr(BacktestService, "_run_strategy_comparisons", lambda *a, **k: ([], []))
    service = BacktestService.__new__(BacktestService)
    actual = service.run_script_strategy(script=SCRIPT, pool="custom", symbols=["A"],
        start_date=date(2026,1,5),end_date=date(2026,1,9), initial_capital=3000,side_cost_pct=0,lot_size=100)
    assert actual["trades"] == report["trades"]
    assert actual["equity_curve"] == report["equity_curve"]
    assert actual["assumptions"]["execution_feedback"] == "actual_fills"


def test_fill_aware_comparison_and_stability_keep_real_execution(tmp_path, monkeypatch):
    from src.core.script_market import MarketBundle
    from src.services.backtest_service import BacktestService
    v = view((10,)*100, (10,)*100)
    bundle = tmp_path / "bundle"
    MarketBundle(dates=v.dates, symbols=v.symbols, open=v.open, close=v.close, members=v.members, meta={}).save(bundle)
    monkeypatch.setattr("src.services.backtest_service.load_market_view", lambda *a, **k: (v, {"bundle_dir":str(bundle)}))
    monkeypatch.setattr("src.repositories.strategy_assets_repo.get_strategy_assets_repo", lambda: SimpleNamespace(get_strategy=lambda key: None))
    service = BacktestService.__new__(BacktestService)
    comparisons, failures = service._run_strategy_comparisons(strategy_ids=[], presets=[dict(name="fills",script=SCRIPT)],
        bundle_dir=bundle,view=v,config=StrategyBacktestConfig(initial_capital=3000,side_cost_pct=0,lot_size=100),
        start_date=v.calendar[0],end_date=v.calendar[-1])
    assert not failures
    assert comparisons[0]["report"]["assumptions"]["execution_feedback"] == "actual_fills"
    result = service.run_script_stability_test(script=SCRIPT,pool="custom",symbols=["A"],
        start_date=v.calendar[0],end_date=v.calendar[-1],initial_capital=3000,side_cost_pct=0,lot_size=100,
        iterations=2,min_window_days=20,seed=42)
    assert result["status"] == "completed" and len(result["samples"]) == 2
    assert all(sample["total_return_pct"] == 0 for sample in result["samples"])
