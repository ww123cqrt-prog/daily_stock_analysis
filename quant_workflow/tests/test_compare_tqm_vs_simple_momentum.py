"""The comparison's custom score must retain its return/drawdown meaning."""

import json

import pandas as pd
import pytest

from quant_workflow.scripts import compare_tqm_vs_simple_momentum as comparison


def _synthetic_prices() -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-05", periods=7),
        "SZ159915_open": [100, 100, 105, 95, 100, 110, 105],
        "SH513100_open": [100, 100, 100, 100, 100, 90, 99],
        "SH512890_open": [100] * 7,
    })
    for symbol in comparison.SYMBOLS:
        frame[f"{symbol}_close"] = frame[f"{symbol}_open"]
    return frame


def test_simulate_preserves_returns_and_curve_under_custom_score_name() -> None:
    frame = _synthetic_prices()
    signals = [
        "SZ159915", "SZ159915", comparison.CASH, "SH513100",
        "SH513100", comparison.CASH, comparison.CASH,
    ]

    result = comparison.simulate(frame, signals)

    # Hand-derived path includes 20 bps on entry, exit and re-entry, plus a loss.
    assert result["equity_curve"]["wealth"].tolist() == pytest.approx([
        1.0, 1.0, 1.0479, 0.9481, 0.9462038, 0.84988025316, 0.934868278476,
    ], rel=1e-13, abs=1e-14)
    pd.testing.assert_series_equal(result["equity_curve"]["date"], frame["date"])
    assert {key: value for key, value in result.items() if key != "equity_curve"} == pytest.approx({
        "total_return": -0.065131721524,
        "annualized_return": -0.9409094819559611,
        "max_drawdown": -0.18896817142857147,
        "switch_count": 3,
        "invested_fraction": 2 / 3,
        "custom_return_drawdown_score": -4.728944711107941,
    }, rel=1e-13, abs=1e-14)


def test_main_reports_and_saves_custom_return_drawdown_score(monkeypatch, tmp_path, capsys) -> None:
    frame = _synthetic_prices()
    data = {
        symbol: frame[["date", f"{symbol}_open", f"{symbol}_close"]].rename(columns={
            f"{symbol}_open": "open", f"{symbol}_close": "close",
        })
        for symbol in comparison.SYMBOLS
    }
    # Replace only the external snapshot read; signal generation and simulation stay real.
    monkeypatch.setattr(comparison, "load_data", lambda: data)
    monkeypatch.setattr(comparison, "SCRIPT_DIR", tmp_path)

    comparison.main()

    output = capsys.readouterr().out
    assert "自定义收益回撤评分" in output
    assert "年化收益 / (最大回撤绝对值 + 0.01)" in output
    assert "结论（按自定义收益回撤评分）:" in output
    assert "夏普" not in output
    assert "风险调整后收益" not in output
    paths = list((tmp_path / "backtest_results").glob("tqm_vs_simple_momentum_*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text(encoding="utf-8"))
    # Seven days cannot satisfy the 20-day lookback: both real strategies stay in cash.
    for strategy in ("simple_momentum", "tqm_factor"):
        assert saved[strategy] == {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "switch_count": 0,
            "invested_fraction": 0.0,
            "custom_return_drawdown_score": 0.0,
        }
