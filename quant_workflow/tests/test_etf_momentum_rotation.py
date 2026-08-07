from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "etf_momentum_rotation.py"
    spec = importlib.util.spec_from_file_location("etf_momentum_rotation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "close": closes,
            "symbol": symbol,
        }
    )


def test_signal_uses_close_and_shifts_execution_to_next_day() -> None:
    module = load_module()
    first = [100.0] * 20 + [120.0, 121.0]
    second = [100.0] * 20 + [100.0, 100.0]
    table = module.build_signal_table(
        {"SZ159915": frame("SZ159915", first), "SH513100": frame("SH513100", second)},
        lookback_days=20,
        moving_average_days=20,
    )

    assert table.iloc[20]["signal"] == "SZ159915"
    assert table.iloc[20]["target_position"] == "CASH"
    assert table.iloc[21]["target_position"] == "SZ159915"


def test_cash_when_top_asset_is_below_its_moving_average() -> None:
    module = load_module()
    first = list(range(120, 100, -1)) + [100.0, 99.0]
    second = [100.0] * 22
    table = module.build_signal_table(
        {"SZ159915": frame("SZ159915", first), "SH513100": frame("SH513100", second)},
        lookback_days=20,
        moving_average_days=20,
    )

    assert table.iloc[20]["top_symbol"] == "SH513100"
    assert table.iloc[20]["signal"] == "CASH"


def test_signal_table_accepts_three_assets_with_filter_then_rank() -> None:
    module = load_module()
    first = [100.0] * 20 + [130.0]
    second = [100.0] * 20 + [120.0]
    third = [100.0] * 20 + [110.0]

    table = module.build_signal_table(
        {
            "SZ159915": frame("SZ159915", first),
            "SH513100": frame("SH513100", second),
            "SH512890": frame("SH512890", third),
        },
        lookback_days=20,
        moving_average_days=20,
        eligibility_mode="filter_then_rank",
    )

    assert table.iloc[20]["signal"] == "SZ159915"
    assert table.iloc[20]["SZ159915_rank"] == 1
    assert table.iloc[20]["SH513100_rank"] == 2
    assert table.iloc[20]["SH512890_rank"] == 3


def test_filter_then_rank_uses_best_eligible_asset_when_raw_leader_is_below_ma() -> None:
    module = load_module()
    # A large value 20 sessions ago lets the raw leader retain the best
    # 20-session return while its current close remains below its 20-day MA.
    raw_leader = [100.0] + [200.0] * 19 + [180.0]
    eligible_runner_up = [100.0] * 20 + [120.0]
    third = [100.0] * 20 + [105.0]
    data = {
        "SZ159915": frame("SZ159915", raw_leader),
        "SH513100": frame("SH513100", eligible_runner_up),
        "SH512890": frame("SH512890", third),
    }

    legacy = module.build_signal_table(
        data,
        lookback_days=20,
        moving_average_days=20,
    )
    defensive = module.build_signal_table(
        data,
        lookback_days=20,
        moving_average_days=20,
        eligibility_mode="filter_then_rank",
    )

    assert legacy.iloc[20]["top_symbol"] == "SZ159915"
    assert legacy.iloc[20]["signal"] == "CASH"
    assert defensive.iloc[20]["top_symbol"] == "SZ159915"
    assert defensive.iloc[20]["signal"] == "SH513100"


def test_filter_then_rank_ties_follow_caller_symbol_order() -> None:
    module = load_module()
    closes = [100.0] * 20 + [120.0]
    data = {
        "SH512890": frame("SH512890", closes),
        "SZ159915": frame("SZ159915", closes),
        "SH513100": frame("SH513100", closes),
    }

    table = module.build_signal_table(
        data,
        lookback_days=20,
        moving_average_days=20,
        eligibility_mode="filter_then_rank",
    )

    assert table.iloc[20]["signal"] == "SH512890"


def test_switch_pays_two_sides_and_cost_reduces_wealth() -> None:
    module = load_module()
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    table = pd.DataFrame(
        {
            "date": dates,
            "target_position": ["CASH", "SZ159915", "SH513100", "CASH"],
            "SZ159915_open": [100.0, 100.0, 100.0, 100.0],
            "SH513100_open": [100.0, 100.0, 100.0, 100.0],
        }
    )
    result = module.simulate_open_to_open(table, side_cost_bps=10.0)

    # The final row has no following open, so its CASH target is not executed
    # inside the measurable open-to-open periods.
    assert result["metrics"]["switch_count"] == 2
    assert result["metrics"]["total_turnover_sides"] == 3.0
    assert result["metrics"]["total_return"] == pytest.approx(
        (1.0 - 0.001) * (1.0 - 0.002) - 1.0
    )


def test_duplicate_dates_are_rejected() -> None:
    module = load_module()
    data = frame("SZ159915", [100.0, 101.0])
    data.loc[1, "date"] = data.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicate dates"):
        module.normalize_price_frame(data, symbol="SZ159915")


def test_json_sanitizer_preserves_boolean_values() -> None:
    module = load_module()
    assert module.sanitize_json_value(True) is True
    assert module.sanitize_json_value(False) is False


def test_etf_research_drawdown_gate_is_explicit_and_separate_from_robust_gate() -> None:
    module = load_module()
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "public_strategy_candidates.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in payload["candidates"]
        if item["id"] == "etf_20d_two_asset_momentum_rotation"
    )

    assert module.ETF_RESEARCH_MAX_DRAWDOWN_ABS == pytest.approx(0.35)
    assert candidate["acceptance"]["max_drawdown_abs_max"] == pytest.approx(
        module.ETF_RESEARCH_MAX_DRAWDOWN_ABS
    )
    assert "CSI800" in candidate["acceptance"]["note"]
