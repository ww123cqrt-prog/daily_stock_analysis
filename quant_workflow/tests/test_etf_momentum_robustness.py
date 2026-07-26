from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "etf_momentum_robustness.yaml"
)


def load_module(name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {"date": dates, "open": closes, "close": closes, "symbol": symbol}
    )


def test_apply_execution_lag_uses_signal_only_and_preserves_cash() -> None:
    module = load_module("etf_momentum_rotation", "etf_momentum_rotation.py")
    table = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=4, freq="B"),
            "signal": ["CASH", "SZ159915", "SH513100", "CASH"],
            "target_position": ["CASH", "CASH", "SZ159915", "SH513100"],
        }
    )

    delayed = module.with_execution_lag(table, lag_days=2)

    assert delayed["target_position"].tolist() == ["CASH", "CASH", "CASH", "SZ159915"]
    assert table["target_position"].tolist() == ["CASH", "CASH", "SZ159915", "SH513100"]


def test_summarize_periods_excludes_cross_boundary_period() -> None:
    module = load_module("etf_momentum_rotation", "etf_momentum_rotation.py")
    periods = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31", "2025-04-01", "2025-04-02"]),
            "next_date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03"]),
            "net_return": [0.10, 0.05, -0.02],
            "position": ["SZ159915", "SZ159915", "CASH"],
            "transaction_cost_rate": [0.0, 0.0, 0.0],
            "wealth": [1.1, 1.155, 1.1319],
        }
    )

    result = module.summarize_periods(
        periods,
        start_date="2025-04-01",
        end_date="2025-04-02",
    )

    assert result["periods"] == 1
    assert result["total_return"] == pytest.approx(0.05)
    assert result["start_date"] == "2025-04-01"
    assert result["end_date"] == "2025-04-02"


def test_robustness_report_marks_parameter_neighbors_as_diagnostics() -> None:
    module = load_module(
        "evaluate_etf_momentum_robustness", "evaluate_etf_momentum_robustness.py"
    )
    report = module.evaluate_parameter_grid(
        {
            "SZ159915": frame("SZ159915", [100.0] * 30),
            "SH513100": frame("SH513100", [100.0] * 30),
        },
        lookbacks=[20],
        moving_averages=[20],
        side_cost_bps=7.5,
    )

    assert report["primary"] == {"lookback_days": 20, "moving_average_days": 20}
    assert report["selection_allowed"] is False
    assert len(report["variants"]) == 1


def test_cash_metrics_reports_indicator_warmup_separately_from_active_cash() -> None:
    module = load_module(
        "evaluate_etf_momentum_robustness", "evaluate_etf_momentum_robustness.py"
    )
    closes = [100.0] * 21 + [90.0, 91.0]
    data = {
        "SZ159915": frame("SZ159915", closes),
        "SH513100": frame("SH513100", [100.0] * len(closes)),
    }
    table = load_module(
        "etf_momentum_rotation", "etf_momentum_rotation.py"
    ).build_signal_table(data)

    result = module.cash_metrics(table)

    assert result["warmup_rows_excluded"] == 20
    assert result["signal_cash"]["cash_days"] >= 1


def test_complete_quarters_exclude_first_and_last_partial_quarters() -> None:
    module = load_module(
        "evaluate_etf_momentum_robustness", "evaluate_etf_momentum_robustness.py"
    )
    table = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-30", "2025-07-24", freq="B"),
        }
    )

    assert module.complete_quarter_labels(table) == ["2025Q2"]


def test_price_discontinuity_is_reported_as_unresolved_evidence() -> None:
    module = load_module(
        "evaluate_etf_momentum_robustness", "evaluate_etf_momentum_robustness.py"
    )
    prices = {
        "SH513100": frame("SH513100", [5.0, 5.1, 1.0, 1.02]),
        "SZ159915": frame("SZ159915", [2.0, 2.1, 2.0, 2.1]),
    }

    report = module.price_discontinuity_report(prices, absolute_return_threshold=0.30)

    assert report["passed"] is False
    assert report["events"][0]["symbol"] == "SH513100"
    assert report["events"][0]["close_return"] < -0.80


def test_robustness_decision_is_manual_and_goal1_cost_stress_is_preserved() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["policy"]["final_decision"] == "manual_review"
    assert config["policy"]["automatic_rejection_disabled"] is True
    assert config["policy"]["reference_lines_are_not_vetoes"] is True
    assert config["evaluation"]["costs_total_bps"]["baseline"] == 20
    assert config["evaluation"]["costs_total_bps"]["stress"] == 40
    assert config["evaluation"]["earlier_comparison_total_bps"] == 15


def test_manual_assessment_lists_every_unmet_reference_line() -> None:
    module = load_module(
        "evaluate_etf_momentum_robustness", "evaluate_etf_momentum_robustness.py"
    )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    metric = lambda annualized, drawdown=0.10: {  # noqa: E731
        "metrics": {"annualized_return": annualized, "max_drawdown": -drawdown}
    }
    result = module._acceptance(
        baseline=metric(0.04, 0.36),
        stress=metric(-0.01),
        delayed=metric(-0.02),
        quarters=[{"positive": True}, {"positive": False}],
        parameter_grid={
            "variants": [
                {"metrics": {"annualized_return": 0.01, "max_drawdown": -0.20}},
                {"metrics": {"annualized_return": -0.01, "max_drawdown": -0.40}},
            ]
        },
        config=config,
    )

    assert result["status"] == "manual_review_required"
    assert result["automatic_rejection_disabled"] is True
    assert set(result["advisory_not_met"]) == set(result["advisory_checks"])
