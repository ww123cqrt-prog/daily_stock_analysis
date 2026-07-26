from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research_etf_momentum_improvement.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "research_etf_momentum_improvement", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fee_rate_converts_wan_to_percent_and_basis_points() -> None:
    module = load_module()

    fee = module.normalize_commission_wan(0.574)

    assert fee == {
        "commission_wan_each_side": pytest.approx(0.574),
        "commission_bps_each_side": pytest.approx(0.574),
        "commission_pct_each_side": pytest.approx(0.00574),
        "commission_rate_each_side": pytest.approx(0.0000574),
        "round_trip_commission_bps": pytest.approx(1.148),
    }


def test_fee_percentage_display_keeps_broker_precision() -> None:
    module = load_module()

    assert module.format_fee_percentage(0.00574) == "0.00574%"


def test_time_splits_must_be_ordered_and_non_overlapping() -> None:
    module = load_module()

    valid = module.normalize_time_splits(
        {
            "development": {"start": "2019-01-02", "end": "2022-12-31"},
            "validation": {"start": "2023-01-01", "end": "2024-12-31"},
            "later_observation": {
                "start": "2025-01-01",
                "end": "2026-07-24",
            },
        }
    )

    assert [item["name"] for item in valid] == [
        "development",
        "validation",
        "later_observation",
    ]
    with pytest.raises(ValueError, match="overlap"):
        module.normalize_time_splits(
            {
                "development": {"start": "2019-01-02", "end": "2023-01-31"},
                "validation": {"start": "2023-01-01", "end": "2024-12-31"},
            }
        )


def test_candidate_selection_uses_development_period_only() -> None:
    module = load_module()
    variants = [
        {
            "lookback_days": 20,
            "moving_average_days": 20,
            "full_period": {"annualized_return": 0.90, "max_drawdown": -0.20},
            "splits": {
                "development": {
                    "annualized_return": 0.12,
                    "max_drawdown": -0.20,
                }
            },
            "switch_count": 100,
        },
        {
            "lookback_days": 20,
            "moving_average_days": 25,
            "full_period": {"annualized_return": 0.20, "max_drawdown": -0.20},
            "splits": {
                "development": {
                    "annualized_return": 0.18,
                    "max_drawdown": -0.25,
                }
            },
            "switch_count": 120,
        },
    ]

    candidate = module.select_development_candidate(
        variants,
        max_drawdown_abs=0.35,
    )

    assert candidate["lookback_days"] == 20
    assert candidate["moving_average_days"] == 25


def test_candidate_is_not_promoted_to_qualified_strategy() -> None:
    module = load_module()
    original = {
        "lookback_days": 20,
        "moving_average_days": 20,
        "splits": {
            "development": {"annualized_return": 0.20, "max_drawdown": -0.30},
            "validation": {"annualized_return": 0.30, "max_drawdown": -0.28},
        },
        "full_period": {"annualized_return": 0.27, "max_drawdown": -0.29},
    }
    candidate = {
        "lookback_days": 20,
        "moving_average_days": 25,
        "splits": {
            "development": {"annualized_return": 0.31, "max_drawdown": -0.24},
            "validation": {"annualized_return": 0.28, "max_drawdown": -0.26},
        },
        "full_period": {"annualized_return": 0.32, "max_drawdown": -0.26},
    }

    comparison = module.build_candidate_comparison(
        original=original,
        candidate=candidate,
        validation_split="validation",
        grid_previously_exposed=True,
    )

    assert comparison["status"] == "retrospective_improvement_candidate"
    assert comparison["qualified_strategy"] is False
    assert comparison["validation"]["annualized_return_delta"] == pytest.approx(-0.02)
    assert comparison["validation"]["max_drawdown_improvement"] == pytest.approx(0.02)
