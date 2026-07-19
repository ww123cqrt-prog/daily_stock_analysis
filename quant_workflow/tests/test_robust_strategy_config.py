from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"


def test_checked_in_robust_strategy_config_preserves_approved_gates() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["version"] == 1
    assert config["universe"]["kind"] == "pit_csi800"
    assert config["universe"]["start_date"] == "2014-01-02"
    assert config["data_gates"]["membership_coverage_min"] == 0.99
    assert config["data_gates"]["ohlcv_coverage_min"] == 0.98
    assert config["training"]["windows_trading_days"] == [504, 756, 1008]
    assert config["training"]["tree_seeds"] == [42, 2026, 3407]
    assert len(config["rolling_validation"]["test_quarters"]) == 12
    assert config["rolling_validation"]["test_quarters"][0] == "2023Q1"
    assert config["rolling_validation"]["test_quarters"][-1] == "2025Q4"
    assert config["label"]["expression"] == "Ref($open, -6) / Ref($open, -1) - 1"
    assert config["portfolio"]["single_stock_weight_max"] == 0.03
    assert config["portfolio"]["industry_weight_max"] == 0.25
    assert config["portfolio"]["growth_and_star_weight_max"] == 0.30
    assert config["acceptance"]["positive_excess_quarter_ratio_min"] == 0.70
    assert config["acceptance"]["annual_excess_return_min"] == 0.05
    assert config["acceptance"]["median_quarter_information_ratio_min"] == 0.5
    assert config["acceptance"]["positive_rank_ic_interval_ratio_min"] == 0.70
    assert config["acceptance"]["window_rank_correlation_median_min"] == 0.70
    assert config["acceptance"]["top50_overlap_mean_min"] == 0.60
    assert config["acceptance"]["seed_rank_correlation_median_min"] == 0.70
    assert config["acceptance"]["seed_top50_overlap_mean_min"] == 0.60
    assert config["acceptance"]["development_max_drawdown_abs_max"] == 0.20
    assert config["acceptance"]["shadow_days"] == 60
    assert config["acceptance"]["shadow_task_success_ratio_min"] == 0.95
    assert config["safety"]["broker_connection"] is False
    assert config["safety"]["automatic_orders"] is False


def test_config_defines_five_models_five_strategies_and_fixed_statuses() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert [item["id"] for item in config["models"]] == [
        "linear_ols",
        "linear_ridge",
        "lightgbm",
        "xgboost",
        "catboost",
    ]
    assert [item["id"] for item in config["strategies"]] == [
        "weekly_top50",
        "low_turnover",
        "regime_scaled",
        "sector_constrained",
        "full_risk_control",
    ]
    assert config["statuses"] == [
        "blocked_data",
        "development_failed",
        "research_candidate_ready",
        "shadow_observation",
        "robust_strategy_qualified",
    ]
