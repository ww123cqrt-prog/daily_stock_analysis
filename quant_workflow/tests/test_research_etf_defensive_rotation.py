from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "research_etf_defensive_rotation.py"
)
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "etf_defensive_rotation.yaml"
)
LOCKED_SYMBOLS = ("SZ159915", "SH513100", "SH512890")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "research_etf_defensive_rotation", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _price_frame(symbol: str, closes: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-18", periods=len(closes))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes * 0.999,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000.0),
        }
    )


def _three_asset_data(periods: int = 80) -> dict[str, pd.DataFrame]:
    x = np.arange(periods, dtype=float)
    return {
        "SZ159915": _price_frame("SZ159915", 1.0 + x * 0.004),
        "SH513100": _price_frame("SH513100", 1.0 + x * 0.003),
        "SH512890": _price_frame(
            "SH512890", 1.0 + np.sin(x / 8.0) * 0.03 + x * 0.001
        ),
    }


def test_config_freezes_the_three_etf_rule_without_parameter_search() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert tuple(config["strategy"]["symbols"]) == LOCKED_SYMBOLS
    assert config["strategy"]["lookback_days"] == 20
    assert config["strategy"]["moving_average_days"] == 20
    assert config["strategy"]["eligibility_mode"] == "filter_then_rank"
    assert config["strategy"]["signal_at"] == "close"
    assert config["strategy"]["execution_at"] == "next_common_open"
    assert config["strategy"]["all_ineligible_position"] == "CASH"
    assert config["strategy"]["parameter_search"] is False
    assert config["costs"]["commission_bps_each_side"] == pytest.approx(0.5)
    assert config["costs"]["slippage_bps_each_side"] == [0.0, 1.0, 2.0]
    assert config["costs"]["minimum_commission_applied"] is False
    assert config["costs"]["stamp_duty_applied"] is False
    assert config["costs"]["transfer_fee_applied"] is False
    assert config["data"]["required_runtime_status"] == "research_preview"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("strategy", "all_ineligible_position", "SH512890"),
        ("costs", "minimum_commission_applied", True),
        ("costs", "stamp_duty_applied", True),
        ("costs", "transfer_fee_applied", True),
    ],
)
def test_config_rejects_rules_or_fees_not_implemented_by_the_locked_engine(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    module = load_module()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config[section][field] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="locked .* config mismatch"):
        module.load_config(config_path)


def test_locked_variants_include_three_etf_and_both_two_etf_rules() -> None:
    module = load_module()

    result = module.evaluate_locked_variants(
        _three_asset_data(),
        commission_bps_each_side=0.5,
        slippage_bps_each_side=(0.0, 1.0, 2.0),
    )

    assert set(result) == {
        "three_etf_defensive",
        "two_etf_legacy_exact",
        "two_etf_eligible_first",
    }
    assert result["three_etf_defensive"]["rule"]["symbols"] == list(LOCKED_SYMBOLS)
    assert result["three_etf_defensive"]["rule"]["eligibility_mode"] == (
        "filter_then_rank"
    )
    assert set(result["three_etf_defensive"]["cost_stress"]) == {
        "0bp_slippage",
        "1bp_slippage",
        "2bp_slippage",
    }
    assert result["three_etf_defensive"]["cost_stress"]["0bp_slippage"][
        "metrics"
    ]["side_cost_bps"] == pytest.approx(0.5)


def test_common_period_uses_all_three_etfs_and_keeps_warmup_as_cash() -> None:
    module = load_module()
    data = _three_asset_data(45)
    data["SH512890"] = data["SH512890"].iloc[5:].reset_index(drop=True)

    variants = module.evaluate_locked_variants(
        data,
        commission_bps_each_side=0.5,
        slippage_bps_each_side=(0.0,),
    )
    result = variants["three_etf_defensive"]["cost_stress"]["0bp_slippage"]

    assert result["metrics"]["start_date"] == "2019-01-25"
    periods = result["periods"]
    assert periods.iloc[0]["position"] == "CASH"
    assert (periods.iloc[:20]["position"] == "CASH").all()


def test_block_bootstrap_is_deterministic_and_reports_incremental_interval() -> None:
    module = load_module()
    dates = pd.bdate_range("2023-01-02", periods=120)
    strategy = pd.DataFrame(
        {"date": dates, "net_return": np.tile([0.002, -0.0005], 60)}
    )
    baseline = pd.DataFrame(
        {"date": dates, "net_return": np.tile([0.001, -0.0005], 60)}
    )

    first = module.block_bootstrap_incremental_return(
        strategy,
        baseline,
        block_days=20,
        samples=300,
        seed=20260807,
    )
    second = module.block_bootstrap_incremental_return(
        strategy,
        baseline,
        block_days=20,
        samples=300,
        seed=20260807,
    )

    assert first == second
    assert first["samples"] == 300
    assert first["mean_annualized_return_delta"] > 0
    assert first["ci95_lower"] > 0


def test_unaccepted_data_can_never_emit_a_formal_candidate_label() -> None:
    module = load_module()
    evidence = {
        "max_drawdown": -0.10,
        "full_annualized_delta": 0.20,
        "validation_annualized_delta": 0.15,
        "observation_annualized_delta": 0.12,
        "bootstrap_ci95_lower": 0.05,
        "leave_one_year_positive_ratio": 1.0,
    }

    preview = module.classify_incremental_value(
        evidence,
        formally_accepted=False,
        max_drawdown_abs_max=0.35,
        leave_one_year_positive_ratio_min=0.6333,
    )

    assert preview == {
        "status": "research_preview",
        "formal_candidate_status": None,
        "classification_enabled": False,
        "reason": "ETF snapshot has not completed formal quant_data acceptance",
    }


def test_missing_chinext_index_is_reported_without_etf_substitution(tmp_path: Path) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars/day.txt").write_text(
        "2026-01-05\n2026-01-06\n2026-01-07\n", encoding="utf-8"
    )
    for symbol in ("SH000300", "SH000905"):
        feature_dir = provider / "features" / symbol.lower()
        feature_dir.mkdir(parents=True)
        np.asarray([0.0, 100.0, 101.0, 102.0], dtype="<f4").tofile(
            feature_dir / "open.day.bin"
        )

    result = module.load_index_benchmarks_from_provider(provider)

    assert set(result["available"]) == {"SH000300", "SH000905"}
    assert result["missing"] == ["SZ399006"]
    assert "SZ159915" not in result["available"]
    assert "SZ159915" not in result["missing"]


def test_html_report_has_inline_charts_and_preview_warning() -> None:
    module = load_module()
    report = {
        "status": "research_preview",
        "summary": {
            "usable_for_trading": False,
            "formal_candidate_status": None,
            "data_end": "2026-08-06",
        },
        "strategy": {
            "name": "三ETF防守轮动",
            "symbols": list(LOCKED_SYMBOLS),
            "commission_bps_each_side": 0.5,
            "base_slippage_bps_each_side": 0.0,
        },
        "performance": {
            "primary_metrics": {
                "annualized_return": 0.12,
                "total_return": 0.8,
                "max_drawdown": -0.2,
            },
            "comparators": {
                "two_etf_legacy_exact": {
                    "annualized_return": 0.11,
                    "total_return": 0.7,
                    "max_drawdown": -0.21,
                },
                "SH000300": {
                    "annualized_return": 0.06,
                    "total_return": 0.4,
                    "max_drawdown": -0.3,
                },
            },
            "periods": {
                "validation": {
                    "strategy": {"annualized_return": 0.1, "max_drawdown": -0.2},
                    "legacy_two_etf": {
                        "annualized_return": 0.12,
                        "max_drawdown": -0.18,
                    },
                }
            },
        },
        "stability": {
            "cost_stress": {
                "0bp_slippage": {
                    "side_cost_bps": 0.5,
                    "annualized_return": 0.12,
                    "max_drawdown": -0.2,
                }
            },
            "start_date_sensitivity": [],
            "leave_one_year_positive_ratio": 0.63,
        },
        "holding_behavior": {
            "position_fraction": {"CASH": 0.1, "SH512890": 0.3},
            "switch_count": 20,
            "invested_fraction": 0.9,
        },
        "benchmarks": {"missing": ["SZ399006"]},
        "chart_data": {
            "equity": [
                {"date": "2026-01-05", "strategy": 1.0},
                {"date": "2026-01-06", "strategy": 1.1},
            ],
            "drawdown": [
                {"date": "2026-01-05", "strategy": 0.0},
                {"date": "2026-01-06", "strategy": -0.05},
            ],
            "yearly_returns": [{"year": 2026, "strategy": 0.1}],
        },
        "reproducibility": {},
    }

    html = module.render_html(report)

    assert "research_preview" in html
    assert "不可用于交易" in html
    assert html.count("<svg") >= 3
    assert "SZ399006" in html
    assert "1bp = 0.01%" in html
    assert "成交金额的0.005%" in html
    assert "分阶段表现" in html
    assert "成本压力" in html
    assert "持仓行为" in html
    assert "原两ETF精确旧规则" in html
    assert "沪深300" in html
