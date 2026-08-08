from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "limit_up_intraday_research.py"
    )
    assert path.exists(), "limit-up intraday research module is missing"
    spec = importlib.util.spec_from_file_location("limit_up_intraday_research", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "limit_up_intraday_data.py"
    )
    spec = importlib.util.spec_from_file_location("limit_up_intraday_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def minute_bars() -> pd.DataFrame:
    times = pd.to_datetime(
        [
            "2025-06-10 13:57:00",
            "2025-06-10 13:58:00",
            "2025-06-10 13:59:00",
            "2025-06-10 14:00:00",
            "2025-06-10 14:01:00",
            "2025-06-10 14:02:00",
            "2025-06-10 14:29:00",
            "2025-06-10 14:30:00",
        ]
    )
    close = [9.95, 10.0, 10.0, 10.0, 9.96, 10.0, 9.98, 9.98]
    low = [9.94, 10.0, 10.0, 10.0, 9.95, 9.98, 9.96, 9.97]
    return pd.DataFrame(
        {
            "datetime": times,
            "open": close,
            "high": [9.96, 10.0, 10.0, 10.0, 10.0, 10.0, 9.99, 9.99],
            "low": low,
            "close": close,
            "volume": [100.0, 200.0, 100.0, 50.0, 400.0, 300.0, 200.0, 100.0],
            "money": [995.0, 2000.0, 1000.0, 500.0, 3984.0, 3000.0, 1996.0, 998.0],
            "high_limit": 10.0,
            "low_limit": 8.2,
            "pre_close": 9.09,
        }
    )


def build_research_snapshot(tmp_path: Path) -> Path:
    data_module = load_data_module()
    stage = tmp_path / "snapshot"
    bar_path = stage / "bars" / "2025" / "2025-06-10" / "SH600000.parquet"
    bar_path.parent.mkdir(parents=True)
    minute_bars().to_parquet(bar_path, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "date": "2025-06-10",
                "relative_path": "bars/2025/2025-06-10/SH600000.parquet",
                "retrieval_ever_touched_limit": True,
                "final_status": "broken_board",
                "listing_date": "1999-11-10",
                "listing_age_sessions": 5000,
                "is_st": False,
                "board": "main",
                "industry": "bank",
                "free_float_shares": 10_000.0,
                "momentum_5d": 0.08,
                "momentum_20d": 0.12,
                "consecutive_limit_days": 2,
                "industry_heat": 0.60,
                "next_date": "2025-06-11",
                "next_open": 10.2,
                "next_close": 10.4,
            }
        ]
    ).to_parquet(stage / "events.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-06-10",
                "decision_time": decision_time,
                "status": "passed",
                "expected_count": 1 if decision_time == "14:00" else 0,
                "recovered_count": 1 if decision_time == "14:00" else 0,
                "missing_symbols": "[]",
            }
            for decision_time in ("14:00", "14:30")
        ]
    ).to_parquet(stage / "recall_audit.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-06-10",
                "decision_time": decision_time,
                "sealed_count": 12 if decision_time == "14:00" else 10,
                "broken_board_rate": 0.25 if decision_time == "14:00" else 0.35,
                "max_consecutive_limit_days": 4,
                "index_trend_20d": 0.03,
                "advance_decline_ratio": 1.2,
                "turnover_activity": 1.1,
                "regime": "normal",
            }
            for decision_time in ("14:00", "14:30")
        ]
    ).to_parquet(stage / "market_state.parquet", index=False)
    events = data_module._read_events(stage)
    recall = data_module._read_recall(stage)
    market_state = data_module._read_market_state(stage, events["date"])
    index = data_module._build_file_index(stage, events)
    index.to_parquet(stage / "FILE_INDEX.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "dataset_id": "ashare_limit_up_intraday_jqdata",
        "snapshot_id": "limit-up-jqdata-20250610-test",
        "data_kind": "limit_up_intraday_events",
        "role": "research_input",
        "status": "ready",
        "source": {"provider": "jqdata", "interface": "get_price_1m"},
        "source_retrieved_at": "2026-08-07T14:00:00+08:00",
        "price_basis": "raw",
        "frequency": "1m",
        "decision_times": ["14:00", "14:30"],
        "date_range": {"start": "2025-06-10", "end": "2025-06-10"},
        "event_count": 1,
        "broken_board_event_count": 1,
        "bar_row_count": int(index["row_count"].sum()),
        "events_path": "events.parquet",
        "recall_audit_path": "recall_audit.parquet",
        "market_state_path": "market_state.parquet",
        "file_index_path": "FILE_INDEX.parquet",
        "recall_audit": {
            "passed": True,
            "sample_count": int(len(recall)),
            "minimum_recall": 1.0,
        },
        "forbidden_model_features": list(data_module.FORBIDDEN_MODEL_FEATURES),
        "market_state": {
            "point_in_time": True,
            "decision_times": ["14:00", "14:30"],
            "row_count": int(len(market_state)),
        },
    }
    manifest["content_sha256"] = data_module._content_hash(stage, manifest)
    (stage / data_module.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stage


def test_decision_features_are_cut_off_and_later_broken_board_is_retained() -> None:
    module = load_module()
    bars = minute_bars()

    at_1400 = module.build_decision_snapshot(bars, decision_time="14:00")
    at_1430 = module.build_decision_snapshot(bars, decision_time="14:30")

    assert at_1400["is_candidate"] is True
    assert at_1400["decision_close"] == pytest.approx(10.0)
    assert at_1400["cumulative_volume"] == pytest.approx(450.0)
    assert at_1400["break_count"] == 0
    assert at_1430["is_candidate"] is False
    assert module.post_decision_board_opened(bars, decision_time="14:00") is True


def test_candidate_recall_audit_blocks_any_missing_sealed_stock() -> None:
    module = load_module()

    passed = module.audit_candidate_recall(
        expected={"SH600000", "SZ300001"},
        recovered={"SZ300001", "SH600000", "SH688001"},
    )
    blocked = module.audit_candidate_recall(
        expected={"SH600000", "SZ300001"},
        recovered={"SH600000"},
    )

    assert passed["status"] == "passed"
    assert passed["recall"] == pytest.approx(1.0)
    assert blocked["status"] == "blocked_data"
    assert blocked["missing_symbols"] == ["SZ300001"]


def test_locked_2026_rows_are_rejected_before_research() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="2026 locked"):
        module.assert_research_date_unlocked("2026-01-02")

    module.assert_research_date_unlocked("2025-12-31")


def test_stock_costs_follow_trade_date_and_sell_only_stamp_duty() -> None:
    module = load_module()

    current = module.stock_round_trip_costs(
        buy_date="2024-01-02",
        sell_date="2024-01-03",
        commission_bps_each_side=0.754,
        sell_slippage_bps=5.0,
    )
    historical = module.stock_round_trip_costs(
        buy_date="2022-01-04",
        sell_date="2022-01-05",
        commission_bps_each_side=0.754,
        sell_slippage_bps=0.0,
    )

    assert current["buy_transfer_bps"] == pytest.approx(0.1)
    assert current["sell_transfer_bps"] == pytest.approx(0.1)
    assert current["stamp_duty_bps"] == pytest.approx(5.0)
    assert current["buy_cost_rate"] == pytest.approx((0.754 + 0.1) / 10_000)
    assert current["sell_cost_rate"] == pytest.approx(
        (0.754 + 0.1 + 5.0 + 5.0) / 10_000
    )
    assert historical["buy_transfer_bps"] == pytest.approx(0.2)
    assert historical["stamp_duty_bps"] == pytest.approx(10.0)


def test_fill_layers_separate_open_evidence_from_capacity_proxy() -> None:
    module = load_module()
    layers = module.build_fill_layers(
        minute_bars(),
        decision_time="14:00",
        order_value=20.0,
        capacity_participation_rates=(0.001, 0.005, 0.01),
    )

    assert layers["theoretical_signal"]["filled"] is True
    assert layers["board_open_verified"]["filled"] is True
    assert layers["capacity_proxy_0.1pct"]["filled"] is False
    assert layers["capacity_proxy_0.5pct"]["filled"] is True
    assert layers["capacity_proxy_1pct"]["filled"] is True
    assert layers["capacity_proxy_0.5pct"]["is_real_fill_probability"] is False


def test_trade_metrics_use_expectancy_profit_factor_and_drawdown() -> None:
    module = load_module()
    metrics = module.trade_metrics([0.20, -0.10, -0.05, 0.10])

    assert metrics["expectancy"] == pytest.approx(0.0375)
    assert metrics["profit_factor"] == pytest.approx(2.0)
    assert metrics["average_win"] == pytest.approx(0.15)
    assert metrics["average_loss"] == pytest.approx(-0.075)
    assert metrics["average_win_loss_ratio"] == pytest.approx(2.0)
    assert metrics["max_consecutive_losses"] == 2
    assert metrics["max_drawdown"] < 0


def test_signal_grade_does_not_use_55_percent_realized_win_rate_as_gate() -> None:
    module = load_module()
    strong = module.grade_signal(
        expectancy=0.01,
        expectancy_bootstrap_lower=0.001,
        profit_factor=1.3,
        max_drawdown=-0.20,
        positive_quarter_fraction=0.6333,
        regime_positive_count=2,
        regime_count=3,
        realized_win_rate=0.40,
    )
    potential = module.grade_signal(
        expectancy=0.01,
        expectancy_bootstrap_lower=-0.002,
        profit_factor=1.1,
        max_drawdown=-0.20,
        positive_quarter_fraction=0.50,
        regime_positive_count=1,
        regime_count=3,
        realized_win_rate=0.70,
    )
    missing = module.grade_signal(
        expectancy=-0.001,
        expectancy_bootstrap_lower=-0.01,
        profit_factor=0.9,
        max_drawdown=-0.10,
        positive_quarter_fraction=0.80,
        regime_positive_count=3,
        regime_count=3,
        realized_win_rate=0.80,
    )

    assert strong["status"] == "strong_signal"
    assert potential["status"] == "potential_signal"
    assert missing["status"] == "signal_not_found"


def test_research_rows_keep_features_point_in_time_and_labels_separate(
    tmp_path: Path,
) -> None:
    module = load_module()
    snapshot = build_research_snapshot(tmp_path)

    rows = module.build_research_rows(snapshot, decision_time="14:00")

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["symbol"] == "SH600000"
    assert row["regime"] == "normal"
    assert row["board_open_verified"]
    assert row["gross_next_open_return"] == pytest.approx(0.02)
    expected_net = 1.02 * (1.0 - 0.0005854) / (1.0 + 0.0000854) - 1.0
    assert row["net_next_open_return_slippage_0"] == pytest.approx(expected_net)
    feature_columns = module.model_feature_columns()
    assert set(feature_columns).isdisjoint(
        {
            "retrieval_ever_touched_limit",
            "final_status",
            "next_open",
            "next_close",
            "board_open_verified",
        }
    )
    assert rows.loc[:, feature_columns].notna().all().all()
    assert module.build_research_rows(snapshot, decision_time="14:30").empty


def test_preregistered_experiments_are_four_independent_tracks() -> None:
    module = load_module()
    specs = module.preregistered_experiment_specs()

    assert [(item["decision_time"], item["exit_at"]) for item in specs] == [
        ("14:00", "next_open"),
        ("14:00", "next_close"),
        ("14:30", "next_open"),
        ("14:30", "next_close"),
    ]
    assert len({item["experiment_id"] for item in specs}) == 4


def test_top3_orders_apply_probability_and_positive_net_return_thresholds() -> None:
    module = load_module()
    predictions = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "calibrated_probability": [0.80, 0.70, 0.54, 0.90],
            "predicted_net_return": [0.03, 0.02, 0.04, -0.01],
            "entry_price": [10.0, 33.0, 20.0, 8.0],
        }
    )

    orders = module.size_topk_orders(
        predictions,
        capital=300_000.0,
        topk=3,
        lot_size=100,
        probability_threshold=0.55,
    )

    assert orders["symbol"].tolist() == ["A", "B"]
    assert orders["shares"].tolist() == [15_000, 4_500]
    assert orders["target_weight"].tolist() == pytest.approx([0.5, 0.5])
