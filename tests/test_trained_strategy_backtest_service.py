# -*- coding: utf-8 -*-
"""Tests for rerunning a saved model prediction with a selected strategy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.services.trained_strategy_backtest_service import TrainedStrategyBacktestService


RUN_ID = "a" * 32
EXPERIMENT_ID = "123456789"


def _write_fixture(root: Path) -> tuple[Path, Path, Path, pd.DataFrame]:
    reports_root = root / "reports"
    mlruns_root = root / "mlruns"
    config_path = root / "candidate_pipeline.yaml"
    reports_root.mkdir(parents=True)

    dates = pd.bdate_range("2026-01-02", periods=8)
    instruments = ["SH600000", "SH600001", "SH600002", "SH600003"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    pred = pd.DataFrame(
        {"score": [score for _ in dates for score in (4.0, 3.0, 2.0, 1.0)]},
        index=index,
    )
    label = pd.DataFrame(
        {"LABEL0": [value for _ in dates for value in (0.012, 0.008, -0.002, -0.004)]},
        index=index,
    )
    artifact_root = mlruns_root / EXPERIMENT_ID / RUN_ID / "artifacts"
    artifact_root.mkdir(parents=True)
    pred.to_pickle(artifact_root / "pred.pkl")
    label.to_pickle(artifact_root / "label.pkl")

    report = {
        "status": "completed",
        "provider": "canonical",
        "provider_uri": str(root / "provider"),
        "periods": {"selection": ["2026-01-02", "2026-01-13"]},
        "models": [
            {
                "id": "xgboost",
                "label": "XGBoost",
                "status": "completed",
                "run_id": RUN_ID,
                "experiment_id": EXPERIMENT_ID,
            }
        ],
        "strategy_evaluations": [
            {
                "model_id": "xgboost",
                "strategy_id": "low_turnover_weekly",
                "strategy_label": "每周低换手",
            }
        ],
    }
    (reports_root / "candidate_pipeline_latest.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )
    (reports_root / "latest_training_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment_name": "baseline_lgb_alpha158",
                "run_id": RUN_ID,
                "experiment_id": EXPERIMENT_ID,
                "provider": "canonical",
                "provider_uri": str(root / "provider"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategies": [
                    {
                        "id": "low_turnover_weekly",
                        "label": "每周低换手",
                        "kwargs": {
                            "rebalance_freq": 5,
                            "turnover_cap": 0.2,
                            "max_drawdown_stop": 0.0,
                        },
                    }
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    market = pd.DataFrame(index=dates)
    market["benchmark_return"] = 0.001
    market["close"] = 100.0
    market["ma200"] = 99.0
    market["trend_200"] = 1.0
    market["vol_percentile"] = 0.2
    return reports_root, mlruns_root, config_path, market


def test_reruns_the_selected_saved_prediction_without_training(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    loaded_ranges: list[tuple[str, str]] = []

    def market_loader(_provider: Path, start: str, end: str) -> pd.DataFrame:
        loaded_ranges.append((start, end))
        return market

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        strategy_config_path=config_path,
        market_loader=market_loader,
    )
    result = service.run(
        run_id=RUN_ID,
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    assert loaded_ranges == [("2026-01-02", "2026-01-13")]
    assert result["status"] == "completed"
    assert result["run_id"] == RUN_ID
    assert result["strategy_id"] == "low_turnover_weekly"
    assert result["parameters"] == {
        "start_date": "2026-01-02",
        "end_date": "2026-01-13",
        "topk": 2,
        "cost_bps": 20.0,
    }
    assert result["metrics"]["days"] == 8
    assert result["metrics"]["total_return"] > result["metrics"]["benchmark_total_return"]
    assert len(result["equity_curve"]) == 8
    assert result["equity_curve"][-1]["strategy_cumulative_return"] == pytest.approx(
        result["metrics"]["total_return"]
    )


def test_rejects_a_run_that_is_not_registered_in_the_training_report(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
    )

    with pytest.raises(ValueError, match="not available in the current training results"):
        service.run(
            run_id="b" * 32,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )


def test_latest_trained_model_can_use_a_registered_trading_strategy(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
    )

    result = service.run(
        run_id=RUN_ID,
        model_id="latest",
        strategy_id="low_turnover_weekly",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    assert result["model_label"] == "baseline_lgb_alpha158"
    assert result["strategy_id"] == "low_turnover_weekly"
    assert result["strategy_label"] == "每周低换手"
