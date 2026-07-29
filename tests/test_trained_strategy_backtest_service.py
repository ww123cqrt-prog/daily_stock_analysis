# -*- coding: utf-8 -*-
"""Tests for rerunning a saved model prediction with a selected strategy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from quant_workflow.scripts._mlflow_run_provenance import RunProvenanceError
from src.services.trained_strategy_backtest_service import TrainedStrategyBacktestService


RUN_ID = "a" * 32
EXPERIMENT_ID = "123456789"
PROVIDER_IDENTITY = {
    "provider_contract": "immutable_snapshot_v2",
    "dataset_id": "qlib_ashare_v2",
    "snapshot_id": "qlib-native-001",
    "content_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
}


def _matching_artifact_validator(root: Path):
    def validate(**kwargs):
        artifact = Path(kwargs["artifact_path"])
        return {
            "provider_identity": dict(PROVIDER_IDENTITY),
            "provider_path": str(root / "resolved-native-provider"),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }

    return validate


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
        "provider_uri": str(root / "legacy-provider"),
        "provider_identity": PROVIDER_IDENTITY,
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
                "provider_uri": str(root / "legacy-provider"),
                "provider_identity": PROVIDER_IDENTITY,
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
    loaded_ranges: list[tuple[Path, str, str]] = []

    def market_loader(provider: Path, start: str, end: str) -> pd.DataFrame:
        loaded_ranges.append((provider, start, end))
        return market

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=market_loader,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
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

    assert loaded_ranges == [
        (tmp_path / "resolved-native-provider", "2026-01-02", "2026-01-13")
    ]
    assert result["status"] == "completed"
    assert result["run_id"] == RUN_ID
    assert set(result["artifact_sha256"]) == {"pred.pkl", "label.pkl"}
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
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
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
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
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


def test_rejects_legacy_report_without_immutable_provider_identity(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("provider_identity")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="immutable native V2 provider"):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )


def test_rejects_artifact_hash_failure_before_pickle_deserialization(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)

    def reject_artifact(**_kwargs):
        raise RunProvenanceError("artifact content hash does not match")

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=reject_artifact,
    )

    with (
        patch.object(
            pd,
            "read_pickle",
            side_effect=AssertionError("unvalidated pickle was read"),
        ) as read_pickle,
        pytest.raises(ValueError, match="artifact provenance cannot be verified"),
    ):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )

    read_pickle.assert_not_called()


def test_rejects_artifact_from_another_provider_before_pickle_deserialization(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)

    def mismatched_artifact(**kwargs):
        artifact = Path(kwargs["artifact_path"])
        return {
            "provider_identity": {
                **PROVIDER_IDENTITY,
                "snapshot_id": "another-snapshot",
            },
            "provider_path": str(tmp_path / "another-provider"),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=mismatched_artifact,
    )

    with (
        patch.object(
            pd,
            "read_pickle",
            side_effect=AssertionError("cross-provider pickle was read"),
        ) as read_pickle,
        pytest.raises(ValueError, match="does not match the training report provider"),
    ):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )

    read_pickle.assert_not_called()
