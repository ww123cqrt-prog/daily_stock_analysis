# -*- coding: utf-8 -*-
"""Tests for rerunning a saved model prediction with a selected strategy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from quant_workflow.scripts._mlflow_run_provenance import RunProvenanceError
from src.services.trained_strategy_backtest_service import (
    TrainedStrategyBacktestService,
    _default_market_loader,
)


RUN_ID = "a" * 32
EXPERIMENT_ID = "123456789"
PROVIDER_IDENTITY = {
    "provider_contract": "immutable_snapshot_v2",
    "dataset_id": "qlib_ashare_v2",
    "snapshot_id": "qlib-native-001",
    "content_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
}


def _benchmark_frame(dates: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["SH000300"], dates],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame({"$close": close}, index=index)


def test_default_market_loader_keeps_risk_features_free_of_future_prices(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2025-01-02", periods=180)
    close = [100.0 + index + ((index % 7) - 3) * 0.1 for index in range(len(dates))]
    changed_future = close.copy()
    changed_future[-2] = close[-2] * 1.5
    changed_future[-1] = close[-1] * 0.7
    frames = iter(
        [
            _benchmark_frame(dates, close),
            _benchmark_frame(dates, changed_future),
        ]
    )
    fake_qlib = ModuleType("qlib")
    fake_qlib.__path__ = []
    fake_qlib.init = lambda **_: None
    fake_qlib_data = ModuleType("qlib.data")
    fake_qlib_data.D = SimpleNamespace(features=lambda *_args, **_kwargs: next(frames))

    with patch.dict(
        "sys.modules",
        {"qlib": fake_qlib, "qlib.data": fake_qlib_data},
    ):
        # The production loader now requires a formal direct Provider
        # descriptor.  This test isolates the look-ahead calculation, so the
        # Qlib initialization itself is stubbed.
        with patch(
            "src.services.trained_strategy_backtest_service.init_qlib_workflow"
        ):
            original = _default_market_loader(
                tmp_path, str(dates[0].date()), str(dates[-3].date())
            )
            altered = _default_market_loader(
                tmp_path, str(dates[0].date()), str(dates[-3].date())
            )

    decision_day = dates[-3]
    assert original.loc[decision_day, "benchmark_return"] != pytest.approx(
        altered.loc[decision_day, "benchmark_return"]
    )
    assert original.loc[decision_day, "volatility_20"] == pytest.approx(
        altered.loc[decision_day, "volatility_20"]
    )
    assert original.loc[decision_day, "vol_percentile"] == pytest.approx(
        altered.loc[decision_day, "vol_percentile"]
    )


def _replay_contract() -> dict:
    contract = {
        "contract_version": 2,
        "replay_engine": "dsa_optimized_strategy",
        "replay_capability": "custom_only",
        "exact_strategy_ids": [],
        "periods": {
            "training": ["2025-01-02", "2025-12-31"],
            "selection": ["2026-01-02", "2026-01-13"],
        },
        "original_parameters": {
            "strategy_id": "low_turnover_weekly",
            "start_date": "2026-01-02",
            "end_date": "2026-01-13",
            "topk": 2,
            "cost_bps": 20.0,
        },
        "strategies": [
            {
                "id": "low_turnover_weekly",
                "label": "每周低换手",
                "config": {
                    "rebalance_freq": 5,
                    "turnover_cap": 0.2,
                    "max_drawdown_stop": 0.0,
                },
            }
        ],
    }
    contract["config_hash"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return contract


def _qlib_replay_contract() -> dict:
    contract = {
        "contract_version": 2,
        "replay_engine": "qlib_portana",
        "exact_strategy_ids": ["baseline_daily"],
        "periods": {
            "training": ["2025-01-02", "2025-12-31"],
            "test": ["2026-01-02", "2026-01-13"],
        },
        "original_parameters": {
            "strategy_id": "baseline_daily",
            "start_date": "2026-01-02",
            "end_date": "2026-01-13",
            "topk": 2,
            "cost_bps": 20.0,
        },
        "strategies": [
            {
                "id": "baseline_daily",
                "label": "每日 TopK 等权基线",
                "config": {
                    "rebalance_freq": 1,
                    "turnover_cap": 1.0,
                    "max_drawdown_stop": 0.0,
                },
            },
            _replay_contract()["strategies"][0],
        ],
        "qlib_backtest_assumptions": {
            "account": 10_000_000.0,
            "benchmark": "SH000300",
            "topk": 2,
            "n_drop": 1,
            "deal_price": "close",
            "limit_threshold": 0.095,
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5.0,
        },
    }
    contract["config_hash"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return contract


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
    pd.to_pickle(
        {
            "record": [
                {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
                {
                    "class": "PortAnaRecord",
                    "module_path": "qlib.workflow.record_temp",
                    "kwargs": {
                        "config": {
                            "strategy": {
                                "class": "TopkDropoutStrategy",
                                "module_path": "qlib.contrib.strategy",
                                "kwargs": {
                                    "signal": "<PRED>",
                                    "topk": 2,
                                    "n_drop": 1,
                                },
                            },
                            "backtest": {
                                "start_time": "2026-01-02",
                                "end_time": "2026-01-13",
                                "account": 10_000_000,
                                "benchmark": "SH000300",
                                "exchange_kwargs": {
                                    "limit_threshold": 0.095,
                                    "deal_price": "close",
                                    "open_cost": 0.0005,
                                    "close_cost": 0.0015,
                                    "min_cost": 5,
                                },
                            },
                        }
                    },
                },
            ]
        },
        artifact_root / "task",
    )
    artifact_sha256 = {
        name: hashlib.sha256((artifact_root / name).read_bytes()).hexdigest()
        for name in ("pred.pkl", "label.pkl", "task")
    }

    qlib_report = pd.DataFrame(index=dates)
    qlib_report["return"] = [0.0, 0.011, 0.007, -0.003, 0.008, 0.004, -0.002, 0.006]
    qlib_report["cost"] = [0.0, 0.001, 0.0, 0.0002, 0.0, 0.0002, 0.0, 0.0002]
    qlib_report["bench"] = [0.001] * len(dates)
    qlib_report["turnover"] = [0.0, 1.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2]
    net = qlib_report["return"] - qlib_report["cost"]
    qlib_report["account"] = 10_000_000 * (1.0 + net).cumprod()
    qlib_report["value"] = qlib_report["account"] * 0.95
    excess = net - qlib_report["bench"]
    excess_std = excess.std(ddof=1)
    saved_metrics = {
        "1day.excess_return_with_cost.annualized_return": excess.mean() * 238,
        "1day.excess_return_with_cost.information_ratio": (
            excess.mean() / excess_std * (238 ** 0.5)
        ),
        "1day.excess_return_with_cost.max_drawdown": (
            excess.cumsum() - excess.cumsum().cummax()
        ).min(),
    }

    report = {
        "status": "completed",
        "provider": "canonical",
        "provider_uri": str(root / "legacy-provider"),
        "provider_identity": PROVIDER_IDENTITY,
        "generated_at": "2026-08-02T10:00:00+08:00",
        "periods": {"selection": ["2026-01-02", "2026-01-13"]},
        "backtest_contract": _replay_contract(),
        "models": [
            {
                "id": "xgboost",
                "label": "XGBoost",
                "status": "completed",
                "run_id": RUN_ID,
                "experiment_id": EXPERIMENT_ID,
                "artifact_sha256": {
                    "pred.pkl": artifact_sha256["pred.pkl"],
                    "label.pkl": artifact_sha256["label.pkl"],
                },
            }
        ],
        "strategy_evaluations": [
            {
                "model_id": "xgboost",
                "strategy_id": "low_turnover_weekly",
                "strategy_label": "每周低换手",
                "strategy_config": {
                    "rebalance_freq": 5,
                    "turnover_cap": 0.2,
                    "max_drawdown_stop": 0.0,
                },
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
                "generated_at": "2026-08-02T09:00:00+08:00",
                "metrics": saved_metrics,
                "artifact_sha256": artifact_sha256,
                "replay_contract": _qlib_replay_contract(),
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
    market.attrs["qlib_report"] = qlib_report
    return reports_root, mlruns_root, config_path, market


def test_catalog_keeps_multiple_historical_runs_and_sanitizes_paths(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    registry = reports_root / "training_runs"
    registry.mkdir()
    first = json.loads(
        (reports_root / "latest_training_result.json").read_text(encoding="utf-8")
    )
    (registry / f"{RUN_ID}.json").write_text(json.dumps(first), encoding="utf-8")
    second = {
        **first,
        "run_id": "b" * 32,
        "generated_at": "2026-08-02T11:00:00+08:00",
    }
    (registry / f"{'b' * 32}.json").write_text(json.dumps(second), encoding="utf-8")

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    catalog = service.catalog()

    baseline_runs = [item for item in catalog["runs"] if item["source_type"] == "baseline"]
    assert [item["run_id"] for item in baseline_runs] == ["b" * 32, RUN_ID]
    assert all(item["experiment_id"] == EXPERIMENT_ID for item in baseline_runs)
    assert all(item["exact_replay_available"] for item in baseline_runs)
    assert all(item["custom_replay_available"] for item in baseline_runs)
    serialized = json.dumps(catalog, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "provider_uri" not in serialized


def test_catalog_skips_candidate_child_registry_and_uses_candidate_model_identity(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    latest = reports_root / "latest_training_result.json"
    child = json.loads(latest.read_text(encoding="utf-8"))
    child["result_scope"] = "candidate_child"
    latest.unlink()
    registry = reports_root / "training_runs"
    registry.mkdir()
    (registry / f"{RUN_ID}.json").write_text(
        json.dumps(child),
        encoding="utf-8",
    )
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    matching = [row for row in service.catalog()["runs"] if row["run_id"] == RUN_ID]

    assert len(matching) == 1
    assert matching[0]["source_type"] == "candidate"
    assert matching[0]["model_id"] == "xgboost"
    assert matching[0]["experiment_id"] == EXPERIMENT_ID


def test_baseline_exact_replay_uses_qlib_portana_and_matches_saved_metrics(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    calls: list[tuple[Path, dict, pd.DataFrame]] = []

    def qlib_runner(
        provider: Path,
        task_config: dict,
        pred: pd.DataFrame,
    ) -> pd.DataFrame:
        calls.append((provider, task_config, pred))
        return market.attrs["qlib_report"].copy()

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: pytest.fail(
            "exact Qlib replay must not use the simplified market/label engine"
        ),
        qlib_backtest_runner=qlib_runner,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    result = service.run(
        run_id=RUN_ID,
        model_id="latest",
        strategy_id="baseline_daily",
        replay_mode="exact",
        start_date=None,
        end_date=None,
        topk=None,
        cost_bps=None,
    )

    assert len(calls) == 1
    provider, task_config, replay_pred = calls[0]
    assert provider == tmp_path / "resolved-native-provider"
    assert task_config["strategy"]["class"] == "TopkDropoutStrategy"
    assert task_config["backtest"]["exchange_kwargs"]["close_cost"] == 0.0015
    assert replay_pred.shape == (32, 1)
    assert result["replay_engine"] == "qlib_portana"
    assert result["exact_metric_validation"]["status"] == "pass"
    assert result["metrics"]["annual_excess_return"] == pytest.approx(
        result["exact_metric_validation"]["saved"]["annual_excess_return"]
    )
    assert set(result["artifact_sha256"]) == {"pred.pkl", "label.pkl", "task"}
    assert result["parameters"]["cost_bps"] == 20.0


def test_baseline_custom_replay_reports_the_engine_that_actually_executed(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        qlib_backtest_runner=lambda *_: pytest.fail(
            "custom replay must use the DSA configurable strategy engine"
        ),
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    result = service.run(
        run_id=RUN_ID,
        model_id="latest",
        strategy_id="baseline_daily",
        replay_mode="custom",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    assert result["replay_engine"] == "dsa_optimized_strategy"
    assert result["source_replay_engine"] == "qlib_portana"
    assert result["modified_from_original"] is True


def test_baseline_non_original_strategy_is_not_marked_exact_replayable(
    tmp_path: Path,
) -> None:
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

    run = next(
        item
        for item in service.catalog()["runs"]
        if item["source_type"] == "baseline"
    )
    strategies = {item["id"]: item for item in run["strategies"]}

    assert strategies["baseline_daily"]["exact_replay_available"] is True
    assert strategies["low_turnover_weekly"]["exact_replay_available"] is False
    with pytest.raises(ValueError, match="selected strategy"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="low_turnover_weekly",
            replay_mode="exact",
            start_date=None,
            end_date=None,
            topk=None,
            cost_bps=None,
        )


def test_exact_replay_rejects_metric_drift(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    latest_path = reports_root / "latest_training_result.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["metrics"]["1day.excess_return_with_cost.annualized_return"] += 0.01
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        qlib_backtest_runner=lambda *_: market.attrs["qlib_report"].copy(),
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="does not match the saved Qlib metrics"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="baseline_daily",
            replay_mode="exact",
            start_date=None,
            end_date=None,
            topk=None,
            cost_bps=None,
        )


def test_candidate_dsa_strategy_is_custom_only_and_rejects_exact_replay(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["strategy_evaluations"][0]["strategy_config"] = {
        "rebalance_freq": 1,
        "turnover_cap": 1.0,
        "max_drawdown_stop": -0.5,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config_path.write_text("strategies: []\n", encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    candidate = next(
        row for row in service.catalog()["runs"] if row["source_type"] == "candidate"
    )
    assert candidate["replay_capability"] == "custom_only"
    assert candidate["exact_replay_available"] is False
    assert candidate["custom_replay_available"] is True
    assert all(
        strategy["exact_replay_available"] is False
        for strategy in candidate["strategies"]
    )
    assert "重建回测" in candidate["replay_unavailable_reason"]

    with pytest.raises(ValueError, match="exact replay contract"):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            replay_mode="exact",
            start_date="2026-01-09",
            end_date="2026-01-09",
            topk=1,
            cost_bps=999.0,
        )


def test_legacy_candidate_contract_cannot_claim_exact_dsa_replay(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contract = dict(report["backtest_contract"])
    contract.pop("replay_capability")
    contract["exact_strategy_ids"] = ["low_turnover_weekly"]
    contract.pop("config_hash")
    contract["config_hash"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report["backtest_contract"] = contract
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    candidate = next(
        row for row in service.catalog()["runs"] if row["source_type"] == "candidate"
    )

    assert candidate["replay_capability"] == "custom_only"
    assert candidate["exact_replay_available"] is False
    assert "重建回测" in candidate["replay_unavailable_reason"]
    with pytest.raises(ValueError, match="exact replay contract"):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            replay_mode="exact",
            start_date=None,
            end_date=None,
            topk=None,
            cost_bps=None,
        )


def test_custom_replay_is_labeled_modified_but_keeps_frozen_strategy(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["strategy_evaluations"][0]["strategy_config"] = {
        "rebalance_freq": 1,
        "turnover_cap": 1.0,
        "max_drawdown_stop": -0.5,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

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
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        replay_mode="custom",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=3,
        cost_bps=30.0,
    )

    assert result["replay_mode"] == "custom"
    assert result["modified_from_original"] is True
    assert result["parameters"]["topk"] == 3
    assert result["strategy_config"] == _replay_contract()["strategies"][0]["config"]


def test_candidate_custom_replay_uses_selected_model_hashes_not_report_hashes(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_sha256"] = {
        "pred.pkl": "0" * 64,
        "label.pkl": "1" * 64,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
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
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        replay_mode="custom",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    assert result["status"] == "completed"


def test_candidate_custom_replay_rejects_selected_model_hash_mismatch(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["models"][0]["artifact_sha256"]["pred.pkl"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="frozen training report"):
        service.run(
            run_id=RUN_ID,
            model_id="xgboost",
            strategy_id="low_turnover_weekly",
            replay_mode="custom",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )


def test_legacy_contract_is_cataloged_as_custom_only(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    latest = reports_root / "latest_training_result.json"
    report = json.loads(latest.read_text(encoding="utf-8"))
    report.pop("replay_contract")
    latest.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    row = next(item for item in service.catalog()["runs"] if item["source_type"] == "baseline")

    assert row["exact_replay_available"] is False
    assert row["custom_replay_available"] is True
    assert row["replay_capability"] == "custom_only"
    assert "旧版" in row["replay_unavailable_reason"]
    with pytest.raises(ValueError, match="exact replay contract"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="baseline_daily",
            replay_mode="exact",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )


def test_qlib_contract_without_task_hash_is_cataloged_as_custom_only(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    latest = reports_root / "latest_training_result.json"
    report = json.loads(latest.read_text(encoding="utf-8"))
    report["artifact_sha256"].pop("task")
    latest.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    row = next(item for item in service.catalog()["runs"] if item["source_type"] == "baseline")

    assert row["exact_replay_available"] is False
    assert row["custom_replay_available"] is True
    assert row["replay_capability"] == "custom_only"
    assert "task" in row["replay_unavailable_reason"]
    with pytest.raises(ValueError, match="exact replay contract"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="baseline_daily",
            replay_mode="exact",
            start_date=None,
            end_date=None,
            topk=None,
            cost_bps=None,
        )


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


def test_run_response_sanitizes_provider_identity_paths(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provider_identity"] = {
        **report["provider_identity"],
        "provider_uri": str(tmp_path / "private-provider"),
        "local_path": str(tmp_path / "another-private-path"),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
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
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    serialized = json.dumps(result, ensure_ascii=False)
    assert result["provider_identity"] == PROVIDER_IDENTITY
    assert "provider_uri" not in serialized
    assert str(tmp_path) not in serialized


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


def test_rejects_candidate_report_marked_as_dry_run(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "candidate_pipeline_latest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dry_run"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="test result cannot be backtested"):
        service.run(
            run_id=RUN_ID,
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


def test_old_baseline_run_remains_replayable_after_latest_changes(tmp_path: Path) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    run_registry = reports_root / "training_runs"
    run_registry.mkdir()
    original = json.loads(
        (reports_root / "latest_training_result.json").read_text(encoding="utf-8")
    )
    (run_registry / f"{RUN_ID}.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    (reports_root / "latest_training_result.json").write_text(
        json.dumps({"status": "completed", "run_id": "b" * 32}),
        encoding="utf-8",
    )
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

    assert result["run_id"] == RUN_ID


def test_candidate_replay_uses_immutable_report_and_frozen_strategy_config(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    latest_path = reports_root / "candidate_pipeline_latest.json"
    original = json.loads(latest_path.read_text(encoding="utf-8"))
    (reports_root / "candidate_pipeline_20260802_100000.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    latest_path.write_text(
        json.dumps({"status": "completed", "models": [], "strategy_evaluations": []}),
        encoding="utf-8",
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "strategies": [
                    {
                        "id": "low_turnover_weekly",
                        "label": "changed later",
                        "kwargs": {
                            "rebalance_freq": 1,
                            "turnover_cap": 0.0,
                            "max_drawdown_stop": -0.5,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
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
        model_id="xgboost",
        strategy_id="low_turnover_weekly",
        start_date="2026-01-02",
        end_date="2026-01-13",
        topk=2,
        cost_bps=20.0,
    )

    assert result["strategy_config"] == {
        "rebalance_freq": 5,
        "turnover_cap": 0.2,
        "max_drawdown_stop": 0.0,
    }


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


def test_exact_replay_binds_artifact_hashes_to_the_frozen_training_report(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "latest_training_result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_sha256"]["pred.pkl"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        qlib_backtest_runner=lambda *_: market.attrs["qlib_report"].copy(),
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="frozen training report"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="baseline_daily",
            replay_mode="exact",
            start_date=None,
            end_date=None,
            topk=None,
            cost_bps=None,
        )


def test_custom_replay_binds_artifact_hashes_to_the_frozen_training_report(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    report_path = reports_root / "latest_training_result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_sha256"]["pred.pkl"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=_matching_artifact_validator(tmp_path),
    )

    with pytest.raises(ValueError, match="frozen training report"):
        service.run(
            run_id=RUN_ID,
            model_id="latest",
            strategy_id="baseline_daily",
            replay_mode="custom",
            start_date="2026-01-02",
            end_date="2026-01-13",
            topk=2,
            cost_bps=20.0,
        )


def test_artifact_deserialization_uses_the_same_bytes_that_were_hashed(
    tmp_path: Path,
) -> None:
    reports_root, mlruns_root, config_path, market = _write_fixture(tmp_path)
    artifact_path = mlruns_root / EXPERIMENT_ID / RUN_ID / "artifacts" / "pred.pkl"
    original = pd.read_pickle(artifact_path)
    original_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    replacement = original.copy()
    replacement.iloc[:, 0] = -999.0

    def replace_after_validation(**kwargs):
        artifact = Path(kwargs["artifact_path"])
        validated_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        replacement.to_pickle(artifact)
        return {
            "provider_identity": dict(PROVIDER_IDENTITY),
            "provider_path": str(tmp_path / "resolved-native-provider"),
            "artifact_sha256": validated_hash,
        }

    service = TrainedStrategyBacktestService(
        reports_root=reports_root,
        mlruns_root=mlruns_root,
        quant_data_root=tmp_path,
        strategy_config_path=config_path,
        market_loader=lambda *_: market,
        provider_resolver=lambda identity, root: root / "resolved-native-provider",
        artifact_validator=replace_after_validation,
    )

    loaded, provenance = service._read_artifact(
        EXPERIMENT_ID,
        RUN_ID,
        "pred.pkl",
        expected_provider_identity=PROVIDER_IDENTITY,
        expected_provider_path=tmp_path / "resolved-native-provider",
        expected_artifact_sha256=original_hash,
    )

    pd.testing.assert_frame_equal(loaded, original)
    assert provenance["artifact_sha256"] == original_hash
    pd.testing.assert_frame_equal(pd.read_pickle(artifact_path), replacement)


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
