"""Synthetic frozen files exercise the default native trained-Run service chain."""
import hashlib
import json
import pandas as pd
import pytest

from quant_qmt_data import QMTClient, save_dataset
from quant_workflow.scripts._mlflow_run_provenance import provider_identity, provider_tags
from src.services.trained_strategy_backtest_service import TrainedStrategyBacktestService


RUN_ID = "a" * 32
STRATEGY_ID = "daily_prediction_topk"


def test_native_trained_run_is_usable_through_public_api(native_run, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.app import create_app
    from api.v1.endpoints import backtest

    service, *_ = native_run
    # The fixture uses real frozen input, Run tags, and prediction artifact.
    monkeypatch.setattr(backtest, "TrainedStrategyBacktestService", lambda: service)
    with TestClient(create_app(static_dir=tmp_path / "static")) as client:
        catalog = client.get("/api/v1/backtest/strategy/trained/catalog")
        assert catalog.status_code == 200
        response = client.post("/api/v1/backtest/strategy/trained",
            headers={"Origin": "http://127.0.0.1:8010"}, json={
                "run_id": RUN_ID, "model_id": "latest", "strategy_id": STRATEGY_ID,
                "replay_mode": "custom", "start_date": "2024-01-02", "end_date": "2024-01-05",
                "topk": 1, "cost_bps": 10, "initial_capital": 10000, "lot_size": 100,
            })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["replay_engine"] == "dsa"
    assert result["parameters"]["initial_capital"] == 10000
    assert result["parameters"]["lot_size"] == 100
    assert result["execution_report"]["trades"]
    assert "input_dir" not in response.text


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    def no_sdk(*args, **kwargs):
        pytest.fail("Frozen trained replay must never call the SDK")

    monkeypatch.setattr(QMTClient, "_request", no_sdk)
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for symbol, opens, closes in (
        ("SH600519", [9., 10., 11., 12.], [9., 12., 11., 13.]),
        ("SZ000001", [20.] * 4, [20.] * 4),
        ("SH000300", [100., 101., 103., 104.], [100., 101., 103., 104.]),
    ):
        for day, opening, closing in zip(dates, opens, closes):
            rows.append(dict(date=day, symbol=symbol, open=opening, close=closing,
                high=max(opening, closing), low=min(opening, closing),
                volume=100000., amount=1000000., preclose=opening, suspend_flag=0))
    bars = pd.DataFrame(rows)
    bars.attrs = dict(provider="qmt", request={"symbols": list(bars.symbol.unique())},
        price_basis="front", volume_units={"SH600519": "shares", "SZ000001": "shares", "SH000300": "native"},
        missing_symbols=[], read_at="2024-01-06T10:00:00+08:00", completed_through="2024-01-05",
        contains_partial=False, qmt_data_dir=str(tmp_path / "qmt-cache"), sdk_version="synthetic-test",
        actual_start="2024-01-02", actual_end="2024-01-05", benchmark_symbols=["SH000300"])
    directory = save_dataset(bars, tmp_path / "saved-input", calendar=dates,
        universe=["SH600519", "SZ000001"])
    identity = provider_identity(directory)
    artifacts = tmp_path / "mlruns" / "1" / RUN_ID / "artifacts"
    artifacts.mkdir(parents=True)
    index = pd.MultiIndex.from_product([[dates[0]], ["SH600519", "SZ000001"]],
        names=["datetime", "instrument"])
    pd.DataFrame({"score": [2., 1.]}, index=index).to_pickle(artifacts / "pred.pkl")
    # Intentionally absurd labels must have no influence on cash/price returns.
    pd.DataFrame({"label": [9999., -9999.]}, index=index).to_pickle(artifacts / "label.pkl")
    hashes = {name: hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
              for name in ("pred.pkl", "label.pkl")}
    tags = artifacts.parent / "tags"
    tags.mkdir()
    for key, value in provider_tags(identity).items():
        (tags / key).write_text(value, encoding="utf-8")
    for name, digest in hashes.items():
        (tags / f"dsa.artifact.{name.removesuffix('.pkl')}_sha256").write_text(digest, encoding="utf-8")
    semantics = dict(feature_cutoff="signal_date_close", prediction_available="signal_date_after_close",
        order_execution="next_trading_day_open", label_future_sessions=2)
    contract = dict(contract_version=3, replay_engine="dsa", exact_strategy_ids=[],
        purpose="data_engineering_acceptance", source_config_sha256="b" * 64,
        runtime_config_sha256="c" * 64, periods={"test": ["2024-01-02", "2024-01-05"]},
        time_semantics=semantics)
    contract["config_hash"] = hashlib.sha256(json.dumps(contract, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = dict(status="completed", run_id=RUN_ID, experiment_id="1", provider="qmt_native",
        provider_identity=identity, artifact_sha256=hashes, replay_contract=contract,
        experiment_name="synthetic-native-test", generated_at="2024-01-06T10:00:00+08:00")
    reports = tmp_path / "registry"
    registry = reports / "training_runs"
    registry.mkdir(parents=True)
    report_path = registry / f"{RUN_ID}.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    service = TrainedStrategyBacktestService(reports_root=reports,
        mlruns_root=tmp_path / "mlruns", quant_data_root=tmp_path / "unused-old-data")
    return service, report, report_path, directory, artifacts


@pytest.mark.parametrize("global_input", [None, "unrelated-input"])
def test_native_provider_resolution_keeps_saved_input_dir(native_run, monkeypatch, global_input):
    service, report, _, directory, _ = native_run
    if global_input is not None:
        monkeypatch.setenv("QMT_DATASET_DIR", global_input)
    else:
        monkeypatch.delenv("QMT_DATASET_DIR", raising=False)
    assert service._resolve_report_provider(report) == directory


def test_native_catalog_offers_custom_predictions_without_frozen_strategy(native_run):
    service, report, _, directory, _ = native_run
    row = service.catalog()["runs"][0]
    assert row["contract_version"] == 3
    assert row["periods"] == report["replay_contract"]["periods"]
    assert row["original_parameters"] == {}
    assert row["exact_replay_available"] is False
    assert row["custom_replay_available"] is True
    assert [strategy["id"] for strategy in row["strategies"]] == [STRATEGY_ID]
    assert row["strategies"][0]["replay_engine"] == "dsa"
    assert "input_dir" not in json.dumps(row)
    assert str(directory) not in json.dumps(row)


def test_native_custom_run_uses_real_prices_lots_cash_fees_and_frozen_benchmark(native_run, monkeypatch):
    service, _, _, _, artifacts = native_run
    monkeypatch.setenv("QMT_DATASET_DIR", "different-global-input")
    (artifacts / "label.pkl").unlink()
    result = service.run(run_id=RUN_ID, model_id="latest", strategy_id=STRATEGY_ID,
        start_date="2024-01-02", end_date="2024-01-05", topk=1, cost_bps=100,
        initial_capital=10100., lot_size=100)
    assert result["status"] == "completed"
    assert result["replay_engine"] == "dsa"
    assert result["source_replay_engine"] == "dsa"
    assert result["contract_version"] == 3
    assert result["original_parameters"] == {}
    assert result["metrics"]["total_return"] == pytest.approx(13000. / 10100. - 1.)
    assert result["metrics"]["benchmark_total_return"] == pytest.approx(.04)
    assert result["execution_assumptions"]["execution"] == "next_trading_day_open"
    assert result["execution_assumptions"]["price_basis"] == "front"
    assert result["execution_assumptions"]["lot_size"] == 100
    ledger = result["execution_report"]
    assert ledger["metrics"]["final_equity"] == pytest.approx(13000.)
    assert ledger["metrics"]["total_cost"] == pytest.approx(100.)
    assert ledger["metrics"]["trade_count"] == 1
    trade = ledger["trades"][0]
    assert (trade["signal_date"], trade["execution_date"]) == ("2024-01-02", "2024-01-03")
    assert (trade["symbol"], trade["shares"], trade["price"], trade["fee"]) == ("SH600519", 1000, 10., 100.)
    assert [point["date"] for point in result["equity_curve"]] == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert result["time_semantics"]["order_execution"] == "next_trading_day_open"
    assert "input_dir" not in json.dumps(result)


def test_native_run_rejects_exact_and_missing_custom_parameters(native_run):
    service, *_ = native_run
    with pytest.raises(ValueError, match="exact|精确"):
        service.run(run_id=RUN_ID, model_id="latest", strategy_id=STRATEGY_ID,
            replay_mode="exact", start_date=None, end_date=None, topk=None, cost_bps=None)
    with pytest.raises(ValueError, match="Custom replay requires"):
        service.run(run_id=RUN_ID, model_id="latest", strategy_id=STRATEGY_ID,
            start_date=None, end_date=None, topk=None, cost_bps=None)


def test_native_artifact_changes_are_rejected_by_default_validator(native_run):
    service, _, _, _, artifacts = native_run
    with (artifacts / "pred.pkl").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance|hash|artifact"):
        service.run(run_id=RUN_ID, model_id="latest", strategy_id=STRATEGY_ID,
            start_date="2024-01-02", end_date="2024-01-05", topk=1, cost_bps=10)


def test_native_bad_v3_contract_does_not_fall_back_to_legacy_strategy(native_run):
    service, report, path, *_ = native_run
    report["replay_contract"]["config_hash"] = "0" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    row = service.catalog()["runs"][0]
    assert row["strategies"] == []
    assert row["custom_replay_available"] is False
