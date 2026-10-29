"""DSA consumers share frozen QMT bars, pools and benchmarks without the SDK."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient as ApiClient

from api.app import create_app
from data_provider.base import DataFetcherManager, DataFetchError
from data_provider.miniqmt_fetcher import QuantQMTDataFetcher
from src.services import script_universe_loader as loader
from tests.test_qmt_native_backtest import save_input


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    directory = save_input(tmp_path, missing=True)
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    return directory


def test_form_api_reads_mixed_stock_etf_input_with_truthful_identity(experiment, tmp_path):
    with ApiClient(create_app(static_dir=tmp_path / "static"),
                   headers={"Origin": "http://127.0.0.1:8010"}) as client:
        response = client.post("/api/v1/backtest/strategy/run", json={
            "strategy_id": "buy_and_hold", "symbols": ["600519", "159915"],
            "start_date": "2024-01-02", "end_date": "2024-01-05",
            "initial_capital": 10000, "side_cost_pct": .1,
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["symbols"] == ["SH600519", "SZ159915"]
    identity = json.loads((experiment / "dataset.json").read_text())
    quality = result["data_quality"]
    assert quality["experiment_id"] == identity["experiment_id"]
    assert quality["bars_sha256"] == identity["bars_sha256"]
    assert quality["inputs_sha256"] == identity["inputs_sha256"]
    assert "input_dir" not in quality
    assert quality["snapshot_id"] is None
    assert quality["dataset_set_version"] is None
    assert quality["price_basis"] == "front"
    assert quality["volume_units"]["SZ159915"] == "units"


def test_benchmark_replay_stays_on_same_input_after_environment_changes(experiment, tmp_path, monkeypatch):
    _, meta = loader.load_market_view("custom", custom_symbols=["600519", "159915"],
        start="2024-01-02", end="2024-01-05", cache_dir=tmp_path / "bundles")
    first = loader.load_benchmark_series(["SH000300", "159915"], input_dir=meta["input_dir"])
    replacement = save_input(tmp_path / "another", scale=2.)
    monkeypatch.setenv("QMT_DATASET_DIR", str(replacement))
    replay = loader.load_benchmark_series(["SH000300", "159915"], input_dir=meta["input_dir"])
    for symbol in first:
        pd.testing.assert_frame_equal(first[symbol], replay[symbol])
        assert replay[symbol].attrs["experiment_id"] == meta["experiment_id"]


def test_ordinary_history_uses_selected_input_instead_of_old_data_root(experiment, tmp_path):
    manager = DataFetcherManager(fetchers=[QuantQMTDataFetcher()])
    frame, source = manager.get_daily_data("159915", "2024-01-02", "2024-01-05")
    assert source == "QuantQMTDataFetcher"
    assert frame["close"].tolist() == [2., 2.4, 2.6]
    assert frame["volume"].tolist() == [1000., 1000., 1000.]
    assert frame.attrs["experiment_id"] == json.loads((experiment / "dataset.json").read_text())["experiment_id"]


def test_stock_etf_and_index_codes_share_frozen_input(experiment, tmp_path):
    view, meta = loader.load_market_view("custom", custom_symbols=["600519.SH", "159915.SZ"],
                                        cache_dir=tmp_path / "bundles")
    assert list(view.symbols) == ["SH600519", "SZ159915"]
    benchmark = loader.load_benchmark_series(["000300.SH"], input_dir=meta["input_dir"])["SH000300"]
    assert benchmark.symbol.eq("SH000300").all()
    assert benchmark["volume"].tolist() == [1000.] * 4
    assert benchmark.attrs["volume_units"]["SH000300"] == "native"
    assert "SH000300" not in view.symbols


def test_missing_symbol_does_not_fall_back_to_a_supplier(experiment):
    manager = DataFetcherManager(fetchers=[QuantQMTDataFetcher()])
    with pytest.raises(DataFetchError):
        manager.get_daily_data("920149", "2024-01-02", "2024-01-05")
