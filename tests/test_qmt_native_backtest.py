"""Frozen QMT inputs exercise the real DSA view and portfolio engines."""
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_qmt_data import QMTClient, load_dataset, save_dataset
from src.core.strategy_backtest import (
    PriceBar, StrategyBacktestConfig, TargetAllocation,
    run_portfolio_backtest, run_portfolio_backtest_view,
)


def save_input(tmp_path, *, missing=False, adjustment="front", scale=1.0,
               universe_kind="custom", membership_intervals=None):
    days = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for symbol, initial in [("SH600519", 10.0), ("SZ159915", 2.0), ("SH000300", 100.0)]:
        for i, day in enumerate(days):
            if missing and symbol == "SZ159915" and i == 1:
                continue
            close = initial * (1 + i / 10) * scale
            rows.append(dict(date=day, symbol=symbol, open=close, high=close,
                             low=close, close=close, volume=1000.0, amount=10000.0,
                             preclose=close, suspend_flag=0))
    frame = pd.DataFrame(rows)
    frame.attrs = dict(
        provider="qmt", request=dict(symbols=list(frame.symbol.unique()),
            start="2024-01-02", end="2024-01-05", adjustment=adjustment, volume_unit="base"),
        price_basis=adjustment, volume_units={"SH600519": "shares", "SZ159915": "units", "SH000300": "native"},
        missing_symbols=[], read_at="2024-01-06T08:00:00+08:00",
        completed_through="2024-01-05", contains_partial=False,
        qmt_data_dir=str(tmp_path / "qmt-cache"), sdk_version="synthetic",
        actual_start="2024-01-02", actual_end="2024-01-05",
        benchmark_symbols=["SH000300"],
    )
    directory = tmp_path / "input"
    selected = ["SH600519"] if universe_kind == "historical" else ["SH600519", "SZ159915"]
    save_dataset(frame, directory, calendar=days, universe=selected,
                 universe_kind=universe_kind, membership_intervals=membership_intervals)
    return directory


@pytest.fixture
def frozen_input(tmp_path, monkeypatch):
    directory = save_input(tmp_path, missing=True)
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    return directory


def test_one_historical_membership_shape_serves_training_and_backtest(tmp_path):
    from quant_workflow.quant_data_qlib_provider import QuantDataInstrumentProvider
    from src.services.script_universe_loader import load_market_view
    directory = save_input(tmp_path, universe_kind="historical", membership_intervals=[
        {"symbol": "SH600519", "start_date": "2024-01-03", "end_date": "2024-01-04"},
    ])
    view, _ = load_market_view("csi800", quant_data_root=directory,
                               cache_dir=tmp_path / "bundles")
    provider = QuantDataInstrumentProvider(input_dir=directory)
    spans = provider.list_instruments({"market": "all"})
    assert spans["SH600519"] == [(pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"))]
    assert view.members[:, 0].tolist() == [False, True, True, False]
    custom_view, _ = load_market_view("custom", custom_symbols=["SH600519"],
                                      quant_data_root=directory, cache_dir=tmp_path / "custom")
    assert custom_view.members[:, 0].tolist() == view.members[:, 0].tolist()


def test_frozen_input_preserves_calendar_nan_units_and_actual_identity(frozen_input, tmp_path):
    from src.services.script_universe_loader import load_market_view
    view, meta = load_market_view("custom", custom_symbols=["600519", "159915"],
                                  start="2024-01-02", end="2024-01-05", cache_dir=tmp_path / "bundles")
    assert view.calendar == tuple(date(2024, 1, d) for d in (2, 3, 4, 5))
    assert list(view.symbols) == ["SH600519", "SZ159915"]
    assert np.isnan(view.close[1, 1])
    assert not view.members[1, 1]
    assert view.volume[0, 1] == 1000.0
    metadata = json.loads((frozen_input / "dataset.json").read_text())
    assert meta["experiment_id"] == metadata["experiment_id"]
    assert meta["input_dir"] == str(frozen_input.resolve())
    assert meta["bars_sha256"] == metadata["bars_sha256"]
    assert meta["price_basis"] == "front"
    assert meta.get("snapshot_id") is None
    assert meta.get("dataset_set_version") not in ("production.v2", "local_snapshots")
    assert np.isnan(view.index_close).all()
    assert (Path(meta["bundle_dir"]) / "market.npz").is_file()


def test_explicit_input_and_benchmark_reuse_ignore_sdk_and_other_environment(frozen_input, tmp_path, monkeypatch):
    from src.services.script_universe_loader import load_market_view, load_benchmark_series
    monkeypatch.setenv("QMT_DATASET_DIR", str(tmp_path / "unrelated"))
    _, meta = load_market_view("etf", quant_data_root=frozen_input, cache_dir=tmp_path / "bundles")
    frames = load_benchmark_series(["SH000300", "159915"], input_dir=meta["input_dir"],
                                   start="2024-01-02", end="2024-01-05")
    assert frames["SH000300"]["close"].tolist() == pytest.approx([100., 110., 120., 130.])
    assert frames["SZ159915"]["close"].tolist() == [2., 2.4, 2.6]
    assert frames["SH000300"].attrs["experiment_id"] == meta["experiment_id"]


@pytest.mark.parametrize("damage", ["bars", "json", "incomplete"])
def test_damaged_frozen_input_never_falls_back_to_live(frozen_input, tmp_path, damage):
    from src.services.script_universe_loader import MarketDataUnavailable, load_market_view
    if damage == "bars":
        with (frozen_input / "bars.parquet").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "json":
        (frozen_input / "dataset.json").write_text("{incomplete", encoding="utf-8")
    else:
        (frozen_input / "dataset.json").unlink()
    with pytest.raises(MarketDataUnavailable, match="input|输入"):
        load_market_view("etf", cache_dir=tmp_path / "bundles")


def test_missing_requested_symbol_is_not_silently_removed(frozen_input, tmp_path):
    from src.services.script_universe_loader import MarketDataUnavailable, load_market_view
    with pytest.raises(MarketDataUnavailable, match="BJ920149"):
        load_market_view("custom", custom_symbols=["SH600519", "BJ920149"], cache_dir=tmp_path / "bundles")


def test_csi800_requires_actual_frozen_historical_membership(frozen_input, tmp_path):
    from src.services.script_universe_loader import MarketDataUnavailable, load_market_view
    with pytest.raises(MarketDataUnavailable, match="CSI800|historical|历史"):
        load_market_view("csi800", cache_dir=tmp_path / "bundles")


def test_historical_membership_is_applied_to_the_frozen_calendar(tmp_path, monkeypatch):
    from src.services.script_universe_loader import load_market_view
    directory = save_input(tmp_path, universe_kind="historical", membership_intervals=[
        {"symbol": "SH600519", "start_date": "2024-01-03", "end_date": "2024-01-04"},
    ])
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    view, _ = load_market_view("csi800", cache_dir=tmp_path / "bundles")
    assert view.members[:, 0].tolist() == [False, True, True, False]


@pytest.mark.parametrize("pool, selected", [("custom", ["SH600519", "BJ920149"]),
    ("all_a", ["SH600519", "BJ920149"]), ("etf", ["SH510300"])])
@pytest.mark.parametrize("benchmark", ["SH000300", "000300.SH"])
def test_live_read_freezes_selected_calendar_and_benchmark_once(pool, selected, benchmark, tmp_path, monkeypatch):
    from src.services import script_universe_loader as loader
    monkeypatch.delenv("QMT_DATASET_DIR", raising=False)
    source = load_dataset(save_input(tmp_path / "source"))
    calls = []
    def read(symbols, **kwargs):
        calls.append(("bars", list(symbols), kwargs))
        frames = []
        for symbol in symbols:
            item = source.loc[source.symbol.eq("SH600519")].copy()
            item["symbol"] = symbol
            frames.append(item)
        frame = pd.concat(frames, ignore_index=True)
        frame.attrs = {**source.attrs, "request": {**source.attrs["request"], "symbols": symbols},
                       "volume_units": {s: "units" if s == "SH510300" else "shares" for s in symbols}}
        return frame
    def universe(sector):
        calls.append(("universe", sector))
        return ["SH510300"] if "ETF" in sector else ["BJ920149"] if "京" in sector else ["SH600519"]
    def calendar(market, **kwargs):
        calls.append(("calendar", market))
        return source.attrs["calendar"]
    monkeypatch.setattr(QMTClient, "from_env", classmethod(lambda cls: SimpleNamespace(
        read_bars=read, trading_dates=calendar, universe=universe)))
    view, meta = loader.load_market_view(pool, custom_symbols=selected if pool == "custom" else None,
        start="2024-01-02", end="2024-01-05", benchmark_symbols=[benchmark], cache_dir=tmp_path / "bundles")
    frozen = load_dataset(meta["input_dir"])
    assert frozen.attrs["universe"] == selected
    assert frozen.attrs["universe_kind"] == ("custom" if pool == "custom" else "current")
    assert "SH000300" in frozen.attrs["benchmark_symbols"]
    assert "SH000300" not in list(view.symbols)
    assert len([call for call in calls if call[0] == "bars"]) == 1
    before = list(calls)
    benchmarks = loader.load_benchmark_series(["SH000300"], input_dir=meta["input_dir"])
    assert not benchmarks["SH000300"].empty
    assert calls == before


def test_form_execution_uses_full_view_calendar(frozen_input, monkeypatch):
    from src.services import backtest_service as module
    allocation = TargetAllocation("test", date(2024, 1, 2), {"SH600519": 1.0}, "known signal")
    monkeypatch.setattr(module, "build_buy_and_hold_allocations", lambda **kwargs: [allocation])
    report = module.BacktestService.__new__(module.BacktestService).run_strategy(
        strategy_id="buy_and_hold", symbols=["600519", "159915"],
        start_date=date(2024, 1, 2), end_date=date(2024, 1, 5), initial_capital=10000.)
    assert report["trades"][0]["execution_date"] == "2024-01-03"
    assert report["data_quality"]["experiment_id"]
    assert report["data_quality"]["snapshot_id"] is None
    assert report["strategy"]["execution"] == "next_trading_day_open"


def test_live_empty_calendar_does_not_save_a_formal_input(tmp_path, monkeypatch):
    from src.services.script_universe_loader import MarketDataUnavailable, load_market_view
    source = load_dataset(save_input(tmp_path / "source"))
    report_dir = tmp_path / "reports"
    monkeypatch.setenv("QUANT_WORKFLOW_REPORTS_ROOT", str(report_dir))
    monkeypatch.delenv("QMT_DATASET_DIR", raising=False)
    monkeypatch.setattr(QMTClient, "from_env", classmethod(lambda cls: SimpleNamespace(
        read_bars=lambda *args, **kwargs: source,
        trading_dates=lambda *args, **kwargs: [])))
    with pytest.raises(MarketDataUnavailable):
        load_market_view("custom", custom_symbols=["600519"], start="2024-01-02", end="2024-01-05")
    assert list((report_dir / "inputs").glob("*/dataset.json")) == []


def test_complete_data_has_equivalent_existing_engine_results(tmp_path, monkeypatch):
    from src.services.script_universe_loader import load_market_view
    directory = save_input(tmp_path)
    view, _ = load_market_view("custom", custom_symbols=["600519"], quant_data_root=directory,
                               cache_dir=tmp_path / "bundles")
    allocations = [TargetAllocation("buy", date(2024, 1, 1), {"SH600519": 1.0}, "buy"),
                   TargetAllocation("sell", date(2024, 1, 4), {}, "sell")]
    bars = [PriceBar("SH600519", day, float(view.open[i, 0]), float(view.close[i, 0]))
            for i, day in enumerate(view.calendar)]
    config = StrategyBacktestConfig(initial_capital=10000., side_cost_pct=0.1, lot_size=100)
    original = run_portfolio_backtest(strategy_name="same", allocations=allocations, bars=bars, config=config)
    vector = run_portfolio_backtest_view(strategy_name="same", allocations=allocations, view=view, config=config)
    for field in ("final_equity", "total_return_pct", "total_cost", "trade_count", "max_drawdown_pct"):
        assert vector["metrics"][field] == pytest.approx(original["metrics"][field])
    assert vector["trades"] == original["trades"]


@pytest.mark.parametrize("basis, prices, shares", [("front", [50., 50., 55., 55.], 200),
                                                 ("back", [100., 100., 110., 110.], 100)])
def test_synthetic_ex_date_research_price_units_cash_and_fee(basis, prices, shares, tmp_path, monkeypatch):
    from src.services.backtest_service import BacktestService
    directory = save_input(tmp_path / "initial", adjustment=basis)
    frame = load_dataset(directory)
    frame = frame.loc[frame.symbol.eq("SH600519")].copy()
    for field in ("open", "high", "low", "close"):
        frame[field] = prices
    frame.attrs["synthetic_case"] = "two-for-one split; research prices, not a corporate-action ledger"
    directory = save_dataset(frame, tmp_path / "split-input", calendar=frame.attrs["calendar"], universe=["SH600519"])
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    report = BacktestService.__new__(BacktestService).run_strategy(strategy_id="buy_and_hold",
        symbols=["600519"], start_date=date(2024, 1, 2), end_date=date(2024, 1, 5),
        initial_capital=10100., side_cost_pct=1., lot_size=100)
    trade = report["trades"][0]
    assert trade["shares"] == shares
    assert trade["notional"] == 10000.
    assert trade["fee"] == 100.
    assert report["metrics"]["final_equity"] == 11000.
    assert report["assumptions"]["price_basis"] == basis
    assert report["assumptions"]["execution_basis"] == "research_price_simulation"


def test_historical_fetcher_reads_same_input_without_volume_multiplication(frozen_input):
    from data_provider.miniqmt_fetcher import QuantQMTDataFetcher
    fetcher = QuantQMTDataFetcher()
    raw = fetcher._fetch_raw_data("159915", "2024-01-02", "2024-01-05")
    result = fetcher._normalize_data(raw, "159915")
    assert result["volume"].tolist() == [1000., 1000., 1000.]
    assert raw.attrs["experiment_id"]


def test_manager_registers_frozen_input_without_legacy_catalog(frozen_input, monkeypatch):
    from data_provider import DataFetcherManager
    monkeypatch.setattr("data_provider.miniqmt_bridge.MiniQMTBridgeClient.configured", lambda: False)
    manager = DataFetcherManager()
    frame, source = manager.get_daily_data("600519", start_date="2024-01-02", end_date="2024-01-05")
    assert source == "QuantQMTDataFetcher"
    assert len(frame) == 4


def test_warm_bundle_cannot_hide_changed_input(frozen_input, tmp_path):
    from src.services.script_universe_loader import MarketDataUnavailable, load_market_view
    load_market_view("etf", cache_dir=tmp_path / "bundles")
    with (frozen_input / "bars.parquet").open("ab") as stream:
        stream.write(b"changed after bundle creation")
    with pytest.raises(MarketDataUnavailable):
        load_market_view("etf", cache_dir=tmp_path / "bundles")


def test_in_process_script_output_uses_same_input_and_benchmark(frozen_input, monkeypatch):
    """Exercise strategy evaluation; OS sandbox policy is tested separately."""
    from src.core.script_market import MarketView
    from src.core.script_strategy import execute_strategy_script
    from src.services import backtest_service as module
    def evaluate(source, *, bundle_dir, start_date, end_date, **kwargs):
        result = execute_strategy_script(source, view=MarketView.load(bundle_dir),
                                         start_date=start_date, end_date=end_date)
        return result.allocations, result.logs, result.check
    monkeypatch.setattr(module, "run_script_in_sandbox", evaluate)
    report = module.BacktestService.__new__(module.BacktestService).run_script_strategy(
        script="def on_bar(context, data):\n    context.target_equal(['SH600519'])",
        pool="custom", symbols=["600519", "159915"], benchmarks=["SH000300"],
        start_date=date(2024, 1, 2), end_date=date(2024, 1, 5), initial_capital=10000.)
    assert report["trades"][0]["execution_date"] == "2024-01-03"
    assert report["data_quality"]["experiment_id"]
    assert report["assumptions"]["price_basis"] == "front"
    assert report["strategy"]["execution"] == "next_trading_day_open"


@pytest.mark.parametrize("path, method, payload", [
    ("/strategy/run", "run_strategy", {"strategy_id": "buy_and_hold", "symbols": ["600519"]}),
    ("/strategy/script", "run_script_strategy", {"script": "def on_bar(context, data):\n    pass"}),
    ("/strategy/script/stability", "run_script_stability_test", {"script": "def on_bar(context, data):\n    pass"}),
])
def test_api_returns_controlled_unavailable_for_native_input(path, method, payload, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from api.app import create_app
    from src.services.backtest_service import BacktestService
    from src.services.script_universe_loader import MarketDataUnavailable
    def unavailable(*args, **kwargs):
        raise MarketDataUnavailable("private input path")
    monkeypatch.setattr(BacktestService, method, unavailable)
    with TestClient(create_app(static_dir=tmp_path / "static"),
                    headers={"Origin": "http://127.0.0.1:8010"}) as client:
        response = client.post("/api/v1/backtest" + path, json=payload)
    assert response.status_code == 503
    assert response.json()["error"] == "market_data_unavailable"
    assert "private input path" not in response.text
