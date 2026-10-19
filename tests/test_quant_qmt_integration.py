"""DSA's default local data path delegates to quant_qmt_data."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient as ApiClient

from api.app import create_app
from quant_qmt_data import quant_data_contract as contract
from src.services import script_universe_loader as loader


def publish_prices(root, dataset="cn_index_daily", symbol="SZ399006", snapshot_id="index-a"):
    stage = root / "staging" / snapshot_id
    bars = stage / "bars"
    bars.mkdir(parents=True)
    frame = pd.DataFrame(dict(date=pd.to_datetime(["2026-08-03", "2026-08-04"]),
                              open=[10., 11.], high=[11., 12.], low=[9., 10.], close=[10., 11.],
                              volume=[100., 110.], amount=[1000., 1210.]))
    frame.to_parquet(bars / f"{symbol}.parquet", index=False)
    contract.finalize_price_snapshot(
        stage, dataset_id=dataset, snapshot_id=stage.name, role="canonical", status="ready",
        source={"provider": "fixture", "volume_unit": "shares"}, price_basis="derived",
        adjustment={"method": "none"}, producer={"script": "integration-test", "version": "1"},
        source_retrieved_at="2026-08-05T10:00:00+08:00")
    if not (root / "DATA_MANIFEST.json").exists():
        contract.write_root_manifest(root, {})
    return contract.publish_snapshot_group([(stage, root / "parquet/index_1d" / dataset)],
                                           quant_data_root=root, update_root_catalog=True)[0]


def test_status_uses_local_provider_without_old_service(tmp_path, monkeypatch):
    root = tmp_path / "data"
    publish_prices(root)
    monkeypatch.setenv("QUANT_DATA_ROOT", str(root))
    from quant_workflow.scripts.quant_data_service_client import QuantDataServiceClient
    monkeypatch.setattr(QuantDataServiceClient, "ready", lambda *_: pytest.fail("old service was called"))
    with ApiClient(create_app(static_dir=tmp_path / "static")) as client:
        response = client.get("/api/v1/data/status")
    assert response.status_code == 200
    status = response.json()
    assert status["backend"] == "quant_qmt_data"
    assert status["validation_scope"] == "snapshot_metadata"
    assert status["available_datasets"] == 1
    assert status["datasets"][0]["snapshot_id"] == "index-a"
    assert str(root) not in response.text


def test_benchmark_derives_symbol_from_provider_index_and_pins_version(tmp_path):
    root = tmp_path / "data"
    publish_prices(root)
    first = loader.load_benchmark_series(["SZ399006"], quant_data_root=root)["SZ399006"]
    assert first["symbol"].tolist() == ["SZ399006"] * 2
    assert first["close"].tolist() == [10., 11.]
    identity = first.attrs["dataset_identity"]
    publish_prices(root, snapshot_id="index-b")
    replay = loader.load_benchmark_series(["SZ399006"], quant_data_root=root,
                                          expected_snapshots={"cn_index_daily": identity})["SZ399006"]
    assert replay.attrs["dataset_identity"] == identity
    pd.testing.assert_frame_equal(first, replay)


def test_ordinary_read_follows_committed_root_not_early_latest(tmp_path):
    root = tmp_path / "data"
    current = publish_prices(root)
    pointer = current.parent.parent / "LATEST.json"
    pointer.write_text(json.dumps({"snapshot_id": "not-committed", "snapshot_path": "missing"}))
    snapshot = loader._read_snapshot_latest(root, "index_1d", "cn_index_daily")
    assert snapshot["snapshot_id"] == "index-a"


def test_etf_routing_uses_verified_price_index_and_rejects_ambiguous_bare_code():
    mapping = loader._etf_code_to_symbol({"file_index": pd.DataFrame({
        "symbol": ["SZ159915", "SH513100", "SH999999", "SZ999999"]})})
    assert mapping["159915"] == "SZ159915"
    assert mapping["SH513100"] == "SH513100"
    assert "999999" not in mapping


@pytest.mark.parametrize("dataset", ["cn_etf_daily", "ashare_daily_raw", "ashare_daily_qfq"])
def test_update_cli_delegates_to_provider(dataset, monkeypatch, tmp_path):
    from quant_workflow.scripts import quant_data_cli as cli
    import quant_qmt_data as provider
    calls = []
    def stock(**kwargs):
        calls.append(kwargs)
        return tmp_path / "raw", tmp_path / "qfq"
    def etf(**kwargs):
        calls.append(kwargs)
        kwargs["_result"].update(status="no_gap", failed_symbols=[])
        return tmp_path / "etf"
    monkeypatch.setattr(provider, "update_ashare_snapshot", stock)
    monkeypatch.setattr(provider, "update_cn_etf_snapshot", etf)
    assert cli._update(tmp_path, dataset, "2026-10-16", None) == 0
    assert len(calls) == 1
    assert calls[0]["quant_data_dir"] == tmp_path
    assert calls[0].get("through", calls[0].get("end_date")) == "2026-10-16"


def test_script_default_root_is_lightweight_store(monkeypatch, tmp_path):
    from src.services import quant_workflow_paths as paths
    from quant_workflow.scripts import _paths
    monkeypatch.delenv("QUANT_DATA_ROOT", raising=False)
    monkeypatch.setenv("WORKING_PROJECTS_ROOT", str(tmp_path))
    expected = tmp_path / "quant_qmt_data/data"
    assert paths.quant_data_root() == expected
    assert _paths.quant_data_root() == expected


def test_web_has_a_direct_stock_update_without_training_or_a_full_daily_pipeline():
    from src.services.quant_job_service import _build_default_command_specs, QUANT_DATA_NETWORK_JOB_TYPES
    spec = _build_default_command_specs()["update_ashare_miniqmt"]
    command = spec["command"]({"end": "2026-10-16"})
    assert command[-4:] == ["update", "ashare_daily_qfq", "--through", "2026-10-16"]
    assert command[1].endswith("quant_data_cli.py")
    assert spec["allowed_params"] == {"end"}
    assert {"update_ashare_miniqmt", "download_etf_momentum"}.issubset(QUANT_DATA_NETWORK_JOB_TYPES)


def test_qlib_consumer_reads_frozen_prices_through_package(tmp_path, monkeypatch):
    pytest.importorskip("qlib")
    from types import SimpleNamespace
    from collections import OrderedDict
    from quant_workflow.quant_data_qlib_provider import QuantDataFeatureProvider
    import quant_qmt_data as provider

    root = tmp_path / "data"
    publish_prices(root)
    snapshot = provider.bar_snapshot("cn_index_daily", quant_data_root=root)
    consumer = object.__new__(QuantDataFeatureProvider)
    consumer._context = SimpleNamespace(
        quant_data_root=root, snapshot_path=snapshot["snapshot_path"],
        index_snapshot_path=None, cleaning={"mode": "canonical"},
        descriptor={"dataset": snapshot["identity"]})
    consumer._rows = {row.symbol: row for row in snapshot["file_index"].itertuples()}
    consumer._index_rows, consumer._frames, consumer._cache_limit = {}, OrderedDict(), 8
    original = provider.read_bars
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs["expected_identity"])
        return original(*args, **kwargs)
    monkeypatch.setattr(provider, "read_bars", record)
    assert consumer._load_frame("SZ399006")["close"].tolist() == [10., 11.]
    assert calls == [snapshot["identity"]]


@pytest.mark.parametrize("stocks", [False, True])
def test_update_cli_runs_real_temporary_increment_and_zero_network_replay(tmp_path, monkeypatch, stocks):
    """Only the supplier transport is synthetic; snapshot writes/reads are real."""
    import hashlib
    import quant_qmt_data as provider
    from quant_qmt_data import miniqmt_etf_source as source
    from quant_qmt_data.miniqmt_bridge import MiniQMTBridgeClient, canonical_bytes
    from quant_workflow.scripts import quant_data_cli as cli

    monkeypatch.setenv("QUANT_WORKFLOW_REPORTS_ROOT", str(tmp_path / "reports"))
    calls = []
    days = ["2026-08-03", "2026-08-04", "2026-08-05"]
    def transport(request):
        calls.append(request)
        selected = [day for day in days if pd.Timestamp(request["start"]) <= pd.Timestamp(day) <= pd.Timestamp(request["end"])]
        if request["operation"] == "calendar":
            data = {symbol: {"dates": selected} for symbol in request["symbols"]}
        else:
            assert request["operation"] == "update_pair"
            rows = [dict(date=day, open=10., high=11., low=9., close=10., volume=100.,
                         amount=100000., suspendFlag=0, preClose=10.) for day in selected]
            data = {symbol: dict(raw=rows, qfq=rows, actions=[], calendar=selected,
                                  volume_unit="shares" if stocks else "lots",
                                  action_history_through=days[-1]) for symbol in request["symbols"]}
        return dict(status="ok", provider="miniqmt", request=request,
                    request_sha256=hashlib.sha256(canonical_bytes(request)).hexdigest(),
                    data=data, data_sha256=hashlib.sha256(canonical_bytes(data)).hexdigest(),
                    mode="read_existing_cache", network_download_requested=False)
    monkeypatch.setattr(source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=transport))
    dataset, symbol = ("ashare_daily_qfq", "SZ000001") if stocks else ("cn_etf_daily", "SZ159915")
    initialize = provider.initialize_ashare_snapshot if stocks else provider.initialize_cn_etf_snapshot
    initialize(quant_data_dir=tmp_path, codes=[symbol], start_date=days[0], end_date=days[1])
    old = provider.read_bars(dataset, symbol, quant_data_root=tmp_path, price_basis="qfq")
    frozen = provider.bar_snapshot(dataset, quant_data_root=tmp_path, price_basis="qfq")
    old_file = frozen["bars_dir"] / f"{symbol}.parquet"
    old_bytes = old_file.read_bytes()
    if not stocks:
        from datetime import date
        from src.repositories.quant_price_repo import QuantPriceRepository
        loaded = QuantPriceRepository(quant_data_root=tmp_path).load(
            [symbol], start_date=date(2026, 8, 3), end_date=date(2026, 8, 4))
        assert len(loaded.bars) == 2
        assert set(loaded.dataset_identities) == {"cn_etf_daily"}
    calls.clear()
    assert cli._update(tmp_path, dataset, days[-1], None) == 0
    assert any(item["operation"] == "update_pair" for item in calls)
    new = provider.read_bars(dataset, symbol, quant_data_root=tmp_path, price_basis="qfq")
    assert len(new) == len(old) + 1
    values = ["date", "open", "high", "low", "close", "volume", "amount"]
    pd.testing.assert_frame_equal(new.iloc[:len(old)][values].reset_index(drop=True), old[values])
    assert old_file.read_bytes() == old_bytes
    calls.clear()
    assert cli._update(tmp_path, dataset, days[-1], None) == 0
    assert calls == []
