"""Local backtests must use frozen Parquet inputs, without a data service."""

import json
from datetime import date

import pandas as pd
import pytest
from quant_qmt_data import quant_data_contract as contract

from src.services.script_universe_loader import load_market_view, load_benchmark_series
from src.repositories.quant_price_repo import QuantPriceRepository


def _snapshot(root, dataset_id, symbol, close, *, snapshot_id=None):
    family = "etf_1d" if dataset_id == "cn_etf_daily" else "ashare_1d"
    snapshot_id = snapshot_id or dataset_id + "-fixture"
    stage = root / "staging" / snapshot_id
    component = "qfq" if family == "etf_1d" else "bars"
    (stage / component).mkdir(parents=True)
    frame = pd.DataFrame({
        "symbol": [symbol, symbol], "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [close - .1, close], "high": [close + .1, close + .3],
        "low": [close - .2, close - .1], "close": [close, close + .2],
        "volume": [100., 120.], "amount": [close * 100., (close + .2) * 120.],
    })
    frame.to_parquet(stage / component / f"{symbol}.parquet", index=False)
    (stage / "factors").mkdir()
    pd.DataFrame({"date": frame["date"], "factor": 1., "offset": 0.}).to_parquet(
        stage / "factors" / f"{symbol}.parquet", index=False,
    )
    pd.DataFrame(columns=["symbol", "ex_date", "effective_trade_date", "action_type"]).to_parquet(
        stage / "corporate_actions.parquet", index=False,
    )
    common = dict(role="canonical", status="ready",
                  source={"provider": "fixture", "volume_unit": "shares"},
                  producer={"script": "test_local_snapshot_backtest", "version": "1"},
                  source_retrieved_at="2024-01-03T16:00:00+08:00")
    adjustment = dict(method="provider_native_paired", adjusted_at=common["source_retrieved_at"],
                      adjustment_as_of="2024-01-03", adjusted_fields=["open", "high", "low", "close"],
                      normalization_rule="latest_valid_trade_date_factor_equals_1_and_offset_equals_0",
                      algorithm_version="provider-native-affine-paired-v1")
    publications = []
    if family == "etf_1d":
        (stage / "raw").mkdir()
        frame.to_parquet(stage / "raw" / f"{symbol}.parquet", index=False)
        pd.DataFrame({"symbol": [symbol], "code": [symbol[2:]]}).to_parquet(
            stage / "ETF_UNIVERSE.parquet", index=False,
        )
        contract.finalize_paired_price_snapshot(
            stage, dataset_id=dataset_id, snapshot_id=snapshot_id, **common,
            adjustment={**adjustment, "paired_raw_component_path": "raw", "volume_adjusted": False},
        )
    else:
        raw = root / "staging" / (snapshot_id + "-raw")
        (raw / "bars").mkdir(parents=True)
        frame.to_parquet(raw / "bars" / f"{symbol}.parquet", index=False)
        contract.finalize_price_snapshot(
            raw, dataset_id="ashare_daily_raw", snapshot_id=raw.name, **common,
            price_basis="raw", adjustment={"method": "none"},
        )
        contract.finalize_price_snapshot(
            stage, dataset_id=dataset_id, snapshot_id=snapshot_id, **common,
            price_basis="qfq", adjustment={**adjustment, "paired_raw_snapshot_id": raw.name},
        )
        publications.append((raw, root / "parquet" / family / "ashare_daily_raw"))
    dataset_root = root / "parquet" / family / dataset_id
    publications.append((stage, dataset_root))
    if not (root / "DATA_MANIFEST.json").exists():
        contract.write_root_manifest(root, {})
    snapshot = contract.publish_snapshot_group(
        publications, quant_data_root=root, update_root_catalog=True,
    )[-1]
    return json.loads((dataset_root / "LATEST.json").read_text()), snapshot


@pytest.fixture
def local_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _snapshot(root, "ashare_daily_qfq", "SH600519", 10.)
    _snapshot(root, "cn_etf_daily", "SZ159915", 2.)
    (root / "ACTIVE_RELEASE_SET.json").write_text('{"archived": true}')
    monkeypatch.setenv("QUANT_DATA_ROOT", str(root))
    monkeypatch.setattr(
        "src.repositories.quant_price_repo.QuantDataServiceClient",
        lambda **kwargs: pytest.fail("the local default must not create a service client"),
    )
    return root


def test_default_etf_pool_reads_local_snapshot_beside_archived_active(local_root, tmp_path):
    view, meta = load_market_view("etf", start="2024-01-02", end="2024-01-03",
                                  cache_dir=tmp_path / "cache")
    assert view.close_at("SZ159915", 1) == 2.2
    assert meta["snapshot_id"] == "cn_etf_daily-fixture"
    assert meta["manifest_sha256"]
    assert "release_set_sha256" not in meta


def test_default_form_reader_preserves_both_local_snapshot_identities(local_root):
    result = QuantPriceRepository().load(["600519", "159915"],
                start_date=date(2024, 1, 2), end_date=date(2024, 1, 3))
    assert {(bar.symbol, bar.close) for bar in result.bars} == {
        ("SH600519", 10.), ("SH600519", 10.2), ("SZ159915", 2.), ("SZ159915", 2.2),
    }
    assert result.dataset_set_version != "production.v2"
    assert result.dataset_identities["ashare_daily_qfq"]["snapshot_id"] == "ashare_daily_qfq-fixture"
    assert result.dataset_identities["cn_etf_daily"]["snapshot_id"] == "cn_etf_daily-fixture"


def test_benchmark_reuses_pinned_snapshot_after_latest_changes(local_root, tmp_path):
    _, meta = load_market_view("custom", custom_symbols=["600519", "159915"],
        start="2024-01-02", end="2024-01-03", quant_data_root=local_root,
        cache_dir=tmp_path / "cache")
    _snapshot(local_root, "cn_etf_daily", "SZ159915", 8., snapshot_id="etf-next")
    frames = load_benchmark_series(["159915"], start="2024-01-02", end="2024-01-03",
                                  expected_snapshots=meta["datasets"])
    assert frames["SZ159915"]["close"].tolist() == [2., 2.2]


def test_changed_snapshot_manifest_is_rejected(local_root, tmp_path):
    path = local_root / "parquet/etf_1d/cn_etf_daily/snapshots/cn_etf_daily-fixture/DATASET_MANIFEST.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(contract.DataContractError, match="manifest hash mismatch"):
        load_market_view("etf", quant_data_root=local_root, cache_dir=tmp_path / "cache")


def test_cached_etf_custom_pool_keeps_its_actual_snapshot_identity(local_root, tmp_path, monkeypatch):
    from src.services import script_universe_loader as loader
    kwargs = dict(custom_symbols=["159915"], quant_data_root=local_root,
                  cache_dir=tmp_path / "cache")
    first, _ = loader.load_market_view("custom", **kwargs)
    monkeypatch.setattr(loader, "_load_bars_long", lambda *a, **kw: pytest.fail("a valid warm bundle must not reread bars"))
    second, meta = loader.load_market_view("custom", **kwargs)
    assert first.close.tolist() == second.close.tolist()
    assert meta["snapshot_id"] == "cn_etf_daily-fixture"


def test_changed_parquet_cannot_reuse_a_warm_bundle(local_root, tmp_path):
    kwargs = dict(quant_data_root=local_root, cache_dir=tmp_path / "cache")
    load_market_view("etf", **kwargs)
    path = local_root / "parquet/etf_1d/cn_etf_daily/snapshots/cn_etf_daily-fixture/qfq/SZ159915.parquet"
    frame = pd.read_parquet(path)
    frame["close"] = [3., 4.]
    frame.to_parquet(path, index=False)
    with pytest.raises(contract.DataContractError, match="indexed bar file is unreadable") as error:
        load_market_view("etf", **kwargs)
    assert isinstance(error.value.__cause__, contract.DataContractError)
    assert "hash mismatch" in str(error.value.__cause__)
