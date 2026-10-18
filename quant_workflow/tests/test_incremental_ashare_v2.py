from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bars(symbol: str, closes: list[float], dates: list[str]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "preclose": close.shift(1).fillna(close - 0.1),
            "volume": 1000.0,
            "amount": close * 1000.0,
            "turn": 0.1,
            "pctChg": 0.0,
            "tradestatus": 1,
            "isST": 0,
            "symbol": symbol,
            "source_id": "baostock_raw",
        }
    )


def _tickflow_metadata_fixture():
    raw = _bars("SH600519", [10.0, 11.0], ["2026-08-13", "2026-08-14"])
    factors = pd.DataFrame({
        "date": raw.date, "factor": [1.0, 1.0],
        "symbol": ["SH600519", None], "offset": [0.0, float("nan")],
        "factor_source": ["tickflow_adj_factor", "tickflow_adj_factor"],
    })
    return raw, raw.copy(), factors


def test_restore_tickflow_metadata_does_not_change_prices_or_known_fields():
    mod = load_script("incremental_update_ashare_v2")
    raw, qfq, factors = _tickflow_metadata_fixture()
    before = factors.copy(deep=True)
    restored = mod.restore_tickflow_factor_metadata(raw, qfq, factors, symbol="SH600519")
    assert restored.symbol.tolist() == ["SH600519", "SH600519"]
    assert restored.offset.tolist() == [0.0, 0.0]
    pd.testing.assert_frame_equal(factors, before)
    pd.testing.assert_frame_equal(restored.drop(columns=["symbol", "offset"]),
                                  before.drop(columns=["symbol", "offset"]))
    pd.testing.assert_frame_equal(raw, qfq)
    pd.testing.assert_frame_equal(
        mod.restore_tickflow_factor_metadata(raw, qfq, restored, symbol="SH600519"),
        restored,
    )


@pytest.mark.parametrize("damage", ["changed_price", "wrong_raw_symbol", "wrong_qfq_symbol",
    "wrong_factor_symbol", "different_dates", "duplicate_dates", "unknown_source",
    "nonfinite_price", "nonpositive_factor", "wrong_known_offset"])
def test_restore_tickflow_metadata_cannot_invent_unproven_identity_or_offset(damage):
    mod = load_script("incremental_update_ashare_v2")
    raw, qfq, factors = _tickflow_metadata_fixture()
    if damage == "changed_price":
        qfq.loc[1, "close"] += 0.01
    elif damage.startswith("wrong_") and damage.endswith("symbol"):
        {"wrong_raw_symbol": raw, "wrong_qfq_symbol": qfq,
         "wrong_factor_symbol": factors}[damage].loc[1, "symbol"] = "SH600000"
    elif damage == "different_dates":
        factors.loc[1, "date"] = pd.Timestamp("2026-08-17")
    elif damage == "duplicate_dates":
        for frame in (raw, qfq, factors):
            frame.loc[1, "date"] = frame.loc[0, "date"]
    elif damage == "unknown_source":
        factors.loc[1, "factor_source"] = "unverified_affine_fit"
    elif damage == "nonfinite_price":
        qfq.loc[1, "open"] = float("nan")
    elif damage == "nonpositive_factor":
        factors.loc[1, "factor"] = 0.0
    elif damage == "wrong_known_offset":
        factors.loc[1, "offset"] = 0.1
    with pytest.raises(mod.DataContractError):
        mod.restore_tickflow_factor_metadata(raw, qfq, factors, symbol="SH600519")


def test_restore_tickflow_metadata_keeps_optional_columns_absent():
    mod = load_script("incremental_update_ashare_v2")
    raw, qfq, factors = _tickflow_metadata_fixture()
    factors = factors.drop(columns=["symbol", "offset"])
    pd.testing.assert_frame_equal(
        mod.restore_tickflow_factor_metadata(raw, qfq, factors, symbol="SH600519"),
        factors,
    )


def _events(symbol: str, first_factor: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": [f"sh.{symbol[2:]}", f"sh.{symbol[2:]}"],
            "dividOperateDate": pd.to_datetime(["2026-08-01", "2026-08-04"]),
            "foreAdjustFactor": [first_factor, 1.0],
            "backAdjustFactor": [1.0, 1.0],
            "adjustFactor": [1.0, 1.0],
            "symbol": symbol,
            "source_id": "baostock_query_adjust_factor",
        }
    )


def _publish_base(root: Path) -> None:
    contract = load_script("quant_data_contract")
    dates = ["2026-08-01", "2026-08-04"]
    raw_stage = root / "staging" / "base" / "raw-base"
    qfq_stage = root / "staging" / "base" / "qfq-base"
    (raw_stage / "bars").mkdir(parents=True, exist_ok=True)
    (qfq_stage / "bars").mkdir(parents=True, exist_ok=True)
    (qfq_stage / "factors").mkdir(parents=True, exist_ok=True)
    actions: list[pd.DataFrame] = []
    for symbol, factor in (("SH000001", 0.5), ("SH000002", 0.8)):
        raw = _bars(symbol, [10.0, 11.0], dates)
        qfq = raw.copy()
        qfq[["open", "high", "low", "close", "preclose"]] = qfq[
            ["open", "high", "low", "close", "preclose"]
        ].mul([factor, factor, factor, factor, factor], axis="columns")
        qfq["source_id"] = "baostock_raw_x_foreAdjustFactor"
        qfq["factor"] = [factor, 1.0]
        factors = pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "source_event_date": pd.to_datetime(dates),
                "factor": [factor, 1.0],
                "factor_source": "baostock_foreAdjustFactor",
                "factor_backfilled_before_first_event": False,
                "factor_paired_fallback": False,
                "factor_terminal_normalized": False,
                "provider_qfq_terminal_factor": float("nan"),
            }
        )
        raw.to_parquet(raw_stage / "bars" / f"{symbol}.parquet", index=False)
        qfq.to_parquet(qfq_stage / "bars" / f"{symbol}.parquet", index=False)
        factors.to_parquet(qfq_stage / "factors" / f"{symbol}.parquet", index=False)
        actions.append(_events(symbol, factor).assign(action_type="adjustment_factor_event"))
    actions_frame = pd.concat(actions, ignore_index=True)
    actions_frame.to_parquet(raw_stage / "corporate_actions.parquet", index=False)
    actions_frame.to_parquet(qfq_stage / "corporate_actions.parquet", index=False)
    contract.finalize_price_snapshot(
        raw_stage,
        dataset_id="ashare_daily_raw",
        snapshot_id="raw-base",
        role="canonical",
        status="ready",
        source={"provider": "baostock", "interface": "test"},
        price_basis="raw",
        adjustment={"method": "none"},
        producer={"script": "test", "version": "1"},
        source_retrieved_at="2026-08-04T18:00:00+08:00",
    )
    contract.finalize_price_snapshot(
        qfq_stage,
        dataset_id="ashare_daily_qfq",
        snapshot_id="qfq-base",
        role="canonical",
        status="ready",
        source={"provider": "baostock", "interface": "test"},
        price_basis="qfq",
        adjustment={
            "method": "local_factor",
            "adjusted_at": "2026-08-04T18:00:00+08:00",
            "adjustment_as_of": "2026-08-04",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_factor_equals_1",
            "algorithm_version": "test-v1",
        },
        producer={"script": "test", "version": "1"},
        source_retrieved_at="2026-08-04T18:00:00+08:00",
        upstream_snapshots=[
            {"dataset_id": "ashare_daily_raw", "snapshot_id": "raw-base"}
        ],
    )
    raw_root = root / "parquet/ashare_1d/ashare_daily_raw"
    qfq_root = root / "parquet/ashare_1d/ashare_daily_qfq"
    raw_published, qfq_published = contract.publish_snapshot_group(
        [(raw_stage, raw_root), (qfq_stage, qfq_root)],
        quant_data_root=root,
    )
    contract.write_root_manifest(
        root,
        {
            "ashare_daily_raw": contract.catalog_entry_from_snapshot(
                raw_published, quant_data_root=root, dataset_root=raw_root
            ),
            "ashare_daily_qfq": contract.catalog_entry_from_snapshot(
                qfq_published, quant_data_root=root, dataset_root=qfq_root
            ),
        },
    )


def _write_bundle(root: Path) -> Path:
    source = load_script("download_baostock_v2")
    bundle = root / "bundle"
    dates = ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
    index_rows: list[dict[str, object]] = []
    for symbol, factor in (("SH000001", 0.5), ("SH000002", 0.4)):
        raw_path = bundle / "raw" / f"{symbol}.parquet"
        event_path = bundle / "factor_events" / f"{symbol}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        raw = _bars(symbol, [11.0, 11.2, 11.3, 11.4], dates)
        raw.to_parquet(raw_path, index=False)
        events = _events(symbol, factor)
        events.to_parquet(event_path, index=False)
        index_rows.append(
            {
                "symbol": symbol,
                "provider_code": f"sh.{symbol[2:]}",
                "raw_path": f"raw/{symbol}.parquet",
                "qfq_provider_path": None,
                "factor_events_path": f"factor_events/{symbol}.parquet",
                "raw_rows": len(raw),
                "factor_event_rows": len(events),
                "first_date": "2026-08-04",
                "last_date": "2026-08-07",
                "raw_sha256": source.sha256_file(raw_path),
                "factor_events_sha256": source.sha256_file(event_path),
                "qfq_provider_sha256": None,
            }
        )
    index = pd.DataFrame(index_rows)
    source.atomic_parquet(index, bundle / "BUNDLE_FILE_INDEX.parquet")
    source.atomic_json(
        bundle / "BUNDLE_MANIFEST.json",
        {
            "schema_version": 2,
            "bundle_kind": "baostock_raw_factor_qfq",
            "bundle_mode": "incremental",
            "status": "ready",
            "source_retrieved_at": "2026-08-07T18:00:00+08:00",
            "requested_start": "2026-07-27",
            "requested_through": "2026-08-07",
            "factor_requested_start": "1990-01-01",
            "base_latest_trade_date": "2026-08-04",
            "requested_symbols": 2,
            "completed_symbols": 2,
            "empty_symbols": 0,
            "failed_symbols": 0,
            "index_path": "BUNDLE_FILE_INDEX.parquet",
        },
    )
    return bundle


def _write_provider_paired_bundle(root: Path) -> Path:
    source = load_script("download_baostock_v2")
    bundle = root / "provider-paired-bundle"
    qfq_pointer = json.loads(
        (root / "parquet/ashare_1d/ashare_daily_qfq/LATEST.json").read_text(
            encoding="utf-8"
        )
    )
    dates = ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]
    index_rows: list[dict[str, object]] = []
    for symbol, provider_scale in (("SH000001", 1.0), ("SH000002", 0.4)):
        raw_path = bundle / "raw" / f"{symbol}.parquet"
        qfq_path = bundle / "qfq_provider" / f"{symbol}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        qfq_path.parent.mkdir(parents=True, exist_ok=True)
        raw = _bars(symbol, [11.0, 11.2, 11.3, 11.4], dates)
        provider_qfq = raw.copy()
        provider_qfq[["open", "high", "low", "close", "preclose"]] = provider_qfq[
            ["open", "high", "low", "close", "preclose"]
        ].mul(provider_scale)
        provider_qfq["source_id"] = "eastmoney_qfq_provider"
        raw["source_id"] = "eastmoney_raw"
        raw.to_parquet(raw_path, index=False)
        provider_qfq.to_parquet(qfq_path, index=False)
        index_rows.append(
            {
                "symbol": symbol,
                "provider_code": f"sh.{symbol[2:]}",
                "raw_path": f"raw/{symbol}.parquet",
                "qfq_provider_path": f"qfq_provider/{symbol}.parquet",
                "qfq_provider_sha256": source.sha256_file(qfq_path),
                "raw_rows": len(raw),
                "first_date": "2026-08-04",
                "last_date": "2026-08-07",
                "raw_sha256": source.sha256_file(raw_path),
            }
        )
    source.atomic_parquet(pd.DataFrame(index_rows), bundle / "BUNDLE_FILE_INDEX.parquet")
    source.atomic_json(
        bundle / "BUNDLE_MANIFEST.json",
        {
            "schema_version": 3,
            "bundle_kind": "provider_paired_raw_qfq",
            "bundle_mode": "incremental",
            "status": "ready",
            "source": {
                "provider": "eastmoney",
                "interface": "push2his_kline_fqt_0_and_fqt_1",
            },
            "source_retrieved_at": "2026-08-07T18:00:00+08:00",
            "base_qfq_snapshot_id": qfq_pointer["snapshot_id"],
            "base_qfq_manifest_sha256": qfq_pointer["manifest_sha256"],
            "requested_start": "2026-07-27",
            "requested_through": "2026-08-07",
            "base_latest_trade_date": "2026-08-04",
            "requested_symbols": 2,
            "completed_symbols": 2,
            "empty_symbols": 0,
            "failed_symbols": 0,
            "index_path": "BUNDLE_FILE_INDEX.parquet",
        },
    )
    return bundle


def test_incremental_price_import_accepts_provider_native_paired_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    module = load_script("incremental_update_ashare_v2")

    result = module.build_incremental_price_snapshots(
        bundle_dir=bundle,
        quant_data_root=root,
    )

    qfq_manifest = json.loads(
        (result.qfq_stage / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert qfq_manifest["source"]["provider"] == "eastmoney"
    assert qfq_manifest["adjustment"]["method"] == "provider_native_paired"
    assert qfq_manifest["adjustment"]["paired_raw_snapshot_id"] == result.raw_stage.name
    qfq = pd.read_parquet(result.qfq_stage / "bars/SH000002.parquet")
    assert qfq.loc[qfq["date"] == "2026-08-05", "close"].iloc[0] == pytest.approx(8.96)


def test_provider_native_paired_bundle_requires_qfq_for_every_symbol(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    index.loc[0, "qfq_provider_path"] = None
    index.loc[0, "qfq_provider_sha256"] = None
    index.to_parquet(index_path, index=False)
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="qfq audit file is required"):
        module.build_incremental_price_snapshots(bundle_dir=bundle, quant_data_root=root)


def test_provider_native_paired_bundle_rejects_raw_qfq_date_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    source = load_script("download_baostock_v2")
    symbol = "SH000001"
    qfq_path = bundle / "qfq_provider" / f"{symbol}.parquet"
    qfq = pd.read_parquet(qfq_path).iloc[:-1].copy()
    qfq.to_parquet(qfq_path, index=False)
    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    selected = index["symbol"].eq(symbol)
    index.loc[selected, "qfq_provider_sha256"] = source.sha256_file(qfq_path)
    index.to_parquet(index_path, index=False)
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="paired raw/qfq dates"):
        module.build_incremental_price_snapshots(bundle_dir=bundle, quant_data_root=root)


def test_provider_native_paired_bundle_rejects_different_base_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_qfq_snapshot_id"] = "different-qfq-base"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="current qfq snapshot identity"):
        module.build_incremental_price_snapshots(bundle_dir=bundle, quant_data_root=root)


def test_provider_native_paired_result_reports_actual_latest_trade_date(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requested_through"] = "2026-08-08"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = load_script("incremental_update_ashare_v2")

    result = module.build_incremental_price_snapshots(
        bundle_dir=bundle,
        quant_data_root=root,
    )

    assert result.latest_trade_date == "2026-08-07"
    qfq_manifest = json.loads(
        (result.qfq_stage / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert qfq_manifest["adjustment"]["adjustment_as_of"] == "2026-08-07"


def test_provider_native_paired_accepts_cent_rounded_provider_prices(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    source = load_script("download_baostock_v2")
    symbol = "SH000002"
    raw = pd.read_parquet(bundle / f"raw/{symbol}.parquet")
    qfq_path = bundle / f"qfq_provider/{symbol}.parquet"
    qfq = raw.copy()
    qfq[["open", "high", "low", "close", "preclose"]] = (
        qfq[["open", "high", "low", "close", "preclose"]] * (1.0 / 3.0)
    ).round(2)
    qfq["source_id"] = "eastmoney_qfq_provider"
    qfq.to_parquet(qfq_path, index=False)
    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    selected = index["symbol"].eq(symbol)
    index.loc[selected, "qfq_provider_sha256"] = source.sha256_file(qfq_path)
    index.to_parquet(index_path, index=False)
    module = load_script("incremental_update_ashare_v2")

    result = module.build_incremental_price_snapshots(
        bundle_dir=bundle,
        quant_data_root=root,
    )

    assert result.latest_trade_date == "2026-08-07"


def test_provider_native_paired_rejects_future_ohlc_ratio_disagreement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_provider_paired_bundle(root)
    source = load_script("download_baostock_v2")
    symbol = "SH000001"
    qfq_path = bundle / f"qfq_provider/{symbol}.parquet"
    qfq = pd.read_parquet(qfq_path)
    future = pd.to_datetime(qfq["date"]) > pd.Timestamp("2026-08-04")
    qfq.loc[future, "high"] = qfq.loc[future, "high"] * 1.01
    qfq.to_parquet(qfq_path, index=False)
    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    selected = index["symbol"].eq(symbol)
    index.loc[selected, "qfq_provider_sha256"] = source.sha256_file(qfq_path)
    index.to_parquet(index_path, index=False)
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="OHLC reconstruction"):
        module.build_incremental_price_snapshots(
            bundle_dir=bundle,
            quant_data_root=root,
        )


def test_incremental_price_import_reuses_history_and_recalculates_only_changed_factor_symbol(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_bundle(root)
    module = load_script("incremental_update_ashare_v2")

    result = module.build_incremental_price_snapshots(
        bundle_dir=bundle,
        quant_data_root=root,
    )

    assert result.base_latest_trade_date == "2026-08-04"
    assert result.latest_trade_date == "2026-08-07"
    assert result.factor_changed_symbols == ("SH000002",)
    assert result.reused_symbols == ("SH000001",)
    changed = pd.read_parquet(result.qfq_stage / "bars/SH000002.parquet")
    unchanged = pd.read_parquet(result.qfq_stage / "bars/SH000001.parquet")
    assert changed.loc[changed["date"] == "2026-08-01", "close"].iloc[0] == pytest.approx(4.0)
    assert unchanged.loc[unchanged["date"] == "2026-08-01", "close"].iloc[0] == pytest.approx(5.0)
    assert changed["date"].max() == pd.Timestamp("2026-08-07")
    assert result.downloaded_rows == 8
    assert result.qfq_recalculated_symbols == ("SH000002",)


def test_incremental_price_import_uses_partial_qfq_audit_for_future_factor_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_bundle(root)
    source = load_script("download_baostock_v2")
    symbol = "SH000002"

    events_path = bundle / "factor_events" / f"{symbol}.parquet"
    events = pd.read_parquet(events_path)
    events.loc[events.index[-1], "dividOperateDate"] = pd.Timestamp("2026-08-12")
    events.to_parquet(events_path, index=False)

    raw_path = bundle / "raw" / f"{symbol}.parquet"
    provider_qfq_path = bundle / "qfq_provider" / f"{symbol}.parquet"
    provider_qfq_path.parent.mkdir(parents=True, exist_ok=True)
    provider_qfq = pd.read_parquet(raw_path)
    for column in source.PRICE_COLUMNS:
        provider_qfq[column] = provider_qfq[column] * 0.4
    provider_qfq.to_parquet(provider_qfq_path, index=False)

    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    selected = index["symbol"].eq(symbol)
    index.loc[selected, "factor_events_sha256"] = source.sha256_file(events_path)
    index.loc[selected, "qfq_provider_path"] = f"qfq_provider/{symbol}.parquet"
    index.loc[selected, "qfq_provider_sha256"] = source.sha256_file(
        provider_qfq_path
    )
    index.to_parquet(index_path, index=False)

    module = load_script("incremental_update_ashare_v2")
    result = module.build_incremental_price_snapshots(
        bundle_dir=bundle,
        quant_data_root=root,
    )

    factors = pd.read_parquet(result.qfq_stage / "factors" / f"{symbol}.parquet")
    assert factors["factor"].iloc[-1] == pytest.approx(1.0)
    assert factors["factor_terminal_normalized"].all()


def test_events_changed_ignores_parquet_datetime_unit_difference() -> None:
    module = load_script("incremental_update_ashare_v2")
    values = {
        "event_date": ["2026-08-04"],
        "foreAdjustFactor": [1.0],
        "backAdjustFactor": [1.0],
        "adjustFactor": [1.0],
        "symbol": ["SH000001"],
    }
    old = pd.DataFrame(values)
    new = pd.DataFrame(values)
    old["event_date"] = pd.to_datetime(old["event_date"]).astype("datetime64[us]")
    new["event_date"] = pd.to_datetime(new["event_date"]).astype("datetime64[ns]")

    assert not module._events_changed(old, new, symbol="SH000001")


def test_incremental_price_import_rejects_full_bundle_without_touching_latest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _publish_base(root)
    bundle = _write_bundle(root)
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_mode"] = "full"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="incremental"):
        module.build_incremental_price_snapshots(bundle_dir=bundle, quant_data_root=root)

    pointer = json.loads(
        (root / "parquet/ashare_1d/ashare_daily_qfq/LATEST.json").read_text()
    )
    assert pointer["snapshot_id"] == "qfq-base"


def test_incremental_bundle_rejects_index_outside_bundle(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    bundle = _write_bundle(root)
    outside_index = root / "outside-index.parquet"
    (bundle / "BUNDLE_FILE_INDEX.parquet").replace(outside_index)
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_path"] = "../outside-index.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="inside bundle"):
        module._validate_bundle(bundle)


def test_incremental_bundle_rejects_data_file_outside_bundle(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    bundle = _write_bundle(root)
    index_path = bundle / "BUNDLE_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    source_path = bundle / str(index.loc[0, "raw_path"])
    outside_path = root / "outside-raw.parquet"
    source_path.replace(outside_path)
    index.loc[0, "raw_path"] = "../outside-raw.parquet"
    index.to_parquet(index_path, index=False)
    module = load_script("incremental_update_ashare_v2")

    with pytest.raises(module.DataContractError, match="inside bundle"):
        module._validate_bundle(bundle)


def test_reused_snapshot_files_do_not_share_inode(tmp_path: Path) -> None:
    module = load_script("incremental_update_ashare_v2")
    source = tmp_path / "source.parquet"
    target = tmp_path / "snapshot" / "target.parquet"
    source.write_bytes(b"immutable snapshot content")

    module._copy_or_link(source, target)

    assert not source.samefile(target)
