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
