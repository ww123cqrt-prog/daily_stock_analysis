from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "quant_data_contract.py"
    )
    spec = importlib.util.spec_from_file_location("quant_data_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_bars(path: Path, *, source: str = "provider.raw") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100.0, 110.0],
            "source_id": [source, source],
        }
    ).to_parquet(path, index=False)


def write_factors(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "factor": [0.98, 1.0],
        }
    ).to_parquet(path, index=False)


def finalize_qfq_snapshot(
    module,
    quant_data: Path,
    snapshot_id: str = "snap-001",
    dataset_id: str = "etf_momentum_daily_qfq",
) -> Path:
    snapshot = quant_data / "staging" / "run-001" / snapshot_id
    write_bars(snapshot / "bars" / "SZ159915.parquet", source="tencent.raw")
    write_factors(snapshot / "factors" / "SZ159915.parquet")
    pd.DataFrame(
        {
            "symbol": ["SZ159915"],
            "ex_date": pd.to_datetime(["2022-01-13"]),
            "action_type": ["split"],
            "ratio": [5.0],
            "source": ["fund_announcement"],
        }
    ).to_parquet(snapshot / "corporate_actions.parquet", index=False)
    module.finalize_price_snapshot(
        snapshot,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={"provider": "tencent", "interface": "ifzq"},
        price_basis="qfq",
        adjustment={
            "method": "provider_native_paired",
            "adjusted_at": "2026-07-25T23:00:00+08:00",
            "adjustment_as_of": "2026-07-24",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_valid_trade_date_factor_equals_1",
            "algorithm_version": "provider-native-paired-v1",
            "paired_raw_snapshot_id": "etf-momentum-tencent-raw-snap-001",
        },
        producer={"script": "download_etf_momentum_data.py", "version": "2"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
    )
    return snapshot


def finalize_raw_snapshot(module, quant_data: Path, snapshot_id: str) -> Path:
    snapshot = quant_data / "staging" / f"run-{snapshot_id}" / snapshot_id
    write_bars(snapshot / "bars" / "SH600000.parquet", source="baostock.raw")
    module.finalize_price_snapshot(
        snapshot,
        dataset_id="ashare_daily_raw",
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={"provider": "baostock", "interface": "history_raw"},
        price_basis="raw",
        adjustment={"method": "none"},
        producer={"script": "download_baostock_v2.py", "version": "2"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
    )
    return snapshot


def finalize_qlib_snapshot(module, quant_data: Path, snapshot_id: str) -> Path:
    snapshot = quant_data / "staging" / f"run-{snapshot_id}" / snapshot_id
    (snapshot / "calendars").mkdir(parents=True)
    (snapshot / "instruments").mkdir(parents=True)
    (snapshot / "features" / "sh600000").mkdir(parents=True)
    (snapshot / "calendars" / "day.txt").write_text(
        "2026-07-23\n2026-07-24\n", encoding="utf-8"
    )
    (snapshot / "instruments" / "all.txt").write_text(
        "SH600000\t2026-07-23\t2026-07-24\n", encoding="utf-8"
    )
    (snapshot / "features" / "sh600000" / "close.day.bin").write_bytes(
        b"feature-bytes"
    )
    module.atomic_write_json(
        snapshot / "PROVIDER_MANIFEST.json",
        {
            "schema_version": 2,
            "status": "ready",
            "source_snapshot": {
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "qfq-source-001",
                "content_sha256": "a" * 64,
                "price_basis": "qfq",
            },
        },
    )
    module.finalize_qlib_snapshot(
        snapshot,
        dataset_id="qlib_ashare_v2",
        snapshot_id=snapshot_id,
        source={"provider": "quant_workflow", "interface": "v2_snapshot_to_qlib"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
        producer={"script": "build_qlib_v2.py", "version": "2"},
        upstream_snapshot={
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "qfq-source-001",
            "content_sha256": "a" * 64,
            "price_basis": "qfq",
        },
    )
    return snapshot


def test_finalize_snapshot_records_price_basis_adjustment_and_file_index(
    tmp_path: Path,
) -> None:
    module = load_module()
    snapshot = finalize_qfq_snapshot(module, tmp_path / "quant_data")

    manifest = json.loads(
        (snapshot / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    index = pd.read_parquet(snapshot / "FILE_INDEX.parquet")
    factor_index = pd.read_parquet(snapshot / "FACTOR_INDEX.parquet")

    assert manifest["schema_version"] == 2
    assert manifest["dataset_id"] == "etf_momentum_daily_qfq"
    assert manifest["price_basis"] == "qfq"
    assert manifest["adjustment"]["method"] == "provider_native_paired"
    assert manifest["adjustment"]["factor_path"] == "factors"
    assert manifest["factor_index_path"] == "FACTOR_INDEX.parquet"
    assert manifest["corporate_actions_path"] == "corporate_actions.parquet"
    assert len(manifest["corporate_actions_sha256"]) == 64
    assert manifest["date_range"] == {
        "start": "2026-07-23",
        "end": "2026-07-24",
    }
    assert index.loc[0, "relative_path"] == "bars/SZ159915.parquet"
    assert len(index.loc[0, "sha256"]) == 64
    assert factor_index.loc[0, "relative_path"] == "factors/SZ159915.parquet"
    assert len(factor_index.loc[0, "sha256"]) == 64


def test_canonical_qfq_without_reproducible_adjustment_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    module = load_module()
    snapshot = tmp_path / "quant_data" / "staging" / "run" / "snapshot"
    write_bars(snapshot / "bars" / "SZ159915.parquet")

    with pytest.raises(module.DataContractError, match="adjustment evidence"):
        module.finalize_price_snapshot(
            snapshot,
            dataset_id="bad_qfq",
            snapshot_id="snapshot",
            role="canonical",
            status="ready",
            source={"provider": "unknown", "interface": "unknown"},
            price_basis="qfq",
            adjustment={"method": "provider_native_paired"},
            producer={"script": "test", "version": "1"},
            source_retrieved_at="2026-07-25T22:59:00+08:00",
        )


def test_publish_and_resolve_immutable_snapshot(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "parquet" / "etf_1d" / "etf_momentum_daily_qfq"

    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "etf_momentum_daily_qfq": module.catalog_entry_from_snapshot(
                published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    resolved = module.resolve_dataset(
        "etf_momentum_daily_qfq",
        quant_data_root=quant_data,
        require_basis="qfq",
    )

    assert published == dataset_root / "snapshots" / "snap-001"
    assert resolved.snapshot_path == published
    assert resolved.manifest["snapshot_id"] == "snap-001"
    assert not staging.exists()


def test_resolve_rejects_wrong_price_basis(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "etf_momentum_daily_qfq": module.catalog_entry_from_snapshot(
                published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )

    with pytest.raises(module.DataContractError, match="price basis"):
        module.resolve_dataset(
            "etf_momentum_daily_qfq",
            quant_data_root=quant_data,
            require_basis="raw",
        )


def test_failed_publication_keeps_previous_latest_pointer(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    dataset_root = quant_data / "datasets" / "etf_qfq"
    first = finalize_qfq_snapshot(module, quant_data, snapshot_id="snap-001")
    module.publish_snapshot(
        first,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    previous = (dataset_root / "LATEST.json").read_bytes()

    invalid = quant_data / "staging" / "run-002" / "snap-002"
    invalid.mkdir(parents=True)
    (invalid / "DATASET_MANIFEST.json").write_text("{}", encoding="utf-8")

    with pytest.raises(module.DataContractError):
        module.publish_snapshot(
            invalid,
            dataset_root=dataset_root,
            quant_data_root=quant_data,
        )

    assert (dataset_root / "LATEST.json").read_bytes() == previous
    assert not (dataset_root / "snapshots" / "snap-002").exists()


def test_resolve_rejects_bar_content_changed_after_publication(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "etf_momentum_daily_qfq": module.catalog_entry_from_snapshot(
                published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    bars_path = published / "bars" / "SZ159915.parquet"
    bars = pd.read_parquet(bars_path)
    bars.loc[0, "close"] = 99.0
    bars.to_parquet(bars_path, index=False)

    with pytest.raises(module.DataContractError, match="bar hash mismatch"):
        module.resolve_dataset(
            "etf_momentum_daily_qfq",
            quant_data_root=quant_data,
            require_basis="qfq",
        )


def test_resolve_rejects_catalog_and_latest_pointer_drift(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    entry = module.catalog_entry_from_snapshot(
        published,
        quant_data_root=quant_data,
        dataset_root=dataset_root,
    )
    entry["snapshot_id"] = "stale-snapshot"
    module.write_root_manifest(
        quant_data,
        {"etf_momentum_daily_qfq": entry},
    )

    with pytest.raises(module.DataContractError, match="out of sync"):
        module.resolve_dataset(
            "etf_momentum_daily_qfq",
            quant_data_root=quant_data,
            require_basis="qfq",
        )


def test_catalog_entry_contains_lineage_and_content_metadata(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )

    entry = module.catalog_entry_from_snapshot(
        published,
        quant_data_root=quant_data,
        dataset_root=dataset_root,
    )

    assert entry["date_range"] == {"start": "2026-07-23", "end": "2026-07-24"}
    assert entry["row_count"] == 2
    assert entry["symbol_count"] == 1
    assert entry["file_index_path"] == "FILE_INDEX.parquet"
    assert entry["factor_index_path"] == "FACTOR_INDEX.parquet"
    assert len(entry["content_sha256"]) == 64
    assert entry["producer"]["script"] == "download_etf_momentum_data.py"


def test_resolved_run_keeps_its_snapshot_after_new_latest_is_published(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    dataset_root = quant_data / "datasets" / "etf_qfq"

    first_staging = finalize_qfq_snapshot(module, quant_data, snapshot_id="snap-001")
    first_published = module.publish_snapshot(
        first_staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "etf_momentum_daily_qfq": module.catalog_entry_from_snapshot(
                first_published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    locked_for_run = module.resolve_dataset(
        "etf_momentum_daily_qfq",
        quant_data_root=quant_data,
        require_basis="qfq",
    )

    second_staging = finalize_qfq_snapshot(
        module, quant_data, snapshot_id="snap-002"
    )
    second_published = module.publish_snapshot(
        second_staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "etf_momentum_daily_qfq": module.catalog_entry_from_snapshot(
                second_published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    current = module.resolve_dataset(
        "etf_momentum_daily_qfq",
        quant_data_root=quant_data,
        require_basis="qfq",
    )

    assert locked_for_run.snapshot_path == first_published
    assert locked_for_run.manifest["snapshot_id"] == "snap-001"
    assert current.snapshot_path == second_published
    assert current.manifest["snapshot_id"] == "snap-002"


def test_validation_rejects_factor_dates_that_do_not_cover_bars(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    snapshot = finalize_qfq_snapshot(module, quant_data)
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-07-24"]), "factor": [1.0]}
    ).to_parquet(snapshot / "factors" / "SZ159915.parquet", index=False)

    with pytest.raises(module.DataContractError, match="factor dates"):
        module.validate_snapshot(snapshot, verify_hashes=False)


def test_describe_returns_registered_legacy_metadata_without_resolving_snapshot(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    module.write_root_manifest(
        quant_data,
        {
            "legacy_raw": {
                "dataset_id": "legacy_raw",
                "data_kind": "daily_bars",
                "role": "legacy",
                "status": "legacy_unknown",
                "source": {"provider": "unknown", "interface": "unknown"},
                "price_basis": "raw",
                "path": "parquet/legacy_raw",
                "adjustment": {"method": "none"},
            }
        },
    )

    description = module.describe_dataset(
        "legacy_raw", quant_data_root=quant_data
    )

    assert description["status"] == "legacy_unknown"
    assert description["path"] == "parquet/legacy_raw"
    assert description["snapshot_available"] is False


def test_publish_snapshot_group_switches_both_latest_pointers(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_stage = finalize_raw_snapshot(module, quant_data, "raw-001")
    qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-001",
        dataset_id="ashare_daily_qfq",
    )
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"

    published = module.publish_snapshot_group(
        [(raw_stage, raw_root), (qfq_stage, qfq_root)],
        quant_data_root=quant_data,
    )

    assert {item.name for item in published} == {"raw-001", "qfq-001"}
    assert json.loads((raw_root / "LATEST.json").read_text())["snapshot_id"] == "raw-001"
    assert json.loads((qfq_root / "LATEST.json").read_text())["snapshot_id"] == "qfq-001"


def test_publish_snapshot_group_rolls_back_all_pointers_on_write_failure(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_stage = finalize_raw_snapshot(module, quant_data, "raw-002")
    qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-002",
        dataset_id="ashare_daily_qfq",
    )
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    raw_root.mkdir(parents=True)
    qfq_root.mkdir(parents=True)
    module.atomic_write_json(raw_root / "LATEST.json", {"snapshot_id": "old-raw"})
    module.atomic_write_json(qfq_root / "LATEST.json", {"snapshot_id": "old-qfq"})
    original_write = module.atomic_write_json
    calls = 0

    def fail_second_pointer(path, payload):
        nonlocal calls
        if Path(path).name == "LATEST.json":
            calls += 1
            if calls == 2:
                raise OSError("simulated pointer failure")
        return original_write(path, payload)

    with patch.object(module, "atomic_write_json", side_effect=fail_second_pointer):
        with pytest.raises(OSError, match="simulated pointer failure"):
            module.publish_snapshot_group(
                [(raw_stage, raw_root), (qfq_stage, qfq_root)],
                quant_data_root=quant_data,
            )

    assert json.loads((raw_root / "LATEST.json").read_text())["snapshot_id"] == "old-raw"
    assert json.loads((qfq_root / "LATEST.json").read_text())["snapshot_id"] == "old-qfq"


def test_publish_snapshot_group_rolls_back_pointers_when_catalog_write_fails(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    old_raw = finalize_raw_snapshot(module, quant_data, "raw-old")
    old_qfq = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-old",
        dataset_id="ashare_daily_qfq",
    )
    published_old_raw, published_old_qfq = module.publish_snapshot_group(
        [(old_raw, raw_root), (old_qfq, qfq_root)],
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "ashare_daily_raw": module.catalog_entry_from_snapshot(
                published_old_raw,
                quant_data_root=quant_data,
                dataset_root=raw_root,
            ),
            "ashare_daily_qfq": module.catalog_entry_from_snapshot(
                published_old_qfq,
                quant_data_root=quant_data,
                dataset_root=qfq_root,
            ),
        },
    )

    new_raw = finalize_raw_snapshot(module, quant_data, "raw-new")
    new_qfq = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-new",
        dataset_id="ashare_daily_qfq",
    )
    original_write = module.atomic_write_json
    failed = False

    def fail_catalog_once(path, payload):
        nonlocal failed
        if Path(path).name == "DATA_MANIFEST.json" and not failed:
            failed = True
            raise OSError("simulated catalog failure")
        return original_write(path, payload)

    with patch.object(module, "atomic_write_json", side_effect=fail_catalog_once):
        with pytest.raises(OSError, match="simulated catalog failure"):
            module.publish_snapshot_group(
                [(new_raw, raw_root), (new_qfq, qfq_root)],
                quant_data_root=quant_data,
                update_root_catalog=True,
            )

    raw = module.resolve_dataset(
        "ashare_daily_raw", quant_data_root=quant_data, require_basis="raw"
    )
    qfq = module.resolve_dataset(
        "ashare_daily_qfq", quant_data_root=quant_data, require_basis="qfq"
    )
    assert raw.manifest["snapshot_id"] == "raw-old"
    assert qfq.manifest["snapshot_id"] == "qfq-old"
    assert new_raw.is_dir()
    assert new_qfq.is_dir()


def test_finalize_publish_and_resolve_qlib_v2_snapshot(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qlib_snapshot(module, quant_data, "qlib-001")
    dataset_root = quant_data / "qlib" / "cn_data_ashare_v2"

    published = module.publish_qlib_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "qlib_ashare_v2": module.catalog_entry_from_snapshot(
                published,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    resolved = module.resolve_dataset(
        "qlib_ashare_v2",
        quant_data_root=quant_data,
        require_basis="qfq",
    )

    assert resolved.snapshot_path == published
    assert resolved.manifest["data_kind"] == "qlib_provider"
    assert resolved.manifest["instrument_count"] == 1
    assert resolved.catalog_entry["upstream_snapshots"][0]["snapshot_id"] == "qfq-source-001"


def test_qlib_snapshot_validation_rejects_unindexed_feature_tampering(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    snapshot = finalize_qlib_snapshot(module, quant_data, "qlib-002")
    feature = snapshot / "features" / "sh600000" / "close.day.bin"
    feature.write_bytes(b"tamper-bytes!")

    with pytest.raises(module.DataContractError, match="Qlib file hash mismatch"):
        module.validate_dataset_snapshot(snapshot, verify_hashes=True)
