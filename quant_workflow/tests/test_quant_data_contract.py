from __future__ import annotations

import importlib.util
import json
import os
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


def finalize_qlib_snapshot(
    module,
    quant_data: Path,
    snapshot_id: str,
    *,
    upstream_snapshot: dict[str, str] | None = None,
    dataset_id: str = "qlib_ashare_v2",
    dependency_snapshots: list[dict[str, str]] | None = None,
    producer: dict[str, str] | None = None,
) -> Path:
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
    upstream_snapshot = upstream_snapshot or {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-source-001",
        "content_sha256": "a" * 64,
        "price_basis": "qfq",
    }
    module.atomic_write_json(
        snapshot / "PROVIDER_MANIFEST.json",
        {
            "schema_version": 2,
            "status": "ready",
            "source_snapshot": upstream_snapshot,
        },
    )
    module.finalize_qlib_snapshot(
        snapshot,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        source={"provider": "quant_workflow", "interface": "v2_snapshot_to_qlib"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
        producer=producer or {"script": "build_qlib_v2.py", "version": "2"},
        upstream_snapshot=upstream_snapshot,
        dependency_snapshots=dependency_snapshots,
    )
    return snapshot


def rewrite_as_legacy_qlib_snapshot(
    module,
    snapshot: Path,
    *,
    omit_dependencies: bool = False,
) -> None:
    provider_manifest_path = snapshot / "PROVIDER_MANIFEST.json"
    provider_manifest = json.loads(provider_manifest_path.read_text(encoding="utf-8"))
    provider_manifest.pop("producer", None)
    if omit_dependencies:
        provider_manifest.pop("dependency_snapshots", None)
    module.atomic_write_json(provider_manifest_path, provider_manifest)
    index = module._build_qlib_file_index(snapshot)
    index.to_parquet(snapshot / "FILE_INDEX.parquet", index=False)
    dataset_manifest_path = snapshot / "DATASET_MANIFEST.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if omit_dependencies:
        dataset_manifest.pop("dependency_snapshots", None)
    dataset_manifest["content_sha256"] = module._content_hash(
        snapshot,
        dataset_manifest,
    )
    module.atomic_write_json(dataset_manifest_path, dataset_manifest)


def test_qlib_structural_validation_skips_file_hashing(tmp_path: Path) -> None:
    module = load_module()
    snapshot = finalize_qlib_snapshot(module, tmp_path / "quant_data", "qlib-001")

    with patch.object(
        module,
        "sha256_file",
        side_effect=AssertionError("structural validation must not hash files"),
    ):
        manifest = module.validate_qlib_snapshot(snapshot, verify_hashes=False)

    assert manifest["snapshot_id"] == "qlib-001"


def test_qlib_provider_manifest_records_and_validates_producer(
    tmp_path: Path,
) -> None:
    module = load_module()
    snapshot = finalize_qlib_snapshot(module, tmp_path / "quant_data", "qlib-001")
    provider_manifest_path = snapshot / "PROVIDER_MANIFEST.json"
    provider_manifest = json.loads(provider_manifest_path.read_text(encoding="utf-8"))

    assert provider_manifest["producer"] == {
        "script": "build_qlib_v2.py",
        "version": "2",
    }

    provider_manifest["producer"] = {"script": "tampered.py", "version": "1"}
    module.atomic_write_json(provider_manifest_path, provider_manifest)
    with pytest.raises(module.DataContractError, match="producer lineage disagree"):
        module.validate_qlib_snapshot(snapshot, verify_hashes=False)


@pytest.mark.parametrize("version", ["2", "3", "4", "5"])
def test_known_legacy_canonical_without_duplicate_producer_remains_valid(
    tmp_path: Path,
    version: str,
) -> None:
    module = load_module()
    snapshot = finalize_qlib_snapshot(
        module,
        tmp_path / "quant_data",
        f"qlib-{version}",
        producer={"script": "build_qlib_v2.py", "version": version},
    )
    rewrite_as_legacy_qlib_snapshot(module, snapshot)

    validated = module.validate_qlib_snapshot(snapshot, verify_hashes=True)

    assert validated["producer"] == {
        "script": "build_qlib_v2.py",
        "version": version,
    }


@pytest.mark.parametrize(
    ("version", "omit_dependencies"),
    [("1", True), ("2", False)],
)
def test_known_legacy_cleaned_snapshots_remain_valid(
    tmp_path: Path,
    version: str,
    omit_dependencies: bool,
) -> None:
    module = load_module()
    dependency = {
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": "canonical-legacy",
        "content_sha256": "b" * 64,
        "price_basis": "qfq",
    }
    snapshot = finalize_qlib_snapshot(
        module,
        tmp_path / "quant_data",
        f"cleaned-{version}",
        dataset_id="qlib_ashare_cleaned_v2",
        dependency_snapshots=[dependency],
        producer={"script": "build_cleaned_qlib_v2.py", "version": version},
    )
    rewrite_as_legacy_qlib_snapshot(
        module,
        snapshot,
        omit_dependencies=omit_dependencies,
    )

    validated = module.validate_qlib_snapshot(snapshot, verify_hashes=True)

    assert validated["producer"]["version"] == version


@pytest.mark.parametrize(
    "producer",
    [
        {"script": "build_qlib_v2.py", "version": "1"},
        {"script": "build_qlib_v2.py", "version": "6"},
        {"script": "build_cleaned_qlib_v2.py", "version": "3"},
    ],
)
def test_unknown_or_new_provider_versions_cannot_omit_producer(
    tmp_path: Path,
    producer: dict[str, str],
) -> None:
    module = load_module()
    snapshot = finalize_qlib_snapshot(
        module,
        tmp_path / "quant_data",
        "qlib-rejected",
        producer=producer,
    )
    rewrite_as_legacy_qlib_snapshot(module, snapshot)

    with pytest.raises(module.DataContractError, match="producer lineage disagree"):
        module.validate_qlib_snapshot(snapshot, verify_hashes=True)


@pytest.mark.parametrize(
    ("dataset_id", "producer"),
    [
        (
            "qlib_ashare_v2",
            {"script": "build_cleaned_qlib_v2.py", "version": "1"},
        ),
        (
            "qlib_ashare_cleaned_v2",
            {"script": "build_qlib_v2.py", "version": "2"},
        ),
    ],
)
def test_legacy_producer_omission_cannot_cross_dataset_identity(
    tmp_path: Path,
    dataset_id: str,
    producer: dict[str, str],
) -> None:
    module = load_module()
    dependencies = None
    if dataset_id == "qlib_ashare_cleaned_v2":
        dependencies = [
            {
                "dataset_id": "qlib_ashare_v2",
                "snapshot_id": "canonical-legacy",
                "content_sha256": "b" * 64,
                "price_basis": "qfq",
            }
        ]
    snapshot = finalize_qlib_snapshot(
        module,
        tmp_path / "quant_data",
        "cross-dataset-producer",
        dataset_id=dataset_id,
        dependency_snapshots=dependencies,
        producer=producer,
    )
    rewrite_as_legacy_qlib_snapshot(module, snapshot)

    with pytest.raises(module.DataContractError, match="producer lineage disagree"):
        module.validate_qlib_snapshot(snapshot, verify_hashes=True)


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


def test_publication_reuses_its_validated_manifest_for_catalog_entry(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "parquet" / "etf_1d" / "etf_momentum_daily_qfq"
    module.write_root_manifest(quant_data, {})
    original_validate = module.validate_dataset_snapshot

    with patch.object(
        module,
        "validate_dataset_snapshot",
        wraps=original_validate,
    ) as validate:
        published = module.publish_snapshot_group(
            [(staging, dataset_root)],
            quant_data_root=quant_data,
            update_root_catalog=True,
        )[0]

    assert published.is_dir()
    assert validate.call_count == 1


def test_transaction_validation_cache_skips_only_unchanged_snapshot_hashes(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "parquet" / "etf_1d" / "etf_momentum_daily_qfq"
    module.write_root_manifest(quant_data, {})
    validated_cache = {}
    published = module.publish_snapshot_group(
        [(staging, dataset_root)],
        quant_data_root=quant_data,
        update_root_catalog=True,
        validated_cache=validated_cache,
    )[0]

    with patch.object(
        module,
        "validate_dataset_snapshot",
        side_effect=AssertionError("unchanged transaction cache must be reused"),
    ):
        resolved = module.resolve_dataset(
            "etf_momentum_daily_qfq",
            quant_data_root=quant_data,
            require_basis="qfq",
            validated_cache=validated_cache,
        )

    assert resolved.snapshot_path == published

    manifest_path = published / "DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["producer"] = {"script": "tampered", "version": "1"}
    module.atomic_write_json(manifest_path, manifest)

    with pytest.raises(module.DataContractError, match="manifest hash mismatch"):
        module.resolve_dataset(
            "etf_momentum_daily_qfq",
            quant_data_root=quant_data,
            require_basis="qfq",
            validated_cache=validated_cache,
        )


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


def test_validate_latest_pointer_rejects_snapshot_outside_its_dataset_root(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    published = module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    pointer_path = dataset_root / "LATEST.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    escaped = quant_data / "datasets" / "other" / "snapshots" / published.name
    escaped.parent.mkdir(parents=True)
    os.replace(published, escaped)
    pointer["snapshot_path"] = escaped.relative_to(quant_data).as_posix()
    module.atomic_write_json(pointer_path, pointer)

    with pytest.raises(module.DataContractError, match="dataset root"):
        module.validate_latest_pointer(
            pointer_path,
            quant_data_root=quant_data,
            expected_dataset_id="etf_momentum_daily_qfq",
        )


def test_validate_latest_pointer_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    staging = finalize_qfq_snapshot(module, quant_data)
    dataset_root = quant_data / "datasets" / "etf_qfq"
    module.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    pointer_path = dataset_root / "LATEST.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = "0" * 64
    module.atomic_write_json(pointer_path, pointer)

    with pytest.raises(module.DataContractError, match="manifest hash mismatch"):
        module.validate_latest_pointer(
            pointer_path,
            quant_data_root=quant_data,
            expected_dataset_id="etf_momentum_daily_qfq",
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


def test_publish_fsyncs_snapshot_tree_and_destination_before_latest_pointer(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_stage = finalize_raw_snapshot(module, quant_data, "raw-durable")
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"
    events: list[tuple[str, str]] = []
    original_write = module.atomic_write_json

    def record_tree(path: Path) -> None:
        events.append(("tree", str(Path(path))))

    def record_directory(path: Path) -> None:
        events.append(("directory", str(Path(path))))

    def record_write(path: Path, payload: dict) -> None:
        if Path(path).name == "LATEST.json":
            events.append(("pointer", str(Path(path))))
        original_write(path, payload)

    with (
        patch.object(module, "_fsync_snapshot_tree", side_effect=record_tree),
        patch.object(module, "_fsync_directory", side_effect=record_directory),
        patch.object(module, "atomic_write_json", side_effect=record_write),
    ):
        module.publish_snapshot(
            raw_stage,
            dataset_root=raw_root,
            quant_data_root=quant_data,
        )

    tree_index = events.index(("tree", str(raw_stage)))
    destination_sync_index = events.index(
        ("directory", str(raw_root / "snapshots"))
    )
    pointer_index = next(
        index for index, event in enumerate(events) if event[0] == "pointer"
    )
    assert tree_index < destination_sync_index < pointer_index


def test_root_manifest_requires_relative_path_contract(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text(
        json.dumps({"schema_version": 2, "datasets": {}}),
        encoding="utf-8",
    )

    with pytest.raises(module.DataContractError, match="path format"):
        module._load_root_manifest(quant_data)


def test_publish_snapshot_group_rejects_a_concurrent_publisher(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_stage = finalize_raw_snapshot(module, quant_data, "raw-locked")
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"

    with module.dataset_publication_lock(quant_data):
        with pytest.raises(module.DataContractError, match="already in progress"):
            module.publish_snapshot_group(
                [(raw_stage, raw_root)],
                quant_data_root=quant_data,
            )

    assert raw_stage.is_dir()
    assert not (raw_root / "LATEST.json").exists()


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


def test_publish_snapshot_group_reports_rollback_failures(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    raw_stage = finalize_raw_snapshot(module, quant_data, "raw-rollback-error")
    qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-rollback-error",
        dataset_id="ashare_daily_qfq",
    )
    raw_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    raw_root.mkdir(parents=True)
    qfq_root.mkdir(parents=True)
    module.atomic_write_json(raw_root / "LATEST.json", {"snapshot_id": "old-raw"})
    module.atomic_write_json(qfq_root / "LATEST.json", {"snapshot_id": "old-qfq"})
    original_write = module.atomic_write_json
    publishing = True

    def fail_publish_and_raw_restore(path, payload):
        nonlocal publishing
        path = Path(path)
        if path == qfq_root / "LATEST.json" and publishing:
            publishing = False
            raise OSError("simulated publication failure")
        if (
            path == raw_root / "LATEST.json"
            and payload.get("snapshot_id") == "old-raw"
            and not publishing
        ):
            raise OSError("simulated rollback failure")
        return original_write(path, payload)

    with patch.object(
        module, "atomic_write_json", side_effect=fail_publish_and_raw_restore
    ):
        with pytest.raises(OSError) as raised:
            module.publish_snapshot_group(
                [(raw_stage, raw_root), (qfq_stage, qfq_root)],
                quant_data_root=quant_data,
            )

    message = str(raised.value)
    assert "simulated publication failure" in message
    assert "simulated rollback failure" in message
    assert isinstance(raised.value.__cause__, OSError)
    assert "simulated publication failure" in str(raised.value.__cause__)
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
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-source-001",
        dataset_id="ashare_daily_qfq",
    )
    qfq = module.publish_snapshot(
        qfq_stage,
        dataset_root=qfq_root,
        quant_data_root=quant_data,
    )
    qfq_manifest = module.validate_dataset_snapshot(qfq)
    upstream = {
        field: str(qfq_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    staging = finalize_qlib_snapshot(
        module,
        quant_data,
        "qlib-001",
        upstream_snapshot=upstream,
    )
    dataset_root = quant_data / "qlib" / "cn_data_ashare_v2"

    published = module.publish_qlib_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "ashare_daily_qfq": module.catalog_entry_from_snapshot(
                qfq,
                quant_data_root=quant_data,
                dataset_root=qfq_root,
            ),
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


def test_resolve_rejects_qlib_when_current_qfq_snapshot_changes(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    qlib_root = quant_data / "qlib" / "cn_data_ashare_v2"
    first_qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-source-001",
        dataset_id="ashare_daily_qfq",
    )
    first_qfq = module.publish_snapshot(
        first_qfq_stage,
        dataset_root=qfq_root,
        quant_data_root=quant_data,
    )
    first_manifest = module.validate_dataset_snapshot(first_qfq)
    upstream = {
        field: str(first_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    qlib_stage = finalize_qlib_snapshot(
        module,
        quant_data,
        "qlib-001",
        upstream_snapshot=upstream,
    )
    qlib = module.publish_qlib_snapshot(
        qlib_stage,
        dataset_root=qlib_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "ashare_daily_qfq": module.catalog_entry_from_snapshot(
                first_qfq,
                quant_data_root=quant_data,
                dataset_root=qfq_root,
            ),
            "qlib_ashare_v2": module.catalog_entry_from_snapshot(
                qlib,
                quant_data_root=quant_data,
                dataset_root=qlib_root,
            ),
        },
    )
    second_qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-source-002",
        dataset_id="ashare_daily_qfq",
    )
    module.publish_snapshot_group(
        [(second_qfq_stage, qfq_root)],
        quant_data_root=quant_data,
        update_root_catalog=True,
    )

    with pytest.raises(module.DataContractError, match="upstream snapshot is stale"):
        module.resolve_dataset(
            "qlib_ashare_v2",
            quant_data_root=quant_data,
            require_basis="qfq",
        )


def test_resolve_rejects_cleaned_qlib_when_canonical_dependency_changes(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    qfq_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    canonical_root = quant_data / "qlib" / "cn_data_ashare_v2"
    cleaned_root = quant_data / "qlib" / "cn_data_ashare_cleaned_v2"

    qfq_stage = finalize_qfq_snapshot(
        module,
        quant_data,
        snapshot_id="qfq-source-001",
        dataset_id="ashare_daily_qfq",
    )
    qfq = module.publish_snapshot(
        qfq_stage,
        dataset_root=qfq_root,
        quant_data_root=quant_data,
    )
    qfq_manifest = module.validate_dataset_snapshot(qfq)
    qfq_identity = {
        field: str(qfq_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    canonical_stage = finalize_qlib_snapshot(
        module,
        quant_data,
        "qlib-canonical-001",
        upstream_snapshot=qfq_identity,
    )
    canonical = module.publish_qlib_snapshot(
        canonical_stage,
        dataset_root=canonical_root,
        quant_data_root=quant_data,
    )
    canonical_manifest = module.validate_dataset_snapshot(canonical)
    canonical_identity = {
        field: str(canonical_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    cleaned_stage = finalize_qlib_snapshot(
        module,
        quant_data,
        "qlib-cleaned-001",
        upstream_snapshot=qfq_identity,
        dataset_id="qlib_ashare_cleaned_v2",
        dependency_snapshots=[canonical_identity],
    )
    cleaned = module.publish_qlib_snapshot(
        cleaned_stage,
        dataset_root=cleaned_root,
        quant_data_root=quant_data,
    )
    module.write_root_manifest(
        quant_data,
        {
            "ashare_daily_qfq": module.catalog_entry_from_snapshot(
                qfq,
                quant_data_root=quant_data,
                dataset_root=qfq_root,
            ),
            "qlib_ashare_v2": module.catalog_entry_from_snapshot(
                canonical,
                quant_data_root=quant_data,
                dataset_root=canonical_root,
            ),
            "qlib_ashare_cleaned_v2": module.catalog_entry_from_snapshot(
                cleaned,
                quant_data_root=quant_data,
                dataset_root=cleaned_root,
            ),
        },
    )

    assert module.resolve_dataset(
        "qlib_ashare_cleaned_v2",
        quant_data_root=quant_data,
        require_basis="qfq",
    ).snapshot_path == cleaned

    replacement_stage = finalize_qlib_snapshot(
        module,
        quant_data,
        "qlib-canonical-002",
        upstream_snapshot=qfq_identity,
    )
    module.publish_snapshot_group(
        [(replacement_stage, canonical_root)],
        quant_data_root=quant_data,
        update_root_catalog=True,
    )

    with pytest.raises(module.DataContractError, match="upstream snapshot is stale"):
        module.resolve_dataset(
            "qlib_ashare_cleaned_v2",
            quant_data_root=quant_data,
            require_basis="qfq",
        )


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


def test_qlib_snapshot_validation_rejects_symlinked_files(tmp_path: Path) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    snapshot = finalize_qlib_snapshot(module, quant_data, "qlib-symlink")
    feature = snapshot / "features" / "sh600000" / "close.day.bin"
    external = tmp_path / "external-close.day.bin"
    external.write_bytes(feature.read_bytes())
    feature.unlink()
    feature.symlink_to(external)

    with pytest.raises(module.DataContractError, match="symlink"):
        module.validate_dataset_snapshot(snapshot, verify_hashes=True)


def test_qlib_snapshot_validation_rejects_manifest_path_escape(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    snapshot = finalize_qlib_snapshot(module, quant_data, "qlib-path-escape")
    external = tmp_path / "outside-provider-manifest.json"
    external.write_bytes((snapshot / "PROVIDER_MANIFEST.json").read_bytes())
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provider_manifest_path"] = os.path.relpath(external, snapshot)
    manifest["content_sha256"] = module._content_hash(snapshot, manifest)
    module.atomic_write_json(manifest_path, manifest)

    with pytest.raises(module.DataContractError, match="must stay inside snapshot"):
        module.validate_dataset_snapshot(snapshot, verify_hashes=True)
