import fcntl
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_data_manifest.py"
    spec = importlib.util.spec_from_file_location("sync_data_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_pit_membership(quant_data: Path) -> Path:
    pit_dir = quant_data / "reference" / "pit_csi800"
    (pit_dir / "daily").mkdir(parents=True)
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
            "index_code": ["000300.XSHG", "000300.XSHG"],
        }
    )
    membership.to_parquet(pit_dir / "daily" / "2025.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": pd.to_datetime(["2025-01-02"]),
            "end_date": pd.to_datetime(["2025-01-03"]),
            "index_code": ["000300.XSHG"],
        }
    ).to_parquet(pit_dir / "membership_intervals.parquet", index=False)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-03",
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )
    return pit_dir


def seed_reference_snapshot(quant_data: Path, membership_hash: str) -> Path:
    reference = quant_data / "reference"
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "list_date": pd.to_datetime(["2000-01-01"]),
            "delist_date": [pd.NaT],
            "board": ["main"],
        }
    ).to_parquet(reference / "security_master.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "industry_code": ["csrc:bank"],
            "start_date": pd.to_datetime(["2025-01-02"]),
            "end_date": pd.to_datetime(["2025-01-03"]),
        }
    ).to_parquet(reference / "industry_intervals.parquet", index=False)
    membership.assign(is_st=False).to_parquet(
        reference / "st_status.parquet", index=False
    )
    membership.assign(is_suspended=False).to_parquet(
        reference / "suspension_status.parquet", index=False
    )
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "baostock",
                "selected": "baostock",
                "date_range": {
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-03",
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_hash,
                "build_run_id": "reference_20260718_cache_v2",
            }
        ),
        encoding="utf-8",
    )
    return reference


def seed_robust_price_product(quant_data: Path) -> Path:
    products = quant_data / "runtime" / "robust_strategy" / "price_products"
    product = products / "0123456789abcdef"
    (product / "cleaned_parquet").mkdir(parents=True)
    (product / "qlib_provider" / "calendars").mkdir(parents=True)
    (product / "qlib_provider" / "instruments").mkdir(parents=True)
    (product / "cleaned_parquet" / "000001.parquet").write_bytes(b"parquet")
    (product / "qlib_provider" / "calendars" / "day.txt").write_text(
        "2025-01-02\n", encoding="utf-8"
    )
    (product / "qlib_provider" / "instruments" / "all.txt").write_text(
        "SH000001\t2025-01-02\t2025-01-02\n", encoding="utf-8"
    )
    files = {
        "cleaned_parquet/000001.parquet": {},
        "qlib_provider/calendars/day.txt": {},
        "qlib_provider/instruments/all.txt": {},
    }
    (product / "CONTENT.json").write_text(
        json.dumps(
            {
                "build_version": 1,
                "product_key": product.name,
                "tree_sha256": "tree-hash",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    (product / "BUILD.json").write_text(
        json.dumps(
            {
                "build_version": 1,
                "product_key": product.name,
                "price_manifest_sha256": "p" * 64,
                "membership_snapshot_hash": "m" * 64,
                "benchmark_fingerprint": "b" * 64,
                "content_manifest_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    (products / "latest_product.json").parent.mkdir(parents=True, exist_ok=True)
    (products / "latest_product.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "product_key": product.name,
                "product_dir": str(product),
                "cleaned_symbols": 1,
                "output_rows": 1,
            }
        ),
        encoding="utf-8",
    )
    return product


def seed_qlib_v2_snapshot(module, quant_data: Path) -> Path:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "quant_data_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "quant_data_contract_sync_fixture", contract_path
    )
    contract = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(contract)
    qfq_snapshot = (
        quant_data
        / "staging"
        / "qfq-sync"
        / "qfq-source-001"
    )
    (qfq_snapshot / "bars").mkdir(parents=True)
    (qfq_snapshot / "factors").mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000.0, 1100.0],
            "source_id": ["baostock", "baostock"],
        }
    ).to_parquet(qfq_snapshot / "bars" / "SH600000.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "factor": [0.5, 1.0],
        }
    ).to_parquet(qfq_snapshot / "factors" / "SH600000.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "event_date": pd.to_datetime(["2026-07-23"]),
            "action_type": ["adjustment_factor_event"],
        }
    ).to_parquet(qfq_snapshot / "corporate_actions.parquet", index=False)
    contract.finalize_price_snapshot(
        qfq_snapshot,
        dataset_id="ashare_daily_qfq",
        snapshot_id="qfq-source-001",
        role="canonical",
        status="ready",
        source={"provider": "baostock", "interface": "raw_factor"},
        price_basis="qfq",
        adjustment={
            "method": "local_factor",
            "adjusted_at": "2026-07-25T23:00:00+08:00",
            "adjustment_as_of": "2026-07-24",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_factor_equals_1",
            "algorithm_version": "test-v1",
        },
        producer={"script": "test", "version": "1"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
    )
    published_qfq = contract.publish_snapshot(
        qfq_snapshot,
        dataset_root=(
            quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
        ),
        quant_data_root=quant_data,
    )
    qfq_manifest = contract.validate_dataset_snapshot(published_qfq)
    snapshot = quant_data / "staging" / "qlib-sync" / "qlib-sync-001"
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
    upstream = {
        field: str(qfq_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    contract.atomic_write_json(
        snapshot / "PROVIDER_MANIFEST.json",
        {
            "schema_version": 2,
            "status": "ready",
            "source_snapshot": upstream,
        },
    )
    contract.finalize_qlib_snapshot(
        snapshot,
        dataset_id="qlib_ashare_v2",
        snapshot_id="qlib-sync-001",
        source={
            "provider": "quant_workflow",
            "interface": "v2_snapshot_to_qlib",
        },
        source_retrieved_at="2026-07-25T22:59:00+08:00",
        producer={"script": "build_qlib_v2.py", "version": "2"},
        upstream_snapshot=upstream,
    )
    return contract.publish_qlib_snapshot(
        snapshot,
        dataset_root=quant_data / "qlib" / "cn_data_ashare_v2",
        quant_data_root=quant_data,
    )


class SyncDataManifestTest(unittest.TestCase):
    def test_known_native_cleaned_pointer_resolves_stable_dataset_id(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            quant_data = Path(tmp) / "quant_data"
            pointer = (
                quant_data
                / "qlib"
                / "cn_data_ashare_cleaned_v2"
                / "LATEST.json"
            )

            dataset_id = module._dataset_id_for_pointer(
                pointer,
                quant_data=quant_data,
                existing_datasets={},
            )

        self.assertEqual(dataset_id, "qlib_ashare_cleaned_v2")

    def test_blocked_pointer_replaces_stale_blocker_codes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            quant_data = Path(tmp) / "quant_data"
            pointer = (
                quant_data
                / "parquet"
                / "etf_1d"
                / "custom_etf_qfq"
                / "LATEST.json"
            )

            entry = module._blocked_pointer_entry(
                dataset_id="custom_etf_qfq",
                pointer_path=pointer,
                quant_data=quant_data,
                existing_entry={
                    "dataset_id": "custom_etf_qfq",
                    "blocker_codes": ["latest_pointer_invalid"],
                },
                blocker_code="snapshot_manifest_invalid",
            )

        self.assertEqual(entry["blocker_codes"], ["snapshot_manifest_invalid"])

    def test_compatibility_enrichment_preserves_existing_producer_version(self):
        module = load_module()
        compatibility = {
            "dataset_id": "ashare_merged_qfq_compatibility",
            "lifecycle": "compatibility_output",
        }
        datasets = {
            "ashare_daily_qfq": {
                "dataset_id": "ashare_daily_qfq",
                "status": "ready",
                "dataset_root": "parquet/ashare_1d/ashare_daily_qfq",
                "snapshot_id": "qfq-001",
                "content_sha256": "a" * 64,
                "adjustment": {"factor_path": "factors"},
            },
            compatibility["dataset_id"]: compatibility,
        }
        existing_datasets = {
            compatibility["dataset_id"]: {
                "dataset_id": compatibility["dataset_id"],
                "producer": {"script": "old.py", "version": "7"},
            }
        }

        module._enrich_compatibility_entries(
            datasets,
            existing_datasets=existing_datasets,
        )

        self.assertEqual(compatibility["producer"]["version"], "7")
        self.assertEqual(
            compatibility["producer"]["script"],
            "publish_v2_compatibility.py",
        )

    def test_v2_catalog_passes_existing_datasets_to_compatibility_enrichment(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            quant_data = Path(tmp) / "quant_data"
            existing_datasets = {
                "ashare_merged_qfq_compatibility": {
                    "dataset_id": "ashare_merged_qfq_compatibility",
                    "producer": {"version": "7"},
                }
            }
            captured = []

            def capture(_datasets, *, existing_datasets=None):
                captured.append(existing_datasets)

            with patch.object(
                module,
                "_enrich_compatibility_entries",
                side_effect=capture,
            ):
                module._v2_dataset_catalog(
                    quant_data=quant_data,
                    pit_entry=None,
                    reference_entry=None,
                    robust_product_entry=None,
                    existing_datasets=existing_datasets,
                )

        self.assertEqual(captured, [existing_datasets])

    def test_build_manifest_resolves_quant_data_root_before_using_helpers(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            quant_data = root / "nested" / ".." / "quant_data"
            seen = []

            def capture_existing(path):
                seen.append(path)
                return {}

            with patch.object(
                module, "_existing_dataset_catalog", side_effect=capture_existing
            ):
                module.build_manifest(
                    quant_data_root=quant_data,
                    reports_root=root / "reports",
                )

        self.assertEqual(seen, [quant_data.resolve()])

    def test_manifest_marks_v1_paths_as_compatibility_and_old_paths_as_audit(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            parquet = quant_data / "parquet" / "ashare_1d"
            qlib = quant_data / "qlib"
            for name in (
                "merged_baostock_jqdata",
                "merged_baostock_jqdata_cleaned",
            ):
                current = parquet / name
                (current / ".metadata").mkdir(parents=True)
                (current / ".metadata" / "COMPATIBILITY_MANIFEST.json").write_text(
                    json.dumps(
                        {
                            "contract": "v1_ashare_qfq_compatibility",
                            "upstream": {
                                "dataset_id": "ashare_daily_qfq",
                                "snapshot_id": "qfq-001",
                            },
                            "date_range": {"start": "2020-01-02", "end": "2026-07-24"},
                            "symbol_count": 1,
                            "row_count": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                pd.DataFrame({"date": pd.to_datetime(["2026-07-24"])}).to_parquet(
                    current / "600000.parquet", index=False
                )
                old = parquet / f"{name}_old"
                old.mkdir()
                pd.DataFrame({"date": pd.to_datetime(["2026-07-17"])}).to_parquet(
                    old / "600000.parquet", index=False
                )
            for name in (
                "cn_data_baostock_jqdata",
                "cn_data_baostock_jqdata_cleaned",
            ):
                current = qlib / name
                (current / "calendars").mkdir(parents=True)
                (current / "instruments").mkdir()
                (current / "calendars" / "day.txt").write_text(
                    "2026-07-24\n", encoding="utf-8"
                )
                (current / "instruments" / "all.txt").write_text(
                    "SH600000\t2026-07-24\t2026-07-24\n", encoding="utf-8"
                )
                (current / "PROVIDER_MANIFEST.json").write_text(
                    json.dumps(
                        {
                            "source_snapshot": {
                                "dataset_id": "ashare_daily_qfq",
                                "snapshot_id": "qfq-001",
                            },
                            "conversion": {
                                "factor_semantics": "unit_factor_prices_already_qfq"
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                old = qlib / f"{name}_old"
                (old / "calendars").mkdir(parents=True)
                (old / "instruments").mkdir()
                (old / "calendars" / "day.txt").write_text(
                    "2026-07-17\n", encoding="utf-8"
                )
                (old / "instruments" / "all.txt").write_text(
                    "SH600000\t2026-07-17\t2026-07-17\n", encoding="utf-8"
                )

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        compatibility = manifest["datasets"]["ashare_merged_qfq_compatibility"]
        for dataset_id in (
            "ashare_merged_qfq_compatibility",
            "ashare_cleaned_qfq_compatibility",
            "qlib_ashare_merged_compatibility",
            "qlib_ashare_cleaned_compatibility",
        ):
            self.assertEqual(manifest["datasets"][dataset_id]["role"], "legacy")
        self.assertEqual(compatibility["status"], "ready")
        self.assertEqual(compatibility["lifecycle"], "compatibility_output")
        self.assertEqual(
            compatibility["path"],
            "parquet/ashare_1d/merged_baostock_jqdata",
        )
        old = manifest["datasets"]["ashare_merged_qfq_old"]
        self.assertEqual(old["role"], "audit")
        self.assertEqual(old["lifecycle"], "old")
        self.assertEqual(
            old["path"],
            "parquet/ashare_1d/merged_baostock_jqdata_old",
        )
        self.assertNotIn("ashare_merged_qfq_legacy", manifest["datasets"])

    def test_compatibility_factor_path_is_qualified_to_the_upstream_snapshot(self):
        module = load_module()
        qfq = {
            "dataset_id": "ashare_daily_qfq",
            "data_kind": "daily_bars",
            "role": "canonical",
            "status": "ready",
            "source": {"provider": "baostock", "interface": "raw_factor"},
            "price_basis": "qfq",
            "dataset_root": "parquet/ashare_1d/ashare_daily_qfq",
            "snapshot_id": "qfq-001",
            "source_retrieved_at": "2026-07-25T22:59:00+08:00",
            "content_sha256": "a" * 64,
            "adjustment": {
                "method": "local_factor",
                "factor_path": "factors",
                "adjusted_at": "2026-07-25T23:00:00+08:00",
                "adjustment_as_of": "2026-07-24",
            },
        }
        compatibility = {
            "dataset_id": "ashare_merged_qfq_compatibility",
            "lifecycle": "compatibility_output",
            "upstream_dataset_id": "ashare_daily_qfq",
            "upstream_snapshot_id": "qfq-001",
            "upstream_content_sha256": "a" * 64,
            "adjustment": {"method": "local_factor"},
        }
        datasets = {
            "ashare_daily_qfq": qfq,
            "ashare_merged_qfq_compatibility": compatibility,
        }

        module._enrich_compatibility_entries(datasets)

        entry = datasets["ashare_merged_qfq_compatibility"]
        self.assertEqual(
            entry["adjustment"]["factor_path"],
            "parquet/ashare_1d/ashare_daily_qfq/snapshots/qfq-001/factors",
        )
        self.assertEqual(
            entry["adjustment"]["evidence_status"],
            "in_upstream_v2_snapshot",
        )

    def test_blocked_qfq_entry_does_not_erase_compatibility_upstream_metadata(self):
        module = load_module()
        compatibility = {
            "dataset_id": "ashare_merged_qfq_compatibility",
            "lifecycle": "compatibility_output",
            "adjustment": {
                "method": "local_factor",
                "evidence_status": "historical_snapshot",
            },
            "upstream_snapshot_id": "qfq-good",
            "upstream_content_sha256": "b" * 64,
        }
        datasets = {
            "ashare_daily_qfq": {
                "dataset_id": "ashare_daily_qfq",
                "status": "blocked",
                "adjustment": {
                    "method": "local_factor",
                    "evidence_status": "raw_factor_snapshot_missing",
                },
            },
            "ashare_merged_qfq_compatibility": compatibility,
        }

        module._enrich_compatibility_entries(datasets)

        self.assertEqual(compatibility["upstream_snapshot_id"], "qfq-good")
        self.assertEqual(compatibility["upstream_content_sha256"], "b" * 64)
        self.assertEqual(
            compatibility["adjustment"]["evidence_status"],
            "historical_snapshot",
        )

    def test_stale_compatibility_keeps_actual_upstream_lineage(self):
        module = load_module()
        compatibility = {
            "dataset_id": "ashare_merged_qfq_compatibility",
            "lifecycle": "compatibility_output",
            "status": "ready",
            "upstream_dataset_id": "ashare_daily_qfq",
            "upstream_snapshot_id": "qfq-old",
            "upstream_content_sha256": "b" * 64,
            "adjustment": {
                "method": "local_factor",
                "evidence_status": "in_upstream_v2_snapshot",
            },
        }
        datasets = {
            "ashare_daily_qfq": {
                "dataset_id": "ashare_daily_qfq",
                "status": "ready",
                "dataset_root": "parquet/ashare_1d/ashare_daily_qfq",
                "snapshot_id": "qfq-current",
                "content_sha256": "a" * 64,
                "source_retrieved_at": "2026-07-28T12:00:00+08:00",
                "adjustment": {"factor_path": "factors"},
            },
            compatibility["dataset_id"]: compatibility,
        }

        module._enrich_compatibility_entries(datasets)

        self.assertEqual(compatibility["status"], "stale")
        self.assertEqual(compatibility["upstream_snapshot_id"], "qfq-old")
        self.assertEqual(compatibility["upstream_content_sha256"], "b" * 64)
        self.assertEqual(
            compatibility["blocker_codes"],
            ["upstream_snapshot_mismatch"],
        )
        self.assertEqual(
            compatibility["current_upstream_snapshot_id"],
            "qfq-current",
        )

    def test_corrupt_dynamic_latest_pointer_remains_visible_as_blocked(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            dataset_root = quant_data / "parquet" / "etf_1d" / "custom_etf_qfq"
            dataset_root.mkdir(parents=True)
            (dataset_root / "LATEST.json").write_text("{broken", encoding="utf-8")
            quant_data.joinpath("DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "datasets": {
                            "custom_etf_qfq": {
                                "dataset_id": "custom_etf_qfq",
                                "data_kind": "daily_bars",
                                "role": "canonical",
                                "status": "ready",
                                "price_basis": "qfq",
                                "dataset_root": "parquet/etf_1d/custom_etf_qfq",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["datasets"]["custom_etf_qfq"]
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("latest_pointer_invalid", entry["blocker_codes"])

    def test_corrupt_dynamic_snapshot_manifest_remains_visible_as_blocked(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            dataset_root = quant_data / "parquet" / "etf_1d" / "custom_etf_qfq"
            snapshot = dataset_root / "snapshots" / "snap-001"
            snapshot.mkdir(parents=True)
            (snapshot / "DATASET_MANIFEST.json").write_text("{broken", encoding="utf-8")
            broken_hash = hashlib.sha256(
                (snapshot / "DATASET_MANIFEST.json").read_bytes()
            ).hexdigest()
            (dataset_root / "LATEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "dataset_id": "custom_etf_qfq",
                        "snapshot_id": "snap-001",
                        "snapshot_path": "parquet/etf_1d/custom_etf_qfq/snapshots/snap-001",
                        "manifest_sha256": broken_hash,
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["datasets"]["custom_etf_qfq"]
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("snapshot_manifest_invalid", entry["blocker_codes"])

    def test_dynamic_latest_pointer_hash_mismatch_is_blocked(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            published = seed_qlib_v2_snapshot(module, quant_data)
            pointer_path = published.parents[1] / "LATEST.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["manifest_sha256"] = "0" * 64
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["datasets"]["qlib_ashare_v2"]
        self.assertEqual(entry["status"], "blocked")
        self.assertIn("latest_pointer_invalid", entry["blocker_codes"])

    def test_duplicate_dataset_id_cannot_replace_the_registered_dataset_root(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            published = seed_qlib_v2_snapshot(module, quant_data)
            official_root = published.parents[1]
            rogue_root = quant_data / "qlib" / "rogue_provider"
            shutil.copytree(official_root, rogue_root)
            rogue_pointer_path = rogue_root / "LATEST.json"
            rogue_pointer = json.loads(rogue_pointer_path.read_text(encoding="utf-8"))
            rogue_pointer["snapshot_path"] = (
                rogue_root / "snapshots" / published.name
            ).relative_to(quant_data).as_posix()
            rogue_pointer_path.write_text(
                json.dumps(rogue_pointer),
                encoding="utf-8",
            )

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["datasets"]["qlib_ashare_v2"]
        self.assertEqual(entry["status"], "ready")
        self.assertEqual(entry["dataset_root"], "qlib/cn_data_ashare_v2")
        rogue = manifest["datasets"]["rogue_provider"]
        self.assertEqual(rogue["status"], "blocked")
        self.assertIn("dataset_id_root_mismatch", rogue["blocker_codes"])

    def test_manifest_discovers_published_qlib_v2_snapshot(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            published = seed_qlib_v2_snapshot(module, quant_data)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["datasets"]["qlib_ashare_v2"]
        self.assertEqual(entry["status"], "ready")
        self.assertEqual(entry["latest_trade_date"], "2026-07-24")
        self.assertEqual(entry["snapshot_id"], published.name)

    def test_native_qlib_is_stale_when_current_qfq_snapshot_changes(self):
        module = load_module()
        qlib = {
            "dataset_id": "qlib_ashare_v2",
            "data_kind": "qlib_provider",
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "upstream_snapshots": [
                {
                    "dataset_id": "ashare_daily_qfq",
                    "snapshot_id": "qfq-old",
                    "content_sha256": "a" * 64,
                    "price_basis": "qfq",
                }
            ],
        }
        datasets = {
            "ashare_daily_qfq": {
                "dataset_id": "ashare_daily_qfq",
                "status": "ready",
                "snapshot_id": "qfq-current",
                "content_sha256": "b" * 64,
                "price_basis": "qfq",
            },
            "qlib_ashare_v2": qlib,
        }

        module._enrich_derived_entries(datasets)

        self.assertEqual(qlib["status"], "stale")
        self.assertEqual(qlib["blocker_codes"], ["upstream_snapshot_mismatch"])
        self.assertEqual(qlib["current_upstream_snapshot_id"], "qfq-current")

    def test_cleaned_qlib_is_stale_when_canonical_dependency_changes(self):
        module = load_module()
        cleaned = {
            "dataset_id": "qlib_ashare_cleaned_v2",
            "data_kind": "qlib_provider",
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "upstream_snapshots": [
                {
                    "dataset_id": "ashare_daily_qfq",
                    "snapshot_id": "qfq-current",
                    "content_sha256": "a" * 64,
                    "price_basis": "qfq",
                }
            ],
            "dependency_snapshots": [
                {
                    "dataset_id": "qlib_ashare_v2",
                    "snapshot_id": "canonical-old",
                    "content_sha256": "b" * 64,
                    "price_basis": "qfq",
                }
            ],
        }
        datasets = {
            "ashare_daily_qfq": {
                "dataset_id": "ashare_daily_qfq",
                "status": "ready",
                "snapshot_id": "qfq-current",
                "content_sha256": "a" * 64,
                "price_basis": "qfq",
            },
            "qlib_ashare_v2": {
                "dataset_id": "qlib_ashare_v2",
                "status": "ready",
                "snapshot_id": "canonical-current",
                "content_sha256": "c" * 64,
                "price_basis": "qfq",
            },
            "qlib_ashare_cleaned_v2": cleaned,
        }

        module._enrich_derived_entries(datasets)

        self.assertEqual(cleaned["status"], "stale")
        self.assertEqual(cleaned["blocker_codes"], ["dependency_snapshot_mismatch"])
        self.assertEqual(
            cleaned["current_dependency_snapshots"]["qlib_ashare_v2"]["snapshot_id"],
            "canonical-current",
        )

    def test_sync_refuses_to_write_while_dataset_publication_is_in_progress(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            lock_path = (
                quant_data / "runtime" / "locks" / "dataset_publication.lock"
            )
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    with self.assertRaisesRegex(
                        module.DataContractError,
                        "already in progress",
                    ):
                        module.sync_data_manifest(
                            quant_data_root=quant_data,
                            reports_root=root / "reports",
                        )
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def test_sync_manifest_reflects_actual_cleaned_provider_calendar(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            reports = root / "quant_workflow" / "reports"
            cleaned_parquet = quant_data / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
            cleaned_provider = quant_data / "qlib" / "cn_data_baostock_jqdata_cleaned"
            for directory in (cleaned_parquet, cleaned_provider / "calendars", cleaned_provider / "instruments"):
                directory.mkdir(parents=True, exist_ok=True)
            (cleaned_parquet / "000001.parquet").write_bytes(b"fake")
            (cleaned_provider / "calendars" / "day.txt").write_text(
                "2026-07-08\n2026-07-09\n",
                encoding="utf-8",
            )
            (cleaned_provider / "instruments" / "all.txt").write_text(
                "SZ000001\t2026-07-08\t2026-07-09\n",
                encoding="utf-8",
            )
            reports.mkdir(parents=True)
            (reports / "data_quality_report_latest.json").write_text(
                '{"publication_contract":"atomic_json_pointer_to_immutable_markdown_v1"}',
                encoding="utf-8",
            )

            manifest_path = module.sync_data_manifest(quant_data_root=quant_data, reports_root=reports)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root_readme = (quant_data / "README.md").read_text(encoding="utf-8")
            ashare_readme = (
                quant_data / "parquet" / "ashare_1d" / "README.md"
            ).read_text(encoding="utf-8")
            manifest_issues = module.check_manifest(
                quant_data_root=quant_data,
                reports_root=reports,
            )

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            manifest["related_reports"]["quality_report"],
            "data_quality_report_latest.json",
        )
        self.assertEqual(manifest["path_format"], "relative_to_quant_data_root")
        cleaned = manifest["datasets"]["ashare_cleaned_qfq_legacy"]
        self.assertEqual(
            cleaned["path"],
            "parquet/ashare_1d/merged_baostock_jqdata_cleaned",
        )
        self.assertEqual(cleaned["price_basis"], "qfq")
        self.assertEqual(cleaned["status"], "legacy_unknown")
        self.assertEqual(manifest["official_entries"]["cleaned_parquet"]["file_count"], 1)
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["calendar_last"], "2026-07-09")
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["instruments_all"], 1)
        self.assertEqual(manifest["official_entries"]["cleaned_parquet"]["role"], "legacy")
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["role"], "legacy")
        self.assertEqual(manifest["documented_inputs"]["canonical_qlib_provider"]["calendar_last"], "")
        self.assertIn("global_daily", manifest["documented_inputs"])
        self.assertIn("raw_global_daily", manifest["documented_inputs"])
        self.assertIn("raw_ashare_snapshots", manifest["documented_inputs"])
        self.assertIn("raw_ashare_news", manifest["documented_inputs"])
        self.assertEqual(manifest["datasets"]["ashare_daily_raw"]["status"], "blocked")
        self.assertEqual(manifest["datasets"]["ashare_daily_qfq"]["status"], "blocked")
        self.assertEqual(manifest["datasets"]["qlib_ashare_v2"]["status"], "blocked")
        self.assertEqual(
            manifest["datasets"]["qlib_ashare_cleaned_v2"]["status"],
            "blocked",
        )
        self.assertEqual(
            manifest["documented_inputs"]["canonical_qlib_provider"]["path"],
            "qlib/cn_data_ashare_v2",
        )
        self.assertEqual(
            manifest["documented_inputs"]["cleaned_qlib_provider"]["path"],
            "qlib/cn_data_ashare_cleaned_v2",
        )
        self.assertEqual(
            manifest["documented_inputs"]["compatibility_qlib_provider"]["path"],
            "qlib/cn_data_baostock_jqdata",
        )
        self.assertNotIn(str(root), json.dumps(manifest))
        self.assertIn(module.README_BLOCK_START, root_readme)
        self.assertIn("ashare_cleaned_qfq_legacy", ashare_readme)
        self.assertEqual(manifest_issues, [])

    def test_legacy_dataset_file_count_includes_nested_audit_files(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            quant_data = Path(tmp) / "quant_data"
            dataset = quant_data / "parquet" / "global_1d"
            (dataset / "us").mkdir(parents=True)
            (dataset / "jp").mkdir(parents=True)
            (dataset / "us" / "AAPL.parquet").write_bytes(b"us")
            (dataset / "jp" / "7203.T.parquet").write_bytes(b"jp")

            entry = module._legacy_dataset_entry(
                quant_data=quant_data,
                dataset_id="global_daily_legacy",
                path=dataset,
                data_kind="daily_bars",
                source_provider="yahoo",
                source_interface="chart_api_normalized",
                price_basis="derived",
                role="audit",
            )

        self.assertEqual(entry["file_count"], 2)

    def test_manifest_registers_complete_pit_and_marks_reference_pending(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            seed_pit_membership(quant_data)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        pit = manifest["official_entries"]["pit_csi800"]
        self.assertEqual(pit["path"], "reference/pit_csi800")
        self.assertEqual(pit["date_range"]["end_date"], "2025-01-03")
        self.assertEqual(pit["daily_file_count"], 1)
        self.assertEqual(pit["member_day_count"], 2)
        self.assertEqual(pit["symbol_count"], 1)
        self.assertEqual(len(pit["membership_snapshot_hash"]), 64)
        self.assertEqual(pit["selected_source"], "baostock")
        self.assertEqual(
            manifest["pending_entries"]["robust_reference"]["status"],
            "not_ready",
        )

    def test_manifest_registers_complete_reference_snapshot(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)
            with module.membership_read_lock(pit_dir) as snapshot:
                membership_hash = snapshot.fingerprint()
            seed_reference_snapshot(quant_data, membership_hash)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["official_entries"]["robust_reference"]
        self.assertEqual(entry["path"], "reference")
        self.assertEqual(entry["selected_source"], "baostock")
        self.assertEqual(entry["build_run_id"], "reference_20260718_cache_v2")
        self.assertEqual(entry["date_range"]["start_date"], "2025-01-02")
        self.assertEqual(entry["row_counts"]["st_status.parquet"], 2)
        self.assertEqual(len(entry["reference_snapshot_hash"]), 64)
        self.assertNotIn("robust_reference", manifest.get("pending_entries", {}))

    def test_manifest_registers_immutable_robust_price_product(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)
            with module.membership_read_lock(pit_dir) as snapshot:
                membership_hash = snapshot.fingerprint()
            seed_reference_snapshot(quant_data, membership_hash)
            product = seed_robust_price_product(quant_data)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["official_entries"]["robust_price_product"]
        self.assertEqual(
            entry["path"],
            "runtime/robust_strategy/price_products/0123456789abcdef",
        )
        self.assertEqual(
            entry["cleaned_parquet"],
            "runtime/robust_strategy/price_products/0123456789abcdef/cleaned_parquet",
        )
        self.assertEqual(entry["product_key"], product.name)
        self.assertEqual(entry["cleaned_symbols"], 1)
        self.assertNotIn("robust_price_product", manifest["pending_entries"])

    def test_pit_lock_remains_held_during_reference_validation(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)
            with module.membership_read_lock(pit_dir) as snapshot:
                membership_hash = snapshot.fingerprint()
            seed_reference_snapshot(quant_data, membership_hash)
            original_reference_lock = module.reference_read_lock
            exclusive_lock_acquired: list[bool] = []

            @contextmanager
            def probing_reference_lock(reference_dir):
                with (pit_dir / ".build_pit_csi800.lock").open("a+b") as handle:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                    except BlockingIOError:
                        exclusive_lock_acquired.append(False)
                    else:
                        exclusive_lock_acquired.append(True)
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                with original_reference_lock(reference_dir) as snapshot:
                    yield snapshot

            with patch.object(
                module, "reference_read_lock", probing_reference_lock
            ):
                manifest = module.build_manifest(
                    quant_data_root=quant_data,
                    reports_root=root / "reports",
                )

        self.assertIn("robust_reference", manifest["official_entries"])
        self.assertEqual(exclusive_lock_acquired, [False])

    def test_corrupt_membership_source_becomes_pending(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)
            source_path = pit_dir / "MEMBERSHIP_SOURCE.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source["date_range"] = None
            source_path.write_text(json.dumps(source), encoding="utf-8")

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        self.assertNotIn("pit_csi800", manifest["official_entries"])
        self.assertEqual(
            manifest["pending_entries"]["pit_csi800"]["reason_code"],
            "membership_snapshot_invalid",
        )

    def test_corrupt_membership_parquet_becomes_pending(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)
            (pit_dir / "daily" / "2025.parquet").write_bytes(b"not parquet")

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        self.assertNotIn("pit_csi800", manifest["official_entries"])
        self.assertEqual(
            manifest["pending_entries"]["pit_csi800"]["reason_code"],
            "membership_snapshot_invalid",
        )

    def test_manifest_write_fsyncs_file_before_replace_and_directory_after(self):
        module = load_module()
        events: list[str] = []
        original_fsync = module.os.fsync
        original_replace = module.os.replace

        def record_fsync(descriptor):
            events.append("fsync")
            return original_fsync(descriptor)

        def record_replace(source, destination):
            events.append("replace")
            return original_replace(source, destination)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(module.os, "fsync", record_fsync),
                patch.object(module.os, "replace", record_replace),
            ):
                module.sync_data_manifest(
                    quant_data_root=root / "quant_data",
                    reports_root=root / "reports",
                )

        replace_index = events.index("replace")
        self.assertIn("fsync", events[:replace_index])
        self.assertIn("fsync", events[replace_index + 1 :])

    def test_check_manifest_reports_unregistered_governed_dataset_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            module.sync_data_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

            self.assertEqual(
                module.check_manifest(
                    quant_data_root=quant_data,
                    reports_root=root / "reports",
                ),
                [],
            )

            unknown = quant_data / "parquet" / "ashare_1d" / "unknown_source"
            unknown.mkdir(parents=True)
            pd.DataFrame({"date": ["2026-07-24"]}).to_parquet(
                unknown / "000001.parquet", index=False
            )
            issues = module.check_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        self.assertTrue(
            any("unregistered governed data path" in issue for issue in issues),
            issues,
        )

    def test_check_manifest_reports_generated_readme_drift(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            module.sync_data_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )
            readme = quant_data / "README.md"
            readme.write_text("# stale\n", encoding="utf-8")

            issues = module.check_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        self.assertIn("dataset README catalog is stale: README.md", issues)

    def test_sync_preserves_explicit_entries_without_stale_checkout_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            reports = root / "daily_stock_analysis" / "quant_workflow" / "reports"
            quant_data.mkdir()
            reports.mkdir(parents=True)
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "generated_at": "old",
                        "source_report": "/old/daily_stock_analysis-robust/quant_workflow/reports/data_hygiene_inventory_latest.json",
                        "official_entries": {
                            "external_verified_product": {
                                "path": "/shared/immutable",
                                "status": "do_not_move",
                            }
                        },
                        "pending_entries": {"remote": {"status": "pending"}},
                    }
                ),
                encoding="utf-8",
            )

            path = module.sync_data_manifest(
                quant_data_root=quant_data, reports_root=reports
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["official_entries"]["external_verified_product"]["path"],
            "/shared/immutable",
        )
        self.assertEqual(manifest["pending_entries"]["remote"]["status"], "pending")
        self.assertEqual(manifest["source_report"], "data_hygiene_inventory_latest.json")
        self.assertNotIn("daily_stock_analysis-robust", json.dumps(manifest))

    def test_generic_snapshot_is_stale_when_frozen_upstream_pointer_moves(self):
        module = load_module()
        datasets = {
            "pit_csi800": {
                "dataset_id": "pit_csi800",
                "status": "ready",
                "snapshot_id": "pit-002",
                "content_sha256": "b" * 64,
                "price_basis": "not_applicable",
            },
            "robust_reference": {
                "dataset_id": "robust_reference",
                "data_kind": "point_in_time_reference",
                "role": "canonical",
                "status": "ready",
                "upstream_snapshots": [
                    {
                        "dataset_id": "pit_csi800",
                        "snapshot_id": "pit-001",
                        "content_sha256": "a" * 64,
                        "price_basis": "not_applicable",
                    }
                ],
            },
        }

        module._enrich_derived_entries(datasets)

        self.assertEqual(datasets["robust_reference"]["status"], "stale")
        self.assertEqual(
            datasets["robust_reference"]["blocker_codes"],
            ["upstream_snapshot_mismatch"],
        )


if __name__ == "__main__":
    unittest.main()
