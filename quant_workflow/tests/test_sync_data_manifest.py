import fcntl
import importlib.util
import json
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
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-source-001",
        "content_sha256": "a" * 64,
        "price_basis": "qfq",
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
            (reports / "data_quality_report_latest.md").write_text("# quality", encoding="utf-8")

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
        self.assertEqual(manifest["documented_inputs"]["canonical_qlib_provider"]["calendar_last"], "")
        self.assertIn("global_daily", manifest["documented_inputs"])
        self.assertIn("raw_global_daily", manifest["documented_inputs"])
        self.assertIn("raw_ashare_snapshots", manifest["documented_inputs"])
        self.assertIn("raw_ashare_news", manifest["documented_inputs"])
        self.assertEqual(manifest["datasets"]["ashare_daily_raw"]["status"], "blocked")
        self.assertEqual(manifest["datasets"]["ashare_daily_qfq"]["status"], "blocked")
        self.assertEqual(manifest["datasets"]["qlib_ashare_v2"]["status"], "blocked")
        self.assertNotIn(str(root), json.dumps(manifest))
        self.assertIn(module.README_BLOCK_START, root_readme)
        self.assertIn("ashare_cleaned_qfq_legacy", ashare_readme)
        self.assertEqual(manifest_issues, [])

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


if __name__ == "__main__":
    unittest.main()
