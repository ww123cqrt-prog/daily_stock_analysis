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


class SyncDataManifestTest(unittest.TestCase):
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

        self.assertEqual(manifest["official_entries"]["cleaned_parquet"]["file_count"], 1)
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["calendar_last"], "2026-07-09")
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["instruments_all"], 1)
        self.assertEqual(manifest["documented_inputs"]["canonical_qlib_provider"]["calendar_last"], "")

    def test_manifest_registers_complete_pit_and_marks_reference_pending(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            pit_dir = seed_pit_membership(quant_data)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        pit = manifest["official_entries"]["pit_csi800"]
        self.assertEqual(pit["path"], str(pit_dir))
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
            reference = seed_reference_snapshot(quant_data, membership_hash)

            manifest = module.build_manifest(
                quant_data_root=quant_data,
                reports_root=root / "reports",
            )

        entry = manifest["official_entries"]["robust_reference"]
        self.assertEqual(entry["path"], str(reference))
        self.assertEqual(entry["selected_source"], "baostock")
        self.assertEqual(entry["build_run_id"], "reference_20260718_cache_v2")
        self.assertEqual(entry["date_range"]["start_date"], "2025-01-02")
        self.assertEqual(entry["row_counts"]["st_status.parquet"], 2)
        self.assertEqual(len(entry["reference_snapshot_hash"]), 64)
        self.assertNotIn("robust_reference", manifest.get("pending_entries", {}))

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


if __name__ == "__main__":
    unittest.main()
