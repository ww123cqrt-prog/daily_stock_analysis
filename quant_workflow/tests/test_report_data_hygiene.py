import importlib.util
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_data_hygiene.py"
    spec = importlib.util.spec_from_file_location("report_data_hygiene", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_jobs_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                label TEXT NOT NULL,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                command_json TEXT NOT NULL,
                cwd TEXT NOT NULL,
                pid INTEGER,
                return_code INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                error TEXT,
                log_path TEXT NOT NULL,
                artifact_paths_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, label, status, params_json, command_json, cwd,
                return_code, created_at, updated_at, error, log_path, artifact_paths_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "qj_failed",
                "train_baseline_lgb",
                "训练 Qlib LightGBM 选股模型基线",
                "failed",
                "{}",
                "[]",
                str(path.parent),
                1,
                "2026-07-04T13:45:23+08:00",
                "2026-07-04T13:45:40+08:00",
                "Process exited with return code 1.",
                str(path.parent / "qj_failed" / "job.log"),
                "[]",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def write_native_v2_dataset(
    quant_data: Path,
    *,
    dataset_id: str,
    dataset_root: str,
    data_kind: str,
) -> tuple[dict, Path]:
    contract = sys.modules["quant_data_contract"]
    snapshot_id = f"{dataset_id}-snapshot"
    dataset_directory = quant_data / dataset_root
    snapshot = dataset_directory / "snapshots" / snapshot_id
    snapshot.mkdir(parents=True, exist_ok=True)
    if data_kind == "daily_bars":
        (snapshot / "bars").mkdir()
        (snapshot / "factors").mkdir()
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
                "open": [10.0, 10.1],
                "high": [10.2, 10.3],
                "low": [9.9, 10.0],
                "close": [10.1, 10.2],
                "volume": [1000.0, 1200.0],
                "source_id": ["test.raw", "test.raw"],
            }
        ).to_parquet(snapshot / "bars" / "SH600000.parquet", index=False)
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
                "factor": [1.0, 1.0],
            }
        ).to_parquet(snapshot / "factors" / "SH600000.parquet", index=False)
        pd.DataFrame({"symbol": [], "ex_date": []}).to_parquet(
            snapshot / "corporate_actions.parquet", index=False
        )
        contract.finalize_price_snapshot(
            snapshot,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            role="canonical",
            status="ready",
            source={"provider": "test", "interface": "fixture"},
            price_basis="qfq",
            adjustment={
                "method": "provider_native_paired",
                "adjusted_at": "2026-07-25T00:00:00+08:00",
                "adjustment_as_of": "2026-07-24",
                "adjusted_fields": ["open", "high", "low", "close"],
                "normalization_rule": "latest_valid_trade_date_factor_equals_1",
                "algorithm_version": "fixture-v1",
                "paired_raw_snapshot_id": "raw-fixture",
            },
            producer={"script": "test_report_data_hygiene.py", "version": "1"},
            source_retrieved_at="2026-07-25T00:00:00+08:00",
        )
    else:
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
            b"fixture-feature"
        )
        qfq_manifest = json.loads(
            (
                quant_data
                / "parquet"
                / "ashare_1d"
                / "ashare_daily_qfq"
                / "snapshots"
                / "ashare_daily_qfq-snapshot"
                / "DATASET_MANIFEST.json"
            ).read_text(encoding="utf-8")
        )
        upstream = {
            field: qfq_manifest[field]
            for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
        }
        contract.atomic_write_json(
            snapshot / "PROVIDER_MANIFEST.json",
            {"schema_version": 2, "status": "ready", "source_snapshot": upstream},
        )
        dependencies = []
        if dataset_id == "qlib_ashare_cleaned_v2":
            qlib_manifest = json.loads(
                (
                    quant_data
                    / "qlib"
                    / "cn_data_ashare_v2"
                    / "snapshots"
                    / "qlib_ashare_v2-snapshot"
                    / "DATASET_MANIFEST.json"
                ).read_text(encoding="utf-8")
            )
            dependencies = [
                {
                    field: qlib_manifest[field]
                    for field in (
                        "dataset_id",
                        "snapshot_id",
                        "content_sha256",
                        "price_basis",
                    )
                }
            ]
        contract.finalize_qlib_snapshot(
            snapshot,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            source={"provider": "test", "interface": "fixture"},
            source_retrieved_at="2026-07-25T00:00:00+08:00",
            producer={"script": "test_report_data_hygiene.py", "version": "1"},
            upstream_snapshot=upstream,
            dependency_snapshots=dependencies,
        )
    manifest = json.loads(
        (snapshot / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    relative_snapshot = snapshot.relative_to(quant_data).as_posix()
    latest_path = dataset_directory / "LATEST.json"
    latest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_id": dataset_id,
                "snapshot_id": snapshot_id,
                "snapshot_path": relative_snapshot,
                "manifest_sha256": manifest_sha256,
            }
        ),
        encoding="utf-8",
    )
    return (
        {
            **manifest,
            "dataset_root": dataset_root,
            "latest_pointer": latest_path.relative_to(quant_data).as_posix(),
            "snapshot_manifest_sha256": manifest_sha256,
        },
        snapshot,
    )


def write_fixed_reference_datasets(module, quant_data: Path) -> tuple[dict, dict[str, Path]]:
    start_date = "2014-01-02"
    end_date = "2026-07-16"
    reference = quant_data / "reference"
    pit = reference / "pit_csi800"
    (pit / "daily").mkdir(parents=True)
    (pit / ".build_pit_csi800.lock").touch()
    (pit / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "baostock",
                "selected": "baostock",
                "fallback_reason": None,
                "date_range": {"start_date": start_date, "end_date": end_date},
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )
    (pit / "membership_intervals.parquet").write_bytes(b"membership-intervals")
    (pit / "daily" / "2014-01-02.parquet").write_bytes(b"daily-membership")
    _metadata, membership_hash = module.read_membership_snapshot(
        pit,
        start_date=start_date,
        end_date=end_date,
    )

    (quant_data / ".reference.download_robust_reference.lock").touch()
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "baostock",
                "selected": "baostock",
                "fallback_reason": None,
                "industry_classification": "fixture",
                "membership_snapshot_hash": membership_hash,
                "date_range": {"start_date": start_date, "end_date": end_date},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "security_master.parquet",
        "industry_intervals.parquet",
        "st_status.parquet",
        "suspension_status.parquet",
    ):
        (reference / name).write_bytes(name.encode("ascii"))
    reference_hash = module.reference_snapshot_fingerprint(reference)
    date_range = {"start_date": start_date, "end_date": end_date}
    entries = {
        "pit_csi800": {
            "dataset_id": "pit_csi800",
            "data_kind": "point_in_time_membership",
            "role": "canonical",
            "status": "ready",
            "price_basis": "not_applicable",
            "path": pit.relative_to(quant_data).as_posix(),
            "snapshot_available": False,
            "snapshot_hash": membership_hash,
            "date_range": date_range,
        },
        "robust_reference": {
            "dataset_id": "robust_reference",
            "data_kind": "point_in_time_reference",
            "role": "canonical",
            "status": "ready",
            "price_basis": "not_applicable",
            "path": reference.relative_to(quant_data).as_posix(),
            "snapshot_available": False,
            "snapshot_hash": reference_hash,
            "date_range": date_range,
        },
    }
    return entries, {"pit_csi800": pit, "robust_reference": reference}


class ReportDataHygieneTest(unittest.TestCase):
    def test_default_scan_uses_configured_quant_data_root(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            projects = root / "projects"
            configured_quant_data = root / "configured-quant-data"
            reports = root / "reports"
            projects.mkdir()
            configured_quant_data.mkdir()
            marker = configured_quant_data / "configured.log"
            marker.write_text("configured", encoding="utf-8")

            with (
                patch.object(module, "default_projects_root", return_value=projects),
                patch.object(
                    module,
                    "default_quant_data_root",
                    return_value=configured_quant_data,
                ),
            ):
                paths = module.generate_data_hygiene_reports(
                    reports_root=reports,
                    timestamp="20260731_120000",
                    process_lines=[],
                )

            inventory = json.loads(paths.inventory_json.read_text(encoding="utf-8"))
            item_paths = {item["path"] for item in inventory["items"]}

        self.assertIn(str(marker.resolve()), item_paths)
        self.assertNotIn(str((projects / "quant_data" / marker.name).resolve()), item_paths)

    def test_collect_v2_catalog_rejects_empty_immutable_snapshot_shell(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            dataset_root = quant_data / "parquet" / "ashare_1d" / "ashare_daily_qfq"
            snapshot = dataset_root / "snapshots" / "empty-snapshot"
            snapshot.mkdir(parents=True)
            manifest_path = snapshot / "DATASET_MANIFEST.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "dataset_id": "ashare_daily_qfq",
                        "snapshot_id": "empty-snapshot",
                        "data_kind": "daily_bars",
                        "role": "canonical",
                        "status": "ready",
                        "price_basis": "qfq",
                    }
                ),
                encoding="utf-8",
            )
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (dataset_root / "LATEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "dataset_id": "ashare_daily_qfq",
                        "snapshot_id": "empty-snapshot",
                        "snapshot_path": snapshot.relative_to(quant_data).as_posix(),
                        "manifest_sha256": manifest_sha256,
                    }
                ),
                encoding="utf-8",
            )
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "path_format": "relative_to_quant_data_root",
                        "datasets": {
                            "ashare_daily_qfq": {
                                "dataset_id": "ashare_daily_qfq",
                                "dataset_root": dataset_root.relative_to(
                                    quant_data
                                ).as_posix(),
                                "latest_pointer": (dataset_root / "LATEST.json")
                                .relative_to(quant_data)
                                .as_posix(),
                                "snapshot_id": "empty-snapshot",
                                "snapshot_manifest_sha256": manifest_sha256,
                                "data_kind": "daily_bars",
                                "role": "canonical",
                                "status": "ready",
                                "price_basis": "qfq",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            catalog = module.collect_v2_catalog(quant_data)

        self.assertEqual(catalog["canonical_ready_count"], 0)
        self.assertEqual(catalog["status"], "partial")
        self.assertFalse(catalog["datasets"][0]["canonical_valid"])
        self.assertIn("incomplete", catalog["datasets"][0]["error"])

    def test_collect_v2_catalog_rejects_changed_fixed_reference_data(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            entries, _paths = write_fixed_reference_datasets(module, quant_data)
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "path_format": "relative_to_quant_data_root",
                        "datasets": entries,
                    }
                ),
                encoding="utf-8",
            )
            (quant_data / "reference" / "st_status.parquet").write_bytes(b"changed")

            catalog = module.collect_v2_catalog(quant_data)

        by_id = {item["dataset_id"]: item for item in catalog["datasets"]}
        self.assertTrue(by_id["pit_csi800"]["canonical_valid"])
        self.assertFalse(by_id["robust_reference"]["canonical_valid"])
        self.assertIn("hash mismatch", by_id["robust_reference"]["error"])

    def test_collect_v2_catalog_binds_root_entry_to_snapshot_contract(self):
        module = load_module()
        cases = {
            "catalog_key": lambda entry: ({"wrong_catalog_key": entry}),
            "content_sha256": lambda entry: {
                "ashare_daily_qfq": {**entry, "content_sha256": "f" * 64}
            },
            "data_kind": lambda entry: {
                "ashare_daily_qfq": {**entry, "data_kind": "qlib_provider"}
            },
        }
        for label, mutate in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                quant_data = Path(temp_dir) / "quant_data"
                entry, _snapshot = write_native_v2_dataset(
                    quant_data,
                    dataset_id="ashare_daily_qfq",
                    dataset_root="parquet/ashare_1d/ashare_daily_qfq",
                    data_kind="daily_bars",
                )
                (quant_data / "DATA_MANIFEST.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "path_format": "relative_to_quant_data_root",
                            "datasets": mutate(entry),
                        }
                    ),
                    encoding="utf-8",
                )

                catalog = module.collect_v2_catalog(quant_data)

            self.assertEqual(catalog["canonical_ready_count"], 0)
            self.assertEqual(catalog["status"], "partial")
            self.assertTrue(catalog["errors"])

    def test_collect_v2_catalog_rejects_snapshot_changed_during_validation(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            entry, snapshot = write_native_v2_dataset(
                quant_data,
                dataset_id="ashare_daily_qfq",
                dataset_root="parquet/ashare_1d/ashare_daily_qfq",
                data_kind="daily_bars",
            )
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "path_format": "relative_to_quant_data_root",
                        "datasets": {"ashare_daily_qfq": entry},
                    }
                ),
                encoding="utf-8",
            )
            original_resolve = module.resolve_dataset

            def mutate_after_validation(*args, **kwargs):
                resolved = original_resolve(*args, **kwargs)
                bars = snapshot / "bars" / "SH600000.parquet"
                bars.write_bytes(bars.read_bytes() + b"changed-after-validation")
                return resolved

            with patch.object(
                module,
                "resolve_dataset",
                side_effect=mutate_after_validation,
            ):
                catalog = module.collect_v2_catalog(quant_data)

        self.assertEqual(catalog["canonical_ready_count"], 0)
        self.assertEqual(catalog["status"], "partial")
        self.assertFalse(catalog["datasets"][0]["canonical_valid"])
        self.assertIn("changed during validation", catalog["datasets"][0]["error"])

    def test_collect_jobs_opens_existing_database_with_read_only_uri(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "jobs" / "index.sqlite"
            make_jobs_db(db_path)
            real_connect = module.sqlite3.connect
            calls = []

            def recording_connect(*args, **kwargs):
                calls.append((args, kwargs))
                return real_connect(*args, **kwargs)

            with patch.object(module.sqlite3, "connect", side_effect=recording_connect):
                jobs = module.collect_jobs(db_path)

        self.assertEqual(jobs["attention"][0]["status"], "failed")
        self.assertTrue(calls)
        self.assertTrue(str(calls[0][0][0]).startswith("file:"), calls[0])
        self.assertTrue(calls[0][1].get("uri"), calls[0])
        self.assertIn("mode=ro", str(calls[0][0][0]))

    def test_process_inventory_redacts_sensitive_command_fragments(self):
        module = load_module()

        matches = module.collect_process_matches(
            [
                "123 1 S 00:01 python download_jqdata.py --token abc123 JQ_PASSWORD=secret",
                "456 1 S 00:01 python train_baseline_lgb.py --config baseline.yaml",
            ]
        )

        joined = "\n".join(matches)
        self.assertIn("[REDACTED]", joined)
        self.assertNotIn("secret", joined)
        self.assertNotIn("abc123", joined)
        self.assertIn("train_baseline_lgb.py", joined)

    def test_generate_reports_is_inventory_and_dry_run_only(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / "quant_workflow" / "reports"
            quant_data = root / "quant_data"
            datasets = {}
            canonical_snapshots = {}
            for dataset_id, dataset_root, data_kind in (
                ("ashare_daily_qfq", "parquet/ashare_1d/ashare_daily_qfq", "daily_bars"),
                ("qlib_ashare_v2", "qlib/cn_data_ashare_v2", "qlib_provider"),
                (
                    "qlib_ashare_cleaned_v2",
                    "qlib/cn_data_ashare_cleaned_v2",
                    "qlib_provider",
                ),
            ):
                entry, snapshot = write_native_v2_dataset(
                    quant_data,
                    dataset_id=dataset_id,
                    dataset_root=dataset_root,
                    data_kind=data_kind,
                )
                datasets[dataset_id] = entry
                canonical_snapshots[dataset_id] = snapshot

            fixed_entries, fixed_paths = write_fixed_reference_datasets(
                module, quant_data
            )
            datasets.update(fixed_entries)
            canonical_snapshots.update(fixed_paths)

            legacy_parquet = (
                quant_data
                / "parquet"
                / "ashare_1d"
                / "merged_baostock_jqdata_cleaned"
            )
            legacy_qlib = quant_data / "qlib" / "cn_data_baostock_jqdata_cleaned"
            datasets["ashare_cleaned_qfq_compatibility"] = {
                "dataset_id": "ashare_cleaned_qfq_compatibility",
                "data_kind": "daily_bars",
                "role": "legacy",
                "status": "ready",
                "path": legacy_parquet.relative_to(quant_data).as_posix(),
            }
            datasets["qlib_ashare_cleaned_compatibility"] = {
                "dataset_id": "qlib_ashare_cleaned_compatibility",
                "data_kind": "qlib_provider",
                "role": "legacy",
                "status": "ready",
                "path": legacy_qlib.relative_to(quant_data).as_posix(),
            }
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "path_format": "relative_to_quant_data_root",
                        "datasets": datasets,
                    }
                ),
                encoding="utf-8",
            )

            old_parquet = quant_data / "parquet" / "ashare_1d" / "qfq"
            old_qlib = quant_data / "qlib" / "cn_data"
            for directory in (
                legacy_parquet,
                old_parquet,
                legacy_qlib / "calendars",
                legacy_qlib / "instruments",
                old_qlib,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            (legacy_parquet / "000001.parquet").write_bytes(b"not-real-parquet")
            (old_parquet / "000002.parquet").write_bytes(b"not-real-parquet")
            (legacy_qlib / "calendars" / "day.txt").write_text(
                "2024-01-02\n2024-01-03\n", encoding="utf-8"
            )
            (legacy_qlib / "instruments" / "all.txt").write_text(
                "000001\t2024-01-02\t2024-01-03\n", encoding="utf-8"
            )
            reports.mkdir(parents=True)
            (reports / "data_quality_report_latest.json").write_text(
                '{"publication_contract":"atomic_json_pointer_to_immutable_markdown_v1"}',
                encoding="utf-8",
            )
            (reports / "data_cleaning_strategy_latest.md").write_text("# strategy", encoding="utf-8")
            (reports / "latest_training_result.json").write_text('{"run_id":"abc"}', encoding="utf-8")
            make_jobs_db(reports / "jobs" / "index.sqlite")
            root_log = root / "quant_data" / "baostock_download.log"
            temp_script = root / "quant_data" / "monitor_akshare.py"
            root_log.write_text("old log", encoding="utf-8")
            temp_script.write_text("print('legacy')\n", encoding="utf-8")

            fingerprint_limits = []
            original_fingerprint = module.filesystem_state_fingerprint

            def bounded_fingerprint(path, **kwargs):
                fingerprint_limits.append(kwargs)
                return original_fingerprint(path, **kwargs)

            with patch.object(
                module,
                "filesystem_state_fingerprint",
                side_effect=bounded_fingerprint,
            ):
                paths = module.generate_data_hygiene_reports(
                    projects_root=root,
                    reports_root=reports,
                    timestamp="20260706_120000",
                    process_lines=[],
                )

            inventory = json.loads(paths.inventory_json.read_text(encoding="utf-8"))
            plan = json.loads(paths.cleanup_plan_json.read_text(encoding="utf-8"))
            bundle = json.loads(
                paths.bundle_latest_json.read_text(encoding="utf-8")
            )
            bundle_inventory = reports / bundle["inventory_artifact"]
            bundle_cleanup = reports / bundle["cleanup_plan_artifact"]
            bundle_hashes_match = (
                hashlib.sha256(bundle_inventory.read_bytes()).hexdigest()
                == bundle["inventory_sha256"]
                and hashlib.sha256(bundle_cleanup.read_bytes()).hexdigest()
                == bundle["cleanup_plan_sha256"]
            )
            root_log_still_exists = root_log.exists()
            temp_script_still_exists = temp_script.exists()
            canonical_snapshots_still_exist = all(
                path.exists() for path in canonical_snapshots.values()
            )
            public_report_text = "\n".join(
                (
                    paths.inventory_md.read_text(encoding="utf-8"),
                    paths.cleanup_plan_md.read_text(encoding="utf-8"),
                    json.dumps(inventory, ensure_ascii=False),
                    json.dumps(plan, ensure_ascii=False),
                )
            )

        self.assertRegex(
            paths.inventory_md.name,
            r"^data_hygiene_inventory_20260706_120000_[0-9a-f]{16}\.md$",
        )
        self.assertRegex(
            paths.cleanup_plan_md.name,
            r"^data_hygiene_cleanup_plan_20260706_120000_[0-9a-f]{16}\.md$",
        )
        self.assertEqual(
            bundle["publication_contract"],
            "atomic_json_pointer_to_immutable_hygiene_bundle_v1",
        )
        self.assertTrue(bundle_hashes_match)
        self.assertTrue(fingerprint_limits)
        self.assertTrue(
            all(
                limits.get("max_entries") == module.HYGIENE_STATE_SCAN_MAX_ENTRIES
                and limits.get("max_seconds") == module.HYGIENE_STATE_SCAN_MAX_SECONDS
                for limits in fingerprint_limits
            )
        )
        self.assertTrue(root_log_still_exists, "dry-run report must not move runtime logs")
        self.assertTrue(temp_script_still_exists, "dry-run report must not move temp scripts")
        self.assertTrue(
            canonical_snapshots_still_exist,
            "canonical V2 snapshots must stay in place",
        )
        self.assertFalse(plan["contains_delete_action"])
        self.assertTrue(
            all(
                action["requires_user_confirmation"] is True
                for action in plan["actions"]
                if action["action"] in {"move", "archive"}
            )
        )
        categories = {item["category"] for item in inventory["items"]}
        self.assertIn("official", categories)
        self.assertIn("legacy", categories)
        self.assertIn("runtime_log", categories)
        self.assertIn("temp_script", categories)
        self.assertEqual(inventory["jobs"]["attention"][0]["status"], "failed")
        self.assertEqual(inventory["catalog"]["schema_version"], 2)
        self.assertTrue(
            inventory["official_status"]["latest_quality_report_pointer"]["exists"]
        )
        self.assertEqual(inventory["catalog"]["canonical_ready_count"], 5)
        self.assertRegex(inventory["catalog"]["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            inventory["catalog"]["validated_dataset_ids"],
            [
                "ashare_daily_qfq",
                "pit_csi800",
                "qlib_ashare_cleaned_v2",
                "qlib_ashare_v2",
                "robust_reference",
            ],
        )
        official_dataset_ids = {
            item.get("dataset_id")
            for item in inventory["items"]
            if item["category"] == "official"
        }
        self.assertEqual(
            official_dataset_ids,
            {
                "ashare_daily_qfq",
                "qlib_ashare_v2",
                "qlib_ashare_cleaned_v2",
                "pit_csi800",
                "robust_reference",
            },
        )
        by_path = {item["path"]: item for item in inventory["items"]}
        self.assertEqual(by_path[str(legacy_parquet)]["category"], "legacy")
        self.assertEqual(by_path[str(legacy_qlib)]["category"], "legacy")
        keep_sources = {
            action["source"]
            for action in plan["actions"]
            if action["action"] == "keep"
        }
        self.assertTrue(
            {str(path.resolve()) for path in canonical_snapshots.values()}.issubset(
                keep_sources
            )
        )
        self.assertNotIn(str(legacy_parquet), keep_sources)
        self.assertNotIn(str(legacy_qlib), keep_sources)
        self.assertIn("Qlib Provider", public_report_text)
        self.assertNotRegex(
            public_report_text,
            r"(?i)direct provider|native V2|原生\s*V2",
        )


if __name__ == "__main__":
    unittest.main()
