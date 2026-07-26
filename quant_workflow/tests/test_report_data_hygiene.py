import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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


class ReportDataHygieneTest(unittest.TestCase):
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
            official_parquet = root / "quant_data" / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
            official_qlib = root / "quant_data" / "qlib" / "cn_data_baostock_jqdata_cleaned"
            legacy_parquet = root / "quant_data" / "parquet" / "ashare_1d" / "qfq"
            legacy_qlib = root / "quant_data" / "qlib" / "cn_data"
            for directory in (official_parquet, legacy_parquet, official_qlib / "calendars", official_qlib / "instruments", legacy_qlib):
                directory.mkdir(parents=True, exist_ok=True)
            (official_parquet / "000001.parquet").write_bytes(b"not-real-parquet")
            (legacy_parquet / "000002.parquet").write_bytes(b"not-real-parquet")
            (official_qlib / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
            (official_qlib / "instruments" / "all.txt").write_text("000001\t2024-01-02\t2024-01-03\n", encoding="utf-8")
            reports.mkdir(parents=True)
            (reports / "data_quality_report_latest.md").write_text("# quality", encoding="utf-8")
            (reports / "data_cleaning_strategy_latest.md").write_text("# strategy", encoding="utf-8")
            (reports / "latest_training_result.json").write_text('{"run_id":"abc"}', encoding="utf-8")
            make_jobs_db(reports / "jobs" / "index.sqlite")
            root_log = root / "quant_data" / "baostock_download.log"
            temp_script = root / "quant_data" / "monitor_akshare.py"
            root_log.write_text("old log", encoding="utf-8")
            temp_script.write_text("print('legacy')\n", encoding="utf-8")

            paths = module.generate_data_hygiene_reports(
                projects_root=root,
                reports_root=reports,
                timestamp="20260706_120000",
                process_lines=[],
            )

            inventory = json.loads(paths.inventory_json.read_text(encoding="utf-8"))
            plan = json.loads(paths.cleanup_plan_json.read_text(encoding="utf-8"))
            root_log_still_exists = root_log.exists()
            temp_script_still_exists = temp_script.exists()
            official_parquet_still_exists = official_parquet.exists()

        self.assertEqual(paths.inventory_md.name, "data_hygiene_inventory_20260706_120000.md")
        self.assertEqual(paths.cleanup_plan_md.name, "data_hygiene_cleanup_plan_20260706_120000.md")
        self.assertTrue(root_log_still_exists, "dry-run report must not move runtime logs")
        self.assertTrue(temp_script_still_exists, "dry-run report must not move temp scripts")
        self.assertTrue(official_parquet_still_exists, "official parquet must stay in place")
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


if __name__ == "__main__":
    unittest.main()
