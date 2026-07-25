# -*- coding: utf-8 -*-
"""Tests for the local quant workflow job API."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError, URLError

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import create_app
from src.services.quant_job_service import QuantJob, QuantJobService, REPORTS_ROOT


class _FakeHTTPResponse:
    def __init__(self, status_code: int = 200) -> None:
        self._status_code = status_code

    def getcode(self) -> int:
        return self._status_code

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeSocket:
    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class QuantJobsApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from api.v1.endpoints.quant_jobs import _clear_mlflow_status_cache, _clear_quant_job_service_cache

        _clear_mlflow_status_cache()
        _clear_quant_job_service_cache()

    def _make_client(self, service: QuantJobService) -> TestClient:
        static_dir = service.jobs_root / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        patcher = patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service)
        self.addCleanup(patcher.stop)
        patcher.start()
        return TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})

    def _assert_reports_dir_arg(self, command: list[str], flag: str = "--out-dir") -> None:
        self.assertIn(flag, command)
        reports_dir = command[command.index(flag) + 1]
        self.assertTrue(reports_dir.endswith("quant_workflow/reports"), reports_dir)

    def test_sanitize_job_error_redacts_absolute_paths(self) -> None:
        from api.v1.endpoints.quant_jobs import _sanitize_job_error

        message = (
            "Failed to start job: [Errno 2] No such file or directory: "
            "'/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/scripts/missing.py'; "
            "container=/app/quant_data/raw/secret.csv; "
            "fallback=C:\\Users\\cq\\secret\\runner.py; "
            "drive_relative=C:secret\\runner.py; "
            "metrics=E:5 R:42"
        )

        sanitized = _sanitize_job_error(message)

        self.assertIsNotNone(sanitized)
        assert sanitized is not None
        self.assertIn("'[path]'", sanitized)
        self.assertIn("fallback=[path]", sanitized)
        self.assertIn("drive_relative=[path]", sanitized)
        self.assertIn("metrics=E:5 R:42", sanitized)
        self.assertNotIn("/Users/cq", sanitized)
        self.assertNotIn("/app/quant_data", sanitized)
        self.assertNotIn("C:\\Users", sanitized)
        self.assertNotIn("C:secret", sanitized)

    def test_rejects_unknown_job_type_before_any_process_is_started(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            client = self._make_client(service)

            resp = client.post("/api/v1/quant/jobs", json={"job_type": "rm_everything"})

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")
        self.assertIn("Unsupported quant job type", resp.json()["message"])

    def test_submit_job_persists_metadata_and_exposes_combined_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "check_qlib_data": {
                        "label": "检查 Qlib 数据",
                        "command": [sys.executable, "-c", "print('qlib ok')"],
                    },
                },
            )
            client = self._make_client(service)

            submit_resp = client.post("/api/v1/quant/jobs", json={"job_type": "check_qlib_data"})
            self.assertEqual(submit_resp.status_code, 200)
            job_id = submit_resp.json()["job_id"]
            self.assertNotIn("command", submit_resp.json())
            self.assertNotIn("cwd", submit_resp.json())
            self.assertNotIn("log_path", submit_resp.json())

            deadline = time.time() + 5
            detail = {}
            while time.time() < deadline:
                detail = client.get(f"/api/v1/quant/jobs/{job_id}").json()
                if detail["status"] == "completed":
                    break
                time.sleep(0.05)

            jobs_resp = client.get("/api/v1/quant/jobs")
            log_resp = client.get(f"/api/v1/quant/jobs/{job_id}/log")
            db_path = Path(temp_dir) / "jobs" / "index.sqlite"
            db_exists = db_path.exists()

        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["return_code"], 0)
        self.assertEqual(detail["job_type"], "check_qlib_data")
        self.assertNotIn("command", detail)
        self.assertNotIn("cwd", detail)
        self.assertNotIn("log_path", detail)
        self.assertEqual(jobs_resp.status_code, 200)
        self.assertEqual(jobs_resp.json()["items"][0]["job_id"], job_id)
        self.assertNotIn("command", jobs_resp.json()["items"][0])
        self.assertNotIn("cwd", jobs_resp.json()["items"][0])
        self.assertNotIn("log_path", jobs_resp.json()["items"][0])
        self.assertEqual(log_resp.status_code, 200)
        self.assertIn("qlib ok", log_resp.json()["content"])
        self.assertIn("[command]", log_resp.json()["content"])
        self.assertNotIn("log_path", log_resp.json())
        self.assertTrue(db_exists)

    def test_status_lists_available_jobs_and_latest_job_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "report_data_quality": {
                        "label": "数据质量报告",
                        "command": [sys.executable, "-c", "print('quality ok')"],
                    },
                },
            )
            client = self._make_client(service)
            client.post("/api/v1/quant/jobs", json={"job_type": "report_data_quality"})

            deadline = time.time() + 5
            status_body = {}
            while time.time() < deadline:
                status_body = client.get("/api/v1/quant/status").json()
                latest = status_body["latest_jobs"].get("report_data_quality")
                if latest and latest["status"] == "completed":
                    break
                time.sleep(0.05)

        self.assertIn("report_data_quality", status_body["available_job_types"])
        self.assertEqual(status_body["latest_jobs"]["report_data_quality"]["status"], "completed")
        self.assertNotIn("command", status_body["latest_jobs"]["report_data_quality"])
        self.assertNotIn("cwd", status_body["latest_jobs"]["report_data_quality"])
        self.assertNotIn("log_path", status_body["latest_jobs"]["report_data_quality"])
        self.assertEqual(status_body["active_count"], 0)

    def test_job_response_exposes_artifact_names_without_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            artifact_path = reports_dir / "nested" / "quality_report_latest.md"
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text("# quality\n", encoding="utf-8")
            service = QuantJobService(
                jobs_root=reports_dir / "jobs",
                command_specs={},
            )
            service._insert_job(
                QuantJob(
                    job_id="qj_public_artifacts",
                    job_type="report_data_quality",
                    label="数据质量报告",
                    status="completed",
                    params={},
                    command=[sys.executable],
                    cwd=temp_dir,
                    return_code=0,
                    created_at="2026-07-04T01:00:00+08:00",
                    updated_at="2026-07-04T01:00:01+08:00",
                    error=f"Failed to start job: [Errno 2] No such file or directory: '{temp_dir}/missing.py'",
                    log_path=str(service.jobs_root / "qj_public_artifacts" / "job.log"),
                    artifact_paths=[str(artifact_path)],
                )
            )
            client = self._make_client(service)

            resp = client.get("/api/v1/quant/jobs/qj_public_artifacts")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["artifact_paths"], ["quality_report_latest.md"])
        self.assertNotIn(str(temp_dir), json.dumps(body))
        self.assertEqual(body["error"], "Failed to start job: [Errno 2] No such file or directory: '[path]'")
        self.assertNotIn("cwd", body)
        self.assertNotIn("log_path", body)

    def test_default_specs_include_provider_maintenance_dry_run_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            available = service.available_job_types()

        self.assertIn("generate_csi300", available)
        self.assertIn("add_index_benchmark", available)
        self.assertEqual(available["generate_csi300"]["allowed_params"], [])
        self.assertEqual(available["add_index_benchmark"]["allowed_params"], [])

    def test_provider_maintenance_jobs_are_forced_to_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            csi300 = service._build_command("generate_csi300", {})
            benchmark = service._build_command("add_index_benchmark", {})

        self.assertTrue(csi300[0].endswith("daily_stock_analysis/.venv/bin/python"), csi300)
        self.assertTrue(benchmark[0].endswith("daily_stock_analysis/.venv/bin/python"), benchmark)
        self.assertIn("generate_csi300.py", csi300[1])
        self.assertIn("add_index_benchmark.py", benchmark[1])
        self.assertIn("--dry-run", csi300)
        self.assertIn("--dry-run", benchmark)

    def test_provider_maintenance_jobs_reject_api_params_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            client = self._make_client(service)

            csi300_resp = client.post(
                "/api/v1/quant/jobs",
                json={"job_type": "generate_csi300", "params": {"dry_run": False}},
            )
            benchmark_resp = client.post(
                "/api/v1/quant/jobs",
                json={"job_type": "add_index_benchmark", "params": {"dry_run": False}},
            )
            jobs = service.list_jobs()

        self.assertEqual(csi300_resp.status_code, 400)
        self.assertEqual(csi300_resp.json()["error"], "invalid_params")
        self.assertIn("Unsupported params for generate_csi300", csi300_resp.json()["message"])
        self.assertEqual(benchmark_resp.status_code, 400)
        self.assertEqual(benchmark_resp.json()["error"], "invalid_params")
        self.assertIn("Unsupported params for add_index_benchmark", benchmark_resp.json()["message"])
        self.assertEqual(jobs, [])

    def test_data_hygiene_report_job_is_inventory_and_dry_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            command = service._build_command("report_data_hygiene", {})
            available = service.available_job_types()
            artifact_patterns = service.command_specs["report_data_hygiene"]["artifact_patterns"]

            with self.assertRaises(ValueError) as ctx:
                service._build_command("report_data_hygiene", {"delete": True})

        self.assertIn("report_data_hygiene.py", command[1])
        self.assertEqual(available["report_data_hygiene"]["allowed_params"], [])
        self.assertIn("Unsupported params for report_data_hygiene", str(ctx.exception))
        self.assertEqual(
            [Path(pattern).name for pattern in artifact_patterns],
            [
                "data_hygiene_inventory_latest.md",
                "data_hygiene_inventory_latest.json",
                "data_hygiene_cleanup_plan_latest.md",
                "data_hygiene_cleanup_plan_latest.json",
            ],
        )
        self.assertTrue(all(str(pattern).startswith(str(REPORTS_ROOT)) for pattern in artifact_patterns))
        self.assertFalse(any("*" in pattern for pattern in artifact_patterns))

    def test_workbench_capability_report_job_is_read_only_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            command = service._build_command("report_workbench_capabilities", {})
            available = service.available_job_types()
            artifact_patterns = service.command_specs["report_workbench_capabilities"]["artifact_patterns"]

            with self.assertRaises(ValueError) as ctx:
                service._build_command("report_workbench_capabilities", {"start_training": True})

        self.assertTrue(command[0].endswith("daily_stock_analysis/.venv/bin/python"), command)
        self.assertIn("report_workbench_capabilities.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertEqual(available["report_workbench_capabilities"]["allowed_params"], [])
        self.assertIn("Unsupported params for report_workbench_capabilities", str(ctx.exception))
        self.assertEqual(
            [Path(pattern).name for pattern in artifact_patterns],
            [
                "workbench_capability_inventory_latest.md",
                "workbench_capability_inventory_latest.json",
            ],
        )
        self.assertTrue(all(str(pattern).startswith(str(REPORTS_ROOT)) for pattern in artifact_patterns))
        self.assertFalse(any("*" in pattern for pattern in artifact_patterns))

    def test_qlib_research_report_jobs_are_fixed_section_read_only_reports(self) -> None:
        expected_sections = {
            "report_qlib_model_explainability": ("model_explainability", {"topk": "12"}),
            "report_qlib_backtest_detail": ("backtest_detail", {"run_id": "safe_run"}),
            "report_qlib_model_comparison": ("model_comparison", {"max_runs": "6"}),
            "report_qlib_factor_analysis": ("factor_analysis", {"topk": "10"}),
            "report_qlib_rolling_training": (
                "rolling_training",
                {
                    "windows": "4",
                    "train_years": "3",
                    "valid_months": "12",
                    "test_months": "1",
                    "step_months": "1",
                    "config": "baseline_lgb_alpha158.yaml",
                },
            ),
            "report_qlib_risk_model": ("risk_model", {"run_id": "safe_run"}),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            available = service.available_job_types()
            commands = {
                job_type: service._build_command(job_type, params)
                for job_type, (_section, params) in expected_sections.items()
            }

            with self.assertRaises(ValueError) as unsafe_run:
                service._build_command("report_qlib_model_explainability", {"run_id": "../unsafe"})
            with self.assertRaises(ValueError) as too_many_runs:
                service._build_command("report_qlib_model_comparison", {"max_runs": "51"})
            with self.assertRaises(ValueError) as unsafe_config:
                service._build_command("report_qlib_rolling_training", {"config": "../baseline.yaml"})
            with self.assertRaises(ValueError) as encoded_config:
                service._build_command("report_qlib_rolling_training", {"config": "%2e%2e.yaml"})
            with self.assertRaises(ValueError) as wrong_suffix_config:
                service._build_command("report_qlib_rolling_training", {"config": "baseline.txt"})
            with self.assertRaises(ValueError) as bad_char_config:
                service._build_command("report_qlib_rolling_training", {"config": "baseline$.yaml"})
            with self.assertRaises(ValueError) as too_many_windows:
                service._build_command("report_qlib_rolling_training", {"windows": "61"})
            with self.assertRaises(ValueError) as too_many_train_years:
                service._build_command("report_qlib_rolling_training", {"train_years": "21"})
            with self.assertRaises(ValueError) as too_many_months:
                service._build_command("report_qlib_rolling_training", {"test_months": "121"})

        for job_type, (section, _params) in expected_sections.items():
            command = commands[job_type]
            self.assertTrue(command[0].endswith("daily_stock_analysis/.venv/bin/python"), command)
            self.assertIn("report_qlib_research.py", command[1])
            self._assert_reports_dir_arg(command)
            self.assertEqual(command[command.index("--section") + 1], section)
            self.assertNotIn("all", command)
            self.assertIn(job_type, available)

            artifact_patterns = service.command_specs[job_type]["artifact_patterns"]
            artifact_names = [Path(pattern).name for pattern in artifact_patterns]
            self.assertIn(f"qlib_research_{section}_latest.json", artifact_names)
            self.assertIn(f"qlib_research_{section}_latest.md", artifact_names)
            self.assertTrue(all(str(pattern).startswith(str(REPORTS_ROOT)) for pattern in artifact_patterns))

        self.assertIn("run_id may only contain", str(unsafe_run.exception))
        self.assertIn("Param max_runs must be <=", str(too_many_runs.exception))
        self.assertIn("Config must be a file name", str(unsafe_config.exception))
        self.assertIn("Config must be a file name", str(encoded_config.exception))
        self.assertIn("Config must be a .yaml or .yml file", str(wrong_suffix_config.exception))
        self.assertIn("Config file name may only contain", str(bad_char_config.exception))
        self.assertIn("Param windows must be <=", str(too_many_windows.exception))
        self.assertIn("Param train_years must be <=", str(too_many_train_years.exception))
        self.assertIn("Param test_months must be <=", str(too_many_months.exception))

    def test_repair_qlib_provider_web_job_is_forced_to_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs=None)
            command = service._build_command("repair_qlib_provider", {})
            available = service.available_job_types()

            with self.assertRaises(ValueError) as ctx:
                service._build_command("repair_qlib_provider", {"dry_run": False})
            with self.assertRaises(ValueError) as legacy_ctx:
                service._build_command("repair_qlib_provider", {"dry_run": True})

        self.assertIn("repair_qlib_provider.py", command[1])
        self.assertIn("--dry-run", command)
        self.assertEqual(available["repair_qlib_provider"]["allowed_params"], [])
        self.assertIn("Unsupported params", str(ctx.exception))
        self.assertIn("Unsupported params", str(legacy_ctx.exception))

    def test_status_includes_research_snapshot_for_workflow_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T15:30:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "Qlib provider is ready."},
                        "data_sources": [
                            {
                                "name": "merged",
                                "files": 5340,
                                "rows": 7580087,
                                "date_range": "2008-01-02 ~ 2026-07-03",
                                "status": "available",
                            }
                        ],
                        "qlib_provider": {
                            "path": "/tmp/qlib",
                            "calendar_days": 4492,
                            "calendar_range": "2008-01-02 ~ 2026-07-03",
                            "instruments": 5340,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "latest_training_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "experiment_name": "baseline_lgb_alpha158_cleaned",
                        "run_id": "run_20260705",
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "qlib_research_risk_model_latest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T16:00:00+08:00",
                        "mode": "read_only_qlib_research",
                        "section": "risk_model",
                        "title": "风险模型集成诊断",
                        "status": "available",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            client = self._make_client(service)

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "trainable")
        self.assertEqual(body["data_sources"][0]["name"], "merged")
        self.assertEqual(body["qlib_provider"]["calendar_range"], "2008-01-02 ~ 2026-07-03")
        self.assertEqual(body["latest_training"]["run_id"], "run_20260705")
        self.assertEqual(body["qlib_research"]["sections"]["risk_model"]["status"], "available")
        self.assertEqual(body["mlflow"]["url"], "http://127.0.0.1:6006/")
        self.assertEqual(body["mlflow"]["role"], "experiment_raw_records")
        self.assertEqual(body["external_services"]["dsa"]["status"], "available")
        self.assertEqual(body["external_services"]["mlflow"]["url"], "http://127.0.0.1:6006/")
        self.assertEqual(body["external_services"]["catdesk"]["url"], "http://127.0.0.1:8765/")

    def test_status_mlflow_url_can_be_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T15:30:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6606/"}, clear=False),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["mlflow"]["url"], "http://127.0.0.1:6606/")

    def test_status_redacts_mlflow_url_userinfo_query_and_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(json.dumps({"generated_at": "2026-07-05T15:30:00+08:00"}), encoding="utf-8")
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch.dict(
                    os.environ,
                    {"MLFLOW_URL": "http://fake_user:fake_pass@127.0.0.1:6606/?token=fake#frag"},
                    clear=False,
                ),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()["mlflow"]
        self.assertEqual(body["url"], "http://127.0.0.1:6606/")
        self.assertEqual(body["status"], "invalid")
        self.assertNotIn("fake_pass", json.dumps(body))
        self.assertNotIn("token=fake", json.dumps(body))

    def test_status_reports_available_local_mlflow_when_health_check_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.quant_jobs.urlopen", return_value=_FakeHTTPResponse(200)) as mocked_urlopen,
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mlflow"]["url"], "http://127.0.0.1:6006/")
        self.assertEqual(body["mlflow"]["status"], "available")
        self.assertEqual(body["mlflow"]["role"], "experiment_raw_records")
        self.assertIn("checked_at", body["mlflow"])
        mocked_urlopen.assert_called_once()

    def test_status_rechecks_available_local_mlflow_without_cache_reuse(self) -> None:
        from api.v1.endpoints.quant_jobs import _clear_mlflow_status_cache

        _clear_mlflow_status_cache()
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch(
                    "api.v1.endpoints.quant_jobs.urlopen",
                    side_effect=[_FakeHTTPResponse(200), _FakeHTTPResponse(503)],
                ) as mocked_urlopen,
            ):
                first_resp = client.get("/api/v1/quant/status")
                second_resp = client.get("/api/v1/quant/status")

        _clear_mlflow_status_cache()
        self.assertEqual(first_resp.status_code, 200)
        self.assertEqual(second_resp.status_code, 200)
        self.assertEqual(first_resp.json()["mlflow"]["status"], "available")
        self.assertEqual(second_resp.json()["mlflow"]["status"], "unavailable")
        self.assertIn("HTTP 503", second_resp.json()["mlflow"]["error"])
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_status_reuses_recent_unavailable_mlflow_health_check(self) -> None:
        from api.v1.endpoints.quant_jobs import _clear_mlflow_status_cache

        _clear_mlflow_status_cache()
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.quant_jobs.urlopen", return_value=_FakeHTTPResponse(503)) as mocked_urlopen,
            ):
                first_resp = client.get("/api/v1/quant/status")
                second_resp = client.get("/api/v1/quant/status")

        _clear_mlflow_status_cache()
        self.assertEqual(first_resp.status_code, 200)
        self.assertEqual(second_resp.status_code, 200)
        self.assertEqual(first_resp.json()["mlflow"]["status"], "unavailable")
        self.assertEqual(second_resp.json()["mlflow"]["status"], "unavailable")
        mocked_urlopen.assert_called_once()

    def test_status_refreshes_local_mlflow_health_check_after_cache_ttl(self) -> None:
        from api.v1.endpoints.quant_jobs import _clear_mlflow_status_cache, _mlflow_status_payload

        _clear_mlflow_status_cache()
        try:
            with (
                patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6006/"}, clear=False),
                patch("api.v1.endpoints.quant_jobs.time.monotonic", side_effect=[0.0, 0.0, 16.0, 16.0]),
                patch(
                    "api.v1.endpoints.quant_jobs.urlopen",
                    side_effect=[_FakeHTTPResponse(200), _FakeHTTPResponse(503)],
                ) as mocked_urlopen,
            ):
                first = _mlflow_status_payload()
                second = _mlflow_status_payload()
        finally:
            _clear_mlflow_status_cache()

        self.assertEqual(first["status"], "available")
        self.assertEqual(second["status"], "unavailable")
        self.assertIn("HTTP 503", second["error"])
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_status_does_not_overwrite_fresh_nonavailable_mlflow_cache_after_slow_probe(self) -> None:
        from api.v1.endpoints.quant_jobs import _MLFLOW_STATUS_CACHE, _clear_mlflow_status_cache, _mlflow_status_payload

        _clear_mlflow_status_cache()
        try:
            def slow_probe_writes_fresh_cache(_url: str) -> dict[str, str]:
                _MLFLOW_STATUS_CACHE["http://127.0.0.1:6006/"] = (
                    1.5,
                    {
                        "url": "http://127.0.0.1:6006/",
                        "status": "linked",
                        "role": "experiment_raw_records",
                        "checked_at": "2026-07-07T00:00:01+00:00",
                        "health_checked": False,
                    },
                )
                return {
                    "url": "http://127.0.0.1:6006/",
                    "status": "unavailable",
                    "role": "experiment_raw_records",
                    "checked_at": "2026-07-07T00:00:00+00:00",
                    "health_checked": True,
                }

            with (
                patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6006/"}, clear=False),
                patch("api.v1.endpoints.quant_jobs.time.monotonic", side_effect=[0.0, 2.0]),
                patch(
                    "api.v1.endpoints.quant_jobs._probe_mlflow_status_payload",
                    side_effect=slow_probe_writes_fresh_cache,
                ) as probe,
            ):
                _MLFLOW_STATUS_CACHE["http://127.0.0.1:6006/"] = (
                    -20.0,
                    {
                        "url": "http://127.0.0.1:6006/",
                        "status": "stale",
                        "role": "experiment_raw_records",
                        "checked_at": "2026-07-06T23:59:00+00:00",
                    },
                )
                payload = _mlflow_status_payload()
        finally:
            _clear_mlflow_status_cache()

        self.assertEqual(payload["status"], "linked")
        probe.assert_called_once()

    def test_mlflow_health_cache_evicts_oldest_entry_when_full(self) -> None:
        from api.v1.endpoints.quant_jobs import (
            MLFLOW_HEALTH_CACHE_MAX_ENTRIES,
            _MLFLOW_STATUS_CACHE,
            _clear_mlflow_status_cache,
            _mlflow_status_payload,
        )

        _clear_mlflow_status_cache()
        try:
            for index in range(MLFLOW_HEALTH_CACHE_MAX_ENTRIES):
                _MLFLOW_STATUS_CACHE[f"http://127.0.0.1:{6100 + index}/"] = (
                    float(index),
                    {
                        "url": f"http://127.0.0.1:{6100 + index}/",
                        "status": "available",
                        "role": "experiment_raw_records",
                        "checked_at": f"2026-07-07T00:00:{index:02d}+00:00",
                    },
                )

            with (
                patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6606/"}, clear=False),
                patch("api.v1.endpoints.quant_jobs.time.monotonic", side_effect=[100.0, 100.0]),
                patch("api.v1.endpoints.quant_jobs.urlopen", return_value=_FakeHTTPResponse(503)),
            ):
                payload = _mlflow_status_payload()

            self.assertEqual(payload["status"], "unavailable")
            self.assertEqual(len(_MLFLOW_STATUS_CACHE), MLFLOW_HEALTH_CACHE_MAX_ENTRIES)
            self.assertNotIn("http://127.0.0.1:6100/", _MLFLOW_STATUS_CACHE)
            self.assertIn("http://127.0.0.1:6606/", _MLFLOW_STATUS_CACHE)
        finally:
            _clear_mlflow_status_cache()

    def test_status_runs_mlflow_health_check_outside_event_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)
            to_thread = AsyncMock(
                return_value={
                    "url": "http://127.0.0.1:6006/",
                    "status": "available",
                    "role": "experiment_raw_records",
                    "checked_at": "2026-07-06T10:00:00+00:00",
                }
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.quant_jobs.asyncio.to_thread", to_thread),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        to_thread.assert_awaited_once()
        self.assertEqual(to_thread.await_args.args[0].__name__, "_mlflow_status_payload")
        self.assertEqual(resp.json()["mlflow"]["status"], "available")

    def test_status_reports_unavailable_local_mlflow_when_health_check_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.quant_jobs.urlopen", side_effect=URLError("connection refused")),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mlflow"]["status"], "unavailable")
        self.assertIn("connection refused", body["mlflow"]["error"])

    def test_status_redacts_mlflow_probe_error_absolute_paths(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_mlflow_status_payload

        with patch(
            "api.v1.endpoints.quant_jobs.urlopen",
            side_effect=URLError("failed opening /Users/cq/.secret/mlflow/password.txt"),
        ):
            payload = _probe_mlflow_status_payload("http://127.0.0.1:6006/")

        self.assertEqual(payload["status"], "unavailable")
        self.assertTrue(payload["health_checked"])
        self.assertIn("[path]", payload["error"])
        self.assertNotIn("/Users/cq", payload["error"])
        self.assertNotIn("password.txt", payload["error"])

    def test_status_reports_unavailable_local_mlflow_when_http_error_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch(
                    "api.v1.endpoints.quant_jobs.urlopen",
                    side_effect=HTTPError("http://127.0.0.1:6006/", 404, "Not Found", {}, None),
                ),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mlflow"]["status"], "unavailable")
        self.assertIn("HTTP 404", body["mlflow"]["error"])

    def test_status_skips_health_check_for_non_local_mlflow_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-05T15:30:00+08:00", "trainable": True}),
                encoding="utf-8",
            )
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs", command_specs={})
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch.dict(os.environ, {"MLFLOW_URL": "https://mlflow.example.com/"}, clear=False),
                patch("api.v1.endpoints.quant_jobs.urlopen") as mocked_urlopen,
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["mlflow"]["url"], "https://mlflow.example.com/")
        self.assertEqual(body["mlflow"]["status"], "linked")
        self.assertIn("skipped", body["mlflow"]["error"])
        mocked_urlopen.assert_not_called()

    def test_mlflow_health_check_never_probes_metadata_service_urls(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_mlflow_status_payload

        with patch("api.v1.endpoints.quant_jobs.urlopen") as mocked_urlopen:
            payload = _probe_mlflow_status_payload("http://169.254.169.254/latest/meta-data/")

        self.assertEqual(payload["url"], "http://169.254.169.254/latest/meta-data/")
        self.assertEqual(payload["status"], "linked")
        self.assertIn("skipped", payload["error"])
        mocked_urlopen.assert_not_called()

    def test_mlflow_health_check_rejects_unsafe_local_url_before_probe(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_mlflow_status_payload

        with patch("api.v1.endpoints.quant_jobs.urlopen") as mocked_urlopen:
            traversal = _probe_mlflow_status_payload("http://127.0.0.1:6006/%2e%2e/admin")
            userinfo = _probe_mlflow_status_payload("http://user:pass@127.0.0.1:6006/?token=secret#frag")
            bad_port = _probe_mlflow_status_payload("http://127.0.0.1:notaport/")
            disallowed_port = _probe_mlflow_status_payload("http://127.0.0.1:6379/")

        self.assertEqual(traversal["status"], "invalid")
        self.assertIn("path traversal", traversal["error"])
        self.assertEqual(userinfo["status"], "invalid")
        self.assertEqual(userinfo["url"], "http://127.0.0.1:6006/")
        self.assertIn("username or password", userinfo["error"])
        self.assertNotIn("user:pass", json.dumps(userinfo))
        self.assertNotIn("token=secret", json.dumps(userinfo))
        self.assertEqual(bad_port["status"], "invalid")
        self.assertIn("port is invalid", bad_port["error"])
        self.assertEqual(disallowed_port["status"], "invalid")
        self.assertIn("ports 6006 and 6606", disallowed_port["error"])
        mocked_urlopen.assert_not_called()

    def test_mlflow_health_check_strips_query_and_fragment_before_probe(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_mlflow_status_payload

        with patch("api.v1.endpoints.quant_jobs.urlopen", return_value=_FakeHTTPResponse(200)) as mocked_urlopen:
            payload = _probe_mlflow_status_payload("http://127.0.0.1:6006/?token=secret#frag")

        self.assertEqual(payload["status"], "available")
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:6006/")
        self.assertNotIn("token=secret", request.full_url)
        self.assertNotIn("#frag", request.full_url)

    def test_catdesk_health_check_reports_local_port_availability(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_catdesk_status_payload

        with patch("api.v1.endpoints.quant_jobs.socket.create_connection", return_value=_FakeSocket()) as mocked_socket:
            payload = _probe_catdesk_status_payload("http://127.0.0.1:8765/")

        self.assertEqual(payload["url"], "http://127.0.0.1:8765/")
        self.assertEqual(payload["status"], "available")
        self.assertEqual(payload["role"], "external_catalyst_dashboard")
        self.assertTrue(payload["health_checked"])
        mocked_socket.assert_called_once()

    def test_catdesk_health_check_skips_non_local_urls_and_rejects_unsafe_local_urls(self) -> None:
        from api.v1.endpoints.quant_jobs import _probe_catdesk_status_payload

        with patch("api.v1.endpoints.quant_jobs.socket.create_connection") as mocked_socket:
            nonlocal_payload = _probe_catdesk_status_payload("https://catdesk.example.com/")
            traversal = _probe_catdesk_status_payload("http://127.0.0.1:8765/%2e%2e/admin")
            userinfo = _probe_catdesk_status_payload("http://user:pass@127.0.0.1:8765/?token=secret")
            disallowed_port = _probe_catdesk_status_payload("http://127.0.0.1:6006/")

        self.assertEqual(nonlocal_payload["status"], "linked")
        self.assertIn("skipped", nonlocal_payload["error"])
        self.assertEqual(traversal["status"], "invalid")
        self.assertIn("path traversal", traversal["error"])
        self.assertEqual(userinfo["status"], "invalid")
        self.assertNotIn("user:pass", json.dumps(userinfo))
        self.assertNotIn("token=secret", json.dumps(userinfo))
        self.assertEqual(disallowed_port["status"], "invalid")
        self.assertIn("port 8765", disallowed_port["error"])
        mocked_socket.assert_not_called()

    def test_stream_limit_env_parser_uses_safe_default_and_minimum(self) -> None:
        from api.v1.endpoints.quant_jobs import _int_env

        with patch.dict(os.environ, {"MAX_QUANT_JOB_STREAMS": "12"}, clear=False):
            self.assertEqual(_int_env("MAX_QUANT_JOB_STREAMS", 8), 12)
        with patch.dict(os.environ, {"MAX_QUANT_JOB_STREAMS": "0"}, clear=False):
            self.assertEqual(_int_env("MAX_QUANT_JOB_STREAMS", 8), 1)
        with patch.dict(os.environ, {"MAX_QUANT_JOB_STREAMS": "not-a-number"}, clear=False):
            self.assertEqual(_int_env("MAX_QUANT_JOB_STREAMS", 8), 8)

    def test_mlflow_health_timeout_env_parser_uses_safe_default_and_minimum(self) -> None:
        from api.v1.endpoints.quant_jobs import _float_env

        with patch.dict(os.environ, {"MLFLOW_HEALTH_TIMEOUT_SECONDS": "3.5"}, clear=False):
            self.assertEqual(_float_env("MLFLOW_HEALTH_TIMEOUT_SECONDS", 2.0), 3.5)
        with patch.dict(os.environ, {"MLFLOW_HEALTH_TIMEOUT_SECONDS": "0"}, clear=False):
            self.assertEqual(_float_env("MLFLOW_HEALTH_TIMEOUT_SECONDS", 2.0), 0.1)
        with patch.dict(os.environ, {"MLFLOW_HEALTH_TIMEOUT_SECONDS": "not-a-number"}, clear=False):
            self.assertEqual(_float_env("MLFLOW_HEALTH_TIMEOUT_SECONDS", 2.0), 2.0)

    def test_public_error_sanitizer_keeps_api_paths_but_redacts_local_filesystem_paths(self) -> None:
        from api.v1.endpoints.quant_jobs import _sanitize_public_error

        message = (
            "GET /api/v1/quant/status failed after reading "
            "'/Users/cq/WorkingProjects/daily_stock_analysis/.env' and "
            "C:\\Users\\cq\\secrets\\.env; fallback /dev/null is harmless"
        )

        sanitized = _sanitize_public_error(message)

        self.assertIn("/api/v1/quant/status", sanitized)
        self.assertIn("/dev/null", sanitized)
        self.assertNotIn("/Users/cq", sanitized)
        self.assertNotIn("C:\\Users\\cq", sanitized)
        self.assertIn("'[path]'", sanitized)

    def test_status_mlflow_url_uses_default_when_environment_is_blank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T15:30:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            client = self._make_client(service)

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch.dict(os.environ, {"MLFLOW_URL": "   "}, clear=False),
            ):
                resp = client.get("/api/v1/quant/status")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["mlflow"]["url"], "http://127.0.0.1:6006/")

    def test_missing_job_uses_structured_not_found_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            client = self._make_client(service)

            resp = client.get("/api/v1/quant/jobs/qj_missing")

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"], "not_found")
        self.assertIn("qj_missing", resp.json()["message"])

    def test_stream_job_snapshot_runs_sync_db_read_in_worker_thread(self) -> None:
        from api.v1.endpoints.quant_jobs import _list_jobs_for_stream

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            job = QuantJob(
                job_id="qj_stream_thread",
                job_type="check_qlib_data",
                label="检查 Qlib 数据",
                status="completed",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=0,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                completed_at="2026-07-04T01:00:02+08:00",
                updated_at="2026-07-04T01:00:02+08:00",
                log_path=str(Path(temp_dir) / "jobs" / "qj_stream_thread" / "job.log"),
            )
            service._insert_job(job)
            calls = []

            async def fake_to_thread(fn, *args, **kwargs):
                calls.append((fn, args, kwargs))
                return fn(*args, **kwargs)

            with patch("api.v1.endpoints.quant_jobs.asyncio.to_thread", side_effect=fake_to_thread):
                jobs = asyncio.run(_list_jobs_for_stream(service))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], {"limit": 20})
        self.assertEqual(jobs[0].job_id, "qj_stream_thread")

    def test_stream_event_generator_stops_after_max_events(self) -> None:
        from api.v1.endpoints.quant_jobs import _quant_job_event_generator

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )

            async def collect_events() -> list[str]:
                events = []
                async for event in _quant_job_event_generator(
                    service,
                    max_job_events=2,
                    poll_interval_seconds=0,
                ):
                    events.append(event)
                return events

            events = asyncio.run(collect_events())

        self.assertEqual(len(events), 4)
        self.assertIn("event: connected", events[0])
        self.assertEqual(sum("event: jobs" in event for event in events), 2)
        self.assertIn("event: stream_end", events[-1])

    def test_stream_event_generator_omits_internal_command_payload(self) -> None:
        from api.v1.endpoints.quant_jobs import _quant_job_event_generator

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            service._insert_job(
                QuantJob(
                    job_id="qj_stream_public",
                    job_type="check_qlib_data",
                    label="检查 Qlib 数据",
                    status="completed",
                    params={},
                    command=[sys.executable, "-c", "print('internal')"],
                    cwd=temp_dir,
                    return_code=0,
                    created_at="2026-07-04T01:00:00+08:00",
                    updated_at="2026-07-04T01:00:01+08:00",
                    error=f"Failed to start job: cannot execute {temp_dir}/secret_runner.py",
                    log_path=str(Path(temp_dir) / "jobs" / "qj_stream_public" / "job.log"),
                )
            )

            async def collect_events() -> list[str]:
                events = []
                async for event in _quant_job_event_generator(
                    service,
                    max_job_events=1,
                    poll_interval_seconds=0,
                ):
                    events.append(event)
                return events

            events = asyncio.run(collect_events())

        payload = json.loads(events[1].split("data: ", 1)[1])
        self.assertEqual(payload["jobs"][0]["job_id"], "qj_stream_public")
        self.assertNotIn("command", payload["jobs"][0])
        self.assertNotIn("cwd", payload["jobs"][0])
        self.assertNotIn("log_path", payload["jobs"][0])
        self.assertNotIn(str(temp_dir), json.dumps(payload))
        self.assertIn("[path]", payload["jobs"][0]["error"])

    def test_stream_event_generator_sends_error_end_when_snapshot_fails(self) -> None:
        from api.v1.endpoints.quant_jobs import _quant_job_event_generator

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )

            async def collect_events() -> list[str]:
                events = []
                async for event in _quant_job_event_generator(
                    service,
                    max_job_events=1,
                    poll_interval_seconds=0,
                ):
                    events.append(event)
                return events

            with patch("api.v1.endpoints.quant_jobs._list_jobs_for_stream", side_effect=RuntimeError("db unavailable")):
                events = asyncio.run(collect_events())

        self.assertEqual(len(events), 2)
        self.assertIn("event: connected", events[0])
        self.assertIn("event: stream_end", events[1])
        payload = json.loads(events[1].split("data: ", 1)[1])
        self.assertEqual(payload["reason"], "error")
        self.assertEqual(payload["error"], "Unable to stream quant job updates.")
        self.assertNotIn("db unavailable", events[1])

    def test_limited_stream_generator_releases_slot_when_closed(self) -> None:
        from api.v1.endpoints import quant_jobs as quant_jobs_endpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )

            async def open_and_close_stream() -> str:
                quant_jobs_endpoint._QUANT_JOB_STREAM_ACTIVE = 0
                self.assertTrue(quant_jobs_endpoint._try_acquire_stream_slot())
                generator = quant_jobs_endpoint._limited_quant_job_event_generator(service)
                try:
                    return await generator.__anext__()
                finally:
                    await generator.aclose()

            event = asyncio.run(open_and_close_stream())

        self.assertIn("event: connected", event)
        self.assertEqual(quant_jobs_endpoint._QUANT_JOB_STREAM_ACTIVE, 0)

    def test_stream_response_disables_cache_and_proxy_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            from api.v1.endpoints.quant_jobs import _release_stream_slot, stream_quant_jobs

            with patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service):
                resp = asyncio.run(stream_quant_jobs())
                _release_stream_slot()

        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")
        self.assertEqual(resp.headers.get("connection"), "keep-alive")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")
        self.assertEqual(resp.headers.get("x-content-type-options"), "nosniff")

    def test_stream_response_releases_slot_when_construction_fails(self) -> None:
        from api.v1.endpoints import quant_jobs as quant_jobs_endpoint
        from api.v1.endpoints.quant_jobs import stream_quant_jobs

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            quant_jobs_endpoint._QUANT_JOB_STREAM_ACTIVE = 0

            with (
                patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service),
                patch("api.v1.endpoints.quant_jobs.StreamingResponse", side_effect=RuntimeError("stream boom")),
            ):
                with self.assertRaises(RuntimeError):
                    asyncio.run(stream_quant_jobs())

        self.assertEqual(quant_jobs_endpoint._QUANT_JOB_STREAM_ACTIVE, 0)

    def test_stream_response_rejects_when_connection_limit_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            from api.v1.endpoints.quant_jobs import stream_quant_jobs

            with (
                patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service),
                patch("api.v1.endpoints.quant_jobs.MAX_QUANT_JOB_STREAMS", 0),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(stream_quant_jobs())

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.detail["error"], "too_many_streams")

    def test_job_database_uses_wal_and_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            with service._connect() as conn:
                busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        self.assertEqual(busy_timeout, 5000)
        self.assertEqual(str(journal_mode).lower(), "wal")

    def test_service_restart_marks_stale_running_jobs_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            first_service = QuantJobService(
                jobs_root=jobs_root,
                command_specs={},
            )
            job = QuantJob(
                job_id="qj_stale",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=temp_dir,
                pid=999999,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(jobs_root / "qj_stale" / "job.log"),
            )
            first_service._insert_job(job)

            restarted = QuantJobService(
                jobs_root=jobs_root,
                command_specs={
                    "sleepy": {
                        "label": "慢任务",
                        "command": [sys.executable, "-c", "print('noop')"],
                    },
                },
            )
            recovered = restarted.get_job(job.job_id)

        self.assertEqual(recovered.status, "failed")
        self.assertIn("service restarted", recovered.error.lower())

    def test_service_restart_recovers_completed_durable_runner_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            first_service = QuantJobService(
                jobs_root=jobs_root,
                command_specs={},
            )
            job_dir = jobs_root / "qj_recovered"
            job_dir.mkdir(parents=True)
            job = QuantJob(
                job_id="qj_recovered",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable, "-c", "print('done')"],
                cwd=temp_dir,
                pid=999999,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(job_dir / "job.log"),
            )
            first_service._insert_job(job)
            (job_dir / "exit_status.json").write_text('{"return_code": 0}', encoding="utf-8")

            restarted = QuantJobService(
                jobs_root=jobs_root,
                command_specs={
                    "sleepy": {
                        "label": "慢任务",
                        "command": [sys.executable, "-c", "print('noop')"],
                    },
                },
            )
            recovered = restarted.get_job(job.job_id)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.return_code, 0)

    def test_service_restart_rechecks_exit_status_before_marking_stale_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            service = QuantJobService(
                jobs_root=jobs_root,
                command_specs={
                    "sleepy": {
                        "label": "慢任务",
                        "command": [sys.executable, "-c", "print('noop')"],
                    },
                },
            )
            job_dir = jobs_root / "qj_late_exit_status"
            job_dir.mkdir(parents=True)
            job = QuantJob(
                job_id="qj_late_exit_status",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable, "-c", "print('done')"],
                cwd=temp_dir,
                pid=999999,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(job_dir / "job.log"),
            )
            service._insert_job(job)

            original_read_exit_status = service._read_exit_status
            read_count = 0

            def read_exit_status_with_late_file(current_job: QuantJob):
                nonlocal read_count
                read_count += 1
                if read_count == 1:
                    (job_dir / "exit_status.json").write_text('{"return_code": 0}', encoding="utf-8")
                    return None
                return original_read_exit_status(current_job)

            with patch.object(service, "_read_exit_status", side_effect=read_exit_status_with_late_file):
                service._recover_running_jobs()

            recovered = service.get_job(job.job_id)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.return_code, 0)
        self.assertGreaterEqual(read_count, 2)

    def test_recovery_monitor_marks_alive_runner_failed_after_poll_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            service = QuantJobService(
                jobs_root=jobs_root,
                command_specs={},
            )
            job_dir = jobs_root / "qj_recovery_timeout"
            job_dir.mkdir(parents=True)
            job = QuantJob(
                job_id="qj_recovery_timeout",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable, "-c", "print('still alive')"],
                cwd=temp_dir,
                pid=os.getpid(),
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(job_dir / "job.log"),
            )
            service._insert_job(job)

            service._poll_recovered_process(
                job.job_id,
                os.getpid(),
                max_poll_seconds=0,
                poll_interval_seconds=0,
            )
            recovered = service.get_job(job.job_id)

        self.assertEqual(recovered.status, "failed")
        self.assertIn("recovery monitoring window", recovered.error or "")

    def test_latest_jobs_by_type_uses_latest_inserted_row_for_same_second_ties(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            service = QuantJobService(jobs_root=jobs_root, command_specs={})
            created_at = "2026-07-04T01:00:00+08:00"
            for job_id, label in (("qj_first", "first"), ("qj_second", "second")):
                service._insert_job(
                    QuantJob(
                        job_id=job_id,
                        job_type="check_qlib_data",
                        label=label,
                        status="completed",
                        params={},
                        command=[sys.executable],
                        cwd=temp_dir,
                        return_code=0,
                        created_at=created_at,
                        started_at=created_at,
                        completed_at=created_at,
                        updated_at=created_at,
                        log_path=str(jobs_root / job_id / "job.log"),
                    )
                )

            latest = service.get_latest_jobs_by_type()
            listed = service.list_jobs(limit=2, job_type="check_qlib_data")

        self.assertEqual(latest["check_qlib_data"].job_id, "qj_second")
        self.assertEqual([job.job_id for job in listed], ["qj_second", "qj_first"])

    def test_start_job_uses_durable_wrapper_process_group(self) -> None:
        class FakeProcess:
            pid = 424242

            def wait(self) -> int:
                return 0

            def poll(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "check_qlib_data": {
                        "label": "检查 Qlib 数据",
                        "command": [sys.executable, "-c", "print('qlib ok')"],
                    },
                },
            )
            popen_calls = []

            def fake_popen(*args, **kwargs):
                popen_calls.append((args, kwargs))
                return FakeProcess()

            with (
                patch("src.services.quant_job_service.subprocess.Popen", side_effect=fake_popen),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "must-not-leak",
                        "JQ_USERNAME": "jq-user",
                        "JQ_PASSWORD": "jq-password",
                        "TUSHARE_TOKEN": "tushare-token",
                        "AKSHARE_TOKEN": "akshare-token",
                        "JQDATA_CLIENT_ID": "jqdata-client",
                        "QUANT_DATA_ROOT": "/tmp/quant-data",
                        "QUANT_WORKFLOW_ROOT": "/tmp/quant-workflow",
                        "QUANT_API_KEY": "must-not-leak",
                        "QUANT_SECRET_TOKEN": "must-not-leak",
                    },
                    clear=False,
                ),
            ):
                job = service.submit_job("check_qlib_data")

            deadline = time.time() + 5
            recovered = service.get_job(job.job_id)
            while time.time() < deadline:
                recovered = service.get_job(job.job_id)
                if recovered.status == "completed" and popen_calls:
                    break
                time.sleep(0.05)

        self.assertEqual(len(popen_calls), 1)
        self.assertEqual(recovered.status, "completed")
        args, kwargs = popen_calls[0]
        self.assertEqual(args[0][0], sys.executable)
        self.assertEqual(args[0][1], "-c")
        self.assertTrue(kwargs["start_new_session"])
        self.assertIn("QUANT_JOB_COMMAND_JSON", kwargs["env"])
        self.assertIn("QUANT_JOB_EXIT_STATUS_PATH", kwargs["env"])
        self.assertEqual(kwargs["env"]["QUANT_DATA_ROOT"], "/tmp/quant-data")
        self.assertEqual(kwargs["env"]["QUANT_WORKFLOW_ROOT"], "/tmp/quant-workflow")
        self.assertNotIn("JQ_USERNAME", kwargs["env"])
        self.assertNotIn("JQ_PASSWORD", kwargs["env"])
        self.assertNotIn("TUSHARE_TOKEN", kwargs["env"])
        self.assertNotIn("AKSHARE_TOKEN", kwargs["env"])
        self.assertNotIn("JQDATA_CLIENT_ID", kwargs["env"])
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("QUANT_API_KEY", kwargs["env"])
        self.assertNotIn("QUANT_SECRET_TOKEN", kwargs["env"])

    def test_start_job_passes_jq_credentials_only_for_explicit_jqdata_job(self) -> None:
        class FakeProcess:
            pid = 424243

            def wait(self) -> int:
                return 0

            def poll(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "download_jqdata": {
                        "label": "验证 JQData 连接",
                        "command": [sys.executable, "-c", "print('jqdata ok')"],
                        "env_names": {"JQ_USERNAME", "JQ_PASSWORD"},
                    },
                },
            )
            popen_calls = []

            def fake_popen(*args, **kwargs):
                popen_calls.append((args, kwargs))
                return FakeProcess()

            with (
                patch("src.services.quant_job_service.subprocess.Popen", side_effect=fake_popen),
                patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "must-not-leak",
                        "JQ_USERNAME": "jq-user",
                        "JQ_PASSWORD": "jq-password",
                        "JQDATA_CLIENT_ID": "jqdata-client",
                        "TUSHARE_TOKEN": "tushare-token",
                        "AKSHARE_TOKEN": "akshare-token",
                    },
                    clear=False,
                ),
            ):
                job = service.submit_job("download_jqdata")

            deadline = time.time() + 5
            recovered = service.get_job(job.job_id)
            while time.time() < deadline:
                recovered = service.get_job(job.job_id)
                if recovered.status == "completed" and popen_calls:
                    break
                time.sleep(0.05)

        self.assertEqual(recovered.status, "completed")
        env = popen_calls[0][1]["env"]
        self.assertEqual(env["JQ_USERNAME"], "jq-user")
        self.assertEqual(env["JQ_PASSWORD"], "jq-password")
        self.assertNotIn("JQDATA_CLIENT_ID", env)
        self.assertNotIn("TUSHARE_TOKEN", env)
        self.assertNotIn("AKSHARE_TOKEN", env)
        self.assertNotIn("OPENAI_API_KEY", env)

    def test_quant_job_child_env_requires_explicit_data_credential_allowlist(self) -> None:
        from src.services.quant_job_service import _quant_job_child_env

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "must-not-leak",
                "ANTHROPIC_API_KEY": "must-not-leak",
                "QUANT_SECRET_TOKEN": "must-not-leak",
                "TUSHARE_TOKEN": "tushare-token",
                "AKSHARE_TOKEN": "akshare-token",
                "JQ_USERNAME": "jq-user",
                "JQ_PASSWORD": "jq-password",
                "QUANT_WORKFLOW_ROOT": "/tmp/quant-workflow",
                "BAOSTOCK_LOGIN_TIMEOUT": "30",
                "MLFLOW_URL": "http://fake_user:fake_pass@127.0.0.1:6006/?token=fake#frag",
                "MLFLOW_TRACKING_PASSWORD": "must-not-leak",
                "QLIB_PROVIDER_URI": "/tmp/qlib",
            },
            clear=False,
        ):
            env = _quant_job_child_env({"QUANT_JOB_COMMAND_JSON": "[]"})
            explicit_env = _quant_job_child_env(
                {"QUANT_JOB_COMMAND_JSON": "[]"},
                allowed_names={"JQ_USERNAME", "JQ_PASSWORD", "TUSHARE_TOKEN", "AKSHARE_TOKEN"},
            )

        self.assertNotIn("TUSHARE_TOKEN", env)
        self.assertNotIn("AKSHARE_TOKEN", env)
        self.assertNotIn("JQ_USERNAME", env)
        self.assertNotIn("JQ_PASSWORD", env)
        self.assertEqual(env["QUANT_WORKFLOW_ROOT"], "/tmp/quant-workflow")
        self.assertEqual(env["BAOSTOCK_LOGIN_TIMEOUT"], "30")
        self.assertEqual(env["QLIB_PROVIDER_URI"], "/tmp/qlib")
        self.assertEqual(env["QUANT_JOB_COMMAND_JSON"], "[]")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("QUANT_SECRET_TOKEN", env)
        self.assertNotIn("MLFLOW_URL", env)
        self.assertNotIn("MLFLOW_TRACKING_PASSWORD", env)
        self.assertEqual(explicit_env["TUSHARE_TOKEN"], "tushare-token")
        self.assertEqual(explicit_env["AKSHARE_TOKEN"], "akshare-token")
        self.assertEqual(explicit_env["JQ_USERNAME"], "jq-user")
        self.assertEqual(explicit_env["JQ_PASSWORD"], "jq-password")

    def test_waiter_does_not_overwrite_cancelled_job_status(self) -> None:
        class FinishedProcess:
            def wait(self) -> int:
                return -15

        class Handle:
            def close(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            job = QuantJob(
                job_id="qj_cancelled",
                job_type="sleepy",
                label="慢任务",
                status="cancelled",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=None,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                completed_at="2026-07-04T01:00:02+08:00",
                updated_at="2026-07-04T01:00:02+08:00",
                error="Cancelled by user from the web UI.",
                log_path=str(Path(temp_dir) / "jobs" / "qj_cancelled" / "job.log"),
            )
            service._insert_job(job)

            service._wait_for_process(job.job_id, FinishedProcess(), Handle())
            recovered = service.get_job(job.job_id)

        self.assertEqual(recovered.status, "cancelled")
        self.assertEqual(recovered.return_code, -15)
        self.assertEqual(recovered.error, "Cancelled by user from the web UI.")

    def test_cancel_job_does_not_overwrite_job_that_finished_during_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            job = QuantJob(
                job_id="qj_race",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                pid=12345,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(Path(temp_dir) / "jobs" / "qj_race" / "job.log"),
            )
            service._insert_job(job)

            class FinishedDuringCancelProcess:
                def __init__(self) -> None:
                    self.terminated = False

                def terminate(self) -> None:
                    self.terminated = True
                    service._update_job(
                        job.job_id,
                        status="completed",
                        return_code=0,
                        completed_at="2026-07-04T01:00:02+08:00",
                        updated_at="2026-07-04T01:00:02+08:00",
                    )

                def poll(self) -> int | None:
                    return 0 if self.terminated else None

            service._processes[job.job_id] = FinishedDuringCancelProcess()  # type: ignore[assignment]

            returned = service.cancel_job(job.job_id)
            recovered = service.get_job(job.job_id)

        self.assertEqual(returned.status, "completed")
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.return_code, 0)

    def test_cancel_job_rewrites_signal_failure_from_waiter_to_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            job = QuantJob(
                job_id="qj_cancel_race",
                job_type="sleepy",
                label="慢任务",
                status="running",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                updated_at="2026-07-04T01:00:01+08:00",
                log_path=str(Path(temp_dir) / "jobs" / "qj_cancel_race" / "job.log"),
            )
            service._insert_job(job)

            class WaiterWinsRaceProcess:
                pid = None

                def __init__(self) -> None:
                    self.return_code: int | None = None

                def terminate(self) -> None:
                    self.return_code = -signal.SIGTERM
                    service._complete_job_from_return_code(
                        job,
                        self.return_code,
                        "2026-07-04T01:00:02+08:00",
                    )

                def poll(self) -> int | None:
                    return self.return_code

            service._processes[job.job_id] = WaiterWinsRaceProcess()  # type: ignore[assignment]

            returned = service.cancel_job(job.job_id)
            recovered = service.get_job(job.job_id)

        self.assertEqual(returned.status, "cancelled")
        self.assertEqual(recovered.status, "cancelled")
        self.assertEqual(recovered.return_code, -signal.SIGTERM)
        self.assertEqual(recovered.error, "Cancelled by user from the web UI.")

    def test_return_code_cancel_mapping_covers_signal_conventions(self) -> None:
        from src.services.quant_job_service import _return_code_indicates_cancel

        cancel_codes = {
            -signal.SIGTERM,
            -signal.SIGKILL,
            128 + signal.SIGTERM,
            128 + signal.SIGKILL,
        }

        for return_code in cancel_codes:
            with self.subTest(return_code=return_code):
                self.assertTrue(_return_code_indicates_cancel(return_code))

        for return_code in (None, 0, 1, 128 + signal.SIGINT):
            with self.subTest(return_code=return_code):
                self.assertFalse(_return_code_indicates_cancel(return_code))

    def test_start_job_closes_log_handle_when_popen_fails(self) -> None:
        class FakeHandle:
            def __init__(self) -> None:
                self.closed = False

            def write(self, data: bytes) -> int:
                return len(data)

            def flush(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "bad_start": {
                        "label": "启动失败任务",
                        "command": [sys.executable, "-c", "print('never starts')"],
                    },
                },
            )
            handle = FakeHandle()

            with (
                patch("src.services.quant_job_service.Path.open", return_value=handle),
                patch("src.services.quant_job_service.subprocess.Popen", side_effect=RuntimeError("boom")),
            ):
                job = service.submit_job("bad_start")

            recovered = service.get_job(job.job_id)

        self.assertTrue(handle.closed)
        self.assertEqual(recovered.status, "failed")
        self.assertIn("Failed to start job", recovered.error or "")

    def test_completed_job_collects_declared_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            artifact_path = reports_dir / "data_quality_report_latest.md"
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "report_data_quality": {
                        "label": "数据质量报告",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                f"p = Path({str(artifact_path)!r}); "
                                "p.parent.mkdir(parents=True, exist_ok=True); "
                                "p.write_text('quality ok', encoding='utf-8')"
                            ),
                        ],
                        "artifact_patterns": [str(reports_dir / "data_quality_report_*.md")],
                    },
                },
            )
            job = service.submit_job("report_data_quality")

            deadline = time.time() + 5
            recovered = service.get_job(job.job_id)
            while time.time() < deadline:
                recovered = service.get_job(job.job_id)
                if recovered.status == "completed":
                    break
                time.sleep(0.05)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.artifact_paths, [str(artifact_path)])

    def test_artifact_download_rejects_symlink_that_resolves_outside_reports_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            outside_dir = Path(temp_dir) / "outside"
            outside_dir.mkdir()
            outside_file = outside_dir / "secret.md"
            outside_file.write_text("secret", encoding="utf-8")
            symlink_path = reports_dir / "linked_secret.md"
            try:
                symlink_path.symlink_to(outside_file)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not supported on this filesystem: {exc}")

            service = QuantJobService(jobs_root=reports_dir / "jobs", command_specs={})
            job = QuantJob(
                job_id="qj_symlink",
                job_type="report_data_quality",
                label="数据质量报告",
                status="completed",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=0,
                created_at="2026-07-07T00:00:00+08:00",
                started_at="2026-07-07T00:00:01+08:00",
                completed_at="2026-07-07T00:00:02+08:00",
                updated_at="2026-07-07T00:00:02+08:00",
                log_path=str(reports_dir / "jobs" / "qj_symlink" / "job.log"),
                artifact_paths=[str(symlink_path)],
            )
            service._insert_job(job)
            client = self._make_client(service)

            resp = client.get("/api/v1/quant/jobs/qj_symlink/artifacts/0")

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "forbidden")
        self.assertIn("outside the allowed reports directory", resp.json()["message"])

    def test_artifact_download_serves_resolved_path_after_symlink_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            real_artifact = reports_dir / "real_report.md"
            real_artifact.write_text("resolved artifact", encoding="utf-8")
            symlink_path = reports_dir / "linked_report.md"
            try:
                symlink_path.symlink_to(real_artifact)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not supported on this filesystem: {exc}")

            service = QuantJobService(jobs_root=reports_dir / "jobs", command_specs={})
            job = QuantJob(
                job_id="qj_allowed_symlink",
                job_type="report_data_quality",
                label="数据质量报告",
                status="completed",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=0,
                created_at="2026-07-07T00:00:00+08:00",
                started_at="2026-07-07T00:00:01+08:00",
                completed_at="2026-07-07T00:00:02+08:00",
                updated_at="2026-07-07T00:00:02+08:00",
                log_path=str(reports_dir / "jobs" / "qj_allowed_symlink" / "job.log"),
                artifact_paths=[str(symlink_path)],
            )
            service._insert_job(job)
            client = self._make_client(service)

            with patch("api.v1.endpoints.quant_jobs.FileResponse") as file_response:
                file_response.return_value = "ok"
                resp = client.get("/api/v1/quant/jobs/qj_allowed_symlink/artifacts/0")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Path(file_response.call_args.args[0]), real_artifact.resolve())
        self.assertEqual(file_response.call_args.kwargs["filename"], "linked_report.md")

    def test_default_specs_include_read_only_qlib_prediction_summary_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            available = service.available_job_types()
            command = service._build_command(
                "report_qlib_predictions",
                {"run_id": "safe_run_20260705", "topk": "15"},
            )

        self.assertIn("report_qlib_predictions", available)
        self.assertIn("topk", available["report_qlib_predictions"]["allowed_params"])
        self.assertIn("run_id", available["report_qlib_predictions"]["allowed_params"])
        self.assertIn("report_qlib_predictions.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertIn("--run-id", command)
        self.assertIn("safe_run_20260705", command)
        self.assertIn("--topk", command)
        self.assertIn("15", command)

    def test_report_and_refresh_jobs_use_configured_reports_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            refresh_command = service._build_command("prepare_baostock_refresh", {"limit": 3})
            anomaly_command = service._build_command("report_data_anomalies", {})

        self.assertIn("prepare_baostock_refresh.py", refresh_command[1])
        self._assert_reports_dir_arg(refresh_command)
        self.assertIn("--limit", refresh_command)
        self.assertIn("3", refresh_command)
        self.assertIn("report_data_anomalies.py", anomaly_command[1])
        self._assert_reports_dir_arg(anomaly_command)

    def test_qlib_prediction_summary_job_rejects_unsafe_run_id_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            with self.assertRaises(ValueError) as unsafe_ctx:
                service._build_command("report_qlib_predictions", {"run_id": "../unsafe", "topk": "10"})
            with self.assertRaises(ValueError) as long_ctx:
                service._build_command("report_qlib_predictions", {"run_id": "r" * 257, "topk": "10"})

        self.assertIn("run_id may only contain", str(unsafe_ctx.exception))
        self.assertIn("run_id must be <= 256", str(long_ctx.exception))

    def test_qlib_prediction_summary_job_rejects_invalid_topk_before_launch(self) -> None:
        invalid_values = [
            ("0", "Param topk must be >= 1."),
            ("501", "Param topk must be <= 500."),
            ("1.5", "Param topk must be an integer."),
            ("abc", "Param topk must be an integer."),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            for topk, expected_message in invalid_values:
                with self.subTest(topk=topk):
                    with self.assertRaises(ValueError) as ctx:
                        service._build_command("report_qlib_predictions", {"topk": topk})
                    self.assertIn(expected_message, str(ctx.exception))

    def test_completed_job_artifact_can_be_downloaded_by_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            artifact_path = reports_dir / "data_quality_report_latest.md"
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "report_data_quality": {
                        "label": "数据质量报告",
                        "command": [
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                f"p = Path({str(artifact_path)!r}); "
                                "p.parent.mkdir(parents=True, exist_ok=True); "
                                "p.write_text('quality ok', encoding='utf-8')"
                            ),
                        ],
                        "artifact_patterns": [str(reports_dir / "data_quality_report_*.md")],
                    },
                },
            )
            client = self._make_client(service)
            submit_resp = client.post("/api/v1/quant/jobs", json={"job_type": "report_data_quality"})
            self.assertEqual(submit_resp.status_code, 200)
            job_id = submit_resp.json()["job_id"]

            deadline = time.time() + 5
            detail = {}
            while time.time() < deadline:
                detail = client.get(f"/api/v1/quant/jobs/{job_id}").json()
                if detail["status"] == "completed":
                    break
                time.sleep(0.05)

            artifact_resp = client.get(f"/api/v1/quant/jobs/{job_id}/artifacts/0")
            missing_resp = client.get(f"/api/v1/quant/jobs/{job_id}/artifacts/1")

        self.assertEqual(detail["status"], "completed")
        self.assertEqual(artifact_resp.status_code, 200)
        self.assertEqual(artifact_resp.text, "quality ok")
        self.assertIn("inline", artifact_resp.headers.get("content-disposition", ""))
        self.assertEqual(missing_resp.status_code, 404)
        self.assertEqual(missing_resp.json()["error"], "not_found")

    def test_completed_job_artifact_must_stay_under_reports_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            service = QuantJobService(
                jobs_root=base / "safe" / "jobs",
                command_specs={},
            )
            outside_artifact = base / "outside_secret.txt"
            outside_artifact.write_text("secret", encoding="utf-8")
            job = QuantJob(
                job_id="qj_artifact_escape",
                job_type="report_data_quality",
                label="数据质量报告",
                status="completed",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=0,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                completed_at="2026-07-04T01:00:02+08:00",
                updated_at="2026-07-04T01:00:02+08:00",
                log_path=str(service.jobs_root / "qj_artifact_escape" / "job.log"),
                artifact_paths=[str(outside_artifact)],
            )
            service._insert_job(job)
            client = self._make_client(service)

            artifact_resp = client.get("/api/v1/quant/jobs/qj_artifact_escape/artifacts/0")

        self.assertEqual(artifact_resp.status_code, 403)
        self.assertEqual(artifact_resp.json()["error"], "forbidden")
        self.assertIn("outside the allowed reports directory", artifact_resp.json()["message"])

    def test_artifact_download_returns_not_found_when_file_disappears_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            service = QuantJobService(
                jobs_root=base / "reports" / "jobs",
                command_specs={},
            )
            artifact_path = base / "reports" / "race.md"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("race", encoding="utf-8")
            job = QuantJob(
                job_id="qj_artifact_race",
                job_type="report_data_quality",
                label="数据质量报告",
                status="completed",
                params={},
                command=[sys.executable],
                cwd=temp_dir,
                return_code=0,
                created_at="2026-07-04T01:00:00+08:00",
                started_at="2026-07-04T01:00:01+08:00",
                completed_at="2026-07-04T01:00:02+08:00",
                updated_at="2026-07-04T01:00:02+08:00",
                log_path=str(service.jobs_root / "qj_artifact_race" / "job.log"),
                artifact_paths=[str(artifact_path)],
            )
            service._insert_job(job)
            client = self._make_client(service)

            with patch("api.v1.endpoints.quant_jobs.FileResponse", side_effect=RuntimeError("file disappeared")):
                artifact_resp = client.get("/api/v1/quant/jobs/qj_artifact_race/artifacts/0")

        self.assertEqual(artifact_resp.status_code, 404)
        self.assertEqual(artifact_resp.json()["error"], "not_found")
        self.assertIn("no longer available", artifact_resp.json()["message"])

    def test_default_command_paths_can_be_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dsa_root = base / "dsa"
            projects_root = base / "workspace"
            workflow_root = base / "workflow"
            reports_root = base / "isolated_reports"
            workflow_python = base / "workflow-python"
            workflow_root.mkdir(parents=True)
            reports_root.mkdir(parents=True)
            workflow_python.write_text("#!/bin/sh\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": str(projects_root),
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                    "QUANT_WORKFLOW_REPORTS_ROOT": str(reports_root),
                    "QUANT_WORKFLOW_PYTHON": str(workflow_python),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "import src.services.quant_job_service as q; "
                        "specs = q._build_default_command_specs(); "
                        "print(json.dumps({"
                        "'dsa_root': str(q.DSA_ROOT), "
                        "'projects_root': str(q.PROJECTS_ROOT), "
                        "'reports_root': str(q.REPORTS_ROOT), "
                        "'jobs_root': str(q.DEFAULT_JOBS_ROOT), "
                        "'check_command': specs['check_qlib_data']['command'], "
                        "'quality_command': specs['report_data_quality']['command']({}), "
                        "'train_command': specs['train_baseline_lgb']['command']({'config': 'custom.yaml'}), "
                        "'dsa_backtest_command': specs['run_dsa_backtest']['command']({})"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["dsa_root"], str(dsa_root))
        self.assertEqual(payload["projects_root"], str(projects_root))
        self.assertEqual(payload["reports_root"], str(reports_root))
        self.assertEqual(payload["jobs_root"], str(reports_root / "jobs"))
        self.assertEqual(payload["check_command"][0], str(workflow_python))
        self.assertEqual(payload["dsa_backtest_command"][0], str(workflow_python))
        self.assertEqual(payload["check_command"][1], str(workflow_root / "scripts" / "check_qlib_data.py"))
        self.assertEqual(payload["quality_command"][payload["quality_command"].index("--out-dir") + 1], str(reports_root))
        self.assertEqual(payload["train_command"][3], str((workflow_root / "configs" / "custom.yaml").resolve()))

    def test_default_quant_workflow_root_is_embedded_under_dsa_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dsa_root = Path(temp_dir) / "daily_stock_analysis"
            dsa_root.mkdir(parents=True)

            env = os.environ.copy()
            env["DSA_ROOT"] = str(dsa_root)
            env.pop("WORKING_PROJECTS_ROOT", None)
            env.pop("QUANT_WORKFLOW_ROOT", None)
            env.pop("QUANT_WORKFLOW_REPORTS_ROOT", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "import src.services.quant_job_service as q; "
                        "print(json.dumps({"
                        "'workflow_root': str(q.QUANT_WORKFLOW_ROOT), "
                        "'reports_root': str(q.REPORTS_ROOT), "
                        "'default_cwd': str(q.QuantJobService().default_cwd)"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["workflow_root"], str(dsa_root / "quant_workflow"))
        self.assertEqual(payload["reports_root"], str(dsa_root / "quant_workflow" / "reports"))
        self.assertEqual(payload["default_cwd"], str(dsa_root))

    def test_missing_quant_workflow_python_override_warns_before_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            workflow_root = base / "workflow"
            missing_python = base / "missing-python"
            workflow_root.mkdir(parents=True)

            env = os.environ.copy()
            env.update(
                {
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                    "QUANT_WORKFLOW_PYTHON": str(missing_python),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, logging, sys; "
                        "logging.basicConfig(level=logging.WARNING); "
                        "import src.services.quant_job_service as q; "
                        "specs = q._build_default_command_specs(); "
                        "print(json.dumps({"
                        "'expected': sys.executable, "
                        "'check_command': specs['check_qlib_data']['command']"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["check_command"][0], payload["expected"])
        self.assertIn("QUANT_WORKFLOW_PYTHON points to a missing file", result.stderr)

    def test_default_quant_commands_use_dsa_venv_python_even_when_qlib_venv_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dsa_root = base / "daily_stock_analysis"
            projects_root = base / "workspace"
            workflow_root = projects_root / "quant_workflow"
            dsa_python = dsa_root / ".venv" / "bin" / "python"
            qlib_python = projects_root / "qlib" / ".venv" / "bin" / "python"
            workflow_root.mkdir(parents=True)
            dsa_python.parent.mkdir(parents=True)
            qlib_python.parent.mkdir(parents=True)
            dsa_python.write_text("#!/bin/sh\n", encoding="utf-8")
            qlib_python.write_text("#!/bin/sh\n", encoding="utf-8")

            env = os.environ.copy()
            env.pop("QLIB_PYTHON", None)
            env.update(
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": str(projects_root),
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, logging, sys; "
                        "logging.basicConfig(level=logging.WARNING); "
                        "import src.services.quant_job_service as q; "
                        "specs = q._build_default_command_specs(); "
                        "print(json.dumps({"
                        "'expected': str(q.DSA_ROOT / '.venv' / 'bin' / 'python'), "
                        "'check_command': specs['check_qlib_data']['command'], "
                        "'train_command': specs['train_baseline_lgb']['command']({'config': 'custom.yaml'})"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["check_command"][0], payload["expected"])
        self.assertEqual(payload["train_command"][0], payload["expected"])
        self.assertNotEqual(payload["check_command"][0], str(qlib_python))

    def test_legacy_qlib_python_env_does_not_override_shared_backend_python(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dsa_root = base / "daily_stock_analysis"
            workflow_root = base / "quant_workflow"
            dsa_python = dsa_root / ".venv" / "bin" / "python"
            legacy_qlib_python = base / "qlib" / ".venv" / "bin" / "python"
            workflow_root.mkdir(parents=True)
            dsa_python.parent.mkdir(parents=True)
            legacy_qlib_python.parent.mkdir(parents=True)
            dsa_python.write_text("#!/bin/sh\n", encoding="utf-8")
            legacy_qlib_python.write_text("#!/bin/sh\n", encoding="utf-8")

            env = os.environ.copy()
            env.pop("QUANT_WORKFLOW_PYTHON", None)
            env["DSA_ROOT"] = str(dsa_root)
            env["QUANT_WORKFLOW_ROOT"] = str(workflow_root)
            env["QLIB_PYTHON"] = str(legacy_qlib_python)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, logging, sys; "
                        "logging.basicConfig(level=logging.WARNING); "
                        "import src.services.quant_job_service as q; "
                        "specs = q._build_default_command_specs(); "
                        "print(json.dumps({"
                        "'expected': str(q.DSA_ROOT / '.venv' / 'bin' / 'python'), "
                        "'check_command': specs['check_qlib_data']['command'], "
                        "'train_command': specs['train_baseline_lgb']['command']({'config': 'custom.yaml'})"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["check_command"][0], payload["expected"])
        self.assertEqual(payload["train_command"][0], payload["expected"])
        self.assertNotEqual(payload["check_command"][0], str(legacy_qlib_python))
        self.assertIn("QLIB_PYTHON is ignored", result.stderr)

    def test_train_config_param_must_stay_under_configs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            with self.assertRaises(ValueError) as ctx:
                service._build_command("train_baseline_lgb", {"config": "../secret.yaml"})
            with self.assertRaises(ValueError) as extension_ctx:
                service._build_command("train_baseline_lgb", {"config": "baseline_lgb_alpha158.py"})
            with self.assertRaises(ValueError) as encoded_ctx:
                service._build_command("train_baseline_lgb", {"config": "%2e%2e%2fsecret.yaml"})
            with self.assertRaises(ValueError) as double_encoded_ctx:
                service._build_command("train_baseline_lgb", {"config": "%252e%252e%252fsecret.yaml"})

        self.assertIn("Config must be a file name", str(ctx.exception))
        self.assertIn("Config must be a .yaml or .yml file", str(extension_ctx.exception))
        self.assertIn("Config must be a file name", str(encoded_ctx.exception))
        self.assertIn("Config must be a file name", str(double_encoded_ctx.exception))

    def test_train_baseline_can_select_cleaned_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "train_baseline_lgb",
                {"config": "baseline_lgb_alpha158.yaml", "provider": "cleaned"},
            )
            status = service.available_job_types()

        self.assertIn("--provider", command)
        self.assertIn("cleaned", command)
        self.assertIn("provider", status["train_baseline_lgb"]["allowed_params"])

    def test_train_provider_param_is_restricted_to_known_local_providers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            with self.assertRaises(ValueError) as ctx:
                service._build_command("train_baseline_lgb", {"provider": "/tmp/other-provider"})

        self.assertIn("Provider must be one of", str(ctx.exception))

    def test_dsa_backtest_command_includes_stock_code_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "run_dsa_backtest",
                {"code": "600519", "eval_window_days": 1, "force": True},
            )

        self.assertEqual(command.count("--backtest-code"), 1)
        self.assertIn("600519", command)
        self.assertEqual(command.count("--backtest-days"), 1)
        self.assertIn("--backtest-force", command)

    def test_apply_data_cleaning_job_is_whitelisted_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("apply_data_cleaning", {})
            status = service.available_job_types()
            spec = service.command_specs["apply_data_cleaning"]

        self.assertIn("apply_data_cleaning.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertEqual(status["apply_data_cleaning"]["label"], "应用清洗策略生成 cleaned Qlib")
        self.assertEqual(status["apply_data_cleaning"]["allowed_params"], ["dry_run", "no_overwrite", "skip_qlib_convert"])
        self.assertIn("--dry-run", service._build_command("apply_data_cleaning", {"dry_run": True}))
        self.assertTrue(any("data_cleaning_applied_latest.json" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("data_cleaning_applied_latest.md" in item for item in spec["artifact_patterns"]))

    def test_merge_to_qlib_job_supports_isolated_smoke_fixture_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("merge_to_qlib", {"smoke": True})
            status = service.available_job_types()

            with self.assertRaises(ValueError) as ctx:
                service._build_command("merge_to_qlib", {"qlib_dir": "/tmp/unsafe"})

        self.assertIn("merge_to_qlib.py", command[1])
        self.assertIn("--smoke-fixture", command)
        self.assertIn("--smoke-dir", command)
        self.assertTrue(any("reports/smoke/merge_to_qlib" in item for item in command))
        self.assertEqual(status["merge_to_qlib"]["allowed_params"], ["no_overwrite", "smoke"])
        self.assertIn("Unsupported params for merge_to_qlib", str(ctx.exception))

    def test_download_jqdata_job_supports_browser_safe_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "download_jqdata",
                {"universe": "csi300", "limit": 1, "sleep": 0, "dry_run": True},
            )
            status = service.available_job_types()

        self.assertIn("download_jqdata.py", command[1])
        self.assertEqual(status["download_jqdata"]["label"], "验证 JQData 连接（dry-run）")
        self.assertIn("--dry-run", command)
        self.assertIn("--limit", command)
        self.assertIn("1", command)
        self.assertEqual(status["download_jqdata"]["allowed_params"], ["dry_run", "end", "limit", "sleep", "start", "universe"])

    def test_download_jqdata_web_job_defaults_to_small_dry_run_and_rejects_real_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("download_jqdata", {"universe": "csi300"})

            with self.assertRaises(ValueError) as real_download_ctx:
                service._build_command("download_jqdata", {"dry_run": False})
            with self.assertRaises(ValueError) as too_many_ctx:
                service._build_command("download_jqdata", {"limit": 21, "dry_run": True})

        self.assertIn("--dry-run", command)
        self.assertEqual(command[command.index("--limit") + 1], "3")
        self.assertIn("only support dry_run=true", str(real_download_ctx.exception))
        self.assertIn("Param limit must be <= 20", str(too_many_ctx.exception))

    def test_public_ashare_source_jobs_are_whitelisted_with_safe_params(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            eastmoney = service._build_command(
                "download_eastmoney",
                {
                    "start": "2026-07-01",
                    "end": "2026-07-08",
                    "limit": 2,
                    "incremental": True,
                    "overlap_days": 7,
                    "use_system_proxy": True,
                },
            )
            akshare = service._build_command(
                "download_akshare",
                {"start": "2026-07-01", "end": "2026-07-08", "limit": 2, "incremental": True, "overlap_days": 7},
            )
            snapshot = service._build_command("validate_latest_snapshot", {"limit": 3, "timeout": 9})
            global_daily = service._build_command(
                "download_global_daily",
                {"start": "2026-07-01", "end": "2026-07-08", "limit": 2, "incremental": True},
            )
            news = service._build_command("download_ashare_news", {"limit": 5, "timeout": 9})
            status = service.available_job_types()

            with self.assertRaises(ValueError) as unsafe_ctx:
                service._build_command("download_eastmoney", {"output_dir": "/tmp/unsafe"})

        self.assertIn("download_eastmoney.py", eastmoney[1])
        self.assertIn("--incremental", eastmoney)
        self.assertIn("--overlap-days", eastmoney)
        self.assertIn("--use-system-proxy", eastmoney)
        self.assertIn("download_akshare.py", akshare[1])
        self.assertIn("--incremental", akshare)
        self.assertIn("validate_latest_snapshot.py", snapshot[1])
        self.assertIn("--out-dir", snapshot)
        self.assertIn("download_global_daily.py", global_daily[1])
        self.assertIn("--incremental", global_daily)
        self.assertIn("download_ashare_news.py", news[1])
        self.assertEqual(news[news.index("--limit") + 1], "5")
        self.assertEqual(
            status["download_eastmoney"]["allowed_params"],
            [
                "end",
                "incremental",
                "limit",
                "max_consecutive_failures",
                "overlap_days",
                "retries",
                "retry_failed",
                "sleep",
                "start",
                "timeout",
                "use_system_proxy",
            ],
        )
        self.assertEqual(
            status["download_akshare"]["allowed_params"],
            [
                "end",
                "incremental",
                "limit",
                "max_consecutive_failures",
                "overlap_days",
                "retries",
                "retry_failed",
                "sleep",
                "start",
                "timeout",
                "use_system_proxy",
            ],
        )
        self.assertEqual(status["validate_latest_snapshot"]["allowed_params"], ["limit", "timeout", "use_system_proxy"])
        self.assertEqual(
            status["download_global_daily"]["allowed_params"],
            ["end", "incremental", "limit", "overlap_days", "retries", "sleep", "start", "timeout", "use_system_proxy"],
        )
        self.assertEqual(
            status["download_ashare_news"]["allowed_params"],
            ["limit", "retries", "sleep", "timeout", "use_system_proxy"],
        )
        self.assertIn("Unsupported params for download_eastmoney", str(unsafe_ctx.exception))

    def test_market_regime_report_job_is_whitelisted_with_safe_params(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("report_market_regime", {"lookback_days": 120, "min_symbols": 50, "limit": 200})
            status = service.available_job_types()
            spec = service.command_specs["report_market_regime"]

        self.assertIn("report_market_regime.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertIn("--lookback-days", command)
        self.assertIn("120", command)
        self.assertIn("--min-symbols", command)
        self.assertIn("50", command)
        self.assertIn("--limit", command)
        self.assertIn("200", command)
        self.assertEqual(status["report_market_regime"]["label"], "生成市场状态和现金仓位信号")
        self.assertEqual(status["report_market_regime"]["allowed_params"], ["limit", "lookback_days", "min_symbols"])
        self.assertTrue(any("market_regime_latest.json" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("market_regime_latest.md" in item for item in spec["artifact_patterns"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            with self.assertRaises(ValueError) as ctx:
                service._build_command("report_market_regime", {"min_symbols": 0})
        self.assertIn("Param min_symbols must be >= 1", str(ctx.exception))

    def test_data_quality_report_job_supports_browser_safe_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "report_data_quality",
                {"overlap_sample": 5, "merged_sample": 5},
            )
            status = service.available_job_types()

        self.assertIn("report_data_quality.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertIn("--overlap-sample", command)
        self.assertIn("5", command)
        self.assertIn("--merged-sample", command)
        self.assertCountEqual(
            status["report_data_quality"]["allowed_params"],
            ["merged_sample", "overlap_sample"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            with self.assertRaises(ValueError) as zero_ctx:
                service._build_command("report_data_quality", {"merged_sample": 0})
            with self.assertRaises(ValueError) as text_ctx:
                service._build_command("report_data_quality", {"merged_sample": "abc"})
        self.assertIn("Param merged_sample must be >= 1", str(zero_ctx.exception))
        self.assertIn("Param merged_sample must be an integer", str(text_ctx.exception))

    def test_browser_numeric_params_have_upper_bounds_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            cases = [
                ("download_baostock", {"retries": 21}, "Param retries must be <="),
                (
                    "download_baostock",
                    {"max_consecutive_failures": 1001},
                    "Param max_consecutive_failures must be <=",
                ),
                ("report_data_quality", {"overlap_sample": 100001}, "Param overlap_sample must be <="),
                ("report_data_quality", {"merged_sample": 100001}, "Param merged_sample must be <="),
                ("run_dsa_backtest", {"code": "600519", "eval_window_days": 3651}, "Param eval_window_days must be <="),
                ("report_market_regime", {"lookback_days": 5001}, "Param lookback_days must be <="),
                ("report_market_regime", {"min_symbols": 10001}, "Param min_symbols must be <="),
                ("report_market_regime", {"limit": 10001}, "Param limit must be <="),
                ("report_cash_control_backtest", {"limit": 10001}, "Param limit must be <="),
                ("prepare_baostock_refresh", {"limit": 10001}, "Param limit must be <="),
                ("download_baostock", {"limit": 10001}, "Param limit must be <="),
                ("download_eastmoney", {"limit": 10001}, "Param limit must be <="),
                ("download_akshare", {"limit": 10001}, "Param limit must be <="),
                ("optimize_strategy", {"dry_run": True, "phase": 5}, "Param phase must be <="),
            ]

            for job_type, params, expected_message in cases:
                with self.subTest(job_type=job_type, params=params):
                    with self.assertRaises(ValueError) as ctx:
                        service._build_command(job_type, params)
                    self.assertIn(expected_message, str(ctx.exception))

    def test_cash_control_backtest_job_is_whitelisted_with_safe_params(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "report_cash_control_backtest",
                {"lookback_days": 80, "min_symbols": 2, "limit": 20},
            )
            status = service.available_job_types()
            spec = service.command_specs["report_cash_control_backtest"]

        self.assertIn("report_cash_control_backtest.py", command[1])
        self._assert_reports_dir_arg(command)
        self.assertIn("--lookback-days", command)
        self.assertIn("80", command)
        self.assertIn("--min-symbols", command)
        self.assertIn("2", command)
        self.assertIn("--limit", command)
        self.assertIn("20", command)
        self.assertEqual(status["report_cash_control_backtest"]["label"], "生成现金仓位叠加回测")
        self.assertEqual(status["report_cash_control_backtest"]["allowed_params"], ["limit", "lookback_days", "min_symbols"])
        self.assertTrue(any("cash_control_backtest_latest.json" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("cash_control_backtest_latest.md" in item for item in spec["artifact_patterns"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            with self.assertRaises(ValueError) as ctx:
                service._build_command("report_cash_control_backtest", {"portfolio_report": "/tmp/report.pkl"})
        self.assertIn("Unsupported params for report_cash_control_backtest", str(ctx.exception))

    def test_china_astock_comparison_job_is_whitelisted_as_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("run_china_astock_comparison", {})
            status = service.available_job_types()
            spec = service.command_specs["run_china_astock_comparison"]

        self.assertIn("run_china_astock_comparison.py", command[1])
        self.assertEqual(status["run_china_astock_comparison"]["label"], "生成 china-astock 策略对比")
        self.assertEqual(status["run_china_astock_comparison"]["allowed_params"], [])
        self.assertTrue(any("china_astock_strategy_comparison_latest.csv" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("china_astock_strategy_comparison_latest.md" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("china_astock_strategy_comparison_latest.json" in item for item in spec["artifact_patterns"]))
        with self.assertRaises(ValueError) as ctx:
            service._build_command("run_china_astock_comparison", {"unsafe": "value"})
        self.assertIn("Unsupported params for run_china_astock_comparison", str(ctx.exception))

    def test_china_astock_indicator_report_job_is_whitelisted_as_report_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command(
                "report_china_astock_indicators",
                {"symbols": "sz000001, sh600519", "lookback_days": 60, "limit": 2},
            )
            status = service.available_job_types()
            spec = service.command_specs["report_china_astock_indicators"]

        self.assertIn("report_china_astock_indicators.py", command[1])
        self._assert_reports_dir_arg(command, flag="--output-dir")
        self.assertIn("--symbols", command)
        self.assertIn("SZ000001,SH600519", command)
        self.assertIn("--lookback-days", command)
        self.assertIn("60", command)
        self.assertIn("--limit", command)
        self.assertIn("2", command)
        self.assertEqual(status["report_china_astock_indicators"]["label"], "生成 china-astock 技术指标报告")
        self.assertEqual(status["report_china_astock_indicators"]["allowed_params"], ["limit", "lookback_days", "symbols"])
        self.assertTrue(any("china_astock_indicator_report_latest.json" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("china_astock_indicator_report_latest.md" in item for item in spec["artifact_patterns"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            default_command = service._build_command("report_china_astock_indicators", {})
            with self.assertRaises(ValueError) as ctx:
                service._build_command("report_china_astock_indicators", {"lookback_days": 2})
            with self.assertRaises(ValueError) as unsafe_symbols_ctx:
                service._build_command("report_china_astock_indicators", {"symbols": "../unsafe"})
            with self.assertRaises(ValueError) as metachar_symbols_ctx:
                service._build_command("report_china_astock_indicators", {"symbols": "600519;rm"})
            with self.assertRaises(ValueError) as multiline_symbols_ctx:
                service._build_command("report_china_astock_indicators", {"symbols": "600519\n000001"})
            with self.assertRaises(ValueError) as embedded_space_symbols_ctx:
                service._build_command("report_china_astock_indicators", {"symbols": "600519 000001"})
            with self.assertRaises(ValueError) as max_lookback_ctx:
                service._build_command("report_china_astock_indicators", {"lookback_days": 2001})
            with self.assertRaises(ValueError) as max_limit_ctx:
                service._build_command("report_china_astock_indicators", {"limit": 501})
            with self.assertRaises(ValueError) as long_symbol_ctx:
                service._build_command("report_china_astock_indicators", {"symbols": "A" * 21})
            with self.assertRaises(ValueError) as too_many_symbols_ctx:
                service._build_command(
                    "report_china_astock_indicators",
                    {"symbols": ",".join(f"{idx:06d}" for idx in range(101))},
                )
        self.assertIn("--limit", default_command)
        self.assertEqual(default_command[default_command.index("--limit") + 1], "50")
        self.assertIn("lookback_days must be >= 20", str(ctx.exception))
        self.assertIn("symbols may only contain", str(unsafe_symbols_ctx.exception))
        self.assertIn("symbols may only contain", str(metachar_symbols_ctx.exception))
        self.assertIn("symbols may only contain", str(multiline_symbols_ctx.exception))
        self.assertIn("without embedded spaces", str(embedded_space_symbols_ctx.exception))
        self.assertIn("lookback_days must be <= 2000", str(max_lookback_ctx.exception))
        self.assertIn("Param limit must be <= 500", str(max_limit_ctx.exception))
        self.assertIn("Each symbol must be <= 20 characters", str(long_symbol_ctx.exception))
        self.assertIn("symbols may include at most 100 entries", str(too_many_symbols_ctx.exception))

    def test_china_astock_indicator_job_submission_builds_safe_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            client = self._make_client(service)

            with patch.object(service, "_start_job", return_value=None) as start_job:
                resp = client.post(
                    "/api/v1/quant/jobs",
                    json={
                        "job_type": "report_china_astock_indicators",
                        "params": {
                            "symbols": "sz000001, sh600519",
                            "lookback_days": 60,
                            "limit": 2,
                        },
                    },
                )

            body = resp.json()
            job = service.get_job(body["job_id"])

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(body["job_type"], "report_china_astock_indicators")
        self.assertNotIn("command", body)
        self.assertNotIn("cwd", body)
        self.assertNotIn("log_path", body)
        start_job.assert_called_once()
        self.assertIn("report_china_astock_indicators.py", job.command[1])
        self._assert_reports_dir_arg(job.command, flag="--output-dir")
        self.assertIn("--symbols", job.command)
        self.assertIn("SZ000001,SH600519", job.command)
        self.assertIn("--lookback-days", job.command)
        self.assertIn("60", job.command)
        self.assertIn("--limit", job.command)
        self.assertIn("2", job.command)

    def test_china_astock_indicator_job_submission_rejects_invalid_limit_before_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            client = self._make_client(service)

            with patch.object(service, "_start_job", return_value=None) as start_job:
                resp = client.post(
                    "/api/v1/quant/jobs",
                    json={
                        "job_type": "report_china_astock_indicators",
                        "params": {"limit": 501},
                    },
                )
                jobs = service.list_jobs()

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")
        self.assertIn("Param limit must be <= 500", resp.json()["message"])
        start_job.assert_not_called()
        self.assertEqual(jobs, [])

    def test_optimize_strategy_job_is_whitelisted_as_browser_safe_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("optimize_strategy", {"phase": 4, "topk": 30, "dry_run": True})
            status = service.available_job_types()
            spec = service.command_specs["optimize_strategy"]

        self.assertIn("optimize_strategy.py", command[1])
        self.assertIn("--dry-run", command)
        self.assertIn("--skip-training", command)
        self._assert_reports_dir_arg(command, "--output-dir")
        self.assertIn("--phase", command)
        self.assertIn("4", command)
        self.assertIn("--topk", command)
        self.assertIn("30", command)
        self.assertEqual(status["optimize_strategy"]["label"], "策略优化预演")
        self.assertEqual(status["optimize_strategy"]["allowed_params"], ["dry_run", "phase", "topk"])
        self.assertTrue(any("optimization_results_latest.json" in item for item in spec["artifact_patterns"]))
        self.assertTrue(any("optimization_results_latest.md" in item for item in spec["artifact_patterns"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            with self.assertRaises(ValueError) as real_ctx:
                service._build_command("optimize_strategy", {"dry_run": False})
            with self.assertRaises(ValueError) as phase_ctx:
                service._build_command("optimize_strategy", {"dry_run": True, "phase": 5})
            with self.assertRaises(ValueError) as topk_ctx:
                service._build_command("optimize_strategy", {"dry_run": True, "topk": 0})
        self.assertIn("only support dry_run=true", str(real_ctx.exception))
        self.assertIn("Param phase must be <= 4", str(phase_ctx.exception))
        self.assertIn("Param topk must be >= 1", str(topk_ctx.exception))

    def test_candidate_pipeline_job_uses_fixed_script_and_safe_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            real_command = service._build_command("run_candidate_pipeline", {"provider": "canonical"})
            dry_command = service._build_command(
                "run_candidate_pipeline",
                {"provider": "cleaned", "dry_run": True},
            )
            status = service.available_job_types()
            spec = service.command_specs["run_candidate_pipeline"]

        self.assertIn("run_candidate_pipeline.py", real_command[1])
        self._assert_reports_dir_arg(real_command, "--output-dir")
        self.assertIn("--provider", real_command)
        self.assertIn("canonical", real_command)
        self.assertNotIn("--dry-run", real_command)
        self.assertIn("cleaned", dry_command)
        self.assertIn("--dry-run", dry_command)
        self.assertEqual(status["run_candidate_pipeline"]["label"], "运行五模型完整量化流程")
        self.assertEqual(status["run_candidate_pipeline"]["allowed_params"], ["dry_run", "provider"])
        self.assertTrue(any("candidate_pipeline_latest.json" in item for item in spec["artifact_patterns"]))

        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")
            with self.assertRaises(ValueError) as provider_ctx:
                service._build_command("run_candidate_pipeline", {"provider": "/tmp/other-provider"})
            with self.assertRaises(ValueError) as unknown_ctx:
                service._build_command("run_candidate_pipeline", {"spec": "other.yaml"})
        self.assertIn("Provider must be one of", str(provider_ctx.exception))
        self.assertIn("Unsupported params", str(unknown_ctx.exception))

    def test_numeric_job_params_are_rejected_before_subprocess_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "download_baostock": {
                        "label": "下载 Baostock 长历史",
                        "command": [sys.executable, "-c", "print('should not run')"],
                        "allowed_params": {"sleep", "timeout"},
                    },
                },
            )
            client = self._make_client(service)

            resp = client.post("/api/v1/quant/jobs", json={"job_type": "download_baostock", "params": {"sleep": "abc"}})
            huge_sleep = client.post(
                "/api/v1/quant/jobs",
                json={"job_type": "download_baostock", "params": {"sleep": "61"}},
            )
            huge_timeout = client.post(
                "/api/v1/quant/jobs",
                json={"job_type": "download_baostock", "params": {"timeout": "86401"}},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")
        self.assertIn("Param sleep must be a number", resp.json()["message"])
        self.assertEqual(huge_sleep.status_code, 400)
        self.assertIn("Param sleep must be <= 60", huge_sleep.json()["message"])
        self.assertEqual(huge_timeout.status_code, 400)
        self.assertIn("Param timeout must be <= 86400", huge_timeout.json()["message"])


if __name__ == "__main__":
    unittest.main()
