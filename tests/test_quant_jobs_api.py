# -*- coding: utf-8 -*-
"""Tests for the local quant workflow job API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from src.services.quant_job_service import QuantJob, QuantJobService


class QuantJobsApiTestCase(unittest.TestCase):
    def _make_client(self, service: QuantJobService) -> TestClient:
        static_dir = service.jobs_root / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        patcher = patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service)
        self.addCleanup(patcher.stop)
        patcher.start()
        return TestClient(create_app(static_dir=static_dir))

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
        self.assertEqual(jobs_resp.status_code, 200)
        self.assertEqual(jobs_resp.json()["items"][0]["job_id"], job_id)
        self.assertEqual(log_resp.status_code, 200)
        self.assertIn("qlib ok", log_resp.json()["content"])
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
        self.assertEqual(status_body["active_count"], 0)

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

    def test_stream_response_disables_cache_and_proxy_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={},
            )
            from api.v1.endpoints.quant_jobs import stream_quant_jobs

            with patch("api.v1.endpoints.quant_jobs.get_quant_job_service", return_value=service):
                resp = asyncio.run(stream_quant_jobs())

        self.assertEqual(resp.media_type, "text/event-stream")
        self.assertEqual(resp.headers.get("cache-control"), "no-cache")
        self.assertEqual(resp.headers.get("connection"), "keep-alive")
        self.assertEqual(resp.headers.get("x-accel-buffering"), "no")

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

            with patch("src.services.quant_job_service.subprocess.Popen", side_effect=fake_popen):
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
            qlib_python = base / "qlib-python"
            workflow_root.mkdir(parents=True)
            qlib_python.write_text("#!/bin/sh\n", encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": str(projects_root),
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                    "QLIB_PYTHON": str(qlib_python),
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
                        "'jobs_root': str(q.DEFAULT_JOBS_ROOT), "
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

        self.assertEqual(payload["dsa_root"], str(dsa_root))
        self.assertEqual(payload["projects_root"], str(projects_root))
        self.assertEqual(payload["jobs_root"], str(workflow_root / "reports" / "jobs"))
        self.assertEqual(payload["check_command"][0], str(qlib_python))
        self.assertEqual(payload["check_command"][1], str(workflow_root / "scripts" / "check_qlib_data.py"))
        self.assertEqual(payload["train_command"][3], str(workflow_root / "configs" / "custom.yaml"))

    def test_train_config_param_must_stay_under_configs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            with self.assertRaises(ValueError) as ctx:
                service._build_command("train_baseline_lgb", {"config": "../secret.yaml"})

        self.assertIn("Config must be a file name", str(ctx.exception))

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

    def test_market_regime_report_job_is_whitelisted_with_safe_params(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(jobs_root=Path(temp_dir) / "jobs")

            command = service._build_command("report_market_regime", {"lookback_days": 120, "min_symbols": 50, "limit": 200})
            status = service.available_job_types()
            spec = service.command_specs["report_market_regime"]

        self.assertIn("report_market_regime.py", command[1])
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

    def test_numeric_job_params_are_rejected_before_subprocess_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = QuantJobService(
                jobs_root=Path(temp_dir) / "jobs",
                command_specs={
                    "download_baostock": {
                        "label": "下载 Baostock 长历史",
                        "command": [sys.executable, "-c", "print('should not run')"],
                        "allowed_params": {"sleep"},
                    },
                },
            )
            client = self._make_client(service)

            resp = client.post("/api/v1/quant/jobs", json={"job_type": "download_baostock", "params": {"sleep": "abc"}})

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")
        self.assertIn("Param sleep must be a number", resp.json()["message"])


if __name__ == "__main__":
    unittest.main()
