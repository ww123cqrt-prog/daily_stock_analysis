# -*- coding: utf-8 -*-
"""Tests for the research workspace status endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app


class ResearchStatusApiTestCase(unittest.TestCase):
    def test_status_reads_latest_quant_workflow_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            log_path = Path(temp_dir) / "experiment_log.md"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "Qlib provider has enough CSI300 coverage.",
                        },
                        "data_sources": [
                            {
                                "name": "akshare",
                                "files": 899,
                                "rows": 1000,
                                "date_range": "2005-01-04 ~ 2026-06-10",
                                "status": "partial",
                            }
                        ],
                        "qlib_provider": {
                            "path": "/tmp/qlib",
                            "calendar_days": 1316,
                            "calendar_range": "2020-01-02 ~ 2025-06-11",
                            "instruments": 5340,
                        },
                        "latest_training": {
                            "status": "completed",
                            "experiment_name": "baseline_lgb_alpha158",
                            "run_id": "abc123",
                            "metrics": {"IC": 0.01},
                        },
                    }
                ),
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：长历史 baseline 已跑通，但收益质量偏弱。\n"
                "- 下一步：补充风险控制和市场择时实验。\n",
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps(
                    {
                        "run_id": "abc123",
                        "experiment_name": "baseline_lgb_alpha158",
                        "status": "completed",
                        "model": "LGBModel",
                        "strategy": "TopkDropoutStrategy",
                        "dataset_range": "2020-01-02 ~ 2026-07-01",
                        "conclusion": "长历史 baseline 已跑通，但收益质量偏弱。",
                        "next_step": "补充风险控制和市场择时实验。",
                        "metrics": {
                            "1day.excess_return_with_cost.annualized_return": 0.033970674,
                            "1day.excess_return_with_cost.information_ratio": 0.261546,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "trainable")
        self.assertEqual(body["data_sources"][0]["name"], "akshare")
        self.assertEqual(body["latest_training"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["status"], "available")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertIn("长历史 baseline 已跑通", body["experiment_notebook"]["latest_note"])
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["runs"][0]["dataset_range"], "2020-01-02 ~ 2026-07-01")
        matrix_by_id = {item["id"]: item for item in body["capability_matrix"]}
        self.assertEqual(matrix_by_id["dsa_web_core"]["status"], "available")
        self.assertEqual(matrix_by_id["quant_workflow_pipeline"]["route"], "/workflow")
        self.assertEqual(matrix_by_id["qlib_integration"]["route"], "/backtest")
        self.assertEqual(matrix_by_id["china_astock_quant_reuse"]["status"], "partial")
        self.assertEqual(matrix_by_id["external_tools_tonghuashun"]["status"], "read_only")
        self.assertEqual(matrix_by_id["trading_execution"]["status"], "not_planned")
        self.assertIn("不自动下单", matrix_by_id["trading_execution"]["restrictions"])
        self.assertIn("不读取券商密码", matrix_by_id["trading_execution"]["restrictions"])
        inventory_by_id = {item["id"]: item for item in body["function_inventory"]}
        self.assertEqual(inventory_by_id["qlib_data_pipeline"]["route"], "/workflow")
        self.assertEqual(inventory_by_id["china_astock_reference"]["status"], "partial")
        self.assertEqual(inventory_by_id["quant_workflow_script_coverage"]["route"], "/research")
        self.assertIn("add_index_benchmark.py", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("generate_csi300.py", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("supervise_baostock.py", inventory_by_id["quant_workflow_script_coverage"]["gaps"])
        reuse_by_id = {item["id"]: item for item in body["china_astock_reuse"]}
        self.assertEqual(reuse_by_id["indicators"]["priority"], "P1")
        self.assertEqual(reuse_by_id["paper_trading"]["status"], "not_planned")
        self.assertIn("不自动下单", reuse_by_id["paper_trading"]["restrictions"])
        acceptance_by_id = {item["id"]: item for item in body["acceptance_checklist"]}
        self.assertEqual(acceptance_by_id["project_hygiene"]["status"], "partial")
        self.assertIn("无散落未跟踪源码", acceptance_by_id["project_hygiene"]["evidence"])
        self.assertEqual(acceptance_by_id["qlib_web_flow"]["route"], "/workflow")
        self.assertIn("MLflow", acceptance_by_id["qlib_web_flow"]["evidence"])
        self.assertEqual(acceptance_by_id["opencode_review"]["status"], "verified")
        self.assertIn("opencode", acceptance_by_id["opencode_review"]["evidence"])

    def test_status_prefers_latest_training_result_over_stale_status_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                        "latest_training": {
                            "status": "completed",
                            "experiment_name": "baseline_lgb_alpha158",
                            "run_id": "old-run",
                            "provider": "canonical",
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-04T13:51:55+08:00",
                        "status": "completed",
                        "experiment_name": "baseline_lgb_alpha158_cleaned",
                        "run_id": "new-cleaned-run",
                        "provider": "cleaned",
                        "provider_uri": "/tmp/cleaned",
                    }
                ),
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "new-cleaned-run")
        self.assertEqual(body["latest_training"]["provider"], "cleaned")

    def test_status_includes_latest_cash_control_backtest_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            cash_control_path = Path(temp_dir) / "cash_control_backtest_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            cash_control_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T09:00:00+08:00",
                        "mode": "read_only_cash_control_overlay",
                        "comparison": {
                            "baseline": {"cumulative_return": 0.20, "max_drawdown": -0.30},
                            "cash_control": {"cumulative_return": 0.12, "max_drawdown": -0.12},
                            "delta": {"cumulative_return": -0.08, "max_drawdown": 0.18},
                        },
                        "signal_summary": {"latest_regime": "risk_off", "latest_cash_ratio": 0.80},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["cash_control_backtest"]["mode"], "read_only_cash_control_overlay")
        self.assertEqual(body["cash_control_backtest"]["signal_summary"]["latest_regime"], "risk_off")
        self.assertEqual(body["cash_control_backtest"]["comparison"]["cash_control"]["max_drawdown"], -0.12)

    def test_status_includes_latest_qlib_prediction_summary_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            predictions_path = Path(temp_dir) / "qlib_predictions_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T10:00:00+08:00",
                        "mode": "read_only_qlib_prediction_summary",
                        "run_id": "run_1",
                        "latest_date": "2026-07-02",
                        "top": [{"instrument": "SZ000002", "score": 0.25}],
                        "bottom": [{"instrument": "SZ000001", "score": -0.40}],
                        "summary": {"latest_rows": 3},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["qlib_predictions"]["mode"], "read_only_qlib_prediction_summary")
        self.assertEqual(body["qlib_predictions"]["latest_date"], "2026-07-02")
        self.assertEqual(body["qlib_predictions"]["top"][0]["instrument"], "SZ000002")
        self.assertEqual(body["qlib_predictions"]["bottom"][0]["score"], -0.40)

    def test_status_returns_missing_payload_when_artifact_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            missing_path = Path(temp_dir) / "missing.json"

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", missing_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "missing")
        self.assertEqual(body["data_sources"], [])
        self.assertEqual(body["experiment_notebook"]["status"], "missing")
        self.assertEqual(body["experiment_notebook"]["runs"], [])
        self.assertIsNone(body["qlib_predictions"])
        self.assertIn("capability_matrix", body)
        self.assertIn("function_inventory", body)
        self.assertIn("china_astock_reuse", body)
        self.assertIn("acceptance_checklist", body)
        self.assertEqual(
            {item["id"] for item in body["capability_matrix"]},
            {
                "dsa_web_core",
                "quant_workflow_pipeline",
                "qlib_integration",
                "china_astock_quant_reuse",
                "external_tools_tonghuashun",
                "trading_execution",
            },
        )
        self.assertIn("quant_workflow_jobs", {item["id"] for item in body["function_inventory"]})
        self.assertIn("event_backtest", {item["id"] for item in body["china_astock_reuse"]})
        self.assertIn("self_service_docs", {item["id"] for item in body["acceptance_checklist"]})

    def test_status_includes_read_only_tonghuashun_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            app_path = Path(temp_dir) / "同花顺.app"
            app_path.mkdir()
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.research_status.TONGHUASHUN_CANDIDATE_PATHS", (app_path,)),
            ):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        tonghuashun = body["external_tools"]["tonghuashun"]
        self.assertTrue(tonghuashun["installed"])
        self.assertEqual(tonghuashun["status"], "available")
        self.assertEqual(tonghuashun["mode"], "read_only")
        self.assertEqual(tonghuashun["path"], str(app_path))
        self.assertIn("不自动下单", tonghuashun["restrictions"])
        self.assertEqual(body["external_tools"]["iphone_mirroring"]["status"], "deferred")
        matrix_by_id = {item["id"]: item for item in body["capability_matrix"]}
        self.assertEqual(matrix_by_id["external_tools_tonghuashun"]["status"], "read_only")
        self.assertIn("本机应用存在检测", matrix_by_id["external_tools_tonghuashun"]["browser"])
        self.assertIn("不连接券商交易接口", matrix_by_id["external_tools_tonghuashun"]["restrictions"])

    def test_capability_matrix_ignores_non_string_external_tool_restrictions(self) -> None:
        from api.v1.endpoints.research_status import _capability_matrix_payload

        matrix = _capability_matrix_payload(
            {
                "tonghuashun": {
                    "status": "available",
                    "installed": True,
                    "summary": "detected",
                    "capabilities": ["本机应用存在检测", {"unexpected": "object"}],
                    "restrictions": ["不自动下单", {"unexpected": "object"}, ""],
                }
            }
        )

        tonghuashun = {item["id"]: item for item in matrix}["external_tools_tonghuashun"]
        self.assertEqual(tonghuashun["browser"], "detected 能力：本机应用存在检测")
        self.assertEqual(tonghuashun["restrictions"], ["不自动下单"])

    def test_archive_experiment_run_rewrites_active_log_and_archives_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            deleted_runs_path = Path(temp_dir) / "experiment_runs_deleted.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "abc123",
                                "experiment_name": "baseline_lgb_alpha158",
                                "status": "completed",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "run_id": "other456",
                                "experiment_name": "risk_overlay",
                                "status": "completed",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.post("/api/v1/research/experiments/abc123/archive")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            deleted_lines = [json.loads(line) for line in deleted_runs_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "other456")

        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])

        self.assertEqual(deleted_lines[0]["run_id"], "abc123")
        self.assertEqual(deleted_lines[0]["deleted_from"], "experiment_runs.jsonl")
        self.assertIn("deleted_at", deleted_lines[0])

    def test_permanently_delete_experiment_run_removes_active_record_and_marked_note(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            deleted_runs_path = Path(temp_dir) / "experiment_runs_deleted.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                "\n".join(
                    [
                        json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}, ensure_ascii=False),
                        json.dumps({"run_id": "other456", "experiment_name": "risk_overlay"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：应被永久删除。\n"
                "<!-- /experiment-run:abc123 -->\n\n"
                "<!-- experiment-run:other456 -->\n"
                "## 2026-07-04 risk_overlay\n\n"
                "- 结论：应保留。\n"
                "<!-- /experiment-run:other456 -->\n",
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.delete("/api/v1/research/experiments/abc123/permanent")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])
        self.assertNotIn("abc123", log_text)
        self.assertNotIn("应被永久删除", log_text)
        self.assertIn("other456", log_text)
        self.assertIn("应保留", log_text)
        self.assertFalse(deleted_runs_path.exists())

    def test_permanent_delete_rejects_incomplete_note_block_without_removing_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：这个 block 缺少结束标记。\n",
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir))
                resp = client.delete("/api/v1/research/experiments/abc123/permanent")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 409)
        self.assertEqual([item["run_id"] for item in active_lines], ["abc123"])
        self.assertIn("缺少结束标记", log_text)

    def test_permanent_delete_restores_jsonl_record_when_note_rewrite_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：写入失败时应保留。\n"
                "<!-- /experiment-run:abc123 -->\n",
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch(
                    "api.v1.endpoints.research_status._remove_marked_experiment_note",
                    side_effect=OSError("simulated disk write failure"),
                ),
            ):
                client = TestClient(create_app(static_dir=static_dir), raise_server_exceptions=False)
                resp = client.delete("/api/v1/research/experiments/abc123/permanent")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 500)
        self.assertEqual([item["run_id"] for item in active_lines], ["abc123"])
        self.assertIn("写入失败时应保留", log_text)

    def test_restore_text_file_logs_warning_when_best_effort_rollback_fails(self) -> None:
        from api.v1.endpoints.research_status import _restore_text_file

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment_runs.jsonl"

            with (
                patch("pathlib.Path.write_text", side_effect=OSError("restore boom")),
                self.assertLogs("api.v1.endpoints.research_status", level="WARNING") as captured,
            ):
                _restore_text_file(path, True, "original content")

        self.assertTrue(any("rollback restore failed" in message for message in captured.output))

    def test_status_paths_can_be_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dsa_root = base / "dsa"
            projects_root = base / "workspace"
            workflow_root = base / "workflow"
            reports_root = base / "custom_reports"

            env = os.environ.copy()
            env.update(
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": str(projects_root),
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                    "QUANT_WORKFLOW_REPORTS_ROOT": str(reports_root),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "import api.v1.endpoints.research_status as r; "
                        "print(json.dumps({"
                        "'dsa_root': str(r.DSA_ROOT), "
                        "'projects_root': str(r.PROJECTS_ROOT), "
                        "'workflow_root': str(r.QUANT_WORKFLOW_ROOT), "
                        "'reports_root': str(r.QUANT_WORKFLOW_REPORTS_ROOT), "
                        "'status_path': str(r.RESEARCH_STATUS_PATH)"
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
        self.assertEqual(payload["workflow_root"], str(workflow_root))
        self.assertEqual(payload["reports_root"], str(reports_root))
        self.assertEqual(payload["status_path"], str(reports_root / "research_status_latest.json"))


if __name__ == "__main__":
    unittest.main()
