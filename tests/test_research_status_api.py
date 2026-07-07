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

from fastapi import HTTPException
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
            inventory_path = Path(temp_dir) / "data_hygiene_inventory_latest.json"
            cleanup_plan_path = Path(temp_dir) / "data_hygiene_cleanup_plan_latest.json"
            qlib_research_path = Path(temp_dir) / "qlib_research_model_explainability_latest.json"
            quality_report_path = Path(temp_dir) / "data_quality_report_latest.md"
            quality_report_json_path = Path(temp_dir) / "data_quality_report_latest.json"
            cleaning_strategy_path = Path(temp_dir) / "data_cleaning_strategy_latest.md"
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            manifest_path = quant_data_root / "DATA_MANIFEST.json"
            readme_path = quant_data_root / "README.md"
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
            inventory_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T08:00:00+08:00",
                        "items": (
                            [{"path": f"/tmp/official_{idx}", "category": "official"} for idx in range(2)]
                            + [{"path": f"/tmp/candidate_{idx}", "categories": ["candidate"]} for idx in range(4)]
                            + [{"path": f"/tmp/legacy_{idx}", "category": "legacy"} for idx in range(5)]
                            + [
                                {"path": f"/tmp/archive_{idx}", "categories": ["archive_candidate"]}
                                for idx in range(8)
                            ]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            cleanup_plan_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T08:00:00+08:00",
                        "dry_run_only": True,
                        "actions": [
                            {
                                "action": "archive",
                                "source": "/Users/cq/WorkingProjects/quant_data/qlib/cn_data",
                                "destination": "/Users/cq/WorkingProjects/quant_data/archive/qlib/cn_data",
                                "reason": "旧测试 provider，不是当前正式入口。",
                                "risk": "可能影响旧实验复现。",
                                "requires_user_confirmation": True,
                            },
                            {
                                "action": "review",
                                "requires_user_confirmation": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            qlib_research_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T09:00:00+08:00",
                        "mode": "read_only_qlib_research",
                        "section": "model_explainability",
                        "title": "模型效果深度解释",
                        "status": "available",
                        "artifacts": {
                            "latest_json": str(qlib_research_path),
                            "latest_markdown": str(Path(temp_dir) / "qlib_research_model_explainability_latest.md"),
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            quality_report_path.write_text(
                "# A-Share Data Quality Report\n\n"
                "- Stock universe: 5509\n"
                "- Merged usable instruments: 5340\n"
                "- Merged rows: 7,580,087\n"
                "- Calendar range: 2020-01-02T00:00:00+08:00 ~ 2026-07-03T15:00:00+08:00\n",
                encoding="utf-8",
            )
            quality_report_json_path.write_text(
                json.dumps(
                    {
                        "summary": {
                            "stock_universe": 5510,
                            "usable_instruments": 5341,
                            "total_rows": "7,580,088",
                            "date_range": "2020-01-03 ~ 2026-07-06",
                        }
                    }
                ),
                encoding="utf-8",
            )
            cleaning_strategy_path.write_text("# Cleaning Strategy\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T21:49:15+08:00",
                        "source_report": str(inventory_path),
                        "official_entries": {
                            "cleaned_parquet": {"path": "/tmp/cleaned_parquet"},
                            "cleaned_qlib_provider": {"path": "/tmp/cleaned_provider"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            readme_path.write_text("# quant_data\n", encoding="utf-8")

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root),
                patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6106/"}, clear=False),
            ):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "trainable")
        self.assertEqual(body["data_sources"][0]["name"], "akshare")
        self.assertEqual(body["latest_training"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["status"], "available")
        self.assertEqual(body["mlflow"]["url"], "http://127.0.0.1:6106/")
        self.assertEqual(body["mlflow"]["status"], "linked")
        self.assertFalse(body["mlflow"]["health_checked"])
        self.assertIn("workflow page", body["mlflow"]["error"])
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual(body["experiment_notebook"]["log_path"], "experiment_log.md")
        self.assertEqual(body["experiment_notebook"]["runs_path"], "experiment_runs.jsonl")
        self.assertIn("长历史 baseline 已跑通", body["experiment_notebook"]["latest_note"])
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["runs"][0]["dataset_range"], "2020-01-02 ~ 2026-07-01")
        self.assertEqual(body["data_hygiene"]["inventory"]["status"], "available")
        self.assertEqual(body["data_hygiene"]["inventory"]["official_entries"], 2)
        self.assertEqual(body["data_hygiene"]["inventory"]["archive_candidates"], 8)
        self.assertTrue(body["data_hygiene"]["cleanup_plan"]["dry_run_only"])
        self.assertEqual(body["data_hygiene"]["cleanup_plan"]["total_actions"], 2)
        self.assertFalse(body["data_hygiene"]["cleanup_plan"]["truncated"])
        self.assertEqual(body["data_hygiene"]["cleanup_plan"]["preview_limit"], 50)
        self.assertTrue(body["data_hygiene"]["cleanup_plan"]["actions"][0]["requires_confirmation"])
        self.assertEqual(
            body["data_hygiene"]["cleanup_plan"]["actions"][0]["source"],
            "WorkingProjects/quant_data/qlib/cn_data",
        )
        self.assertEqual(
            body["data_hygiene"]["cleanup_plan"]["actions"][0]["destination"],
            "WorkingProjects/quant_data/archive/qlib/cn_data",
        )
        self.assertEqual(body["data_hygiene"]["cleanup_plan"]["actions"][1]["source"], "")
        self.assertIsNone(body["data_hygiene"]["cleanup_plan"]["actions"][1]["destination"])
        self.assertEqual(
            body["data_hygiene"]["quality_report"]["summary"]["date_range"],
            "2020-01-03 ~ 2026-07-06",
        )
        self.assertEqual(body["data_hygiene"]["quality_report"]["summary"]["stock_universe"], "5510")
        self.assertEqual(body["data_hygiene"]["quality_report"]["summary"]["usable_instruments"], "5341")
        self.assertEqual(body["data_hygiene"]["cleaning_strategy"]["status"], "available")
        self.assertEqual(body["data_hygiene"]["manifest"]["status"], "available")
        self.assertEqual(body["data_hygiene"]["manifest"]["source_report"], "data_hygiene_inventory_latest.json")
        self.assertEqual(body["data_hygiene"]["manifest"]["official_entries"], 2)
        self.assertEqual(body["data_hygiene"]["readme"]["status"], "available")
        self.assertNotIn(str(temp_dir), json.dumps(body["data_hygiene"], ensure_ascii=False))
        self.assertNotIn("/Users/cq/", json.dumps(body["data_hygiene"], ensure_ascii=False))
        self.assertEqual(body["qlib_research"]["status"], "available")
        self.assertEqual(body["qlib_research"]["available_sections"], 1)
        self.assertEqual(
            body["qlib_research"]["sections"]["model_explainability"]["artifacts"]["latest_json"],
            "qlib_research_model_explainability_latest.json",
        )
        self.assertEqual(body["qlib_research"]["sections"]["risk_model"]["status"], "missing")
        self.assertNotIn(str(temp_dir), json.dumps(body["qlib_research"], ensure_ascii=False))
        matrix_by_id = {item["id"]: item for item in body["capability_matrix"]}
        self.assertEqual(matrix_by_id["dsa_web_core"]["status"], "available")
        self.assertEqual(matrix_by_id["quant_workflow_pipeline"]["route"], "/workflow")
        self.assertEqual(matrix_by_id["qlib_integration"]["route"], "/backtest")
        self.assertEqual(matrix_by_id["china_astock_quant_reuse"]["status"], "available")
        self.assertEqual(matrix_by_id["china_astock_quant_reuse"]["route"], "/workflow")
        self.assertEqual(matrix_by_id["external_tools_tonghuashun"]["status"], "read_only")
        self.assertEqual(matrix_by_id["trading_execution"]["status"], "not_planned")
        self.assertIn("不自动下单", matrix_by_id["trading_execution"]["restrictions"])
        self.assertIn("不读取券商密码", matrix_by_id["trading_execution"]["restrictions"])
        inventory_by_id = {item["id"]: item for item in body["function_inventory"]}
        self.assertEqual(inventory_by_id["qlib_data_pipeline"]["route"], "/workflow")
        self.assertEqual(inventory_by_id["china_astock_reference"]["status"], "available")
        self.assertIn("report_china_astock_indicators", inventory_by_id["china_astock_reference"]["verification"])
        self.assertEqual(inventory_by_id["quant_workflow_script_coverage"]["route"], "/research")
        self.assertEqual(inventory_by_id["quant_workflow_script_coverage"]["status"], "available")
        self.assertIn("report_data_hygiene.py", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("dry-run cleanup plan", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("add_index_benchmark.py", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("generate_csi300.py", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("强制 dry-run", inventory_by_id["quant_workflow_script_coverage"]["coverage"])
        self.assertIn("supervise_baostock.py", inventory_by_id["quant_workflow_script_coverage"]["gaps"])
        self.assertIn("不允许网页写入 provider", inventory_by_id["quant_workflow_script_coverage"]["gaps"])
        reuse_by_id = {item["id"]: item for item in body["china_astock_reuse"]}
        self.assertEqual(reuse_by_id["indicators"]["priority"], "P1")
        self.assertEqual(reuse_by_id["indicators"]["status"], "available")
        self.assertIn("工作流指标报告", reuse_by_id["indicators"]["target"])
        self.assertEqual(reuse_by_id["paper_trading"]["status"], "not_planned")
        self.assertIn("不自动下单", reuse_by_id["paper_trading"]["restrictions"])
        acceptance_by_id = {item["id"]: item for item in body["acceptance_checklist"]}
        self.assertEqual(acceptance_by_id["project_hygiene"]["status"], "partial")
        self.assertIn("无散落未跟踪源码", acceptance_by_id["project_hygiene"]["evidence"])
        self.assertIn("未提交的正式改动", acceptance_by_id["project_hygiene"]["evidence"])
        self.assertNotIn("已暂存", acceptance_by_id["project_hygiene"]["evidence"])
        self.assertEqual(acceptance_by_id["qlib_web_flow"]["route"], "/workflow")
        self.assertIn("MLflow", acceptance_by_id["qlib_web_flow"]["evidence"])
        self.assertIn("强制 dry-run", acceptance_by_id["quant_workflow_wrapped"]["evidence"])
        self.assertIn("二次确认", acceptance_by_id["quant_workflow_wrapped"]["next"])
        self.assertEqual(acceptance_by_id["opencode_review"]["status"], "verified")
        self.assertIn("opencode", acceptance_by_id["opencode_review"]["evidence"])

    def test_experiment_file_guard_uses_msvcrt_fallback_when_fcntl_is_unavailable(self) -> None:
        from api.v1.endpoints.research_status import _experiment_file_guard

        class FakeMsvcrt:
            LK_LOCK = 1
            LK_UNLCK = 2

            def __init__(self) -> None:
                self.calls: list[tuple[int, int]] = []

            def locking(self, _fileno: int, mode: int, size: int) -> None:
                self.calls.append((mode, size))

        fake_msvcrt = FakeMsvcrt()
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            with (
                patch("api.v1.endpoints.research_status.fcntl", None),
                patch("api.v1.endpoints.research_status.msvcrt", fake_msvcrt),
            ):
                with _experiment_file_guard(reports_dir):
                    self.assertTrue((reports_dir / ".experiment.lock").exists())

        self.assertEqual(fake_msvcrt.calls, [(FakeMsvcrt.LK_LOCK, 1), (FakeMsvcrt.LK_UNLCK, 1)])

    def test_status_prefers_latest_training_result_over_stale_status_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            cleaned_provider = Path(temp_dir) / "cn_data_baostock_jqdata_cleaned"
            (cleaned_provider / "calendars").mkdir(parents=True)
            (cleaned_provider / "instruments").mkdir(parents=True)
            (cleaned_provider / "features" / "SH600000").mkdir(parents=True)
            (cleaned_provider / "calendars" / "day.txt").write_text(
                "2008-01-02\n2026-07-03\n",
                encoding="utf-8",
            )
            (cleaned_provider / "instruments" / "all.txt").write_text(
                "SH600000\t2008-01-02\t2026-07-03\nSZ000001\t2008-01-02\t2026-07-03\n",
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
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
                        "provider_uri": str(cleaned_provider),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT", cleaned_provider.parent),
            ):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "new-cleaned-run")
        self.assertEqual(body["latest_training"]["provider"], "cleaned")
        self.assertEqual(body["qlib_provider"]["path"], str(cleaned_provider))
        self.assertEqual(body["qlib_provider"]["calendar_range"], "2008-01-02 ~ 2026-07-03")
        self.assertEqual(body["qlib_provider"]["calendar_days"], 2)
        self.assertEqual(body["qlib_provider"]["instruments"], 2)

    def test_directory_size_cache_avoids_repeated_provider_scans_until_ttl(self) -> None:
        from api.v1.endpoints.research_status import _clear_directory_size_cache, _directory_size_mb

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                (provider / "first.bin").write_bytes(b"0" * 1024 * 1024)

                with patch("api.v1.endpoints.research_status.time.monotonic", side_effect=[0.0, 0.0, 0.0, 10.0, 61.0, 61.0, 61.0]):
                    first = _directory_size_mb(provider)
                    (provider / "second.bin").write_bytes(b"1" * 1024 * 1024)
                    cached = _directory_size_mb(provider)
                    refreshed = _directory_size_mb(provider)
        finally:
            _clear_directory_size_cache()

        self.assertEqual(first, 1.0)
        self.assertEqual(cached, 1.0)
        self.assertEqual(refreshed, 2.0)

    def test_directory_size_scan_returns_unknown_after_time_budget(self) -> None:
        from api.v1.endpoints.research_status import _clear_directory_size_cache, _directory_size_mb

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                for index in range(100):
                    (provider / f"{index:03d}.bin").write_bytes(b"0")

                with (
                    patch("api.v1.endpoints.research_status.DIRECTORY_SIZE_SCAN_MAX_SECONDS", 0.1),
                    patch("api.v1.endpoints.research_status.time.monotonic", side_effect=[0.0, 0.0, 0.2]),
                ):
                    self.assertIsNone(_directory_size_mb(provider))
        finally:
            _clear_directory_size_cache()

    def test_directory_size_scan_returns_unknown_after_entry_limit(self) -> None:
        from api.v1.endpoints.research_status import _clear_directory_size_cache, _directory_size_mb

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                (provider / "first.bin").write_bytes(b"0")
                (provider / "second.bin").write_bytes(b"1")

                with patch("api.v1.endpoints.research_status.DIRECTORY_SIZE_SCAN_MAX_ENTRIES", 1):
                    self.assertIsNone(_directory_size_mb(provider))
        finally:
            _clear_directory_size_cache()

    def test_data_hygiene_cleanup_plan_reports_truncated_action_preview(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            (reports_dir / "data_hygiene_inventory_latest.json").write_text(
                json.dumps({"generated_at": "2026-07-07T00:00:00+08:00", "items": []}),
                encoding="utf-8",
            )
            (reports_dir / "data_hygiene_cleanup_plan_latest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-07T00:00:00+08:00",
                        "dry_run_only": True,
                        "actions": [{"action": "archive", "source": f"/tmp/item_{idx}"} for idx in range(55)],
                    }
                ),
                encoding="utf-8",
            )

            payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["cleanup_plan"]["total_actions"], 55)
        self.assertEqual(len(payload["cleanup_plan"]["actions"]), 50)
        self.assertTrue(payload["cleanup_plan"]["truncated"])
        self.assertEqual(payload["cleanup_plan"]["preview_limit"], 50)

    def test_data_hygiene_manifest_counts_list_official_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            (quant_data_root / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-07T00:00:00+08:00",
                        "source_report": str(reports_dir / "data_hygiene_inventory_latest.json"),
                        "official_entries": [
                            {"path": "/tmp/cleaned_parquet"},
                            {"path": "/tmp/cleaned_provider"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root):
                from api.v1.endpoints.research_status import _data_hygiene_payload

                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["manifest"]["status"], "available")
        self.assertEqual(payload["manifest"]["official_entries"], 2)

    def test_status_redacts_mlflow_url_credentials_query_and_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"generated_at": "2026-07-02T10:00:00+08:00", "trainable": True}),
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch.dict(
                    os.environ,
                    {"MLFLOW_URL": "http://fake_user:fake_pass@127.0.0.1:6106/?token=fake#frag"},
                    clear=False,
                ),
            ):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        mlflow = resp.json()["mlflow"]
        self.assertEqual(mlflow["url"], "http://127.0.0.1:6106/")
        self.assertEqual(mlflow["status"], "invalid")
        self.assertEqual(mlflow["role"], "experiment_raw_records")
        self.assertIsNone(mlflow["checked_at"])
        self.assertIn("username or password", mlflow["error"])
        self.assertNotIn("fake_pass", json.dumps(mlflow))
        self.assertNotIn("token=fake", json.dumps(mlflow))

    def test_status_marks_external_mlflow_as_unprobed_link(self) -> None:
        from api.v1.endpoints.research_status import _mlflow_payload

        with patch.dict(os.environ, {"MLFLOW_URL": "https://mlflow.example.com/experiments?token=fake"}, clear=False):
            mlflow = _mlflow_payload()

        self.assertEqual(mlflow["url"], "https://mlflow.example.com/experiments")
        self.assertEqual(mlflow["status"], "linked")
        self.assertFalse(mlflow["health_checked"])
        self.assertIn("not probed", mlflow["error"])
        self.assertNotIn("token=fake", json.dumps(mlflow))

    def test_status_rejects_unsafe_mlflow_url_with_sanitized_display(self) -> None:
        from api.v1.endpoints.research_status import _mlflow_payload

        with patch.dict(os.environ, {"MLFLOW_URL": "http://127.0.0.1:6106/%2e%2e/admin"}, clear=False):
            traversal = _mlflow_payload()
        with patch.dict(os.environ, {"MLFLOW_URL": "file:///tmp/mlflow"}, clear=False):
            non_http = _mlflow_payload()

        self.assertEqual(traversal["url"], "http://127.0.0.1:6106/%2e%2e/admin")
        self.assertEqual(traversal["status"], "invalid")
        self.assertIn("path traversal", traversal["error"])
        self.assertEqual(non_http["url"], "http://127.0.0.1:6006/")
        self.assertEqual(non_http["status"], "invalid")
        self.assertIn("http(s)", non_http["error"])

    def test_status_does_not_scan_untrusted_latest_training_provider_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            untrusted_provider = Path(temp_dir) / "outside" / "cn_data"
            untrusted_provider.mkdir(parents=True)
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "untrusted-provider-run",
                        "provider_uri": str(untrusted_provider),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT", Path(temp_dir) / "trusted_qlib_root"),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    side_effect=AssertionError("untrusted provider must not be scanned"),
                ),
            ):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "untrusted-provider-run")
        self.assertEqual(body["qlib_provider"]["path"], "/tmp/stale-cn-data")
        self.assertFalse(body["qlib_provider"]["latest_training_provider_trusted"])
        self.assertIn("outside configured Qlib provider root", body["qlib_provider"]["provider_uri_warning"])

    def test_status_does_not_scan_qlib_root_as_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            qlib_root = Path(temp_dir) / "quant_data" / "qlib"
            qlib_root.mkdir(parents=True)
            status_path = Path(temp_dir) / "research_status_latest.json"
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {"status": "trainable", "summary": "ready"},
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "root-provider-run",
                        "provider_uri": str(qlib_root),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path),
                patch("api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT", qlib_root),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    side_effect=AssertionError("qlib root must not be scanned as a provider"),
                ),
            ):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "root-provider-run")
        self.assertFalse(body["qlib_provider"]["latest_training_provider_trusted"])
        self.assertIn("not a concrete Qlib provider", body["qlib_provider"]["provider_uri_warning"])

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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.post("/api/v1/research/experiments/abc123/archive")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            deleted_lines = [json.loads(line) for line in deleted_runs_path.read_text(encoding="utf-8").splitlines()]
            lock_exists = (Path(temp_dir) / ".experiment.lock").exists()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(lock_exists)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "other456")

        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])

        self.assertEqual(deleted_lines[0]["run_id"], "abc123")
        self.assertEqual(deleted_lines[0]["deleted_from"], "experiment_runs.jsonl")
        self.assertIn("deleted_at", deleted_lines[0])

    def test_archive_experiment_run_restores_active_record_when_archive_write_fails(self) -> None:
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
            runs_text = (
                json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}, ensure_ascii=False)
                + "\n"
                + json.dumps({"run_id": "other456", "experiment_name": "risk_overlay"}, ensure_ascii=False)
                + "\n"
            )
            runs_path.write_text(runs_text, encoding="utf-8")
            deleted_runs_path.mkdir()

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.post("/api/v1/research/experiments/abc123/archive")

            persisted_runs = runs_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 500)
        self.assertEqual(persisted_runs, runs_text)

    def test_archive_experiment_run_restores_snapshots_when_http_exception_is_raised(self) -> None:
        from api.v1.endpoints.research_status import _archive_experiment_run

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            runs_path = reports_dir / "experiment_runs.jsonl"
            deleted_runs_path = reports_dir / "experiment_runs_deleted.jsonl"
            runs_text = json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}) + "\n"
            deleted_runs_text = json.dumps({"run_id": "old999", "deleted_from": "experiment_runs.jsonl"}) + "\n"
            runs_path.write_text(runs_text, encoding="utf-8")
            deleted_runs_path.write_text(deleted_runs_text, encoding="utf-8")

            def mutate_then_fail(_reports_dir: Path, _run_id: str) -> dict[str, object]:
                runs_path.write_text("", encoding="utf-8")
                deleted_runs_path.write_text("", encoding="utf-8")
                raise HTTPException(status_code=503, detail="temporary archive failure")

            with patch("api.v1.endpoints.research_status._remove_active_experiment_run", side_effect=mutate_then_fail):
                with self.assertRaises(HTTPException) as ctx:
                    _archive_experiment_run(reports_dir, "abc123")

            persisted_runs = runs_path.read_text(encoding="utf-8")
            persisted_deleted_runs = deleted_runs_path.read_text(encoding="utf-8")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(persisted_runs, runs_text)
        self.assertEqual(persisted_deleted_runs, deleted_runs_text)

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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.delete("/api/v1/research/experiments/abc123/permanent?confirm=permanent")

            active_lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
            log_text = log_path.read_text(encoding="utf-8")
            lock_exists = (Path(temp_dir) / ".experiment.lock").exists()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(lock_exists)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])
        self.assertNotIn("abc123", log_text)
        self.assertNotIn("应被永久删除", log_text)
        self.assertIn("other456", log_text)
        self.assertIn("应保留", log_text)
        self.assertFalse(deleted_runs_path.exists())

    def test_permanent_delete_requires_backend_confirmation_without_mutating_files(self) -> None:
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
            runs_text = json.dumps({"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}, ensure_ascii=False) + "\n"
            log_text = (
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：没有后端确认时必须保留。\n"
                "<!-- /experiment-run:abc123 -->\n"
            )
            runs_path.write_text(runs_text, encoding="utf-8")
            log_path.write_text(log_text, encoding="utf-8")

            with patch("api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path):
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.delete("/api/v1/research/experiments/abc123/permanent")

            persisted_runs = runs_path.read_text(encoding="utf-8")
            persisted_log = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 400)
        self.assertIn("confirm=permanent", resp.json()["message"])
        self.assertEqual(persisted_runs, runs_text)
        self.assertEqual(persisted_log, log_text)

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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"})
                resp = client.delete("/api/v1/research/experiments/abc123/permanent?confirm=permanent")

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
                client = TestClient(create_app(static_dir=static_dir), headers={"Origin": "http://127.0.0.1:8010"}, raise_server_exceptions=False)
                resp = client.delete("/api/v1/research/experiments/abc123/permanent?confirm=permanent")

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

    def test_default_status_paths_use_embedded_quant_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dsa_root = Path(temp_dir) / "daily_stock_analysis"
            dsa_root.mkdir(parents=True)

            env = os.environ.copy()
            env["DSA_ROOT"] = str(dsa_root)
            env.pop("WORKING_PROJECTS_ROOT", None)
            env.pop("QUANT_WORKFLOW_ROOT", None)
            env.pop("QUANT_WORKFLOW_REPORTS_ROOT", None)
            env.pop("QUANT_DATA_ROOT", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "import api.v1.endpoints.research_status as r; "
                        "print(json.dumps({"
                        "'workflow_root': str(r.QUANT_WORKFLOW_ROOT), "
                        "'reports_root': str(r.QUANT_WORKFLOW_REPORTS_ROOT), "
                        "'quant_data_root': str(r.QUANT_DATA_ROOT), "
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

        self.assertEqual(payload["workflow_root"], str(dsa_root / "quant_workflow"))
        self.assertEqual(payload["reports_root"], str(dsa_root / "quant_workflow" / "reports"))
        self.assertEqual(payload["quant_data_root"], str(dsa_root / "quant_data"))
        self.assertEqual(payload["status_path"], str(dsa_root / "quant_workflow" / "reports" / "research_status_latest.json"))


if __name__ == "__main__":
    unittest.main()
