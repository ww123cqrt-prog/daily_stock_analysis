# -*- coding: utf-8 -*-
"""Tests for local quant workflow path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.services import quant_workflow_paths as paths


class QuantWorkflowPathsTestCase(unittest.TestCase):
    def test_sibling_reports_and_mlruns_are_used_when_embedded_payload_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dsa_root = root / "daily_stock_analysis"
            embedded_workflow = dsa_root / "quant_workflow"
            sibling_workflow = root / "quant_workflow"
            sibling_reports = sibling_workflow / "reports"
            sibling_mlruns = sibling_workflow / "mlruns"
            embedded_workflow.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            sibling_mlruns.mkdir(parents=True)
            (sibling_reports / "research_status_latest.json").write_text("{}", encoding="utf-8")
            (sibling_mlruns / "0").mkdir()

            with patch.dict(
                os.environ,
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": "",
                    "QUANT_WORKFLOW_ROOT": "",
                    "QUANT_WORKFLOW_REPORTS_ROOT": "",
                    "QUANT_WORKFLOW_MLRUNS_DIR": "",
                },
                clear=False,
            ):
                self.assertEqual(paths.workflow_root(), embedded_workflow)
                self.assertEqual(paths.reports_root(), sibling_reports)
                self.assertEqual(paths.mlruns_root(), sibling_mlruns)

    def test_embedded_reports_take_precedence_when_they_have_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dsa_root = root / "daily_stock_analysis"
            embedded_reports = dsa_root / "quant_workflow" / "reports"
            sibling_reports = root / "quant_workflow" / "reports"
            embedded_reports.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            (embedded_reports / "latest_training_result.json").write_text("{}", encoding="utf-8")
            (sibling_reports / "research_status_latest.json").write_text("{}", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": "",
                    "QUANT_WORKFLOW_ROOT": "",
                    "QUANT_WORKFLOW_REPORTS_ROOT": "",
                },
                clear=False,
            ):
                self.assertEqual(paths.reports_root(), embedded_reports)

    def test_explicit_environment_overrides_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "DSA_ROOT": str(root / "daily_stock_analysis"),
                "QUANT_WORKFLOW_ROOT": str(root / "custom-workflow"),
                "QUANT_WORKFLOW_REPORTS_ROOT": str(root / "custom-reports"),
                "QUANT_WORKFLOW_MLRUNS_DIR": str(root / "custom-mlruns"),
                "QUANT_DATA_ROOT": str(root / "custom-data"),
            }

            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(paths.workflow_root(), root / "custom-workflow")
                self.assertEqual(paths.reports_root(), root / "custom-reports")
                self.assertEqual(paths.mlruns_root(), root / "custom-mlruns")
                self.assertEqual(paths.quant_data_root(), root / "custom-data")


if __name__ == "__main__":
    unittest.main()
