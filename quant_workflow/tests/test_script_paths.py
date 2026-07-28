import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_paths.py"
    spec = importlib.util.spec_from_file_location("script_paths", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptPathsTest(unittest.TestCase):
    def test_default_roots_follow_workflow_location(self):
        module = load_module()
        workflow_root = Path(__file__).resolve().parents[1]
        dsa_root = workflow_root.parent

        self.assertEqual(module.workflow_root(), workflow_root)
        self.assertEqual(module.projects_root(), dsa_root.parent)
        sibling_quant_data = dsa_root.parent / "quant_data"
        expected_quant_data = dsa_root / "quant_data"
        if not expected_quant_data.exists() and sibling_quant_data.exists():
            expected_quant_data = sibling_quant_data
        self.assertEqual(module.quant_data_root(), expected_quant_data)
        sibling_reports = dsa_root.parent / "quant_workflow" / "reports"
        expected_reports = workflow_root / "reports"
        self.assertEqual(module.reports_root(), expected_reports)

        sibling_mlruns = dsa_root.parent / "quant_workflow" / "mlruns"
        expected_mlruns = workflow_root / "mlruns"
        self.assertEqual(module.mlruns_root(), expected_mlruns)

    def test_working_projects_override_moves_default_project_roots(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            projects_root = Path(temp_dir) / "WorkingProjects"
            with patch.dict(os.environ, {"WORKING_PROJECTS_ROOT": str(projects_root)}, clear=False):
                self.assertEqual(module.projects_root(), projects_root)
                self.assertEqual(module.workflow_root(), Path(__file__).resolve().parents[1])
                self.assertEqual(module.quant_data_root(), Path(__file__).resolve().parents[2] / "quant_data")
                self.assertEqual(module.china_astock_root(), projects_root / "china-astock-quant")

    def test_sibling_quant_data_is_used_when_repo_local_data_is_absent(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            sibling_data = temp_root / "quant_data"
            workflow_root.mkdir(parents=True)
            sibling_data.mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_DATA_ROOT",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.quant_data_root(), sibling_data)

    def test_repo_local_reports_and_mlruns_are_default_even_when_sibling_payload_exists(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            sibling_workflow = temp_root / "quant_workflow"
            sibling_reports = sibling_workflow / "reports"
            sibling_mlruns = sibling_workflow / "mlruns"
            workflow_root.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            sibling_mlruns.mkdir(parents=True)
            (sibling_reports / "latest_training_result.json").write_text("{}", encoding="utf-8")
            (sibling_mlruns / "0").mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_WORKFLOW_ROOT",
                        "QUANT_WORKFLOW_REPORTS_ROOT",
                        "QUANT_WORKFLOW_MLRUNS_DIR",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.reports_root(), workflow_root / "reports")
                self.assertEqual(module.mlruns_root(), workflow_root / "mlruns")

    def test_repo_local_reports_and_mlruns_win_when_both_payloads_exist(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            repo_reports = workflow_root / "reports"
            repo_mlruns = workflow_root / "mlruns"
            sibling_reports = temp_root / "quant_workflow" / "reports"
            sibling_mlruns = temp_root / "quant_workflow" / "mlruns"
            repo_reports.mkdir(parents=True)
            repo_mlruns.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            sibling_mlruns.mkdir(parents=True)
            (repo_reports / "research_status_latest.json").write_text("{}", encoding="utf-8")
            (repo_mlruns / "0").mkdir()
            (sibling_reports / "latest_training_result.json").write_text("{}", encoding="utf-8")
            (sibling_mlruns / "models").mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_WORKFLOW_ROOT",
                        "QUANT_WORKFLOW_REPORTS_ROOT",
                        "QUANT_WORKFLOW_MLRUNS_DIR",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.reports_root(), repo_reports)
                self.assertEqual(module.mlruns_root(), repo_mlruns)

    def test_specific_env_overrides_take_precedence(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "QUANT_WORKFLOW_ROOT": str(root / "workflow"),
                "QUANT_WORKFLOW_REPORTS_ROOT": str(root / "reports"),
                "QUANT_WORKFLOW_MLRUNS_DIR": str(root / "mlruns"),
                "QUANT_DATA_ROOT": str(root / "data"),
                "QLIB_PROVIDER_URI": str(root / "provider"),
                "QLIB_CLEANED_PROVIDER_URI": str(root / "provider-cleaned"),
                "QLIB_COMPATIBILITY_PROVIDER_URI": str(root / "provider-compat"),
                "CHINA_ASTOCK_QUANT_ROOT": str(root / "astock"),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(module.workflow_root(), root / "workflow")
                self.assertEqual(module.reports_root(), root / "reports")
                self.assertEqual(module.mlruns_root(), root / "mlruns")
                self.assertEqual(module.quant_data_root(), root / "data")
                self.assertEqual(module.qlib_provider_uri(), root / "provider")
                self.assertEqual(module.qlib_provider_uri("cleaned"), root / "provider-cleaned")
                self.assertEqual(
                    module.qlib_compatibility_provider_uri(), root / "provider-compat"
                )
                self.assertEqual(
                    module.qlib_compatibility_provider_uri("cleaned"),
                    root / "provider-cleaned",
                )
                self.assertEqual(module.china_astock_root(), root / "astock")

    def test_unknown_provider_name_is_rejected(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.qlib_provider_uri("unknown")

    def test_canonical_provider_resolves_native_v2_snapshot_once_per_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            first = root / "qlib" / "native" / "snapshots" / "snapshot-001"
            second = root / "qlib" / "native" / "snapshots" / "snapshot-002"
            resolved = [
                SimpleNamespace(
                    snapshot_path=first,
                    manifest={"benchmark_complete": True},
                ),
                SimpleNamespace(
                    snapshot_path=second,
                    manifest={"benchmark_complete": True},
                ),
            ]
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root), "QLIB_PROVIDER_URI": ""},
                clear=False,
            ), patch.object(module, "resolve_dataset", side_effect=resolved) as resolver:
                module._resolve_native_qlib.cache_clear()
                self.assertEqual(module.qlib_provider_uri(), first)
                self.assertEqual(module.qlib_provider_uri(), first)

            self.assertEqual(resolver.call_count, 1)
            resolver.assert_called_once_with(
                "qlib_ashare_v2", quant_data_root=root.resolve(), require_basis="qfq"
            )

    def test_canonical_provider_fails_closed_when_native_benchmarks_are_incomplete(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            resolved = SimpleNamespace(
                snapshot_path=root / "snapshot",
                manifest={"benchmark_complete": False},
            )
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root), "QLIB_PROVIDER_URI": ""},
                clear=False,
            ), patch.object(module, "resolve_dataset", return_value=resolved):
                module._resolve_native_qlib.cache_clear()
                with self.assertRaisesRegex(RuntimeError, "benchmarks are incomplete"):
                    module.qlib_provider_uri()

    def test_ashare_qfq_bars_resolve_native_snapshot_once_per_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            first = root / "parquet" / "native" / "snapshots" / "snapshot-001"
            second = root / "parquet" / "native" / "snapshots" / "snapshot-002"
            resolved = [
                SimpleNamespace(snapshot_path=first, manifest={}),
                SimpleNamespace(snapshot_path=second, manifest={}),
            ]
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root)},
                clear=False,
            ), patch.object(module, "resolve_dataset", side_effect=resolved) as resolver:
                module._resolve_native_ashare_qfq.cache_clear()
                self.assertEqual(module.ashare_qfq_bars_dir(), first / "bars")
                self.assertEqual(module.ashare_qfq_bars_dir(), first / "bars")

            self.assertEqual(resolver.call_count, 1)
            resolver.assert_called_once_with(
                "ashare_daily_qfq",
                quant_data_root=root.resolve(),
                require_basis="qfq",
            )


if __name__ == "__main__":
    unittest.main()
