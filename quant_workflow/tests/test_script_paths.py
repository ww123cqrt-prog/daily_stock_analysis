import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
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
        self.assertEqual(module.quant_data_root(), dsa_root / "quant_data")
        self.assertEqual(module.reports_root(), workflow_root / "reports")
        self.assertEqual(module.mlruns_root(), workflow_root / "mlruns")

    def test_working_projects_override_moves_default_project_roots(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            projects_root = Path(temp_dir) / "WorkingProjects"
            with patch.dict(os.environ, {"WORKING_PROJECTS_ROOT": str(projects_root)}, clear=False):
                self.assertEqual(module.projects_root(), projects_root)
                self.assertEqual(module.workflow_root(), Path(__file__).resolve().parents[1])
                self.assertEqual(module.quant_data_root(), Path(__file__).resolve().parents[2] / "quant_data")
                self.assertEqual(module.china_astock_root(), projects_root / "china-astock-quant")

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
                "CHINA_ASTOCK_QUANT_ROOT": str(root / "astock"),
            }
            with patch.dict(os.environ, env, clear=False):
                self.assertEqual(module.workflow_root(), root / "workflow")
                self.assertEqual(module.reports_root(), root / "reports")
                self.assertEqual(module.mlruns_root(), root / "mlruns")
                self.assertEqual(module.quant_data_root(), root / "data")
                self.assertEqual(module.qlib_provider_uri(), root / "provider")
                self.assertEqual(module.qlib_provider_uri("cleaned"), root / "provider-cleaned")
                self.assertEqual(module.china_astock_root(), root / "astock")

    def test_unknown_provider_name_is_rejected(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.qlib_provider_uri("unknown")


if __name__ == "__main__":
    unittest.main()
