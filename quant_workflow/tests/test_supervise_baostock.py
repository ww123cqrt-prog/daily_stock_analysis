import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "supervise_baostock.py"
    spec = importlib.util.spec_from_file_location("supervise_baostock", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SuperviseBaostockTest(unittest.TestCase):
    def test_python_defaults_to_current_interpreter(self):
        module = load_module()

        self.assertEqual(module.PYTHON, Path(sys.executable))

    def test_legacy_qlib_python_env_does_not_override_current_interpreter(self):
        override = Path("/tmp/dsa-shared-python")
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "supervise_baostock.py"
        env = os.environ.copy()
        env["QLIB_PYTHON"] = str(override)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, json; "
                    "from pathlib import Path; "
                    f"path=Path({str(script_path)!r}); "
                    "spec=importlib.util.spec_from_file_location('supervise_baostock', path); "
                    "module=importlib.util.module_from_spec(spec); "
                    "spec.loader.exec_module(module); "
                    "print(json.dumps({'python': str(module.PYTHON)}))"
                ),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn(f'"python": "{sys.executable}"', result.stdout)
        self.assertNotIn(str(override), result.stdout)

    def test_quant_workflow_python_env_can_override_current_interpreter(self):
        override = Path("/tmp/dsa-shared-python")
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "supervise_baostock.py"
        env = os.environ.copy()
        env["QUANT_WORKFLOW_PYTHON"] = str(override)
        env["QLIB_PYTHON"] = "/tmp/legacy-qlib-python"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.util, json; "
                    "from pathlib import Path; "
                    f"path=Path({str(script_path)!r}); "
                    "spec=importlib.util.spec_from_file_location('supervise_baostock', path); "
                    "module=importlib.util.module_from_spec(spec); "
                    "spec.loader.exec_module(module); "
                    "print(json.dumps({'python': str(module.PYTHON)}))"
                ),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn(f'"python": "{override}"', result.stdout)
        self.assertNotIn("/tmp/legacy-qlib-python", result.stdout)


if __name__ == "__main__":
    unittest.main()
