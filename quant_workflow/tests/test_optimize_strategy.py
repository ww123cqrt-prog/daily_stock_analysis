import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "optimize_strategy.py"
    spec = importlib.util.spec_from_file_location("optimize_strategy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OptimizeStrategyTest(unittest.TestCase):
    def test_data_dir_honors_quant_data_root_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "optimize_strategy.py"
            env = os.environ.copy()
            env["QUANT_DATA_ROOT"] = str(quant_data)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.util, json; "
                        "from pathlib import Path; "
                        f"path=Path({str(script_path)!r}); "
                        "spec=importlib.util.spec_from_file_location('optimize_strategy', path); "
                        "module=importlib.util.module_from_spec(spec); "
                        "spec.loader.exec_module(module); "
                        "print(json.dumps({'data_dir': str(module.DATA_DIR)}))"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["data_dir"], str(quant_data / "qlib" / "cn_data_baostock_jqdata"))

    def test_dry_run_writes_timestamped_report_and_latest_aliases(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"

            status = module.run_report(
                phase=4,
                topk=3,
                skip_training=True,
                dry_run=True,
                output_dir=out_dir,
                timestamp="20260705_180000",
            )

            result_json = out_dir / "optimization_results_20260705_180000.json"
            result_md = out_dir / "optimization_results_20260705_180000.md"
            latest_json = out_dir / "optimization_results_latest.json"
            latest_md = out_dir / "optimization_results_latest.md"

            self.assertEqual(status["status"], "completed")
            self.assertTrue(status["dry_run"])
            self.assertEqual(status["output_json"], str(result_json.resolve()))
            self.assertEqual(status["output_md"], str(result_md.resolve()))
            self.assertTrue(result_json.exists())
            self.assertTrue(result_md.exists())
            self.assertEqual(latest_json.read_text(encoding="utf-8"), result_json.read_text(encoding="utf-8"))
            self.assertEqual(latest_md.read_text(encoding="utf-8"), result_md.read_text(encoding="utf-8"))

            payload = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], 4)
            self.assertTrue(payload["dry_run"])
            self.assertIn("phase4_combined", payload["results"])
            self.assertIn("phase4_conservative", payload["results"])
            self.assertIn("只读策略优化预演", result_md.read_text(encoding="utf-8"))

    def test_rejects_invalid_web_safe_params(self):
        module = load_module()

        for kwargs, message in (
            ({"phase": 5, "topk": 3}, "phase"),
            ({"phase": 1, "topk": 0}, "topk"),
            ({"phase": 1, "topk": 501}, "topk"),
        ):
            with self.subTest(kwargs=kwargs):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with self.assertRaises(ValueError) as ctx:
                        module.run_report(
                            skip_training=True,
                            dry_run=True,
                            output_dir=Path(temp_dir),
                            timestamp="20260705_180000",
                            **kwargs,
                        )
                self.assertIn(message, str(ctx.exception))

    def test_cli_supports_browser_safe_dry_run(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "optimize_strategy.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dry-run",
                    "--skip-training",
                    "--phase",
                    "4",
                    "--topk",
                    "3",
                    "--output-dir",
                    str(Path(temp_dir) / "reports"),
                    "--timestamp",
                    "20260705_181500",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(temp_dir) / "reports" / "optimization_results_latest.json").exists())
            self.assertIn("只读策略优化", result.stdout)


if __name__ == "__main__":
    unittest.main()
