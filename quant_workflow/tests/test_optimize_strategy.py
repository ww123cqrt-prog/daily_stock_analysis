import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "optimize_strategy.py"
    spec = importlib.util.spec_from_file_location("optimize_strategy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OptimizeStrategyTest(unittest.TestCase):
    def test_data_dir_honors_explicit_provider_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "native-provider"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "optimize_strategy.py"
            env = os.environ.copy()
            env["QLIB_PROVIDER_URI"] = str(provider)

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
        self.assertEqual(payload["data_dir"], str(provider))

    def test_training_runtime_config_locks_resolved_native_provider(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "baseline.yaml"
            source.write_text(
                "qlib_init:\n  provider_uri: RESOLVE_QLIB_ASHARE_V2_AT_RUNTIME\n"
                "task: {}\n",
                encoding="utf-8",
            )
            provider = root / "native-snapshot"

            runtime = module.write_training_config_with_provider(
                source,
                provider,
                output_dir=root / "runtime-configs",
            )

            payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))
            self.assertEqual(payload["qlib_init"]["provider_uri"], str(provider))
            self.assertEqual(payload["task"], {})

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
            report_text = result_md.read_text(encoding="utf-8")
            self.assertIn("只读交易策略优化预演", report_text)
            self.assertEqual(payload["metric_basis"]["candidate_return"], "absolute_net_return")
            self.assertIn("不能直接横向比较", report_text)

    def test_phase2_dry_run_supports_default_topk_and_returns_five_candidates(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"

            module.run_report(
                phase=2,
                topk=50,
                skip_training=True,
                dry_run=True,
                output_dir=out_dir,
                timestamp="20260715_130000",
            )

            payload = json.loads(
                (out_dir / "optimization_results_latest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["results"]), 5)
            self.assertEqual(len(payload["summary"]), 5)

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
            self.assertIn("只读交易策略优化", result.stdout)


if __name__ == "__main__":
    unittest.main()
