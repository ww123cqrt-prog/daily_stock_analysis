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
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
    spec = importlib.util.spec_from_file_location("train_baseline_lgb", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainBaselineLgbTest(unittest.TestCase):
    def test_resolves_provider_uris_under_quant_data_root_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
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
                        "spec=importlib.util.spec_from_file_location('train_baseline_lgb', path); "
                        "module=importlib.util.module_from_spec(spec); "
                        "spec.loader.exec_module(module); "
                        "print(json.dumps({"
                        "'canonical': str(module.resolve_provider_uri('canonical')), "
                        "'cleaned': str(module.resolve_provider_uri('cleaned'))"
                        "}))"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["canonical"], str(quant_data / "qlib" / "cn_data_baostock_jqdata"))
        self.assertEqual(payload["cleaned"], str(quant_data / "qlib" / "cn_data_baostock_jqdata_cleaned"))

    def test_resolves_canonical_and_cleaned_provider_uris(self):
        module = load_module()

        canonical = module.resolve_provider_uri("canonical")
        cleaned = module.resolve_provider_uri("cleaned")

        self.assertTrue(str(canonical).endswith("cn_data_baostock_jqdata"))
        self.assertTrue(str(cleaned).endswith("cn_data_baostock_jqdata_cleaned"))

    def test_rejects_unknown_provider_name(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.resolve_provider_uri("remote-shell")

    def test_writes_runtime_config_with_selected_provider_without_mutating_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "baseline.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "qlib_init": {"provider_uri": "/old/provider", "region": "cn"},
                        "task": {"model": {"class": "LGBModel"}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            runtime = module.write_runtime_config_with_provider(
                source,
                Path("/new/provider"),
                "cleaned",
                base / "runtime",
            )

            source_payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            runtime_payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))

        self.assertEqual(source_payload["qlib_init"]["provider_uri"], "/old/provider")
        self.assertEqual(runtime_payload["qlib_init"]["provider_uri"], "/new/provider")
        self.assertIn("cleaned", runtime.name)

    def test_builds_csi300_lines_from_provider_instrument_ranges(self):
        module = load_module()
        ranges = {
            "SZ000001": ("2008-01-02", "2026-07-03"),
            "SH600000": ("2009-01-05", "2026-07-03"),
        }

        lines = module.build_csi300_lines(["000001", "600000", "430001"], ranges)

        self.assertEqual(
            lines,
            [
                "SZ000001\t2008-01-02\t2026-07-03",
                "SH600000\t2009-01-05\t2026-07-03",
            ],
        )

    def test_ensure_csi300_instruments_writes_selected_provider_file(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir)
            instruments_dir = provider / "instruments"
            instruments_dir.mkdir(parents=True)
            (instruments_dir / "all.txt").write_text(
                "SZ000001\t2008-01-02\t2026-07-03\n"
                "SH600000\t2009-01-05\t2026-07-03\n",
                encoding="utf-8",
            )

            module.ensure_csi300_instruments(provider, csi300_symbols=["000001", "600000"])

            content = (instruments_dir / "csi300.txt").read_text(encoding="utf-8")

        self.assertIn("SZ000001\t2008-01-02\t2026-07-03", content)
        self.assertIn("SH600000\t2009-01-05\t2026-07-03", content)

    def test_append_experiment_record_writes_jsonl_and_marked_markdown(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            info = {
                "run_id": "cleaned-run-1",
                "status": "completed",
                "experiment_name": "baseline_lgb_alpha158_cleaned",
                "provider": "cleaned",
                "provider_uri": "/tmp/qlib_cleaned",
                "source_config": "/tmp/baseline.yaml",
                "metrics": {
                    "1day.excess_return_with_cost.annualized_return": 0.198637,
                    "1day.excess_return_with_cost.information_ratio": 1.705515,
                    "1day.excess_return_with_cost.max_drawdown": -0.091497,
                },
                "params": {
                    "model.class": "LGBModel",
                    "record": "[{'class': 'SignalRecord'}, {'class': 'PortAnaRecord'}]",
                    "dataset.kwargs.segments.train": "['2020-01-02', '2022-12-31']",
                    "dataset.kwargs.segments.test": "['2024-01-01', '2026-07-01']",
                },
            }

            appended = module.append_experiment_record(info, report_dir=report_dir)
            runs = [json.loads(line) for line in (report_dir / "experiment_runs.jsonl").read_text(encoding="utf-8").splitlines()]
            log_text = (report_dir / "experiment_log.md").read_text(encoding="utf-8")
            lock_exists = (report_dir / ".experiment.lock").exists()

        self.assertTrue(appended)
        self.assertTrue(lock_exists)
        self.assertEqual(runs[0]["run_id"], "cleaned-run-1")
        self.assertEqual(runs[0]["provider"], "cleaned")
        self.assertEqual(runs[0]["qlib_provider"], "/tmp/qlib_cleaned")
        self.assertIn("<!-- experiment-run:cleaned-run-1 -->", log_text)
        self.assertIn("baseline_lgb_alpha158_cleaned", log_text)


if __name__ == "__main__":
    unittest.main()
