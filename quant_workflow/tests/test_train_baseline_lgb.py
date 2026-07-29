import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
    spec = importlib.util.spec_from_file_location("train_baseline_lgb", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainBaselineLgbTest(unittest.TestCase):
    def test_explicit_provider_uri_is_validated_and_used_without_reresolving_latest(self):
        module = load_module()
        pinned = Path("/quant_data/qlib/cn_data_ashare_v2/snapshots/native-v2")

        with patch.object(module, "qlib_provider_uri", return_value=pinned) as resolver:
            resolved = module.resolve_provider_uri("canonical", pinned)

        self.assertEqual(resolved, pinned)
        resolver.assert_called_once_with(
            "canonical",
            snapshot_override=pinned,
        )

    def test_rejects_legacy_provider_environment_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            canonical = quant_data / "qlib" / "cn_data_baostock_jqdata"
            cleaned = quant_data / "qlib" / "cn_data_baostock_jqdata_cleaned"
            canonical.mkdir(parents=True)
            cleaned.mkdir(parents=True)
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
            env = os.environ.copy()
            env["QUANT_DATA_ROOT"] = str(quant_data)
            env["QLIB_PROVIDER_URI"] = str(canonical)
            env["QLIB_CLEANED_PROVIDER_URI"] = str(cleaned)

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
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legacy compatibility providers are rejected", result.stderr)

    def test_resolves_canonical_and_cleaned_provider_uris(self):
        module = load_module()

        canonical = module.resolve_provider_uri("canonical")
        cleaned = module.resolve_provider_uri("cleaned")

        manifest = json.loads(
            (canonical / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["dataset_id"], "qlib_ashare_v2")
        self.assertTrue(manifest["benchmark_complete"])
        cleaned_manifest = json.loads(
            (cleaned / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(cleaned_manifest["dataset_id"], "qlib_ashare_cleaned_v2")
        self.assertTrue(cleaned_manifest["benchmark_complete"])

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

    def test_runtime_config_extends_dates_to_provider_latest_calendar_day(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            provider = base / "provider"
            (provider / "calendars").mkdir(parents=True)
            (provider / "calendars" / "day.txt").write_text(
                "2026-07-08\n2026-07-09\n",
                encoding="utf-8",
            )
            source = base / "baseline.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "qlib_init": {"provider_uri": "/old/provider", "region": "cn"},
                        "task": {
                            "dataset": {
                                "kwargs": {
                                    "handler": {"kwargs": {"end_time": "2026-07-03"}},
                                    "segments": {"test": ["2024-01-01", "2026-07-01"]},
                                }
                            },
                            "record": [
                                {"class": "SignalRecord"},
                                {
                                    "class": "PortAnaRecord",
                                    "config": {"backtest": {"end_time": "2026-07-01"}},
                                },
                            ],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            runtime = module.write_runtime_config_with_provider(source, provider, "cleaned", base / "runtime")

            runtime_payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))
            source_payload = yaml.safe_load(source.read_text(encoding="utf-8"))

        task = runtime_payload["task"]
        self.assertEqual(task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"], "2026-07-09")
        self.assertEqual(task["dataset"]["kwargs"]["segments"]["test"], ["2024-01-01", "2026-07-08"])
        self.assertEqual(task["record"][1]["kwargs"]["config"]["backtest"]["end_time"], "2026-07-08")
        self.assertEqual(source_payload["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"], "2026-07-03")

    def test_runtime_config_keeps_portana_config_under_qlib_kwargs(self):
        module = load_module()
        payload = {
            "task": {
                "record": [
                    {"class": "SignalRecord"},
                    {
                        "class": "PortAnaRecord",
                        "config": {"backtest": {"end_time": "2026-07-01"}},
                    },
                ]
            }
        }

        module.update_runtime_config_dates(payload, "2026-07-09", backtest_end_day="2026-07-08")

        record = payload["task"]["record"][1]
        self.assertEqual(record["kwargs"]["config"]["backtest"]["end_time"], "2026-07-08")

    def test_runtime_config_normalizes_portana_config_even_without_calendar_day(self):
        module = load_module()
        payload = {
            "task": {
                "record": [
                    {
                        "class": "PortAnaRecord",
                        "config": {"backtest": {"end_time": "2026-07-01"}},
                    },
                ]
            }
        }

        module.update_runtime_config_dates(payload, "", backtest_end_day="")

        record = payload["task"]["record"][0]
        self.assertEqual(record["kwargs"]["config"], {"backtest": {"end_time": "2026-07-01"}})

    def test_repairs_mlruns_artifact_locations_to_active_root(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            mlruns = Path(temp_dir) / "mlruns"
            exp_dir = mlruns / "123"
            exp_dir.mkdir(parents=True)
            meta = exp_dir / "meta.yaml"
            meta.write_text(
                yaml.safe_dump(
                    {
                        "experiment_id": "123",
                        "name": "baseline",
                        "artifact_location": "file:///old/quant_workflow/mlruns/123",
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            fixed = module.ensure_mlruns_artifact_locations(mlruns)

            payload = yaml.safe_load(meta.read_text(encoding="utf-8"))
        self.assertEqual(fixed, 1)
        self.assertEqual(payload["artifact_location"], (mlruns / "123").resolve().as_uri())

    def test_required_training_artifacts_must_exist_in_active_mlruns(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            artifacts = mlruns / "123" / "run-1" / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "pred.pkl").write_bytes(b"prediction")

            with self.assertRaisesRegex(RuntimeError, "label.pkl"):
                module.require_local_training_artifacts(
                    "run-1",
                    "123",
                    mlruns_dir=mlruns,
                )

            (artifacts / "label.pkl").write_bytes(b"label")
            local = module.require_local_training_artifacts(
                "run-1",
                "123",
                mlruns_dir=mlruns,
            )

        self.assertEqual(local, ["label.pkl", "pred.pkl"])

    def test_local_run_artifacts_returns_empty_for_missing_run(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                module.local_run_artifacts("missing-run", "123", mlruns_dir=Path(temp_dir) / "mlruns"),
                [],
            )

    def test_training_records_hashes_after_artifacts_are_local(self):
        module = load_module()
        client = object()
        expected = {
            "pred.pkl": "a" * 64,
            "label.pkl": "b" * 64,
        }

        with patch.object(
            module,
            "tag_run_artifact_hashes",
            return_value=expected,
        ) as tag_hashes:
            result = module.tag_training_run_artifacts(
                "run-1",
                mlruns_dir=Path("/tmp/mlruns"),
                client=client,
            )

        self.assertEqual(result, expected)
        tag_hashes.assert_called_once_with(
            client,
            "run-1",
            mlruns_dir=Path("/tmp/mlruns"),
        )

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

    def test_load_csi300_symbols_uses_local_quant_data_cache_without_network_project(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            stock_list = Path(temp_dir) / "stock_lists.json"
            stock_list.write_text(
                json.dumps({"csi300": ["000001", "600000"]}),
                encoding="utf-8",
            )
            module.ASTOCK_PROJECT = Path(temp_dir) / "missing-network-project"

            symbols = module.load_csi300_symbols(stock_list)

        self.assertEqual(symbols, ["000001", "600000"])

    def test_ensure_csi300_instruments_refuses_to_mutate_provider(self):
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

            with self.assertRaisesRegex(FileNotFoundError, "rebuild qlib_ashare_v2"):
                module.ensure_csi300_instruments(
                    provider, csi300_symbols=["000001", "600000"]
                )

            self.assertFalse((instruments_dir / "csi300.txt").exists())

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

    def test_record_training_result_can_preserve_latest_for_smoke_run(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            latest = report_dir / "latest_training_result.json"
            latest.write_text('{"run_id":"formal-run"}\n', encoding="utf-8")

            module.record_training_result(
                {
                    "run_id": "smoke-run",
                    "status": "completed",
                    "experiment_name": "native_v2_smoke",
                },
                publish_latest=False,
                report_dir=report_dir,
            )

            latest_payload = json.loads(latest.read_text(encoding="utf-8"))
            history = [
                json.loads(line)
                for line in (report_dir / "experiment_runs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(latest_payload["run_id"], "formal-run")
        self.assertEqual(history[0]["run_id"], "smoke-run")


if __name__ == "__main__":
    unittest.main()
