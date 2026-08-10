import importlib.util
import errno
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline_lgb.py"
    spec = importlib.util.spec_from_file_location("train_baseline_lgb", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainBaselineLgbTest(unittest.TestCase):
    def test_immutable_training_registry_rejects_existing_content_mismatch(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            info = {
                "run_id": "run-immutable-1",
                "status": "completed",
                "provider_identity": {"snapshot_id": "native-v2"},
            }

            path = module.write_immutable_training_run(info, report_dir=report_dir)
            self.assertEqual(
                module.write_immutable_training_run(info, report_dir=report_dir),
                path,
            )
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["status"] = "failed"
            path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "immutable_training_run_mismatch"
            ):
                module.write_immutable_training_run(info, report_dir=report_dir)

    def test_immutable_training_registry_rejects_unsafe_existing_files(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            registry = report_dir / "training_runs"
            registry.mkdir()
            info = {"run_id": "unsafe-run", "status": "completed"}
            path = registry / "unsafe-run.json"

            target = report_dir / "target.json"
            target.write_text(json.dumps(info), encoding="utf-8")
            path.symlink_to(target)
            with self.assertRaisesRegex(
                ValueError, "immutable_training_run_mismatch"
            ):
                module.write_immutable_training_run(info, report_dir=report_dir)

            path.unlink()
            path.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "immutable_training_run_mismatch"
            ):
                module.write_immutable_training_run(info, report_dir=report_dir)

            path.write_bytes(b"x" * (module.MAX_IMMUTABLE_TRAINING_RUN_BYTES + 1))
            with self.assertRaisesRegex(
                ValueError, "immutable_training_run_mismatch"
            ):
                module.write_immutable_training_run(info, report_dir=report_dir)

    def test_build_training_replay_contract_freezes_original_parameters(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "baseline.yaml"
            runtime = root / "runtime.yaml"
            strategies = root / "candidate.yaml"
            source.write_text("task: {}\n", encoding="utf-8")
            runtime.write_text(
                yaml.safe_dump(
                    {
                        "task": {
                            "dataset": {
                                "kwargs": {
                                    "segments": {
                                        "train": ["2020-01-02", "2022-12-31"],
                                        "valid": ["2023-01-01", "2023-12-31"],
                                        "test": ["2024-01-01", "2026-07-10"],
                                    }
                                }
                            },
                            "record": [
                                {
                                    "class": "PortAnaRecord",
                                    "kwargs": {
                                        "config": {
                                            "strategy": {
                                                "kwargs": {"topk": 50, "n_drop": 5}
                                            },
                                            "backtest": {
                                                "start_time": "2024-01-02",
                                                "end_time": "2026-07-10",
                                                "benchmark": "SH000300",
                                                "exchange_kwargs": {
                                                    "open_cost": 0.0005,
                                                    "close_cost": 0.0015,
                                                },
                                            },
                                        }
                                    },
                                }
                            ],
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            strategies.write_text(
                yaml.safe_dump(
                    {
                        "cost_bps": 15,
                        "strategies": [
                            {
                                "id": "baseline_daily",
                                "label": "每日 TopK 等权基线",
                                "kwargs": {
                                    "rebalance_freq": 1,
                                    "turnover_cap": 1.0,
                                    "max_drawdown_stop": 0.0,
                                },
                            }
                        ],
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            contract = module.build_training_replay_contract(
                source_config=source,
                runtime_config=runtime,
                strategy_config=strategies,
            )

        self.assertEqual(contract["contract_version"], 2)
        self.assertEqual(contract["replay_engine"], "qlib_portana")
        self.assertEqual(contract["exact_strategy_ids"], ["baseline_daily"])
        self.assertEqual(contract["original_parameters"], {
            "strategy_id": "baseline_daily",
            "start_date": "2024-01-02",
            "end_date": "2026-07-10",
            "topk": 50,
            "cost_bps": 20.0,
        })
        self.assertEqual(contract["periods"]["training"], ["2020-01-02", "2022-12-31"])
        self.assertEqual(contract["strategies"][0]["config"]["rebalance_freq"], 1)
        self.assertEqual(contract["qlib_backtest_assumptions"]["n_drop"], 5)
        self.assertEqual(contract["qlib_backtest_assumptions"]["open_cost"], 0.0005)
        self.assertEqual(contract["qlib_backtest_assumptions"]["close_cost"], 0.0015)
        self.assertRegex(contract["config_hash"], r"^[0-9a-f]{64}$")

    def test_approved_identity_resolves_exact_snapshot_without_reading_latest(self):
        module = load_module()
        pinned = Path("/quant_data/qlib/cn_data_ashare_v2/snapshots/native-v2")
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }

        with (
            patch.object(
                module,
                "resolve_approved_provider_identity",
                return_value=pinned,
                create=True,
            ) as approved_resolver,
            patch.object(
                module,
                "qlib_native_snapshot_uri",
                side_effect=AssertionError("LATEST was resolved"),
            ),
        ):
            resolved = module.resolve_provider_uri(
                "canonical", expected_provider_identity=identity
            )

        self.assertEqual(resolved, pinned)
        approved_resolver.assert_called_once_with(
            "canonical",
            identity,
            quant_data_root=module.QUANT_DATA,
        )

    def test_explicit_provider_uri_is_validated_and_used_without_reresolving_latest(self):
        module = load_module()
        pinned = Path("/quant_data/qlib/cn_data_ashare_v2/snapshots/native-v2")

        with patch.object(module, "qlib_native_snapshot_uri", return_value=pinned) as resolver:
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

        descriptor = json.loads(
            (canonical / "PROVIDER_DESCRIPTOR.json").read_text(encoding="utf-8")
        )
        self.assertEqual(descriptor["storage_mode"], "direct_parquet")
        self.assertEqual(descriptor["dataset"]["dataset_id"], "ashare_daily_qfq")
        self.assertEqual(
            set(descriptor["index_dataset"]["required_symbols"]),
            {"SH000300", "SH000905", "SH000906", "SZ399006", "SH000688"},
        )
        cleaned_descriptor = json.loads(
            (cleaned / "PROVIDER_DESCRIPTOR.json").read_text(encoding="utf-8")
        )
        self.assertEqual(cleaned_descriptor["cleaning"]["mode"], "cleaned")

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

    def test_runtime_config_contains_direct_provider_components(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "baseline.yaml"
            source.write_text(
                yaml.safe_dump({"qlib_init": {"provider_uri": "/old/provider"}}),
                encoding="utf-8",
            )
            direct_provider = base / "direct" / "provider"
            direct_provider.mkdir(parents=True)
            (direct_provider / "PROVIDER_DESCRIPTOR.json").write_text(
                "{}\n", encoding="utf-8"
            )
            direct_config = {
                "provider_uri": "/direct/provider",
                "calendar_provider": {"class": "QuantDataCalendarProvider"},
                "instrument_provider": {"class": "QuantDataInstrumentProvider"},
                "feature_provider": {"class": "QuantDataFeatureProvider"},
                "provider": {"class": "QuantDataProvider"},
            }
            with patch.object(
                module,
                "qlib_init_config_for_uri",
                return_value=direct_config,
            ):
                runtime = module.write_runtime_config_with_provider(
                    source,
                    direct_provider,
                    "canonical",
                    base / "runtime",
                    extend_dates_to_provider_latest=False,
                )

            payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))

        self.assertEqual(payload["qlib_init"]["provider_uri"], "/direct/provider")
        self.assertEqual(
            payload["qlib_init"]["feature_provider"]["class"],
            "QuantDataFeatureProvider",
        )

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

    def test_equivalence_runtime_config_preserves_registered_fixed_dates(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            provider = base / "provider"
            (provider / "calendars").mkdir(parents=True)
            (provider / "calendars" / "day.txt").write_text(
                "2026-07-23\n2026-07-24\n",
                encoding="utf-8",
            )
            source = base / "equivalence.yaml"
            source.write_text(
                yaml.safe_dump(
                    {
                        "qlib_init": {"provider_uri": "placeholder"},
                        "task": {
                            "dataset": {
                                "kwargs": {
                                    "handler": {"kwargs": {"end_time": "2024-12-31"}},
                                    "segments": {
                                        "test": ["2024-10-08", "2024-12-31"]
                                    },
                                }
                            },
                            "record": [
                                {
                                    "class": "PortAnaRecord",
                                    "kwargs": {
                                        "config": {
                                            "backtest": {
                                                "start_time": "2024-10-08",
                                                "end_time": "2024-12-31",
                                            }
                                        }
                                    },
                                }
                            ],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            runtime = module.write_runtime_config_with_provider(
                source,
                provider,
                "canonical",
                base / "runtime",
                extend_dates_to_provider_latest=False,
            )
            payload = yaml.safe_load(runtime.read_text(encoding="utf-8"))

        dataset = payload["task"]["dataset"]["kwargs"]
        backtest = payload["task"]["record"][0]["kwargs"]["config"]["backtest"]
        self.assertEqual(dataset["handler"]["kwargs"]["end_time"], "2024-12-31")
        self.assertEqual(dataset["segments"]["test"], ["2024-10-08", "2024-12-31"])
        self.assertEqual(backtest["end_time"], "2024-12-31")

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
            with self.assertRaisesRegex(RuntimeError, "task"):
                module.require_local_training_artifacts(
                    "run-1",
                    "123",
                    mlruns_dir=mlruns,
                )

            (artifacts / "task").write_bytes(b"task")
            local = module.require_local_training_artifacts(
                "run-1",
                "123",
                mlruns_dir=mlruns,
            )

        self.assertEqual(local, ["label.pkl", "pred.pkl", "task"])

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
            "task": "c" * 64,
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

    def test_publication_revalidates_provider_before_tagging_or_writing_latest(self):
        module = load_module()
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
        client = object()

        with (
            patch.object(
                module,
                "revalidate_provider_identity",
                side_effect=RuntimeError("provider changed during training"),
                create=True,
            ) as revalidate,
            patch.object(module, "tag_run_with_provider") as tag_provider,
            patch.object(module, "record_training_result") as record_result,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed during training"):
                module.publish_training_result(
                    {"run_id": "run-1", "status": "completed"},
                    provider="canonical",
                    provider_uri=Path("/tmp/native-v2"),
                    expected_provider_identity=identity,
                    source_config=Path("/tmp/source.yaml"),
                    runtime_config=Path("/tmp/runtime.yaml"),
                    client=client,
                )

        revalidate.assert_called_once()
        tag_provider.assert_not_called()
        record_result.assert_not_called()

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

    def test_experiment_file_lock_uses_cross_platform_adapter_and_unlocks_on_error(
        self,
    ):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(module, "lock_file", create=True) as lock_file,
                patch.object(module, "unlock_file", create=True) as unlock_file,
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    with module.experiment_file_lock(Path(temp_dir)):
                        raise RuntimeError("write failed")

        lock_file.assert_called_once()
        lock_handle = lock_file.call_args.args[0]
        self.assertEqual(lock_file.call_args.kwargs, {"exclusive": True})
        self.assertIn("b", lock_handle.mode)
        unlock_file.assert_called_once_with(lock_handle)

    def test_windows_file_lock_retries_until_the_lock_is_available(self):
        import _file_lock as file_lock

        class FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            def __init__(self):
                self.calls = []
                self.attempts = 0

            def locking(self, _fileno, mode, size):
                self.calls.append((mode, size))
                if mode == self.LK_NBLCK and self.attempts == 0:
                    self.attempts += 1
                    raise OSError(errno.EACCES, "lock is busy")

        fake_msvcrt = FakeMsvcrt()
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "lock"
            with lock_path.open("a+b") as handle:
                with (
                    patch.object(file_lock, "IS_WINDOWS", True),
                    patch.dict(sys.modules, {"msvcrt": fake_msvcrt}),
                    patch.object(file_lock.time, "sleep") as fake_sleep,
                ):
                    file_lock.lock_file(handle, exclusive=True)
                    file_lock.unlock_file(handle)

        self.assertEqual(
            fake_msvcrt.calls,
            [
                (FakeMsvcrt.LK_NBLCK, 1),
                (FakeMsvcrt.LK_NBLCK, 1),
                (FakeMsvcrt.LK_UNLCK, 1),
            ],
        )
        fake_sleep.assert_called_once_with(0.1)

    def test_windows_file_lock_does_not_retry_unrelated_os_errors(self):
        import _file_lock as file_lock

        class FakeMsvcrt:
            LK_NBLCK = 1

            def locking(self, _fileno, _mode, _size):
                raise OSError(errno.EBADF, "bad handle")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (Path(temp_dir) / "lock").open("a+b") as handle:
                with (
                    patch.object(file_lock, "IS_WINDOWS", True),
                    patch.dict(sys.modules, {"msvcrt": FakeMsvcrt()}),
                    patch.object(file_lock.time, "sleep") as fake_sleep,
                ):
                    with self.assertRaisesRegex(OSError, "bad handle"):
                        file_lock.lock_file(handle, exclusive=True)

        fake_sleep.assert_not_called()

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
            result_output = report_dir / "runs" / "smoke-run.json"
            latest.write_text('{"run_id":"formal-run"}\n', encoding="utf-8")

            module.record_training_result(
                {
                    "run_id": "smoke-run",
                    "status": "completed",
                    "experiment_name": "native_v2_smoke",
                },
                publish_latest=False,
                report_dir=report_dir,
                result_output=result_output,
            )

            latest_payload = json.loads(latest.read_text(encoding="utf-8"))
            isolated_payload = json.loads(result_output.read_text(encoding="utf-8"))
            history = [
                json.loads(line)
                for line in (report_dir / "experiment_runs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            immutable_run = json.loads(
                (report_dir / "training_runs" / "smoke-run.json").read_text(
                    encoding="utf-8"
                )
            )

            with self.assertRaisesRegex(
                ValueError, "immutable_training_run_mismatch"
            ):
                module.record_training_result(
                    {
                        "run_id": "smoke-run",
                        "status": "completed",
                        "experiment_name": "must-not-overwrite",
                    },
                    publish_latest=False,
                    report_dir=report_dir,
                )
            immutable_after_duplicate = json.loads(
                (report_dir / "training_runs" / "smoke-run.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(latest_payload["run_id"], "formal-run")
        self.assertEqual(isolated_payload["run_id"], "smoke-run")
        self.assertEqual(history[0]["run_id"], "smoke-run")
        self.assertEqual(immutable_run["experiment_name"], "native_v2_smoke")
        self.assertEqual(immutable_after_duplicate, immutable_run)

    def test_training_equivalence_scope_never_enters_official_result_indexes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            latest = report_dir / "latest_training_result.json"
            latest.write_text('{"run_id":"formal-run"}\n', encoding="utf-8")

            result_path = module.record_training_result(
                {
                    "run_id": "equivalence-web-run",
                    "status": "completed",
                    "experiment_name": "dsa_training_equivalence",
                },
                publish_latest=False,
                report_dir=report_dir,
                result_scope="training_equivalence",
                equivalence_channel="web",
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            latest_web = json.loads(
                (
                    report_dir
                    / "test_runs"
                    / "training_equivalence"
                    / "latest_web.json"
                ).read_text(encoding="utf-8")
            )
            formal_latest = json.loads(latest.read_text(encoding="utf-8"))
            official_registry_exists = (report_dir / "training_runs").exists()
            experiment_runs_exists = (report_dir / "experiment_runs.jsonl").exists()
            experiment_log_exists = (report_dir / "experiment_log.md").exists()

        self.assertEqual(result_path.name, "training_result.json")
        self.assertEqual(payload["result_scope"], "training_equivalence")
        self.assertTrue(payload["test_only"])
        self.assertEqual(payload["equivalence_channel"], "web")
        self.assertIn("不可用于研究或交易", payload["warning"])
        self.assertEqual(latest_web["run_id"], "equivalence-web-run")
        self.assertEqual(formal_latest["run_id"], "formal-run")
        self.assertFalse(official_registry_exists)
        self.assertFalse(experiment_runs_exists)
        self.assertFalse(experiment_log_exists)

    def test_training_equivalence_scope_rejects_official_latest_publication(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "cannot publish official latest"):
                module.record_training_result(
                    {
                        "run_id": "equivalence-web-run",
                        "status": "completed",
                        "experiment_name": "dsa_training_equivalence",
                    },
                    publish_latest=True,
                    report_dir=Path(temp_dir),
                    result_scope="training_equivalence",
                    equivalence_channel="web",
                )

    def test_candidate_child_scope_is_persisted_without_replacing_latest(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            report_dir = Path(temp_dir)
            latest = report_dir / "latest_training_result.json"
            result_output = report_dir / "candidate" / "child-result.json"
            latest.write_text('{"run_id":"formal-run"}\n', encoding="utf-8")

            result_path = module.record_training_result(
                {
                    "run_id": "candidate-child-run",
                    "status": "completed",
                    "experiment_name": "candidate-models",
                },
                publish_latest=False,
                report_dir=report_dir,
                result_output=result_output,
                result_scope="candidate_child",
            )

            isolated = json.loads(result_path.read_text(encoding="utf-8"))
            immutable = json.loads(
                (
                    report_dir
                    / "training_runs"
                    / "candidate-child-run.json"
                ).read_text(encoding="utf-8")
            )
            formal_latest = json.loads(latest.read_text(encoding="utf-8"))

        self.assertEqual(isolated["result_scope"], "candidate_child")
        self.assertEqual(immutable["result_scope"], "candidate_child")
        self.assertEqual(formal_latest["run_id"], "formal-run")

    def test_training_equivalence_config_is_bounded_and_deterministic(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "baseline_lgb_equivalence.yaml"
        )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = payload["task"]["model"]["kwargs"]
        handler = payload["task"]["dataset"]["kwargs"]["handler"]["kwargs"]
        segments = payload["task"]["dataset"]["kwargs"]["segments"]

        self.assertLessEqual(model["num_threads"], 4)
        self.assertTrue(model["deterministic"])
        self.assertEqual(model["seed"], 42)
        self.assertEqual(model["feature_fraction_seed"], 42)
        self.assertEqual(model["bagging_seed"], 42)
        self.assertEqual(model["data_random_seed"], 42)
        self.assertEqual(handler["start_time"], "2024-01-02")
        self.assertEqual(handler["end_time"], "2024-12-31")
        self.assertEqual(segments["train"], ["2024-01-02", "2024-06-28"])
        self.assertEqual(segments["valid"], ["2024-07-01", "2024-09-30"])
        self.assertEqual(segments["test"], ["2024-10-08", "2024-12-31"])

    def test_publication_forwards_isolated_result_output(self):
        module = load_module()
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
        result_output = Path("/tmp/candidate-result.json")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.yaml"
            runtime = root / "runtime.yaml"
            strategy = root / "candidate.yaml"
            source.write_text("task: {}\n", encoding="utf-8")
            runtime.write_text(
                yaml.safe_dump({"task": {"dataset": {"kwargs": {"segments": {}}}, "record": []}}),
                encoding="utf-8",
            )
            strategy.write_text("cost_bps: 15\nstrategies: []\n", encoding="utf-8")
            client = Mock()
            with (
                patch.object(
                    module,
                    "revalidate_provider_identity",
                    return_value=identity,
                    create=True,
                ),
                patch.object(module, "tag_run_with_provider"),
                patch.object(module, "record_training_result") as record_result,
                patch.object(module, "CANDIDATE_STRATEGY_CONFIG", strategy),
            ):
                module.publish_training_result(
                    {"run_id": "candidate-run", "status": "completed"},
                    provider="canonical",
                    provider_uri=Path("/tmp/native-v2"),
                    expected_provider_identity=identity,
                    source_config=source,
                    runtime_config=runtime,
                    publish_latest=False,
                    result_output=result_output,
                    client=client,
                    result_scope="candidate_child",
                )

        record_result.assert_called_once()
        self.assertFalse(record_result.call_args.kwargs["publish_latest"])
        self.assertEqual(
            record_result.call_args.kwargs["result_output"],
            result_output,
        )
        self.assertEqual(
            record_result.call_args.kwargs["result_scope"],
            "candidate_child",
        )
        client.set_tag.assert_any_call(
            "candidate-run",
            "dsa.result_scope",
            "candidate_child",
        )
        self.assertIn(
            "replay_contract",
            record_result.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
