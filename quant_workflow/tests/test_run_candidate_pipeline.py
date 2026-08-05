import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_candidate_pipeline.py"
SPEC_PATH = Path(__file__).resolve().parents[1] / "configs" / "candidate_pipeline.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("run_candidate_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prediction_frame(start: str, periods: int = 5, instruments: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range(start, periods=periods)
    symbols = [f"SH600{i:03d}" for i in range(instruments)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "instrument"])
    score = np.tile(np.arange(instruments, dtype=float), periods)
    label = score * 0.001
    return pd.DataFrame({"score": score}, index=index), pd.DataFrame({"label": label}, index=index)


def benchmark_frame(dates: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["SH000300"], dates],
        names=["instrument", "datetime"],
    )
    return pd.DataFrame({"$close": close}, index=index)


class CandidatePipelineTest(unittest.TestCase):
    def test_atomic_write_fsyncs_file_and_parent_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "candidate.json"
            with (
                patch.object(module.os, "fsync") as fsync,
                patch.object(module, "_fsync_directory", create=True) as fsync_directory,
            ):
                module._atomic_write(target, "{\"status\": \"ok\"}\n")
                written = target.read_text(encoding="utf-8")

        self.assertTrue(written)
        fsync.assert_called_once()
        fsync_directory.assert_called_once_with(target.parent)

    def test_approved_identity_resolves_pipeline_snapshot_without_reading_latest(self):
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
                "qlib_provider_uri",
                side_effect=AssertionError("LATEST was resolved"),
            ),
        ):
            resolved = module.resolve_pipeline_provider(
                "canonical", expected_provider_identity=identity
            )

        self.assertEqual(resolved, pinned)
        approved_resolver.assert_called_once_with(
            "canonical",
            identity,
            quant_data_root=module.QUANT_DATA,
        )

    def test_model_training_command_pins_the_parent_snapshot(self):
        module = load_module()
        result_output = Path("/tmp/candidate-result.json")
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }

        command = module._training_command(
            config_path=Path("/tmp/candidate.yaml"),
            experiment_name="candidate-test",
            provider="canonical",
            expected_provider_identity=identity,
            result_output=result_output,
        )

        self.assertNotIn("--provider-uri", command)
        self.assertIn("--no-publish-latest", command)
        self.assertEqual(
            command[command.index("--result-scope") + 1],
            "candidate_child",
        )
        self.assertEqual(
            Path(command[command.index("--result-output") + 1]),
            result_output,
        )
        self.assertEqual(
            json.loads(command[command.index("--provider-identity-json") + 1]),
            identity,
        )

    def test_model_training_reads_its_batch_result_instead_of_operational_latest(self):
        module = load_module()
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_config = root / "base.yaml"
            base_config.write_text("{}\n", encoding="utf-8")
            operational_latest = root / "latest_training_result.json"
            operational_latest.write_text("not candidate JSON", encoding="utf-8")

            def fake_training(command, *, cwd, check):
                self.assertEqual(cwd, str(module.WORKFLOW_ROOT))
                self.assertFalse(check)
                self.assertIn("--no-publish-latest", command)
                result_path = Path(command[command.index("--result-output") + 1])
                source_config = Path(command[command.index("--config") + 1])
                result_path.write_text(
                    json.dumps(
                        {
                            "run_id": "candidate-run",
                            "experiment_id": "candidate-experiment-id",
                            "experiment_name": "candidate-experiment",
                            "source_config": str(source_config),
                            "provider_identity": identity,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(
                    module,
                    "LATEST_TRAINING_RESULT",
                    operational_latest,
                    create=True,
                ),
                patch.object(module, "_resolve_base_config", return_value=base_config),
                patch.object(
                    module,
                    "build_candidate_config",
                    return_value={"qlib_init": {}},
                ),
                patch.object(module, "apply_common_periods"),
                patch.object(module.subprocess, "run", side_effect=fake_training),
                patch.object(module, "_tag_candidate_run") as tag_candidate,
                patch.object(
                    module,
                    "_load_artifacts",
                    return_value=("prediction", "label"),
                ),
            ):
                info, prediction, label = module._run_model_training(
                    {"id": "linear_ridge"},
                    {},
                    provider="canonical",
                    provider_uri=root / "provider",
                    provider_latest_day="2026-07-31",
                    latest_prediction_day="2026-07-30",
                    batch_id="candidate-alpha158-20260801_120000",
                    config_dir=root / "runtime_configs",
                    experiment_name="candidate-experiment",
                    expected_provider_identity=identity,
                )

            self.assertEqual(
                operational_latest.read_text(encoding="utf-8"),
                "not candidate JSON",
            )

        self.assertEqual(info["run_id"], "candidate-run")
        self.assertEqual(prediction, "prediction")
        self.assertEqual(label, "label")
        tag_candidate.assert_called_once_with(
            info,
            "candidate-alpha158-20260801_120000",
            "linear_ridge",
        )

    def test_model_result_must_match_the_parent_snapshot_identity(self):
        module = load_module()
        expected = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }

        module._require_provider_identity(expected, expected)
        changed = {**expected, "snapshot_id": "native-v2-new"}
        with self.assertRaisesRegex(RuntimeError, "different immutable provider"):
            module._require_provider_identity(changed, expected)

    def test_candidate_artifacts_are_all_validated_before_pickle_loading(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            artifacts = mlruns / "123" / "safe_run" / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "pred.pkl").write_bytes(b"prediction")
            (artifacts / "label.pkl").write_bytes(b"label")
            info = {"experiment_id": "123", "run_id": "safe_run"}

            with (
                patch.object(
                    module,
                    "validate_file_store_run_artifact",
                    side_effect=[
                        {
                            "artifact_sha256": module.hashlib.sha256(
                                b"prediction"
                            ).hexdigest()
                        },
                        module.RunProvenanceError("label hash mismatch"),
                    ],
                ) as validate,
                patch.object(
                    module.pd,
                    "read_pickle",
                    side_effect=AssertionError("unvalidated pickle was read"),
                ) as read_pickle,
                self.assertRaisesRegex(RuntimeError, "label hash mismatch"),
            ):
                module._load_artifacts(
                    info,
                    mlruns_dir=mlruns,
                    quant_data_root_path=root / "quant_data",
                )

        self.assertEqual(validate.call_count, 2)
        read_pickle.assert_not_called()

    def test_candidate_artifacts_must_match_the_batch_provider_identity(self):
        module = load_module()
        expected = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "native-v2",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            artifacts = mlruns / "123" / "safe_run" / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "pred.pkl").write_bytes(b"prediction")
            (artifacts / "label.pkl").write_bytes(b"label")
            info = {"experiment_id": "123", "run_id": "safe_run"}
            mismatched = {**expected, "snapshot_id": "another-snapshot"}

            with (
                patch.object(
                    module,
                    "validate_file_store_run_artifact",
                    side_effect=[
                        {
                            "provider_identity": expected,
                            "artifact_sha256": module.hashlib.sha256(
                                b"prediction"
                            ).hexdigest(),
                        },
                        {
                            "provider_identity": mismatched,
                            "artifact_sha256": module.hashlib.sha256(
                                b"label"
                            ).hexdigest(),
                        },
                    ],
                ),
                patch.object(
                    module.pd,
                    "read_pickle",
                    side_effect=AssertionError("cross-provider pickle was read"),
                ) as read_pickle,
                self.assertRaisesRegex(RuntimeError, "different immutable provider"),
            ):
                module._load_artifacts(
                    info,
                    mlruns_dir=mlruns,
                    quant_data_root_path=root / "quant_data",
                    expected_provider_identity=expected,
                )

        read_pickle.assert_not_called()

    def test_candidate_deserializes_the_exact_bytes_bound_to_the_validated_hash(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            artifacts = mlruns / "123" / "safe_run" / "artifacts"
            artifacts.mkdir(parents=True)
            original_pred, original_label = prediction_frame("2025-01-02")
            replacement_pred = original_pred.assign(score=-999.0)
            pred_path = artifacts / "pred.pkl"
            label_path = artifacts / "label.pkl"
            original_pred.to_pickle(pred_path)
            original_label.to_pickle(label_path)
            original_pred_bytes = pred_path.read_bytes()
            original_label_bytes = label_path.read_bytes()
            expected = {
                "provider_contract": "immutable_snapshot_v2",
                "dataset_id": "qlib_ashare_v2",
                "snapshot_id": "snapshot",
                "content_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
            }

            def validate(**kwargs):
                path = Path(kwargs["artifact_path"])
                loaded = original_pred_bytes if path.name == "pred.pkl" else original_label_bytes
                digest = module.hashlib.sha256(loaded).hexdigest()
                if path.name == "pred.pkl":
                    replacement_pred.to_pickle(path)
                return {
                    "provider_identity": expected,
                    "provider_path": str(root / "quant_data" / "provider"),
                    "artifact_sha256": digest,
                }

            with patch.object(
                module, "validate_file_store_run_artifact", side_effect=validate
            ):
                pred, label = module._load_artifacts(
                    {"experiment_id": "123", "run_id": "safe_run"},
                    mlruns_dir=mlruns,
                    quant_data_root_path=root / "quant_data",
                    expected_provider_identity=expected,
                )

        pd.testing.assert_frame_equal(pred, pd.read_pickle(io.BytesIO(original_pred_bytes)))
        pd.testing.assert_frame_equal(label, original_label)

    def test_checked_in_spec_defines_five_distinct_models_and_holdout(self):
        module = load_module()

        spec = module.load_pipeline_spec(SPEC_PATH)

        self.assertEqual(
            [item["id"] for item in spec["models"]],
            ["linear_ols", "linear_ridge", "lightgbm", "xgboost", "catboost"],
        )
        self.assertEqual(spec["selection_period"], ["2024-01-01", "2025-12-31"])
        self.assertEqual(spec["holdout_period"][0], "2026-01-01")
        self.assertEqual(len({item["id"] for item in spec["models"]}), 5)

    def test_build_candidate_config_replaces_only_model_and_keeps_common_dataset(self):
        module = load_module()
        base = {
            "qlib_init": {"provider_uri": "/provider", "region": "cn"},
            "task": {
                "model": {"class": "OldModel"},
                "dataset": {"kwargs": {"segments": {"train": ["2020-01-01", "2022-12-31"]}}},
                "record": [{"class": "SignalRecord"}],
            },
        }
        model = {
            "id": "linear_ols",
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {"estimator": "ols"},
        }

        built = module.build_candidate_config(base, model)

        self.assertEqual(built["task"]["model"]["class"], "LinearModel")
        self.assertEqual(built["task"]["model"]["kwargs"], {"estimator": "ols"})
        self.assertEqual(
            built["task"]["dataset"]["kwargs"]["segments"]["train"],
            ["2020-01-01", "2022-12-31"],
        )
        self.assertEqual(base["task"]["model"], {"class": "OldModel"})

    def test_linear_candidates_fill_and_scale_features_before_training(self):
        module = load_module()
        base = {
            "task": {
                "model": {"class": "OldModel"},
                "dataset": {"kwargs": {"handler": {"kwargs": {"instruments": "csi300"}}}},
            }
        }
        linear = {
            "id": "linear_ols",
            "class": "LinearModel",
            "module_path": "qlib.contrib.model.linear",
            "kwargs": {"estimator": "ols"},
        }

        built = module.build_candidate_config(base, linear)
        handler_kwargs = built["task"]["dataset"]["kwargs"]["handler"]["kwargs"]

        self.assertEqual(
            [item["class"] for item in handler_kwargs["infer_processors"]],
            ["RobustZScoreNorm", "Fillna"],
        )
        self.assertEqual([item["class"] for item in handler_kwargs["learn_processors"]], ["DropnaLabel"])

    def test_slice_frame_uses_only_requested_period(self):
        module = load_module()
        pred_2025, _ = prediction_frame("2025-12-29", periods=5)
        pred_2026, _ = prediction_frame("2026-01-05", periods=3)
        pred = pd.concat([pred_2025, pred_2026])

        selected = module.slice_frame(pred, "2026-01-01", "2026-12-31")

        dates = selected.index.get_level_values("datetime")
        self.assertGreaterEqual(dates.min(), pd.Timestamp("2026-01-01"))
        self.assertEqual(dates.max(), pd.Timestamp("2026-01-07"))

    def test_rank_models_uses_selection_metrics_not_holdout_results(self):
        module = load_module()
        candidates = [
            {
                "id": "selection_winner",
                "status": "completed",
                "selection": {"information_ratio": 1.1, "annual_excess_return": 0.10, "rank_ic": 0.05},
                "holdout": {"information_ratio": -9.0, "annual_excess_return": -0.90},
            },
            {
                "id": "holdout_winner",
                "status": "completed",
                "selection": {"information_ratio": 0.7, "annual_excess_return": 0.08, "rank_ic": 0.04},
                "holdout": {"information_ratio": 9.0, "annual_excess_return": 0.90},
            },
        ]

        ranked = module.rank_models(candidates, top_n=1)

        self.assertEqual([item["id"] for item in ranked], ["selection_winner"])

    def test_pipeline_status_is_not_completed_when_any_model_failed(self):
        module = load_module()
        candidates = [
            {"id": "ok", "status": "completed"},
            {"id": "failed", "status": "failed", "error": "training failed"},
        ]

        self.assertEqual(module.pipeline_status(candidates), "failed")

    def test_incomplete_pipeline_cannot_report_a_passing_holdout_verdict(self):
        module = load_module()
        champion = {
            "holdout": {
                "annual_excess_return": 0.30,
                "information_ratio": 2.0,
                "max_drawdown": -0.05,
            }
        }

        verdict = module.holdout_verdict(champion, pipeline_complete=False)

        self.assertEqual(verdict["status"], "unavailable_incomplete_pipeline")

    def test_provider_dates_distinguish_data_prediction_and_evaluable_label_days(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir)
            (provider / "calendars").mkdir(parents=True)
            (provider / "calendars" / "day.txt").write_text(
                "2026-07-09\n2026-07-10\n2026-07-13\n2026-07-14\n",
                encoding="utf-8",
            )

            latest, prediction_day, label_day = module.provider_days(provider)

        self.assertEqual(latest, "2026-07-14")
        self.assertEqual(prediction_day, "2026-07-13")
        self.assertEqual(label_day, "2026-07-10")

    def test_benchmark_return_matches_alpha158_label_horizon(self):
        module = load_module()
        dates = pd.bdate_range("2026-01-02", periods=4)
        close = pd.Series([100.0, 101.0, 103.0, 106.0], index=dates)

        result = module.benchmark_label_returns(close)

        self.assertAlmostEqual(result.iloc[0], 103.0 / 101.0 - 1.0)
        self.assertAlmostEqual(result.iloc[1], 106.0 / 103.0 - 1.0)
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_market_risk_features_do_not_use_future_benchmark_prices(self):
        module = load_module()
        dates = pd.bdate_range("2025-01-02", periods=180)
        close = [100.0 + index + ((index % 7) - 3) * 0.1 for index in range(len(dates))]
        changed_future = close.copy()
        changed_future[-2] = close[-2] * 1.5
        changed_future[-1] = close[-1] * 0.7
        frames = iter(
            [
                benchmark_frame(dates, close),
                benchmark_frame(dates, changed_future),
            ]
        )
        fake_qlib = ModuleType("qlib")
        fake_qlib.__path__ = []
        fake_qlib.init = lambda **_: None
        fake_qlib_data = ModuleType("qlib.data")
        fake_qlib_data.D = SimpleNamespace(
            features=lambda *_args, **_kwargs: next(frames)
        )

        with patch.dict(
            "sys.modules",
            {"qlib": fake_qlib, "qlib.data": fake_qlib_data},
        ):
            original = module.load_market_data(
                Path("/unused/provider"),
                str(dates[0].date()),
                str(dates[-1].date()),
            )
            altered = module.load_market_data(
                Path("/unused/provider"),
                str(dates[0].date()),
                str(dates[-1].date()),
            )

        decision_day = dates[-3]
        self.assertNotAlmostEqual(
            original.loc[decision_day, "benchmark_return"],
            altered.loc[decision_day, "benchmark_return"],
        )
        self.assertAlmostEqual(
            original.loc[decision_day, "volatility_20"],
            altered.loc[decision_day, "volatility_20"],
        )
        self.assertAlmostEqual(
            original.loc[decision_day, "vol_percentile"],
            altered.loc[decision_day, "vol_percentile"],
        )

    def test_dry_run_writes_five_models_to_isolated_test_namespace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "reports"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--output-dir",
                    str(out_dir),
                    "--timestamp",
                    "20260715_170000",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            test_dir = out_dir / "test_runs" / "candidate_pipeline"
            report = json.loads(
                (test_dir / "candidate_pipeline_test_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            timestamped = test_dir / "candidate_pipeline_test_20260715_170000.json"
            markdown = (
                test_dir / "candidate_pipeline_test_latest.md"
            ).read_text(encoding="utf-8")
            progress = json.loads(
                (test_dir / "candidate_pipeline_test_progress_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(timestamped.exists())
            self.assertFalse((out_dir / "candidate_pipeline_latest.json").exists())
            self.assertFalse((out_dir / "candidate_pipeline_latest.md").exists())
            self.assertFalse(
                (out_dir / "candidate_pipeline_progress_latest.json").exists()
            )

        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["dry_run"])
        self.assertEqual(len(report["models"]), 5)
        self.assertEqual(len(report["selected_models"]), 2)
        self.assertEqual(len(report["strategy_evaluations"]), 10)
        self.assertTrue(
            all(row.get("strategy_config") for row in report["strategy_evaluations"])
        )
        self.assertTrue(
            all(
                model["backtest_eligibility"]["eligible"] is False
                and "run_id" in model["backtest_eligibility"]["missing"]
                for model in report["models"]
            )
        )
        self.assertEqual(report["selection_basis"], "selection_period_only")
        self.assertEqual(report["backtest_contract"]["contract_version"], 2)
        self.assertEqual(
            report["backtest_contract"]["replay_engine"],
            "dsa_optimized_strategy",
        )
        self.assertEqual(
            report["backtest_contract"]["replay_capability"],
            "custom_only",
        )
        self.assertEqual(report["backtest_contract"]["exact_strategy_ids"], [])
        self.assertEqual(
            report["backtest_contract"]["original_parameters"],
            {
                "start_date": report["periods"]["selection"][0],
                "end_date": report["periods"]["selection"][1],
                "topk": 50,
                "cost_bps": 15.0,
            },
        )
        self.assertRegex(
            report["backtest_contract"]["config_hash"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            report["provider_identity"]["provider_contract"],
            "path_only_override",
        )
        self.assertFalse(report["experiment_notebook_appended"])
        self.assertIn("candidate", report["experiment_notebook_policy"])
        self.assertEqual(progress["stage"], "completed")
        self.assertTrue(progress["dry_run"])
        self.assertIn("测试数据，不可用于研究或交易", markdown)
        self.assertIn("不自动下单", markdown)
        self.assertIn("2026 留出期不参与挑选", markdown)


if __name__ == "__main__":
    unittest.main()
