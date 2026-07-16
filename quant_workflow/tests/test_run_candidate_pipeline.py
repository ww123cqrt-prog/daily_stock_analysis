import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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


class CandidatePipelineTest(unittest.TestCase):
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

    def test_dry_run_writes_five_models_ten_strategy_rows_and_latest_alias(self):
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
            report = json.loads((out_dir / "candidate_pipeline_latest.json").read_text(encoding="utf-8"))
            timestamped = out_dir / "candidate_pipeline_20260715_170000.json"
            markdown = (out_dir / "candidate_pipeline_latest.md").read_text(encoding="utf-8")
            self.assertTrue(timestamped.exists())

        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["dry_run"])
        self.assertEqual(len(report["models"]), 5)
        self.assertEqual(len(report["selected_models"]), 2)
        self.assertEqual(len(report["strategy_evaluations"]), 10)
        self.assertEqual(report["selection_basis"], "selection_period_only")
        self.assertFalse(report["experiment_notebook_appended"])
        self.assertIn("candidate", report["experiment_notebook_policy"])
        self.assertIn("不自动下单", markdown)
        self.assertIn("2026 留出期不参与挑选", markdown)


if __name__ == "__main__":
    unittest.main()
