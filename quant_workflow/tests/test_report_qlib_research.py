import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKFLOW_ROOT / "scripts" / "report_qlib_research.py"


class FakeImportanceModel:
    def get_feature_importance(self):
        return pd.Series({"RESI5": 3.0, "WVMA5": 1.5, "ROC5": 2.0})


def load_module():
    spec = importlib.util.spec_from_file_location("report_qlib_research", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_run(root: Path, run_id: str = "safe_run") -> Path:
    run_dir = root / "mlruns" / "123" / run_id
    artifacts = run_dir / "artifacts"
    portfolio = artifacts / "portfolio_analysis"
    metrics = run_dir / "metrics"
    params = run_dir / "params"
    portfolio.mkdir(parents=True)
    metrics.mkdir()
    params.mkdir()
    (run_dir.parent / "meta.yaml").write_text("name: baseline_lgb_alpha158_cleaned\n", encoding="utf-8")
    (run_dir / "meta.yaml").write_text("status: FINISHED\nstart_time: 1783200000000\n", encoding="utf-8")
    (params / "model.class").write_text("LGBModel\n", encoding="utf-8")
    (params / "dataset.kwargs.segments.train").write_text("('2020-01-02', '2023-12-29')\n", encoding="utf-8")
    (metrics / "l2.train").write_text("0 0.010 0\n", encoding="utf-8")
    (metrics / "l2.valid").write_text("0 0.014 0\n", encoding="utf-8")
    (metrics / "1day.excess_return_with_cost.annualized_return").write_text("0 0.08 0\n", encoding="utf-8")
    pd.to_pickle(FakeImportanceModel(), artifacts / "params.pkl")
    write_prediction_and_label(artifacts)
    write_portfolio_artifacts(portfolio)
    return run_dir


def write_prediction_and_label(artifacts: Path) -> None:
    rows = []
    scores = []
    labels = []
    for date in pd.to_datetime(["2026-07-01", "2026-07-02"]):
        for idx in range(10):
            rows.append((date, f"SH6000{idx:02d}"))
            score = idx / 10
            scores.append(score)
            labels.append(score * 0.02 - 0.005)
    index = pd.MultiIndex.from_tuples(rows, names=["datetime", "instrument"])
    pd.DataFrame({"score": scores}, index=index).to_pickle(artifacts / "pred.pkl")
    pd.DataFrame({"LABEL0": labels}, index=index).to_pickle(artifacts / "label.pkl")


def write_portfolio_artifacts(portfolio: Path) -> None:
    dates = pd.bdate_range("2026-06-01", periods=8, name="datetime")
    report = pd.DataFrame(
        {
            "return": [0.01, -0.02, 0.015, 0.004, -0.01, 0.02, 0.003, 0.005],
            "bench": [0.006, -0.012, 0.010, 0.002, -0.006, 0.012, 0.001, 0.003],
            "turnover": [0.35] * 8,
            "cost": [0.0002] * 8,
            "account": [1000000] * 8,
            "cash": [100000] * 8,
            "value": [900000] * 8,
        },
        index=dates,
    )
    report.to_pickle(portfolio / "report_normal_1day.pkl")
    pd.DataFrame({"ffr": [0.9] * 8, "pa": [1.0] * 8}, index=dates).to_pickle(
        portfolio / "indicators_normal_1day.pkl"
    )
    positions = {
        dates[-2]: {"position": {"SH600000": {"weight": 0.04, "amount": 100, "price": 10, "count_day": 2}}},
        dates[-1]: {
            "position": {
                "SH600000": {"weight": 0.06, "amount": 100, "price": 10, "count_day": 3},
                "SZ000001": {"weight": 0.03, "amount": 200, "price": 8, "count_day": 1},
            }
        },
    }
    pd.to_pickle(positions, portfolio / "positions_normal_1day.pkl")


class ReportQlibResearchTest(unittest.TestCase):
    def test_find_run_dir_uses_latest_training_and_rejects_unsafe_run_id(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_run(root, "safe_run")
            latest = root / "latest_training_result.json"
            latest.write_text(json.dumps({"run_id": "safe_run"}), encoding="utf-8")

            found = module.find_run_dir(latest_training_path=latest, mlruns_dir=root / "mlruns")

            self.assertEqual(found, run_dir)
            with self.assertRaises(ValueError):
                module.find_run_dir("../unsafe", latest_training_path=latest, mlruns_dir=root / "mlruns")

    def test_model_explainability_and_factor_reports_are_read_only(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = write_run(root)
            out_dir = root / "reports"

            explain = module.write_report(
                "model_explainability",
                module.model_explainability(run_dir, topk=2),
                out_dir,
            )
            factor = module.factor_analysis(run_dir, topk=2)
            markdown = (out_dir / "qlib_research_model_explainability_latest.md").read_text(encoding="utf-8")
            serialized = json.loads((out_dir / "qlib_research_model_explainability_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(explain["mode"], "read_only_qlib_research")
        self.assertEqual(serialized["section"], "model_explainability")
        self.assertEqual([item["feature"] for item in explain["feature_importance"]["top"]], ["RESI5", "ROC5"])
        self.assertEqual(factor["score_factor"]["daily_count"], 2)
        self.assertGreater(factor["score_factor"]["mean_rank_ic"], 0.9)
        self.assertIn("read-only", markdown)
        self.assertIn("不自动下单", markdown)

    def test_backtest_detail_and_risk_model_use_portfolio_artifacts(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = write_run(Path(temp_dir))

            detail = module.backtest_detail(run_dir)
            risk = module.risk_model(run_dir)

        self.assertEqual(detail["metrics"]["strategy"]["days"], 8)
        self.assertEqual(detail["positions"]["holding_count"], 2)
        self.assertEqual(detail["trade_activity"]["event_days"], 2)
        self.assertEqual(risk["constraint_checks"]["max_single_weight_5pct"], "breach")
        self.assertIn("Review turnover cap", " ".join(risk["recommendations"]))

    def test_model_comparison_compares_recent_local_runs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_run(root, "run_a")
            write_run(root, "run_b")

            comparison = module.model_comparison(mlruns_dir=root / "mlruns", max_runs=2)

        self.assertEqual(comparison["status"], "available")
        self.assertEqual(comparison["run_count"], 2)
        self.assertIn("1day.excess_return_with_cost.annualized_return", comparison["best_by_metric"])
        self.assertEqual(len(comparison["prediction_correlations"]), 1)

    def test_rolling_training_plan_reads_provider_calendar(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider = root / "qlib_provider"
            (provider / "calendars").mkdir(parents=True)
            dates = pd.bdate_range("2020-01-02", "2026-07-03")
            (provider / "calendars" / "day.txt").write_text(
                "\n".join(str(date.date()) for date in dates),
                encoding="utf-8",
            )
            config = root / "baseline.yaml"
            config.write_text(
                "task:\n  dataset:\n    kwargs:\n      segments:\n        train: [2020-01-02, 2023-12-29]\n",
                encoding="utf-8",
            )

            plan = module.rolling_training_plan(
                provider_uri=provider,
                config_path=config,
                windows=2,
                train_years=3,
                valid_months=6,
                test_months=1,
                step_months=1,
            )

        self.assertEqual(plan["status"], "available")
        self.assertEqual(plan["calendar_days"], len(dates))
        self.assertEqual(len(plan["windows"]), 2)
        self.assertEqual(plan["windows"][0]["window"], 1)


if __name__ == "__main__":
    unittest.main()
