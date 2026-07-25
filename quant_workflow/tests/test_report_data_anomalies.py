import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_data_anomalies.py"
    spec = importlib.util.spec_from_file_location("report_data_anomalies", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReportDataAnomaliesTest(unittest.TestCase):
    def test_report_records_the_candidate_merged_directory(self):
        module = load_module()
        candidate = Path("/tmp/daily-candidate/cleaned-parquet")

        report = module.build_report(
            summary_df=pd.DataFrame(),
            events_df=pd.DataFrame(),
            coverage={"files": 0, "rows": 0, "start": "", "end": ""},
            stock_csv=Path("/tmp/stocks.csv"),
            events_csv=Path("/tmp/events.csv"),
            merged_dir=candidate,
        )

        self.assertIn(str(candidate), report)

    def test_classifies_structural_price_issue_as_blocker(self):
        module = load_module()
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
                "open": [10.0, 12.0, 12.0],
                "high": [11.0, 11.0, 11.0],
                "low": [9.0, 10.0, 10.0],
                "close": [10.5, 12.0, 12.0],
                "volume": [1000, 0, 0],
            }
        )

        summary, events = module.analyze_frame("000001", df, pd.Timestamp("2024-12-31"))

        issue_types = {event["issue_type"] for event in events}
        self.assertIn("duplicate_dates", issue_types)
        self.assertIn("bad_ohlc", issue_types)
        self.assertEqual(summary["severity"], "blocker")
        self.assertEqual(summary["cleaning_action"], "exclude_symbol_from_training_until_repaired")

    def test_large_return_is_review_not_blocker(self):
        module = load_module()
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "open": [10.0, 16.0, 16.5],
                "high": [10.2, 16.5, 16.8],
                "low": [9.8, 15.8, 16.2],
                "close": [10.0, 16.0, 16.5],
                "volume": [1000, 1000, 1000],
            }
        )

        summary, events = module.analyze_frame("000002", df, pd.Timestamp("2024-12-31"))

        self.assertIn("abs_return_gt_50pct", {event["issue_type"] for event in events})
        self.assertEqual(summary["severity"], "review")
        self.assertEqual(summary["cleaning_action"], "review_and_winsorize_or_mask_event_dates")


if __name__ == "__main__":
    unittest.main()
