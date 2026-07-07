import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_baostock_refresh.py"
    spec = importlib.util.spec_from_file_location("prepare_baostock_refresh", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareBaostockRefreshTest(unittest.TestCase):
    def test_normalizes_numeric_symbol_with_leading_zero_padding(self):
        module = load_module()

        self.assertEqual(module.normalize_symbol(18), "000018")

    def test_selects_symbols_ending_before_target_date(self):
        module = load_module()
        summary = pd.DataFrame(
            [
                {"symbol": "000001", "end": "2026-06-24", "severity": "warning"},
                {"symbol": "600519", "end": "2025-06-11", "severity": "warning"},
                {"symbol": "000002", "end": "2026-07-03", "severity": "ok"},
                {"symbol": "000003", "end": "", "severity": "blocker"},
            ]
        )

        selected = module.select_refresh_symbols(summary, pd.Timestamp("2026-07-03"))

        self.assertEqual(list(selected["symbol"]), ["000003", "600519", "000001"])

    def test_writes_empty_plan_when_anomaly_report_is_missing(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "reports"
            missing = Path(tmp) / "missing_anomaly_stocks.csv"
            argv = [
                "prepare_baostock_refresh.py",
                "--anomaly-stocks",
                str(missing),
                "--out-dir",
                str(out_dir),
                "--target-end",
                "2026-07-03",
                "--limit",
                "3",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            plan = out_dir / "baostock_refresh_plan_latest.md"
            symbols = out_dir / "baostock_refresh_symbols_latest.csv"

            self.assertTrue(plan.exists())
            self.assertTrue(symbols.exists())
            text = plan.read_text(encoding="utf-8")
            self.assertIn("Baostock Refresh Plan", text)
            self.assertIn("Symbols to refresh: 0", text)
            self.assertIn("Run `report_data_anomalies.py` first", text)


if __name__ == "__main__":
    unittest.main()
