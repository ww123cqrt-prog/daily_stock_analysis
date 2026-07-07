import importlib.util
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
