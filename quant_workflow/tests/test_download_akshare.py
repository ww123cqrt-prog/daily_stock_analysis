import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_akshare.py"
    spec = importlib.util.spec_from_file_location("download_akshare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DownloadAkShareTest(unittest.TestCase):
    def test_fetch_one_uses_qfq_daily_hist_and_normalizes_schema(self):
        module = load_module()
        calls = []

        def stock_zh_a_hist(**kwargs):
            calls.append(kwargs)
            return pd.DataFrame(
                {
                    "日期": ["2026-07-08"],
                    "开盘": [10.0],
                    "收盘": [10.2],
                    "最高": [10.5],
                    "最低": [9.9],
                    "成交量": [123],
                    "成交额": [123456.7],
                    "换手率": [0.75],
                    "涨跌幅": [1.5],
                }
            )

        original = sys.modules.get("akshare")
        sys.modules["akshare"] = types.SimpleNamespace(stock_zh_a_hist=stock_zh_a_hist)
        try:
            frame = module.fetch_one("000001", "2026-07-01", "2026-07-08", timeout=8)
        finally:
            if original is None:
                sys.modules.pop("akshare", None)
            else:
                sys.modules["akshare"] = original

        self.assertEqual(calls[0]["symbol"], "000001")
        self.assertEqual(calls[0]["period"], "daily")
        self.assertEqual(calls[0]["adjust"], "qfq")
        self.assertEqual(calls[0]["start_date"], "20260701")
        self.assertEqual(frame["volume"].tolist(), [12300])
        self.assertEqual(frame["turn"].tolist(), [0.75])


if __name__ == "__main__":
    unittest.main()
