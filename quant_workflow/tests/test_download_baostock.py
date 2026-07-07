import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_baostock.py"
    spec = importlib.util.spec_from_file_location("download_baostock", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DownloadBaostockTest(unittest.TestCase):
    def test_loads_symbols_file_with_comments_and_suffixes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            symbols_file = Path(tmp) / "symbols.txt"
            symbols_file.write_text(
                "\n".join(
                    [
                        "# refresh target symbols",
                        "000001.XSHE",
                        "600519",
                        "SH688001",
                        "000001.XSHE",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            symbols = module.load_symbols_file(symbols_file)

        self.assertEqual(symbols, ["000001", "600519", "688001"])

    def test_filters_codes_in_symbols_file_order(self):
        module = load_module()
        all_codes = [
            ("000001", "sz.000001"),
            ("600519", "sh.600519"),
            ("688001", "sh.688001"),
        ]

        filtered = module.filter_codes_by_symbols(all_codes, ["600519", "000001"])

        self.assertEqual(filtered, [("600519", "sh.600519"), ("000001", "sz.000001")])

    def test_incremental_start_uses_existing_last_date_minus_overlap(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000001.parquet"
            pd.DataFrame({"date": ["2026-06-20", "2026-06-24"]}).to_parquet(path, index=False)

            start = module.incremental_start_for_file(path, "2008-01-01", overlap_days=7)

        self.assertEqual(start, "2026-06-17")

    def test_merge_existing_frame_overwrites_overlap_with_fresh_rows(self):
        module = load_module()
        existing = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-20", "2026-06-21"]),
                "close": [10.0, 11.0],
            }
        )
        fresh = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-06-21", "2026-06-22"]),
                "close": [12.0, 13.0],
            }
        )

        merged = module.merge_existing_frame(existing, fresh)

        self.assertEqual(merged["close"].tolist(), [10.0, 12.0, 13.0])


if __name__ == "__main__":
    unittest.main()
