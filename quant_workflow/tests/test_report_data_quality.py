import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_data_quality.py"
    spec = importlib.util.spec_from_file_location("report_data_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_prices(directory: Path, symbol: str) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=3),
            "open": [10.0, 10.2, 10.4],
            "high": [10.1, 10.3, 10.5],
            "low": [9.9, 10.1, 10.3],
            "close": [10.0, 10.2, 10.4],
            "volume": [1000, 1100, 1200],
        }
    )
    frame.to_parquet(directory / f"{symbol}.parquet", index=False)


class ReportDataQualityTest(unittest.TestCase):
    def test_incomplete_benchmark_is_a_training_readiness_blocker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text("SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8")
            feature = provider / "features" / "sh000300" / "close.day.bin"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(np.array([0.0, 3900.0], dtype="<f4").tobytes())

            stats = module.qlib_provider_stats(provider)
            ready, detail = module.build_training_readiness(
                stats,
                {f"{index:06d}" for index in range(100)},
                {"duplicate_dates": 0, "null_dates": 0, "missing_columns": 0, "bad_ohlc_rows": 0},
            )

        self.assertFalse(stats["benchmark_complete"])
        self.assertFalse(ready)
        self.assertIn("benchmark", detail["summary"].lower())
    def test_scan_merged_files_respects_sample_limit(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_prices(data_dir, "000001")
            write_prices(data_dir, "000002")
            write_prices(data_dir, "000003")

            summary, issues = module.scan_merged_files(data_dir, sample_limit=1)

        self.assertEqual(summary["symbol"].tolist(), ["000001"])
        self.assertTrue(issues.empty)

    def test_overlap_consistency_compares_public_sources_against_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            baostock = base / "baostock"
            jqdata = base / "jqdata"
            eastmoney = base / "eastmoney"
            akshare = base / "akshare"
            for directory in [baostock, jqdata, eastmoney, akshare]:
                directory.mkdir()
                write_prices(directory, "000001")
            module.BAOSTOCK_DIR = baostock
            module.JQDATA_DIR = jqdata
            module.EASTMONEY_DIR = eastmoney
            module.AKSHARE_DIR = akshare

            overlap = module.overlap_consistency(sample_limit=10)

        self.assertCountEqual(overlap["source"].tolist(), ["jqdata", "eastmoney", "akshare"])
        self.assertTrue((overlap["overlap_days"] == 3).all())

    def test_write_research_status_custom_out_dir_does_not_touch_official_latest(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "isolated_reports"
            out_dir.mkdir()
            official_latest = base / "official_reports" / "research_status_latest.json"
            official_latest.parent.mkdir()
            official_latest.write_text('{"generated_at":"original"}', encoding="utf-8")
            (official_latest.parent / "latest_training_result.json").write_text(
                '{"run_id":"real-run"}',
                encoding="utf-8",
            )
            (out_dir / "latest_training_result.json").write_text(
                '{"run_id":"isolated-run"}',
                encoding="utf-8",
            )

            status_path = module.write_research_status(
                out_dir=out_dir,
                source_rows=[
                    {
                        "name": "merged",
                        "files": 1,
                        "rows": 3,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-04",
                        "size_mb": 0.01,
                    }
                ],
                qlib_stats={
                    "calendar_days": 3,
                    "calendar_start": "2024-01-02",
                    "calendar_end": "2024-01-04",
                    "instruments": 1,
                    "feature_dirs": 1,
                    "size_mb": 0.01,
                },
                merged_success={"000001"},
                issue_counts={
                    "bad_ohlc_rows": 0,
                    "non_positive_volume": 0,
                    "abs_return_gt_50pct": 0,
                },
            )

            isolated_payload = json.loads(status_path.read_text(encoding="utf-8"))
            official_payload = json.loads(official_latest.read_text(encoding="utf-8"))

        self.assertEqual(status_path, out_dir / "research_status_latest.json")
        self.assertEqual(isolated_payload["data_sources"][0]["name"], "merged")
        self.assertEqual(isolated_payload["latest_training"]["run_id"], "isolated-run")
        self.assertEqual(official_payload, {"generated_at": "original"})

    def test_main_writes_empty_report_when_stock_list_and_data_are_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            module.MERGED_DIR = root / "merged"
            module.QLIB_DIR = root / "qlib"
            argv = [
                "report_data_quality.py",
                "--out-dir",
                str(out_dir),
                "--overlap-sample",
                "5",
                "--merged-sample",
                "5",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            report = out_dir / "data_quality_report_latest.md"
            status = out_dir / "research_status_latest.json"

            self.assertTrue(report.exists())
            self.assertTrue(status.exists())
            text = report.read_text(encoding="utf-8")

        self.assertIn("A-Share Data Quality Report", text)
        self.assertIn("Qlib Provider", text)
        self.assertIn("Stock universe: 0", text)


if __name__ == "__main__":
    unittest.main()
