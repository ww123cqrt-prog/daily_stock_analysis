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


def write_benchmark_features(
    provider: Path,
    *,
    symbols: tuple[str, ...] = ("SH000300", "SH000905", "SH000906"),
) -> None:
    calendar = ["2026-07-17", "2026-07-20"]
    for symbol_index, symbol in enumerate(symbols, start=1):
        feature_dir = provider / "features" / symbol.lower()
        feature_dir.mkdir(parents=True, exist_ok=True)
        for field in ("open", "high", "low", "close", "volume", "factor"):
            values = np.array(
                [
                    0.0,
                    1.0 if field == "factor" else 10.0 * symbol_index,
                    1.0 if field == "factor" else 10.0 * symbol_index + 1.0,
                ],
                dtype="<f4",
            )
            if field == "volume":
                values[1:] = [1000.0, 1100.0]
            values.tofile(feature_dir / f"{field}.day.bin")


class ReportDataQualityTest(unittest.TestCase):
    def test_source_index_normalizes_native_v2_exchange_prefixes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("SH600000.parquet", "SZ000001.parquet", "830001.parquet"):
                (root / name).write_bytes(b"")

            symbols = module.load_source_index(root)

        self.assertEqual(symbols, {"600000", "000001", "830001"})

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

    def test_all_three_benchmarks_and_six_fields_are_required(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)

            complete = module.qlib_provider_stats(provider)
            (provider / "features" / "sh000906" / "volume.day.bin").unlink()
            incomplete = module.qlib_provider_stats(provider)

        self.assertTrue(complete["benchmark_complete"])
        self.assertEqual(complete["benchmark_feature_dirs"], 3)
        self.assertEqual(set(complete["benchmark_details"]), set(module.BENCHMARK_SYMBOLS))
        self.assertFalse(incomplete["benchmark_complete"])
        self.assertIn(
            "volume", incomplete["benchmark_details"]["SH000906"]["missing_fields"]
        )

    def test_benchmark_factor_must_cover_every_calendar_day(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)
            np.array([0.0, 1.0, np.nan], dtype="<f4").tofile(
                provider / "features" / "sh000300" / "factor.day.bin"
            )

            stats = module.qlib_provider_stats(provider)

        self.assertFalse(stats["benchmark_complete"])
        self.assertEqual(stats["benchmark_details"]["SH000300"]["missing_days"], 1)
        self.assertIn(
            "factor", stats["benchmark_details"]["SH000300"]["missing_fields"]
        )

    def test_help_does_not_resolve_large_dataset_defaults(self):
        module = load_module()
        with patch.object(
            module,
            "ashare_qfq_bars_dir",
            side_effect=AssertionError("A-share snapshot was resolved for --help"),
        ), patch.object(
            module,
            "qlib_provider_uri",
            side_effect=AssertionError("Qlib snapshot was resolved for --help"),
        ), patch.object(sys, "argv", ["report_data_quality.py", "--help"]):
            with self.assertRaises(SystemExit) as raised:
                module.main()

        self.assertEqual(raised.exception.code, 0)

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
