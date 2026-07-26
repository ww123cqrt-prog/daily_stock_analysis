import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_ashare_sources.py"
    spec = importlib.util.spec_from_file_location("_ashare_sources", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AShareSourcesTest(unittest.TestCase):
    def test_conservative_source_profile_rejects_unsafe_cli_values(self):
        module = load_module()

        module.validate_source_safety_profile(
            sleep_seconds=2,
            retries=1,
            max_consecutive_failures=3,
            max_requests=6000,
            overlap_days=7,
        )

        unsafe_cases = [
            ({"sleep_seconds": 1.9}, "sleep must be at least 2"),
            ({"retries": 2}, "retries must be exactly 1"),
            ({"max_consecutive_failures": 4}, "max_consecutive_failures must be between 1 and 3"),
            ({"max_requests": 6001}, "max_requests must be between 1 and 6000"),
            ({"overlap_days": 8}, "overlap_days must be between 0 and 7"),
        ]
        safe = {
            "sleep_seconds": 2,
            "retries": 1,
            "max_consecutive_failures": 3,
            "max_requests": 6000,
            "overlap_days": 7,
        }
        for override, expected_message in unsafe_cases:
            with self.subTest(override=override):
                with self.assertRaises(ValueError) as ctx:
                    module.validate_source_safety_profile(**(safe | override))
                self.assertIn(expected_message, str(ctx.exception))

    def test_source_entrypoint_lock_rejects_a_second_downloader_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "source_download.lock.json"
            owner = module.DailyUpdateLock(lock_path)
            owner.acquire("first-source")
            entrypoint = Mock(return_value=0)
            try:
                rc = module.run_source_entrypoint(
                    "eastmoney",
                    entrypoint,
                    lock_path=lock_path,
                )
            finally:
                owner.release()

            accepted_rc = module.run_source_entrypoint(
                "eastmoney",
                entrypoint,
                lock_path=lock_path,
            )

        self.assertEqual(rc, 3)
        self.assertEqual(accepted_rc, 0)
        entrypoint.assert_called_once_with()

    def test_normalizes_dates_symbols_and_chinese_daily_columns(self):
        module = load_module()

        frame = module.normalize_daily_frame(
            pd.DataFrame(
                {
                    "日期": ["20260708", "2026-07-09"],
                    "开盘": ["10.0", "10.2"],
                    "收盘": ["10.1", "10.3"],
                    "最高": ["10.4", "10.5"],
                    "最低": ["9.8", "10.0"],
                    "成交量": ["12", "13"],
                    "成交额": ["12000", "13390"],
                    "换手率": ["0.5", "0.6"],
                    "涨跌幅": ["1.0", "1.98"],
                }
            ),
            volume_multiplier=100,
        )

        self.assertEqual(module.normalize_symbol("SH600519.XSHG"), "600519")
        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-08", "2026-07-09"])
        self.assertEqual(frame["volume"].tolist(), [1200, 1300])
        self.assertEqual(frame["turn"].tolist(), [0.5, 0.6])
        self.assertEqual(frame["pctChg"].tolist(), [1.0, 1.98])

    def test_run_daily_download_merges_incremental_rows_and_writes_summary(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "eastmoney_qfq"
            out_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2026-07-01", "2026-07-02"],
                    "open": [10.0, 10.1],
                    "close": [10.1, 10.2],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "volume": [1000, 1100],
                }
            ).to_parquet(out_dir / "000001.parquet", index=False)

            def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
                self.assertEqual(symbol, "000001")
                self.assertEqual(start, "2026-07-01")
                self.assertEqual(end, "2026-07-08")
                return pd.DataFrame(
                    {
                        "date": ["2026-07-02", "2026-07-03"],
                        "open": [11.0, 11.1],
                        "close": [11.2, 11.3],
                        "high": [11.4, 11.5],
                        "low": [10.8, 10.9],
                        "volume": [1200, 1300],
                    }
                )

            summary = module.run_daily_download(
                source_name="eastmoney",
                output_dir=out_dir,
                symbols=["000001"],
                fetch_one=fetch_one,
                start="2026-07-01",
                end="2026-07-08",
                incremental=True,
                overlap_days=2,
                sleep_seconds=0,
                summary_dir=base,
                runtime_status_dir=base / "runtime" / "status",
            )

            merged = pd.read_parquet(out_dir / "000001.parquet")

            self.assertEqual(summary["ok"], 1)
            self.assertEqual(summary["fail"], 0)
            self.assertEqual(merged["close"].tolist(), [10.1, 11.2, 11.3])
            self.assertEqual((base / "eastmoney_download_summary_latest.json").exists(), True)

    def test_default_download_state_goes_under_reports_not_quant_data_root(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "eastmoney_qfq"
            reports = base / "reports"
            quant_data = base / "quant_data"

            def fetch_one(_symbol: str, _start: str, _end: str) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "date": ["2026-07-08"],
                        "open": [10.0],
                        "close": [10.1],
                        "high": [10.2],
                        "low": [9.9],
                        "volume": [1000],
                    }
                )

            original_reports_root = module.reports_root
            original_quant_data = module.QUANT_DATA
            module.reports_root = lambda: reports
            module.QUANT_DATA = quant_data
            try:
                module.run_daily_download(
                    source_name="eastmoney",
                    output_dir=out_dir,
                    symbols=["000001"],
                    fetch_one=fetch_one,
                    start="2026-07-01",
                    end="2026-07-08",
                    incremental=True,
                    overlap_days=2,
                    sleep_seconds=0,
                )
            finally:
                module.reports_root = original_reports_root
                module.QUANT_DATA = original_quant_data

            self.assertTrue((reports / "download_state" / "eastmoney_current.json").exists())
            self.assertTrue((reports / "download_state" / "eastmoney_download_summary_latest.json").exists())
            self.assertTrue((reports / "runtime" / "status" / "eastmoney_download_summary_latest.json").exists())
            self.assertFalse((quant_data / "eastmoney_current.json").exists())

    def test_incremental_range_after_end_is_no_new_without_fetching_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "eastmoney_qfq"
            out_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2026-07-08"],
                    "open": [10.0],
                    "close": [10.1],
                    "high": [10.2],
                    "low": [9.9],
                    "volume": [1000],
                }
            ).to_parquet(out_dir / "000001.parquet", index=False)
            fetch_one = Mock(side_effect=AssertionError("fetch_one should not run when start > end"))

            summary = module.run_daily_download(
                source_name="eastmoney",
                output_dir=out_dir,
                symbols=["000001"],
                fetch_one=fetch_one,
                start="2026-01-01",
                end="2026-07-03",
                incremental=True,
                overlap_days=0,
                sleep_seconds=0,
                summary_dir=base,
                runtime_status_dir=base / "runtime" / "status",
            )

        self.assertEqual(summary["no_new"], 1)
        self.assertEqual(summary["fail"], 0)
        self.assertEqual(summary["ok"], 0)
        fetch_one.assert_not_called()

    def test_remote_denial_trips_immediate_circuit_breaker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            fetch_one = Mock(side_effect=RuntimeError("Remote end closed connection without response"))

            summary = module.run_daily_download(
                source_name="eastmoney",
                output_dir=base / "eastmoney_qfq",
                symbols=["000001", "000002", "000004"],
                fetch_one=fetch_one,
                start="2026-07-01",
                end="2026-07-08",
                incremental=True,
                overlap_days=7,
                sleep_seconds=0,
                summary_dir=base,
                runtime_status_dir=base / "runtime" / "status",
            )

        self.assertEqual(fetch_one.call_count, 1)
        self.assertEqual(summary["fail"], 1)
        self.assertEqual(summary["stopped_reason"], "source_denied")

    def test_provider_denial_signatures_cover_common_http_errors(self):
        module = load_module()

        for message in [
            "403 Client Error: Forbidden",
            "429 Client Error: Too Many Requests",
            "HTTP Error 403: Forbidden",
            "10001011 黑名单用户",
        ]:
            with self.subTest(message=message):
                self.assertTrue(module.is_source_denial_error(RuntimeError(message)))

    def test_safety_stop_never_reports_a_success_exit_code(self):
        module = load_module()

        for reason in ["source_denied", "consecutive_failures", "request_budget"]:
            with self.subTest(reason=reason):
                self.assertNotEqual(
                    module.download_exit_code({"ok": 10, "no_new": 5, "stopped_reason": reason}),
                    0,
                )

        self.assertEqual(module.download_exit_code({"ok": 1, "no_new": 0, "stopped_reason": "completed"}), 0)

    def test_default_failure_limit_stops_after_three_symbols(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            fetch_one = Mock(side_effect=RuntimeError("temporary network failure"))

            summary = module.run_daily_download(
                source_name="eastmoney",
                output_dir=base / "eastmoney_qfq",
                symbols=["000001", "000002", "000004", "000005"],
                fetch_one=fetch_one,
                start="2026-07-01",
                end="2026-07-08",
                incremental=True,
                overlap_days=7,
                sleep_seconds=0,
                summary_dir=base,
                runtime_status_dir=base / "runtime" / "status",
            )

        self.assertEqual(fetch_one.call_count, 3)
        self.assertEqual(summary["stopped_reason"], "consecutive_failures")

    def test_request_budget_stops_before_fetching_another_symbol(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            fetch_one = Mock(
                return_value=pd.DataFrame(
                    {
                        "date": ["2026-07-08"],
                        "open": [10.0],
                        "close": [10.1],
                        "high": [10.2],
                        "low": [9.9],
                        "volume": [1000],
                    }
                )
            )

            summary = module.run_daily_download(
                source_name="eastmoney",
                output_dir=base / "eastmoney_qfq",
                symbols=["000001", "000002"],
                fetch_one=fetch_one,
                start="2026-07-01",
                end="2026-07-08",
                incremental=False,
                overlap_days=7,
                sleep_seconds=0,
                max_requests=1,
                summary_dir=base,
                runtime_status_dir=base / "runtime" / "status",
            )

        self.assertEqual(fetch_one.call_count, 1)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["requests_made"], 1)
        self.assertEqual(summary["stopped_reason"], "request_budget")

    def test_load_symbols_file_is_explicit_universe_and_preserves_order(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            stock_list = base / "stock_lists.json"
            symbols_file = base / "symbols.txt"
            stock_list.write_text("{not-json", encoding="utf-8")
            symbols_file.write_text("600519\n000001\n600519\n", encoding="utf-8")

            original_stock_list = module.STOCK_LIST
            module.STOCK_LIST = stock_list
            try:
                symbols = module.load_symbols(symbols_file)
            finally:
                module.STOCK_LIST = original_stock_list

        self.assertEqual(symbols, ["600519", "000001"])


if __name__ == "__main__":
    unittest.main()
