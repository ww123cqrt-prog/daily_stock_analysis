import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_baostock.py"
    spec = importlib.util.spec_from_file_location("download_baostock", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DownloadBaostockTest(unittest.TestCase):
    def test_query_one_runs_without_unix_signal_timer(self):
        module = load_module()

        class EmptySuccessResult:
            error_code = "0"
            error_msg = "success"
            fields = ["date", "code"]

            def next(self):
                return False

        query = Mock(return_value=EmptySuccessResult())
        fake_bs = types.SimpleNamespace(query_history_k_data_plus=query)

        with patch.object(module, "signal", types.SimpleNamespace()):
            result = module.query_one(
                fake_bs,
                "sz.000001",
                "2026-07-23",
                "2026-07-24",
                "2",
                retries=1,
                timeout=5,
            )

        self.assertTrue(result.empty)
        query.assert_called_once()

    def test_blacklist_login_failure_is_recorded_as_immediate_circuit_break(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            stock_list = root / "stock_lists.json"
            stock_list.write_text(json.dumps({"all": ["000001.XSHE"]}), encoding="utf-8")
            fake_bs = types.SimpleNamespace(
                login=lambda: types.SimpleNamespace(error_code="10001011", error_msg="黑名单用户，请与管理员联系"),
                logout=lambda: None,
            )
            with patch.object(module, "STOCK_LIST", stock_list), \
                    patch.object(module, "REPORTS_ROOT", reports), \
                    patch.dict(sys.modules, {"baostock": fake_bs}), \
                    patch.object(sys, "argv", ["download_baostock.py", "--out", str(root / "out")]):
                rc = module.main()

            status = json.loads(
                (reports / "runtime" / "status" / "baostock_download_summary_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rc, 2)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["failure_type"], "source_blacklisted")
        self.assertTrue(status["manual_review_required"])
        self.assertEqual(status["stopped_reason"], "source_denied")
        self.assertEqual(status["requests_made"], 0)

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

    def test_incremental_range_after_end_is_no_new_without_querying_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_list = root / "stock_lists.json"
            stock_list.write_text(json.dumps({"all": ["000001.XSHE"]}), encoding="utf-8")
            out_dir = root / "parquet"
            out_dir.mkdir()
            pd.DataFrame({"date": pd.to_datetime(["2026-07-08"])}).to_parquet(out_dir / "000001.parquet", index=False)

            fake_bs = types.SimpleNamespace(
                login=lambda: types.SimpleNamespace(error_code="0", error_msg="success"),
                logout=lambda: None,
            )
            query_one = Mock(side_effect=AssertionError("query_one should not run when start > end"))
            argv = [
                "download_baostock.py",
                "--incremental",
                "--overlap-days",
                "0",
                "--end",
                "20260703",
                "--sleep",
                "2",
                "--out",
                str(out_dir),
            ]
            with patch.object(module, "STOCK_LIST", stock_list), \
                    patch.object(module, "EMPTY_CODES_FILE", root / "empty.json"), \
                    patch.object(module, "TIMEOUT_CODES_FILE", root / "timeout.json"), \
                    patch.object(module, "CURRENT_FILE", root / "current.json"), \
                    patch.object(module, "QUANT_DATA", root), \
                    patch.object(module, "query_one", query_one), \
                    patch.dict(sys.modules, {"baostock": fake_bs}), \
                    patch.object(sys, "argv", argv):
                rc = module.main()

        self.assertEqual(rc, 0)
        query_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
