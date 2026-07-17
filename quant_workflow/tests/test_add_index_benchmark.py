import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("add_index_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AddIndexBenchmarkTest(unittest.TestCase):
    def test_download_rejects_baostock_error_code(self):
        module = load_module()

        class ErrorResult:
            error_code = "10002007"
            error_msg = "network error"

            def next(self):
                return False

        class FakeBaostock:
            def query_history_k_data_plus(self, *args, **kwargs):
                return ErrorResult()

        with self.assertRaisesRegex(RuntimeError, "baostock benchmark query failed"):
            module._download_index_data(
                FakeBaostock(), "sh.000300", "2026-01-02", "2026-01-05"
            )

    def test_missing_required_trading_day_is_not_forward_or_back_filled(self):
        module = load_module()

        class Result:
            error_code = "0"
            error_msg = "success"

            def __init__(self):
                self.done = False

            def next(self):
                if self.done:
                    return False
                self.done = True
                return True

            def get_row_data(self):
                return ["2026-01-05", "10", "11", "9", "10.5", "1000"]

        class FakeBaostock:
            def login(self):
                return type("Login", (), {"error_code": "0", "error_msg": ""})()

            def logout(self):
                return None

            def query_history_k_data_plus(self, *args, **kwargs):
                return Result()

        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text(
                "2026-01-02\n2026-01-05\n2026-01-06\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "missing required trading days"):
                module.add_benchmark(provider, baostock_module=FakeBaostock())

            self.assertFalse((provider / "features" / "sh000300").exists())

    def test_dry_run_does_not_write_or_call_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            status = module.add_benchmark(
                provider, dry_run=True, baostock_module=object()
            )

        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["calendar_days"], 2)
        self.assertFalse((provider / "features" / "sh000300").exists())

    def test_dry_run_without_calendar_reports_empty_benchmark_plan(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "missing-provider"
            buffer = io.StringIO()

            with redirect_stdout(buffer):
                status = module.add_benchmark(
                    provider, dry_run=True, baostock_module=object()
                )

        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["calendar_days"], 0)
        self.assertIn("Dry run OK", buffer.getvalue())
        self.assertFalse((provider / "features" / "sh000300").exists())

    def test_remove_symbol_if_present_removes_only_benchmark_symbol(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all.txt"
            path.write_text(
                "SH000300\t2020-01-01\t2020-12-31\nSH600000\t2020-01-01\t2020-12-31\n",
                encoding="utf-8",
            )

            module._remove_symbol_if_present(path, "SH000300")
            text = path.read_text(encoding="utf-8")

        self.assertNotIn("SH000300", text)
        self.assertIn("SH600000", text)

    def test_adds_csi300_csi500_csi800_ohlcv_without_polluting_stock_pools(self):
        module = load_module()

        class Result:
            def __init__(self, rows):
                self.rows = list(rows)
                self.index = 0

            def next(self):
                if self.index >= len(self.rows):
                    return False
                self.index += 1
                return True

            def get_row_data(self):
                return self.rows[self.index - 1]

        class FakeBaostock:
            def __init__(self):
                self.codes = []
                self.logged_out = False

            def login(self):
                return type("Login", (), {"error_code": "0", "error_msg": ""})()

            def logout(self):
                self.logged_out = True

            def query_history_k_data_plus(self, code, fields, **kwargs):
                self.codes.append(code)
                self.assert_fields = fields
                return Result(
                    [
                        ["2026-01-02", "10", "11", "9", "10.5", "1000"],
                        ["2026-01-05", "10.5", "12", "10", "11.5", "1200"],
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")
            instruments = provider / "instruments"
            instruments.mkdir()
            for name in ["all.txt", "csi300.txt", "csi500.txt", "csi800.txt"]:
                (instruments / name).write_text(
                    "SH000300\t2026-01-02\t2026-01-05\n"
                    "SH000905\t2026-01-02\t2026-01-05\n"
                    "SH000906\t2026-01-02\t2026-01-05\n"
                    "SH600000\t2026-01-02\t2026-01-05\n",
                    encoding="utf-8",
                )
            fake = FakeBaostock()

            status = module.add_benchmark(provider, baostock_module=fake)

            self.assertEqual(status["status"], "completed")
            self.assertEqual(
                set(status["benchmark_symbols"]), {"SH000300", "SH000905", "SH000906"}
            )
            self.assertEqual(set(fake.codes), {"sh.000300", "sh.000905", "sh.000906"})
            self.assertTrue(fake.logged_out)
            for symbol in ["sh000300", "sh000905", "sh000906"]:
                feature_dir = provider / "features" / symbol
                self.assertTrue((feature_dir / "open.day.bin").is_file())
                raw = np.fromfile(feature_dir / "open.day.bin", dtype="<f4")
                np.testing.assert_allclose(raw, [0.0, 10.0, 10.5])
                volume = np.fromfile(feature_dir / "volume.day.bin", dtype="<f4")
                np.testing.assert_allclose(volume, [0.0, 1000.0, 1200.0])
            for path in instruments.glob("*.txt"):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("SH000300", text)
                self.assertNotIn("SH000905", text)
                self.assertNotIn("SH000906", text)
                self.assertIn("SH600000", text)


if __name__ == "__main__":
    unittest.main()
