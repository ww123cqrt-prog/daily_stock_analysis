import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("add_index_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_complete_provider(
    provider: Path,
    *,
    calendar_text: str = "2026-01-02\n2026-01-05\n",
    price: float = 10.0,
) -> None:
    calendar = provider / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True, exist_ok=True)
    calendar.write_text(calendar_text, encoding="utf-8")
    instruments = provider / "instruments"
    instruments.mkdir(parents=True, exist_ok=True)
    (instruments / "all.txt").write_text(
        "SH600000\t2026-01-02\t2026-01-05\n", encoding="utf-8"
    )
    day_count = len(calendar_text.strip().splitlines())
    for symbol in ("sh000300", "sh000905", "sh000906"):
        feature_dir = provider / "features" / symbol
        feature_dir.mkdir(parents=True, exist_ok=True)
        for field in ("open", "high", "low", "close", "factor"):
            values = np.array([0.0] + [price] * day_count, dtype="<f4")
            if field == "factor":
                values[1:] = 1.0
            values.tofile(feature_dir / f"{field}.day.bin")
        np.array([0.0] + [1000.0] * day_count, dtype="<f4").tofile(
            feature_dir / "volume.day.bin"
        )


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

            class ProviderMustNotBeCalled:
                def __getattr__(self, name):
                    raise AssertionError(f"baostock must not be called: {name}")

            reused = module.add_benchmark(
                provider,
                reuse_existing=True,
                baostock_module=ProviderMustNotBeCalled(),
                required_start_date="2026-01-02",
            )
            self.assertEqual(reused["status"], "reused-existing")
            self.assertEqual(
                set(reused["benchmark_symbols"]),
                {"SH000300", "SH000905", "SH000906"},
            )

    def test_reuse_existing_rejects_incomplete_benchmark_bins(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "existing benchmark"):
                module.add_benchmark(
                    provider,
                    reuse_existing=True,
                    baostock_module=object(),
                    required_start_date="2026-01-02",
                )

    def test_imports_validated_benchmarks_with_identical_calendar(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            write_complete_provider(source, price=12.0)
            write_complete_provider(target, price=8.0)

            status = module.add_benchmark(
                target,
                import_provider_dir=source,
                required_start_date="2026-01-02",
            )

            self.assertEqual(status["status"], "imported-existing")
            for symbol in ("sh000300", "sh000905", "sh000906"):
                raw = np.fromfile(
                    target / "features" / symbol / "open.day.bin", dtype="<f4"
                )
                np.testing.assert_allclose(raw, [0.0, 12.0, 12.0])
            self.assertFalse(
                (target / module.BENCHMARK_TRANSACTION_FILENAME).exists()
            )

    def test_import_rejects_calendar_mismatch_without_writes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            write_complete_provider(
                source,
                calendar_text="2026-01-02\n2026-01-06\n",
                price=12.0,
            )
            write_complete_provider(target, price=8.0)
            before = {
                path.relative_to(target).as_posix(): path.read_bytes()
                for path in (target / "features").rglob("*.bin")
            }

            with self.assertRaisesRegex(RuntimeError, "calendar differs"):
                module.add_benchmark(
                    target,
                    import_provider_dir=source,
                    required_start_date="2026-01-02",
                )

            after = {
                path.relative_to(target).as_posix(): path.read_bytes()
                for path in (target / "features").rglob("*.bin")
            }
            self.assertEqual(after, before)

    def test_import_recovers_interrupted_directory_transaction(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            write_complete_provider(source, price=12.0)
            write_complete_provider(target, price=8.0)
            real_replace = module.os.replace
            crashed = False

            def crash_after_first_stage(source_path, target_path):
                nonlocal crashed
                result = real_replace(source_path, target_path)
                if str(source_path).endswith(".stage") and not crashed:
                    crashed = True
                    raise SystemExit("simulated benchmark publish crash")
                return result

            with mock.patch.object(module.os, "replace", crash_after_first_stage):
                with self.assertRaisesRegex(
                    SystemExit, "simulated benchmark publish crash"
                ):
                    module.add_benchmark(
                        target,
                        import_provider_dir=source,
                        required_start_date="2026-01-02",
                    )

            self.assertTrue(
                (target / module.BENCHMARK_TRANSACTION_FILENAME).is_file()
            )
            status = module.add_benchmark(
                target,
                import_provider_dir=source,
                required_start_date="2026-01-02",
            )
            self.assertEqual(status["status"], "imported-existing")
            self.assertFalse(
                (target / module.BENCHMARK_TRANSACTION_FILENAME).exists()
            )
            for symbol in ("sh000300", "sh000905", "sh000906"):
                raw = np.fromfile(
                    target / "features" / symbol / "open.day.bin", dtype="<f4"
                )
                np.testing.assert_allclose(raw, [0.0, 12.0, 12.0])


if __name__ == "__main__":
    unittest.main()
