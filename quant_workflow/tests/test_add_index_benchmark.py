import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("add_index_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AddIndexBenchmarkTest(unittest.TestCase):
    def test_dry_run_does_not_write_or_call_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            status = module.add_benchmark(provider, dry_run=True, baostock_module=object())

        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["calendar_days"], 2)
        self.assertFalse((provider / "features" / "sh000300").exists())

    def test_dry_run_without_calendar_reports_empty_benchmark_plan(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "missing-provider"
            buffer = io.StringIO()

            with redirect_stdout(buffer):
                status = module.add_benchmark(provider, dry_run=True, baostock_module=object())

        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["calendar_days"], 0)
        self.assertIn("Dry run OK", buffer.getvalue())
        self.assertFalse((provider / "features" / "sh000300").exists())

    def test_remove_symbol_if_present_removes_only_benchmark_symbol(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all.txt"
            path.write_text(
                "SH000300\t2020-01-01\t2020-12-31\n"
                "SH600000\t2020-01-01\t2020-12-31\n",
                encoding="utf-8",
            )

            module._remove_symbol_if_present(path, "SH000300")
            text = path.read_text(encoding="utf-8")

        self.assertNotIn("SH000300", text)
        self.assertIn("SH600000", text)


if __name__ == "__main__":
    unittest.main()
