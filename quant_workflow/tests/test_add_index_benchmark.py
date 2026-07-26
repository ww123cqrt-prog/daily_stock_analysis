import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("add_index_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AddIndexBenchmarkTest(unittest.TestCase):
    def test_tencent_source_pages_backward_and_returns_close_by_date(self):
        module = load_module()

        class FakeResponse:
            def __init__(self, rows):
                self._rows = rows

            def raise_for_status(self):
                return None

            def json(self):
                return {"code": 0, "data": {"sh000300": {"day": self._rows}}}

        class FakeSession:
            def __init__(self):
                self.trust_env = None
                self.params = []

            def get(self, _url, *, params, timeout):
                self.params.append((params["param"], timeout))
                if ",,2026-07-21," in params["param"]:
                    return FakeResponse(
                        [
                            ["2026-07-20", "4575.67", "4598.32"],
                            ["2026-07-21", "4630.85", "4739.23"],
                        ]
                    )
                return FakeResponse(
                    [
                        ["2026-07-17", "4661.62", "4529.10"],
                        ["2026-07-19", "4500.00", "4510.00"],
                    ]
                )

            def close(self):
                return None

        session = FakeSession()
        with patch("requests.Session", return_value=session):
            data = module._download_tencent_index(
                "2026-07-17",
                "2026-07-21",
                use_system_proxy=True,
            )

        self.assertTrue(session.trust_env)
        self.assertEqual(len(session.params), 2)
        self.assertEqual(
            data,
            {
                "2026-07-17": 4529.10,
                "2026-07-19": 4510.00,
                "2026-07-20": 4598.32,
                "2026-07-21": 4739.23,
            },
        )

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

    def test_baostock_failure_uses_complete_eastmoney_fallback_without_forward_fill(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-20\n2026-07-21\n", encoding="utf-8")

            def failed(_start, _end):
                raise RuntimeError("source unavailable")

            status = module.add_benchmark(
                provider,
                source_fetchers=[
                    ("baostock", failed),
                    ("eastmoney", lambda _start, _end: {"2026-07-20": 4000.0, "2026-07-21": 4010.0}),
                ],
                cache_providers=[],
            )
            start_idx, values = module.read_feature_bin(provider / "features" / "sh000300" / "close.day.bin")

        self.assertEqual(status["source"], "eastmoney")
        self.assertEqual(start_idx, 0)
        self.assertEqual(values.tolist(), [4000.0, 4010.0])

    def test_incomplete_sources_fail_closed_without_replacing_existing_benchmark(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-20\n2026-07-21\n", encoding="utf-8")
            feature = provider / "features" / "sh000300" / "close.day.bin"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"previous-official-benchmark")

            with self.assertRaises(module.BenchmarkSourceError):
                module.add_benchmark(
                    provider,
                    source_fetchers=[
                        ("baostock", lambda _start, _end: {"2026-07-20": 4000.0}),
                        ("eastmoney", lambda _start, _end: {}),
                    ],
                    cache_providers=[],
                )

            self.assertEqual(feature.read_bytes(), b"previous-official-benchmark")

    def test_validated_local_provider_cache_is_a_legal_fallback(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            provider = base / "provider"
            cache = base / "cache"
            for root in [provider, cache]:
                calendar = root / "calendars" / "day.txt"
                calendar.parent.mkdir(parents=True)
                calendar.write_text("2026-07-20\n2026-07-21\n", encoding="utf-8")
            cache_feature = cache / "features" / "sh000300" / "close.day.bin"
            cache_feature.parent.mkdir(parents=True)
            module.write_feature_bin(cache_feature, 0, [4000.0, 4010.0])

            status = module.add_benchmark(
                provider,
                source_fetchers=[("baostock", lambda _start, _end: (_ for _ in ()).throw(RuntimeError("down")))],
                cache_providers=[cache],
            )

        self.assertEqual(status["source"], f"local_cache:{cache}")

    def test_default_cache_search_includes_official_and_runtime_providers(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            quant_data = Path(tmp) / "quant_data"
            official = quant_data / "qlib" / "official"
            runtime = quant_data / "runtime" / "run-1" / "provider"
            current = quant_data / "runtime" / "current" / "provider"
            for provider in [official, runtime, current]:
                feature = provider / "features" / "sh000300" / "close.day.bin"
                feature.parent.mkdir(parents=True)
                feature.write_bytes(b"cache")

            with patch.object(module, "QUANT_DATA", quant_data):
                providers = module._default_cache_providers(current)

        self.assertIn(official, providers)
        self.assertIn(runtime, providers)
        self.assertNotIn(current, providers)

    def test_all_benchmark_sources_failed_writes_sanitized_runtime_status(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            provider = base / "provider"
            status_dir = base / "runtime" / "status"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-20\n", encoding="utf-8")

            with self.assertRaises(module.BenchmarkSourceError):
                module.add_benchmark(
                    provider,
                    source_fetchers=[("baostock", lambda _start, _end: (_ for _ in ()).throw(RuntimeError("blacklisted")))],
                    cache_providers=[],
                    status_dir=status_dir,
                )

            status = json.loads((status_dir / "benchmark_update_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["attempts"][0]["source"], "baostock")


if __name__ == "__main__":
    unittest.main()
