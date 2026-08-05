import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "add_index_benchmark.py"
    spec = importlib.util.spec_from_file_location("add_index_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_sources_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_benchmark_sources.py"
    spec = importlib.util.spec_from_file_location("benchmark_sources", path)
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
    def test_eastmoney_source_maps_daily_ohlcv_for_all_benchmarks(self):
        module = load_sources_module()

        class FakeResponse:
            def __init__(self, code):
                self.code = code

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "rc": 0,
                    "data": {
                        "code": self.code,
                        "klines": [
                            "2026-07-23,10.0,10.1,10.2,9.9,1000,0,0,0,0,0",
                            "2026-07-24,10.2,10.3,10.4,10.1,1200,0,0,0,0,0",
                        ],
                    },
                }

        class FakeSession:
            def __init__(self):
                self.trust_env = None
                self.params = []
                self.headers = []

            def get(self, _url, *, params, headers, timeout):
                self.params.append((params, timeout))
                self.headers.append(headers)
                return FakeResponse(params["secid"].split(".", 1)[1])

            def close(self):
                return None

        session = FakeSession()
        with mock.patch("requests.Session", return_value=session):
            frames = module.download_eastmoney_benchmarks(
                "2026-07-23",
                "2026-07-24",
                use_system_proxy=True,
                sleep_fn=lambda _seconds: None,
            )

        self.assertTrue(session.trust_env)
        self.assertEqual(set(frames), {"SH000300", "SH000905", "SH000906"})
        self.assertEqual(
            {params["secid"] for params, _timeout in session.params},
            {"1.000300", "1.000905", "1.000906"},
        )
        self.assertTrue(
            all("Mozilla/5.0" in headers["User-Agent"] for headers in session.headers)
        )
        self.assertEqual(
            frames["SH000300"].columns.tolist(),
            ["open", "high", "low", "close", "volume"],
        )
        self.assertEqual(frames["SH000300"].loc["2026-07-24", "low"], 10.1)

    def test_eastmoney_source_uses_bounded_doh_curl_fallback(self):
        module = load_sources_module()
        import requests

        class FailingSession:
            trust_env = None

            def get(self, *_args, **_kwargs):
                raise requests.ConnectionError("synthetic DNS interception")

            def close(self):
                return None

        def curl_result(command, **kwargs):
            secid = next(
                item.split("=", 1)[1]
                for index, item in enumerate(command)
                if command[index - 1] == "--data-urlencode" and item.startswith("secid=")
            )
            code = secid.split(".", 1)[1]
            payload = {
                "rc": 0,
                "data": {
                    "code": code,
                    "klines": [
                        "2026-07-24,10.2,10.3,10.4,10.1,1200,0,0,0,0,0"
                    ],
                },
            }
            return mock.Mock(stdout=json.dumps(payload), stderr="", returncode=0)

        with mock.patch("requests.Session", return_value=FailingSession()), mock.patch(
            "subprocess.run", side_effect=curl_result
        ) as run:
            frames = module.download_eastmoney_benchmarks(
                "2026-07-24",
                "2026-07-24",
                sleep_fn=lambda _seconds: None,
                max_attempts=1,
            )

        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIn("--doh-url", command)
            self.assertIn("--noproxy", command)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["timeout"], 35)
        self.assertEqual(frames["SH000906"].loc["2026-07-24", "close"], 10.3)

    def test_tencent_robust_source_downloads_all_three_ohlcv_series(self):
        module = load_module()

        class FakeResponse:
            def __init__(self, code):
                self.code = code

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "code": 0,
                    "data": {
                        self.code: {
                            "day": [
                                ["2026-07-23", "10.0", "10.1", "10.2", "9.9", "1000"],
                                ["2026-07-24", "10.2", "10.3", "10.4", "10.1", "1200"],
                            ]
                        }
                    },
                }

        class FakeSession:
            def __init__(self):
                self.trust_env = None
                self.codes = []

            def get(self, _url, *, params, timeout):
                code = params["param"].split(",", 1)[0]
                self.codes.append((code, timeout))
                return FakeResponse(code)

            def close(self):
                return None

        session = FakeSession()
        with mock.patch("requests.Session", return_value=session):
            frames = module._download_tencent_benchmarks(
                "2026-07-23",
                "2026-07-24",
                use_system_proxy=True,
                sleep_fn=lambda _seconds: None,
            )

        self.assertTrue(session.trust_env)
        self.assertEqual(set(frames), {"SH000300", "SH000905", "SH000906"})
        self.assertEqual({code for code, _ in session.codes}, {"sh000300", "sh000905", "sh000906"})
        self.assertEqual(
            frames["SH000905"].columns.tolist(),
            ["open", "high", "low", "close", "volume"],
        )
        self.assertEqual(frames["SH000905"].loc["2026-07-24", "close"], 10.3)

    def test_tencent_source_retries_transient_failure_with_bounded_backoff(self):
        module = load_module()
        import requests

        class FakeResponse:
            def __init__(self, code):
                self.code = code

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": {
                        self.code: {
                            "day": [
                                [
                                    "2026-07-24",
                                    "10",
                                    "10",
                                    "10",
                                    "10",
                                    "1000",
                                ]
                            ]
                        }
                    }
                }

        class FakeSession:
            def __init__(self):
                self.calls = 0
                self.trust_env = False

            def get(self, _url, *, params, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise requests.ConnectionError("temporary")
                return FakeResponse(params["param"].split(",", 1)[0])

            def close(self):
                return None

        sleeps: list[float] = []
        session = FakeSession()
        with mock.patch("requests.Session", return_value=session):
            result = module._download_tencent_benchmarks(
                "2026-07-24",
                "2026-07-24",
                sleep_fn=sleeps.append,
                request_interval_seconds=0.25,
                retry_backoff_seconds=1.0,
                max_attempts=3,
            )

        self.assertEqual(set(result), set(module.BENCHMARKS))
        self.assertEqual(session.calls, 4)
        self.assertIn(1.0, sleeps)
        self.assertGreaterEqual(sleeps.count(0.25), 4)

    def test_add_benchmark_can_use_tencent_without_loading_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "provider"
            (provider / "calendars").mkdir(parents=True)
            (provider / "calendars" / "day.txt").write_text(
                "2026-01-02\n2026-01-05\n", encoding="utf-8"
            )
            (provider / "instruments").mkdir()
            (provider / "instruments" / "all.txt").write_text(
                "SH600000\t2026-01-02\t2026-01-05\n", encoding="utf-8"
            )
            frame = pd.DataFrame(
                {
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000.0, 1200.0],
                },
                index=["2026-01-02", "2026-01-05"],
            )
            with mock.patch.object(
                module,
                "_download_tencent_benchmarks",
                return_value={symbol: frame for symbol in module.BENCHMARKS},
            ), mock.patch.object(
                module, "_load_baostock", side_effect=AssertionError("Baostock must not load")
            ):
                status = module.add_benchmark(
                    provider,
                    benchmark_source="tencent",
                    required_start_date="2026-01-02",
                )

            self.assertEqual(status["source"], "tencent")
            for symbol in module.BENCHMARKS:
                self.assertTrue((provider / "features" / symbol.lower() / "close.day.bin").is_file())

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
        with mock.patch("requests.Session", return_value=session):
            data = module._download_tencent_index(
                "2026-07-17", "2026-07-21", use_system_proxy=True
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

    def test_legacy_fallback_uses_complete_second_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp) / "provider"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-20\n2026-07-21\n", encoding="utf-8")

            status = module.add_benchmark(
                provider,
                source_fetchers=[
                    ("baostock", lambda _start, _end: (_ for _ in ()).throw(RuntimeError("down"))),
                    ("eastmoney", lambda _start, _end: {"2026-07-20": 4000.0, "2026-07-21": 4010.0}),
                ],
                cache_providers=[],
            )
            start_idx, values = module.read_feature_bin(
                provider / "features" / "sh000300" / "close.day.bin"
            )

        self.assertEqual(status["source"], "eastmoney")
        self.assertEqual(start_idx, 0)
        self.assertEqual(values.tolist(), [4000.0, 4010.0])

    def test_legacy_incomplete_sources_fail_without_replacing_existing(self):
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

    def test_legacy_validated_local_cache_remains_available_for_audit(self):
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
            module.write_feature_bin(cache_feature, 0, [4000.0, 4010.0])

            status = module.add_benchmark(
                provider,
                source_fetchers=[
                    ("baostock", lambda _start, _end: (_ for _ in ()).throw(RuntimeError("down")))
                ],
                cache_providers=[cache],
            )

        self.assertEqual(status["source"], f"local_cache:{cache}")

    def test_default_cache_search_excludes_target_provider(self):
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

            with mock.patch.object(module, "QUANT_DATA", quant_data):
                providers = module._default_cache_providers(current)

        self.assertIn(official, providers)
        self.assertIn(runtime, providers)
        self.assertNotIn(current, providers)

    def test_legacy_all_sources_failed_writes_sanitized_status(self):
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
                    source_fetchers=[
                        ("baostock", lambda _start, _end: (_ for _ in ()).throw(RuntimeError("blacklisted")))
                    ],
                    cache_providers=[],
                    status_dir=status_dir,
                )

            status = json.loads(
                (status_dir / "benchmark_update_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["attempts"][0]["source"], "baostock")


if __name__ == "__main__":
    unittest.main()
