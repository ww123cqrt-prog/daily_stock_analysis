import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_jqdata.py"
    spec = importlib.util.spec_from_file_location("download_jqdata", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeJQData:
    def __init__(self):
        self.auth_calls = []
        self.price_calls = []

    def auth(self, username, password):
        self.auth_calls.append((username, password))

    def is_auth(self):
        return True

    def get_query_count(self):
        return {"spare": 1000, "total": 1000}

    def get_index_stocks(self, code, date=None):
        return ["000001.XSHE", "600519.XSHG"]

    def get_all_securities(self, types=None):
        return pd.DataFrame(index=["000001.XSHE", "600519.XSHG", "300750.XSHE"])

    def get_price(self, code, start_date, end_date, frequency, skip_paused, fq):
        self.price_calls.append(
            {
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": frequency,
                "skip_paused": skip_paused,
                "fq": fq,
            }
        )
        return pd.DataFrame(
            {
                "date": ["2025-03-03", "2025-03-04"],
                "open": [10.0, 10.2],
                "close": [10.1, 10.3],
                "high": [10.4, 10.5],
                "low": [9.8, 10.0],
                "volume": [1000.0, 1100.0],
            }
        )


class DownloadJQDataTest(unittest.TestCase):
    def test_multiprocessing_configuration_skips_fork_on_windows(self):
        module = load_module()

        class SpawnOnlyMultiprocessing:
            @staticmethod
            def get_all_start_methods():
                return ["spawn"]

            @staticmethod
            def set_start_method(*_args, **_kwargs):
                raise AssertionError("fork must not be configured on Windows")

        self.assertFalse(
            module._configure_multiprocessing_start_method(
                SpawnOnlyMultiprocessing()
            )
        )

    def test_load_env_file_sets_missing_values_without_overwriting_existing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# local credentials",
                        "JQ_USERNAME=file-user",
                        "JQ_PASSWORD=file-secret",
                    ]
                ),
                encoding="utf-8",
            )
            environ = {"JQ_USERNAME": "existing-user"}

            module.load_env_file(env_path, environ=environ)

        self.assertEqual(environ["JQ_USERNAME"], "existing-user")
        self.assertEqual(environ["JQ_PASSWORD"], "file-secret")

    def test_authenticate_prefers_configured_private_env_file_without_logging_secret(self):
        module = load_module()
        fake_jq = FakeJQData()
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "jqdata.env"
            env_path.write_text(
                "JQ_USERNAME=private-user\nJQ_PASSWORD=private-secret\n",
                encoding="utf-8",
            )
            environ = {"JQDATA_ENV_FILE": str(env_path)}
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                authenticated = module.authenticate(fake_jq, environ)

        self.assertTrue(authenticated)
        self.assertEqual(fake_jq.auth_calls, [("private-user", "private-secret")])
        self.assertNotIn("private-secret", output.getvalue())

    def test_main_default_env_file_order_honors_configured_private_env_file(self):
        module = load_module()
        fake_jq = FakeJQData()
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "jqdata.env"
            env_path.write_text(
                "JQ_USERNAME=private-user\nJQ_PASSWORD=private-secret\n",
                encoding="utf-8",
            )
            environ = {"JQDATA_ENV_FILE": str(env_path)}
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = module.main(
                    [
                        "--universe",
                        "csi300",
                        "--limit",
                        "1",
                        "--sleep",
                        "0",
                        "--dry-run",
                    ],
                    jq_module=fake_jq,
                    environ=environ,
                )

        self.assertEqual(result, 0)
        self.assertEqual(fake_jq.auth_calls, [("private-user", "private-secret")])
        self.assertIn("JQData dry run OK", output.getvalue())
        self.assertNotIn("private-secret", output.getvalue())

    def test_authenticate_env_file_none_skips_default_files(self):
        module = load_module()
        fake_jq = FakeJQData()
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            authenticated = module.authenticate(fake_jq, {}, env_file=None)

        self.assertFalse(authenticated)
        self.assertEqual(fake_jq.auth_calls, [])
        self.assertIn("credentials are not configured", stderr.getvalue())

    def test_dry_run_authenticates_resolves_universe_and_writes_no_files(self):
        module = load_module()
        fake_jq = FakeJQData()
        environ = {"JQ_USERNAME": "dry-user", "JQ_PASSWORD": "dry-secret"}
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "jqdata"
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = module.main(
                    [
                        "--universe",
                        "csi300",
                        "--limit",
                        "1",
                        "--sleep",
                        "0",
                        "--out",
                        str(out_dir),
                        "--dry-run",
                    ],
                    jq_module=fake_jq,
                    environ=environ,
                )

            text = output.getvalue()

        self.assertEqual(result, 0)
        self.assertEqual(fake_jq.auth_calls, [("dry-user", "dry-secret")])
        self.assertEqual(fake_jq.price_calls, [])
        self.assertFalse(out_dir.exists())
        self.assertIn("JQData dry run OK", text)
        self.assertIn("resolved stocks: 2", text)
        self.assertIn("000001.XSHE", text)
        self.assertNotIn("dry-secret", text)

    def test_dry_run_defaults_to_small_sample_without_writing_files(self):
        module = load_module()
        fake_jq = FakeJQData()
        environ = {"JQ_USERNAME": "dry-user", "JQ_PASSWORD": "dry-secret"}
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "jqdata"
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = module.main(
                    [
                        "--universe",
                        "all",
                        "--sleep",
                        "0",
                        "--out",
                        str(out_dir),
                        "--dry-run",
                    ],
                    jq_module=fake_jq,
                    environ=environ,
                )

            text = output.getvalue()

        self.assertEqual(result, 0)
        self.assertEqual(fake_jq.price_calls, [])
        self.assertFalse(out_dir.exists())
        self.assertIn("resolved stocks: 3", text)
        self.assertIn("selected stocks: 3", text)

    def test_custom_universe_is_normalized_and_deduplicated(self):
        module = load_module()

        codes = module.resolve_universe(
            FakeJQData(),
            "000001, SH600519;300750.XSHE 000001",
            "2026-07-05",
        )

        self.assertEqual(codes, ["000001.XSHE", "600519.XSHG", "300750.XSHE"])

    def test_invalid_custom_universe_fails_before_price_query(self):
        module = load_module()
        fake_jq = FakeJQData()
        environ = {"JQ_USERNAME": "dry-user", "JQ_PASSWORD": "dry-secret"}
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            with contextlib.redirect_stderr(stderr):
                result = module.main(
                    [
                        "--universe",
                        "../../bad",
                        "--out",
                        str(Path(temp_dir) / "jqdata"),
                        "--dry-run",
                    ],
                    jq_module=fake_jq,
                    environ=environ,
                )

        self.assertEqual(result, 2)
        self.assertEqual(fake_jq.price_calls, [])
        self.assertIn("Unsupported security code format", stderr.getvalue())
        self.assertNotIn("dry-secret", stderr.getvalue())

    def test_download_writes_to_custom_output_directory(self):
        module = load_module()
        fake_jq = FakeJQData()
        environ = {"JQ_USERNAME": "real-user", "JQ_PASSWORD": "real-secret"}
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "jqdata"

            result = module.main(
                [
                    "--universe",
                    "csi300",
                    "--start",
                    "2025-03-03",
                    "--end",
                    "2025-03-04",
                    "--limit",
                    "1",
                    "--sleep",
                    "0",
                    "--out",
                    str(out_dir),
                ],
                jq_module=fake_jq,
                environ=environ,
            )
            frame = pd.read_parquet(out_dir / "000001.parquet")

        self.assertEqual(result, 0)
        self.assertEqual(len(fake_jq.price_calls), 1)
        self.assertEqual(frame["close"].tolist(), [10.1, 10.3])


if __name__ == "__main__":
    unittest.main()
