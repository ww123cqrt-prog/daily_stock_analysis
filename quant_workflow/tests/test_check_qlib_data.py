import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_qlib_data.py"
    spec = importlib.util.spec_from_file_location("check_qlib_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckQlibDataTest(unittest.TestCase):
    def test_parse_args_accepts_candidate_provider_directory(self):
        module = load_module()
        provider = Path("/tmp/candidate-provider")

        args = module.parse_args(["--provider-dir", str(provider)])

        self.assertEqual(args.provider_dir, provider)

    def test_parse_args_accepts_native_cleaned_provider(self):
        module = load_module()

        args = module.parse_args(["--provider", "cleaned"])

        self.assertEqual(args.provider, "cleaned")
        self.assertIsNone(args.provider_dir)

    def test_import_does_not_resolve_default_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "legacy-provider"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_qlib_data.py"
            env = os.environ.copy()
            env["QLIB_PROVIDER_URI"] = str(provider)

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.util, json; "
                        "from pathlib import Path; "
                        f"path=Path({str(script_path)!r}); "
                        "spec=importlib.util.spec_from_file_location('check_qlib_data', path); "
                        "module=importlib.util.module_from_spec(spec); "
                        "spec.loader.exec_module(module); "
                        "print(json.dumps({'imported': True}))"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["imported"])

    def test_required_sample_covers_stock_indexes_and_all_ohlcv_factor_fields(self):
        module = load_module()

        self.assertEqual(
            module.REQUIRED_INSTRUMENTS,
            (
                "SH600000",
                "SH000300",
                "SH000905",
                "SH000906",
                "SZ399006",
                "SH000688",
            ),
        )
        self.assertEqual(
            module.REQUIRED_FIELDS,
            ("$open", "$high", "$low", "$close", "$volume", "$factor"),
        )

    def test_reads_provider_calendar_range(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            provider = Path(tmp)
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2008-01-02\n2026-06-24\n", encoding="utf-8")

            start, end = module.read_provider_calendar_range(provider)

        self.assertEqual((start, end), ("2008-01-02", "2026-06-24"))


if __name__ == "__main__":
    unittest.main()
