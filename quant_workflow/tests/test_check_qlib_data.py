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
    def test_provider_uri_honors_quant_data_root_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            quant_data = Path(temp_dir) / "quant_data"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_qlib_data.py"
            env = os.environ.copy()
            env["QUANT_DATA_ROOT"] = str(quant_data)

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
                        "print(json.dumps({'provider_uri': str(module.PROVIDER_URI)}))"
                    ),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["provider_uri"],
            str(quant_data / "qlib" / "cn_data_baostock_jqdata"),
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
