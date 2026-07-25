import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_latest_snapshot.py"
    spec = importlib.util.spec_from_file_location("validate_latest_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    text = 'var hq_str_sh600519="贵州茅台,10.00,9.90,10.20,10.50,9.80,0,0,12300,123456.7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-07-08,15:00:00,00";'
    encoding = ""

    def raise_for_status(self):
        return None


class FakeSession:
    def get(self, url, headers, timeout):
        return FakeResponse()


class ValidateLatestSnapshotTest(unittest.TestCase):
    def test_fetch_sina_snapshot_parses_latest_price(self):
        module = load_module()

        rows = module.fetch_sina_snapshots(["600519"], session=FakeSession(), timeout=3)

        self.assertEqual(rows[0]["symbol"], "600519")
        self.assertEqual(rows[0]["name"], "贵州茅台")
        self.assertEqual(rows[0]["close"], 10.2)
        self.assertEqual(rows[0]["trade_date"], "2026-07-08")

    def test_build_validation_report_compares_snapshot_to_merged_latest_close(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            merged_dir = base / "merged"
            out_dir = base / "reports"
            merged_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2026-07-08"],
                    "open": [10.0],
                    "close": [10.0],
                    "high": [10.5],
                    "low": [9.8],
                    "volume": [12300],
                }
            ).to_parquet(merged_dir / "600519.parquet", index=False)

            payload = module.build_validation_report(
                snapshots=[{"symbol": "600519", "name": "贵州茅台", "close": 10.2, "trade_date": "2026-07-08"}],
                merged_dir=merged_dir,
                out_dir=out_dir,
            )

            self.assertEqual(payload["compared"], 1)
            self.assertAlmostEqual(payload["rows"][0]["close_rel_diff"], 0.02)
            self.assertTrue((out_dir / "latest_snapshot_validation_latest.json").exists())
            self.assertTrue((out_dir / "latest_snapshot_validation_latest.md").exists())

    def test_write_raw_snapshot_capture_preserves_original_response(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = module.write_raw_snapshot_capture(
                raw_text='var hq_str_sh600519="raw";',
                symbols=["600519"],
                raw_root=root,
                captured_at="2026-07-25T10:11:12+08:00",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["source"], "sina_snapshot")
        self.assertEqual(payload["symbols"], ["600519"])
        self.assertEqual(payload["raw_text"], 'var hq_str_sh600519="raw";')
        self.assertEqual(len(payload["raw_text_sha256"]), 64)
        self.assertIn("captured_date=2026-07-25", str(path))


if __name__ == "__main__":
    unittest.main()
