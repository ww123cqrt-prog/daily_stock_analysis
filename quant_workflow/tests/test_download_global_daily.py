from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_global_daily.py"
    spec = importlib.util.spec_from_file_location("download_global_daily", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def yahoo_payload(*, dates: list[int], closes: list[float]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": dates,
                    "indicators": {
                        "quote": [
                            {
                                "open": closes,
                                "high": closes,
                                "low": closes,
                                "close": closes,
                                "volume": [1000 for _ in closes],
                            }
                        ],
                        "adjclose": [{"adjclose": closes}],
                    },
                }
            ],
            "error": None,
        }
    }


class DownloadGlobalDailyTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_load_universe_keeps_market_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symbols.json"
            path.write_text(
                json.dumps({"markets": {"us": ["AAPL"], "jp": ["7203.T"], "kr": ["005930.KS"]}}),
                encoding="utf-8",
            )

            universe = self.module.load_universe(path)

        self.assertEqual(universe["us"], ["AAPL"])
        self.assertEqual(universe["jp"], ["7203.T"])
        self.assertEqual(universe["kr"], ["005930.KS"])

    def test_incremental_download_keeps_one_parquet_and_raw_capture_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "parquet" / "global_1d"
            raw = root / "raw" / "global_1d" / "yahoo"
            summary = root / "reports" / "download_state"
            calls = []

            def fetch(symbol: str, start: str, end: str) -> dict:
                calls.append((symbol, start, end))
                return yahoo_payload(dates=[1783454400, 1783540800], closes=[100.0, 101.0])

            result = self.module.run_download(
                universe={"us": ["AAPL"]},
                output_root=output,
                raw_root=raw,
                summary_dir=summary,
                start="2026-07-01",
                end="2026-07-10",
                incremental=True,
                overlap_days=5,
                limit=0,
                sleep_seconds=0,
                fetch_payload=fetch,
            )

            parquet = output / "us" / "AAPL.parquet"
            captures = list(raw.rglob("*.json"))
            payload = json.loads(captures[0].read_text(encoding="utf-8"))

            self.assertEqual(result["ok"], 1)
            self.assertTrue(parquet.exists())
            self.assertEqual(list((output / "us").glob("AAPL*.parquet")), [parquet])
            self.assertEqual(len(captures), 1)
            self.assertEqual(payload["source"], "yahoo_chart")
            self.assertEqual(payload["market"], "us")
            self.assertEqual(payload["symbol"], "AAPL")
            self.assertEqual(len(payload["payload_sha256"]), 64)
            self.assertEqual(payload["request"]["start"], calls[0][1])
            self.assertIn("chart", payload["payload"])
            self.assertTrue((summary / "global_daily_download_summary_latest.json").exists())
            self.assertEqual(list(root.rglob("*.tmp")), [])
