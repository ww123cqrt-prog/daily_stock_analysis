import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "market_data_clients.py"
    spec = importlib.util.spec_from_file_location("market_data_clients", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload=None, text=""):
        self.payload = payload or {}
        self.text = text
        self.encoding = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}, "timeout": timeout})
        return self.response


class MarketDataClientsTest(unittest.TestCase):
    def test_symbol_code_helpers_normalize_markets(self):
        module = load_module()

        self.assertEqual(module.eastmoney_secid("SH600519"), "1.600519")
        self.assertEqual(module.eastmoney_secid("000001.SZ"), "0.000001")
        self.assertEqual(module.sina_quote_code("600519"), "sh600519")
        self.assertEqual(module.sina_quote_code("000001"), "sz000001")

    def test_fetch_eastmoney_qfq_daily_returns_raw_daily_frame(self):
        module = load_module()
        session = FakeSession(
            FakeResponse(
                {
                    "data": {
                        "klines": [
                            "2026-07-08,10.00,10.20,10.50,9.90,123,123456.7,0,1.50,0,0.75",
                        ],
                    }
                }
            )
        )

        frame = module.fetch_eastmoney_qfq_daily("600519", "2026-07-01", "2026-07-08", session=session, timeout=9)

        self.assertEqual(session.calls[0]["params"]["secid"], "1.600519")
        self.assertEqual(session.calls[0]["params"]["beg"], "20260701")
        self.assertEqual(session.calls[0]["params"]["end"], "20260708")
        self.assertEqual(frame["volume"].tolist(), ["123"])
        self.assertEqual(frame["turn"].tolist(), ["0.75"])

    def test_parse_sina_quote_text_returns_normalized_snapshot_rows(self):
        module = load_module()
        text = (
            'var hq_str_sh600519="贵州茅台,10.00,9.90,10.20,10.50,9.80,0,0,12300,123456.7,'
            '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2026-07-08,15:00:00,00";'
        )

        rows = module.parse_sina_quote_text(text)

        self.assertEqual(rows[0]["symbol"], "600519")
        self.assertEqual(rows[0]["name"], "贵州茅台")
        self.assertEqual(rows[0]["close"], 10.2)
        self.assertAlmostEqual(rows[0]["pct"], (10.2 / 9.9 - 1) * 100)
        self.assertEqual(rows[0]["source"], "sina_snapshot")

    def test_fetch_eastmoney_a_snapshot_normalizes_scaled_fields(self):
        module = load_module()
        session = FakeSession(
            FakeResponse(
                {
                    "data": {
                        "f57": "600519",
                        "f58": "贵州茅台",
                        "f43": 1020,
                        "f170": 150,
                        "f116": 123456789,
                        "f117": 98765432,
                        "f168": 75,
                        "f10": 250,
                        "f9": 3123,
                        "f23": 987,
                    }
                }
            )
        )

        payload = module.fetch_eastmoney_a_snapshot("600519", session=session, timeout=5)

        self.assertEqual(session.calls[0]["params"]["secid"], "1.600519")
        self.assertEqual(payload["symbol"], "600519")
        self.assertEqual(payload["name"], "贵州茅台")
        self.assertEqual(payload["price"], 10.2)
        self.assertEqual(payload["pct"], 1.5)
        self.assertEqual(payload["turnover_rate"], 0.75)
        self.assertEqual(payload["source"], "eastmoney_snapshot")


if __name__ == "__main__":
    unittest.main()
