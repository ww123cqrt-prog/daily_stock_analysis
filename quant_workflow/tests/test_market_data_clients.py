import importlib.util
import unittest
from datetime import datetime, timezone
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

    def test_fetch_eastmoney_daily_supports_raw_mode(self):
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

        module.fetch_eastmoney_daily(
            "600519",
            "2026-07-01",
            "2026-07-08",
            fqt=0,
            session=session,
            timeout=9,
        )

        self.assertEqual(session.calls[0]["params"]["fqt"], "0")

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

    def test_parse_yahoo_chart_payload_returns_daily_bars(self):
        module = load_module()
        timestamp = int(datetime(2026, 7, 8, 20, 0, tzinfo=timezone.utc).timestamp())
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"exchangeTimezoneName": "America/New_York"},
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [210.0],
                                    "high": [215.0],
                                    "low": [208.0],
                                    "close": [214.0],
                                    "volume": [123456],
                                }
                            ],
                            "adjclose": [{"adjclose": [213.5]}],
                        },
                    }
                ],
                "error": None,
            }
        }

        frame = module.parse_yahoo_chart_payload(payload)

        self.assertEqual(frame["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-07-08"])
        self.assertEqual(frame["close"].tolist(), [214.0])
        self.assertEqual(frame["adj_close"].tolist(), [213.5])
        self.assertEqual(frame["volume"].tolist(), [123456])

    def test_fetch_yahoo_chart_payload_returns_unmodified_json(self):
        module = load_module()
        raw_payload = {"chart": {"result": [], "error": None}}
        session = FakeSession(FakeResponse(raw_payload))

        payload = module.fetch_yahoo_chart_payload(
            "7203.T",
            "2026-07-01",
            "2026-07-08",
            session=session,
            timeout=7,
        )

        self.assertIs(payload, raw_payload)
        self.assertTrue(session.calls[0]["url"].endswith("/7203.T"))
        self.assertEqual(session.calls[0]["params"]["interval"], "1d")
        self.assertGreater(session.calls[0]["params"]["period2"], session.calls[0]["params"]["period1"])

    def test_parse_eastmoney_news_jsonp_returns_payload(self):
        module = load_module()
        text = 'jQuery_catdesk({"result":{"cmsArticleWebOld":[{"code":"1","title":"news"}]}});'

        payload = module.parse_eastmoney_news_jsonp(text)

        self.assertEqual(payload["result"]["cmsArticleWebOld"][0]["title"], "news")


if __name__ == "__main__":
    unittest.main()
