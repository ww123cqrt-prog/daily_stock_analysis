import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_eastmoney.py"
    spec = importlib.util.spec_from_file_location("download_eastmoney", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return FakeResponse(self.payload)


class DownloadEastMoneyTest(unittest.TestCase):
    def test_fetch_one_normalizes_kline_rows_to_baostock_schema(self):
        module = load_module()
        session = FakeSession(
            {
                "data": {
                    "klines": [
                        "2026-07-08,10.00,10.20,10.50,9.90,123,123456.7,0,1.50,0,0.75",
                    ],
                }
            }
        )

        frame = module.fetch_one("600519", "2026-07-01", "2026-07-08", session=session, timeout=9)

        self.assertEqual(session.calls[0][1]["secid"], "1.600519")
        self.assertEqual(session.calls[0][1]["klt"], "101")
        self.assertEqual(session.calls[0][1]["fqt"], "1")
        self.assertIn("User-Agent", session.calls[0][2])
        self.assertEqual(frame.columns.tolist(), ["date", "open", "close", "high", "low", "volume", "amount", "turn", "pctChg"])
        self.assertEqual(frame["volume"].tolist(), [12300])
        self.assertEqual(frame["pctChg"].tolist(), [1.5])


if __name__ == "__main__":
    unittest.main()
