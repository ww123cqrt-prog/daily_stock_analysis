import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_market_regime.py"
    spec = importlib.util.spec_from_file_location("report_market_regime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_prices(directory: Path, symbol: str, prices: list[float]) -> None:
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [100000] * len(prices),
        }
    )
    frame.to_parquet(directory / f"{symbol}.parquet", index=False)


class ReportMarketRegimeTest(unittest.TestCase):
    def test_classifies_broad_decline_as_risk_off_with_high_cash(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "merged"
            data_dir.mkdir()
            prices = [100 - i for i in range(80)]
            write_prices(data_dir, "000001", prices)
            write_prices(data_dir, "000002", [price * 1.1 for price in prices])

            result = module.analyze_market_regime(data_dir, lookback_days=60, min_symbols=2)

        self.assertEqual(result["regime"], "risk_off")
        self.assertGreaterEqual(result["cash_ratio"], 0.6)
        self.assertLess(result["metrics"]["return_20d"], 0)
        self.assertLess(result["metrics"]["breadth_20d"], 0.45)

    def test_classifies_broad_advance_as_risk_on_with_low_cash(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "merged"
            data_dir.mkdir()
            prices = [50 + i for i in range(80)]
            write_prices(data_dir, "000001", prices)
            write_prices(data_dir, "000002", [price * 0.9 for price in prices])

            result = module.analyze_market_regime(data_dir, lookback_days=60, min_symbols=2)

        self.assertEqual(result["regime"], "risk_on")
        self.assertLessEqual(result["cash_ratio"], 0.2)
        self.assertGreater(result["metrics"]["return_20d"], 0)
        self.assertGreater(result["metrics"]["breadth_20d"], 0.55)

    def test_classifies_mixed_market_as_neutral(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "merged"
            data_dir.mkdir()
            write_prices(data_dir, "000001", [100.0] * 80)
            write_prices(data_dir, "000002", [100 + (0.1 if i % 2 == 0 else -0.1) for i in range(80)])

            result = module.analyze_market_regime(data_dir, lookback_days=60, min_symbols=2)

        self.assertEqual(result["regime"], "neutral")
        self.assertEqual(result["cash_ratio"], 0.30)

    def test_classifies_moderate_broad_decline_as_defensive(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "merged"
            data_dir.mkdir()
            prices = [100 - i * 0.09 for i in range(80)]
            write_prices(data_dir, "000001", prices)
            write_prices(data_dir, "000002", [price * 1.02 for price in prices])

            result = module.analyze_market_regime(data_dir, lookback_days=60, min_symbols=2)

        self.assertEqual(result["regime"], "defensive")
        self.assertEqual(result["cash_ratio"], 0.50)

    def test_writes_latest_markdown_and_json_reports(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "merged"
            out_dir = root / "reports"
            data_dir.mkdir()
            prices = [20 + i * 0.2 for i in range(70)]
            write_prices(data_dir, "000001", prices)
            write_prices(data_dir, "000002", [price * 1.05 for price in prices])

            payload = module.generate_report(data_dir, out_dir, lookback_days=60, min_symbols=2)

            latest_md = out_dir / "market_regime_latest.md"
            latest_json = out_dir / "market_regime_latest.json"
            markdown = latest_md.read_text(encoding="utf-8")
            serialized = latest_json.read_text(encoding="utf-8")

        self.assertEqual(payload["regime"], "risk_on")
        self.assertIn("Market Regime Report", markdown)
        self.assertIn("cash_ratio", serialized)

    def test_writes_insufficient_data_report_for_empty_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "empty"
            out_dir = root / "reports"
            data_dir.mkdir()

            payload = module.generate_report(data_dir, out_dir, lookback_days=60, min_symbols=2)
            markdown = (out_dir / "market_regime_latest.md").read_text(encoding="utf-8")

        self.assertEqual(payload["regime"], "insufficient_data")
        self.assertEqual(payload["cash_ratio"], 0.80)
        self.assertIn("Market Regime Report", markdown)
        self.assertIn("Suggested cash_ratio", markdown)


if __name__ == "__main__":
    unittest.main()
