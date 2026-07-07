import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_china_astock_indicators.py"
    spec = importlib.util.spec_from_file_location("report_china_astock_indicators", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_indicator_fixture(root: Path) -> None:
    strategies = root / "strategies"
    strategies.mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (strategies / "__init__.py").write_text("", encoding="utf-8")
    (strategies / "_indicators.py").write_text(
        "\n".join(
            [
                "import pandas as pd",
                "",
                "def sma(series, window):",
                "    return pd.Series(series).rolling(window, min_periods=1).mean()",
                "",
                "def ema(series, window):",
                "    return pd.Series(series).ewm(span=window, adjust=False).mean()",
                "",
                "def rsi(series, window):",
                "    close = pd.Series(series)",
                "    delta = close.diff().fillna(0)",
                "    gain = delta.clip(lower=0).rolling(window, min_periods=1).mean()",
                "    loss = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()",
                "    rs = gain / loss.replace(0, 1e-9)",
                "    return 100 - (100 / (1 + rs))",
                "",
                "def atr(high, low, close, window):",
                "    high = pd.Series(high)",
                "    low = pd.Series(low)",
                "    close = pd.Series(close)",
                "    prev_close = close.shift(1).fillna(close)",
                "    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)",
                "    return tr.rolling(window, min_periods=1).mean()",
                "",
                "def bollinger(series, window=20, mult=2.0):",
                "    close = pd.Series(series)",
                "    mid = close.rolling(window, min_periods=1).mean()",
                "    std = close.rolling(window, min_periods=1).std().fillna(0)",
                "    return mid - mult * std, mid, mid + mult * std",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_price_fixture(path: Path) -> None:
    rows = []
    for symbol, start in (("000001", 10.0), ("600519", 100.0), ("000002", 20.0)):
        for day in range(30):
            close = start + day * 0.5
            rows.append(
                {
                    "date": f"2026-01-{day + 1:02d}",
                    "symbol": symbol,
                    "open": close - 0.2,
                    "high": close + 0.4,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + day,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


class ReportChinaAstockIndicatorsTest(unittest.TestCase):
    def test_generates_read_only_indicator_report_and_latest_aliases(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            china_root = base / "china-astock-quant"
            write_indicator_fixture(china_root)
            input_csv = base / "prices.csv"
            output_dir = base / "reports"
            write_price_fixture(input_csv)

            status = module.run_report(
                input_csv=input_csv,
                output_dir=output_dir,
                china_astock_root=china_root,
                timestamp="20260706_010203",
                symbols="000001,600519",
                lookback_days=25,
                limit=2,
            )

            output_json = output_dir / "china_astock_indicator_report_20260706_010203.json"
            output_md = output_dir / "china_astock_indicator_report_20260706_010203.md"
            latest_json = output_dir / "china_astock_indicator_report_latest.json"
            latest_md = output_dir / "china_astock_indicator_report_latest.md"

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["research_only"], True)
            self.assertEqual(status["trading_enabled"], False)
            self.assertEqual(status["symbols_requested"], ["000001", "600519"])
            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())
            self.assertEqual(latest_json.read_text(encoding="utf-8"), output_json.read_text(encoding="utf-8"))
            self.assertEqual(latest_md.read_text(encoding="utf-8"), output_md.read_text(encoding="utf-8"))

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            symbols = [row["symbol"] for row in payload["rows"]]
            self.assertEqual(symbols, ["000001", "600519"])
            self.assertIn("sma20", payload["rows"][0])
            self.assertIn("ema20", payload["rows"][0])
            self.assertIn("rsi14", payload["rows"][0])
            self.assertIn("atr14", payload["rows"][0])
            self.assertIn("bollinger_position", payload["rows"][0])
            self.assertIn("只读研究报告", output_md.read_text(encoding="utf-8"))
            self.assertIn("不连接券商", output_md.read_text(encoding="utf-8"))

    def test_rejects_missing_price_columns(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            china_root = base / "china-astock-quant"
            write_indicator_fixture(china_root)
            input_csv = base / "bad.csv"
            input_csv.write_text("date,symbol,close\n2026-01-01,000001,10\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                module.run_report(input_csv=input_csv, output_dir=base / "reports", china_astock_root=china_root)

            self.assertIn("required columns", str(ctx.exception))

    def test_omitted_limit_defaults_to_bounded_sample(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            china_root = base / "china-astock-quant"
            write_indicator_fixture(china_root)
            input_csv = base / "prices.csv"
            write_price_fixture(input_csv)

            status = module.run_report(
                input_csv=input_csv,
                output_dir=base / "reports",
                china_astock_root=china_root,
                timestamp="20260706_020304",
            )

        self.assertEqual(status["limit"], 50)

    def test_missing_default_input_dir_writes_unavailable_rows_for_requested_symbols(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            china_root = base / "china-astock-quant"
            write_indicator_fixture(china_root)

            status = module.run_report(
                input_dir=base / "missing-input",
                output_dir=base / "reports",
                china_astock_root=china_root,
                timestamp="20260706_030405",
                symbols="000001,600519",
                lookback_days=60,
                limit=2,
            )

            latest_md = base / "reports" / "china_astock_indicator_report_latest.md"
            markdown = latest_md.read_text(encoding="utf-8")

        self.assertEqual(status["status"], "completed")
        self.assertEqual([row["symbol"] for row in status["rows"]], ["000001", "600519"])
        self.assertIn("data_unavailable", status["rows"][0]["tags"])
        self.assertIn("000001", markdown)
        self.assertIn("只读研究报告", markdown)

    def test_rejects_limit_above_browser_safe_bound(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            china_root = base / "china-astock-quant"
            write_indicator_fixture(china_root)
            input_csv = base / "prices.csv"
            write_price_fixture(input_csv)

            with self.assertRaises(ValueError) as ctx:
                module.run_report(
                    input_csv=input_csv,
                    output_dir=base / "reports",
                    china_astock_root=china_root,
                    limit=501,
                )

        self.assertIn("Expected integer <= 500", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
