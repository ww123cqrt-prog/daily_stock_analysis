import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKFLOW_ROOT / "scripts" / "report_cash_control_backtest.py"


def load_module():
    spec = importlib.util.spec_from_file_location("report_cash_control_backtest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_module_with_env(env: dict[str, str]):
    spec = importlib.util.spec_from_file_location("report_cash_control_backtest_env", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.dict(os.environ, env, clear=False):
        spec.loader.exec_module(module)
    return module


def write_portfolio_report(path: Path, returns: list[float]) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2024-01-02", periods=len(returns))
    frame = pd.DataFrame(
        {
            "return": returns,
            "bench": returns,
            "account": 1000000,
            "cash": 0,
            "value": 1000000,
        },
        index=pd.DatetimeIndex(dates, name="datetime"),
    )
    frame.to_pickle(path)
    return dates


def write_prices(directory: Path, symbol: str, prices: list[float]) -> None:
    dates = pd.bdate_range("2023-12-01", periods=len(prices))
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


class ReportCashControlBacktestTest(unittest.TestCase):
    def test_module_paths_support_local_overrides_without_hardcoded_data_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quant_data = root / "custom_quant_data"
            mlruns = root / "custom_mlruns"

            module = load_module_with_env(
                {
                    "WORKING_PROJECTS_ROOT": str(root),
                    "QUANT_DATA_ROOT": str(quant_data),
                    "QUANT_WORKFLOW_MLRUNS_DIR": str(mlruns),
                }
            )

        self.assertEqual(module.WORKFLOW, WORKFLOW_ROOT)
        self.assertEqual(module.ROOT, root)
        self.assertEqual(module.QUANT_DATA, quant_data)
        self.assertEqual(module.MLRUNS_DIR, mlruns)

    def test_find_portfolio_report_rejects_unsafe_run_id_before_globbing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            fallback = mlruns / "0" / "safe_run" / "artifacts" / "portfolio_analysis"
            fallback.mkdir(parents=True)
            (fallback / "report_normal_1day.pkl").write_bytes(b"placeholder")

            with self.assertRaises(ValueError):
                module.find_portfolio_report_path(
                    latest_training_path=root / "missing.json",
                    mlruns_dir=mlruns,
                    run_id="../unsafe",
                )

    def test_find_portfolio_report_does_not_fall_back_to_another_run(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            other = (
                mlruns
                / "0"
                / "other_run"
                / "artifacts"
                / "portfolio_analysis"
                / "report_normal_1day.pkl"
            )
            other.parent.mkdir(parents=True)
            other.write_bytes(b"placeholder")
            latest = root / "latest.json"
            latest.write_text(json.dumps({"run_id": "missing_run"}), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "missing_run"):
                module.find_portfolio_report_path(
                    latest_training_path=latest,
                    mlruns_dir=mlruns,
                )

    def test_cash_control_report_validates_run_artifact_provenance(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = (
                root
                / "mlruns"
                / "0"
                / "safe_run"
                / "artifacts"
                / "portfolio_analysis"
                / "report_normal_1day.pkl"
            )
            report.parent.mkdir(parents=True)
            report.write_bytes(b"placeholder")
            expected = {
                "provider_identity": {"snapshot_id": "snapshot-001"},
                "provider_path": str(root / "snapshot-001"),
            }

            with patch.object(
                module,
                "validate_file_store_run_artifact",
                return_value=expected,
            ) as validate:
                provenance = module.validate_portfolio_provenance(
                    portfolio_report_path=report,
                    run_id="safe_run",
                    mlruns_dir=root / "mlruns",
                    quant_data_root_path=root / "quant_data",
                )

            self.assertEqual(provenance, expected)
            validate.assert_called_once_with(
                artifact_path=report,
                artifact_relative_path=Path(
                    "portfolio_analysis/report_normal_1day.pkl"
                ),
                run_id="safe_run",
                mlruns_dir=root / "mlruns",
                quant_data_root=root / "quant_data",
            )

    def test_return_metrics_use_a_share_trading_days_and_sharpe_name(self):
        module = load_module()
        returns = pd.Series(
            [0.01, -0.02, 0.03, 0.01],
            index=pd.bdate_range("2024-01-02", periods=4),
            name="return",
        )

        metrics = module.calculate_return_metrics(returns, trading_days_per_year=244)
        expected_sharpe = returns.mean() / returns.std() * (244 ** 0.5)

        self.assertAlmostEqual(metrics["sharpe_ratio"], expected_sharpe)
        self.assertNotIn("information_ratio", metrics)

    def test_cash_overlay_scales_daily_strategy_returns_and_reduces_drawdown(self):
        module = load_module()
        returns = pd.Series(
            [0.02, -0.10, -0.08, 0.01],
            index=pd.bdate_range("2024-01-02", periods=4),
            name="return",
        )
        signals = pd.DataFrame(
            {
                "cash_ratio": [0.10, 0.80, 0.80, 0.50],
                "position_ratio": [0.90, 0.20, 0.20, 0.50],
                "regime": ["risk_on", "risk_off", "risk_off", "defensive"],
            },
            index=returns.index,
        )

        overlay = module.apply_cash_control_overlay(returns, signals)
        baseline_metrics = module.calculate_return_metrics(returns)
        overlay_metrics = module.calculate_return_metrics(overlay)

        self.assertAlmostEqual(float(overlay.iloc[0]), 0.018)
        self.assertAlmostEqual(float(overlay.iloc[1]), -0.020)
        self.assertLess(abs(overlay_metrics["max_drawdown"]), abs(baseline_metrics["max_drawdown"]))
        self.assertEqual(overlay_metrics["days"], 4)

    def test_generates_read_only_latest_reports_from_portfolio_and_market_data(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "report_normal_1day.pkl"
            market_dir = root / "market"
            out_dir = root / "reports"
            market_dir.mkdir()
            write_portfolio_report(report_path, [0.01, -0.04, -0.03, 0.02, 0.01])
            declining_prices = [100 - i * 0.8 for i in range(60)]
            write_prices(market_dir, "000001", declining_prices)
            write_prices(market_dir, "000002", [price * 1.03 for price in declining_prices])

            payload = module.generate_report(
                portfolio_report_path=report_path,
                market_data_dir=market_dir,
                out_dir=out_dir,
                lookback_days=20,
                min_symbols=2,
                run_id="safe_run",
                provenance={
                    "artifact_sha256": "b" * 64,
                    "provider_identity": {
                        "provider_contract": "immutable_snapshot_v2",
                        "dataset_id": "qlib_ashare_v2",
                        "snapshot_id": "snapshot-001",
                    },
                    "provider_path": str(root / "snapshot-001"),
                },
            )

            latest_json = out_dir / "cash_control_backtest_latest.json"
            latest_md = out_dir / "cash_control_backtest_latest.md"
            serialized = json.loads(latest_json.read_text(encoding="utf-8"))
            markdown = latest_md.read_text(encoding="utf-8")

        self.assertEqual(payload["mode"], "read_only_cash_control_overlay")
        self.assertIn("baseline", payload["comparison"])
        self.assertIn("cash_control", payload["comparison"])
        self.assertIn("regime_counts", payload["signal_summary"])
        self.assertEqual(payload["run_id"], "safe_run")
        self.assertEqual(payload["artifact_sha256"], "b" * 64)
        self.assertEqual(payload["provenance_status"], "validated")
        self.assertEqual(
            payload["provider_identity"]["provider_contract"],
            "immutable_snapshot_v2",
        )
        self.assertEqual(serialized["mode"], "read_only_cash_control_overlay")
        self.assertIn("Cash-Control Backtest", markdown)
        self.assertIn("read-only", markdown)
        self.assertIn("不连接券商", markdown)

    def test_generates_empty_latest_report_when_portfolio_report_is_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"

            payload = module.generate_report(
                portfolio_report_path=root / "missing_report_normal_1day.pkl",
                market_data_dir=root / "market",
                out_dir=out_dir,
                lookback_days=20,
                min_symbols=2,
            )

            latest_md = out_dir / "cash_control_backtest_latest.md"
            markdown = latest_md.read_text(encoding="utf-8")

        self.assertEqual(payload["mode"], "read_only_cash_control_overlay")
        self.assertEqual(payload["comparison"]["baseline"]["days"], 0)
        self.assertIn("missing_report_normal_1day.pkl", payload["unavailable_reason"])
        self.assertIn("Cash-Control Backtest", markdown)
        self.assertIn("read-only research report", markdown)

    def test_preflight_failure_does_not_read_unvalidated_portfolio_artifact(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "report_normal_1day.pkl"
            report_path.write_bytes(b"untrusted pickle")

            with patch.object(
                module,
                "load_portfolio_returns",
                side_effect=AssertionError("unvalidated pickle was read"),
            ):
                payload = module.generate_report(
                    portfolio_report_path=report_path,
                    market_data_dir=root / "market",
                    out_dir=root / "reports",
                    run_id="safe_run",
                    preflight_unavailable_reason="provider validation failed",
                )

        self.assertEqual(payload["comparison"]["baseline"]["days"], 0)
        self.assertEqual(payload["provenance_status"], "unavailable")
        self.assertEqual(payload["unavailable_reason"], "provider validation failed")

    def test_direct_report_call_refuses_existing_pickle_without_provenance(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "report_normal_1day.pkl"
            report_path.write_bytes(b"untrusted pickle")

            with patch.object(
                module,
                "load_portfolio_returns",
                side_effect=AssertionError("unvalidated pickle was read"),
            ) as load_returns:
                payload = module.generate_report(
                    portfolio_report_path=report_path,
                    market_data_dir=root / "market",
                    out_dir=root / "reports",
                    run_id="safe_run",
                )

        self.assertEqual(payload["provenance_status"], "unavailable")
        self.assertIn("provenance was not validated", payload["unavailable_reason"])
        load_returns.assert_not_called()

    def test_main_validates_exact_run_and_returns_nonzero_on_provenance_failure(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            report = (
                mlruns
                / "0"
                / "safe_run"
                / "artifacts"
                / "portfolio_analysis"
                / "report_normal_1day.pkl"
            )
            report.parent.mkdir(parents=True)
            report.write_bytes(b"untrusted pickle")
            latest = root / "latest_training_result.json"
            latest.write_text(json.dumps({"run_id": "safe_run"}), encoding="utf-8")
            out_dir = root / "reports"

            with (
                patch.object(
                    module.sys,
                    "argv",
                    [
                        "report_cash_control_backtest.py",
                        "--latest-training",
                        str(latest),
                        "--mlruns-dir",
                        str(mlruns),
                        "--market-data-dir",
                        str(root / "market"),
                        "--out-dir",
                        str(out_dir),
                    ],
                ),
                patch.object(
                    module,
                    "validate_portfolio_provenance",
                    side_effect=ValueError("provider validation failed"),
                ) as validate,
                patch.object(
                    module,
                    "load_portfolio_returns",
                    side_effect=AssertionError("unvalidated pickle was read"),
                ),
            ):
                exit_code = module.main()

            payload = json.loads(
                (out_dir / "cash_control_backtest_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(exit_code, 1)
        validate.assert_called_once_with(
            portfolio_report_path=report,
            run_id="safe_run",
            mlruns_dir=mlruns,
            quant_data_root_path=module.QUANT_DATA.resolve(),
        )
        self.assertEqual(payload["run_id"], "safe_run")
        self.assertEqual(payload["provenance_status"], "unavailable")
        self.assertIn("provider validation failed", payload["unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
