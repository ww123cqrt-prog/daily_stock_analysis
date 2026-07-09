import importlib.util
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKFLOW_ROOT / "scripts" / "report_catdesk_signal_validation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("report_catdesk_signal_validation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReportCatdeskSignalValidationTest(unittest.TestCase):
    def _generate_minimal_report(self, allow_backfilled_qlib: bool):
        module = load_module()
        captured = {}
        base_row = {
            "code": "000001",
            "name": "平安银行",
            "theme": "金融",
            "filtered_score": 80.0,
            "validation_return_pct": 1.2,
            "hit": True,
        }

        def fake_load_latest_qlib_scores(signal_date, top_reference_date, run_cutoff):
            captured["signal_date"] = signal_date
            captured["top_reference_date"] = top_reference_date
            captured["run_cutoff"] = run_cutoff
            return {
                "available": True,
                "prediction_path": "/tmp/pred.pkl",
                "latest_date": signal_date,
                "scores": {"SZ000001": 0.8},
                "percentiles": {"SZ000001": 0.9},
            }

        def fake_strategy_comparison(rows, top_n):
            return {
                module.DEFAULT_SELECTED_STRATEGY: {
                    "selected": rows[:top_n],
                    "summary": {"count": len(rows[:top_n])},
                    "label": module.STRATEGY_LABELS[module.DEFAULT_SELECTED_STRATEGY],
                    "description": "",
                    "missed_top_actual_winners": [],
                    "captured_top_actual_winner_count": len(rows[:top_n]),
                }
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            with ExitStack() as stack:
                stack.enter_context(patch.object(module, "quant_data_root", return_value=root / "quant_data"))
                stack.enter_context(patch.object(module, "reports_root", return_value=root / "reports_root"))
                stack.enter_context(
                    patch.object(
                        module,
                        "load_catdesk_candidates",
                        return_value=(
                            {"date": "2026-07-08", "generated_at": "2026-07-08T09:00:00+08:00"},
                            [
                                {
                                    "raw": {"candidate": {"code": "000001", "name": "平安银行"}},
                                    "theme": "金融",
                                    "theme_rank": 1,
                                }
                            ],
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        module,
                        "classify_catdesk_report",
                        return_value={
                            "catdesk_report_status": "as_of_morning",
                            "catdesk_report_reason": "",
                        },
                    )
                )
                stack.enter_context(
                    patch.object(
                        module,
                        "find_market_regime_path",
                        return_value=(
                            root / "market_regime_20260708_090000.json",
                            {
                                "market_regime_status": "as_of_signal_date",
                                "market_regime_reason": "",
                            },
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        module,
                        "load_market_regime",
                        return_value={"regime": "neutral", "position_ratio": 0.5, "cash_ratio": 0.5},
                    )
                )
                stack.enter_context(patch.object(module, "load_latest_qlib_scores", side_effect=fake_load_latest_qlib_scores))
                stack.enter_context(patch.object(module, "read_price_point_on_or_before", return_value=module.PricePoint(close=10.0)))
                stack.enter_context(patch.object(module, "score_candidate", return_value=dict(base_row)))
                stack.enter_context(patch.object(module, "apply_validation", side_effect=lambda rows, **_kwargs: rows))
                stack.enter_context(
                    patch.object(
                        module,
                        "attach_strategy_scores",
                        side_effect=lambda rows: [
                            {**row, "strategy_scores": {module.DEFAULT_SELECTED_STRATEGY: row["filtered_score"]}}
                            for row in rows
                        ],
                    )
                )
                stack.enter_context(patch.object(module, "strategy_comparison", side_effect=fake_strategy_comparison))
                stack.enter_context(patch.object(module, "rank_by_strategy", side_effect=lambda rows, _strategy: rows))
                stack.enter_context(patch.object(module, "summarize", side_effect=lambda rows: {"count": len(rows)}))
                stack.enter_context(patch.object(module, "build_markdown", return_value="# ok\n"))

                payload = module.generate_report(
                    catdesk_report=root / "catdesk.json",
                    signal_date="2026-07-08",
                    validation_date="2026-07-09",
                    candidate_set="ashare",
                    top_n=1,
                    selected_strategy=module.DEFAULT_SELECTED_STRATEGY,
                    use_online_validation=False,
                    use_system_proxy=False,
                    allow_backfilled_qlib=allow_backfilled_qlib,
                    timeout=5.0,
                    out_dir=out_dir,
                )

        return captured, payload

    def test_signal_price_uses_previous_trade_date_when_requested_date_has_no_bar(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            frame = pd.DataFrame(
                [
                    {"date": "2026-07-02", "close": 10.0, "pctChg": 1.0, "amount": 100.0, "turn": 2.0},
                    {"date": "2026-07-03", "close": 11.0, "pctChg": 2.0, "amount": 110.0, "turn": 3.0},
                ]
            )
            frame.to_parquet(source_dir / "000001.parquet")

            point = module.read_price_point_on_or_before(source_dir, "000001", "2026-07-05")

        self.assertEqual(point.close, 11.0)
        self.assertEqual(point.trade_date, "2026-07-03")
        self.assertEqual(point.requested_date, "2026-07-05")
        self.assertEqual(point.fallback_reason, "requested_date_missing_use_previous_trade_date")

    def test_validation_markdown_uses_payload_dates_not_hardcoded_dates(self):
        module = load_module()
        payload = {
            "generated_at": "2026-07-09T17:30:00+08:00",
            "title": "CatDesk + DSA/Qlib Signal Validation",
            "signal_date": "2026-07-06",
            "validation_date": "2026-07-07",
            "selected": [],
            "all_candidates": [],
            "selected_summary": {},
            "raw_summary": {},
            "strategies": {},
            "selected_strategy_label": "主线题材桶",
            "inputs": {
                "catdesk_report": "/tmp/2026-07-06-morning.json",
                "signal_price_dir": "/tmp/prices",
                "market_regime_report": "/tmp/market.json",
            },
            "qlib": {"prediction_path": "", "latest_date": "2026-07-01"},
            "market_regime": {"regime": "risk_off", "position_ratio": 0.2, "cash_ratio": 0.8},
            "input_quality": {
                "catdesk_report_status": "as_of_morning",
                "market_regime_status": "as_of_signal_date",
                "return_basis": "close_to_close_research_only",
            },
        }

        markdown = module.build_markdown(payload)

        self.assertIn("2026-07-06 candidate source", markdown)
        self.assertIn("Local daily bars on or before 2026-07-06", markdown)
        self.assertIn("2026-07-07 close/snapshot", markdown)
        self.assertNotIn("2026-07-08 candidate source", markdown)
        self.assertNotIn("Local daily bars on or before 2026-07-08", markdown)
        self.assertNotIn("AlphaSift", markdown)

    def test_business_day_pairs_skip_weekends(self):
        module = load_module()

        pairs = module.business_day_pairs("2026-07-08", 4)

        self.assertEqual(
            pairs,
            [
                ("2026-07-08", "2026-07-09"),
                ("2026-07-07", "2026-07-08"),
                ("2026-07-06", "2026-07-07"),
                ("2026-07-03", "2026-07-06"),
            ],
        )

    def test_catdesk_report_generated_after_morning_cutoff_is_backfilled_sample(self):
        module = load_module()
        payload = {"date": "2026-07-08", "generated_at": "2026-07-08T18:43:30+08:00"}

        quality = module.classify_catdesk_report(payload, "2026-07-08")

        self.assertEqual(quality["catdesk_report_status"], "backfilled_sample")
        self.assertIn("after 2026-07-08T09:30:00+08:00", quality["catdesk_report_reason"])

    def test_market_regime_path_uses_latest_file_on_or_before_signal_date(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = Path(temp_dir)
            old = reports / "market_regime_20260706_212319.json"
            future_latest = reports / "market_regime_latest.json"
            old.write_text("{}", encoding="utf-8")
            future_latest.write_text("{}", encoding="utf-8")

            path, status = module.find_market_regime_path(reports, "2026-07-06")

        self.assertEqual(path, old)
        self.assertEqual(status["market_regime_status"], "as_of_signal_date")

    def test_market_regime_does_not_fall_forward_to_future_latest(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = Path(temp_dir)
            future_latest = reports / "market_regime_latest.json"
            future_latest.write_text("{}", encoding="utf-8")

            path, status = module.find_market_regime_path(reports, "2026-07-06")

        self.assertEqual(path, reports / "market_regime_20260706_missing.json")
        self.assertEqual(status["market_regime_status"], "missing_as_of_signal_date")

    def test_qlib_scores_are_limited_to_signal_date_or_earlier(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [
                    pd.Timestamp("2026-07-03"),
                    pd.Timestamp("2026-07-08"),
                ],
                "instrument": ["SZ000001", "SZ000001"],
                "score": [0.1, 0.9],
            }
        )
        module.find_prediction_path = lambda: Path("/tmp/pred.pkl")
        module.load_prediction_scores = lambda _path: frame

        payload = module.load_latest_qlib_scores("2026-07-05", top_reference_date="2026-07-05")

        self.assertTrue(payload["available"])
        self.assertEqual(payload["latest_date"], "2026-07-03")
        self.assertEqual(payload["scores"], {"SZ000001": 0.1})

    def test_qlib_scores_are_unavailable_when_only_future_scores_exist(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        module.find_prediction_path = lambda: Path("/tmp/pred.pkl")
        module.load_prediction_scores = lambda _path: frame

        payload = module.load_latest_qlib_scores("2026-07-05", top_reference_date="2026-07-05")

        self.assertFalse(payload["available"])
        self.assertIn("No Qlib prediction rows", payload["reason"])

    def test_qlib_scores_reject_run_generated_after_validation_cutoff(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        module.find_prediction_path = lambda: Path("/tmp/mlruns/1/run_1/artifacts/pred.pkl")
        module.load_prediction_scores = lambda _path: frame
        module.load_qlib_run_metadata = lambda _path: {
            "run_id": "run_1",
            "run_generated_at": "2026-07-09T14:22:49+08:00",
        }

        payload = module.load_latest_qlib_scores(
            "2026-07-08",
            top_reference_date="2026-07-08",
            run_cutoff="2026-07-09T09:30:00+08:00",
        )

        self.assertFalse(payload["available"])
        self.assertEqual(payload["run_id"], "run_1")
        self.assertIn("after validation cutoff", payload["reason"])

    def test_qlib_scores_reject_when_strict_cutoff_has_no_generation_time(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        module.find_prediction_path = lambda: Path("/tmp/mlruns/1/run_1/artifacts/pred.pkl")
        module.load_prediction_scores = lambda _path: frame
        module.load_qlib_run_metadata = lambda _path: {"run_id": "run_1"}

        payload = module.load_latest_qlib_scores(
            "2026-07-08",
            top_reference_date="2026-07-08",
            run_cutoff="2026-07-09T09:30:00+08:00",
        )

        self.assertFalse(payload["available"])
        self.assertEqual(payload["run_id"], "run_1")
        self.assertIn("generation time is unavailable", payload["reason"])

    def test_qlib_scores_accept_meta_yaml_end_time_before_validation_cutoff(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        before_cutoff_ms = int(pd.Timestamp("2026-07-09T08:00:00+08:00").timestamp() * 1000)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "mlruns" / "1" / "run_meta"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            pred_path = artifacts / "pred.pkl"
            pred_path.write_bytes(b"pred")
            (run_dir / "meta.yaml").write_text(
                f"run_id: run_meta\nstart_time: {before_cutoff_ms - 1000}\nend_time: {before_cutoff_ms}\n",
                encoding="utf-8",
            )
            module.find_prediction_path = lambda: pred_path
            module.load_prediction_scores = lambda _path: frame

            payload = module.load_latest_qlib_scores(
                "2026-07-08",
                top_reference_date="2026-07-08",
                run_cutoff="2026-07-09T09:30:00+08:00",
            )

        self.assertTrue(payload["available"])
        self.assertEqual(payload["run_id"], "run_meta")
        self.assertEqual(payload["latest_date"], "2026-07-08")

    def test_qlib_scores_reject_meta_yaml_end_time_after_validation_cutoff(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        after_cutoff_ms = int(pd.Timestamp("2026-07-09T14:00:00+08:00").timestamp() * 1000)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "mlruns" / "1" / "run_late"
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True)
            pred_path = artifacts / "pred.pkl"
            pred_path.write_bytes(b"pred")
            (run_dir / "meta.yaml").write_text(
                f"run_id: run_late\nstart_time: {after_cutoff_ms - 1000}\nend_time: {after_cutoff_ms}\n",
                encoding="utf-8",
            )
            module.find_prediction_path = lambda: pred_path
            module.load_prediction_scores = lambda _path: frame

            payload = module.load_latest_qlib_scores(
                "2026-07-08",
                top_reference_date="2026-07-08",
                run_cutoff="2026-07-09T09:30:00+08:00",
            )

        self.assertFalse(payload["available"])
        self.assertEqual(payload["run_id"], "run_late")
        self.assertIn("after validation cutoff", payload["reason"])

    def test_generate_report_defaults_to_strict_qlib_run_cutoff(self):
        captured, payload = self._generate_minimal_report(allow_backfilled_qlib=False)

        self.assertEqual(captured["signal_date"], "2026-07-08")
        self.assertEqual(captured["top_reference_date"], "2026-07-08")
        self.assertEqual(captured["run_cutoff"], "2026-07-09T09:30:00+08:00")
        self.assertEqual(payload["input_quality"]["qlib_run_cutoff"], "2026-07-09T09:30:00+08:00")
        self.assertFalse(payload["input_quality"]["qlib_backfilled_allowed"])
        self.assertFalse(payload["inputs"]["qlib_backfilled_allowed"])

    def test_generate_report_marks_explicit_backfilled_qlib_mode(self):
        captured, payload = self._generate_minimal_report(allow_backfilled_qlib=True)

        self.assertEqual(captured["run_cutoff"], "")
        self.assertEqual(payload["input_quality"]["qlib_run_cutoff"], "")
        self.assertTrue(payload["input_quality"]["qlib_backfilled_allowed"])
        self.assertTrue(payload["inputs"]["qlib_backfilled_allowed"])

    def test_qlib_scores_can_be_used_without_cutoff_for_backfilled_research(self):
        module = load_module()
        frame = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2026-07-08")],
                "instrument": ["SZ000001"],
                "score": [0.9],
            }
        )
        module.find_prediction_path = lambda: Path("/tmp/mlruns/1/run_1/artifacts/pred.pkl")
        module.load_prediction_scores = lambda _path: frame
        module.load_qlib_run_metadata = lambda _path: {
            "run_id": "run_1",
            "run_generated_at": "2026-07-09T14:22:49+08:00",
        }

        payload = module.load_latest_qlib_scores(
            "2026-07-08",
            top_reference_date="2026-07-08",
            run_cutoff="",
        )

        self.assertTrue(payload["available"])
        self.assertEqual(payload["latest_date"], "2026-07-08")
        self.assertEqual(payload["scores"], {"SZ000001": 0.9})

    def test_cli_passes_allow_backfilled_qlib_to_single_report(self):
        module = load_module()
        captured = {}

        def fake_generate_report(**kwargs):
            captured.update(kwargs)
            return {
                "selected": [],
                "all_candidates": [],
                "selected_strategy_label": "主线题材桶",
                "selected_strategy": "dominant_theme_bucket",
                "selected_summary": {},
                "raw_summary": {},
                "artifacts": {"latest_markdown": "latest.md", "latest_json": "latest.json"},
            }

        module.generate_report = fake_generate_report
        argv = [
            "report_catdesk_signal_validation.py",
            "--allow-backfilled-qlib",
            "--no-online-validation",
        ]

        with patch.object(module.sys, "argv", argv):
            exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["allow_backfilled_qlib"])

    def test_cli_passes_allow_backfilled_qlib_to_rolling_report(self):
        module = load_module()
        captured = {}

        def fake_generate_rolling_report(**kwargs):
            captured.update(kwargs)
            return {
                "items": [{"status": "ok"}],
                "artifacts": {"latest_markdown": "rolling.md", "latest_json": "rolling.json"},
            }

        module.generate_rolling_report = fake_generate_rolling_report
        argv = [
            "report_catdesk_signal_validation.py",
            "--rolling-days",
            "2",
            "--allow-backfilled-qlib",
            "--no-online-validation",
        ]

        with patch.object(module.sys, "argv", argv):
            exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["allow_backfilled_qlib"])


if __name__ == "__main__":
    unittest.main()
