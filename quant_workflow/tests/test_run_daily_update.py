import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_update.py"
    spec = importlib.util.spec_from_file_location("run_daily_update", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so @dataclass KW_ONLY resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    base = dict(
        skip_download=False,
        force=False,
        overlap_days=45,
        sleep=0.3,
        timeout=60,
        retries=3,
        max_consecutive_failures=50,
        limit=0,
        end="20260708",
        skip_eastmoney=False,
        skip_akshare=False,
        skip_snapshot_validation=False,
        skip_robust_shadow=False,
        use_system_proxy=False,
        snapshot_limit=50,
        overlap_sample=0,
        merged_sample=0,
        dry_run=False,
        continue_on_error=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class ShouldRunHeavyTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_forced_always_runs(self):
        self.assertTrue(
            self.module.should_run_heavy(
                force=True, download_ran=True, before="2026-07-03", after="2026-07-03"
            )
        )

    def test_skipped_download_runs_rebuild(self):
        self.assertTrue(
            self.module.should_run_heavy(
                force=False, download_ran=False, before="2026-07-03", after="2026-07-03"
            )
        )

    def test_new_trading_day_runs(self):
        self.assertTrue(
            self.module.should_run_heavy(
                force=False, download_ran=True, before="2026-07-03", after="2026-07-04"
            )
        )

    def test_no_new_trading_day_skips(self):
        self.assertFalse(
            self.module.should_run_heavy(
                force=False, download_ran=True, before="2026-07-03", after="2026-07-03"
            )
        )

    def test_missing_dates_are_conservative(self):
        self.assertTrue(
            self.module.should_run_heavy(
                force=False, download_ran=True, before="", after=""
            )
        )


class BuildStagesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_full_chain_includes_download_first(self):
        stages = self.module.build_stages(_args())
        names = [s.name for s in stages]
        self.assertEqual(names[0], "download_baostock")
        self.assertIn("download_eastmoney", names)
        self.assertIn("download_akshare", names)
        self.assertIn("merge_to_qlib", names)
        self.assertIn("add_index_benchmark", names)
        self.assertIn("validate_latest_snapshot", names)
        self.assertIn("apply_data_cleaning", names)
        self.assertIn("check_qlib_data", names)
        self.assertIn("sync_data_manifest", names)
        # anomaly detection must precede cleaning (cleaning consumes its output)
        self.assertLess(
            names.index("report_data_anomalies"), names.index("apply_data_cleaning")
        )
        self.assertLess(
            names.index("add_index_benchmark"), names.index("apply_data_cleaning")
        )
        self.assertLess(
            names.index("check_qlib_data"), names.index("sync_data_manifest")
        )

    def test_skip_download_omits_download_stage(self):
        stages = self.module.build_stages(_args(skip_download=True))
        names = [s.name for s in stages]
        self.assertNotIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertEqual(names[0], "merge_to_qlib")

    def test_download_stage_is_incremental_nonfatal_and_not_heavy(self):
        stages = self.module.build_stages(_args(limit=5))
        download = next(s for s in stages if s.name == "download_baostock")
        self.assertFalse(download.heavy)
        # Baostock download is non-fatal: merge reads on-disk parquet, so a
        # transient download failure must not block the rebuild.
        self.assertFalse(download.required)
        self.assertIn("--incremental", download.argv)
        self.assertIn("--limit", download.argv)
        self.assertIn("5", download.argv)

    def test_new_public_source_downloads_are_incremental_and_nonfatal(self):
        stages = self.module.build_stages(_args(limit=5))
        for name in ["download_eastmoney", "download_akshare"]:
            stage = next(s for s in stages if s.name == name)
            self.assertFalse(stage.heavy)
            self.assertFalse(stage.required)
            self.assertIn("--incremental", stage.argv)
            self.assertIn("--overlap-days", stage.argv)
            self.assertIn("--limit", stage.argv)
            self.assertIn("5", stage.argv)

    def test_rebuild_stages_marked_heavy(self):
        stages = self.module.build_stages(_args())
        heavy = {s.name for s in stages if s.heavy}
        self.assertEqual(
            heavy,
            {
                "merge_to_qlib",
                "report_data_anomalies",
                "apply_data_cleaning",
                "report_data_quality",
                "validate_latest_snapshot",
                "check_qlib_data",
                "add_index_benchmark",
            },
        )
        manifest = next(s for s in stages if s.name == "sync_data_manifest")
        self.assertFalse(manifest.heavy)
        self.assertFalse(manifest.required)

    def test_source_downloads_can_be_individually_disabled(self):
        stages = self.module.build_stages(
            _args(skip_eastmoney=True, skip_akshare=True, skip_snapshot_validation=True)
        )
        names = [s.name for s in stages]

        self.assertIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertNotIn("validate_latest_snapshot", names)

    def test_system_proxy_flag_only_applies_to_public_sources(self):
        stages = self.module.build_stages(_args(use_system_proxy=True))
        baostock = next(s for s in stages if s.name == "download_baostock")
        eastmoney = next(s for s in stages if s.name == "download_eastmoney")
        akshare = next(s for s in stages if s.name == "download_akshare")
        snapshot = next(s for s in stages if s.name == "validate_latest_snapshot")

        self.assertNotIn("--use-system-proxy", baostock.argv)
        self.assertIn("--use-system-proxy", eastmoney.argv)
        self.assertIn("--use-system-proxy", akshare.argv)
        self.assertIn("--use-system-proxy", snapshot.argv)

    def test_active_robust_shadow_is_appended_after_manifest_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            robust = reports / "robust_strategy"
            robust.mkdir(parents=True)
            (robust / "confirmation_latest.json").write_text(
                json.dumps({"status": "shadow_observation"}), encoding="utf-8"
            )
            (robust / "frozen_strategy_latest.json").write_text(
                json.dumps(
                    {
                        "confirmation_status": "shadow_observation",
                        "shadow_tracking_started": True,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(self.module, "REPORTS_ROOT", reports):
                stages = self.module.build_stages(_args(end="20260716"))

        names = [stage.name for stage in stages]
        self.assertEqual(names[-2:], ["sync_data_manifest", "run_robust_shadow_day"])
        shadow = stages[-1]
        self.assertFalse(shadow.heavy)
        self.assertFalse(shadow.required)
        self.assertIn("run_robust_shadow_day.py", shadow.argv[1])
        self.assertEqual(shadow.argv[shadow.argv.index("--as-of-date") + 1], "20260716")
        self.assertEqual(
            shadow.argv[shadow.argv.index("--frozen") + 1],
            str(robust / "frozen_strategy_latest.json"),
        )

    def test_robust_shadow_is_omitted_when_not_confirmed_skipped_or_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            robust = reports / "robust_strategy"
            robust.mkdir(parents=True)
            with patch.object(self.module, "REPORTS_ROOT", reports):
                self.assertNotIn(
                    "run_robust_shadow_day",
                    [stage.name for stage in self.module.build_stages(_args())],
                )

                (robust / "confirmation_latest.json").write_text(
                    json.dumps({"status": "shadow_observation"}), encoding="utf-8"
                )
                (robust / "frozen_strategy_latest.json").write_text(
                    json.dumps(
                        {
                            "confirmation_status": "shadow_observation",
                            "shadow_tracking_started": True,
                        }
                    ),
                    encoding="utf-8",
                )
                (robust / "shadow_state_latest.json").write_text(
                    json.dumps({"status": "robust_strategy_qualified"}),
                    encoding="utf-8",
                )
                self.assertNotIn(
                    "run_robust_shadow_day",
                    [stage.name for stage in self.module.build_stages(_args())],
                )
                self.assertNotIn(
                    "run_robust_shadow_day",
                    [
                        stage.name
                        for stage in self.module.build_stages(
                            _args(skip_robust_shadow=True)
                        )
                    ],
                )


class LatestSourceDateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_returns_empty_for_missing_dir(self):
        self.assertEqual(self.module.latest_source_date(Path("/no/such/dir")), "")

    def test_returns_max_date_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd.DataFrame(
                {"date": pd.to_datetime(["2026-07-01", "2026-07-03"])}
            ).to_parquet(d / "000001.parquet")
            pd.DataFrame(
                {"date": pd.to_datetime(["2026-06-30", "2026-07-02"])}
            ).to_parquet(d / "600000.parquet")
            self.assertEqual(self.module.latest_source_date(d), "2026-07-03")

    def test_latest_any_source_date_returns_max_date_across_source_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            baostock = base / "baostock"
            eastmoney = base / "eastmoney"
            baostock.mkdir()
            eastmoney.mkdir()
            pd.DataFrame({"date": pd.to_datetime(["2026-07-03"])}).to_parquet(
                baostock / "000001.parquet"
            )
            pd.DataFrame({"date": pd.to_datetime(["2026-07-08"])}).to_parquet(
                eastmoney / "000001.parquet"
            )

            latest, by_source = self.module.latest_any_source_date(
                {"baostock": baostock, "eastmoney": eastmoney}
            )

        self.assertEqual(latest, "2026-07-08")
        self.assertEqual(by_source["baostock"], "2026-07-03")
        self.assertEqual(by_source["eastmoney"], "2026-07-08")


class DryRunMainTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_dry_run_writes_summary_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(self.module, "REPORTS_ROOT", reports),
                patch.object(sys, "argv", ["run_daily_update.py", "--dry-run"]),
                patch.object(self.module.subprocess, "run") as mock_run,
            ):
                rc = self.module.main()
            self.assertEqual(rc, 0)
            mock_run.assert_not_called()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(summary["dry_run"])
            self.assertTrue(all(s["status"] == "planned" for s in summary["stages"]))
            # dry-run previews the full pipeline regardless of the trading-day gate
            self.assertTrue(
                any(s["name"] == "apply_data_cleaning" for s in summary["stages"])
            )

    def test_optional_source_download_failure_does_not_fail_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "download_eastmoney" else "ok"
            return module.StageResult(
                stage.name,
                status,
                0.0,
                1 if status == "failed" else 0,
                required=stage.required,
            )

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(
                    module,
                    "latest_any_source_date",
                    side_effect=[
                        ("2026-07-03", {}),
                        ("2026-07-08", {}),
                        ("2026-07-08", {}),
                    ],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys, "argv", ["run_daily_update.py", "--continue-on-error"]
                ),
            ):
                rc = module.main()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok_with_warnings")
        self.assertTrue(
            any(
                s["name"] == "download_eastmoney" and s["status"] == "failed"
                for s in summary["stages"]
            )
        )

    def test_all_source_downloads_failing_is_nonfatal_when_forced_rebuild_still_runs(
        self,
    ):
        # Redundancy contract: every source download is non-fatal. Even if all
        # of them fail on the network, the on-disk parquet is intact, so the
        # rebuild (merge/clean/quality) must still run and the run must NOT
        # fail — it completes with ok_with_warnings.
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(
                stage.name,
                status,
                0.0,
                1 if status == "failed" else 0,
                required=stage.required,
            )

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(
                    module,
                    "latest_any_source_date",
                    side_effect=[
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                    ],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(sys, "argv", ["run_daily_update.py", "--force"]),
            ):
                rc = module.main()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok_with_warnings")
        # baostock download failure is recorded but non-required (non-fatal)
        self.assertTrue(
            any(
                s["name"] == "download_baostock"
                and not s["required"]
                and s["status"] == "failed"
                for s in summary["stages"]
            )
        )
        # rebuild still ran (forced), proving a dead source did not block it
        self.assertTrue(
            any(
                s["name"] == "merge_to_qlib" and s["status"] == "ok"
                for s in summary["stages"]
            )
        )

    def test_all_source_downloads_failing_without_rebuild_fails_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(
                stage.name,
                status,
                0.0,
                1 if status == "failed" else 0,
                required=stage.required,
            )

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(
                    module,
                    "latest_any_source_date",
                    side_effect=[
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                    ],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(sys, "argv", ["run_daily_update.py"]),
            ):
                rc = module.main()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(
            any(
                s["name"] == "source_refresh_gate"
                and s["required"]
                and s["status"] == "failed"
                for s in summary["stages"]
            )
        )

    def test_manifest_sync_failure_is_warning_not_daily_update_failure(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "sync_data_manifest" else "ok"
            return module.StageResult(
                stage.name,
                status,
                0.0,
                1 if status == "failed" else 0,
                required=stage.required,
            )

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(
                    module,
                    "latest_any_source_date",
                    side_effect=[
                        ("2026-07-03", {"baostock": "2026-07-03"}),
                        ("2026-07-08", {"baostock": "2026-07-08"}),
                        ("2026-07-08", {"baostock": "2026-07-08"}),
                    ],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(sys, "argv", ["run_daily_update.py"]),
            ):
                rc = module.main()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok_with_warnings")
        self.assertTrue(
            any(
                s["name"] == "sync_data_manifest"
                and not s["required"]
                and s["status"] == "failed"
                for s in summary["stages"]
            )
        )


if __name__ == "__main__":
    unittest.main()
