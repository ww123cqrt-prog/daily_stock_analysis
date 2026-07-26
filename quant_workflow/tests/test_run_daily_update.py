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
        ashare_source="none",
        overlap_days=7,
        sleep=2.0,
        timeout=60,
        retries=1,
        max_consecutive_failures=3,
        max_requests=6000,
        limit=0,
        end="20260708",
        skip_baostock=False,
        skip_eastmoney=False,
        skip_akshare=False,
        skip_global_daily=False,
        skip_ashare_news=False,
        skip_snapshot_validation=False,
        use_system_proxy=False,
        snapshot_limit=50,
        news_limit=0,
        overlap_sample=0,
        merged_sample=0,
        min_source_coverage=0.8,
        benchmark_cache_dir=None,
        benchmark_offline=True,
        quality_min_calendar_days=252,
        quality_min_instruments=100,
        skip_if_robust_training=False,
        dry_run=False,
        continue_on_error=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _snapshot(day: str, *, eligible: bool = True, ratio: float = 1.0):
    return {
        "candidate_date": day,
        "expected_symbols": 1,
        "covered_symbols": 1 if eligible else 0,
        "coverage_ratio": ratio,
        "minimum_ratio": 0.8,
        "eligible": eligible,
        "source_dates": {"baostock": day},
        "sources": {},
    }


class ShouldRunHeavyTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_forced_always_runs(self):
        self.assertTrue(
            self.module.should_run_heavy(force=True, download_ran=True, before="2026-07-03", after="2026-07-03")
        )

    def test_skipped_download_runs_rebuild(self):
        self.assertTrue(
            self.module.should_run_heavy(force=False, download_ran=False, before="2026-07-03", after="2026-07-03")
        )

    def test_new_trading_day_runs(self):
        self.assertTrue(
            self.module.should_run_heavy(force=False, download_ran=True, before="2026-07-03", after="2026-07-04")
        )

    def test_no_new_trading_day_skips(self):
        self.assertFalse(
            self.module.should_run_heavy(force=False, download_ran=True, before="2026-07-03", after="2026-07-03")
        )

    def test_missing_dates_are_conservative(self):
        self.assertTrue(
            self.module.should_run_heavy(force=False, download_ran=True, before="", after="")
        )


class BuildStagesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_full_chain_includes_download_first(self):
        stages = self.module.build_stages(_args(ashare_source="baostock", news_limit=50))
        names = [s.name for s in stages]
        self.assertEqual(names[0], "download_baostock")
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertIn("download_global_daily", names)
        self.assertIn("download_ashare_news", names)
        self.assertIn("merge_to_qlib", names)
        self.assertIn("add_index_benchmark", names)
        self.assertIn("validate_latest_snapshot", names)
        self.assertIn("apply_data_cleaning", names)
        self.assertIn("check_qlib_data", names)
        self.assertIn("publish_official", names)
        self.assertNotIn("sync_data_manifest", names)
        # anomaly detection must precede cleaning (cleaning consumes its output)
        self.assertLess(names.index("report_data_anomalies"), names.index("apply_data_cleaning"))
        self.assertLess(names.index("add_index_benchmark"), names.index("apply_data_cleaning"))
        self.assertLess(names.index("check_qlib_data"), names.index("publish_official"))

    def test_skip_download_omits_download_stage(self):
        stages = self.module.build_stages(_args(skip_download=True))
        names = [s.name for s in stages]
        self.assertNotIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertEqual(names[0], "merge_to_qlib")

    def test_download_stage_is_incremental_nonfatal_and_not_heavy(self):
        stages = self.module.build_stages(_args(ashare_source="baostock", limit=5))
        download = next(s for s in stages if s.name == "download_baostock")
        self.assertFalse(download.heavy)
        # Baostock download is non-fatal: merge reads on-disk parquet, so a
        # transient download failure must not block the rebuild.
        self.assertFalse(download.required)
        self.assertIn("--incremental", download.argv)
        self.assertIn("--limit", download.argv)
        self.assertIn("5", download.argv)
        self.assertEqual(download.argv[download.argv.index("--max-requests") + 1], "6000")

    def test_eastmoney_family_sources_are_mutually_exclusive(self):
        for source, name in [("eastmoney", "download_eastmoney"), ("akshare", "download_akshare")]:
            stages = self.module.build_stages(_args(ashare_source=source, limit=5))
            names = [stage.name for stage in stages]
            self.assertEqual(len({item for item in names if item in {"download_eastmoney", "download_akshare"}}), 1)
            stage = next(s for s in stages if s.name == name)
            self.assertFalse(stage.heavy)
            self.assertFalse(stage.required)
            self.assertIn("--incremental", stage.argv)
            self.assertIn("--overlap-days", stage.argv)
            self.assertIn("--limit", stage.argv)
            self.assertIn("5", stage.argv)

        stages = self.module.build_stages(_args(ashare_source="eastmoney", limit=5))
        global_daily = next(s for s in stages if s.name == "download_global_daily")
        self.assertFalse(global_daily.heavy)
        self.assertFalse(global_daily.required)
        self.assertIn("--incremental", global_daily.argv)
        self.assertIn("--limit", global_daily.argv)

        self.assertNotIn("download_ashare_news", [stage.name for stage in stages])

    def test_default_arguments_quarantine_blocked_ashare_sources(self):
        args = self.module.parse_args([])

        self.assertEqual(args.ashare_source, "none")
        self.assertEqual(args.sleep, 2.0)
        self.assertEqual(args.retries, 1)
        self.assertEqual(args.max_consecutive_failures, 3)
        self.assertEqual(args.max_requests, 6000)
        self.assertEqual(args.overlap_days, 7)
        self.assertEqual(args.news_limit, 0)
        names = [stage.name for stage in self.module.build_stages(args)]
        self.assertFalse(set(names) & self.module.ASHARE_DOWNLOAD_STAGE_NAMES)
        self.assertNotIn("download_ashare_news", names)
        benchmark = next(stage for stage in self.module.build_stages(args) if stage.name == "add_index_benchmark")
        self.assertTrue(args.benchmark_offline)
        self.assertIn("--offline", benchmark.argv)

    def test_rebuild_stages_marked_heavy(self):
        stages = self.module.build_stages(_args())
        heavy = {s.name for s in stages if s.heavy}
        self.assertEqual(
            heavy,
            {"merge_to_qlib", "report_data_anomalies", "apply_data_cleaning",
             "report_data_quality", "validate_latest_snapshot", "check_qlib_data",
             "add_index_benchmark", "publish_official"},
        )
        publish = next(s for s in stages if s.name == "publish_official")
        self.assertTrue(publish.heavy)
        self.assertTrue(publish.required)

    def test_source_downloads_can_be_individually_disabled(self):
        stages = self.module.build_stages(
            _args(ashare_source="baostock", skip_baostock=True, skip_snapshot_validation=True)
        )
        names = [s.name for s in stages]

        self.assertNotIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertNotIn("validate_latest_snapshot", names)

    def test_system_proxy_flag_only_applies_to_public_sources(self):
        baostock_stages = self.module.build_stages(_args(ashare_source="baostock", use_system_proxy=True))
        eastmoney_stages = self.module.build_stages(_args(ashare_source="eastmoney", use_system_proxy=True))
        akshare_stages = self.module.build_stages(_args(ashare_source="akshare", use_system_proxy=True))
        baostock = next(s for s in baostock_stages if s.name == "download_baostock")
        eastmoney = next(s for s in eastmoney_stages if s.name == "download_eastmoney")
        akshare = next(s for s in akshare_stages if s.name == "download_akshare")
        snapshot = next(s for s in eastmoney_stages if s.name == "validate_latest_snapshot")

        self.assertNotIn("--use-system-proxy", baostock.argv)
        self.assertIn("--use-system-proxy", eastmoney.argv)
        self.assertIn("--use-system-proxy", akshare.argv)
        self.assertIn("--use-system-proxy", snapshot.argv)


class LatestSourceDateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_returns_empty_for_missing_dir(self):
        self.assertEqual(self.module.latest_source_date(Path("/no/such/dir")), "")

    def test_returns_max_date_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-07-03"])}).to_parquet(d / "000001.parquet")
            pd.DataFrame({"date": pd.to_datetime(["2026-06-30", "2026-07-02"])}).to_parquet(d / "600000.parquet")
            self.assertEqual(self.module.latest_source_date(d), "2026-07-03")

    def test_latest_any_source_date_returns_max_date_across_source_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            baostock = base / "baostock"
            eastmoney = base / "eastmoney"
            baostock.mkdir()
            eastmoney.mkdir()
            pd.DataFrame({"date": pd.to_datetime(["2026-07-03"])}).to_parquet(baostock / "000001.parquet")
            pd.DataFrame({"date": pd.to_datetime(["2026-07-08"])}).to_parquet(eastmoney / "000001.parquet")

            latest, by_source = self.module.latest_any_source_date({"baostock": baostock, "eastmoney": eastmoney})

        self.assertEqual(latest, "2026-07-08")
        self.assertEqual(by_source["baostock"], "2026-07-03")
        self.assertEqual(by_source["eastmoney"], "2026-07-08")


class SourceRefreshFailureGateTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_non_ashare_download_failure_does_not_trigger_ashare_refresh_gate(self):
        results = [
            self.module.StageResult("download_baostock", "ok", 0, 0, required=False),
            self.module.StageResult("download_eastmoney", "ok", 0, 0, required=False),
            self.module.StageResult("download_akshare", "ok", 0, 0, required=False),
            self.module.StageResult("download_global_daily", "failed", 0, 1, required=False),
            self.module.StageResult("download_ashare_news", "failed", 0, 1, required=False),
            self.module.StageResult("merge_to_qlib", "skipped", 0, 0, required=True),
        ]

        self.assertFalse(self.module.source_refresh_failed_without_rebuild(results))

    def test_all_ashare_downloads_failed_still_triggers_gate_with_global_success(self):
        results = [
            self.module.StageResult(name, "failed", 0, 1, required=False)
            for name in self.module.ASHARE_DOWNLOAD_STAGE_NAMES
        ] + [
            self.module.StageResult("download_global_daily", "ok", 0, 0, required=False),
            self.module.StageResult("merge_to_qlib", "skipped", 0, 0, required=True),
        ]

        self.assertTrue(self.module.source_refresh_failed_without_rebuild(results))


class SourceCoverageTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_one_new_symbol_cannot_pass_market_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "akshare"
            source.mkdir()
            symbols = [f"{index:06d}" for index in range(10)]
            for index, symbol in enumerate(symbols):
                day = "2026-07-21" if index == 0 else "2026-07-20"
                pd.DataFrame({"date": [day]}).to_parquet(source / f"{symbol}.parquet")

            coverage = self.module.source_coverage_snapshot(
                {"akshare": source},
                expected_symbols=set(symbols),
                minimum_ratio=0.8,
            )

        self.assertEqual(coverage["candidate_date"], "2026-07-21")
        self.assertEqual(coverage["covered_symbols"], 1)
        self.assertFalse(coverage["eligible"])

    def test_one_complete_source_can_pass_while_other_sources_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            complete = base / "eastmoney"
            failed_a = base / "baostock"
            failed_b = base / "akshare"
            for directory in [complete, failed_a, failed_b]:
                directory.mkdir()
            symbols = [f"{index:06d}" for index in range(10)]
            for symbol in symbols:
                pd.DataFrame({"date": ["2026-07-21"]}).to_parquet(complete / f"{symbol}.parquet")

            coverage = self.module.source_coverage_snapshot(
                {"baostock": failed_a, "eastmoney": complete, "akshare": failed_b},
                expected_symbols=set(symbols),
                minimum_ratio=0.8,
            )

        self.assertTrue(coverage["eligible"])
        self.assertEqual(coverage["coverage_ratio"], 1.0)


class DryRunMainTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_dry_run_writes_summary_and_runs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(self.module, "REPORTS_ROOT", reports), \
                    patch.object(self.module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(self.module, "source_coverage_snapshot", return_value=_snapshot("2026-07-03")), \
                    patch.object(sys, "argv", ["run_daily_update.py", "--dry-run"]), \
                    patch.object(self.module.subprocess, "run") as mock_run:
                rc = self.module.main()
            self.assertEqual(rc, 0)
            mock_run.assert_not_called()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["status"], "planned")
            self.assertTrue(all(s["status"] == "planned" for s in summary["stages"]))
            # dry-run previews the full pipeline regardless of the trading-day gate
            self.assertTrue(any(s["name"] == "apply_data_cleaning" for s in summary["stages"]))

    def test_duplicate_trigger_records_controlled_skip(self):
        module = self.module
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            held = module.LockHeldError({"pid": 4242, "run_id": "existing"})
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module.DailyUpdateLock, "acquire", side_effect=held), \
                    patch.object(sys, "argv", ["run_daily_update.py", "--dry-run"]):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))
            runtime = json.loads((reports / "runtime" / "status" / "daily_update_status_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "controlled_skip")
        self.assertEqual(runtime["reason"], "already_running")
        self.assertFalse(runtime["official_published"])
        self.assertFalse(runtime["manifest_synced"])

    def test_active_robust_training_records_controlled_skip_before_any_stage(self):
        module = self.module
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "robust_training_active", return_value=True), \
                    patch.object(module, "execute_pipeline") as execute, \
                    patch.object(sys, "argv", ["run_daily_update.py", "--skip-if-robust-training"]):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))
            runtime = json.loads(
                (reports / "runtime" / "status" / "daily_update_status_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "controlled_skip")
        self.assertEqual(summary["reason"], "robust_training_active")
        self.assertFalse(runtime["official_published"])
        self.assertFalse(runtime["manifest_synced"])
        execute.assert_not_called()

    def test_robust_training_detector_catches_main_project_research_worker(self):
        module = self.module
        completed = module.subprocess.CompletedProcess(
            args=["/bin/ps", "-axo", "command="],
            returncode=0,
            stdout=(
                "/Users/cq/WorkingProjects/daily_stock_analysis/.venv/bin/python "
                "quant_workflow/scripts/run_robust_signal_research.py --config smoke.yaml\n"
            ),
            stderr="",
        )
        with patch.object(module.subprocess, "run", return_value=completed):
            active = module.robust_training_active()

        self.assertTrue(active)

    def test_optional_source_download_failure_does_not_fail_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "download_eastmoney" else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(
                        sys,
                        "argv",
                        ["run_daily_update.py", "--ashare-source", "eastmoney", "--continue-on-error"],
                    ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok_with_warnings")
        self.assertTrue(any(s["name"] == "download_eastmoney" and s["status"] == "failed" for s in summary["stages"]))

    def test_one_selected_source_can_publish_when_coverage_passes(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            return module.StageResult(stage.name, "ok", 0.0, 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(
                        sys,
                        "argv",
                        ["run_daily_update.py", "--ashare-source", "eastmoney", "--continue-on-error"],
                    ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(any(item["name"] == "publish_official" and item["status"] == "ok" for item in summary["stages"]))

    def test_cleaning_failure_never_reaches_official_publish(self):
        module = self.module
        called: list[str] = []

        def fake_run_stage(stage, dry_run):
            called.append(stage.name)
            failed = stage.name == "apply_data_cleaning"
            return module.StageResult(stage.name, "failed" if failed else "ok", 0.0, 1 if failed else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(sys, "argv", ["run_daily_update.py"]):
                rc = module.main()

        self.assertEqual(rc, 1)
        self.assertIn("apply_data_cleaning", called)
        self.assertNotIn("publish_official", called)

    def test_all_source_downloads_failing_is_nonfatal_when_forced_rebuild_still_runs(self):
        # Redundancy contract: every source download is non-fatal. Even if all
        # of them fail on the network, the on-disk parquet is intact, so the
        # rebuild (merge/clean/quality) must still run and the run must NOT
        # fail — it completes with ok_with_warnings.
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-03")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(
                        sys,
                        "argv",
                        ["run_daily_update.py", "--ashare-source", "baostock", "--force"],
                    ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok_with_warnings")
        # baostock download failure is recorded but non-required (non-fatal)
        self.assertTrue(any(
            s["name"] == "download_baostock" and not s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))
        # rebuild still ran (forced), proving a dead source did not block it
        self.assertTrue(any(s["name"] == "merge_to_qlib" and s["status"] == "ok" for s in summary["stages"]))

    def test_all_source_downloads_failing_without_rebuild_fails_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-03")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(sys, "argv", ["run_daily_update.py", "--ashare-source", "baostock"]):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any(
            s["name"] == "source_refresh_gate" and s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))

    def test_publish_failure_is_required_and_fails_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "publish_official" else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(module, "REPORTS_ROOT", reports), \
                    patch.object(module, "load_expected_symbols", return_value={"000001"}), \
                    patch.object(module, "source_coverage_snapshot", side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")]), \
                    patch.object(module, "run_stage", side_effect=fake_run_stage), \
                    patch.object(sys, "argv", ["run_daily_update.py"]):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any(
            s["name"] == "publish_official" and s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))


if __name__ == "__main__":
    unittest.main()
