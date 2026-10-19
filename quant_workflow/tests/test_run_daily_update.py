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
        skip_etf_momentum=False,
        use_system_proxy=False,
        news_limit=0,
        overlap_sample=0,
        merged_sample=0,
        min_source_coverage=0.8,
        ashare_v2_bundle_dir=None,
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
        self.assertIn("download_etf_momentum", names)
        self.assertIn("validate_cn_etf_daily_release", names)
        self.assertIn("generate_etf_momentum_report", names)
        self.assertIn("validate_ashare_daily_qfq", names)
        self.assertIn("validate_cn_index_daily", names)
        self.assertIn("check_qlib_direct_data", names)
        self.assertIn("check_qlib_direct_cleaned_data", names)
        self.assertNotIn("validate_qlib_ashare_v2", names)
        self.assertNotIn("validate_qlib_ashare_cleaned_v2", names)
        self.assertNotIn("qlib_ashare_v2", " ".join(" ".join(s.argv) for s in stages))
        self.assertIn("report_data_quality", names)
        self.assertIn("sync_data_manifest_check", names)
        self.assertNotIn("merge_to_qlib", names)
        self.assertNotIn("publish_official", names)
        self.assertLess(
            names.index("validate_cn_index_daily"),
            names.index("check_qlib_direct_data"),
        )
        cleaned_check = next(
            stage for stage in stages if stage.name == "check_qlib_direct_cleaned_data"
        )
        self.assertEqual(
            cleaned_check.argv[-2:], ["--provider", "cleaned"]
        )

    def test_skip_download_omits_download_stage(self):
        stages = self.module.build_stages(_args(skip_download=True))
        names = [s.name for s in stages]
        self.assertNotIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertEqual(names[0], "validate_ashare_daily_qfq")

    def test_download_stage_is_incremental_nonfatal(self):
        stages = self.module.build_stages(_args(ashare_source="baostock", limit=5))
        download = next(s for s in stages if s.name == "download_baostock")
        # Source downloads update audit caches only. They never publish V2.
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
            self.assertFalse(stage.required)
            self.assertIn("--incremental", stage.argv)
            self.assertIn("--overlap-days", stage.argv)
            self.assertIn("--limit", stage.argv)
            self.assertIn("5", stage.argv)

        stages = self.module.build_stages(_args(ashare_source="eastmoney", limit=5))
        global_daily = next(s for s in stages if s.name == "download_global_daily")
        self.assertFalse(global_daily.required)
        self.assertIn("--incremental", global_daily.argv)
        self.assertIn("--limit", global_daily.argv)

        self.assertNotIn("download_ashare_news", [stage.name for stage in stages])

    def test_default_arguments_use_miniqmt_and_quarantine_other_ashare_sources(self):
        args = self.module.parse_args([])

        self.assertEqual(args.ashare_source, "miniqmt")
        self.assertEqual(args.sleep, 2.0)
        self.assertEqual(args.retries, 1)
        self.assertEqual(args.max_consecutive_failures, 3)
        self.assertEqual(args.max_requests, 6000)
        self.assertEqual(args.overlap_days, 7)
        self.assertEqual(args.news_limit, 0)
        names = [stage.name for stage in self.module.build_stages(args)]
        self.assertIn("update_ashare_miniqmt", names)
        self.assertFalse(set(names) & self.module.ASHARE_DOWNLOAD_STAGE_NAMES)
        self.assertNotIn("download_ashare_news", names)
        self.assertNotIn("update_ashare_v2_bundle", names)
        self.assertIn("validate_cn_index_daily", names)
        self.assertIn("check_qlib_direct_data", names)
        self.assertIn("check_qlib_direct_cleaned_data", names)

    def test_etf_momentum_update_and_report_are_isolated_nonfatal_stages(self):
        stages = self.module.build_stages(_args())
        update = next(stage for stage in stages if stage.name == "download_etf_momentum")
        validation = next(
            stage for stage in stages if stage.name == "validate_cn_etf_daily_release"
        )
        report = next(
            stage for stage in stages if stage.name == "generate_etf_momentum_report"
        )

        self.assertFalse(update.required)
        self.assertIn("quant_data_cli.py", update.argv[1])
        self.assertIn("update", update.argv)
        self.assertIn("cn_etf_daily", update.argv)
        self.assertIn("--through", update.argv)
        self.assertFalse(validation.required)
        self.assertIn("quant_data_cli.py", validation.argv[1])
        self.assertEqual(validation.argv[-2:], ["cn_etf_daily", "--release"])
        self.assertFalse(report.required)
        self.assertIn("generate_etf_momentum_strategy_report.py", report.argv[1])
        self.assertLess(stages.index(update), stages.index(validation))
        self.assertLess(stages.index(validation), stages.index(report))

    def test_etf_momentum_stages_can_be_skipped(self):
        stages = self.module.build_stages(_args(skip_etf_momentum=True))
        names = [stage.name for stage in stages]
        self.assertNotIn("download_etf_momentum", names)
        self.assertNotIn("validate_cn_etf_daily_release", names)
        self.assertNotIn("generate_etf_momentum_report", names)

    def test_formal_index_update_stage_uses_cn_index_daily_updater(self):
        stages = self.module.build_stages(_args())
        names = [stage.name for stage in stages]
        update = next(stage for stage in stages if stage.name == "update_cn_index_daily")

        self.assertTrue(update.required)
        self.assertIn("quant_data_cli.py", update.argv[1])
        self.assertEqual(update.argv[update.argv.index("update") + 1], "cn_index_daily")
        self.assertIn("--through", update.argv)
        self.assertLess(names.index("update_cn_index_daily"), names.index("validate_cn_index_daily"))

    def test_verified_bundle_uses_contract_updater_before_native_validation(self):
        bundle = Path("/verified/bundle")
        stages = self.module.build_stages(_args(ashare_v2_bundle_dir=bundle))
        names = [stage.name for stage in stages]
        update = next(stage for stage in stages if stage.name == "update_ashare_v2_bundle")

        self.assertTrue(update.required)
        self.assertIn("quant_data_cli.py", update.argv[1])
        self.assertIn("ashare_daily_qfq", update.argv)
        self.assertEqual(update.argv[update.argv.index("--bundle-dir") + 1], str(bundle))
        self.assertLess(
            names.index("update_ashare_v2_bundle"),
            names.index("validate_ashare_daily_qfq"),
        )
        self.assertNotIn("publish_official", names)

    def test_source_downloads_can_be_individually_disabled(self):
        stages = self.module.build_stages(
            _args(ashare_source="baostock", skip_baostock=True)
        )
        names = [s.name for s in stages]

        self.assertNotIn("download_baostock", names)
        self.assertNotIn("download_eastmoney", names)
        self.assertNotIn("download_akshare", names)
        self.assertIn("validate_cn_index_daily", names)

    def test_system_proxy_flag_only_applies_to_public_sources(self):
        baostock_stages = self.module.build_stages(_args(ashare_source="baostock", use_system_proxy=True))
        eastmoney_stages = self.module.build_stages(_args(ashare_source="eastmoney", use_system_proxy=True))
        akshare_stages = self.module.build_stages(_args(ashare_source="akshare", use_system_proxy=True))
        baostock = next(s for s in baostock_stages if s.name == "download_baostock")
        eastmoney = next(s for s in eastmoney_stages if s.name == "download_eastmoney")
        akshare = next(s for s in akshare_stages if s.name == "download_akshare")

        self.assertNotIn("--use-system-proxy", baostock.argv)
        self.assertIn("--use-system-proxy", eastmoney.argv)
        self.assertIn("--use-system-proxy", akshare.argv)


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

    def test_latest_dataset_date_reads_governed_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "datasets": {
                            "cn_etf_daily": {
                                "latest_trade_date": "2026-07-24"
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest = self.module.latest_dataset_date("cn_etf_daily", root=root)

        self.assertEqual(latest, "2026-07-24")

    def test_latest_dataset_date_fails_closed_when_catalog_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self.module.latest_dataset_date(
                    "cn_etf_daily", root=Path(tmp)
                ),
                "",
            )


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

        self.assertFalse(self.module.selected_ashare_refresh_failed(results))

    def test_all_ashare_downloads_failed_still_triggers_gate_with_global_success(self):
        results = [
            self.module.StageResult(name, "failed", 0, 1, required=False)
            for name in self.module.ASHARE_DOWNLOAD_STAGE_NAMES
        ] + [
            self.module.StageResult("download_global_daily", "ok", 0, 0, required=False),
            self.module.StageResult("merge_to_qlib", "skipped", 0, 0, required=True),
        ]

        self.assertTrue(self.module.selected_ashare_refresh_failed(results))


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
            self.assertTrue(
                any(s["name"] == "validate_cn_index_daily" for s in summary["stages"])
            )
            self.assertFalse(
                any(s["name"] == "publish_official" for s in summary["stages"])
            )

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

    def test_selected_source_download_failure_fails_refresh_request(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "download_eastmoney" else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys,
                    "argv",
                    ["run_daily_update.py", "--ashare-source", "eastmoney", "--continue-on-error"],
                ),
            ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any(s["name"] == "download_eastmoney" and s["status"] == "failed" for s in summary["stages"]))
        self.assertTrue(
            any(
                s["name"] == "source_refresh_gate"
                and s["required"]
                and s["status"] == "failed"
                for s in summary["stages"]
            )
        )

    def test_selected_legacy_source_never_claims_canonical_publication(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            return module.StageResult(stage.name, "ok", 0.0, 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys,
                    "argv",
                    ["run_daily_update.py", "--ashare-source", "eastmoney", "--continue-on-error"],
                ),
            ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(summary["status"], "ok")
        self.assertFalse(summary["official_published"])
        self.assertFalse(summary["manifest_synced"])
        self.assertNotIn("publish_official", [item["name"] for item in summary["stages"]])

    def test_etf_report_keeps_previous_artifact_when_update_or_release_check_fails(self):
        module = self.module

        for failed_stage in (
            "download_etf_momentum",
            "validate_cn_etf_daily_release",
        ):
            with self.subTest(failed_stage=failed_stage), tempfile.TemporaryDirectory() as tmp:
                called: list[str] = []

                def fake_run_stage(stage, dry_run):
                    called.append(stage.name)
                    status = "failed" if stage.name == failed_stage else "ok"
                    return module.StageResult(
                        stage.name,
                        status,
                        0.0,
                        1 if status == "failed" else 0,
                        required=stage.required,
                    )

                reports = Path(tmp) / "reports"
                with (
                    patch.object(module, "REPORTS_ROOT", reports),
                    patch.object(module, "load_expected_symbols", return_value={"000001"}),
                    patch.object(
                        module,
                        "source_coverage_snapshot",
                        return_value=_snapshot("2026-07-08"),
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

                report_stage = next(
                    stage
                    for stage in summary["stages"]
                    if stage["name"] == "generate_etf_momentum_report"
                )
                self.assertEqual(rc, 0)
                self.assertEqual(summary["status"], "ok_with_warnings")
                self.assertEqual(report_stage["status"], "skipped")
                self.assertIn(failed_stage, report_stage["details"]["reason"])
                self.assertNotIn("generate_etf_momentum_report", called)

    def test_native_validation_failure_stops_before_downstream_checks(self):
        module = self.module
        called: list[str] = []

        def fake_run_stage(stage, dry_run):
            called.append(stage.name)
            failed = stage.name == "validate_cn_index_daily"
            return module.StageResult(stage.name, "failed" if failed else "ok", 0.0, 1 if failed else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(sys, "argv", ["run_daily_update.py"]),
            ):
                rc = module.main()

        self.assertEqual(rc, 1)
        self.assertIn("validate_cn_index_daily", called)
        self.assertNotIn("check_qlib_direct_data", called)

    def test_failed_audit_download_never_triggers_legacy_rebuild(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-03")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys,
                    "argv",
                    ["run_daily_update.py", "--ashare-source", "baostock"],
                ),
            ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        # baostock download failure is recorded but non-required (non-fatal)
        self.assertTrue(any(
            s["name"] == "download_baostock" and not s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))
        self.assertNotIn("merge_to_qlib", [s["name"] for s in summary["stages"]])
        self.assertTrue(
            any(s["name"] == "validate_cn_index_daily" and s["status"] == "ok" for s in summary["stages"])
        )

    def test_all_source_downloads_failing_without_rebuild_fails_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name.startswith("download_") else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-03")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(sys, "argv", ["run_daily_update.py", "--ashare-source", "baostock"]),
            ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any(
            s["name"] == "source_refresh_gate" and s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))

    def test_bundle_update_failure_is_required_and_fails_daily_update(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            status = "failed" if stage.name == "update_ashare_v2_bundle" else "ok"
            return module.StageResult(stage.name, status, 0.0, 1 if status == "failed" else 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    side_effect=[_snapshot("2026-07-03"), _snapshot("2026-07-08")],
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_daily_update.py",
                        "--ashare-v2-bundle-dir",
                        str(Path(tmp) / "bundle"),
                    ],
                ),
            ):
                rc = module.main()
            summary = json.loads((reports / "daily_update_summary_latest.json").read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(any(
            s["name"] == "update_ashare_v2_bundle" and s["required"] and s["status"] == "failed"
            for s in summary["stages"]
        ))

    def test_successful_bundle_update_is_the_only_canonical_publication(self):
        module = self.module

        def fake_run_stage(stage, dry_run):
            return module.StageResult(stage.name, "ok", 0.0, 0, required=stage.required)

        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            bundle = Path(tmp) / "bundle"
            with (
                patch.object(module, "REPORTS_ROOT", reports),
                patch.object(module, "load_expected_symbols", return_value={"000001"}),
                patch.object(
                    module,
                    "source_coverage_snapshot",
                    return_value=_snapshot("2026-07-08"),
                ),
                patch.object(module, "run_stage", side_effect=fake_run_stage),
                patch.object(
                    sys,
                    "argv",
                    ["run_daily_update.py", "--ashare-v2-bundle-dir", str(bundle)],
                ),
            ):
                rc = module.main()
            summary = json.loads(
                (reports / "daily_update_summary_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(rc, 0)
        self.assertTrue(summary["official_published"])
        self.assertTrue(summary["manifest_synced"])


if __name__ == "__main__":
    unittest.main()
