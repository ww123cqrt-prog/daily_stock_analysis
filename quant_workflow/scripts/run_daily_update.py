#!/usr/bin/env python3
"""Run the daily local A-share data update pipeline end to end.

Chains the existing stable quant_workflow scripts into one auditable run:

    1. download_*               incremental refresh of anonymous public sources
    2. merge_to_qlib.py         multi-source merge -> canonical Qlib provider
    3. add_index_benchmark.py   SH000300 benchmark features for Qlib backtest
    4. validate_latest_snapshot Sina latest snapshot validation (non-fatal)
    5. report_data_anomalies.py anomaly detection (produces cleaning inputs)
    6. apply_data_cleaning.py   cleaned parquet -> official cleaned provider
    7. report_data_quality.py   cross-source validation
    8. check_qlib_data.py       final provider calendar/instrument sanity check
    9. sync_data_manifest.py    refresh quant_data/DATA_MANIFEST.json
   10. run_robust_shadow_day.py active frozen-strategy paper observation

Every stage runs the current interpreter (sys.executable) against the embedded
quant_workflow scripts, so reports and providers resolve to the same canonical
locations the running DSA service reads.

Safety:
    * Baostock/EastMoney/AkShare/Sina use anonymous public interfaces; no
      credentials are needed or read here.
    * JQData is NOT downloaded here (its account data window ends before the
      Baostock series); existing JQData parquet is only reused during merge and
      cross-source validation.
    * The pipeline never deletes real data. Merge and cleaning overwrite their
      own derived outputs in place, exactly like the manual /workflow buttons.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
REPORTS_ROOT = reports_root()
QUANT_DATA = quant_data_root()
BAOSTOCK_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "baostock_qfq"
EASTMONEY_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "eastmoney_qfq"
AKSHARE_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "akshare_qfq"
SOURCE_DIRS = {
    "baostock": BAOSTOCK_DIR,
    "eastmoney": EASTMONEY_DIR,
    "akshare": AKSHARE_DIR,
}


@dataclass
class Stage:
    name: str
    argv: list[str]
    heavy: bool = False
    required: bool = True


@dataclass
class StageResult:
    name: str
    status: str  # ok | failed | planned | skipped
    seconds: float
    returncode: int
    required: bool = True


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def latest_source_date(directory: Path, sample: int = 40) -> str:
    """Return the max trading date across a spread sample of source parquet files.

    A spread sample (not just the first N) keeps both SZ (000xxx) and SH
    (6xxxxx) symbols represented so a market-wide refresh is detected reliably
    without reading all 5000+ files.
    """
    import pandas as pd

    if not directory.exists():
        return ""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return ""
    if sample > 0 and len(files) > sample:
        step = len(files) / sample
        files = [files[int(i * step)] for i in range(sample)]
    latest = None
    for path in files:
        try:
            dates = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue
        series = pd.to_datetime(dates["date"], errors="coerce").dropna()
        if series.empty:
            continue
        current = series.max()
        if latest is None or current > latest:
            latest = current
    return str(latest.date()) if latest is not None else ""


def latest_any_source_date(
    source_dirs: dict[str, Path], sample: int = 40
) -> tuple[str, dict[str, str]]:
    by_source = {
        name: latest_source_date(directory, sample=sample)
        for name, directory in source_dirs.items()
    }
    latest = max((value for value in by_source.values() if value), default="")
    return latest, by_source


def should_run_heavy(
    *, force: bool, download_ran: bool, before: str, after: str
) -> bool:
    """Decide whether to rebuild merge/clean/report stages.

    Rebuild when forced, when the download step was skipped (caller wants a
    rebuild of existing data), or when a new trading day actually arrived.
    """
    if force or not download_ran:
        return True
    if not before or not after:
        return True
    return after > before


def _python() -> str:
    return sys.executable


def _script(name: str) -> list[str]:
    return [_python(), str(WORKFLOW_ROOT / "scripts" / name)]


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def robust_shadow_tracking_is_active(report_dir: Path) -> bool:
    robust_dir = Path(report_dir) / "robust_strategy"
    confirmation = _read_json_object(robust_dir / "confirmation_latest.json")
    frozen = _read_json_object(robust_dir / "frozen_strategy_latest.json")
    state = _read_json_object(robust_dir / "shadow_state_latest.json")
    if state.get("status") in {"robust_strategy_qualified", "development_failed"}:
        return False
    return (
        confirmation.get("status") == "shadow_observation"
        and frozen.get("confirmation_status") == "shadow_observation"
        and frozen.get("shadow_tracking_started") is True
    )


def build_stages(args: argparse.Namespace) -> "list[Stage]":
    """Compose the ordered pipeline. Each stage is (name, argv, heavy)."""
    out = ["--out-dir", str(REPORTS_ROOT)]
    stages: list[Stage] = []

    if not args.skip_download:
        # All source downloads are non-fatal: merge reads on-disk parquet, so a
        # transient network failure (any single source) must not block the
        # rebuild while other sources' data is intact. This is the whole point
        # of multi-source redundancy.
        download_specs = [
            ("download_baostock", "download_baostock.py", False, False),
            ("download_eastmoney", "download_eastmoney.py", args.skip_eastmoney, False),
            ("download_akshare", "download_akshare.py", args.skip_akshare, False),
        ]
        for stage_name, script_name, skip, required in download_specs:
            if skip:
                continue
            dl = _script(script_name) + [
                "--incremental",
                "--overlap-days",
                str(args.overlap_days),
                "--sleep",
                str(args.sleep),
                "--timeout",
                str(args.timeout),
                "--retries",
                str(args.retries),
                "--max-consecutive-failures",
                str(args.max_consecutive_failures),
                "--end",
                str(args.end),
            ]
            if args.limit > 0:
                dl += ["--limit", str(args.limit)]
            if (
                stage_name in {"download_eastmoney", "download_akshare"}
                and args.use_system_proxy
            ):
                dl += ["--use-system-proxy"]
            stages.append(Stage(stage_name, dl, heavy=False, required=required))

    stages.append(Stage("merge_to_qlib", _script("merge_to_qlib.py"), heavy=True))
    stages.append(
        Stage("add_index_benchmark", _script("add_index_benchmark.py"), heavy=True)
    )
    if not args.skip_snapshot_validation:
        snapshot = _script("validate_latest_snapshot.py") + out
        if args.snapshot_limit > 0:
            snapshot += ["--limit", str(args.snapshot_limit)]
        if args.use_system_proxy:
            snapshot += ["--use-system-proxy"]
        stages.append(
            Stage("validate_latest_snapshot", snapshot, heavy=True, required=False)
        )
    stages.append(
        Stage(
            "report_data_anomalies",
            _script("report_data_anomalies.py") + out,
            heavy=True,
        )
    )
    stages.append(
        Stage(
            "apply_data_cleaning", _script("apply_data_cleaning.py") + out, heavy=True
        )
    )
    quality = _script("report_data_quality.py") + out
    if args.overlap_sample > 0:
        quality += ["--overlap-sample", str(args.overlap_sample)]
    if args.merged_sample > 0:
        quality += ["--merged-sample", str(args.merged_sample)]
    stages.append(Stage("report_data_quality", quality, heavy=True))
    stages.append(Stage("check_qlib_data", _script("check_qlib_data.py"), heavy=True))
    stages.append(
        Stage(
            "sync_data_manifest",
            _script("sync_data_manifest.py") + out,
            heavy=False,
            required=True,
        )
    )
    if not args.skip_robust_shadow and robust_shadow_tracking_is_active(REPORTS_ROOT):
        robust_dir = REPORTS_ROOT / "robust_strategy"
        stages.append(
            Stage(
                "run_robust_shadow_day",
                _script("run_robust_shadow_day.py")
                + [
                    "--report-dir",
                    str(robust_dir),
                    "--frozen",
                    str(robust_dir / "frozen_strategy_latest.json"),
                    "--confirmation-latest",
                    str(robust_dir / "confirmation_latest.json"),
                    "--provider-latest",
                    str(robust_dir / "provider_latest.json"),
                    "--as-of-date",
                    str(args.end),
                ],
                heavy=False,
                required=False,
            )
        )
    return stages


def source_refresh_failed_without_rebuild(results: list[StageResult]) -> bool:
    """Return true when the run failed to refresh any source and rebuilt nothing.

    Individual source failures are warnings because existing parquet can still
    support a rebuild. If every attempted source failed and every heavy stage
    was skipped, the daily run produced no useful update and should not be
    reported as successful.
    """
    download_results = [r for r in results if r.name.startswith("download_")]
    if not download_results or not all(r.status == "failed" for r in download_results):
        return False
    heavy_results = [
        r
        for r in results
        if r.name
        in {
            "merge_to_qlib",
            "add_index_benchmark",
            "validate_latest_snapshot",
            "report_data_anomalies",
            "apply_data_cleaning",
            "report_data_quality",
            "check_qlib_data",
        }
    ]
    return bool(heavy_results) and all(r.status == "skipped" for r in heavy_results)


def run_stage(stage: "Stage", dry_run: bool) -> "StageResult":
    _log(f"stage {stage.name}: {' '.join(stage.argv)}")
    if dry_run:
        return StageResult(stage.name, "planned", 0.0, 0, required=stage.required)
    t0 = time.time()
    proc = subprocess.run(stage.argv, cwd=str(WORKFLOW_ROOT))
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "failed"
    _log(f"stage {stage.name}: {status} rc={proc.returncode} in {elapsed / 60:.1f}m")
    return StageResult(
        stage.name, status, elapsed, proc.returncode, required=stage.required
    )


def write_summary(results: "list[StageResult]", meta: dict) -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **meta,
        "stages": [
            {
                "name": r.name,
                "status": r.status,
                "seconds": round(r.seconds, 2),
                "returncode": r.returncode,
                "required": r.required,
            }
            for r in results
        ],
    }
    status = "ok"
    if any(r.status == "failed" and r.required for r in results):
        status = "failed"
    elif any(r.status == "failed" for r in results):
        status = "ok_with_warnings"
    elif all(r.status == "skipped" for r in results):
        status = "skipped"
    payload["status"] = status

    dated = REPORTS_ROOT / f"daily_update_summary_{ts}.json"
    dated.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest = REPORTS_ROOT / "daily_update_summary_latest.json"
    latest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return latest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the daily A-share data update pipeline."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip Baostock download; rebuild merge/clean/reports from existing parquet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run heavy rebuild stages even when no new trading day arrived.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=45,
        help="Calendar-day overlap for incremental Baostock download.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3, help="Per-symbol sleep for Baostock."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-symbol Baostock query timeout (seconds).",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Baostock per-symbol retries."
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=50,
        help="Abort download after this many consecutive failures.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on symbols downloaded (0 = all).",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y%m%d"),
        help="Download end date for source refreshes, YYYYMMDD or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--skip-eastmoney",
        action="store_true",
        help="Skip EastMoney public-source refresh.",
    )
    parser.add_argument(
        "--skip-akshare",
        action="store_true",
        help="Skip AkShare public-source refresh.",
    )
    parser.add_argument(
        "--skip-snapshot-validation",
        action="store_true",
        help="Skip Sina latest snapshot validation.",
    )
    parser.add_argument(
        "--skip-robust-shadow",
        action="store_true",
        help="Skip the active frozen-strategy paper observation stage.",
    )
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Let EastMoney/AkShare/Sina stages inherit macOS/system proxy settings.",
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=50,
        help="Number of merged symbols sampled for Sina snapshot validation.",
    )
    parser.add_argument(
        "--overlap-sample",
        type=int,
        default=0,
        help="Baostock/JQData overlap sample size for the quality report (0 = all).",
    )
    parser.add_argument(
        "--merged-sample",
        type=int,
        default=0,
        help="Merged-file sample size for the quality report (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pipeline without executing any stage.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later stages even if one fails (default stops).",
    )
    args = parser.parse_args()

    _log(f"daily update start; reports={REPORTS_ROOT} quant_data={QUANT_DATA}")

    before, before_by_source = latest_any_source_date(SOURCE_DIRS)
    _log(f"latest source date before: {before or 'n/a'} {before_by_source}")

    stages = build_stages(args)
    results: list[StageResult] = []
    download_ran = not args.skip_download
    heavy_enabled = None  # decided after download

    for stage in stages:
        if stage.heavy:
            if heavy_enabled is None:
                if args.dry_run:
                    # A dry run previews the whole pipeline, so show every stage
                    # command instead of applying today's no-new-data gate.
                    after = before
                    after_by_source = before_by_source
                    heavy_enabled = True
                else:
                    after, after_by_source = (
                        latest_any_source_date(SOURCE_DIRS)
                        if download_ran
                        else (before, before_by_source)
                    )
                    heavy_enabled = should_run_heavy(
                        force=args.force,
                        download_ran=download_ran,
                        before=before,
                        after=after,
                    )
                _log(
                    f"latest source date after: {after or 'n/a'} {after_by_source}; "
                    f"heavy stages {'enabled' if heavy_enabled else 'skipped (no new trading day)'}"
                )
            if not heavy_enabled:
                _log(f"stage {stage.name}: skipped")
                results.append(
                    StageResult(stage.name, "skipped", 0.0, 0, required=stage.required)
                )
                continue

        result = run_stage(stage, args.dry_run)
        results.append(result)
        if (
            result.status == "failed"
            and result.required
            and not args.continue_on_error
            and not args.dry_run
        ):
            _log(
                f"stage {stage.name} failed; stopping pipeline (use --continue-on-error to override)"
            )
            break
        if result.status == "failed" and not result.required:
            _log(f"stage {stage.name} failed but is non-fatal; continuing")

    if source_refresh_failed_without_rebuild(results):
        _log(
            "all source downloads failed and rebuild stages were skipped; marking daily update failed"
        )
        results.append(
            StageResult(
                "source_refresh_gate",
                "failed",
                0.0,
                1,
                required=True,
            )
        )

    after_final, after_final_by_source = latest_any_source_date(SOURCE_DIRS)
    summary_path = write_summary(
        results,
        {
            "source_date_before": before,
            "source_date_after": after_final,
            "source_dates_before": before_by_source,
            "source_dates_after": after_final_by_source,
            "dry_run": bool(args.dry_run),
            "skip_download": bool(args.skip_download),
            "forced": bool(args.force),
        },
    )
    _log(f"daily update summary: {summary_path}")

    failed = [r for r in results if r.status == "failed" and r.required]
    if failed:
        _log(f"daily update FAILED at: {', '.join(r.name for r in failed)}")
        return 1
    _log("daily update completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
