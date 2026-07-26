#!/usr/bin/env python3
"""Run the scheduled A-share source refresh and candidate publication pipeline.

Source caches are updated independently. Derived merged and cleaned artifacts
are built under a run-scoped staging directory, validated there, and promoted
only by the final rollback-capable publish stage.
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
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import DailyUpdateLock, LockHeldError, write_json_atomic  # noqa: E402
from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
REPORTS_ROOT = reports_root()
QUANT_DATA = quant_data_root()
PARQUET_ROOT = QUANT_DATA / "parquet" / "ashare_1d"
QLIB_ROOT = QUANT_DATA / "qlib"
BAOSTOCK_DIR = PARQUET_ROOT / "baostock_qfq"
EASTMONEY_DIR = PARQUET_ROOT / "eastmoney_qfq"
AKSHARE_DIR = PARQUET_ROOT / "akshare_qfq"
JQDATA_DIR = PARQUET_ROOT / "jqdata"
STOCK_LIST = QUANT_DATA / "stock_lists.json"
SOURCE_DIRS = {
    "baostock": BAOSTOCK_DIR,
    "eastmoney": EASTMONEY_DIR,
    "akshare": AKSHARE_DIR,
}
ASHARE_DOWNLOAD_STAGE_NAMES = frozenset(
    {"download_baostock", "download_eastmoney", "download_akshare"}
)


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
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class CandidatePaths:
    run_root: Path
    merged_parquet: Path
    merged_qlib: Path
    cleaned_parquet: Path
    cleaned_qlib: Path
    backup_root: Path


def _log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _read_latest_date(path: Path) -> str:
    import pandas as pd

    try:
        dates = pd.read_parquet(path, columns=["date"])
    except Exception:
        return ""
    if "date" not in dates:
        return ""
    series = pd.to_datetime(dates["date"], errors="coerce").dropna()
    return str(series.max().date()) if not series.empty else ""


def latest_source_date(directory: Path, sample: int = 40) -> str:
    if not directory.exists():
        return ""
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return ""
    if sample > 0 and len(files) > sample:
        step = len(files) / sample
        files = [files[int(index * step)] for index in range(sample)]
    return max((_read_latest_date(path) for path in files), default="")


def latest_any_source_date(source_dirs: dict[str, Path], sample: int = 40) -> tuple[str, dict[str, str]]:
    by_source = {name: latest_source_date(directory, sample=sample) for name, directory in source_dirs.items()}
    latest = max((value for value in by_source.values() if value), default="")
    return latest, by_source


def load_expected_symbols(path: Path = STOCK_LIST) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("all", []) if isinstance(payload, dict) else []
    symbols: set[str] = set()
    for value in values:
        raw = str(value).strip().upper()
        if not raw:
            continue
        raw = raw.split(".", 1)[0]
        if raw.startswith(("SH", "SZ")):
            raw = raw[2:]
        if len(raw) == 6 and raw.isdigit():
            symbols.add(raw)
    return symbols


def source_coverage_snapshot(
    source_dirs: dict[str, Path],
    *,
    expected_symbols: set[str],
    minimum_ratio: float,
) -> dict[str, Any]:
    latest_by_symbol: dict[str, dict[str, str]] = {}
    source_dates: dict[str, str] = {}
    for source_name, directory in source_dirs.items():
        symbol_dates: dict[str, str] = {}
        if directory.exists():
            for path in sorted(directory.glob("*.parquet")):
                latest = _read_latest_date(path)
                if latest:
                    symbol_dates[path.stem] = latest
        latest_by_symbol[source_name] = symbol_dates
        source_dates[source_name] = max(symbol_dates.values(), default="")

    candidate_date = max((value for value in source_dates.values() if value), default="")
    covered: set[str] = set()
    per_source: dict[str, dict[str, Any]] = {}
    for source_name, symbol_dates in latest_by_symbol.items():
        current = {symbol for symbol, day in symbol_dates.items() if day == candidate_date}
        if expected_symbols:
            current &= expected_symbols
        covered.update(current)
        per_source[source_name] = {
            "files": len(symbol_dates),
            "latest_date": source_dates[source_name],
            "candidate_date_symbols": len(current),
        }

    expected_count = len(expected_symbols)
    ratio = len(covered) / expected_count if expected_count else 0.0
    return {
        "candidate_date": candidate_date,
        "expected_symbols": expected_count,
        "covered_symbols": len(covered),
        "coverage_ratio": round(ratio, 6),
        "minimum_ratio": float(minimum_ratio),
        "eligible": bool(candidate_date and expected_count and ratio >= minimum_ratio),
        "source_dates": source_dates,
        "sources": per_source,
    }


def should_run_heavy(*, force: bool, download_ran: bool, before: str, after: str) -> bool:
    if force or not download_ran:
        return True
    if not before or not after:
        return True
    return after > before


def _python() -> str:
    return sys.executable


def _script(name: str) -> list[str]:
    return [_python(), str(WORKFLOW_ROOT / "scripts" / name)]


def candidate_paths(run_id: str) -> CandidatePaths:
    run_root = QUANT_DATA / "runtime" / "daily_update" / "runs" / run_id
    staging = run_root / "staging"
    return CandidatePaths(
        run_root=run_root,
        merged_parquet=staging / "merged_parquet",
        merged_qlib=staging / "merged_qlib",
        cleaned_parquet=staging / "cleaned_parquet",
        cleaned_qlib=staging / "cleaned_qlib",
        backup_root=run_root / "previous_official",
    )


def build_stages(args: argparse.Namespace, run_id: str = "planned") -> list[Stage]:
    paths = candidate_paths(run_id)
    out = ["--out-dir", str(REPORTS_ROOT)]
    status_dir = REPORTS_ROOT / "runtime" / "status"
    stages: list[Stage] = []

    if not args.skip_download:
        selected_source = getattr(args, "ashare_source", "none")
        download_specs = {
            "baostock": ("download_baostock", "download_baostock.py", args.skip_baostock, False),
            "eastmoney": ("download_eastmoney", "download_eastmoney.py", args.skip_eastmoney, False),
            "akshare": ("download_akshare", "download_akshare.py", args.skip_akshare, False),
        }
        selected_spec = download_specs.get(selected_source)
        selected_specs = [selected_spec] if selected_spec is not None else []
        for stage_name, script_name, skip, required in selected_specs:
            if skip:
                continue
            command = _script(script_name) + [
                "--incremental",
                "--overlap-days", str(args.overlap_days),
                "--sleep", str(args.sleep),
                "--timeout", str(args.timeout),
                "--retries", str(args.retries),
                "--max-consecutive-failures", str(args.max_consecutive_failures),
                "--max-requests", str(args.max_requests),
                "--end", str(args.end),
            ]
            if args.limit > 0:
                command += ["--limit", str(args.limit)]
            if stage_name in {"download_eastmoney", "download_akshare"} and args.use_system_proxy:
                command += ["--use-system-proxy"]
            stages.append(Stage(stage_name, command, required=required))

        if not args.skip_global_daily:
            global_daily = _script("download_global_daily.py") + [
                "--incremental",
                "--overlap-days", str(args.overlap_days),
                "--sleep", str(args.sleep),
                "--timeout", str(args.timeout),
                "--retries", str(args.retries),
                "--end", str(args.end),
            ]
            if args.limit > 0:
                global_daily += ["--limit", str(args.limit)]
            if args.use_system_proxy:
                global_daily += ["--use-system-proxy"]
            stages.append(Stage("download_global_daily", global_daily, required=False))

        if not args.skip_ashare_news and args.news_limit > 0:
            news = _script("download_ashare_news.py") + [
                "--limit", str(args.news_limit),
                "--sleep", str(args.sleep),
                "--timeout", str(args.timeout),
                "--retries", str(args.retries),
            ]
            if args.use_system_proxy:
                news += ["--use-system-proxy"]
            stages.append(Stage("download_ashare_news", news, required=False))

    stages.append(
        Stage(
            "merge_to_qlib",
            _script("merge_to_qlib.py") + [
                "--merged-dir", str(paths.merged_parquet),
                "--qlib-dir", str(paths.merged_qlib),
            ],
            heavy=True,
        )
    )
    benchmark = _script("add_index_benchmark.py") + [
        "--provider-dir", str(paths.merged_qlib),
        "--status-dir", str(status_dir),
    ]
    if args.benchmark_cache_dir is not None:
        benchmark += ["--cache-provider", str(args.benchmark_cache_dir)]
    if args.benchmark_offline:
        benchmark += ["--offline"]
    if args.use_system_proxy:
        benchmark += ["--use-system-proxy"]
    stages.append(Stage("add_index_benchmark", benchmark, heavy=True))

    if not args.skip_snapshot_validation:
        snapshot = _script("validate_latest_snapshot.py") + [
            "--merged-dir", str(paths.merged_parquet),
            *out,
        ]
        if args.snapshot_limit > 0:
            snapshot += ["--limit", str(args.snapshot_limit)]
        if args.use_system_proxy:
            snapshot += ["--use-system-proxy"]
        stages.append(Stage("validate_latest_snapshot", snapshot, heavy=True, required=False))

    stages.append(
        Stage(
            "report_data_anomalies",
            _script("report_data_anomalies.py") + ["--merged-dir", str(paths.merged_parquet), *out],
            heavy=True,
        )
    )
    stages.append(
        Stage(
            "apply_data_cleaning",
            _script("apply_data_cleaning.py") + [
                "--merged-dir", str(paths.merged_parquet),
                "--cleaned-merged-dir", str(paths.cleaned_parquet),
                "--qlib-dir", str(paths.cleaned_qlib),
                "--benchmark-provider-dir", str(paths.merged_qlib),
                *out,
            ],
            heavy=True,
        )
    )
    quality = _script("report_data_quality.py") + [
        "--merged-dir", str(paths.cleaned_parquet),
        "--qlib-dir", str(paths.cleaned_qlib),
        "--min-calendar-days", str(args.quality_min_calendar_days),
        "--min-instruments", str(args.quality_min_instruments),
        "--fail-on-not-trainable",
        *out,
    ]
    if args.overlap_sample > 0:
        quality += ["--overlap-sample", str(args.overlap_sample)]
    if args.merged_sample > 0:
        quality += ["--merged-sample", str(args.merged_sample)]
    stages.append(Stage("report_data_quality", quality, heavy=True))
    stages.append(
        Stage(
            "check_qlib_data",
            _script("check_qlib_data.py") + ["--provider-dir", str(paths.cleaned_qlib)],
            heavy=True,
        )
    )

    publish = _script("publish_daily_candidate.py") + [
        "--candidate-merged-parquet", str(paths.merged_parquet),
        "--target-merged-parquet", str(PARQUET_ROOT / "merged_baostock_jqdata"),
        "--candidate-merged-qlib", str(paths.merged_qlib),
        "--target-merged-qlib", str(QLIB_ROOT / "cn_data_baostock_jqdata"),
        "--candidate-cleaned-parquet", str(paths.cleaned_parquet),
        "--target-cleaned-parquet", str(PARQUET_ROOT / "merged_baostock_jqdata_cleaned"),
        "--candidate-cleaned-qlib", str(paths.cleaned_qlib),
        "--target-cleaned-qlib", str(QLIB_ROOT / "cn_data_baostock_jqdata_cleaned"),
        "--backup-root", str(paths.backup_root),
        "--quant-data-root", str(QUANT_DATA),
        "--reports-root", str(REPORTS_ROOT),
        "--status-dir", str(status_dir),
    ]
    stages.append(Stage("publish_official", publish, heavy=True))
    return stages


def source_refresh_failed_without_rebuild(results: list[StageResult]) -> bool:
    download_results = [result for result in results if result.name in ASHARE_DOWNLOAD_STAGE_NAMES]
    if not download_results or not all(result.status == "failed" for result in download_results):
        return False
    heavy_names = {
        "merge_to_qlib",
        "add_index_benchmark",
        "validate_latest_snapshot",
        "report_data_anomalies",
        "apply_data_cleaning",
        "report_data_quality",
        "check_qlib_data",
        "publish_official",
    }
    heavy_results = [result for result in results if result.name in heavy_names]
    return bool(heavy_results) and all(result.status == "skipped" for result in heavy_results)


def _download_details(stage_name: str) -> dict[str, Any] | None:
    if not stage_name.startswith("download_"):
        return None
    source_name = stage_name.removeprefix("download_")
    candidates = [
        REPORTS_ROOT / "runtime" / "status" / f"{source_name}_download_summary_latest.json",
        REPORTS_ROOT / "download_state" / f"{source_name}_download_summary_latest.json",
    ]
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def run_stage(stage: Stage, dry_run: bool) -> StageResult:
    _log(f"stage {stage.name}: {' '.join(stage.argv)}")
    if dry_run:
        return StageResult(stage.name, "planned", 0.0, 0, required=stage.required)
    started = time.time()
    proc = subprocess.run(stage.argv, cwd=str(WORKFLOW_ROOT))
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 else "failed"
    _log(f"stage {stage.name}: {status} rc={proc.returncode} in {elapsed / 60:.1f}m")
    return StageResult(
        stage.name,
        status,
        elapsed,
        proc.returncode,
        required=stage.required,
        details=_download_details(stage.name),
    )


def stage_payload(result: StageResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": result.name,
        "status": result.status,
        "seconds": round(result.seconds, 2),
        "returncode": result.returncode,
        "required": result.required,
    }
    if result.details is not None:
        payload["details"] = result.details
    return payload


def summary_status(results: list[StageResult]) -> str:
    if any(result.status == "failed" and result.required for result in results):
        return "failed"
    if any(result.status == "failed" for result in results):
        return "ok_with_warnings"
    if results and all(result.status == "planned" for result in results):
        return "planned"
    if results and all(result.status == "skipped" for result in results):
        return "skipped"
    return "ok"


def write_summary(results: list[StageResult], meta: dict[str, Any]) -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    override = meta.pop("status_override", None)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **meta,
        "stages": [stage_payload(result) for result in results],
        "status": override or summary_status(results),
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    write_json_atomic(REPORTS_ROOT / f"daily_update_summary_{timestamp}.json", payload)
    latest = REPORTS_ROOT / "daily_update_summary_latest.json"
    write_json_atomic(latest, payload)
    return latest


def write_runtime_status(run_id: str, status: str, **details: Any) -> Path:
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "status": status,
        **details,
    }
    root = REPORTS_ROOT / "runtime" / "status"
    write_json_atomic(root / f"daily_update_status_{run_id}.json", payload)
    latest = root / "daily_update_status_latest.json"
    write_json_atomic(latest, payload)
    return latest


def robust_training_active() -> bool:
    """Return true when a known robust-strategy Python worker is active.

    Process command lines are inspected locally but never copied into logs or
    status reports.
    """
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "command="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return True
    if result.returncode != 0:
        return True
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "python" not in lowered:
            continue
        if "/daily_stock_analysis-robust" in lowered:
            return True
        if "quant_workflow/scripts/" in lowered and "robust" in lowered:
            return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily A-share data update pipeline.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--ashare-source",
        choices=["none", "baostock", "eastmoney", "akshare"],
        default="none",
        help="Run at most one A-share history source; none keeps network history quarantined.",
    )
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--skip-baostock", action="store_true")
    parser.add_argument("--skip-eastmoney", action="store_true")
    parser.add_argument("--skip-akshare", action="store_true")
    parser.add_argument("--skip-global-daily", action="store_true")
    parser.add_argument("--skip-ashare-news", action="store_true")
    parser.add_argument("--skip-snapshot-validation", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument("--snapshot-limit", type=int, default=50)
    parser.add_argument("--news-limit", type=int, default=0)
    parser.add_argument("--overlap-sample", type=int, default=0)
    parser.add_argument("--merged-sample", type=int, default=0)
    parser.add_argument("--min-source-coverage", type=float, default=0.8)
    parser.add_argument("--benchmark-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--benchmark-offline",
        dest="benchmark_offline",
        action="store_true",
        default=True,
        help="Use validated local benchmark caches only (default).",
    )
    parser.add_argument(
        "--allow-benchmark-network",
        dest="benchmark_offline",
        action="store_false",
        help="Explicitly allow benchmark fallback requests; never used by the scheduled or web profile.",
    )
    parser.add_argument("--quality-min-calendar-days", type=int, default=252)
    parser.add_argument("--quality-min-instruments", type=int, default=100)
    parser.add_argument("--skip-if-robust-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.overlap_days < 0 or args.overlap_days > 7:
        parser.error("--overlap-days must be between 0 and 7 for scheduled source safety")
    if args.sleep < 2.0:
        parser.error("--sleep must be at least 2 seconds for scheduled source safety")
    if args.retries != 1:
        parser.error("--retries must be 1 for scheduled source safety")
    if args.max_consecutive_failures < 1 or args.max_consecutive_failures > 3:
        parser.error("--max-consecutive-failures must be between 1 and 3")
    if args.max_requests < 1 or args.max_requests > 6000:
        parser.error("--max-requests must be between 1 and 6000")
    return args


def execute_pipeline(args: argparse.Namespace, run_id: str) -> int:
    _log(f"daily update start; reports={REPORTS_ROOT} quant_data={QUANT_DATA}")
    expected_symbols = load_expected_symbols()
    before = source_coverage_snapshot(
        SOURCE_DIRS,
        expected_symbols=expected_symbols,
        minimum_ratio=args.min_source_coverage,
    )
    _log(
        "source coverage before: "
        f"date={before['candidate_date'] or 'n/a'} ratio={before['coverage_ratio']:.3f}"
    )
    write_runtime_status(run_id, "running", current_stage="starting", source_coverage_before=before)

    stages = build_stages(args, run_id=run_id)
    results: list[StageResult] = []
    download_ran = not args.skip_download
    heavy_enabled: bool | None = None
    coverage_rejected = False
    after = before

    for stage in stages:
        if stage.heavy:
            if heavy_enabled is None:
                if args.dry_run:
                    heavy_enabled = True
                elif args.force or not download_ran:
                    after = source_coverage_snapshot(
                        SOURCE_DIRS,
                        expected_symbols=expected_symbols,
                        minimum_ratio=args.min_source_coverage,
                    )
                    heavy_enabled = True
                else:
                    after = source_coverage_snapshot(
                        SOURCE_DIRS,
                        expected_symbols=expected_symbols,
                        minimum_ratio=args.min_source_coverage,
                    )
                    date_advanced = should_run_heavy(
                        force=False,
                        download_ran=True,
                        before=before["candidate_date"],
                        after=after["candidate_date"],
                    )
                    heavy_enabled = bool(date_advanced and after["eligible"])
                    coverage_rejected = bool(date_advanced and not after["eligible"])
                _log(
                    "source coverage after: "
                    f"date={after['candidate_date'] or 'n/a'} ratio={after['coverage_ratio']:.3f}; "
                    f"heavy stages {'enabled' if heavy_enabled else 'skipped'}"
                )
            if not heavy_enabled:
                result = StageResult(stage.name, "skipped", 0.0, 0, required=stage.required)
                results.append(result)
                write_runtime_status(
                    run_id,
                    "running",
                    current_stage=stage.name,
                    stage_status="skipped",
                    stages=[stage_payload(item) for item in results],
                )
                continue

        if stage.name == "publish_official" and any(
            item.required and item.status == "failed" for item in results
        ):
            result = StageResult(stage.name, "skipped", 0.0, 0, required=True)
        else:
            write_runtime_status(
                run_id,
                "running",
                current_stage=stage.name,
                stages=[stage_payload(item) for item in results],
            )
            result = run_stage(stage, args.dry_run)
        results.append(result)
        write_runtime_status(
            run_id,
            "running",
            current_stage=stage.name,
            stage_status=result.status,
            stages=[stage_payload(item) for item in results],
        )
        if result.status == "failed" and result.required and not args.continue_on_error and not args.dry_run:
            _log(f"stage {stage.name} failed; stopping pipeline")
            break
        if result.status == "failed" and not result.required:
            _log(f"stage {stage.name} failed but is non-fatal; continuing")

    if coverage_rejected:
        results.append(
            StageResult(
                "source_coverage_gate",
                "failed",
                0.0,
                1,
                required=True,
                details=after,
            )
        )
    if source_refresh_failed_without_rebuild(results):
        results.append(StageResult("source_refresh_gate", "failed", 0.0, 1, required=True))

    if not args.dry_run and after is before:
        after = source_coverage_snapshot(
            SOURCE_DIRS,
            expected_symbols=expected_symbols,
            minimum_ratio=args.min_source_coverage,
        )
    published = any(item.name == "publish_official" and item.status == "ok" for item in results)
    summary_path = write_summary(
        results,
        {
            "run_id": run_id,
            "source_date_before": before["candidate_date"],
            "source_date_after": after["candidate_date"],
            "source_dates_before": before["source_dates"],
            "source_dates_after": after["source_dates"],
            "source_coverage_before": before,
            "source_coverage_after": after,
            "dry_run": bool(args.dry_run),
            "skip_download": bool(args.skip_download),
            "forced": bool(args.force),
            "official_published": published,
            "manifest_synced": published,
            "candidate_root": str(candidate_paths(run_id).run_root),
        },
    )
    final_status = summary_status(results)
    write_runtime_status(
        run_id,
        final_status,
        summary_path=str(summary_path),
        official_published=published,
        manifest_synced=published,
        stages=[stage_payload(item) for item in results],
        source_coverage=after,
    )
    _log(f"daily update summary: {summary_path}")

    failed = [item for item in results if item.status == "failed" and item.required]
    if failed:
        _log(f"daily update FAILED at: {', '.join(item.name for item in failed)}")
        return 1
    _log("daily update completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    lock = DailyUpdateLock(REPORTS_ROOT / "runtime" / "status" / "daily_update.lock.json")
    try:
        lock.acquire(run_id)
    except LockHeldError as exc:
        owner_run_id = str(exc.owner.get("run_id", "unknown"))
        write_runtime_status(
            run_id,
            "controlled_skip",
            reason="already_running",
            active_run_id=owner_run_id,
            official_published=False,
            manifest_synced=False,
        )
        write_summary(
            [],
            {
                "run_id": run_id,
                "status_override": "controlled_skip",
                "reason": "already_running",
                "active_run_id": owner_run_id,
                "dry_run": bool(args.dry_run),
                "official_published": False,
                "manifest_synced": False,
            },
        )
        _log(f"daily update controlled skip: active run {owner_run_id}")
        return 0

    try:
        if args.skip_if_robust_training and robust_training_active():
            write_runtime_status(
                run_id,
                "controlled_skip",
                reason="robust_training_active",
                official_published=False,
                manifest_synced=False,
            )
            write_summary(
                [],
                {
                    "run_id": run_id,
                    "status_override": "controlled_skip",
                    "reason": "robust_training_active",
                    "dry_run": bool(args.dry_run),
                    "official_published": False,
                    "manifest_synced": False,
                },
            )
            _log("daily update controlled skip: robust training is active")
            return 0
        return execute_pipeline(args, run_id)
    except Exception as exc:
        failure = StageResult(
            "orchestrator",
            "failed",
            0.0,
            1,
            required=True,
            details={"error_type": type(exc).__name__},
        )
        summary_path = write_summary(
            [failure],
            {
                "run_id": run_id,
                "dry_run": bool(args.dry_run),
                "official_published": False,
                "manifest_synced": False,
            },
        )
        write_runtime_status(
            run_id,
            "failed",
            current_stage="orchestrator",
            error_type=type(exc).__name__,
            summary_path=str(summary_path),
        )
        _log(f"daily update FAILED in orchestrator: {type(exc).__name__}")
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
