#!/usr/bin/env python3
"""Synchronize quant_data/DATA_MANIFEST.json with actual local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root, reports_root as default_reports_root  # noqa: E402


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def read_first_last(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    first = ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if not first:
                first = value
            last = value
    return first, last


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def human_size(path: Path) -> str:
    size = directory_size_bytes(path)
    if not path.exists():
        return "missing"
    if size < 1024 * 1024:
        return f"{round(size / 1024, 1)}K"
    if size < 1024 * 1024 * 1024:
        return f"{round(size / (1024 * 1024))}M"
    return f"{round(size / (1024 * 1024 * 1024), 2)}G"


def parquet_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "size": human_size(path),
        "file_count": len(list(path.glob("*.parquet"))) if path.exists() else 0,
    }


def qlib_entry(path: Path, role: str) -> dict[str, Any]:
    first, last = read_first_last(path / "calendars" / "day.txt")
    return {
        "path": str(path),
        "role": role,
        "size": human_size(path),
        "calendar_first": first,
        "calendar_last": last,
        "instruments_all": count_lines(path / "instruments" / "all.txt"),
    }


def build_manifest(*, quant_data_root: Path, reports_root: Path) -> dict[str, Any]:
    quant_data = Path(quant_data_root).expanduser()
    reports = Path(reports_root).expanduser()
    parquet_root = quant_data / "parquet" / "ashare_1d"
    qlib_root = quant_data / "qlib"

    cleaned_parquet = parquet_root / "merged_baostock_jqdata_cleaned"
    cleaned_provider = qlib_root / "cn_data_baostock_jqdata_cleaned"
    canonical_provider = qlib_root / "cn_data_baostock_jqdata"

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_report": str(reports / "data_hygiene_inventory_latest.json"),
        "cleanup_plan": str(reports / "data_hygiene_cleanup_plan_latest.json"),
        "policy": {
            "owner": "local DSA quant workbench",
            "dry_run_first": True,
            "no_delete_without_explicit_user_confirmation": True,
            "no_move_without_explicit_user_confirmation": True,
            "do_not_store_secrets": True,
            "data_root": str(quant_data),
        },
        "official_entries": {
            "cleaned_parquet": {
                **parquet_entry(
                    cleaned_parquet,
                    "Official cleaned A-share daily parquet dataset for DSA and Qlib conversion.",
                ),
                "status": "do_not_move",
            },
            "cleaned_qlib_provider": {
                **qlib_entry(
                    cleaned_provider,
                    "Official Qlib provider for current training and backtesting.",
                ),
                "status": "do_not_move",
            },
        },
        "documented_inputs": {
            "baostock_qfq": parquet_entry(
                parquet_root / "baostock_qfq",
                "Baostock source parquet used to reproduce merged and cleaned data.",
            ),
            "jqdata": parquet_entry(
                parquet_root / "jqdata",
                "JQData source parquet used to supplement and cross-check A-share data.",
            ),
            "eastmoney_qfq": parquet_entry(
                parquet_root / "eastmoney_qfq",
                "EastMoney public-source parquet used as redundant daily data input.",
            ),
            "akshare_qfq": parquet_entry(
                parquet_root / "akshare_qfq",
                "AkShare public-source parquet used as optional redundant daily data input.",
            ),
            "merged_baostock_jqdata": parquet_entry(
                parquet_root / "merged_baostock_jqdata",
                "Merged pre-cleaning parquet dataset retained for comparison and reproducibility.",
            ),
            "canonical_qlib_provider": qlib_entry(
                canonical_provider,
                "Merged pre-cleaning Qlib provider retained as a comparison baseline.",
            ),
        },
        "archive_candidates": {
            "require_confirmation": True,
            "note": "These are documented by the dry-run cleanup plan only. They have not been moved or deleted.",
            "examples": [
                str(parquet_root / "qfq"),
                str(parquet_root / "qfq_full"),
                str(qlib_root / "cn_data"),
                str(qlib_root / "cn_data_jqdata"),
                str(qlib_root / "cn_data_jqdata_full"),
            ],
        },
        "root_file_policy": {
            "current_state": "Root-level logs, pid files, runtime JSON files, and temporary scripts may still exist for compatibility.",
            "desired_state": "Runtime state should live under quant_workflow/reports/runtime_state, logs under quant_workflow/logs/quant_data, and one-off scripts under quant_workflow/scripts/archive after explicit confirmation.",
            "no_action_taken": True,
        },
        "related_reports": {
            "quality_report": str(reports / "data_quality_report_latest.md"),
            "cleaning_strategy": str(reports / "data_cleaning_strategy_latest.md"),
            "latest_training_result": str(reports / "latest_training_result.json"),
            "daily_update_summary": str(reports / "daily_update_summary_latest.json"),
            "research_status": str(reports / "research_status_latest.json"),
        },
    }
    return manifest


def sync_data_manifest(*, quant_data_root: Path | None = None, reports_root: Path | None = None) -> Path:
    quant_data = Path(quant_data_root or default_quant_data_root()).expanduser()
    reports = Path(reports_root or default_reports_root()).expanduser()
    manifest = build_manifest(quant_data_root=quant_data, reports_root=reports)
    quant_data.mkdir(parents=True, exist_ok=True)
    path = quant_data / "DATA_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize quant_data/DATA_MANIFEST.json with actual artifacts.")
    parser.add_argument("--quant-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Reports root. Defaults to quant_workflow/reports.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = sync_data_manifest(quant_data_root=args.quant_data_root, reports_root=args.out_dir)
    print(f"Synced data manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
