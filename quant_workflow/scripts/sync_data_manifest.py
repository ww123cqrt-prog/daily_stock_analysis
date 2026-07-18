#!/usr/bin/env python3
"""Synchronize quant_data/DATA_MANIFEST.json with actual local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root, reports_root as default_reports_root  # noqa: E402
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from _robust_reference_integrity import (  # noqa: E402
    REFERENCE_SNAPSHOT_FILES,
    ReferenceIntegrityError,
    reference_read_lock,
)


REFERENCE_PARQUET_FILES = tuple(
    name for name in REFERENCE_SNAPSHOT_FILES if name.endswith(".parquet")
)
DATA_READ_ERRORS = (
    KeyError,
    TypeError,
    ValueError,
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    pa.ArrowException,
)


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


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _pit_manifest_entry_locked(pit_dir: Path, snapshot) -> dict[str, Any]:
    snapshot.ensure_transaction_clean()
    raw_source = json.loads(
        (pit_dir / "MEMBERSHIP_SOURCE.json").read_text(encoding="utf-8")
    )
    date_range = raw_source["date_range"]
    if not isinstance(date_range, dict):
        raise TypeError("membership date_range must be a mapping")
    source = snapshot.validate_source(
        start_date=str(date_range["start_date"]),
        end_date=str(date_range["end_date"]),
    )
    daily_files = snapshot.daily_files()
    membership_hash = snapshot.fingerprint()
    member_day_count = sum(parquet_row_count(path) for path in daily_files)
    symbols: set[str] = set()
    for path in daily_files:
        symbols.update(
            str(value)
            for value in pq.read_table(path, columns=["symbol"])
            .column("symbol")
            .to_pylist()
        )
    entry = {
        "path": str(pit_dir),
        "role": "Official point-in-time CSI800 membership snapshot.",
        "status": "do_not_move",
        "date_range": source["date_range"],
        "requested_source": source["requested"],
        "selected_source": source["selected"],
        "daily_file_count": len(daily_files),
        "member_day_count": member_day_count,
        "symbol_count": len(symbols),
        "interval_row_count": parquet_row_count(
            pit_dir / "membership_intervals.parquet"
        ),
        "membership_snapshot_hash": membership_hash,
    }
    if "fallback_reason" in source:
        entry["fallback_reason"] = source["fallback_reason"]
    return entry


def pit_manifest_entry(
    pit_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not pit_dir.is_dir():
        return None, {
            "path": str(pit_dir),
            "status": "not_ready",
            "reason_code": "membership_snapshot_missing",
        }
    try:
        with membership_read_lock(pit_dir) as snapshot:
            return _pit_manifest_entry_locked(pit_dir, snapshot), None
    except (*DATA_READ_ERRORS, MembershipIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "membership_snapshot_invalid")
        return None, {
            "path": str(pit_dir),
            "status": "not_ready",
            "reason_code": reason_code,
        }


def reference_manifest_entry(
    reference_dir: Path,
    pit_entry: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    missing = [
        name for name in REFERENCE_SNAPSHOT_FILES if not (reference_dir / name).is_file()
    ]
    if missing:
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": "reference_snapshot_incomplete",
            "missing_files": missing,
        }
    if pit_entry is None:
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": "membership_snapshot_not_ready",
        }
    try:
        with reference_read_lock(reference_dir) as snapshot:
            snapshot.ensure_transaction_clean()
            date_range = pit_entry["date_range"]
            source = snapshot.validate_source(
                start_date=str(date_range["start_date"]),
                end_date=str(date_range["end_date"]),
                expected_membership_snapshot_hash=str(
                    pit_entry["membership_snapshot_hash"]
                ),
            )
            reference_hash = snapshot.fingerprint()
            entry = {
                "path": str(reference_dir),
                "role": "Official PIT security, industry, ST, and suspension metadata.",
                "status": "do_not_move",
                "date_range": source["date_range"],
                "requested_source": source["requested"],
                "selected_source": source["selected"],
                "industry_classification": source["industry_classification"],
                "membership_snapshot_hash": source["membership_snapshot_hash"],
                "reference_snapshot_hash": reference_hash,
                "row_counts": {
                    name: parquet_row_count(reference_dir / name)
                    for name in REFERENCE_PARQUET_FILES
                },
            }
            if "fallback_reason" in source:
                entry["fallback_reason"] = source["fallback_reason"]
            if "build_run_id" in source:
                entry["build_run_id"] = source["build_run_id"]
            return entry, None
    except (*DATA_READ_ERRORS, ReferenceIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "reference_snapshot_invalid")
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": reason_code,
        }


def robust_manifest_entries(reference_dir: Path) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    pit_dir = reference_dir / "pit_csi800"
    if not pit_dir.is_dir():
        pit_entry, pit_pending = pit_manifest_entry(pit_dir)
        _, reference_pending = reference_manifest_entry(reference_dir, None)
        return pit_entry, pit_pending, None, reference_pending
    try:
        with membership_read_lock(pit_dir) as snapshot:
            pit_entry = _pit_manifest_entry_locked(pit_dir, snapshot)
            reference_entry, reference_pending = reference_manifest_entry(
                reference_dir, pit_entry
            )
            return pit_entry, None, reference_entry, reference_pending
    except (*DATA_READ_ERRORS, MembershipIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "membership_snapshot_invalid")
        return (
            None,
            {
                "path": str(pit_dir),
                "status": "not_ready",
                "reason_code": reason_code,
            },
            None,
            {
                "path": str(reference_dir),
                "status": "not_ready",
                "reason_code": "membership_snapshot_not_ready",
            },
        )


def build_manifest(*, quant_data_root: Path, reports_root: Path) -> dict[str, Any]:
    quant_data = Path(quant_data_root).expanduser()
    reports = Path(reports_root).expanduser()
    parquet_root = quant_data / "parquet" / "ashare_1d"
    qlib_root = quant_data / "qlib"

    cleaned_parquet = parquet_root / "merged_baostock_jqdata_cleaned"
    cleaned_provider = qlib_root / "cn_data_baostock_jqdata_cleaned"
    canonical_provider = qlib_root / "cn_data_baostock_jqdata"
    reference_root = quant_data / "reference"
    pit_entry, pit_pending, reference_entry, reference_pending = (
        robust_manifest_entries(reference_root)
    )

    official_entries: dict[str, Any] = {
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
    }
    if pit_entry is not None:
        official_entries["pit_csi800"] = pit_entry
    if reference_entry is not None:
        official_entries["robust_reference"] = reference_entry
    pending_entries: dict[str, Any] = {}
    if pit_pending is not None:
        pending_entries["pit_csi800"] = pit_pending
    if reference_pending is not None:
        pending_entries["robust_reference"] = reference_pending

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
        "official_entries": official_entries,
        "pending_entries": pending_entries,
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
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
