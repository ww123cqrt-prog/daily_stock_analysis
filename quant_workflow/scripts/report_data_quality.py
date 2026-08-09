"""Generate a data quality report for the local A-share Qlib dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (  # noqa: E402
    qlib_workflow_provider_descriptor,
    qlib_workflow_provider_uri,
    quant_data_root,
    reports_root,
    workflow_root,
)
from _symbols import normalize_symbol  # noqa: E402
from quant_data_contract import (  # noqa: E402
    AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS,
    AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
    DataContractError,
    atomic_write_json,
    filesystem_state_fingerprint,
    resolve_dataset,
)


QUANT_DATA = quant_data_root()
WORKFLOW = workflow_root()

BAOSTOCK_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "baostock_qfq"
JQDATA_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "jqdata"
EASTMONEY_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "eastmoney_qfq"
AKSHARE_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "akshare_qfq"
MERGED_DIR: Path | None = None
QLIB_DIR: Path | None = None
STOCK_LIST = QUANT_DATA / "stock_lists.json"
EMPTY_CODES = QUANT_DATA / "baostock_empty_codes.json"
TIMEOUT_CODES = QUANT_DATA / "baostock_timeout_codes.json"
REPORT_DIR = reports_root()
LATEST_TRAINING_RESULT_FILENAME = "latest_training_result.json"
BENCHMARK_SYMBOLS = ("SH000300", "SH000905", "SH000906")
BENCHMARK_FEATURES = {symbol.lower() for symbol in BENCHMARK_SYMBOLS}
BENCHMARK_FIELDS = ("open", "high", "low", "close", "volume", "factor")
REQUIRED_INSTRUMENT_FEATURE_FILES = tuple(
    f"{field}.day.bin" for field in BENCHMARK_FIELDS
)
QLIB_INSTRUMENT_RE = re.compile(r"^(?:SH|SZ|BJ)\d{6}$", re.IGNORECASE)
MIN_REQUIRED_FEATURE_COVERAGE = 0.50
QLIB_OHLC_RELATIVE_TOLERANCE = 2e-6
QLIB_OHLC_ABSOLUTE_TOLERANCE = 1e-4

NUMERIC_FIELDS = ["open", "close", "high", "low", "volume"]

DATA_SOURCE_METADATA = {
    "ashare_daily_qfq": {
        "dataset_id": "ashare_daily_qfq",
        "role": "canonical",
        "lifecycle": "immutable_snapshot_v2",
    },
    "baostock": {
        "dataset_id": "ashare_baostock_qfq_legacy",
        "role": "audit",
        "lifecycle": "legacy_unknown",
    },
    "jqdata": {
        "dataset_id": "ashare_jqdata_qfq_legacy",
        "role": "audit",
        "lifecycle": "legacy_unknown",
    },
    "eastmoney": {
        "dataset_id": "ashare_eastmoney_qfq_legacy",
        "role": "audit",
        "lifecycle": "legacy_unknown",
    },
    "akshare": {
        "dataset_id": "ashare_akshare_qfq_legacy",
        "role": "audit",
        "lifecycle": "legacy_unknown",
    },
    "merged": {
        "dataset_id": "ashare_merged_qfq_compatibility",
        "role": "legacy",
        "lifecycle": "compatibility_output",
    },
}


@dataclass
class SourceStats:
    name: str
    files: int
    rows: int
    first_date: str
    last_date: str
    size_mb: float


def native_dataset_identity(resolved) -> dict[str, str]:
    entry = resolved.catalog_entry
    return {
        "dataset_id": str(resolved.dataset_id),
        "snapshot_id": str(entry["snapshot_id"]),
        "manifest_sha256": str(entry["snapshot_manifest_sha256"]),
        "content_sha256": str(entry["content_sha256"]),
    }


def direct_provider_identity(descriptor: Path, quant_data_root_path: Path) -> dict[str, str]:
    payload = json.loads(Path(descriptor).read_text(encoding="utf-8"))
    dataset = payload.get("dataset") or {}
    index = payload.get("index_dataset") or {}
    cleaning = payload.get("cleaning") or {}
    descriptor = Path(descriptor).resolve()
    root = Path(quant_data_root_path).resolve()
    return {
        "provider_contract": "direct_parquet_v1",
        "dataset_id": str(dataset.get("dataset_id") or ""),
        "snapshot_id": str(dataset.get("snapshot_id") or ""),
        "manifest_sha256": str(dataset.get("manifest_sha256") or ""),
        "content_sha256": str(dataset.get("content_sha256") or ""),
        "descriptor_path": descriptor.relative_to(root).as_posix(),
        "descriptor_sha256": hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        "cleaning_mode": str(cleaning.get("mode") or ""),
        "index_dataset_id": str(index.get("dataset_id") or ""),
        "index_snapshot_id": str(index.get("snapshot_id") or ""),
        "index_content_sha256": str(index.get("content_sha256") or ""),
        "index_manifest_sha256": str(index.get("manifest_sha256") or ""),
    }


def build_quality_data_contract(
    *,
    ashare_resolution,
    canonical_qlib_resolution,
    cleaned_qlib_resolution,
    has_path_override: bool,
    canonical_qlib_identity: dict[str, str] | None = None,
    cleaned_qlib_identity: dict[str, str] | None = None,
) -> dict:
    contract = {
        "status": "unverified" if has_path_override else "verified",
        "reason": (
            "Explicit data path overrides are diagnostic only and are not proof of the current formal Parquet/direct Provider snapshot."
            if has_path_override
            else "Resolved from the current formal Parquet snapshots and immutable LATEST pointers; Qlib reads them through the direct Provider descriptor."
        ),
    }
    if ashare_resolution is not None:
        contract["ashare_daily_qfq"] = native_dataset_identity(ashare_resolution)

    providers = {}
    if canonical_qlib_identity is not None:
        providers["canonical"] = dict(canonical_qlib_identity)
        contract["qlib_provider"] = providers["canonical"]
    elif canonical_qlib_resolution is not None:
        providers["canonical"] = native_dataset_identity(
            canonical_qlib_resolution
        )
        contract["qlib_provider"] = providers["canonical"]
    if cleaned_qlib_identity is not None:
        providers["cleaned"] = dict(cleaned_qlib_identity)
    elif cleaned_qlib_resolution is not None:
        providers["cleaned"] = native_dataset_identity(cleaned_qlib_resolution)
    if providers:
        contract["qlib_providers"] = providers
    return contract


def native_snapshot_resolutions(*resolutions) -> dict[str, object]:
    return {
        str(resolution.dataset_id): resolution
        for resolution in resolutions
        if resolution is not None
    }


def capture_native_snapshot_state(resolutions: dict[str, object]) -> dict[str, dict]:
    return {
        dataset_id: filesystem_state_fingerprint(resolution.snapshot_path)
        for dataset_id, resolution in sorted(resolutions.items())
    }


def revalidate_native_snapshot_resolutions(
    resolutions: dict[str, object],
) -> None:
    for dataset_id, expected in sorted(resolutions.items()):
        expected_entry = expected.catalog_entry
        refreshed = resolve_dataset(
            dataset_id,
            quant_data_root=QUANT_DATA,
            require_basis=expected_entry.get("price_basis"),
        )
        if (
            refreshed.snapshot_path.resolve(strict=True)
            != expected.snapshot_path.resolve(strict=True)
            or native_dataset_identity(refreshed) != native_dataset_identity(expected)
        ):
            raise DataContractError(
                f"native snapshot identity changed during quality scan: {dataset_id}"
            )


def revalidate_native_snapshot_content_for_scan(
    resolutions: dict[str, object],
    *,
    scan_mode: str,
) -> str:
    if scan_mode == "sampled":
        return "skipped_sampled"
    if scan_mode != "full":
        raise ValueError(f"unsupported quality scan mode: {scan_mode}")
    revalidate_native_snapshot_resolutions(resolutions)
    return "verified"


def bind_snapshot_state(
    data_contract: dict,
    *,
    before: dict[str, dict],
    after_scan: dict[str, dict],
    after_revalidation: dict[str, dict],
    fingerprint_failed: bool,
    content_revalidation_status: str,
) -> None:
    if content_revalidation_status not in {"verified", "failed", "skipped_sampled"}:
        raise ValueError("invalid content revalidation status")
    stable = (
        bool(before)
        and not fingerprint_failed
        and content_revalidation_status == "verified"
        and before == after_scan == after_revalidation
    )
    data_contract["snapshot_state"] = {
        "stable": stable,
        "before": before,
        "after_scan": after_scan,
        "after_revalidation": after_revalidation,
        "after": after_revalidation,
        "content_revalidated": content_revalidation_status == "verified",
        "content_revalidation_status": content_revalidation_status,
    }
    if stable:
        return
    data_contract["status"] = "unverified"
    if content_revalidation_status == "failed":
        data_contract["reason"] = (
            "Formal Parquet/direct Provider snapshot content could not be revalidated after the "
            "data-quality scan."
        )
    elif fingerprint_failed:
        data_contract["reason"] = (
            "Formal Parquet/direct Provider snapshot filesystem state could not be verified across the "
            "data-quality scan."
        )
    elif content_revalidation_status == "skipped_sampled":
        data_contract["reason"] = (
            "Sampled quality scans skip the second full formal snapshot content "
            "revalidation; run a full quality scan before training."
        )
    else:
        data_contract["reason"] = (
            "Formal Parquet/direct Provider snapshot filesystem state changed during the data-quality scan."
        )


def read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def read_stock_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    raw_codes = data.get("all", []) if isinstance(data, dict) else []
    return [str(code).split(".")[0] for code in raw_codes if str(code).strip()]


def file_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def source_stats(name: str, directory: Path) -> SourceStats:
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    rows = 0
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for path in files:
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            continue
        rows += len(df)
        mins.append(dates.min())
        maxs.append(dates.max())

    return SourceStats(
        name=name,
        files=len(files),
        rows=rows,
        first_date=str(min(mins).date()) if mins else "",
        last_date=str(max(maxs).date()) if maxs else "",
        size_mb=file_size_mb(directory),
    )


def load_source_index(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {
        normalize_symbol(path.stem, strip_any_suffix=True, strip_prefix=True)
        for path in directory.glob("*.parquet")
    }


def _blocking_merged_summary(symbol: str, row_count: int, reason: str) -> dict:
    summary = {
        "symbol": symbol,
        "rows": row_count,
        "start": "",
        "end": "",
        "duplicate_dates": 0,
        "null_dates": row_count,
        "missing_columns": reason,
        "bad_ohlc_rows": row_count,
        "abs_return_gt_30pct": 0,
        "abs_return_gt_50pct": 0,
    }
    for field in NUMERIC_FIELDS:
        summary[f"null_{field}"] = row_count
        summary[f"non_positive_{field}"] = 0
    return summary


def scan_merged_files(directory: Path, sample_limit: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    issue_rows: list[dict] = []
    files = sorted(directory.glob("*.parquet"))
    if sample_limit > 0:
        files = files[:sample_limit]

    for i, path in enumerate(files, 1):
        sym = path.stem
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            issue_rows.append(
                {
                    "symbol": sym,
                    "issue": "unreadable_parquet",
                    "count": 1,
                    "detail": type(exc).__name__,
                }
            )
            summary_rows.append(
                _blocking_merged_summary(sym, 0, "unreadable_parquet")
            )
            continue
        if "date" not in df.columns:
            issue_rows.append({"symbol": sym, "issue": "missing_date_column", "count": 1, "detail": ""})
            summary_rows.append(_blocking_merged_summary(sym, len(df), "date"))
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        row_count = len(df)
        duplicate_dates = int(df["date"].duplicated().sum())
        null_dates = int(df["date"].isna().sum())
        missing_cols = [c for c in NUMERIC_FIELDS if c not in df.columns]

        summary = {
            "symbol": sym,
            "rows": row_count,
            "start": str(df["date"].min().date()) if not df["date"].dropna().empty else "",
            "end": str(df["date"].max().date()) if not df["date"].dropna().empty else "",
            "duplicate_dates": duplicate_dates,
            "null_dates": null_dates,
            "missing_columns": ",".join(missing_cols),
        }

        for col in NUMERIC_FIELDS:
            if col not in df.columns:
                summary[f"null_{col}"] = row_count
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            summary[f"null_{col}"] = int(values.isna().sum())
            summary[f"non_positive_{col}"] = int((values <= 0).sum())

        if {"open", "high", "low", "close"}.issubset(df.columns):
            open_ = pd.to_numeric(df["open"], errors="coerce")
            high = pd.to_numeric(df["high"], errors="coerce")
            low = pd.to_numeric(df["low"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            bad_ohlc = (
                (high < low)
                | (high < open_)
                | (high < close)
                | (low > open_)
                | (low > close)
            )
            summary["bad_ohlc_rows"] = int(bad_ohlc.fillna(False).sum())
        else:
            summary["bad_ohlc_rows"] = row_count

        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            pct = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            summary["abs_return_gt_30pct"] = int((pct.abs() > 0.30).sum())
            summary["abs_return_gt_50pct"] = int((pct.abs() > 0.50).sum())
        else:
            summary["abs_return_gt_30pct"] = 0
            summary["abs_return_gt_50pct"] = 0

        summary_rows.append(summary)

        for field in [
            "duplicate_dates",
            "null_dates",
            "missing_columns",
            "bad_ohlc_rows",
            "abs_return_gt_50pct",
        ]:
            value = summary.get(field)
            if value and value != "":
                issue_rows.append({"symbol": sym, "issue": field, "count": value, "detail": summary.get("missing_columns", "")})

        if i % 1000 == 0:
            print(f"  scanned {i}/{len(files)} A-share qfq files")

    return pd.DataFrame(summary_rows), pd.DataFrame(issue_rows)


def _compare_overlap_to_baostock(
    *,
    source_name: str,
    source_dir: Path,
    sample_limit: int = 0,
) -> list[dict]:
    bs_files = load_source_index(BAOSTOCK_DIR)
    source_files = load_source_index(source_dir)
    symbols = sorted(bs_files & source_files)
    if sample_limit > 0:
        symbols = symbols[:sample_limit]

    rows: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        bs = pd.read_parquet(BAOSTOCK_DIR / f"{sym}.parquet")
        other = pd.read_parquet(source_dir / f"{sym}.parquet")
        bs["date"] = pd.to_datetime(bs["date"], errors="coerce")
        other["date"] = pd.to_datetime(other["date"], errors="coerce")
        joined = bs.merge(other, on="date", suffixes=("_bs", "_other"))
        if joined.empty:
            rows.append({"source": source_name, "symbol": sym, "overlap_days": 0})
            continue

        row = {"source": source_name, "symbol": sym, "overlap_days": len(joined)}
        for col in ["open", "close", "high", "low"]:
            diff = (
                pd.to_numeric(joined[f"{col}_bs"], errors="coerce")
                - pd.to_numeric(joined[f"{col}_other"], errors="coerce")
            ).abs()
            base = pd.to_numeric(joined[f"{col}_other"], errors="coerce").abs().replace(0, np.nan)
            rel = (diff / base).replace([np.inf, -np.inf], np.nan)
            row[f"{col}_max_abs_diff"] = float(diff.max()) if not diff.empty else math.nan
            row[f"{col}_median_rel_diff"] = float(rel.median()) if not rel.dropna().empty else math.nan

        if "volume_bs" in joined.columns and "volume_other" in joined.columns:
            vol_bs = pd.to_numeric(joined["volume_bs"], errors="coerce")
            vol_other = pd.to_numeric(joined["volume_other"], errors="coerce").replace(0, np.nan)
            ratio = (vol_bs / vol_other).replace([np.inf, -np.inf], np.nan)
            row["volume_ratio_median"] = float(ratio.median()) if not ratio.dropna().empty else math.nan
            row["volume_ratio_min"] = float(ratio.min()) if not ratio.dropna().empty else math.nan
            row["volume_ratio_max"] = float(ratio.max()) if not ratio.dropna().empty else math.nan

        rows.append(row)
        if i % 1000 == 0:
            print(f"  compared {i}/{len(symbols)} baostock/{source_name} overlap files")

    return rows


def overlap_consistency(sample_limit: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    for source_name, source_dir in [
        ("jqdata", JQDATA_DIR),
        ("eastmoney", EASTMONEY_DIR),
        ("akshare", AKSHARE_DIR),
    ]:
        if not source_dir.exists():
            continue
        rows.extend(
            _compare_overlap_to_baostock(
                source_name=source_name,
                source_dir=source_dir,
                sample_limit=sample_limit,
            )
        )
    return pd.DataFrame(rows)


def _read_qlib_calendar(path: Path) -> tuple[list[str], list[str]]:
    try:
        if path.is_symlink() or not path.is_file():
            return [], ["calendar file is unavailable"]
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return [], ["calendar file is unreadable"]
    issues: list[str] = []
    previous = ""
    for line_number, raw in enumerate(lines, 1):
        try:
            normalized = datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
        except ValueError:
            issues.append(f"line {line_number}: invalid ISO date")
            continue
        if normalized != raw:
            issues.append(f"line {line_number}: date is not canonical ISO format")
        if previous and raw <= previous:
            issues.append(f"line {line_number}: dates are not strictly increasing")
        previous = raw
    if not lines:
        issues.append("calendar is empty")
    return lines, issues


def _read_qlib_instrument_intervals(
    path: Path,
    calendar: list[str],
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    try:
        if path.is_symlink() or not path.is_file():
            return {}, ["instrument file is unavailable"]
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}, ["instrument file is unreadable"]
    calendar_positions = {value: index for index, value in enumerate(calendar)}
    intervals: dict[str, tuple[int, int]] = {}
    issues: list[str] = []
    for line_number, raw in enumerate(lines, 1):
        parts = raw.split("\t")
        if len(parts) != 3:
            issues.append(f"line {line_number}: expected symbol, start, and end")
            continue
        symbol, start, end = parts
        normalized_symbol = symbol.lower()
        if not QLIB_INSTRUMENT_RE.fullmatch(symbol):
            issues.append(f"line {line_number}: invalid instrument symbol")
            continue
        if normalized_symbol in intervals:
            issues.append(f"line {line_number}: duplicate instrument symbol")
            continue
        if start not in calendar_positions or end not in calendar_positions:
            issues.append(f"line {line_number}: interval is outside the calendar")
            continue
        start_index = calendar_positions[start]
        end_index = calendar_positions[end]
        if start_index > end_index:
            issues.append(f"line {line_number}: interval start is after end")
            continue
        intervals[normalized_symbol] = (start_index, end_index)
    if not lines:
        issues.append("instrument file is empty")
    return intervals, issues


def _read_qlib_feature_binary(
    path: Path,
    *,
    calendar_size: int,
) -> tuple[int, np.ndarray] | None:
    path = Path(path)
    if calendar_size <= 0 or path.is_symlink() or not path.is_file():
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    file_fd: int | None = None
    try:
        file_fd = os.open(path, flags)
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 8
            or before.st_size % 4 != 0
            or before.st_size > (calendar_size + 1) * 4
        ):
            return None
        with os.fdopen(file_fd, "rb", closefd=False) as handle:
            payload = handle.read(before.st_size)
            after = os.fstat(file_fd)
        if len(payload) != before.st_size:
            return None
        current = path.lstat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or identity_before != identity_after
            or identity_before != identity_current
        ):
            return None
        raw = np.frombuffer(payload, dtype="<f4")
        start_value = float(raw[0])
        if (
            not math.isfinite(start_value)
            or not start_value.is_integer()
            or start_value < 0
            or int(start_value) + len(raw) - 1 > calendar_size
        ):
            return None
        return int(start_value), raw[1:].copy()
    except (OSError, ValueError):
        return None
    finally:
        if file_fd is not None:
            os.close(file_fd)


def _valid_qlib_field_mask(field: str, values: np.ndarray) -> np.ndarray | None:
    if field not in BENCHMARK_FIELDS or bool(np.isinf(values).any()):
        return None
    finite = np.isfinite(values)
    finite_values = values[finite]
    if not finite_values.size:
        return None
    if field in {"open", "high", "low", "close", "factor"}:
        if bool((finite_values <= 0).any()):
            return None
    elif field == "volume" and bool((finite_values < 0).any()):
        return None
    return finite


def _invalid_qlib_ohlc_positions(
    values_by_field: dict[str, np.ndarray],
) -> np.ndarray:
    try:
        open_ = values_by_field["open"]
        high = values_by_field["high"]
        low = values_by_field["low"]
        close = values_by_field["close"]
    except KeyError:
        return np.array([], dtype=np.int64)
    if not (open_.shape == high.shape == low.shape == close.shape):
        return np.array([], dtype=np.int64)
    finite = np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    if not bool(finite.any()):
        return np.array([], dtype=np.int64)
    upper_reference = np.maximum.reduce([open_, low, close])
    lower_reference = np.minimum.reduce([open_, high, close])
    invalid = (
        (high < upper_reference)
        & ~np.isclose(
            high,
            upper_reference,
            rtol=QLIB_OHLC_RELATIVE_TOLERANCE,
            atol=QLIB_OHLC_ABSOLUTE_TOLERANCE,
        )
    ) | (
        (low > lower_reference)
        & ~np.isclose(
            low,
            lower_reference,
            rtol=QLIB_OHLC_RELATIVE_TOLERANCE,
            atol=QLIB_OHLC_ABSOLUTE_TOLERANCE,
        )
    )
    return np.flatnonzero(finite & invalid)


def _valid_qlib_ohlc_relationships(values_by_field: dict[str, np.ndarray]) -> bool:
    try:
        required = [values_by_field[field] for field in ("open", "high", "low", "close")]
    except KeyError:
        return False
    if not (required[0].shape == required[1].shape == required[2].shape == required[3].shape):
        return False
    finite = np.isfinite(required[0]) & np.isfinite(required[1]) & np.isfinite(required[2]) & np.isfinite(required[3])
    return bool(finite.any()) and not len(_invalid_qlib_ohlc_positions(values_by_field))


def _aligned_benchmark_feature(
    path: Path,
    calendar_size: int,
    *,
    field: str,
) -> np.ndarray | None:
    loaded = _read_qlib_feature_binary(path, calendar_size=calendar_size)
    if loaded is None:
        return None
    start_idx, values = loaded
    if _valid_qlib_field_mask(field, values) is None:
        return None
    aligned = np.full(calendar_size, np.nan, dtype=np.float32)
    aligned[start_idx : start_idx + len(values)] = values
    return aligned


def benchmark_feature_stats(provider_dir: Path, calendar: list[str]) -> dict:
    details: dict[str, dict] = {}
    total_missing_days = 0
    first_observations: list[str] = []
    for symbol in BENCHMARK_SYMBOLS:
        feature_dir = provider_dir / "features" / symbol.lower()
        missing_fields: list[str] = []
        masks: list[np.ndarray] = []
        first_positions: list[int] = []
        aligned_fields: dict[str, np.ndarray] = {}
        invalid_ohlc_dates: list[str] = []
        price_missing_days = len(calendar)
        factor_missing_days = len(calendar)
        for field in BENCHMARK_FIELDS:
            aligned = _aligned_benchmark_feature(
                feature_dir / f"{field}.day.bin",
                len(calendar),
                field=field,
            )
            if aligned is None:
                missing_fields.append(field)
                continue
            aligned_fields[field] = aligned
            valid = np.isfinite(aligned)
            valid &= aligned >= 0 if field == "volume" else aligned > 0
            positions = np.flatnonzero(valid)
            if not len(positions):
                missing_fields.append(field)
                continue
            if field == "factor":
                factor_missing_days = int((~valid).sum())
                if factor_missing_days:
                    missing_fields.append(field)
                continue
            first_positions.append(int(positions[0]))
            masks.append(valid)
        prices_complete = len(masks) == 5
        same_dates = False
        if prices_complete:
            first = max(first_positions)
            same_dates = all(np.array_equal(masks[0], mask) for mask in masks[1:])
            price_missing_days = int((~masks[0][first:]).sum())
            first_observation = calendar[first]
            first_observations.append(first_observation)
            invalid_positions = _invalid_qlib_ohlc_positions(aligned_fields)
            invalid_ohlc_dates = [calendar[int(index)] for index in invalid_positions]
            ohlc_valid = not invalid_ohlc_dates
            if not ohlc_valid:
                missing_fields.append("ohlc_relationship")
        else:
            first_observation = ""
            ohlc_valid = False
        missing_days = max(price_missing_days, factor_missing_days)
        complete = (
            not missing_fields
            and prices_complete
            and same_dates
            and ohlc_valid
            and missing_days == 0
        )
        details[symbol] = {
            "complete": complete,
            "first_observation": first_observation,
            "missing_days": missing_days,
            "missing_fields": missing_fields,
            "invalid_ohlc_dates": invalid_ohlc_dates,
        }
        total_missing_days += missing_days
    return {
        "benchmark_complete": bool(calendar)
        and all(details[symbol]["complete"] for symbol in BENCHMARK_SYMBOLS),
        "benchmark_first_observation": min(first_observations)
        if first_observations
        else "",
        "benchmark_missing_days": total_missing_days,
        "benchmark_details": details,
    }


def qlib_provider_stats(provider_dir: Path | None = None) -> dict:
    provider_dir = Path(provider_dir or qlib_workflow_provider_uri())
    descriptor = (
        provider_dir
        if provider_dir.name == "PROVIDER_DESCRIPTOR.json"
        else provider_dir / "PROVIDER_DESCRIPTOR.json"
    )
    if descriptor.is_file():
        return direct_qlib_provider_stats(provider_dir)
    cal = provider_dir / "calendars" / "day.txt"
    inst = provider_dir / "instruments" / "all.txt"
    calendar_lines, calendar_issues = _read_qlib_calendar(cal)
    calendar = calendar_lines if not calendar_issues else []
    instrument_intervals, instrument_issues = _read_qlib_instrument_intervals(
        inst,
        calendar,
    )
    instrument_names = set(instrument_intervals)
    feature_dirs = (
        [
            path
            for path in (provider_dir / "features").iterdir()
            if path.is_dir() and not path.is_symlink()
        ]
        if (provider_dir / "features").exists()
        else []
    )
    feature_dirs_by_name = {path.name.lower(): path for path in feature_dirs}
    feature_names = set(feature_dirs_by_name)
    instrument_feature_names = feature_names & instrument_names
    missing_instrument_features = sorted(instrument_names - feature_names)
    extra_features = sorted(feature_names - instrument_names)
    unexpected_extra = [name for name in extra_features if name not in BENCHMARK_FEATURES]
    complete_instrument_features = []
    incomplete_instrument_features = []
    missing_instrument_feature_files = []
    invalid_instrument_feature_files = []
    misaligned_instrument_features = []
    valid_feature_coverages: list[float] = []
    for instrument_name in sorted(instrument_names):
        feature_dir = feature_dirs_by_name.get(instrument_name)
        available_files = (
            {
                path.name
                for path in feature_dir.iterdir()
                if path.is_file() and not path.is_symlink()
            }
            if feature_dir is not None
            else set()
        )
        missing_files = [
            name
            for name in REQUIRED_INSTRUMENT_FEATURE_FILES
            if name not in available_files
        ]
        start_index, end_index = instrument_intervals[instrument_name]
        validations = {
            name: validate_qlib_feature_binary(
                feature_dir / name,
                calendar_size=len(calendar),
                field=name.removesuffix(".day.bin"),
                expected_start_index=start_index,
                expected_end_index=end_index,
            )
            for name in REQUIRED_INSTRUMENT_FEATURE_FILES
            if name in available_files
        }
        invalid_files = [
            name for name, validation in validations.items() if validation is None
        ]
        valid_masks = [
            validation["mask_sha256"]
            for validation in validations.values()
            if validation is not None
        ]
        valid_feature_coverages.extend(
            validation["coverage"]
            for validation in validations.values()
            if validation is not None
        )
        masks_aligned = len(valid_masks) == len(REQUIRED_INSTRUMENT_FEATURE_FILES) and (
            len(set(valid_masks)) == 1
        )
        if not missing_files and not invalid_files and masks_aligned:
            values_by_field = {
                name.removesuffix(".day.bin"): validation["values"]
                for name, validation in validations.items()
                if validation is not None
            }
            if not _valid_qlib_ohlc_relationships(values_by_field):
                invalid_files.append("ohlc_relationship")
        if not masks_aligned and not missing_files and not invalid_files:
            misaligned_instrument_features.append(instrument_name)
        if missing_files or invalid_files or not masks_aligned:
            incomplete_instrument_features.append(instrument_name)
            missing_instrument_feature_files.extend(
                f"{instrument_name}/{name}" for name in missing_files
            )
            invalid_instrument_feature_files.extend(
                f"{instrument_name}/{name}" for name in invalid_files
            )
        else:
            complete_instrument_features.append(instrument_name)
    return {
        "calendar_valid": bool(calendar_lines) and not calendar_issues,
        "invalid_calendar_entries": len(calendar_issues),
        "invalid_calendar_entry_sample": "; ".join(calendar_issues[:20]),
        "instrument_records_valid": bool(instrument_intervals)
        and not instrument_issues,
        "invalid_instrument_records": len(instrument_issues),
        "invalid_instrument_record_sample": "; ".join(instrument_issues[:20]),
        "calendar_days": len(calendar_lines),
        "calendar_start": calendar[0] if calendar else "",
        "calendar_end": calendar[-1] if calendar else "",
        "instruments": len(instrument_names),
        "feature_dirs": len(feature_dirs),
        "instrument_feature_dirs": len(instrument_feature_names),
        "missing_instrument_feature_dirs": len(missing_instrument_features),
        "missing_instrument_feature_sample": ", ".join(
            missing_instrument_features[:20]
        ),
        "complete_instrument_feature_dirs": len(complete_instrument_features),
        "incomplete_instrument_feature_dirs": len(incomplete_instrument_features),
        "missing_instrument_feature_files": len(missing_instrument_feature_files),
        "missing_instrument_feature_file_sample": ", ".join(
            missing_instrument_feature_files[:20]
        ),
        "invalid_instrument_feature_files": len(invalid_instrument_feature_files),
        "invalid_instrument_feature_file_sample": ", ".join(
            invalid_instrument_feature_files[:20]
        ),
        "required_feature_min_coverage": MIN_REQUIRED_FEATURE_COVERAGE,
        "minimum_valid_instrument_feature_coverage": (
            min(valid_feature_coverages) if valid_feature_coverages else 0.0
        ),
        "misaligned_instrument_feature_dirs": len(
            misaligned_instrument_features
        ),
        "misaligned_instrument_feature_dir_sample": ", ".join(
            misaligned_instrument_features[:20]
        ),
        "benchmark_feature_dirs": len([name for name in extra_features if name in BENCHMARK_FEATURES]),
        "unexpected_extra_feature_dirs": len(unexpected_extra),
        "unexpected_extra_feature_sample": ", ".join(unexpected_extra[:20]),
        "size_mb": file_size_mb(provider_dir),
        **benchmark_feature_stats(provider_dir, calendar),
    }


def direct_qlib_provider_stats(provider_dir: Path) -> dict:
    """Validate the metadata and frozen Parquet view used by the direct Provider.

    Direct mode has no per-field ``.bin`` files.  The descriptor and its frozen
    file indexes prove the storage identity; the existing full bar scan proves
    row-level quality.  The returned keys intentionally match the old report
    shape so the readiness rules remain shared.
    """

    from quant_workflow.quant_data_qlib_provider import (
        _load_descriptor,
        discover_quant_data_root,
    )

    provider_dir = Path(provider_dir)
    descriptor = (
        provider_dir
        if provider_dir.name == "PROVIDER_DESCRIPTOR.json"
        else provider_dir / "PROVIDER_DESCRIPTOR.json"
    )
    root = discover_quant_data_root(descriptor)
    context = _load_descriptor(descriptor, root)
    cleaning = context.cleaning
    excluded = set(cleaning.get("plan", {}).get("excluded_symbols", []))
    instrument_index = context.file_index[
        ~context.file_index["symbol"].astype(str).isin(excluded)
    ].copy()
    calendar = [value.date().isoformat() for value in context.calendar]
    instrument_names = set(instrument_index["symbol"].astype(str))
    benchmark_stats = direct_benchmark_feature_stats(context, calendar)
    complete = len(instrument_names)
    return {
        "calendar_valid": True,
        "invalid_calendar_entries": 0,
        "invalid_calendar_entry_sample": "",
        "instrument_records_valid": True,
        "invalid_instrument_records": 0,
        "invalid_instrument_record_sample": "",
        "calendar_days": len(calendar),
        "calendar_start": calendar[0] if calendar else "",
        "calendar_end": calendar[-1] if calendar else "",
        "instruments": len(instrument_names),
        "feature_dirs": complete,
        "instrument_feature_dirs": complete,
        "missing_instrument_feature_dirs": 0,
        "missing_instrument_feature_sample": "",
        "complete_instrument_feature_dirs": complete,
        "incomplete_instrument_feature_dirs": 0,
        "missing_instrument_feature_files": 0,
        "missing_instrument_feature_file_sample": "",
        "invalid_instrument_feature_files": 0,
        "invalid_instrument_feature_file_sample": "",
        "required_feature_min_coverage": MIN_REQUIRED_FEATURE_COVERAGE,
        "minimum_valid_instrument_feature_coverage": 1.0,
        "misaligned_instrument_feature_dirs": 0,
        "misaligned_instrument_feature_dir_sample": "",
        "benchmark_feature_dirs": (
            len(context.index_file_index) if context.index_file_index is not None else 0
        ),
        "unexpected_extra_feature_dirs": 0,
        "unexpected_extra_feature_sample": "",
        "size_mb": file_size_mb(provider_dir),
        **benchmark_stats,
    }


def direct_benchmark_feature_stats(context: Any, calendar: list[str]) -> dict:
    required = tuple(
        context.descriptor.get("index_dataset", {}).get("required_symbols")
        or BENCHMARK_SYMBOLS
    )
    details: dict[str, dict[str, Any]] = {}
    first_observations: list[str] = []
    total_missing_days = 0
    direct_fields = ("open", "high", "low", "close", "volume")
    index_frame = context.index_file_index
    index_snapshot = context.index_snapshot_path
    for symbol in required:
        detail = {
            "complete": False,
            "first_observation": "",
            "missing_days": 0,
            "missing_fields": [],
            "invalid_ohlc_dates": [],
        }
        row = None
        if index_frame is not None:
            matches = index_frame[index_frame["symbol"].astype(str) == symbol]
            if len(matches) == 1:
                row = matches.iloc[0]
        if row is None or index_snapshot is None:
            detail["missing_fields"] = list(direct_fields)
            details[symbol] = detail
            continue
        path = index_snapshot / str(row["relative_path"])
        try:
            frame = pd.read_parquet(path)
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            frame = frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
        except Exception:
            detail["missing_fields"] = list(direct_fields)
            details[symbol] = detail
            continue
        if frame.empty:
            detail["missing_fields"] = list(direct_fields)
            details[symbol] = detail
            continue
        detail["first_observation"] = frame["date"].iloc[0].date().isoformat()
        first_observations.append(detail["first_observation"])
        missing_fields = [field for field in direct_fields if field not in frame.columns]
        detail["missing_fields"] = missing_fields
        if not missing_fields:
            tolerance = np.maximum(
                np.abs(frame["high"].to_numpy(dtype=float)),
                np.abs(frame["low"].to_numpy(dtype=float)),
            ) * QLIB_OHLC_RELATIVE_TOLERANCE + QLIB_OHLC_ABSOLUTE_TOLERANCE
            high = frame["high"].to_numpy(dtype=float)
            low = frame["low"].to_numpy(dtype=float)
            open_values = frame["open"].to_numpy(dtype=float)
            close = frame["close"].to_numpy(dtype=float)
            invalid_mask = (
                (high + tolerance < np.maximum(open_values, close))
                | (low - tolerance > np.minimum(open_values, close))
                | (low <= 0)
                | (high <= 0)
            )
            detail["invalid_ohlc_dates"] = [
                value.date().isoformat()
                for value in frame.loc[invalid_mask, "date"].tolist()
            ]
            expected = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
            expected = expected[expected >= frame["date"].iloc[0]]
            present = pd.DatetimeIndex(frame["date"])
            detail["missing_days"] = int(len(expected.difference(present)))
        detail["complete"] = not detail["missing_fields"] and not detail["invalid_ohlc_dates"] and detail["missing_days"] == 0
        total_missing_days += detail["missing_days"]
        details[symbol] = detail
    return {
        "benchmark_complete": bool(details) and all(item["complete"] for item in details.values()),
        "benchmark_first_observation": min(first_observations) if first_observations else "",
        "benchmark_missing_days": total_missing_days,
        "benchmark_details": details,
    }


def validate_qlib_feature_binary(
    path: Path,
    *,
    calendar_size: int,
    field: str,
    expected_start_index: int,
    expected_end_index: int,
) -> dict[str, Any] | None:
    loaded = _read_qlib_feature_binary(path, calendar_size=calendar_size)
    if loaded is None:
        return None
    start_index, values = loaded
    expected_count = expected_end_index - expected_start_index + 1
    if start_index != expected_start_index or len(values) != expected_count:
        return None
    finite_mask = _valid_qlib_field_mask(field, values)
    if finite_mask is None:
        return None
    finite_count = int(finite_mask.sum())
    coverage = finite_count / expected_count
    if coverage < MIN_REQUIRED_FEATURE_COVERAGE:
        return None
    return {
        "coverage": coverage,
        "finite_count": finite_count,
        "value_count": expected_count,
        "mask_sha256": hashlib.sha256(finite_mask.tobytes()).hexdigest(),
        "values": values,
    }


def source_status(name: str, stats: SourceStats, merged_symbols: set[str]) -> str:
    if stats.files == 0 or stats.rows == 0:
        return "missing"
    if name in {"akshare", "eastmoney", "jqdata"}:
        return "partial"
    if name in {"ashare_daily_qfq", "merged"} and stats.files < max(len(merged_symbols), 1):
        return "partial"
    return "available"


def data_source_metadata(name: str) -> dict[str, str]:
    return dict(
        DATA_SOURCE_METADATA.get(
            name,
            {"dataset_id": name, "role": "audit", "lifecycle": "unknown"},
        )
    )


def load_latest_training_result(reports_dir: Path) -> dict | None:
    latest_training_result = reports_dir / LATEST_TRAINING_RESULT_FILENAME
    if not latest_training_result.exists():
        return None
    try:
        payload = json.loads(latest_training_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_training_readiness(
    qlib_stats: dict,
    merged_success: set[str],
    issue_counts: dict[str, int],
    *,
    min_calendar_days: int = 252,
    min_instruments: int = 100,
    min_merged_instruments: int = 100,
    scan_mode: str = "full",
) -> tuple[bool, dict[str, object]]:
    blockers = []
    sampled_scan = scan_mode != "full"
    if sampled_scan:
        blockers.append(
            "Data quality checks used a sample; a full scan is required before training"
        )
    if qlib_stats.get("calendar_valid") is not True:
        blockers.append(
            "Qlib calendar contract is invalid "
            f"({qlib_stats.get('invalid_calendar_entries', 'unknown')} issues)"
        )
    if qlib_stats.get("instrument_records_valid") is not True:
        blockers.append(
            "Qlib instrument contract is invalid "
            f"({qlib_stats.get('invalid_instrument_records', 'unknown')} issues)"
        )
    if qlib_stats["calendar_days"] < min_calendar_days:
        blockers.append(f"Qlib calendar has fewer than {min_calendar_days} trading days")
    if qlib_stats["instruments"] < min_instruments:
        blockers.append(f"Qlib provider has fewer than {min_instruments} instruments")
    if not qlib_stats.get("benchmark_complete", False):
        details = qlib_stats.get("benchmark_details") or {}
        benchmark_blockers: list[str] = []
        benchmark_symbols = tuple(
            dict.fromkeys([*BENCHMARK_SYMBOLS, *(str(symbol) for symbol in details)])
        )
        for symbol in benchmark_symbols:
            detail = details.get(symbol) or {}
            invalid_dates = list(detail.get("invalid_ohlc_dates") or [])
            if invalid_dates:
                if len(invalid_dates) == 1:
                    benchmark_blockers.append(
                        f"{symbol} benchmark OHLC relationship is invalid on {invalid_dates[0]}"
                    )
                else:
                    benchmark_blockers.append(
                        f"{symbol} benchmark OHLC relationship is invalid on "
                        f"{len(invalid_dates)} days (first: {invalid_dates[0]})"
                    )
                continue
            missing_fields = [
                field
                for field in detail.get("missing_fields") or []
                if field != "ohlc_relationship"
            ]
            if missing_fields:
                benchmark_blockers.append(
                    f"{symbol} benchmark fields are missing or invalid: "
                    + ", ".join(missing_fields)
                )
                continue
            missing_days = int(detail.get("missing_days") or 0)
            if missing_days:
                benchmark_blockers.append(
                    f"{symbol} benchmark is missing {missing_days} trading days after inception"
                )
                continue
            if detail.get("complete") is not True:
                benchmark_blockers.append(f"{symbol} benchmark validation is incomplete")
        blockers.extend(
            benchmark_blockers
            or [f"{'/'.join(benchmark_symbols)} benchmark validation is incomplete"]
        )
    instrument_feature_dirs = qlib_stats.get("instrument_feature_dirs")
    if not isinstance(instrument_feature_dirs, int) or isinstance(
        instrument_feature_dirs, bool
    ):
        instrument_feature_dirs = max(
            int(qlib_stats.get("feature_dirs", 0))
            - int(qlib_stats.get("benchmark_feature_dirs", 0))
            - int(qlib_stats.get("unexpected_extra_feature_dirs", 0)),
            0,
        )
    missing_instrument_feature_dirs = qlib_stats.get(
        "missing_instrument_feature_dirs"
    )
    if not isinstance(missing_instrument_feature_dirs, int) or isinstance(
        missing_instrument_feature_dirs, bool
    ):
        missing_instrument_feature_dirs = max(
            int(qlib_stats.get("instruments", 0)) - instrument_feature_dirs,
            0,
        )
    if instrument_feature_dirs < min_instruments:
        blockers.append(
            "Qlib provider has fewer than "
            f"{min_instruments} instrument feature directories"
        )
    if missing_instrument_feature_dirs > 0:
        blockers.append(
            "Qlib provider is missing feature directories for "
            f"{missing_instrument_feature_dirs} listed instruments"
        )
    required_file_metrics = {}
    for field in (
        "complete_instrument_feature_dirs",
        "incomplete_instrument_feature_dirs",
        "missing_instrument_feature_files",
        "invalid_instrument_feature_files",
        "misaligned_instrument_feature_dirs",
    ):
        value = qlib_stats.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            required_file_metrics[field] = value
    if len(required_file_metrics) != 5:
        blockers.append(
            "Qlib provider required feature files were not fully validated"
        )
    else:
        complete_feature_dirs = required_file_metrics[
            "complete_instrument_feature_dirs"
        ]
        incomplete_feature_dirs = required_file_metrics[
            "incomplete_instrument_feature_dirs"
        ]
        missing_feature_files = required_file_metrics[
            "missing_instrument_feature_files"
        ]
        invalid_feature_files = required_file_metrics[
            "invalid_instrument_feature_files"
        ]
        misaligned_feature_dirs = required_file_metrics[
            "misaligned_instrument_feature_dirs"
        ]
        if complete_feature_dirs < min_instruments:
            blockers.append(
                "Qlib provider has fewer than "
                f"{min_instruments} instruments with all required feature files"
            )
        if incomplete_feature_dirs > 0 or missing_feature_files > 0:
            blockers.append(
                "Qlib provider required feature files are incomplete for "
                f"{incomplete_feature_dirs} instruments "
                f"({missing_feature_files} missing files)"
            )
        if invalid_feature_files > 0:
            blockers.append(
                "Qlib provider has invalid required feature files "
                f"({invalid_feature_files} invalid files)"
            )
        if misaligned_feature_dirs > 0:
            blockers.append(
                "Qlib provider required feature masks are misaligned for "
                f"{misaligned_feature_dirs} instruments"
            )
    if len(merged_success) < min_merged_instruments:
        blockers.append(f"Merged local dataset has fewer than {min_merged_instruments} instruments")
    for key in ["duplicate_dates", "null_dates", "missing_columns", "bad_ohlc_rows"]:
        if issue_counts.get(key, 0) > 0:
            blockers.append(f"Structural issue remains: {key}={issue_counts[key]}")

    if blockers:
        return False, {
            "status": "partial" if sampled_scan else "limited",
            "summary": "数据仍需修复后再正式训练：" + "; ".join(blockers[:4]),
            "failure_reasons": blockers,
        }

    warnings = []
    if issue_counts.get("non_positive_volume", 0) > 0:
        warnings.append(f"{issue_counts['non_positive_volume']} symbols have non-positive volume rows")
    if issue_counts.get("abs_return_gt_50pct", 0) > 0:
        warnings.append(f"{issue_counts['abs_return_gt_50pct']} symbols have absolute daily returns above 50%")

    summary = "Qlib provider has enough calendar/instrument coverage for the baseline run."
    if warnings:
        summary += " Warnings: " + "; ".join(warnings[:3])

    return True, {
        "status": "trainable",
        "summary": summary,
        "failure_reasons": [],
    }


def build_training_threshold_contract(
    *,
    min_calendar_days: int,
    min_instruments: int,
    min_merged_instruments: int,
) -> dict[str, int | bool]:
    authoritative = (
        min_calendar_days >= AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS
        and min_instruments >= AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS
        and min_merged_instruments >= AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS
    )
    return {
        "min_calendar_days": min_calendar_days,
        "min_instruments": min_instruments,
        "min_merged_instruments": min_merged_instruments,
        "authoritative_min_calendar_days": (
            AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS
        ),
        "authoritative_min_instruments": AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
        "authoritative": authoritative,
    }


def build_provider_quality_record(
    *,
    provider_key: str,
    identity: dict,
    qlib_stats: dict,
    merged_success: set[str],
    issue_counts: dict[str, int],
    min_calendar_days: int,
    min_instruments: int,
    min_merged_instruments: int,
    scan_mode: str,
    verification_status: str,
    verification_block_reason: str,
) -> dict:
    trainable, training_readiness = build_training_readiness(
        qlib_stats=qlib_stats,
        merged_success=merged_success,
        issue_counts=issue_counts,
        min_calendar_days=min_calendar_days,
        min_instruments=min_instruments,
        min_merged_instruments=min_merged_instruments,
        scan_mode=scan_mode,
    )
    if verification_status != "verified":
        trainable = False
        failure_reasons = list(training_readiness.get("failure_reasons") or [])
        failure_reasons.append(verification_block_reason)
        if scan_mode == "full" or training_readiness.get("status") == "trainable":
            training_readiness = {
                "status": "blocked",
                "summary": verification_block_reason,
                "failure_reasons": failure_reasons,
            }
        else:
            training_readiness = {
                **training_readiness,
                "failure_reasons": failure_reasons,
            }

    record = {
        "provider": provider_key,
        "dataset_id": identity.get("dataset_id"),
        "snapshot_id": identity.get("snapshot_id"),
        "manifest_sha256": identity.get("manifest_sha256"),
        "content_sha256": identity.get("content_sha256"),
        "calendar_valid": qlib_stats.get("calendar_valid") is True,
        "invalid_calendar_entries": int(
            qlib_stats.get("invalid_calendar_entries", 0)
        ),
        "invalid_calendar_entry_sample": str(
            qlib_stats.get("invalid_calendar_entry_sample") or ""
        ),
        "instrument_records_valid": qlib_stats.get("instrument_records_valid")
        is True,
        "invalid_instrument_records": int(
            qlib_stats.get("invalid_instrument_records", 0)
        ),
        "invalid_instrument_record_sample": str(
            qlib_stats.get("invalid_instrument_record_sample") or ""
        ),
        "calendar_days": int(qlib_stats.get("calendar_days", 0)),
        "calendar_start": str(qlib_stats.get("calendar_start") or ""),
        "calendar_end": str(qlib_stats.get("calendar_end") or ""),
        "instruments": int(qlib_stats.get("instruments", 0)),
        "feature_dirs": int(qlib_stats.get("feature_dirs", 0)),
        "instrument_feature_dirs": int(
            qlib_stats.get("instrument_feature_dirs", 0)
        ),
        "missing_instrument_feature_dirs": int(
            qlib_stats.get("missing_instrument_feature_dirs", 0)
        ),
        "complete_instrument_feature_dirs": int(
            qlib_stats.get("complete_instrument_feature_dirs", 0)
        ),
        "incomplete_instrument_feature_dirs": int(
            qlib_stats.get("incomplete_instrument_feature_dirs", 0)
        ),
        "missing_instrument_feature_files": int(
            qlib_stats.get("missing_instrument_feature_files", 0)
        ),
        "missing_instrument_feature_file_sample": str(
            qlib_stats.get("missing_instrument_feature_file_sample") or ""
        ),
        "invalid_instrument_feature_files": int(
            qlib_stats.get("invalid_instrument_feature_files", 0)
        ),
        "invalid_instrument_feature_file_sample": str(
            qlib_stats.get("invalid_instrument_feature_file_sample") or ""
        ),
        "required_feature_min_coverage": float(
            qlib_stats.get("required_feature_min_coverage", 0.0)
        ),
        "minimum_valid_instrument_feature_coverage": float(
            qlib_stats.get("minimum_valid_instrument_feature_coverage", 0.0)
        ),
        "misaligned_instrument_feature_dirs": int(
            qlib_stats.get("misaligned_instrument_feature_dirs", 0)
        ),
        "misaligned_instrument_feature_dir_sample": str(
            qlib_stats.get("misaligned_instrument_feature_dir_sample") or ""
        ),
        "benchmark_complete": qlib_stats.get("benchmark_complete") is True,
        "benchmark_missing_days": int(
            qlib_stats.get("benchmark_missing_days", 0)
        ),
        "benchmark_details": qlib_stats.get("benchmark_details") or {},
        "trainable": trainable,
        "failure_reasons": list(
            training_readiness.get("failure_reasons") or []
        ),
        "training_readiness": training_readiness,
    }
    if identity.get("provider_contract") == "direct_parquet_v1":
        for field in (
            "provider_contract",
            "descriptor_path",
            "descriptor_sha256",
            "cleaning_mode",
            "index_dataset_id",
            "index_snapshot_id",
            "index_content_sha256",
            "index_manifest_sha256",
        ):
            record[field] = identity.get(field)
    return record


def write_research_status(
    out_dir: Path,
    source_rows: list[dict],
    qlib_stats: dict,
    merged_success: set[str],
    issue_counts: dict[str, int],
    provider_dir: Path | None = None,
    min_calendar_days: int = 252,
    min_instruments: int = 100,
    min_merged_instruments: int = 100,
    scan_mode: str = "full",
    scan_limit: int = 0,
    scanned_instruments: int | None = None,
    quality_report_id: str | None = None,
    quality_evidence_sha256: str | None = None,
    verification_status: str | None = None,
    data_contract: dict | None = None,
    provider_quality: dict | None = None,
    status_filename: str = "research_status_latest.json",
) -> Path:
    provider_dir = Path(provider_dir or qlib_workflow_provider_uri())
    trainable, training_readiness = build_training_readiness(
        qlib_stats=qlib_stats,
        merged_success=merged_success,
        issue_counts=issue_counts,
        min_calendar_days=min_calendar_days,
        min_instruments=min_instruments,
        min_merged_instruments=min_merged_instruments,
        scan_mode=scan_mode,
    )
    latest_training = load_latest_training_result(out_dir)
    effective_verification_status = verification_status or (
        "verified" if scan_mode == "full" else "unverified"
    )
    if effective_verification_status != "verified":
        trainable = False
        if training_readiness.get("status") == "trainable":
            training_readiness = {
                "status": "blocked",
                "summary": (
                    "Data quality evidence is not bound to the current formal Parquet/direct Provider "
                    "snapshot; rerun without path overrides before training."
                ),
            }
    canonical_quality = (
        provider_quality.get("canonical")
        if isinstance(provider_quality, dict)
        else None
    )
    if isinstance(canonical_quality, dict):
        trainable = canonical_quality.get("trainable") is True
        canonical_readiness = canonical_quality.get("training_readiness")
        if isinstance(canonical_readiness, dict):
            training_readiness = dict(canonical_readiness)

    data_sources = []
    for item in source_rows:
        stats = SourceStats(**item)
        data_sources.append(
            {
                "name": stats.name,
                **data_source_metadata(stats.name),
                "files": stats.files,
                "rows": stats.rows,
                "date_range": f"{stats.first_date} ~ {stats.last_date}",
                "size_mb": round(stats.size_mb, 2),
                "status": source_status(stats.name, stats, merged_success),
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trainable": trainable,
        "training_readiness": training_readiness,
        "quality_scan": {
            "mode": scan_mode,
            "coverage": "full" if scan_mode == "full" else "partial",
            "verification_status": effective_verification_status,
            "requested_sample": scan_limit if scan_mode != "full" else 0,
            "scanned_instruments": scanned_instruments,
        },
        "data_sources": data_sources,
        "qlib_provider": {
            "path": str(provider_dir),
            "calendar_days": qlib_stats["calendar_days"],
            "calendar_range": f"{qlib_stats['calendar_start']} ~ {qlib_stats['calendar_end']}",
            "instruments": qlib_stats["instruments"],
            "feature_dirs": qlib_stats["feature_dirs"],
            "complete_instrument_feature_dirs": qlib_stats.get(
                "complete_instrument_feature_dirs", 0
            ),
            "incomplete_instrument_feature_dirs": qlib_stats.get(
                "incomplete_instrument_feature_dirs", 0
            ),
            "missing_instrument_feature_files": qlib_stats.get(
                "missing_instrument_feature_files", 0
            ),
            "invalid_instrument_feature_files": qlib_stats.get(
                "invalid_instrument_feature_files", 0
            ),
            "benchmark_complete": qlib_stats.get("benchmark_complete", False),
            "benchmark_missing_days": qlib_stats.get("benchmark_missing_days", 0),
            "size_mb": round(qlib_stats["size_mb"], 2),
        },
        "latest_training": latest_training,
        "quality_issues": issue_counts,
    }
    if quality_report_id is not None:
        payload["quality_report_id"] = quality_report_id
    if quality_evidence_sha256 is not None:
        payload["quality_evidence_sha256"] = quality_evidence_sha256
    if data_contract is not None:
        payload["data_contract"] = data_contract
    if provider_quality is not None:
        payload["provider_quality"] = provider_quality

    status_path = out_dir / status_filename
    atomic_write_json(status_path, payload)
    return status_path


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--overlap-sample", type=int, default=0, help="0 means all overlaps")
    parser.add_argument(
        "--merged-sample",
        type=int,
        default=0,
        help="0 means all A-share qfq files (legacy option name)",
    )
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--qlib-dir", type=Path, default=None)
    parser.add_argument(
        "--min-calendar-days",
        type=int,
        default=AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS,
    )
    parser.add_argument(
        "--min-instruments",
        type=int,
        default=AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS,
    )
    parser.add_argument("--fail-on-not-trainable", action="store_true")
    args = parser.parse_args(argv)
    if args.min_calendar_days < 0 or args.min_instruments < 0:
        parser.error("minimum calendar days and instruments must be non-negative")
    scan_mode = "sampled" if args.merged_sample > 0 else "full"
    configured_legacy_dir = args.merged_dir or MERGED_DIR or os.environ.get("MERGED_DIR")
    configured_qlib_dir = args.qlib_dir or QLIB_DIR or os.environ.get("QLIB_DIR")
    has_path_override = configured_legacy_dir is not None or configured_qlib_dir is not None

    ashare_resolution = None
    if configured_legacy_dir is not None:
        args.merged_dir = Path(configured_legacy_dir)
    else:
        ashare_resolution = resolve_dataset(
            "ashare_daily_qfq",
            quant_data_root=QUANT_DATA,
            require_basis="qfq",
        )
        args.merged_dir = ashare_resolution.snapshot_path / "bars"

    qlib_resolution = None
    cleaned_qlib_resolution = None
    direct_canonical_identity = None
    direct_cleaned_identity = None
    direct_index_resolution = None
    if configured_qlib_dir is not None:
        args.qlib_dir = Path(configured_qlib_dir)
    else:
        canonical_descriptor = qlib_workflow_provider_descriptor("canonical")
        cleaned_descriptor = qlib_workflow_provider_descriptor("cleaned")
        direct_canonical_identity = direct_provider_identity(
            canonical_descriptor,
            QUANT_DATA,
        )
        direct_cleaned_identity = direct_provider_identity(
            cleaned_descriptor,
            QUANT_DATA,
        )
        direct_index_resolution = resolve_dataset(
            "cn_index_daily",
            quant_data_root=QUANT_DATA,
            require_basis="derived",
        )
        args.qlib_dir = canonical_descriptor.parent

    bars_source_name = "merged" if configured_legacy_dir else "ashare_daily_qfq"
    data_contract = build_quality_data_contract(
        ashare_resolution=ashare_resolution,
        canonical_qlib_resolution=qlib_resolution,
        cleaned_qlib_resolution=cleaned_qlib_resolution,
        has_path_override=has_path_override,
        canonical_qlib_identity=direct_canonical_identity,
        cleaned_qlib_identity=direct_cleaned_identity,
    )
    snapshot_resolutions = native_snapshot_resolutions(
        ashare_resolution,
        qlib_resolution,
        cleaned_qlib_resolution,
        direct_index_resolution,
    )
    snapshot_state_before: dict[str, dict] = {}
    snapshot_fingerprint_failed = False
    if not has_path_override:
        try:
            snapshot_state_before = capture_native_snapshot_state(
                snapshot_resolutions
            )
        except (DataContractError, OSError):
            snapshot_fingerprint_failed = True

    out_dir = args.out_dir
    if scan_mode != "full":
        out_dir = out_dir / "test_runs" / "data_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    stock_codes = read_stock_universe(STOCK_LIST)
    stock_set = set(stock_codes)
    empty_codes = set(read_json_list(EMPTY_CODES))
    timeout_codes = set(read_json_list(TIMEOUT_CODES))

    source_rows = []
    for name, directory in [
        ("baostock", BAOSTOCK_DIR),
        ("jqdata", JQDATA_DIR),
        ("eastmoney", EASTMONEY_DIR),
        ("akshare", AKSHARE_DIR),
        (bars_source_name, args.merged_dir),
    ]:
        stats = source_stats(name, directory)
        source_rows.append(stats.__dict__)

    source_df = pd.DataFrame(source_rows)
    source_csv = out_dir / f"data_quality_sources_{ts}.csv"
    source_df.to_csv(source_csv, index=False)

    print(f"Scanning {bars_source_name} files...")
    merged_summary, merged_issues = scan_merged_files(args.merged_dir, sample_limit=args.merged_sample)
    summary_csv = out_dir / f"data_quality_ashare_qfq_summary_{ts}.csv"
    issues_csv = out_dir / f"data_quality_issues_{ts}.csv"
    merged_summary.to_csv(summary_csv, index=False)
    merged_issues.to_csv(issues_csv, index=False)

    print("Comparing Baostock cross-source overlaps...")
    overlap = overlap_consistency(args.overlap_sample)
    overlap_csv = out_dir / f"data_quality_overlap_{ts}.csv"
    overlap.to_csv(overlap_csv, index=False)

    qlib_stats_by_provider = {
        "canonical": qlib_provider_stats(args.qlib_dir),
    }
    if cleaned_qlib_resolution is not None or direct_cleaned_identity is not None:
        qlib_stats_by_provider["cleaned"] = qlib_provider_stats(
            cleaned_qlib_resolution.snapshot_path
            if cleaned_qlib_resolution is not None
            else qlib_workflow_provider_descriptor("cleaned").parent
        )
    qlib_stats = qlib_stats_by_provider["canonical"]
    bs_success = load_source_index(BAOSTOCK_DIR)
    merged_success = load_source_index(args.merged_dir)

    if not has_path_override:
        snapshot_state_after_scan: dict[str, dict] = {}
        snapshot_state_after_revalidation: dict[str, dict] = {}
        snapshot_content_revalidation_status = "failed"
        try:
            snapshot_state_after_scan = capture_native_snapshot_state(
                snapshot_resolutions
            )
        except (DataContractError, OSError):
            snapshot_fingerprint_failed = True
        try:
            snapshot_content_revalidation_status = (
                revalidate_native_snapshot_content_for_scan(
                    snapshot_resolutions,
                    scan_mode=scan_mode,
                )
            )
        except (DataContractError, OSError, ValueError):
            snapshot_content_revalidation_status = "failed"
        try:
            snapshot_state_after_revalidation = capture_native_snapshot_state(
                snapshot_resolutions
            )
        except (DataContractError, OSError):
            snapshot_fingerprint_failed = True
        bind_snapshot_state(
            data_contract,
            before=snapshot_state_before,
            after_scan=snapshot_state_after_scan,
            after_revalidation=snapshot_state_after_revalidation,
            fingerprint_failed=snapshot_fingerprint_failed,
            content_revalidation_status=snapshot_content_revalidation_status,
        )

    missing_success = sorted(stock_set - merged_success)
    classified = bs_success | empty_codes | timeout_codes
    unclassified = sorted(stock_set - classified)

    issue_counts = {}
    for col in ["duplicate_dates", "null_dates", "bad_ohlc_rows", "abs_return_gt_50pct", "non_positive_volume"]:
        issue_counts[col] = int((merged_summary[col] > 0).sum()) if col in merged_summary.columns else 0
    missing_cols_count = int((merged_summary["missing_columns"].astype(str) != "").sum()) if "missing_columns" in merged_summary.columns else 0
    issue_counts["missing_columns"] = missing_cols_count

    overlap_metrics = {}
    if not overlap.empty:
        overlap_metrics = {
            "symbols": len(overlap),
            "sources": ", ".join(sorted(overlap["source"].dropna().unique())) if "source" in overlap.columns else "",
            "median_close_rel_diff": float(overlap["close_median_rel_diff"].median()),
            "p95_close_rel_diff": float(overlap["close_median_rel_diff"].quantile(0.95)),
            "median_volume_ratio": float(overlap["volume_ratio_median"].median()),
            "p05_volume_ratio": float(overlap["volume_ratio_median"].quantile(0.05)),
            "p95_volume_ratio": float(overlap["volume_ratio_median"].quantile(0.95)),
        }

    rows = []
    for item in source_rows:
        metadata = data_source_metadata(item["name"])
        rows.append(
            {
                "source": item["name"],
                "role": metadata["role"],
                "files": item["files"],
                "rows": item["rows"],
                "date_range": f"{item['first_date']} ~ {item['last_date']}",
                "size_mb": f"{item['size_mb']:.1f}",
            }
        )

    generated_at = datetime.now().astimezone()
    training_thresholds = build_training_threshold_contract(
        min_calendar_days=args.min_calendar_days,
        min_instruments=args.min_instruments,
        min_merged_instruments=args.min_instruments,
    )
    verification_status = (
        "verified"
        if (
            scan_mode == "full"
            and data_contract["status"] == "verified"
            and training_thresholds["authoritative"] is True
        )
        else "unverified"
    )
    if training_thresholds["authoritative"] is not True:
        verification_block_reason = (
            "Data quality thresholds are below the fixed authoritative "
            "minimums; rerun with at least "
            f"{AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS} calendar days and "
            f"{AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS} instruments."
        )
    elif data_contract["status"] != "verified":
        verification_block_reason = (
            f"{data_contract['reason']} Rerun the full data quality report after "
            "the formal Parquet/direct Provider snapshots are stable."
        )
    else:
        verification_block_reason = (
            "Data quality evidence is not bound to the current formal Parquet/direct Provider "
            "snapshot; rerun without path overrides before training."
        )
    provider_identities = data_contract.get("qlib_providers") or {}
    provider_quality = {
        provider_key: build_provider_quality_record(
            provider_key=provider_key,
            identity=provider_identities.get(provider_key) or {},
            qlib_stats=provider_stats,
            merged_success=merged_success,
            issue_counts=issue_counts,
            min_calendar_days=args.min_calendar_days,
            min_instruments=args.min_instruments,
            min_merged_instruments=args.min_instruments,
            scan_mode=scan_mode,
            verification_status=verification_status,
            verification_block_reason=verification_block_reason,
        )
        for provider_key, provider_stats in qlib_stats_by_provider.items()
    }
    quality_trainable = provider_quality["canonical"]["trainable"] is True
    quality_readiness = provider_quality["canonical"]["training_readiness"]
    canonical_stats = next(
        item for item in source_rows if item["name"] == bars_source_name
    )
    canonical_date_range = (
        f"{canonical_stats['first_date']} ~ {canonical_stats['last_date']}"
        if canonical_stats["first_date"] and canonical_stats["last_date"]
        else ""
    )
    quality_payload = {
        "schema_version": 3,
        "evidence_contract": "immutable_data_quality_evidence_v2",
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "scan_mode": scan_mode,
        "coverage_status": "full" if scan_mode == "full" else "partial",
        "verification_status": verification_status,
        "requested_sample": args.merged_sample if scan_mode == "sampled" else 0,
        "scanned_instruments": int(len(merged_summary)),
        "trainable": quality_trainable,
        "training_readiness": quality_readiness,
        "training_thresholds": training_thresholds,
        "data_contract": data_contract,
        "provider_quality": provider_quality,
        "summary": {
            "stock_universe": len(stock_set),
            "usable_instruments": len(merged_success),
            "total_rows": int(canonical_stats["rows"]),
            "date_range": canonical_date_range,
            "source": bars_source_name,
        },
    }

    ashare_identity = data_contract.get("ashare_daily_qfq") or {}
    qlib_identity = data_contract.get("qlib_provider") or {}
    cleaned_qlib_identity = (
        (data_contract.get("qlib_providers") or {}).get("cleaned") or {}
    )

    sampled_notice = (
        "> **Sampled report cannot authorize training.** This diagnostic output "
        "does not replace the formal full-scan quality publication.\n\n"
        if scan_mode != "full"
        else ""
    )
    report = f"""# A-Share Data Quality Report

Generated: {generated_at.strftime("%Y-%m-%d %H:%M:%S")}

{sampled_notice}## Executive Summary

- Stock universe: {len(stock_set)}
- Canonical qfq usable instruments: {len(merged_success)}
- Baostock empty/no-data instruments: {len(empty_codes)}
- Baostock timeout instruments: {len(timeout_codes)}
- Unclassified instruments: {len(unclassified)}
- Canonical qfq rows: {int(source_df[source_df['name'] == bars_source_name]['rows'].iloc[0]):,}
- Canonical qfq date range: {canonical_date_range}
- Qlib provider: `{args.qlib_dir}`
- Scan mode: {scan_mode}
- Coverage status: {"full" if scan_mode == "full" else "partial"}
- Verification status: {verification_status}
- Training readiness: {quality_readiness["status"]}
- Minimum calendar days used: {training_thresholds["min_calendar_days"]}
- Minimum instruments used: {training_thresholds["min_instruments"]}
- Thresholds authoritative: {training_thresholds["authoritative"]}

## Formal Parquet / Direct Qlib Provider Contract

- Contract status: {data_contract["status"]}
- Contract reason: {data_contract["reason"]}
- A-share dataset ID: {ashare_identity.get("dataset_id", "unverified override")}
- A-share snapshot ID: {ashare_identity.get("snapshot_id", "unverified override")}
- A-share manifest SHA-256: {ashare_identity.get("manifest_sha256", "unverified override")}
- A-share content SHA-256: {ashare_identity.get("content_sha256", "unverified override")}
- Qlib dataset ID: {qlib_identity.get("dataset_id", "unverified override")}
- Qlib snapshot ID: {qlib_identity.get("snapshot_id", "unverified override")}
- Qlib manifest SHA-256: {qlib_identity.get("manifest_sha256", "unverified override")}
- Qlib content SHA-256: {qlib_identity.get("content_sha256", "unverified override")}
- Cleaned Qlib dataset ID: {cleaned_qlib_identity.get("dataset_id", "not available")}
- Cleaned Qlib snapshot ID: {cleaned_qlib_identity.get("snapshot_id", "not available")}
- Cleaned Qlib manifest SHA-256: {cleaned_qlib_identity.get("manifest_sha256", "not available")}
- Cleaned Qlib content SHA-256: {cleaned_qlib_identity.get("content_sha256", "not available")}

## Source Coverage

`ashare_daily_qfq` is the canonical immutable Parquet input. Qlib reads it through
the metadata-only direct Provider; rows marked `audit` are retained for source
comparison only, and `legacy` is a compatibility output.

{markdown_table(rows, ["source", "role", "files", "rows", "date_range", "size_mb"])}

## Qlib Provider

- Calendar days: {qlib_stats["calendar_days"]}
- Calendar range: {qlib_stats["calendar_start"]} ~ {qlib_stats["calendar_end"]}
- Instruments: {qlib_stats["instruments"]}
- Feature directories: {qlib_stats["feature_dirs"]}
- Instrument feature directories: {qlib_stats["instrument_feature_dirs"]}
- Listed instruments missing feature directories: {qlib_stats["missing_instrument_feature_dirs"]}
- Instruments with all six required feature files: {qlib_stats["complete_instrument_feature_dirs"]}
- Instruments with incomplete required feature files: {qlib_stats["incomplete_instrument_feature_dirs"]}
- Missing required instrument feature files: {qlib_stats["missing_instrument_feature_files"]} `{qlib_stats["missing_instrument_feature_file_sample"]}`
- Invalid required instrument feature files: {qlib_stats["invalid_instrument_feature_files"]} `{qlib_stats["invalid_instrument_feature_file_sample"]}`
- Benchmark-only feature directories: {qlib_stats["benchmark_feature_dirs"]}
- Benchmark complete through calendar end: {qlib_stats["benchmark_complete"]}
- Benchmark missing days after first observation: {qlib_stats["benchmark_missing_days"]}
- Unexpected extra feature directories: {qlib_stats["unexpected_extra_feature_dirs"]} `{qlib_stats["unexpected_extra_feature_sample"]}`
- Size: {qlib_stats["size_mb"]:.1f} MB

## Structural Checks

- Symbols with duplicate dates: {issue_counts["duplicate_dates"]}
- Symbols with null dates: {issue_counts["null_dates"]}
- Symbols with missing required columns: {issue_counts["missing_columns"]}
- Symbols with OHLC inconsistencies: {issue_counts["bad_ohlc_rows"]}
- Symbols with non-positive volume rows: {issue_counts["non_positive_volume"]}
- Symbols with absolute daily return above 50%: {issue_counts["abs_return_gt_50pct"]}

## Baostock Cross-source Overlap

- Compared rows: {overlap_metrics.get("symbols", 0)}
- Sources: {overlap_metrics.get("sources", "")}
- Median close relative difference: {overlap_metrics.get("median_close_rel_diff", math.nan):.6f}
- P95 close relative difference: {overlap_metrics.get("p95_close_rel_diff", math.nan):.6f}
- Median volume ratio, Baostock/other source: {overlap_metrics.get("median_volume_ratio", math.nan):.6f}
- P05/P95 volume ratio: {overlap_metrics.get("p05_volume_ratio", math.nan):.6f} / {overlap_metrics.get("p95_volume_ratio", math.nan):.6f}

Interpretation: small close differences are expected because Baostock and JQData
round adjusted prices differently. Volume ratio below 1 for some stocks is also
expected in adjusted-volume space after corporate actions.

## Missing / Skipped Instruments

- Canonical qfq missing success count: {len(missing_success)}
- Missing success sample: `{", ".join(missing_success[:40])}`
- Empty/no-data sample: `{", ".join(sorted(empty_codes)[:40])}`
- Timeout codes: `{", ".join(sorted(timeout_codes))}`

## Artifacts

- Source summary CSV: `{source_csv}`
- A-share qfq per-symbol summary CSV: `{summary_csv}`
- Issues CSV: `{issues_csv}`
- Baostock cross-source overlap CSV: `{overlap_csv}`
"""

    markdown_sha256 = hashlib.sha256(report.encode("utf-8")).hexdigest()
    quality_payload["markdown_sha256"] = markdown_sha256

    markdown_artifact_stem = f"data_quality_report_{ts}_{markdown_sha256[:16]}"
    report_path = out_dir / f"{markdown_artifact_stem}.md"
    quality_payload["markdown_artifact"] = report_path.name
    atomic_write_text(report_path, report)
    evidence_bytes = (
        json.dumps(quality_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    report_id = f"data-quality-{evidence_sha256[:16]}"
    evidence_artifact_stem = f"data_quality_report_{ts}_{evidence_sha256[:16]}"
    latest_stem = (
        "data_quality_report_sampled_latest"
        if scan_mode != "full"
        else "data_quality_report_latest"
    )
    latest_path = out_dir / f"{latest_stem}.md"
    report_json_path = out_dir / f"{evidence_artifact_stem}.json"
    latest_json_path = out_dir / f"{latest_stem}.json"
    atomic_write_json(report_json_path, quality_payload)
    atomic_write_text(latest_path, report)
    atomic_write_json(
        latest_json_path,
        {
            "schema_version": 3,
            "publication_contract": (
                "atomic_json_pointer_to_immutable_quality_evidence_v2"
            ),
            "report_id": report_id,
            "evidence_artifact": report_json_path.name,
            "evidence_sha256": evidence_sha256,
        },
    )
    research_status_path = write_research_status(
        out_dir=out_dir,
        source_rows=source_rows,
        qlib_stats=qlib_stats,
        merged_success=merged_success,
        issue_counts=issue_counts,
        provider_dir=args.qlib_dir,
        min_calendar_days=args.min_calendar_days,
        min_instruments=args.min_instruments,
        min_merged_instruments=args.min_instruments,
        scan_mode=scan_mode,
        scan_limit=args.merged_sample,
        scanned_instruments=int(len(merged_summary)),
        quality_report_id=report_id,
        quality_evidence_sha256=evidence_sha256,
        verification_status=verification_status,
        data_contract=data_contract,
        provider_quality=provider_quality,
        status_filename=(
            "research_status_sampled_latest.json"
            if scan_mode != "full"
            else "research_status_latest.json"
        ),
    )

    print(f"Report: {report_path}")
    print(f"Latest: {latest_path}")
    print(f"Report JSON: {report_json_path}")
    print(f"Latest JSON: {latest_json_path}")
    print(f"Research status: {research_status_path}")
    print(f"Sources: {source_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Issues: {issues_csv}")
    print(f"Overlap: {overlap_csv}")
    return 1 if args.fail_on_not_trainable and not quality_trainable else 0


if __name__ == "__main__":
    raise SystemExit(main())
