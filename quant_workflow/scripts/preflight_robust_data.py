#!/usr/bin/env python3
"""Fail-closed data gate for the robust point-in-time CSI800 workflow."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root, reports_root  # noqa: E402
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)


QUANT_DATA = quant_data_root()
DEFAULT_REFERENCE_DIR = QUANT_DATA / "reference"
DEFAULT_PIT_DIR = DEFAULT_REFERENCE_DIR / "pit_csi800"
DEFAULT_PRICE_DIR = (
    QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
)
DEFAULT_PROVIDER_DIR = qlib_provider_uri("cleaned")
DEFAULT_OUT_DIR = reports_root() / "robust_strategy"
REQUIRED_BENCHMARKS = ("SH000300", "SH000905", "SH000906")
REFERENCE_TABLES = {
    "security_master": (
        "security_master.parquet",
        {"symbol", "list_date", "delist_date", "board"},
    ),
    "industry_intervals": (
        "industry_intervals.parquet",
        {"symbol", "industry_code", "start_date", "end_date"},
    ),
    "st_status": ("st_status.parquet", {"date", "symbol", "is_st"}),
    "suspension_status": (
        "suspension_status.parquet",
        {"date", "symbol", "is_suspended"},
    ),
}
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
MEMBERSHIP_SOURCE_FILENAME = "MEMBERSHIP_SOURCE.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _check(check_id: str, passed: bool, message: str, **details: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "message": message,
        "details": details,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _to_dates(values: Iterable[Any]) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce").dt.normalize()


def _read_calendar(
    provider_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[pd.Timestamp], str | None]:
    path = provider_dir / "calendars" / "day.txt"
    if not path.is_file():
        return [], f"calendar file is missing: {path}"
    days = _to_dates(path.read_text(encoding="utf-8").splitlines()).dropna()
    selected = days[(days >= start) & (days <= end)].drop_duplicates().sort_values()
    if selected.empty:
        return [], f"calendar has no trading days in {start.date()} to {end.date()}"
    return selected.tolist(), None


def _read_membership(pit_dir: Path) -> tuple[pd.DataFrame, str | None]:
    files = sorted((pit_dir / "daily").glob("*.parquet"))
    if not files:
        return pd.DataFrame(
            columns=["date", "symbol", "index_code"]
        ), "no PIT membership parquet files found"
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            return pd.DataFrame(
                columns=["date", "symbol", "index_code"]
            ), f"cannot read {path.name}: {type(exc).__name__}"
        required = {"date", "symbol", "index_code"}
        if not required.issubset(frame.columns):
            return pd.DataFrame(
                columns=["date", "symbol", "index_code"]
            ), f"{path.name} is missing required columns"
        frames.append(frame.loc[:, ["date", "symbol", "index_code"]])
    membership = pd.concat(frames, ignore_index=True)
    membership["date"] = pd.to_datetime(
        membership["date"], errors="coerce"
    ).dt.normalize()
    membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
    membership["index_code"] = (
        membership["index_code"].astype(str).str.strip().str.upper()
    )
    return membership, None


def _membership_source_provenance(
    pit_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    path = pit_dir / MEMBERSHIP_SOURCE_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, {"reason_code": "membership_source_missing"}, {}
    except (OSError, json.JSONDecodeError):
        return False, {"reason_code": "membership_source_invalid"}, {}
    if not isinstance(raw, dict) or raw.get("selected") not in {
        "jqdata",
        "baostock",
    }:
        return False, {"reason_code": "membership_source_invalid"}, {}
    date_range = raw.get("date_range")
    indices = raw.get("indices")
    try:
        source_start = pd.Timestamp(date_range["start_date"]).normalize()
        source_end = pd.Timestamp(date_range["end_date"]).normalize()
    except (KeyError, TypeError, ValueError):
        return False, {"reason_code": "membership_source_invalid"}, {}
    if pd.isna(source_start) or pd.isna(source_end):
        return False, {"reason_code": "membership_source_invalid"}, {}
    expected_indices = {"csi300": "000300.XSHG", "csi500": "000905.XSHG"}
    if indices != expected_indices or source_start > start or source_end < end:
        return (
            False,
            {
                "reason_code": "membership_source_range_incomplete",
                "source_start": str(source_start.date()),
                "source_end": str(source_end.date()),
            },
            {},
        )
    sanitized = {
        key: raw[key]
        for key in ("requested", "selected", "fallback_reason", "date_range", "indices")
        if key in raw
    }
    return True, {"reason_code": None}, sanitized


def _find_symbol_file(directory: Path, symbol: str) -> Path | None:
    normalized = str(symbol).strip().upper()
    names = [normalized, normalized.lower()]
    if len(normalized) >= 6 and normalized[-6:].isdigit():
        names.append(normalized[-6:])
    for name in names:
        candidate = directory / f"{name}.parquet"
        if candidate.is_file():
            return candidate
    return None


def _benchmark_gaps(
    provider_dir: Path, expected_dates: pd.DatetimeIndex
) -> dict[str, list[str]]:
    calendar_path = provider_dir / "calendars" / "day.txt"
    if not calendar_path.is_file():
        return {symbol: ["calendar_missing"] for symbol in REQUIRED_BENCHMARKS}
    full_calendar = (
        _to_dates(calendar_path.read_text(encoding="utf-8").splitlines())
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    calendar_positions = {day: index for index, day in enumerate(full_calendar)}
    gaps: dict[str, list[str]] = {}
    for symbol in REQUIRED_BENCHMARKS:
        feature_dir = provider_dir / "features" / symbol.lower()
        symbol_gaps: list[str] = []
        for field in OHLCV_COLUMNS:
            path = feature_dir / f"{field}.day.bin"
            if not path.is_file():
                symbol_gaps.append(f"{field}:missing_file")
                continue
            try:
                raw = np.fromfile(path, dtype="<f4")
            except OSError:
                symbol_gaps.append(f"{field}:unreadable")
                continue
            if len(raw) < 2 or not math.isfinite(float(raw[0])):
                symbol_gaps.append(f"{field}:invalid_binary")
                continue
            start_index = int(raw[0])
            values = raw[1:]
            for day in expected_dates:
                absolute = calendar_positions.get(day)
                offset = None if absolute is None else absolute - start_index
                value = (
                    float("nan")
                    if offset is None or offset < 0 or offset >= len(values)
                    else float(values[offset])
                )
                valid = math.isfinite(value) and (
                    value >= 0 if field == "volume" else value > 0
                )
                if not valid:
                    symbol_gaps.append(f"{field}:{day.date()}")
                    if len(symbol_gaps) >= 20:
                        break
        if symbol_gaps:
            gaps[symbol] = symbol_gaps[:20]
    return gaps


def _read_price_file(path: Path, symbol: str) -> tuple[pd.DataFrame, str | None]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(), f"{symbol}: unreadable parquet ({type(exc).__name__})"
    required = {"date", *OHLCV_COLUMNS}
    if not required.issubset(frame.columns):
        return pd.DataFrame(), f"{symbol}: missing OHLCV columns"
    frame = frame.loc[:, ["date", *OHLCV_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = symbol
    for column in OHLCV_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, None


def _valid_ohlcv(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    values = frame.loc[:, OHLCV_COLUMNS]
    finite = values.apply(
        lambda column: column.map(
            lambda value: math.isfinite(value) if pd.notna(value) else False
        )
    ).all(axis=1)
    positive_prices = (values.loc[:, ["open", "high", "low", "close"]] > 0).all(axis=1)
    nonnegative_volume = values["volume"] >= 0
    ordered = (values["low"] <= values[["open", "close", "high"]].min(axis=1)) & (
        values["high"] >= values[["open", "close", "low"]].max(axis=1)
    )
    return finite & positive_prices & nonnegative_volume & ordered


def _metadata_check(
    reference_dir: Path, membership: pd.DataFrame
) -> tuple[bool, dict[str, Any]]:
    missing: list[str] = []
    invalid: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, (file_name, columns) in REFERENCE_TABLES.items():
        path = reference_dir / file_name
        if not path.is_file():
            missing.append(file_name)
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            invalid.append(f"{file_name}: {type(exc).__name__}")
            continue
        absent = sorted(columns - set(frame.columns))
        if absent:
            invalid.append(f"{file_name}: missing {', '.join(absent)}")
            continue
        tables[name] = frame

    members = set(membership["symbol"].dropna().astype(str))
    symbol_gaps: dict[str, int] = {}
    for name, frame in tables.items():
        known = set(frame["symbol"].dropna().astype(str).str.upper())
        gap = len(members - known)
        if gap:
            symbol_gaps[name] = gap
    member_day_gaps: dict[str, int] = {}
    if not membership.empty and "security_master" in tables:
        security = tables["security_master"].copy()
        security["symbol"] = security["symbol"].astype(str).str.upper()
        security["list_date"] = pd.to_datetime(
            security["list_date"], errors="coerce"
        ).dt.normalize()
        security["delist_date"] = pd.to_datetime(
            security["delist_date"], errors="coerce"
        ).dt.normalize()
        joined = membership[["date", "symbol"]].merge(
            security[["symbol", "list_date", "delist_date", "board"]],
            on="symbol",
            how="left",
            validate="many_to_one",
        )
        if "suspension_status" in tables:
            suspension = tables["suspension_status"].copy()
            suspension["date"] = pd.to_datetime(
                suspension["date"], errors="coerce"
            ).dt.normalize()
            suspension["symbol"] = suspension["symbol"].astype(str).str.upper()
            joined = joined.merge(
                suspension[["date", "symbol", "is_suspended"]],
                on=["date", "symbol"],
                how="left",
                validate="many_to_one",
            )
        else:
            joined["is_suspended"] = False
        post_delist = joined["delist_date"].notna() & (
            joined["delist_date"] <= joined["date"]
        )
        valid = (
            joined["list_date"].notna()
            & joined["board"].notna()
            & (joined["list_date"] <= joined["date"])
            & (~post_delist | joined["is_suspended"].eq(True))
        )
        gap = int((~valid).sum())
        if gap:
            member_day_gaps["security_master"] = gap
    if not membership.empty and "industry_intervals" in tables:
        industry = tables["industry_intervals"].copy()
        industry["symbol"] = industry["symbol"].astype(str).str.upper()
        industry["start_date"] = pd.to_datetime(
            industry["start_date"], errors="coerce"
        ).dt.normalize()
        industry["end_date"] = pd.to_datetime(
            industry["end_date"], errors="coerce"
        ).dt.normalize()
        covered = pd.Series(False, index=membership.index)
        overlap = False
        for symbol, indices in membership.groupby("symbol").groups.items():
            ranges = industry[industry["symbol"] == symbol].sort_values(
                ["start_date", "end_date"]
            )
            if ranges.empty:
                continue
            interval_index = pd.IntervalIndex.from_arrays(
                ranges["start_date"], ranges["end_date"], closed="both"
            )
            if interval_index.is_overlapping:
                overlap = True
                break
            member_indices = pd.Index(indices)
            covered.loc[member_indices] = (
                interval_index.get_indexer(membership.loc[member_indices, "date"]) >= 0
            )
        gap = int((~covered).sum())
        if gap or overlap:
            member_day_gaps["industry_intervals"] = max(gap, 1 if overlap else 0)
    expected_keys = (
        pd.MultiIndex.from_frame(membership[["date", "symbol"]])
        if not membership.empty
        else pd.MultiIndex.from_arrays([[], []])
    )
    for name in ("st_status", "suspension_status"):
        if name not in tables or membership.empty:
            continue
        daily = tables[name].copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
        daily["symbol"] = daily["symbol"].astype(str).str.upper()
        duplicate_count = int(daily.duplicated(["date", "symbol"]).sum())
        available_keys = pd.MultiIndex.from_frame(
            daily[["date", "symbol"]].drop_duplicates()
        )
        gap = int((~expected_keys.isin(available_keys)).sum()) + duplicate_count
        if gap:
            member_day_gaps[name] = gap
    return not missing and not invalid and not symbol_gaps and not member_day_gaps, {
        "missing_files": missing,
        "invalid_tables": invalid,
        "member_symbol_gaps": symbol_gaps,
        "member_day_gaps": member_day_gaps,
    }


def run_preflight(
    *,
    pit_dir: Path,
    price_dir: Path,
    provider_dir: Path,
    reference_dir: Path,
    start_date: str,
    end_date: str,
    as_of_date: str | None = None,
    expected_members_per_day: int = 800,
    membership_coverage_min: float = 0.99,
    ohlcv_coverage_min: float = 0.98,
) -> dict[str, Any]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    as_of = pd.Timestamp(as_of_date or date.today().isoformat()).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if end > as_of:
        raise ValueError("end_date must not be after as_of_date")
    if expected_members_per_day <= 0:
        raise ValueError("expected_members_per_day must be positive")
    if not 0 < membership_coverage_min <= 1 or not 0 < ohlcv_coverage_min <= 1:
        raise ValueError("coverage thresholds must be in (0, 1]")

    checks: list[dict[str, Any]] = []
    pit_dir = Path(pit_dir)
    reference_dir = Path(reference_dir)
    try:
        with membership_read_lock(pit_dir) as membership_snapshot:
            membership_snapshot.ensure_transaction_clean()
            membership_source = membership_snapshot.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
            )
            membership_snapshot_hash = membership_snapshot.fingerprint()
            with reference_read_lock(reference_dir) as reference_precheck:
                reference_precheck.ensure_transaction_clean()
                reference_precheck.validate_source(
                    start_date=str(start.date()),
                    end_date=str(end.date()),
                    expected_membership_snapshot_hash=membership_snapshot_hash,
                )
            membership, membership_error = _read_membership(pit_dir)
    except MembershipIntegrityError as exc:
        check_id = (
            "membership_provenance"
            if exc.reason_code.startswith("membership_source_")
            else "membership_snapshot_integrity"
        )
        checks.append(
            _check(
                check_id,
                False,
                "membership snapshot is unavailable or unverified",
                reason_code=exc.reason_code,
            )
        )
        return {
            "status": "blocked_data",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "range": {
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "as_of_date": str(as_of.date()),
            },
            "inputs": {
                "pit_dir": str(pit_dir.resolve()),
                "price_dir": str(Path(price_dir).resolve()),
                "provider_dir": str(Path(provider_dir).resolve()),
                "reference_dir": str(Path(reference_dir).resolve()),
            },
            "membership_source": {},
            "membership_snapshot_hash": None,
            "reference_source": {},
            "reference_snapshot_hash": None,
            "summary": {
                "trading_days": 0,
                "membership_coverage": 0.0,
                "ohlcv_coverage": 0.0,
                "failed_checks": [check_id],
            },
            "checks": checks,
        }
    except ReferenceIntegrityError as exc:
        check_id = (
            "reference_transaction_clean"
            if exc.reason_code == "reference_transaction_incomplete"
            else "reference_provenance"
        )
        checks.append(
            _check(
                check_id,
                False,
                "reference snapshot is unavailable or unverified",
                reason_code=exc.reason_code,
            )
        )
        return {
            "status": "blocked_data",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "range": {
                "start_date": str(start.date()),
                "end_date": str(end.date()),
                "as_of_date": str(as_of.date()),
            },
            "inputs": {
                "pit_dir": str(pit_dir.resolve()),
                "price_dir": str(Path(price_dir).resolve()),
                "provider_dir": str(Path(provider_dir).resolve()),
                "reference_dir": str(reference_dir.resolve()),
            },
            "membership_source": membership_source,
            "membership_snapshot_hash": None,
            "reference_source": {},
            "reference_snapshot_hash": None,
            "summary": {
                "trading_days": 0,
                "membership_coverage": 0.0,
                "ohlcv_coverage": 0.0,
                "failed_checks": [check_id],
            },
            "checks": checks,
        }
    checks.append(
        _check(
            "membership_snapshot_integrity",
            True,
            "membership snapshot lock, journal, and files are verified",
        )
    )
    checks.append(
        _check(
            "reference_transaction_clean",
            True,
            "no incomplete reference publication transaction",
        )
    )
    calendar, calendar_error = _read_calendar(Path(provider_dir), start, end)
    checks.append(
        _check(
            "trading_calendar",
            not calendar_error,
            calendar_error or f"{len(calendar)} trading days found",
        )
    )

    checks.append(
        _check(
            "pit_membership_files",
            not membership_error,
            membership_error or "PIT membership files are readable",
        )
    )
    checks.append(
        _check(
            "membership_provenance",
            True,
            "historical membership source and range are verified",
            reason_code=None,
        )
    )

    invalid_membership_dates = (
        int(membership["date"].isna().sum()) if not membership.empty else 0
    )
    duplicates = (
        int(membership.duplicated(["date", "symbol"], keep=False).sum())
        if not membership.empty
        else 0
    )
    checks.append(
        _check(
            "membership_duplicates",
            invalid_membership_dates == 0 and duplicates == 0,
            "membership keys are unique"
            if not duplicates and not invalid_membership_dates
            else "membership contains invalid or duplicate date-symbol keys",
            duplicate_rows=duplicates,
            invalid_date_rows=invalid_membership_dates,
        )
    )

    future_membership = (
        int((membership["date"] > as_of).sum()) if not membership.empty else 0
    )
    checks.append(
        _check(
            "future_dates",
            future_membership == 0,
            "no future membership dates"
            if not future_membership
            else "membership contains future dates",
            future_membership_rows=future_membership,
        )
    )

    expected_dates = pd.DatetimeIndex(calendar)
    selected = membership[
        (membership["date"] >= start) & (membership["date"] <= end)
    ].copy()
    member_counts = (
        selected.groupby("date")["symbol"].nunique()
        if not selected.empty
        else pd.Series(dtype=int)
    )
    valid_membership_days = sum(
        int(member_counts.get(day, 0)) == expected_members_per_day
        for day in expected_dates
    )
    membership_coverage = _ratio(valid_membership_days, len(expected_dates))
    membership_gate = bool(calendar) and membership_coverage >= membership_coverage_min
    checks.append(
        _check(
            "membership_coverage",
            membership_gate,
            f"membership coverage {membership_coverage:.4f}; required {membership_coverage_min:.4f}",
            valid_days=valid_membership_days,
            expected_days=len(expected_dates),
            expected_members_per_day=expected_members_per_day,
        )
    )

    price_frames: list[pd.DataFrame] = []
    price_errors: list[str] = []
    price_duplicate_rows = 0
    future_price_rows = 0
    for symbol in sorted(set(selected["symbol"].dropna().astype(str))):
        path = _find_symbol_file(Path(price_dir), symbol)
        if path is None:
            price_errors.append(f"{symbol}: missing parquet")
            continue
        frame, error = _read_price_file(path, symbol)
        if error:
            price_errors.append(error)
            continue
        price_duplicate_rows += int(frame.duplicated(["date"], keep=False).sum())
        future_price_rows += int((frame["date"] > as_of).sum())
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
        frame["valid_ohlcv"] = _valid_ohlcv(frame)
        price_frames.append(frame.loc[:, ["date", "symbol", "valid_ohlcv"]])

    checks.append(
        _check(
            "price_duplicates",
            price_duplicate_rows == 0,
            "price dates are unique"
            if not price_duplicate_rows
            else "price files contain duplicate dates",
            duplicate_rows=price_duplicate_rows,
        )
    )
    if future_price_rows:
        future_check = next(check for check in checks if check["id"] == "future_dates")
        future_check["passed"] = False
        future_check["message"] = "membership or price data contains future dates"
        future_check["details"]["future_price_rows"] = future_price_rows

    prices = (
        pd.concat(price_frames, ignore_index=True)
        if price_frames
        else pd.DataFrame(columns=["date", "symbol", "valid_ohlcv"])
    )
    prices = prices.drop_duplicates(["date", "symbol"], keep=False)
    expected_membership = selected.drop_duplicates(["date", "symbol"])[
        ["date", "symbol"]
    ]
    joined = expected_membership.merge(prices, on=["date", "symbol"], how="left")
    try:
        with reference_read_lock(reference_dir) as coverage_reference:
            coverage_reference.ensure_transaction_clean()
            coverage_reference.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
                expected_membership_snapshot_hash=membership_snapshot_hash,
            )
            suspension_coverage = pd.read_parquet(
                reference_dir / "suspension_status.parquet",
                columns=["date", "symbol", "is_suspended"],
            )
        suspension_coverage["date"] = pd.to_datetime(
            suspension_coverage["date"], errors="coerce"
        ).dt.normalize()
        suspension_coverage["symbol"] = (
            suspension_coverage["symbol"].astype(str).str.upper()
        )
        joined = joined.merge(
            suspension_coverage,
            on=["date", "symbol"],
            how="left",
            validate="one_to_one",
        )
        covered_rows = joined["valid_ohlcv"].eq(True) | joined[
            "is_suspended"
        ].eq(True)
    except (OSError, ValueError, ReferenceIntegrityError):
        covered_rows = joined["valid_ohlcv"].eq(True)
    valid_ohlcv_rows = int(covered_rows.sum()) if not joined.empty else 0
    expected_member_days = len(expected_dates) * expected_members_per_day
    ohlcv_coverage = _ratio(valid_ohlcv_rows, expected_member_days)
    checks.append(
        _check(
            "ohlcv_coverage",
            ohlcv_coverage >= ohlcv_coverage_min,
            f"OHLCV coverage {ohlcv_coverage:.4f}; required {ohlcv_coverage_min:.4f}",
            valid_member_days=valid_ohlcv_rows,
            expected_member_days=expected_member_days,
            missing_or_invalid_examples=price_errors[:20],
            missing_or_invalid_count=len(price_errors),
        )
    )

    benchmark_gaps = _benchmark_gaps(Path(provider_dir), expected_dates)
    checks.append(
        _check(
            "required_benchmarks",
            not benchmark_gaps,
            "all required benchmarks cover the preflight period"
            if not benchmark_gaps
            else "required benchmark data is missing",
            gaps=benchmark_gaps,
        )
    )

    reference_source: dict[str, Any] = {}
    reference_snapshot_hash: str | None = None
    provenance_error: str | None = None
    metadata_ok = False
    metadata_details: dict[str, Any] = {
        "missing_files": [],
        "invalid_tables": [],
        "member_symbol_gaps": {},
        "member_day_gaps": {},
    }
    try:
        with reference_read_lock(reference_dir) as reference_snapshot:
            reference_snapshot.ensure_transaction_clean()
            reference_source = reference_snapshot.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
                expected_membership_snapshot_hash=membership_snapshot_hash,
            )
            metadata_ok, metadata_details = _metadata_check(reference_dir, selected)
            try:
                reference_snapshot_hash = reference_snapshot.fingerprint()
            except ReferenceIntegrityError as exc:
                if exc.reason_code != "reference_snapshot_invalid":
                    raise
                metadata_ok = False
                metadata_details["reason_code"] = exc.reason_code
    except ReferenceIntegrityError as exc:
        provenance_error = exc.reason_code
        reference_source = {}
        reference_snapshot_hash = None
        metadata_details["reason_code"] = exc.reason_code
        if exc.reason_code == "reference_transaction_incomplete":
            transaction_check = next(
                item for item in checks if item["id"] == "reference_transaction_clean"
            )
            transaction_check["passed"] = False
            transaction_check["message"] = (
                "reference publication transaction is incomplete"
            )
            transaction_check["details"]["reason_code"] = exc.reason_code
    checks.append(
        _check(
            "reference_provenance",
            provenance_error is None,
            "reference source and requested range are verified"
            if provenance_error is None
            else "reference source is missing, invalid, or incomplete",
            reason_code=provenance_error,
        )
    )
    checks.append(
        _check(
            "reference_metadata",
            metadata_ok,
            "listing, board, industry, ST, and suspension metadata are present"
            if metadata_ok
            else "required reference metadata is missing or incomplete",
            **metadata_details,
        )
    )

    stability_reason: str | None = None
    try:
        with membership_read_lock(pit_dir) as final_membership:
            final_membership.ensure_transaction_clean()
            final_membership.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
            )
            final_membership_hash = final_membership.fingerprint()
            with reference_read_lock(reference_dir) as final_reference:
                final_reference.ensure_transaction_clean()
                if final_membership_hash != membership_snapshot_hash:
                    raise MembershipIntegrityError(
                        "membership_snapshot_changed",
                        "membership snapshot changed during preflight",
                    )
                final_reference.validate_source(
                    start_date=str(start.date()),
                    end_date=str(end.date()),
                    expected_membership_snapshot_hash=final_membership_hash,
                )
                final_reference_hash = final_reference.fingerprint()
                if final_reference_hash != reference_snapshot_hash:
                    raise ReferenceIntegrityError(
                        "reference_snapshot_changed",
                        "reference snapshot changed during preflight",
                    )
    except (MembershipIntegrityError, ReferenceIntegrityError) as exc:
        stability_reason = exc.reason_code
    checks.append(
        _check(
            "snapshot_stability",
            stability_reason is None,
            "membership and reference snapshots remained stable through preflight"
            if stability_reason is None
            else "membership or reference snapshot changed during preflight",
            reason_code=stability_reason,
        )
    )

    status = (
        "ready" if checks and all(item["passed"] for item in checks) else "blocked_data"
    )
    return {
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "range": {
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "as_of_date": str(as_of.date()),
        },
        "inputs": {
            "pit_dir": str(Path(pit_dir).resolve()),
            "price_dir": str(Path(price_dir).resolve()),
            "provider_dir": str(Path(provider_dir).resolve()),
            "reference_dir": str(Path(reference_dir).resolve()),
        },
        "membership_source": membership_source,
        "membership_snapshot_hash": membership_snapshot_hash,
        "reference_source": reference_source,
        "reference_snapshot_hash": reference_snapshot_hash,
        "summary": {
            "trading_days": len(expected_dates),
            "membership_coverage": membership_coverage,
            "ohlcv_coverage": ohlcv_coverage,
            "failed_checks": [item["id"] for item in checks if not item["passed"]],
        },
        "checks": checks,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Robust Strategy Data Preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Generated: `{payload['generated_at']}`",
        f"- Range: `{payload['range']['start_date']}` to `{payload['range']['end_date']}`",
        f"- PIT membership coverage: `{payload['summary']['membership_coverage']:.4f}`",
        f"- OHLCV coverage: `{payload['summary']['ohlcv_coverage']:.4f}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| `{check['id']}` | {'PASS' if check['passed'] else 'FAIL'} | {check['message']} |"
        )
    return "\n".join(lines) + "\n"


def write_report(payload: dict[str, Any], out_dir: Path) -> None:
    _atomic_write(
        out_dir / "preflight_latest.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(out_dir / "preflight_latest.md", _markdown(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate all data required by the robust PIT CSI800 strategy."
    )
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--price-dir", type=Path, default=DEFAULT_PRICE_DIR)
    parser.add_argument("--provider-dir", type=Path, default=DEFAULT_PROVIDER_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--start-date", default="2014-01-02")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--expected-members-per-day", type=int, default=800)
    parser.add_argument("--membership-coverage-min", type=float, default=0.99)
    parser.add_argument("--ohlcv-coverage-min", type=float, default=0.98)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    try:
        payload = run_preflight(
            pit_dir=args.pit_dir,
            price_dir=args.price_dir,
            provider_dir=args.provider_dir,
            reference_dir=args.reference_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            as_of_date=args.as_of_date,
            expected_members_per_day=args.expected_members_per_day,
            membership_coverage_min=args.membership_coverage_min,
            ohlcv_coverage_min=args.ohlcv_coverage_min,
        )
    except (OSError, ValueError) as exc:
        payload = {
            "status": "blocked_data",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "range": {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "as_of_date": args.as_of_date,
            },
            "inputs": {},
            "summary": {
                "trading_days": 0,
                "membership_coverage": 0.0,
                "ohlcv_coverage": 0.0,
                "failed_checks": ["invalid_preflight_input"],
            },
            "checks": [_check("invalid_preflight_input", False, str(exc))],
        }
    write_report(payload, args.out_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
