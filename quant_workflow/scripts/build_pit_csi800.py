#!/usr/bin/env python3
"""Build point-in-time CSI300 + CSI500 membership from JQData."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from _symbols import to_qlib_name  # noqa: E402
from download_jqdata import _load_jq_module, authenticate  # noqa: E402


CSI300 = "000300.XSHG"
CSI500 = "000905.XSHG"
INDEX_SIZES = {CSI300: 300, CSI500: 500}
EXPECTED_UNION_SIZE = 800
DEFAULT_START_DATE = "2007-01-15"
STATUS_FILENAME = "build_status_latest.json"
TRANSACTION_JOURNAL_FILENAME = ".build_pit_csi800.transaction.json"
REQUIRED_DAILY_COLUMNS = ("date", "symbol", "index_code")
REQUIRED_INTERVAL_COLUMNS = ("symbol", "index_code", "start_date", "end_date")
NORMALIZED_SYMBOL_PATTERN = re.compile(r"^(?:SH|SZ|BJ)\d{6}$")
BLOCKED_RETURN_CODE = 2
SUMMARY_SAMPLE_LIMIT = 20


class BuildBlocked(Exception):
    """A predictable, sanitized data gate failure."""

    def __init__(
        self,
        reason_code: str,
        *,
        expected_dates: Sequence[date] = (),
        completed_dates: Sequence[date] = (),
        anomalies: Sequence[dict[str, Any]] = (),
        skipped_years: Sequence[int] = (),
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.expected_dates = tuple(expected_dates)
        self.completed_dates = tuple(completed_dates)
        self.anomalies = tuple(anomalies)
        self.skipped_years = tuple(skipped_years)


class MembershipValidationError(Exception):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__("membership_validation_failed")
        self.details = details


class ExistingPartitionError(Exception):
    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__("existing_partition_invalid")
        self.details = details


class JQDataProvider:
    """Small adapter that keeps the builder independent from the SDK in tests."""

    def __init__(self, jq_module: Any) -> None:
        self.jq_module = jq_module

    def get_trading_days(self, start_date: date, end_date: date):
        return self.jq_module.get_trade_days(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

    def get_index_members(self, index_code: str, on_date: date):
        return self.jq_module.get_index_stocks(index_code, date=on_date.isoformat())


def default_output_dir() -> Path:
    return quant_data_root() / "reference" / "pit_csi800"


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise BuildBlocked("invalid_date") from exc


def _normalize_date(value: Any) -> date:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid date value") from exc
    if pd.isna(parsed):
        raise ValueError("invalid date value")
    return parsed.date()


def _normalize_symbol(value: Any) -> str:
    normalized = to_qlib_name(value, strip_any_suffix=True, strip_prefix=True)
    if not NORMALIZED_SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("invalid security code")
    return normalized


def _normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [
        column for column in REQUIRED_DAILY_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError("daily membership columns are missing")
    normalized = frame.loc[:, REQUIRED_DAILY_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="raise"
    ).dt.normalize()
    normalized["symbol"] = normalized["symbol"].map(_normalize_symbol)
    normalized["index_code"] = (
        normalized["index_code"].astype(str).str.strip().str.upper()
    )
    return normalized.sort_values(["date", "index_code", "symbol"]).reset_index(
        drop=True
    )


@contextmanager
def _exclusive_build_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".build_pit_csi800.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_transaction_locks(output_dir: Path, status_path: Path | None):
    lock_directories = {output_dir.resolve()}
    if status_path is not None:
        lock_directories.add(status_path.parent.resolve())
    with ExitStack() as stack:
        for directory in sorted(lock_directories, key=str):
            stack.enter_context(_exclusive_build_lock(directory))
        yield


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _journal_path(output_dir: Path) -> Path:
    return output_dir / TRANSACTION_JOURNAL_FILENAME


def _transaction_owner_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.transaction-owner.json")


def _owner_matches(path: Path, transaction_id: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("transaction_id") == transaction_id


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_transaction_backup(backup: Path, target: Path) -> None:
    rollback_stage = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rollback")
    try:
        try:
            os.link(backup, rollback_stage)
        except OSError:
            shutil.copy2(backup, rollback_stage)
        os.replace(rollback_stage, target)
    finally:
        rollback_stage.unlink(missing_ok=True)


def _rollback_transaction_entries(entries: Sequence[dict[str, Any]]) -> None:
    rollback_errors: list[Exception] = []
    for entry in reversed(entries):
        target = Path(entry["target"])
        backup = Path(entry["backup"])
        try:
            if entry["had_original"]:
                if not backup.is_file():
                    raise FileNotFoundError("transaction backup is missing")
                _restore_transaction_backup(backup, target)
            else:
                target.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - catastrophic filesystem failure.
            rollback_errors.append(exc)
    if rollback_errors:
        raise RuntimeError("transaction rollback failed") from rollback_errors[0]


def _cleanup_transaction_entries(entries: Sequence[dict[str, Any]]) -> None:
    for entry in entries:
        Path(entry["stage"]).unlink(missing_ok=True)
        Path(entry["backup"]).unlink(missing_ok=True)


def _commit_file_transaction(
    parquet_payloads: Mapping[Path, pd.DataFrame],
    json_payloads: Mapping[Path, dict[str, Any]],
    *,
    output_dir: Path,
) -> None:
    """Install data and status files with a recoverable transaction journal."""
    transaction_id = uuid.uuid4().hex
    root = output_dir.resolve()
    prepared_json_payloads = dict(json_payloads)
    owner_by_target: dict[Path, Path] = {}
    for raw_target in set(parquet_payloads) | set(json_payloads):
        target = Path(raw_target)
        if not target.resolve().is_relative_to(root):
            owner = _transaction_owner_path(target)
            owner_by_target[target] = owner
            prepared_json_payloads[owner] = {"transaction_id": transaction_id}
    owner_targets = set(owner_by_target.values())
    targets = sorted(
        set(parquet_payloads) | set(prepared_json_payloads),
        key=lambda target: (0 if target in owner_targets else 1, str(target)),
    )
    entries: list[dict[str, Any]] = []
    journal = _journal_path(output_dir)
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = target.with_name(f".{target.name}.{transaction_id}.stage")
            backup = target.with_name(f".{target.name}.{transaction_id}.backup")
            if target in parquet_payloads:
                parquet_payloads[target].to_parquet(stage, index=False)
            else:
                stage.write_text(
                    json.dumps(
                        prepared_json_payloads[target],
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            had_original = target.is_file()
            if had_original:
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
            entries.append(
                {
                    "target": str(target.resolve()),
                    "stage": str(stage.resolve()),
                    "backup": str(backup.resolve()),
                    "had_original": had_original,
                    "installed_sha256": _file_sha256(stage),
                    "transaction_id": transaction_id,
                    "external_owner_path": str(
                        (
                            target
                            if target in owner_targets
                            else owner_by_target.get(target, "")
                        )
                    ),
                }
            )

        _atomic_write_json(
            {
                "transaction_id": transaction_id,
                "state": "prepared",
                "entries": entries,
            },
            journal,
        )
        for entry in entries:
            os.replace(entry["stage"], entry["target"])
        journal.unlink()
    except Exception:
        _rollback_transaction_entries(entries)
        journal.unlink(missing_ok=True)
        _cleanup_transaction_entries(entries)
        raise
    _cleanup_transaction_entries(entries)


def _commit_parquet_transaction(payloads: Mapping[Path, pd.DataFrame]) -> None:
    if not payloads:
        return
    common_parent = Path(
        os.path.commonpath([str(path.parent.resolve()) for path in payloads])
    )
    _commit_file_transaction(payloads, {}, output_dir=common_parent)


def _recover_transaction(
    output_dir: Path,
    *,
    allowed_external_targets: Sequence[Path] = (),
) -> None:
    journal = _journal_path(output_dir)
    if not journal.is_file():
        return
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        entries = payload["entries"]
        allowed = {path.resolve() for path in allowed_external_targets}
        allowed_owners = {_transaction_owner_path(path).resolve() for path in allowed}
        root = output_dir.resolve()
        rollback_entries: list[dict[str, Any]] = []
        for entry in entries:
            target = Path(entry["target"]).resolve()
            stage = Path(entry["stage"]).resolve()
            backup = Path(entry["backup"]).resolve()
            is_internal = target.is_relative_to(root)
            if not (is_internal or target in allowed or target in allowed_owners):
                raise ValueError("transaction target is outside allowed roots")
            if stage.parent != target.parent or backup.parent != target.parent:
                raise ValueError("transaction artifact path is invalid")
            if is_internal:
                rollback_entries.append(entry)
                continue
            transaction_id = str(
                entry.get("transaction_id") or payload.get("transaction_id") or ""
            )
            owner_text = str(entry.get("external_owner_path") or "")
            owner_path = Path(owner_text).resolve() if owner_text else None
            if owner_path is not None and _owner_matches(owner_path, transaction_id):
                rollback_entries.append(entry)
                continue
            # Compatibility for journals created before ownership sidecars existed.
            installed_sha256 = entry.get("installed_sha256")
            if (
                owner_path is None
                and isinstance(installed_sha256, str)
                and target.is_file()
            ):
                if _file_sha256(target) == installed_sha256:
                    rollback_entries.append(entry)
        _rollback_transaction_entries(rollback_entries)
        journal.unlink()
        _cleanup_transaction_entries(entries)
    except Exception as exc:
        raise BuildBlocked(
            "transaction_recovery_failed",
            anomalies=[
                {
                    "stage": "transaction_recovery",
                    "reason_code": "transaction_recovery_failed",
                }
            ],
        ) from exc


def _reason_code_for_provider_error(exc: Exception) -> str:
    text = str(exc).lower()
    permission_markers = (
        "permission",
        "not authorized",
        "unauthorized",
        "no privilege",
        "权限",
        "无权",
        "认证",
    )
    quota_markers = (
        "quota",
        "query count",
        "rate limit",
        "流量",
        "调用次数",
        "超过限制",
    )
    if isinstance(exc, PermissionError) or any(
        marker in text for marker in permission_markers
    ):
        return "jqdata_permission_denied"
    if any(marker in text for marker in quota_markers):
        return "jqdata_quota_exceeded"
    return "jqdata_fetch_failed"


def _normalize_trading_days(
    values: Sequence[Any], start_date: date, end_date: date
) -> list[date]:
    days: set[date] = set()
    for value in values:
        normalized = _normalize_date(value)
        if start_date <= normalized <= end_date:
            days.add(normalized)
    return sorted(days)


def _membership_details(
    on_date: date,
    csi300_members: set[str],
    csi500_members: set[str],
) -> dict[str, Any]:
    overlap = csi300_members & csi500_members
    union = csi300_members | csi500_members
    return {
        "date": on_date.isoformat(),
        "reason_code": "membership_validation_failed",
        "csi300_count": len(csi300_members),
        "csi500_count": len(csi500_members),
        "overlap_count": len(overlap),
        "union_count": len(union),
    }


def build_daily_membership(
    on_date: date,
    csi300_values: Sequence[Any],
    csi500_values: Sequence[Any],
) -> pd.DataFrame:
    try:
        csi300_members = {_normalize_symbol(value) for value in csi300_values}
        csi500_members = {_normalize_symbol(value) for value in csi500_values}
    except (TypeError, ValueError) as exc:
        raise MembershipValidationError(
            {
                "date": on_date.isoformat(),
                "reason_code": "invalid_symbol",
            }
        ) from exc

    details = _membership_details(on_date, csi300_members, csi500_members)
    if (
        details["csi300_count"] != INDEX_SIZES[CSI300]
        or details["csi500_count"] != INDEX_SIZES[CSI500]
        or details["overlap_count"] != 0
        or details["union_count"] != EXPECTED_UNION_SIZE
    ):
        raise MembershipValidationError(details)

    records = [
        (pd.Timestamp(on_date), symbol, index_code)
        for index_code, members in ((CSI300, csi300_members), (CSI500, csi500_members))
        for symbol in sorted(members)
    ]
    return pd.DataFrame(records, columns=REQUIRED_DAILY_COLUMNS)


def _validate_partition_day(frame: pd.DataFrame, on_date: date) -> bool:
    day_frame = frame.loc[frame["date"] == pd.Timestamp(on_date)]
    if len(day_frame) != EXPECTED_UNION_SIZE:
        return False
    counts = day_frame.groupby("index_code").size().to_dict()
    if counts != INDEX_SIZES:
        return False
    return day_frame["symbol"].nunique() == EXPECTED_UNION_SIZE


def _read_existing_partition(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=REQUIRED_DAILY_COLUMNS)
    return _normalize_daily_frame(pd.read_parquet(path))


def _validate_partition_dates(
    frame: pd.DataFrame,
    partition_year: int,
    authoritative_dates: Sequence[date],
) -> None:
    authoritative = set(authoritative_dates)
    observed = {pd.Timestamp(value).date() for value in frame["date"].unique()}
    invalid = sorted(
        value
        for value in observed
        if value.year != partition_year or value not in authoritative
    )
    if invalid:
        raise ExistingPartitionError(
            {
                "stage": "existing_partition",
                "reason_code": "existing_partition_invalid",
                "year": partition_year,
                "invalid_date_count": len(invalid),
                "sample_dates": [
                    value.isoformat() for value in invalid[:SUMMARY_SAMPLE_LIMIT]
                ],
            }
        )


def _validate_partition(
    frame: pd.DataFrame,
    partition_year: int,
    authoritative_dates: Sequence[date],
) -> None:
    _validate_partition_dates(frame, partition_year, authoritative_dates)
    invalid_membership_dates = sorted(
        pd.Timestamp(timestamp).date()
        for timestamp, group in frame.groupby("date")
        if not _validate_partition_day(group, pd.Timestamp(timestamp).date())
    )
    if invalid_membership_dates:
        raise ExistingPartitionError(
            {
                "stage": "existing_partition",
                "reason_code": "existing_partition_invalid",
                "year": partition_year,
                "invalid_membership_date_count": len(invalid_membership_dates),
                "sample_dates": [
                    value.isoformat()
                    for value in invalid_membership_dates[:SUMMARY_SAMPLE_LIMIT]
                ],
            }
        )


def _replace_dates(
    existing: pd.DataFrame,
    replacements: Sequence[pd.DataFrame],
    replaced_dates: Sequence[date],
) -> pd.DataFrame:
    replaced = {pd.Timestamp(value) for value in replaced_dates}
    retained = (
        existing.loc[~existing["date"].isin(replaced)]
        if not existing.empty
        else existing
    )
    frames = [frame for frame in (retained, *replacements) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True)
    return _normalize_daily_frame(combined)


def compress_membership_intervals(
    daily_membership: pd.DataFrame,
    *,
    authoritative_dates: Sequence[date],
) -> pd.DataFrame:
    """Compress memberships across the authoritative trading-day sequence."""
    frame = _normalize_daily_frame(daily_membership).drop_duplicates()
    if frame.empty:
        return pd.DataFrame(columns=("symbol", "index_code", "start_date", "end_date"))

    trading_days = sorted({pd.Timestamp(value) for value in authoritative_dates})
    positions = {value: offset for offset, value in enumerate(trading_days)}
    observed_dates = {pd.Timestamp(value) for value in frame["date"].unique()}
    if not observed_dates.issubset(positions):
        raise ValueError("daily membership contains a non-authoritative date")
    records: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for (symbol, index_code), group in frame.groupby(
        ["symbol", "index_code"], sort=True
    ):
        dates = sorted(pd.Timestamp(value) for value in group["date"].unique())
        start = previous = dates[0]
        for current in dates[1:]:
            if positions[current] != positions[previous] + 1:
                records.append((symbol, index_code, start, previous))
                start = current
            previous = current
        records.append((symbol, index_code, start, previous))

    return (
        pd.DataFrame(
            records,
            columns=("symbol", "index_code", "start_date", "end_date"),
        )
        .sort_values(["symbol", "index_code", "start_date"])
        .reset_index(drop=True)
    )


def _normalize_intervals_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != REQUIRED_INTERVAL_COLUMNS:
        raise ValueError("interval membership columns are invalid")
    normalized = frame.loc[:, REQUIRED_INTERVAL_COLUMNS].copy()
    normalized["symbol"] = normalized["symbol"].astype("string")
    normalized["index_code"] = normalized["index_code"].astype("string")
    for column in ("start_date", "end_date"):
        normalized[column] = pd.to_datetime(
            normalized[column], errors="raise"
        ).dt.normalize()
    return normalized.sort_values(
        ["symbol", "index_code", "start_date", "end_date"]
    ).reset_index(drop=True)


def _intervals_content_matches(path: Path, expected: pd.DataFrame) -> bool:
    if not path.is_file():
        return False
    try:
        observed = _normalize_intervals_frame(pd.read_parquet(path))
        normalized_expected = _normalize_intervals_frame(expected)
        return observed.equals(normalized_expected)
    except Exception:
        return False


def _combine_daily_partitions(
    partitions: Mapping[int, pd.DataFrame],
    authoritative_dates_by_year: Mapping[int, Sequence[date]],
) -> pd.DataFrame:
    if not partitions:
        raise ValueError("no daily partitions")
    frames: list[pd.DataFrame] = []
    for partition_year, partition in sorted(partitions.items()):
        if partition_year not in authoritative_dates_by_year:
            raise ExistingPartitionError(
                {
                    "stage": "existing_partition",
                    "reason_code": "existing_partition_invalid",
                    "year": partition_year,
                }
            )
        _validate_partition(
            partition,
            partition_year,
            authoritative_dates_by_year[partition_year],
        )
        frames.append(partition)
    frame = _normalize_daily_frame(pd.concat(frames, ignore_index=True))
    return frame


def _build_status(
    status: str,
    start_date: date,
    end_date: date,
    *,
    expected_dates: Sequence[date] = (),
    completed_dates: Sequence[date] = (),
    anomalies: Sequence[dict[str, Any]] = (),
    skipped_years: Sequence[int] = (),
    reason_code: str | None = None,
) -> dict[str, Any]:
    expected = sorted(set(expected_dates))
    completed = sorted(set(completed_dates) & set(expected))
    missing = sorted(set(expected) - set(completed))
    payload: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "expected_date_count": len(expected),
        "completed_date_count": len(completed),
        "missing_summary": {
            "count": len(missing),
            "sample_dates": [
                value.isoformat() for value in missing[:SUMMARY_SAMPLE_LIMIT]
            ],
        },
        "anomaly_summary": {
            "count": len(anomalies),
            "items": list(anomalies[:SUMMARY_SAMPLE_LIMIT]),
        },
        "skipped_years": sorted(set(skipped_years)),
    }
    if reason_code is not None:
        payload["reason_code"] = reason_code
    return payload


def _blocked_from_provider_error(
    exc: Exception,
    *,
    stage: str,
    expected_dates: Sequence[date],
    completed_dates: Sequence[date],
    skipped_years: Sequence[int],
    on_date: date | None = None,
    index_code: str | None = None,
) -> BuildBlocked:
    reason_code = _reason_code_for_provider_error(exc)
    anomaly: dict[str, Any] = {"stage": stage, "reason_code": reason_code}
    if on_date is not None:
        anomaly["date"] = on_date.isoformat()
    if index_code is not None:
        anomaly["index_code"] = index_code
    return BuildBlocked(
        reason_code,
        expected_dates=expected_dates,
        completed_dates=completed_dates,
        anomalies=[anomaly],
        skipped_years=skipped_years,
    )


def _build_pit_csi800_locked(
    provider: Any,
    start_date: date,
    end_date: date,
    output_dir: Path,
    *,
    force: bool = False,
    as_of_date: date | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    calendar_limit = date.today() if as_of_date is None else as_of_date
    try:
        raw_trading_days = provider.get_trading_days(start_date, end_date)
        trading_days = _normalize_trading_days(raw_trading_days, start_date, end_date)
    except Exception as exc:
        raise _blocked_from_provider_error(
            exc,
            stage="trading_calendar",
            expected_dates=(),
            completed_dates=(),
            skipped_years=(),
        ) from None
    daily_dir = output_dir / "daily"
    completed_dates: set[date] = set()
    skipped_years: list[int] = []
    requested_years = set(range(start_date.year, end_date.year + 1))
    partition_paths: dict[int, Path] = {}
    for path in sorted(daily_dir.glob("*.parquet")):
        if not re.fullmatch(r"\d{4}", path.stem):
            raise BuildBlocked(
                "existing_partition_invalid",
                expected_dates=trading_days,
                anomalies=[
                    {
                        "stage": "existing_partition",
                        "reason_code": "existing_partition_invalid",
                        "partition": path.name,
                    }
                ],
            )
        partition_paths[int(path.stem)] = path

    authoritative_dates_by_year: dict[int, list[date]] = {}
    bounded_years = requested_years | set(partition_paths)
    calendar_years = range(min(bounded_years), max(bounded_years) + 1)
    for year in calendar_years:
        year_start = date(year, 1, 1)
        if year_start > calendar_limit:
            authoritative_dates_by_year[year] = []
            continue
        year_end = min(date(year, 12, 31), calendar_limit)
        try:
            raw_year_days = provider.get_trading_days(year_start, year_end)
            year_days = _normalize_trading_days(
                raw_year_days,
                year_start,
                year_end,
            )
            if year >= 2007 and year_end == date(year, 12, 31) and not year_days:
                raise BuildBlocked(
                    "calendar_inconsistent",
                    expected_dates=trading_days,
                    anomalies=[
                        {
                            "stage": "annual_trading_calendar",
                            "reason_code": "calendar_inconsistent",
                            "year": year,
                        }
                    ],
                )
            authoritative_dates_by_year[year] = year_days
        except BuildBlocked:
            raise
        except Exception as exc:
            raise _blocked_from_provider_error(
                exc,
                stage="annual_trading_calendar",
                expected_dates=trading_days,
                completed_dates=completed_dates,
                skipped_years=skipped_years,
            ) from None

    authoritative_request_dates = sorted(
        value
        for year_dates in authoritative_dates_by_year.values()
        for value in year_dates
        if start_date <= value <= end_date
    )
    if trading_days != authoritative_request_dates:
        raise BuildBlocked(
            "calendar_inconsistent",
            expected_dates=authoritative_request_dates,
            anomalies=[
                {
                    "stage": "trading_calendar",
                    "reason_code": "calendar_inconsistent",
                    "range_date_count": len(trading_days),
                    "authoritative_date_count": len(authoritative_request_dates),
                }
            ],
        )
    if not authoritative_request_dates:
        raise BuildBlocked("no_trading_days")
    trading_days = authoritative_request_dates

    existing_partitions: dict[int, pd.DataFrame] = {}
    for year, partition_path in sorted(partition_paths.items()):
        try:
            existing = _read_existing_partition(partition_path)
            _validate_partition(
                existing,
                year,
                authoritative_dates_by_year[year],
            )
        except ExistingPartitionError as exc:
            raise BuildBlocked(
                "existing_partition_invalid",
                expected_dates=trading_days,
                completed_dates=completed_dates,
                anomalies=[exc.details],
                skipped_years=skipped_years,
            ) from None
        except Exception:
            raise BuildBlocked(
                "existing_partition_invalid",
                expected_dates=trading_days,
                completed_dates=completed_dates,
                anomalies=[
                    {
                        "stage": "existing_partition",
                        "reason_code": "existing_partition_invalid",
                        "year": year,
                    }
                ],
                skipped_years=skipped_years,
            ) from None
        existing_partitions[year] = existing

    changed_partitions: dict[Path, pd.DataFrame] = {}
    years = sorted(requested_years)
    for year in years:
        year_days = [value for value in trading_days if value.year == year]
        partition_path = daily_dir / f"{year}.parquet"
        existing = existing_partitions.get(
            year,
            pd.DataFrame(columns=REQUIRED_DAILY_COLUMNS),
        )
        valid_existing = {
            value for value in year_days if _validate_partition_day(existing, value)
        }
        completed_dates.update(valid_existing)
        days_to_fetch = (
            list(year_days)
            if force
            else [value for value in year_days if value not in valid_existing]
        )
        if not days_to_fetch:
            skipped_years.append(year)
            continue

        replacements: list[pd.DataFrame] = []
        for on_date in days_to_fetch:
            fetched: dict[str, Sequence[Any]] = {}
            for index_code in (CSI300, CSI500):
                try:
                    fetched[index_code] = provider.get_index_members(
                        index_code, on_date
                    )
                except Exception as exc:
                    raise _blocked_from_provider_error(
                        exc,
                        stage="index_membership",
                        expected_dates=trading_days,
                        completed_dates=completed_dates,
                        skipped_years=skipped_years,
                        on_date=on_date,
                        index_code=index_code,
                    ) from None
            try:
                replacements.append(
                    build_daily_membership(
                        on_date,
                        fetched[CSI300],
                        fetched[CSI500],
                    )
                )
            except MembershipValidationError as exc:
                raise BuildBlocked(
                    "membership_validation_failed",
                    expected_dates=trading_days,
                    completed_dates=completed_dates,
                    anomalies=[exc.details],
                    skipped_years=skipped_years,
                ) from None

        updated = _replace_dates(existing, replacements, days_to_fetch)
        try:
            _validate_partition(
                updated,
                year,
                authoritative_dates_by_year[year],
            )
        except ExistingPartitionError as exc:
            raise BuildBlocked(
                "existing_partition_invalid",
                expected_dates=trading_days,
                completed_dates=completed_dates,
                anomalies=[exc.details],
                skipped_years=skipped_years,
            ) from None
        existing_partitions[year] = updated
        changed_partitions[partition_path] = updated

    intervals_path = output_dir / "membership_intervals.parquet"
    completed_dates.update(trading_days)
    status = _build_status(
        "completed",
        start_date,
        end_date,
        expected_dates=trading_days,
        completed_dates=completed_dates,
        skipped_years=skipped_years,
    )
    status_payloads = {} if status_path is None else {status_path: status}

    try:
        all_daily = _combine_daily_partitions(
            existing_partitions,
            authoritative_dates_by_year,
        )
        authoritative_dates = sorted(
            {
                value
                for year_dates in authoritative_dates_by_year.values()
                for value in year_dates
            }
        )
        intervals = compress_membership_intervals(
            all_daily,
            authoritative_dates=authoritative_dates,
        )
    except ExistingPartitionError as exc:
        raise BuildBlocked(
            "existing_partition_invalid",
            expected_dates=trading_days,
            completed_dates=completed_dates,
            anomalies=[exc.details],
            skipped_years=skipped_years,
        ) from None
    except Exception:
        raise BuildBlocked(
            "existing_partition_invalid",
            expected_dates=trading_days,
            completed_dates=completed_dates,
            anomalies=[
                {
                    "stage": "interval_compression",
                    "reason_code": "existing_partition_invalid",
                }
            ],
            skipped_years=skipped_years,
        ) from None

    if not changed_partitions and _intervals_content_matches(intervals_path, intervals):
        if status_payloads:
            _commit_file_transaction(
                {},
                status_payloads,
                output_dir=output_dir,
            )
        return status

    transaction_payloads = dict(changed_partitions)
    transaction_payloads[intervals_path] = intervals
    _commit_file_transaction(
        transaction_payloads,
        status_payloads,
        output_dir=output_dir,
    )
    return status


def _safe_write_status(status: dict[str, Any], status_path: Path) -> bool:
    try:
        _commit_file_transaction(
            {},
            {
                _transaction_owner_path(status_path): {
                    "transaction_id": f"status-{uuid.uuid4().hex}"
                },
                status_path: status,
            },
            output_dir=status_path.parent,
        )
        return True
    except Exception:
        print("ERROR: status_write_failed", file=sys.stderr)
        return False


def build_pit_csi800(
    provider: Any,
    start_date: date,
    end_date: date,
    output_dir: Path,
    *,
    force: bool = False,
    as_of_date: date | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    with _exclusive_transaction_locks(output_dir, status_path):
        if (
            status_path is not None
            and status_path.parent.resolve() != output_dir.resolve()
        ):
            _recover_transaction(status_path.parent)
        _recover_transaction(
            output_dir,
            allowed_external_targets=() if status_path is None else (status_path,),
        )
        try:
            return _build_pit_csi800_locked(
                provider,
                start_date,
                end_date,
                output_dir,
                force=force,
                as_of_date=as_of_date,
                status_path=status_path,
            )
        except BuildBlocked as exc:
            if status_path is not None:
                _safe_write_status(
                    _build_status(
                        "blocked_data",
                        start_date,
                        end_date,
                        expected_dates=exc.expected_dates,
                        completed_dates=exc.completed_dates,
                        anomalies=exc.anomalies,
                        skipped_years=exc.skipped_years,
                        reason_code=exc.reason_code,
                    ),
                    status_path,
                )
            raise
        except Exception:
            if status_path is not None:
                _safe_write_status(
                    _build_status(
                        "blocked_data",
                        start_date,
                        end_date,
                        reason_code="build_failed",
                        anomalies=[{"stage": "build", "reason_code": "build_failed"}],
                    ),
                    status_path,
                )
            raise


def _create_default_provider(
    jq_module: Any | None,
    environ: MutableMapping[str, str] | None,
) -> JQDataProvider:
    env = os.environ if environ is None else environ
    try:
        jq = _load_jq_module() if jq_module is None else jq_module
    except Exception as exc:
        raise BuildBlocked("jqdata_sdk_unavailable") from exc
    try:
        authenticated = authenticate(jq, env)
    except Exception as exc:
        raise BuildBlocked(_reason_code_for_provider_error(exc)) from None
    if not authenticated:
        raise BuildBlocked("jqdata_auth_failed")
    return JQDataProvider(jq)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build point-in-time CSI800 historical membership."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--status-out-dir",
        type=Path,
        default=None,
        help="Directory for build_status_latest.json; defaults to --output-dir.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _publish_failure_status(
    output_dir: Path,
    status_path: Path,
    status: dict[str, Any],
) -> bool:
    try:
        with _exclusive_transaction_locks(output_dir, status_path):
            _recover_transaction(
                output_dir,
                allowed_external_targets=(status_path,),
            )
            return _safe_write_status(status, status_path)
    except BuildBlocked:
        print("ERROR: transaction_recovery_failed", file=sys.stderr)
        return False


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: Any | None = None,
    jq_module: Any | None = None,
    environ: MutableMapping[str, str] | None = None,
    today: date | None = None,
) -> int:
    args = _argument_parser().parse_args(argv)
    current_date = date.today() if today is None else today
    output_dir = (
        args.output_dir if args.output_dir is not None else default_output_dir()
    )
    status_dir = args.status_out_dir if args.status_out_dir is not None else output_dir
    status_path = status_dir / STATUS_FILENAME
    raw_end_date = args.end_date or current_date.isoformat()

    try:
        start_date = _parse_date(args.start_date)
        end_date = _parse_date(raw_end_date)
    except BuildBlocked as exc:
        fallback_start = current_date
        fallback_end = current_date
        status = _build_status(
            "blocked_data",
            fallback_start,
            fallback_end,
            reason_code=exc.reason_code,
        )
        _publish_failure_status(output_dir, status_path, status)
        print(f"ERROR: blocked_data ({exc.reason_code})", file=sys.stderr)
        return BLOCKED_RETURN_CODE

    if start_date > end_date:
        exc = BuildBlocked("invalid_date_range")
        status = _build_status(
            "blocked_data",
            start_date,
            end_date,
            reason_code=exc.reason_code,
        )
        _publish_failure_status(output_dir, status_path, status)
        print(f"ERROR: blocked_data ({exc.reason_code})", file=sys.stderr)
        return BLOCKED_RETURN_CODE
    if start_date > current_date or end_date > current_date:
        exc = BuildBlocked("future_date_not_allowed")
        status = _build_status(
            "blocked_data",
            start_date,
            end_date,
            reason_code=exc.reason_code,
        )
        _publish_failure_status(output_dir, status_path, status)
        print(f"ERROR: blocked_data ({exc.reason_code})", file=sys.stderr)
        return BLOCKED_RETURN_CODE

    try:
        active_provider = (
            provider
            if provider is not None
            else _create_default_provider(jq_module, environ)
        )
    except BuildBlocked as exc:
        status = _build_status(
            "blocked_data",
            start_date,
            end_date,
            reason_code=exc.reason_code,
        )
        _publish_failure_status(output_dir, status_path, status)
        print(f"ERROR: blocked_data ({exc.reason_code})", file=sys.stderr)
        return BLOCKED_RETURN_CODE

    try:
        status = build_pit_csi800(
            active_provider,
            start_date,
            end_date,
            output_dir,
            force=args.force,
            as_of_date=current_date,
            status_path=status_path,
        )
    except BuildBlocked as exc:
        print(f"ERROR: blocked_data ({exc.reason_code})", file=sys.stderr)
        return BLOCKED_RETURN_CODE
    except Exception:
        print("ERROR: blocked_data (build_failed)", file=sys.stderr)
        return BLOCKED_RETURN_CODE

    print(
        f"PIT CSI800 build completed: {status['completed_date_count']} trading dates; "
        f"data={output_dir}; status={status_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
