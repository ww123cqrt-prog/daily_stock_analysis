#!/usr/bin/env python3
"""Build point-in-time reference metadata for the robust CSI800 strategy."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from download_jqdata import _load_jq_module, authenticate  # noqa: E402


DEFAULT_START_DATE = "2014-01-02"
STATUS_FILENAME = "download_robust_reference_latest.json"
BLOCKED_RETURN_CODE = 2
SYMBOL_CHUNK_SIZE = 200
TABLE_COLUMNS = {
    "security_master.parquet": ("symbol", "list_date", "delist_date", "board"),
    "industry_intervals.parquet": ("symbol", "industry_code", "start_date", "end_date"),
    "st_status.parquet": ("date", "symbol", "is_st"),
    "suspension_status.parquet": ("date", "symbol", "is_suspended"),
}
QLIB_SYMBOL_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")
JQ_SYMBOL_PATTERN = re.compile(r"^(\d{6})\.(XSHG|XSHE|XBSE)$")
EXCHANGE_TO_JQ = {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBSE"}
JQ_TO_EXCHANGE = {value: key for key, value in EXCHANGE_TO_JQ.items()}


class BlockedData(Exception):
    """A fail-closed data or provider error identified only by a safe code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def default_pit_dir() -> Path:
    return quant_data_root() / "reference" / "pit_csi800"


def default_reference_dir() -> Path:
    return quant_data_root() / "reference"


def qlib_to_jq_symbol(value: Any) -> str:
    matched = QLIB_SYMBOL_PATTERN.fullmatch(str(value).strip().upper())
    if matched is None:
        raise BlockedData("invalid_pit_symbol")
    exchange, digits = matched.groups()
    return f"{digits}.{EXCHANGE_TO_JQ[exchange]}"


def jq_to_qlib_symbol(value: Any) -> str:
    matched = JQ_SYMBOL_PATTERN.fullmatch(str(value).strip().upper())
    if matched is None:
        raise BlockedData("invalid_provider_symbol")
    digits, exchange = matched.groups()
    return f"{JQ_TO_EXCHANGE[exchange]}{digits}"


def board_from_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    if symbol.startswith(("SH60", "SZ00")):
        return "main"
    if symbol.startswith("SZ30"):
        return "growth"
    if symbol.startswith("SH68"):
        return "star"
    return "other"


def _parse_date(value: str, reason_code: str = "invalid_date") -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise BlockedData(reason_code) from exc


def _normalize_timestamp(value: Any, *, allow_none: bool = False) -> pd.Timestamp:
    if value is None or pd.isna(value):
        if allow_none:
            return pd.NaT
        raise BlockedData("invalid_provider_date")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise BlockedData("invalid_provider_date") from exc
    if pd.isna(parsed) or parsed.tzinfo is not None:
        raise BlockedData("invalid_provider_date")
    return parsed.normalize()


def _read_pit_membership(
    pit_dir: Path,
    start_date: date,
    end_date: date,
    today: date,
) -> tuple[pd.DataFrame, int]:
    files = sorted((pit_dir / "daily").glob("*.parquet"))
    if not files:
        raise BlockedData("missing_pit_data")
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise BlockedData("invalid_pit_data") from exc
        if not {"date", "symbol"}.issubset(frame.columns):
            raise BlockedData("invalid_pit_schema")
        frames.append(frame.loc[:, ["date", "symbol"]].copy())
    membership = pd.concat(frames, ignore_index=True)
    try:
        membership["date"] = pd.to_datetime(
            membership["date"], errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise BlockedData("invalid_pit_date") from exc
    if (
        membership["date"].isna().any()
        or getattr(membership["date"].dt, "tz", None) is not None
    ):
        raise BlockedData("invalid_pit_date")
    membership["symbol"] = membership["symbol"].astype(str).str.strip().str.upper()
    for symbol in membership["symbol"].unique():
        qlib_to_jq_symbol(symbol)
    if membership.duplicated(["date", "symbol"]).any():
        raise BlockedData("duplicate_pit_member_day")
    if (membership["date"].dt.date > today).any():
        raise BlockedData("future_pit_date")
    selected = membership.loc[
        membership["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    if selected.empty:
        raise BlockedData("missing_pit_data")
    return selected.sort_values(["date", "symbol"]).reset_index(drop=True), len(files)


def _provider_reason(exc: Exception) -> str:
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


def _call_provider(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except BlockedData:
        raise
    except Exception as exc:
        raise BlockedData(_provider_reason(exc)) from exc


def _info_value(info: Any, names: Sequence[str]) -> Any:
    if isinstance(info, Mapping):
        for name in names:
            if name in info:
                return info[name]
    else:
        for name in names:
            if hasattr(info, name):
                return getattr(info, name)
    return None


def _build_security_master(
    provider: Any, membership: pd.DataFrame, today: date
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    member_bounds = membership.groupby("symbol")["date"].agg(["min", "max"])
    for symbol in sorted(member_bounds.index):
        info = _call_provider(provider.get_security_info, qlib_to_jq_symbol(symbol))
        if info is None:
            raise BlockedData("missing_security_master_member")
        list_raw = _info_value(info, ("start_date", "list_date"))
        if list_raw is None:
            raise BlockedData("missing_security_master_member")
        list_date = _normalize_timestamp(list_raw)
        delist_raw = _info_value(info, ("end_date", "delist_date"))
        delist_date = _normalize_timestamp(delist_raw, allow_none=True)
        if pd.notna(delist_date) and delist_date.date() > today:
            if delist_date.year >= 2200:
                delist_date = pd.NaT
            else:
                raise BlockedData("future_security_master_date")
        if list_date > member_bounds.loc[symbol, "min"]:
            raise BlockedData("invalid_security_master_member_day")
        if pd.notna(delist_date) and delist_date < member_bounds.loc[symbol, "max"]:
            raise BlockedData("invalid_security_master_member_day")
        rows.append(
            {
                "symbol": symbol,
                "list_date": list_date,
                "delist_date": delist_date,
                "board": board_from_symbol(symbol),
            }
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS["security_master.parquet"])


def _industry_code(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    sw_l1 = entry.get("sw_l1")
    if not isinstance(sw_l1, Mapping):
        return None
    value = sw_l1.get("industry_code")
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def _fetch_daily_industries(provider: Any, membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timestamp, day_members in membership.groupby("date", sort=True):
        symbols = sorted(day_members["symbol"].tolist())
        jq_codes = [qlib_to_jq_symbol(symbol) for symbol in symbols]
        raw = _call_provider(
            provider.get_industry, jq_codes, date=timestamp.date().isoformat()
        )
        if not isinstance(raw, Mapping):
            raise BlockedData("invalid_industry_schema")
        normalized: dict[str, Any] = {}
        for raw_symbol, entry in raw.items():
            symbol = jq_to_qlib_symbol(raw_symbol)
            if symbol in normalized:
                raise BlockedData("duplicate_industry_member_day")
            normalized[symbol] = entry
        if set(normalized) - set(symbols):
            raise BlockedData("invalid_industry_schema")
        for symbol in symbols:
            industry_code = _industry_code(normalized.get(symbol))
            if industry_code is None:
                raise BlockedData("missing_industry_member_day")
            rows.append(
                {"date": timestamp, "symbol": symbol, "industry_code": industry_code}
            )
    return pd.DataFrame(rows, columns=["date", "symbol", "industry_code"])


def _compress_industry_intervals(
    daily: pd.DataFrame, trading_dates: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    ordinal = {
        timestamp: offset for offset, timestamp in enumerate(sorted(trading_dates))
    }
    intervals: list[dict[str, Any]] = []
    for symbol, group in daily.groupby("symbol", sort=True):
        ordered = group.sort_values("date").reset_index(drop=True)
        start = ordered.loc[0, "date"]
        previous_date = start
        industry_code = ordered.loc[0, "industry_code"]
        for row in ordered.iloc[1:].itertuples(index=False):
            continuous = ordinal[row.date] == ordinal[previous_date] + 1
            if row.industry_code != industry_code or not continuous:
                intervals.append(
                    {
                        "symbol": symbol,
                        "industry_code": industry_code,
                        "start_date": start,
                        "end_date": previous_date,
                    }
                )
                start = row.date
                industry_code = row.industry_code
            previous_date = row.date
        intervals.append(
            {
                "symbol": symbol,
                "industry_code": industry_code,
                "start_date": start,
                "end_date": previous_date,
            }
        )
    return pd.DataFrame(intervals, columns=TABLE_COLUMNS["industry_intervals.parquet"])


def _coerce_bool(value: Any, reason_code: str) -> bool:
    if value is None or pd.isna(value):
        raise BlockedData(reason_code)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise BlockedData(reason_code)


def _chunks(values: Sequence[str], size: int = SYMBOL_CHUNK_SIZE):
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def _year_groups(membership: pd.DataFrame):
    for _, frame in membership.groupby(membership["date"].dt.year, sort=True):
        yield frame, frame["date"].min().date(), frame["date"].max().date()


def _normalize_st_matrix(
    raw: Any, requested_symbols: Sequence[str]
) -> dict[tuple[pd.Timestamp, str], bool]:
    if (
        not isinstance(raw, pd.DataFrame)
        or raw.index.has_duplicates
        or raw.columns.has_duplicates
    ):
        raise BlockedData("invalid_st_schema")
    try:
        dates = pd.to_datetime(raw.index, errors="raise").normalize()
    except (TypeError, ValueError) as exc:
        raise BlockedData("invalid_st_schema") from exc
    if dates.has_duplicates or dates.isna().any() or dates.tz is not None:
        raise BlockedData("invalid_st_schema")
    symbols = [jq_to_qlib_symbol(column) for column in raw.columns]
    if len(symbols) != len(set(symbols)) or set(symbols) != set(requested_symbols):
        raise BlockedData("invalid_st_schema")
    lookup: dict[tuple[pd.Timestamp, str], bool] = {}
    for column, symbol in zip(raw.columns, symbols):
        for timestamp, value in zip(dates, raw[column].tolist()):
            if value is None or pd.isna(value):
                continue
            lookup[(timestamp, symbol)] = _coerce_bool(value, "invalid_st_schema")
    return lookup


def _fetch_st_status(provider: Any, membership: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[pd.Timestamp, str], bool] = {}
    for year_members, start_date, end_date in _year_groups(membership):
        symbols = sorted(year_members["symbol"].unique())
        for symbol_chunk in _chunks(symbols):
            jq_codes = [qlib_to_jq_symbol(symbol) for symbol in symbol_chunk]
            raw = _call_provider(
                provider.get_extras,
                "is_st",
                jq_codes,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                df=True,
            )
            chunk_lookup = _normalize_st_matrix(raw, symbol_chunk)
            if set(lookup).intersection(chunk_lookup):
                raise BlockedData("duplicate_st_member_day")
            lookup.update(chunk_lookup)
    rows = []
    for row in membership.itertuples(index=False):
        key = (row.date, row.symbol)
        if key not in lookup:
            raise BlockedData("missing_st_member_day")
        rows.append({"date": row.date, "symbol": row.symbol, "is_st": lookup[key]})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS["st_status.parquet"])


def _normalize_paused_frame(
    raw: Any, requested_symbols: Sequence[str]
) -> dict[tuple[pd.Timestamp, str], bool]:
    if not isinstance(raw, pd.DataFrame):
        raise BlockedData("invalid_paused_schema")
    frame = raw.reset_index()
    date_column = next(
        (
            name
            for name in ("time", "date", "datetime", "index")
            if name in frame.columns
        ),
        None,
    )
    code_column = next(
        (name for name in ("code", "security", "symbol") if name in frame.columns), None
    )
    if date_column is None or "paused" not in frame.columns:
        raise BlockedData("invalid_paused_schema")
    if code_column is None:
        if len(requested_symbols) != 1:
            raise BlockedData("invalid_paused_schema")
        frame["__code"] = qlib_to_jq_symbol(requested_symbols[0])
        code_column = "__code"
    try:
        frame["__date"] = pd.to_datetime(
            frame[date_column], errors="raise"
        ).dt.normalize()
    except (TypeError, ValueError) as exc:
        raise BlockedData("invalid_paused_schema") from exc
    if (
        frame["__date"].isna().any()
        or getattr(frame["__date"].dt, "tz", None) is not None
    ):
        raise BlockedData("invalid_paused_schema")
    frame["__symbol"] = frame[code_column].map(jq_to_qlib_symbol)
    if set(frame["__symbol"]) - set(requested_symbols):
        raise BlockedData("invalid_paused_schema")
    if frame.duplicated(["__date", "__symbol"]).any():
        raise BlockedData("duplicate_paused_member_day")
    lookup: dict[tuple[pd.Timestamp, str], bool] = {}
    for _, row in frame.iterrows():
        lookup[(row["__date"], row["__symbol"])] = _coerce_bool(
            row["paused"], "invalid_paused_schema"
        )
    return lookup


def _fetch_suspension_status(provider: Any, membership: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[pd.Timestamp, str], bool] = {}
    for year_members, start_date, end_date in _year_groups(membership):
        symbols = sorted(year_members["symbol"].unique())
        for symbol_chunk in _chunks(symbols):
            jq_codes = [qlib_to_jq_symbol(symbol) for symbol in symbol_chunk]
            raw = _call_provider(
                provider.get_price,
                jq_codes,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                frequency="daily",
                fields=["paused"],
                skip_paused=False,
                fq=None,
                panel=False,
            )
            chunk_lookup = _normalize_paused_frame(raw, symbol_chunk)
            if set(lookup).intersection(chunk_lookup):
                raise BlockedData("duplicate_paused_member_day")
            lookup.update(chunk_lookup)
    rows = []
    for row in membership.itertuples(index=False):
        key = (row.date, row.symbol)
        if key not in lookup:
            raise BlockedData("missing_paused_member_day")
        rows.append(
            {"date": row.date, "symbol": row.symbol, "is_suspended": lookup[key]}
        )
    return pd.DataFrame(rows, columns=TABLE_COLUMNS["suspension_status.parquet"])


def _validate_tables(
    tables: Mapping[str, pd.DataFrame], membership: pd.DataFrame, today: date
) -> None:
    expected_keys = set(
        membership[["date", "symbol"]].itertuples(index=False, name=None)
    )
    for name, columns in TABLE_COLUMNS.items():
        frame = tables.get(name)
        if frame is None or tuple(frame.columns) != columns:
            raise BlockedData("invalid_reference_schema")
        if frame.empty or frame.isna().all(axis=1).any():
            raise BlockedData("incomplete_reference_data")
    security = tables["security_master.parquet"]
    if security.duplicated("symbol").any() or set(security["symbol"]) != set(
        membership["symbol"]
    ):
        raise BlockedData("invalid_security_master_coverage")
    industry = tables["industry_intervals.parquet"]
    if industry.duplicated(["symbol", "start_date", "end_date"]).any():
        raise BlockedData("duplicate_industry_interval")
    industry = industry.copy()
    for column in ("start_date", "end_date"):
        industry[column] = pd.to_datetime(
            industry[column], errors="coerce"
        ).dt.normalize()
    if industry[["start_date", "end_date"]].isna().any(axis=None):
        raise BlockedData("invalid_industry_interval")
    if (industry["start_date"] > industry["end_date"]).any():
        raise BlockedData("invalid_industry_interval")
    if (industry["start_date"].dt.date > today).any() or (
        industry["end_date"].dt.date > today
    ).any():
        raise BlockedData("future_industry_interval")
    if set(industry["symbol"]) != set(membership["symbol"]):
        raise BlockedData("invalid_industry_coverage")
    trading_dates = pd.DatetimeIndex(membership["date"].drop_duplicates()).sort_values()
    for symbol, member_rows in membership.groupby("symbol", sort=False):
        member_dates = set(member_rows["date"])
        intervals = industry[industry["symbol"] == symbol].sort_values(
            ["start_date", "end_date"]
        )
        covered: set[pd.Timestamp] = set()
        previous_end: pd.Timestamp | None = None
        for row in intervals.itertuples(index=False):
            start = pd.Timestamp(row.start_date)
            end = pd.Timestamp(row.end_date)
            if previous_end is not None and start <= previous_end:
                raise BlockedData("overlapping_industry_intervals")
            interval_dates = set(
                trading_dates[(trading_dates >= start) & (trading_dates <= end)]
            )
            if (
                not interval_dates
                or not interval_dates.issubset(member_dates)
                or covered.intersection(interval_dates)
            ):
                raise BlockedData("invalid_industry_coverage")
            covered.update(interval_dates)
            previous_end = end
        if covered != member_dates:
            raise BlockedData("invalid_industry_coverage")
    for name in ("st_status.parquet", "suspension_status.parquet"):
        frame = tables[name]
        if frame.duplicated(["date", "symbol"]).any():
            raise BlockedData("duplicate_reference_member_day")
        keys = set(frame[["date", "symbol"]].itertuples(index=False, name=None))
        if keys != expected_keys:
            raise BlockedData("incomplete_reference_member_day")
        if (frame["date"].dt.date > today).any():
            raise BlockedData("future_reference_date")


def build_reference_tables(
    provider: Any, membership: pd.DataFrame, today: date
) -> dict[str, pd.DataFrame]:
    security = _build_security_master(provider, membership, today)
    daily_industry = _fetch_daily_industries(provider, membership)
    industry = _compress_industry_intervals(daily_industry, membership["date"].unique())
    st = _fetch_st_status(provider, membership)
    suspension = _fetch_suspension_status(provider, membership)
    tables = {
        "security_master.parquet": security,
        "industry_intervals.parquet": industry,
        "st_status.parquet": st,
        "suspension_status.parquet": suspension,
    }
    _validate_tables(tables, membership, today)
    return tables


def _status_payload(
    status: str,
    reason_code: str,
    start_date: date,
    end_date: date,
    *,
    pit_file_count: int = 0,
    member_count: int = 0,
    member_day_count: int = 0,
    table_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    counts = table_counts or {}
    return {
        "status": status,
        "reason_code": reason_code,
        "date_range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "pit_file_count": int(pit_file_count),
        "member_count": int(member_count),
        "member_day_count": int(member_day_count),
        "security_master_count": int(counts.get("security_master.parquet", 0)),
        "industry_interval_count": int(counts.get("industry_intervals.parquet", 0)),
        "st_status_count": int(counts.get("st_status.parquet", 0)),
        "suspension_status_count": int(counts.get("suspension_status.parquet", 0)),
    }


def _write_json_stage(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_write_status(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json_stage(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_backup(backup: Path, target: Path) -> None:
    rollback = target.with_name(f".{target.name}.{uuid.uuid4().hex}.rollback")
    try:
        try:
            os.link(backup, rollback)
        except OSError:
            shutil.copy2(backup, rollback)
        os.replace(rollback, target)
    finally:
        rollback.unlink(missing_ok=True)


def _publish_transaction(
    tables: Mapping[str, pd.DataFrame],
    reference_dir: Path,
    status_payload: Mapping[str, Any],
    status_path: Path,
) -> None:
    transaction_id = uuid.uuid4().hex
    entries: list[dict[str, Any]] = []
    try:
        reference_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        for name in TABLE_COLUMNS:
            target = reference_dir / name
            stage = target.with_name(f".{target.name}.{transaction_id}.stage")
            tables[name].to_parquet(stage, index=False)
            entries.append(
                {
                    "target": target,
                    "stage": stage,
                    "backup": None,
                    "had_original": target.is_file(),
                }
            )
        status_stage = status_path.with_name(
            f".{status_path.name}.{transaction_id}.stage"
        )
        _write_json_stage(status_payload, status_stage)
        entries.append(
            {
                "target": status_path,
                "stage": status_stage,
                "backup": None,
                "had_original": status_path.is_file(),
            }
        )
        for entry in entries:
            if entry["had_original"]:
                target = entry["target"]
                backup = target.with_name(f".{target.name}.{transaction_id}.backup")
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
                entry["backup"] = backup
        for entry in entries:
            os.replace(entry["stage"], entry["target"])
    except Exception:
        rollback_error: Exception | None = None
        for entry in reversed(entries):
            try:
                if entry["had_original"]:
                    backup = entry["backup"]
                    if backup is None or not backup.is_file():
                        raise FileNotFoundError("transaction backup missing")
                    _restore_backup(backup, entry["target"])
                else:
                    entry["target"].unlink(missing_ok=True)
            except (
                Exception
            ) as exc:  # pragma: no cover - catastrophic rollback failure.
                rollback_error = rollback_error or exc
        if rollback_error is not None:
            raise RuntimeError(
                "reference transaction rollback failed"
            ) from rollback_error
        raise
    finally:
        for entry in entries:
            entry["stage"].unlink(missing_ok=True)
            backup = entry["backup"]
            if backup is not None:
                backup.unlink(missing_ok=True)


def _create_default_provider(
    env_file: Path | None,
    environ: MutableMapping[str, str],
) -> Any:
    jq_module = _load_jq_module()
    authenticated = (
        authenticate(jq_module, environ, env_file=env_file)
        if env_file is not None
        else authenticate(jq_module, environ)
    )
    if not authenticated:
        raise BlockedData("jqdata_auth_failed")
    return jq_module


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download strict PIT reference metadata from JQData."
    )
    parser.add_argument("--pit-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--status-out-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: Any | None = None,
    today: date | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    current_date = date.today() if today is None else today
    args = _argument_parser().parse_args(argv)
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date or current_date.isoformat())
    reference_dir = (
        default_reference_dir() if args.reference_dir is None else args.reference_dir
    )
    pit_dir = default_pit_dir() if args.pit_dir is None else args.pit_dir
    status_dir = reference_dir if args.status_out_dir is None else args.status_out_dir
    status_path = status_dir / STATUS_FILENAME
    pit_file_count = 0
    member_count = 0
    member_day_count = 0

    def blocked(reason_code: str) -> int:
        payload = _status_payload(
            "blocked_data",
            reason_code,
            start_date,
            end_date,
            pit_file_count=pit_file_count,
            member_count=member_count,
            member_day_count=member_day_count,
        )
        try:
            _atomic_write_status(payload, status_path)
        except Exception:
            print(
                "download_robust_reference: blocked_data (status_publish_failed)",
                file=sys.stderr,
            )
            return BLOCKED_RETURN_CODE
        print(
            f"download_robust_reference: blocked_data ({reason_code})", file=sys.stderr
        )
        return BLOCKED_RETURN_CODE

    if start_date > end_date:
        return blocked("invalid_date_range")
    if end_date > current_date:
        return blocked("future_requested_date")
    try:
        membership, pit_file_count = _read_pit_membership(
            pit_dir, start_date, end_date, current_date
        )
        member_count = int(membership["symbol"].nunique())
        member_day_count = len(membership)
        active_provider = provider
        if active_provider is None:
            active_provider = _create_default_provider(
                args.env_file, os.environ if environ is None else environ
            )
        tables = build_reference_tables(active_provider, membership, current_date)
    except BlockedData as exc:
        return blocked(exc.reason_code)
    except Exception as exc:
        return blocked(_provider_reason(exc))

    counts = {name: len(frame) for name, frame in tables.items()}
    completed = _status_payload(
        "completed",
        "completed",
        start_date,
        end_date,
        pit_file_count=pit_file_count,
        member_count=member_count,
        member_day_count=member_day_count,
        table_counts=counts,
    )
    try:
        _publish_transaction(tables, reference_dir, completed, status_path)
    except Exception:
        return blocked("atomic_publish_failed")
    print(
        f"download_robust_reference: completed; members={member_count}; member_days={member_day_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
