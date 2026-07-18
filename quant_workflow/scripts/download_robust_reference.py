#!/usr/bin/env python3
"""Build point-in-time reference metadata for the robust CSI800 strategy."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import stat
import sys
import unicodedata
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from _file_lock import lock_file, unlock_file  # noqa: E402
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from _robust_reference_policy import (  # noqa: E402
    MAX_POST_DELIST_MEMBERSHIP_LAG_DAYS,
)
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from download_jqdata import _load_jq_module, authenticate  # noqa: E402


DEFAULT_START_DATE = "2014-01-02"
STATUS_FILENAME = "download_robust_reference_latest.json"
SOURCE_FILENAME = "REFERENCE_SOURCE.json"
TRANSACTION_JOURNAL_FILENAME = ".download_robust_reference.transaction.json"
WRITER_LOCK_FILENAME = ".download_robust_reference.lock"
TRANSACTION_JOURNAL_VERSION = 1
BLOCKED_RETURN_CODE = 2
SYMBOL_CHUNK_SIZE = 200
BAOSTOCK_MAX_ATTEMPTS = 2
BAOSTOCK_RELOGIN_ERROR_CODES = {f"1000200{suffix}" for suffix in range(1, 9)}
BAOSTOCK_BLACKLIST_ERROR_CODE = "10001011"
BAOSTOCK_CACHE_SCHEMA_VERSION = 2
BAOSTOCK_SOCKET_TIMEOUT_SECONDS = 60.0
IS_WINDOWS = os.name == "nt"
TABLE_COLUMNS = {
    "security_master.parquet": ("symbol", "list_date", "delist_date", "board"),
    "industry_intervals.parquet": ("symbol", "industry_code", "start_date", "end_date"),
    "st_status.parquet": ("date", "symbol", "is_st"),
    "suspension_status.parquet": ("date", "symbol", "is_suspended"),
}
QLIB_SYMBOL_PATTERN = re.compile(r"^(SH|SZ|BJ)(\d{6})$")
JQ_SYMBOL_PATTERN = re.compile(r"^(\d{6})\.(XSHG|XSHE|XBSE)$")
BAOSTOCK_SYMBOL_PATTERN = re.compile(r"^(sh|sz|bj)\.(\d{6})$", re.IGNORECASE)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
EXCHANGE_TO_JQ = {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBSE"}
JQ_TO_EXCHANGE = {value: key for key, value in EXCHANGE_TO_JQ.items()}
JQDATA_FALLBACK_REASONS = {
    "jqdata_auth_failed",
    "jqdata_sdk_unavailable",
    "jqdata_permission_denied",
    "jqdata_quota_exceeded",
}


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


def qlib_to_baostock_symbol(value: Any) -> str:
    matched = QLIB_SYMBOL_PATTERN.fullmatch(str(value).strip().upper())
    if matched is None:
        raise BlockedData("invalid_pit_symbol")
    exchange, digits = matched.groups()
    return f"{exchange.lower()}.{digits}"


def baostock_to_qlib_symbol(value: Any) -> str:
    matched = BAOSTOCK_SYMBOL_PATTERN.fullmatch(str(value).strip())
    if matched is None:
        raise BlockedData("invalid_provider_symbol")
    exchange, digits = matched.groups()
    return f"{exchange.upper()}{digits}"


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


def _parse_run_id(value: str) -> str:
    normalized = str(value).strip()
    if RUN_ID_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("run id contains unsupported characters")
    return normalized


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
    auth_markers = (
        "authentication",
        "credential",
        "login failed",
        "not logged in",
        "用户名",
        "密码",
        "登录",
    )
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
    if any(marker in text for marker in auth_markers):
        return "jqdata_auth_failed"
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
        rows.append(
            {
                "symbol": symbol,
                "list_date": list_date,
                "delist_date": delist_date,
                "board": board_from_symbol(symbol),
            }
        )
    security = pd.DataFrame(
        rows, columns=TABLE_COLUMNS["security_master.parquet"]
    )
    _validate_security_membership_bounds(membership, security)
    return security


def _security_delist_dates(
    security: pd.DataFrame | None,
) -> dict[str, pd.Timestamp]:
    if security is None:
        return {}
    if not {"symbol", "delist_date"}.issubset(security.columns):
        raise BlockedData("invalid_security_master_schema")
    normalized = security.loc[:, ["symbol", "delist_date"]].copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()
    if normalized.duplicated("symbol").any():
        raise BlockedData("duplicate_security_master_member")
    return {
        row.symbol: _normalize_timestamp(row.delist_date)
        for row in normalized.itertuples(index=False)
        if pd.notna(row.delist_date)
    }


def _validate_security_membership_bounds(
    membership: pd.DataFrame, security: pd.DataFrame
) -> None:
    if not {"symbol", "list_date", "delist_date"}.issubset(security.columns):
        raise BlockedData("invalid_security_master_schema")
    normalized = security.loc[:, ["symbol", "list_date", "delist_date"]].copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()
    if normalized.duplicated("symbol").any():
        raise BlockedData("duplicate_security_master_member")
    normalized["list_date"] = normalized["list_date"].map(_normalize_timestamp)
    normalized["delist_date"] = normalized["delist_date"].map(
        lambda value: _normalize_timestamp(value, allow_none=True)
    )
    by_symbol = normalized.set_index("symbol")
    member_bounds = membership.groupby("symbol")["date"].agg(["min", "max"])
    if set(member_bounds.index) != set(by_symbol.index):
        raise BlockedData("invalid_security_master_coverage")
    maximum_lag = timedelta(days=MAX_POST_DELIST_MEMBERSHIP_LAG_DAYS)
    for symbol, bounds in member_bounds.iterrows():
        row = by_symbol.loc[symbol]
        if row["list_date"] > bounds["min"]:
            raise BlockedData("invalid_security_master_member_day")
        delist_date = row["delist_date"]
        if pd.notna(delist_date) and bounds["max"] > delist_date + maximum_lag:
            raise BlockedData("invalid_security_master_member_day")


def _materialize_daily_status(
    membership: pd.DataFrame,
    lookup: Mapping[tuple[pd.Timestamp, str], bool],
    security: pd.DataFrame | None,
    *,
    missing_reason: str,
    forced_post_delist_value: bool | None,
) -> np.ndarray:
    delist_dates = _security_delist_dates(security)
    values = np.empty(len(membership), dtype=np.bool_)
    ordered = membership.loc[:, ["date", "symbol"]].copy()
    ordered["position"] = np.arange(len(ordered), dtype=np.int64)
    last_observed: dict[str, bool] = {}
    for timestamp, symbol, position in ordered.sort_values(
        ["date", "symbol"], kind="mergesort"
    ).itertuples(index=False, name=None):
        key = (timestamp, symbol)
        delist_date = delist_dates.get(symbol)
        post_delist = delist_date is not None and timestamp >= delist_date
        if key in lookup:
            value = bool(lookup[key])
            if post_delist and forced_post_delist_value is True and not value:
                raise BlockedData("tradable_after_delist")
            last_observed[symbol] = value
        elif post_delist and forced_post_delist_value is not None:
            value = forced_post_delist_value
        elif post_delist and symbol in last_observed:
            value = last_observed[symbol]
        else:
            raise BlockedData(missing_reason)
        values[position] = value
    return values


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


def _fetch_st_status(
    provider: Any,
    membership: pd.DataFrame,
    security: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    values = _materialize_daily_status(
        membership,
        lookup,
        security,
        missing_reason="missing_st_member_day",
        forced_post_delist_value=None,
    )
    return pd.DataFrame(
        {
            "date": membership["date"].to_numpy(copy=False),
            "symbol": membership["symbol"].to_numpy(copy=False),
            "is_st": values,
        },
        columns=TABLE_COLUMNS["st_status.parquet"],
    )


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


def _fetch_suspension_status(
    provider: Any,
    membership: pd.DataFrame,
    security: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
    values = _materialize_daily_status(
        membership,
        lookup,
        security,
        missing_reason="missing_paused_member_day",
        forced_post_delist_value=True,
    )
    return pd.DataFrame(
        {
            "date": membership["date"].to_numpy(copy=False),
            "symbol": membership["symbol"].to_numpy(copy=False),
            "is_suspended": values,
        },
        columns=TABLE_COLUMNS["suspension_status.parquet"],
    )


class _BaostockResponseError(Exception):
    """Internal signal for a retryable nonzero Baostock response."""

    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


def _baostock_response_frame(response: Any) -> pd.DataFrame:
    error_code = "" if response is None else str(getattr(response, "error_code", ""))
    if error_code != "0":
        raise _BaostockResponseError(error_code)
    fields = list(getattr(response, "fields", []))
    if not fields or len(fields) != len(set(fields)):
        raise BlockedData("invalid_baostock_schema")
    rows: list[list[Any]] = []
    try:
        while response.next():
            row = list(response.get_row_data())
            if len(row) != len(fields):
                raise BlockedData("invalid_baostock_schema")
            rows.append(row)
    except BlockedData:
        raise
    except Exception as exc:
        raise BlockedData("baostock_fetch_failed") from exc
    error_code = str(getattr(response, "error_code", ""))
    if error_code != "0":
        raise _BaostockResponseError(error_code)
    return pd.DataFrame(rows, columns=fields)


def _require_baostock_columns(
    frame: pd.DataFrame, required: Sequence[str], reason_code: str
) -> None:
    if not set(required).issubset(frame.columns):
        raise BlockedData(reason_code)


def _configure_baostock_socket_timeout(
    provider: Any,
    *,
    import_module=importlib.import_module,
) -> bool:
    if getattr(provider, "__name__", None) != "baostock":
        return False
    try:
        context = import_module("baostock.common.context")
        active_socket = getattr(context, "default_socket")
        active_socket.settimeout(BAOSTOCK_SOCKET_TIMEOUT_SECONDS)
    except Exception as exc:
        raise BlockedData("baostock_socket_timeout_failed") from exc
    return True


def _required_text(value: Any, reason_code: str) -> str:
    if value is None or pd.isna(value):
        raise BlockedData(reason_code)
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        raise BlockedData(reason_code)
    return normalized


def _industry_component(value: Any) -> str:
    normalized = _required_text(value, "missing_industry_member_day").casefold()
    return normalized.replace(":", "_").replace(" ", "_")


class JQDataAdapter:
    source = "jqdata"
    industry_classification = "sw_l1"

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def probe(self, membership: pd.DataFrame, today: date) -> None:
        probe_membership = membership.iloc[[0]].copy()
        security = _build_security_master(self.provider, probe_membership, today)
        _fetch_daily_industries(self.provider, probe_membership)
        _fetch_st_status(self.provider, probe_membership, security)
        _fetch_suspension_status(self.provider, probe_membership, security)

    def build(self, membership: pd.DataFrame, today: date) -> dict[str, pd.DataFrame]:
        security = _build_security_master(self.provider, membership, today)
        daily_industry = _fetch_daily_industries(self.provider, membership)
        industry = _compress_industry_intervals(
            daily_industry, membership["date"].unique()
        )
        return {
            "security_master.parquet": security,
            "industry_intervals.parquet": industry,
            "st_status.parquet": _fetch_st_status(
                self.provider, membership, security
            ),
            "suspension_status.parquet": _fetch_suspension_status(
                self.provider, membership, security
            ),
        }

    def close(self) -> None:
        return None


class BaostockAdapter:
    source = "baostock"

    def __init__(self, provider: Any, *, cache_dir: Path | None = None) -> None:
        self.provider = provider
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.industry_classification: str | None = None
        self._logged_in = False

    @staticmethod
    def _write_cache(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            _fsync_file(temporary)
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _same_member_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
        left_keys = left.loc[:, ["date", "symbol"]].sort_values(
            ["date", "symbol"], kind="mergesort", ignore_index=True
        )
        right_keys = right.loc[:, ["date", "symbol"]].sort_values(
            ["date", "symbol"], kind="mergesort", ignore_index=True
        )
        return len(left_keys) == len(right_keys) and np.array_equal(
            left_keys.to_numpy(copy=False), right_keys.to_numpy(copy=False)
        )

    def _industry_cache_path(self, year: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "industry" / f"{year}.parquet"

    def _status_cache_path(self, symbol: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "status" / f"{symbol}.parquet"

    def login(self) -> None:
        try:
            response = self.provider.login()
        except Exception as exc:
            raise BlockedData("baostock_login_failed") from exc
        error_code = str(getattr(response, "error_code", ""))
        if error_code == BAOSTOCK_BLACKLIST_ERROR_CODE:
            raise BlockedData("baostock_blacklisted")
        if response is None or error_code != "0":
            raise BlockedData("baostock_login_failed")
        self._logged_in = True
        try:
            _configure_baostock_socket_timeout(self.provider)
        except BlockedData:
            self.close()
            raise

    def close(self) -> None:
        if not self._logged_in:
            return
        self._logged_in = False
        try:
            self.provider.logout()
        except Exception:
            pass

    def _relogin(self) -> None:
        self.close()
        self.login()

    def _query(self, method: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        for attempt in range(BAOSTOCK_MAX_ATTEMPTS):
            try:
                response = method(*args, **kwargs)
            except Exception as exc:
                raise BlockedData("baostock_fetch_failed") from exc
            try:
                return _baostock_response_frame(response)
            except _BaostockResponseError as exc:
                if exc.error_code == BAOSTOCK_BLACKLIST_ERROR_CODE:
                    raise BlockedData("baostock_blacklisted") from exc
                if (
                    exc.error_code not in BAOSTOCK_RELOGIN_ERROR_CODES
                    or attempt + 1 >= BAOSTOCK_MAX_ATTEMPTS
                ):
                    raise BlockedData("baostock_query_failed")
                self._relogin()
        raise BlockedData("baostock_query_failed")  # pragma: no cover

    def _build_security_master(
        self, membership: pd.DataFrame, today: date
    ) -> pd.DataFrame:
        frame = self._query(self.provider.query_stock_basic, code="", code_name="")
        required = ("code", "code_name", "ipoDate", "outDate", "type", "status")
        _require_baostock_columns(frame, required, "invalid_security_master_schema")
        member_bounds = membership.groupby("symbol")["date"].agg(["min", "max"])
        expected = set(member_bounds.index)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in frame.loc[:, required].itertuples(index=False, name=None):
            code, code_name, ipo_raw, out_raw, security_type, status = raw
            try:
                symbol = baostock_to_qlib_symbol(code)
            except BlockedData:
                continue
            if symbol not in expected:
                continue
            if symbol in seen:
                raise BlockedData("duplicate_security_master_member")
            _required_text(code_name, "invalid_security_master_member")
            if str(security_type).strip() != "1" or str(status).strip() not in {
                "0",
                "1",
            }:
                raise BlockedData("invalid_security_master_member")
            list_date = _normalize_timestamp(ipo_raw)
            if out_raw is None or pd.isna(out_raw) or not str(out_raw).strip():
                delist_date = pd.NaT
            else:
                delist_date = _normalize_timestamp(out_raw)
            if pd.notna(delist_date) and delist_date.date() > today:
                if delist_date.year >= 2200:
                    delist_date = pd.NaT
                else:
                    raise BlockedData("future_security_master_date")
            if list_date > member_bounds.loc[symbol, "min"]:
                raise BlockedData("invalid_security_master_member_day")
            rows.append(
                {
                    "symbol": symbol,
                    "list_date": list_date,
                    "delist_date": delist_date,
                    "board": board_from_symbol(symbol),
                }
            )
            seen.add(symbol)
        if seen != expected:
            raise BlockedData("missing_security_master_member")
        security = pd.DataFrame(
            sorted(rows, key=lambda row: row["symbol"]),
            columns=TABLE_COLUMNS["security_master.parquet"],
        )
        _validate_security_membership_bounds(membership, security)
        return security

    def _fetch_industry_partition(
        self, membership: pd.DataFrame
    ) -> tuple[pd.DataFrame, str]:
        required = (
            "updateDate",
            "code",
            "code_name",
            "industry",
            "industryClassification",
        )
        classifications: set[str] = set()
        industry_codes = np.empty(len(membership), dtype=object)
        industry_codes[:] = None
        date_groups = membership.groupby("date", sort=True).indices
        total_dates = len(date_groups)
        for completed, (timestamp, positions) in enumerate(
            date_groups.items(), start=1
        ):
            day = timestamp.date().isoformat()
            day_symbols = membership.iloc[positions]["symbol"].to_numpy(copy=False)
            symbols = set(day_symbols)
            frame = self._query(self.provider.query_stock_industry, code="", date=day)
            _require_baostock_columns(frame, required, "invalid_industry_schema")
            found: dict[str, str] = {}
            for raw in frame.loc[:, required].itertuples(index=False, name=None):
                update_raw, code, code_name, industry, classification = raw
                try:
                    update_date = _normalize_timestamp(update_raw)
                except BlockedData as exc:
                    raise BlockedData("invalid_industry_update_date") from exc
                if update_date > timestamp:
                    raise BlockedData("future_industry_update_date")
                try:
                    symbol = baostock_to_qlib_symbol(code)
                except BlockedData:
                    continue
                if symbol not in symbols:
                    continue
                if symbol in found:
                    raise BlockedData("duplicate_industry_member_day")
                _required_text(code_name, "missing_industry_member_day")
                classification_text = _required_text(
                    classification, "missing_industry_member_day"
                )
                industry_code = (
                    f"{_industry_component(classification)}:"
                    f"{_industry_component(industry)}"
                )
                classifications.add(classification_text)
                found[symbol] = industry_code
            for position, symbol in zip(positions, day_symbols):
                industry_code = found.get(symbol)
                if industry_code is None:
                    raise BlockedData("missing_industry_member_day")
                industry_codes[position] = industry_code
            if completed % 25 == 0 or completed == total_dates:
                print(
                    "download_robust_reference: industry date "
                    f"{completed}/{total_dates} year={timestamp.year} "
                    f"last_date={day}",
                    flush=True,
                )
        if len(classifications) != 1:
            raise BlockedData("inconsistent_industry_classification")
        classification = next(iter(classifications))
        return (
            pd.DataFrame(
                {
                    "date": membership["date"].to_numpy(copy=False),
                    "symbol": membership["symbol"].to_numpy(copy=False),
                    "industry_code": industry_codes,
                },
                index=membership.index,
            ),
            classification,
        )

    def _read_industry_cache(
        self, path: Path, membership: pd.DataFrame
    ) -> tuple[pd.DataFrame, str]:
        try:
            cached = pd.read_parquet(path)
        except Exception as exc:
            raise BlockedData("invalid_reference_cache") from exc
        required = {
            "date",
            "symbol",
            "industry_code",
            "industry_classification",
        }
        if not required.issubset(cached.columns):
            raise BlockedData("invalid_reference_cache")
        cached = cached.loc[:, sorted(required)].copy()
        try:
            cached["date"] = pd.to_datetime(
                cached["date"], errors="raise"
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise BlockedData("invalid_reference_cache") from exc
        if getattr(cached["date"].dt, "tz", None) is not None:
            raise BlockedData("invalid_reference_cache")
        cached["symbol"] = cached["symbol"].astype(str).str.strip().str.upper()
        if (
            cached[["date", "symbol", "industry_code"]].isna().any(axis=None)
            or cached.duplicated(["date", "symbol"]).any()
            or not self._same_member_keys(cached, membership)
        ):
            raise BlockedData("invalid_reference_cache")
        classifications = {
            _required_text(value, "invalid_reference_cache")
            for value in cached["industry_classification"].unique()
        }
        if len(classifications) != 1:
            raise BlockedData("invalid_reference_cache")
        classification = next(iter(classifications))
        expected_component = _industry_component(classification)
        normalized_codes: list[str] = []
        for value in cached["industry_code"]:
            code = _required_text(value, "invalid_reference_cache")
            components = code.split(":")
            if (
                len(components) != 2
                or components[0].casefold() != expected_component
                or not components[1].strip()
            ):
                raise BlockedData("invalid_reference_cache")
            normalized_codes.append(code)
        cached["industry_code"] = normalized_codes
        ordered = membership.loc[:, ["date", "symbol"]].merge(
            cached.loc[:, ["date", "symbol", "industry_code"]],
            on=["date", "symbol"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if ordered["industry_code"].isna().any():
            raise BlockedData("invalid_reference_cache")
        return (
            ordered,
            classification,
        )

    def _fetch_daily_industries(self, membership: pd.DataFrame) -> pd.DataFrame:
        partitions: list[pd.DataFrame] = []
        classifications: set[str] = set()
        year_groups = list(membership.groupby(membership["date"].dt.year, sort=True))
        for completed, (year, raw_partition) in enumerate(year_groups, start=1):
            partition_members = raw_partition.reset_index(drop=True)
            cache_path = self._industry_cache_path(int(year))
            cache_hit = cache_path is not None and cache_path.is_file()
            if cache_hit:
                partition, classification = self._read_industry_cache(
                    cache_path, partition_members
                )
            else:
                partition, classification = self._fetch_industry_partition(
                    partition_members
                )
                if cache_path is not None:
                    self._write_cache(
                        partition.assign(industry_classification=classification),
                        cache_path,
                    )
            partitions.append(partition)
            classifications.add(classification)
            print(
                "download_robust_reference: industry "
                f"{completed}/{len(year_groups)} year={year} "
                f"cache={'hit' if cache_hit else 'miss'}",
                flush=True,
            )
        if len(classifications) != 1:
            raise BlockedData("inconsistent_industry_classification")
        self.industry_classification = next(iter(classifications))
        combined = pd.concat(partitions, ignore_index=True)
        if not self._same_member_keys(combined, membership):
            raise BlockedData("invalid_reference_cache")
        return pd.DataFrame(
            {
                "date": combined["date"].to_numpy(copy=False),
                "symbol": combined["symbol"].to_numpy(copy=False),
                "industry_code": combined["industry_code"].to_numpy(copy=False),
            },
            index=membership.index,
        )

    def _read_status_cache(self, path: Path, expected_symbol: str) -> pd.DataFrame:
        try:
            cached = pd.read_parquet(path)
        except Exception as exc:
            raise BlockedData("invalid_reference_cache") from exc
        required = ("symbol", "date", "is_st", "is_suspended")
        if not set(required).issubset(cached.columns):
            raise BlockedData("invalid_reference_cache")
        cached = cached.loc[:, required].copy()
        cached["symbol"] = cached["symbol"].astype(str).str.strip().str.upper()
        if set(cached["symbol"].unique()) != {expected_symbol}:
            raise BlockedData("invalid_reference_cache")
        try:
            cached["date"] = pd.to_datetime(
                cached["date"], errors="raise"
            ).dt.normalize()
        except (TypeError, ValueError) as exc:
            raise BlockedData("invalid_reference_cache") from exc
        if getattr(cached["date"].dt, "tz", None) is not None:
            raise BlockedData("invalid_reference_cache")
        if cached["date"].isna().any() or cached.duplicated("date").any():
            raise BlockedData("invalid_reference_cache")
        for column in ("is_st", "is_suspended"):
            if not cached[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise BlockedData("invalid_reference_cache")
            cached[column] = cached[column].astype(bool)
        return cached.set_index("date").loc[:, ["is_st", "is_suspended"]]

    def _fetch_status_tables(
        self,
        membership: pd.DataFrame,
        security: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = ("date", "tradestatus", "isST")
        start_date = membership["date"].min().date().isoformat()
        end_date = membership["date"].max().date().isoformat()
        st_values = np.empty(len(membership), dtype=np.bool_)
        paused_values = np.empty(len(membership), dtype=np.bool_)
        delist_dates = _security_delist_dates(security)
        symbol_groups = membership.groupby("symbol", sort=True).indices
        for completed, (symbol, positions) in enumerate(
            symbol_groups.items(), start=1
        ):
            cache_path = self._status_cache_path(symbol)
            cache_hit = cache_path is not None and cache_path.is_file()
            if cache_hit:
                history = self._read_status_cache(cache_path, symbol)
            else:
                frame = self._query(
                    self.provider.query_history_k_data_plus,
                    qlib_to_baostock_symbol(symbol),
                    "date,tradestatus,isST",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3",
                )
                _require_baostock_columns(
                    frame, required, "invalid_baostock_history_schema"
                )
                try:
                    history_dates = pd.to_datetime(
                        frame["date"], errors="raise"
                    ).dt.normalize()
                except (TypeError, ValueError) as exc:
                    raise BlockedData("invalid_provider_date") from exc
                if (
                    history_dates.isna().any()
                    or getattr(history_dates.dt, "tz", None) is not None
                ):
                    raise BlockedData("invalid_provider_date")
                if history_dates.duplicated().any():
                    raise BlockedData("duplicate_reference_member_day")
                trade_status = frame["tradestatus"].astype(str).str.strip()
                is_st = frame["isST"].astype(str).str.strip()
                if (
                    not trade_status.isin(("0", "1")).all()
                    or not is_st.isin(("0", "1")).all()
                ):
                    raise BlockedData("invalid_baostock_history_schema")
                history = pd.DataFrame(
                    {
                        "is_st": is_st.eq("1").to_numpy(),
                        "is_suspended": trade_status.eq("0").to_numpy(),
                    },
                    index=pd.DatetimeIndex(history_dates),
                )
            member_dates = pd.DatetimeIndex(
                membership.iloc[positions]["date"].to_numpy(copy=False)
            )
            aligned = history.reindex(member_dates)
            delist_date = delist_dates.get(symbol)
            if delist_date is not None:
                post_delist = member_dates >= delist_date
                if post_delist.any():
                    observed_post_delist = aligned.loc[post_delist, "is_suspended"]
                    if (
                        observed_post_delist.notna()
                        & ~observed_post_delist.astype("boolean").fillna(False)
                    ).any():
                        raise BlockedData("tradable_after_delist")
                    carried_st = history["is_st"].sort_index().reindex(
                        member_dates, method="ffill"
                    )
                    missing_post_delist_st = post_delist & aligned["is_st"].isna()
                    aligned.loc[missing_post_delist_st, "is_st"] = carried_st.loc[
                        missing_post_delist_st
                    ]
                    aligned.loc[post_delist, "is_suspended"] = True
            if aligned["is_st"].isna().any():
                raise BlockedData("missing_st_member_day")
            if aligned["is_suspended"].isna().any():
                raise BlockedData("missing_paused_member_day")
            if not cache_hit and cache_path is not None:
                self._write_cache(
                    history.rename_axis("date")
                    .reset_index()
                    .assign(symbol=symbol)
                    .loc[:, ["symbol", "date", "is_st", "is_suspended"]],
                    cache_path,
                )
            st_values[positions] = aligned["is_st"].to_numpy(dtype=np.bool_)
            paused_values[positions] = aligned["is_suspended"].to_numpy(dtype=np.bool_)
            if completed % 100 == 0 or completed == len(symbol_groups):
                print(
                    "download_robust_reference: status "
                    f"{completed}/{len(symbol_groups)} "
                    f"last_symbol={symbol} cache={'hit' if cache_hit else 'miss'}",
                    flush=True,
                )
        dates = membership["date"].to_numpy(copy=False)
        symbols = membership["symbol"].to_numpy(copy=False)
        return (
            pd.DataFrame(
                {"date": dates, "symbol": symbols, "is_st": st_values},
                index=membership.index,
                columns=TABLE_COLUMNS["st_status.parquet"],
            ),
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbols,
                    "is_suspended": paused_values,
                },
                index=membership.index,
                columns=TABLE_COLUMNS["suspension_status.parquet"],
            ),
        )

    def build(self, membership: pd.DataFrame, today: date) -> dict[str, pd.DataFrame]:
        security = self._build_security_master(membership, today)
        st, suspension = self._fetch_status_tables(membership, security)
        daily_industry = self._fetch_daily_industries(membership)
        industry = _compress_industry_intervals(
            daily_industry, membership["date"].unique()
        )
        return {
            "security_master.parquet": security,
            "industry_intervals.parquet": industry,
            "st_status.parquet": st,
            "suspension_status.parquet": suspension,
        }


def _validate_tables(
    tables: Mapping[str, pd.DataFrame], membership: pd.DataFrame, today: date
) -> None:
    expected_keys = membership.loc[:, ["date", "symbol"]].sort_values(
        ["date", "symbol"], kind="mergesort", ignore_index=True
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
    _validate_security_membership_bounds(membership, security)
    industry = tables["industry_intervals.parquet"]
    if industry.duplicated(["symbol", "start_date", "end_date"]).any():
        raise BlockedData("duplicate_industry_interval")
    industry = industry.copy()
    try:
        industry["industry_code"] = industry["industry_code"].map(
            lambda value: _required_text(value, "invalid_industry_interval")
        )
    except BlockedData as exc:
        raise BlockedData("invalid_industry_interval") from exc
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
    status_columns = {
        "st_status.parquet": "is_st",
        "suspension_status.parquet": "is_suspended",
    }
    for name, value_column in status_columns.items():
        frame = tables[name]
        if frame.duplicated(["date", "symbol"]).any():
            raise BlockedData("duplicate_reference_member_day")
        available_keys = frame.loc[:, ["date", "symbol"]].sort_values(
            ["date", "symbol"], kind="mergesort", ignore_index=True
        )
        same_keys = (
            len(available_keys) == len(expected_keys)
            and np.array_equal(
                available_keys["date"].to_numpy(copy=False),
                expected_keys["date"].to_numpy(copy=False),
            )
            and np.array_equal(
                available_keys["symbol"].to_numpy(copy=False),
                expected_keys["symbol"].to_numpy(copy=False),
            )
        )
        if not same_keys:
            raise BlockedData("incomplete_reference_member_day")
        if (frame["date"].dt.date > today).any():
            raise BlockedData("future_reference_date")
        if not frame[value_column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise BlockedData("invalid_reference_schema")

    delist_dates = _security_delist_dates(security)
    if delist_dates:
        delisted_symbols = set(delist_dates)
        residual = membership.loc[
            membership["symbol"].isin(delisted_symbols), ["date", "symbol"]
        ].copy()
        residual["delist_date"] = pd.to_datetime(
            residual["symbol"].map(delist_dates), errors="raise"
        ).dt.normalize()
        residual = residual.loc[residual["date"] >= residual["delist_date"]]
        if not residual.empty:
            suspension = tables["suspension_status.parquet"]
            suspension = suspension.loc[
                suspension["symbol"].isin(delisted_symbols),
                ["date", "symbol", "is_suspended"],
            ]
            residual = residual.merge(
                suspension,
                on=["date", "symbol"],
                how="left",
                sort=False,
                validate="one_to_one",
            )
            if not residual["is_suspended"].eq(True).all():
                raise BlockedData("tradable_after_delist")


def build_reference_tables(
    adapter: Any, membership: pd.DataFrame, today: date
) -> dict[str, pd.DataFrame]:
    active_adapter = adapter if hasattr(adapter, "build") else JQDataAdapter(adapter)
    tables = active_adapter.build(membership, today)
    _validate_tables(tables, membership, today)
    return tables


def _read_existing_reference_tables(reference_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    try:
        for name in TABLE_COLUMNS:
            tables[name] = pd.read_parquet(reference_dir / name)
    except Exception as exc:
        raise BlockedData("reference_snapshot_invalid") from exc
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
    requested: str = "auto",
    selected: str | None = None,
    fallback_reason: str | None = None,
    industry_classification: str | None = None,
    membership_snapshot_hash: str | None = None,
    build_run_id: str | None = None,
) -> dict[str, Any]:
    counts = table_counts or {}
    payload = {
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
        "requested": requested,
        "selected": selected,
        "industry_classification": industry_classification,
        "membership_snapshot_hash": membership_snapshot_hash,
    }
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    if build_run_id is not None:
        payload["build_run_id"] = build_run_id
    return payload


def _source_payload(
    requested: str,
    selected: str,
    start_date: date,
    end_date: date,
    industry_classification: str,
    membership_snapshot_hash: str,
    fallback_reason: str | None,
    build_run_id: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested": requested,
        "selected": selected,
        "date_range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "industry_classification": industry_classification,
        "membership_snapshot_hash": membership_snapshot_hash,
    }
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    if build_run_id is not None:
        payload["build_run_id"] = build_run_id
    return payload


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


def _fsync_directory(path: Path) -> None:
    if IS_WINDOWS:
        # Windows does not expose POSIX-style directory descriptors.  Files are
        # still fsynced before each atomic replace; directory fsync is the only
        # unavailable durability step on this platform.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDWR if IS_WINDOWS else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in sorted({Path(value).resolve() for value in paths}, key=str):
        _fsync_directory(path)


def _write_transaction_journal(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _serialize_transaction_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "target": str(Path(entry["target"]).resolve()),
            "stage": str(Path(entry["stage"]).resolve()),
            "backup": (
                None
                if entry["backup"] is None
                else str(Path(entry["backup"]).resolve())
            ),
            "had_original": bool(entry["had_original"]),
        }
        for entry in entries
    ]


def _transaction_journal_payload(
    transaction_id: str,
    state: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": TRANSACTION_JOURNAL_VERSION,
        "transaction_id": transaction_id,
        "state": state,
        "entries": _serialize_transaction_entries(entries),
    }


def _validated_transaction_entries(
    payload: Any, allowed_targets: Sequence[Path]
) -> tuple[str, list[dict[str, Any]]]:
    if (
        not isinstance(payload, Mapping)
        or payload.get("version") != TRANSACTION_JOURNAL_VERSION
        or payload.get("state") not in {"prepared", "committed"}
        or not isinstance(payload.get("transaction_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["transaction_id"]) is None
        or not isinstance(payload.get("entries"), list)
        or not payload["entries"]
    ):
        raise BlockedData("transaction_recovery_failed")
    allowed = {path.resolve() for path in allowed_targets}
    transaction_id = str(payload["transaction_id"])
    entries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in payload["entries"]:
        if not isinstance(raw, Mapping) or type(raw.get("had_original")) is not bool:
            raise BlockedData("transaction_recovery_failed")
        try:
            target = Path(raw["target"]).resolve()
            stage = Path(raw["stage"]).resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise BlockedData("transaction_recovery_failed") from exc
        had_original = raw["had_original"]
        backup_raw = raw.get("backup")
        if had_original:
            if not isinstance(backup_raw, str):
                raise BlockedData("transaction_recovery_failed")
            backup: Path | None = Path(backup_raw).resolve()
        elif backup_raw is None:
            backup = None
        else:
            raise BlockedData("transaction_recovery_failed")
        if (
            target not in allowed
            or target in seen
            or stage != target.with_name(f".{target.name}.{transaction_id}.stage")
            or (
                backup is not None
                and backup
                != target.with_name(f".{target.name}.{transaction_id}.backup")
            )
            or target == stage
            or target == backup
            or stage == backup
        ):
            raise BlockedData("transaction_recovery_failed")
        entries.append(
            {
                "target": target,
                "stage": stage,
                "backup": backup,
                "had_original": had_original,
            }
        )
        seen.add(target)
    return str(payload["state"]), entries


def _cleanup_transaction_artifacts(
    entries: Sequence[Mapping[str, Any]], journal_path: Path
) -> None:
    directories = {journal_path.parent}
    for entry in entries:
        stage = Path(entry["stage"])
        stage.unlink(missing_ok=True)
        directories.add(stage.parent)
        backup = entry["backup"]
        if backup is not None:
            backup = Path(backup)
            backup.unlink(missing_ok=True)
            directories.add(backup.parent)
    journal_path.unlink(missing_ok=True)
    for directory in directories:
        _fsync_directory(directory)


def _recover_transaction(journal_path: Path, allowed_targets: Sequence[Path]) -> None:
    try:
        journal_mode = journal_path.lstat().st_mode
    except FileNotFoundError:
        return
    if journal_path.is_symlink() or not stat.S_ISREG(journal_mode):
        raise BlockedData("transaction_recovery_failed")
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        state, entries = _validated_transaction_entries(payload, allowed_targets)
        if state == "prepared":
            for entry in reversed(entries):
                target = entry["target"]
                if entry["had_original"]:
                    backup = entry["backup"]
                    if backup is None or not backup.is_file():
                        raise FileNotFoundError("transaction backup missing")
                    _restore_backup(backup, target)
                else:
                    target.unlink(missing_ok=True)
        _cleanup_transaction_artifacts(entries, journal_path)
    except BlockedData:
        raise
    except Exception as exc:
        raise BlockedData("transaction_recovery_failed") from exc


def _acquire_writer_lock(reference_dir: Path) -> Any:
    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = reference_dir.parent / f".{reference_dir.name}{WRITER_LOCK_FILENAME}"
    handle = lock_path.open("a+b")
    try:
        lock_file(handle, exclusive=True)
    except Exception:
        handle.close()
        raise
    return handle


def _release_writer_lock(handle: Any) -> None:
    try:
        unlock_file(handle)
    finally:
        handle.close()


def _publish_transaction(
    tables: Mapping[str, pd.DataFrame],
    reference_dir: Path,
    source_payload: Mapping[str, Any],
    source_path: Path,
    status_payload: Mapping[str, Any],
) -> None:
    transaction_id = uuid.uuid4().hex
    journal_path = reference_dir / TRANSACTION_JOURNAL_FILENAME
    status_path = reference_dir / STATUS_FILENAME
    entries: list[dict[str, Any]] = []
    try:
        reference_dir.mkdir(parents=True, exist_ok=True)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        for name in TABLE_COLUMNS:
            target = reference_dir / name
            stage = target.with_name(f".{target.name}.{transaction_id}.stage")
            entries.append(
                {
                    "target": target,
                    "stage": stage,
                    "backup": None,
                    "had_original": target.is_file(),
                }
            )
            tables[name].to_parquet(stage, index=False)
        source_stage = source_path.with_name(
            f".{source_path.name}.{transaction_id}.stage"
        )
        entries.append(
            {
                "target": source_path,
                "stage": source_stage,
                "backup": None,
                "had_original": source_path.is_file(),
            }
        )
        _write_json_stage(source_payload, source_stage)
        status_stage = status_path.with_name(
            f".{status_path.name}.{transaction_id}.stage"
        )
        entries.append(
            {
                "target": status_path,
                "stage": status_stage,
                "backup": None,
                "had_original": status_path.is_file(),
            }
        )
        _write_json_stage(status_payload, status_stage)
        for entry in entries:
            if entry["had_original"]:
                target = entry["target"]
                backup = target.with_name(f".{target.name}.{transaction_id}.backup")
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
                entry["backup"] = backup
        artifact_directories: set[Path] = set()
        for entry in entries:
            stage = Path(entry["stage"])
            _fsync_file(stage)
            artifact_directories.add(stage.parent)
            backup = entry["backup"]
            if backup is not None:
                backup = Path(backup)
                _fsync_file(backup)
                artifact_directories.add(backup.parent)
        _fsync_directories(list(artifact_directories))
        _write_transaction_journal(
            _transaction_journal_payload(transaction_id, "prepared", entries),
            journal_path,
        )
        for entry in entries:
            os.replace(entry["stage"], entry["target"])
        _fsync_directories([Path(entry["target"]).parent for entry in entries])
        _write_transaction_journal(
            _transaction_journal_payload(transaction_id, "committed", entries),
            journal_path,
        )
        _cleanup_transaction_artifacts(entries, journal_path)
    except Exception:
        if journal_path.is_file():
            try:
                _recover_transaction(
                    journal_path,
                    [entry["target"] for entry in entries],
                )
            except BlockedData as exc:
                raise RuntimeError("reference transaction rollback failed") from exc
        else:
            for entry in entries:
                Path(entry["stage"]).unlink(missing_ok=True)
                backup = entry["backup"]
                if backup is not None:
                    Path(backup).unlink(missing_ok=True)
        raise


def _create_default_provider(
    env_file: Path | None,
    environ: MutableMapping[str, str],
) -> Any:
    try:
        jq_module = _load_jq_module()
    except Exception as exc:
        raise BlockedData("jqdata_sdk_unavailable") from exc
    try:
        authenticated = (
            authenticate(jq_module, environ, env_file=env_file)
            if env_file is not None
            else authenticate(jq_module, environ)
        )
    except Exception as exc:
        raise BlockedData("jqdata_auth_failed") from exc
    if not authenticated:
        raise BlockedData("jqdata_auth_failed")
    return jq_module


def _create_default_baostock_provider() -> Any:
    try:
        return importlib.import_module("baostock")
    except Exception as exc:
        raise BlockedData("baostock_sdk_unavailable") from exc


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download strict PIT reference metadata from one selected source."
    )
    parser.add_argument("--pit-dir", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--status-out-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=_parse_run_id, default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--reference-source",
        choices=("auto", "jqdata", "baostock"),
        default="auto",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Validate and reuse an existing reference snapshot without network calls.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: Any | None = None,
    baostock_provider: Any | None = None,
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
    canonical_status_path = reference_dir / STATUS_FILENAME
    source_path = reference_dir / SOURCE_FILENAME
    journal_path = reference_dir / TRANSACTION_JOURNAL_FILENAME
    pit_file_count = 0
    member_count = 0
    member_day_count = 0
    requested = args.reference_source
    selected: str | None = None
    fallback_reason: str | None = None
    industry_classification: str | None = None
    membership_snapshot_hash: str | None = None
    cleanup_adapters: list[Any] = []

    def blocked(reason_code: str) -> int:
        payload = _status_payload(
            "blocked_data",
            reason_code,
            start_date,
            end_date,
            pit_file_count=pit_file_count,
            member_count=member_count,
            member_day_count=member_day_count,
            requested=requested,
            selected=selected,
            fallback_reason=fallback_reason,
            industry_classification=industry_classification,
            membership_snapshot_hash=membership_snapshot_hash,
            build_run_id=args.run_id,
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

    allowed_transaction_targets = [
        *(reference_dir / name for name in TABLE_COLUMNS),
        source_path,
        canonical_status_path,
    ]
    recovery_lock = None
    try:
        recovery_lock = _acquire_writer_lock(reference_dir)
        _recover_transaction(journal_path, allowed_transaction_targets)
    except BlockedData:
        return blocked("transaction_recovery_failed")
    except Exception:
        return blocked("writer_lock_failed")
    finally:
        if recovery_lock is not None:
            _release_writer_lock(recovery_lock)

    try:
        if start_date > end_date:
            return blocked("invalid_date_range")
        if end_date > current_date:
            return blocked("future_requested_date")
        try:
            with membership_read_lock(pit_dir) as membership_snapshot:
                membership_snapshot.ensure_transaction_clean()
                membership_snapshot.validate_source(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                membership, pit_file_count = _read_pit_membership(
                    pit_dir, start_date, end_date, current_date
                )
                member_count = int(membership["symbol"].nunique())
                member_day_count = len(membership)
                membership_snapshot_hash = membership_snapshot.fingerprint()

            if args.reuse_existing:
                try:
                    with reference_read_lock(reference_dir) as reference_snapshot:
                        reference_snapshot.ensure_transaction_clean()
                        existing_source = reference_snapshot.validate_source(
                            start_date=start_date.isoformat(),
                            end_date=end_date.isoformat(),
                            expected_membership_snapshot_hash=membership_snapshot_hash,
                        )
                        tables = _read_existing_reference_tables(reference_dir)
                        _validate_tables(tables, membership, current_date)
                        reference_snapshot.fingerprint()
                    with membership_read_lock(pit_dir) as final_membership:
                        final_membership.ensure_transaction_clean()
                        final_membership.validate_source(
                            start_date=start_date.isoformat(),
                            end_date=end_date.isoformat(),
                        )
                        if (
                            final_membership.fingerprint()
                            != membership_snapshot_hash
                        ):
                            raise BlockedData("membership_snapshot_changed")
                except ReferenceIntegrityError as exc:
                    return blocked(exc.reason_code)
                except BlockedData as exc:
                    return blocked(exc.reason_code)
                selected = str(existing_source["selected"])
                fallback_reason = existing_source.get("fallback_reason")
                industry_classification = str(
                    existing_source["industry_classification"]
                )
                counts = {name: len(frame) for name, frame in tables.items()}
                completed = _status_payload(
                    "completed",
                    "reused_existing",
                    start_date,
                    end_date,
                    pit_file_count=pit_file_count,
                    member_count=member_count,
                    member_day_count=member_day_count,
                    table_counts=counts,
                    requested=requested,
                    selected=selected,
                    fallback_reason=fallback_reason,
                    industry_classification=industry_classification,
                    membership_snapshot_hash=membership_snapshot_hash,
                    build_run_id=args.run_id,
                )
                try:
                    _atomic_write_status(completed, status_path)
                except Exception:
                    return blocked("status_publish_failed")
                print(
                    "download_robust_reference: reused existing snapshot; "
                    f"members={member_count}; member_days={member_day_count}"
                )
                return 0

            baostock_cache_dir = (
                reference_dir.parent
                / "cache"
                / "robust_reference"
                / f"v{BAOSTOCK_CACHE_SCHEMA_VERSION}"
                / "baostock"
                / membership_snapshot_hash
                / f"{start_date.isoformat()}_{end_date.isoformat()}"
            )

            if requested == "baostock":
                selected = "baostock"
                raw_baostock = baostock_provider
                if raw_baostock is None:
                    raw_baostock = (
                        provider
                        if provider is not None
                        else _create_default_baostock_provider()
                    )
                active_adapter: Any = BaostockAdapter(
                    raw_baostock, cache_dir=baostock_cache_dir
                )
                cleanup_adapters.append(active_adapter)
                active_adapter.login()
            elif requested == "jqdata":
                selected = "jqdata"
                raw_jqdata = provider
                if raw_jqdata is None:
                    raw_jqdata = _create_default_provider(
                        args.env_file, os.environ if environ is None else environ
                    )
                active_adapter = JQDataAdapter(raw_jqdata)
            else:
                selected = "jqdata"
                try:
                    raw_jqdata = provider
                    if raw_jqdata is None:
                        raw_jqdata = _create_default_provider(
                            args.env_file,
                            os.environ if environ is None else environ,
                        )
                    active_adapter = JQDataAdapter(raw_jqdata)
                    active_adapter.probe(membership, current_date)
                except BlockedData as exc:
                    if exc.reason_code not in JQDATA_FALLBACK_REASONS:
                        raise
                    # A lone injected provider retains the historical JQData-only
                    # test contract; production auto and dual-provider tests fall back.
                    if provider is not None and baostock_provider is None:
                        raise
                    fallback_reason = exc.reason_code
                    selected = "baostock"
                    raw_baostock = (
                        baostock_provider
                        if baostock_provider is not None
                        else _create_default_baostock_provider()
                    )
                    active_adapter = BaostockAdapter(
                        raw_baostock, cache_dir=baostock_cache_dir
                    )
                    cleanup_adapters.append(active_adapter)
                    active_adapter.login()

            try:
                tables = build_reference_tables(
                    active_adapter, membership, current_date
                )
            except BlockedData as exc:
                can_fallback = (
                    requested == "auto"
                    and selected == "jqdata"
                    and exc.reason_code in JQDATA_FALLBACK_REASONS
                    and (provider is None or baostock_provider is not None)
                )
                if not can_fallback:
                    raise
                fallback_reason = exc.reason_code
                selected = "baostock"
                raw_baostock = (
                    baostock_provider
                    if baostock_provider is not None
                    else _create_default_baostock_provider()
                )
                active_adapter = BaostockAdapter(
                    raw_baostock, cache_dir=baostock_cache_dir
                )
                cleanup_adapters.append(active_adapter)
                active_adapter.login()
                tables = build_reference_tables(
                    active_adapter, membership, current_date
                )
            industry_classification = active_adapter.industry_classification
            if not industry_classification:
                raise BlockedData("missing_industry_classification")
        except MembershipIntegrityError as exc:
            return blocked(exc.reason_code)
        except BlockedData as exc:
            return blocked(exc.reason_code)
        except Exception as exc:
            reason_code = (
                "baostock_fetch_failed"
                if selected == "baostock"
                else _provider_reason(exc)
            )
            return blocked(reason_code)

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
            requested=requested,
            selected=selected,
            fallback_reason=fallback_reason,
            industry_classification=industry_classification,
            membership_snapshot_hash=membership_snapshot_hash,
            build_run_id=args.run_id,
        )
        source = _source_payload(
            requested,
            selected,
            start_date,
            end_date,
            industry_classification,
            membership_snapshot_hash,
            fallback_reason,
            args.run_id,
        )
        publish_lock = None
        try:
            with membership_read_lock(pit_dir) as final_membership_snapshot:
                final_membership_snapshot.ensure_transaction_clean()
                final_membership_snapshot.validate_source(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                )
                if final_membership_snapshot.fingerprint() != membership_snapshot_hash:
                    raise BlockedData("membership_snapshot_changed")
                publish_lock = _acquire_writer_lock(reference_dir)
                _recover_transaction(journal_path, allowed_transaction_targets)
                _publish_transaction(
                    tables,
                    reference_dir,
                    source,
                    source_path,
                    completed,
                )
        except BlockedData as exc:
            return blocked(exc.reason_code)
        except Exception:
            return blocked("atomic_publish_failed")
        finally:
            if publish_lock is not None:
                _release_writer_lock(publish_lock)
        if status_path.resolve() != canonical_status_path.resolve():
            try:
                _atomic_write_status(completed, status_path)
            except Exception:
                return blocked("status_publish_failed")
        print(
            "download_robust_reference: completed; "
            f"members={member_count}; member_days={member_day_count}"
        )
        return 0
    finally:
        for adapter in reversed(cleanup_adapters):
            try:
                adapter.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
