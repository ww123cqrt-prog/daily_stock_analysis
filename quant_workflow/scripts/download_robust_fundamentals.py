#!/usr/bin/env python3
"""Download leakage-safe historical fundamental inputs for Goal 1 research."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from robust_fundamentals import (
    build_daily_fundamental_features,
    build_fundamental_events,
    fundamental_coverage_report,
    prepare_statement_frame,
)
from _paths import quant_data_root, reports_root


EASTMONEY_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
EASTMONEY_SUMMARY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
SOURCE_CONTRACT_VERSION = 3
PERIOD_CACHE_CONTRACT_VERSION = 1
PAGE_SIZE = 500
SYMBOL_BATCH_SIZE = 20
LOCKED_CONFIRMATION_START = pd.Timestamp("2026-01-01")
STATEMENT_QUERIES = {
    "balance": {
        "type": "RPT_F10_FINANCE_GBALANCE",
        "columns": "ALL",
    },
    "income": {
        "type": "RPT_F10_FINANCE_GINCOME",
        "columns": (
            "SECUCODE,SECURITY_CODE,REPORT_DATE,NOTICE_DATE,UPDATE_DATE,REPORT_TYPE,"
            "TOTAL_OPERATE_INCOME,TOTAL_OPERATE_INCOME_YOY,PARENT_NETPROFIT,"
            "PARENT_NETPROFIT_YOY,BASIC_EPS"
        ),
    },
    "cashflow": {
        "type": "RPT_F10_FINANCE_GCASHFLOW",
        "columns": "ALL",
    },
}
SUMMARY_QUERIES = {
    "balance": {
        "report_name": "RPT_DMSK_FN_BALANCE",
        "columns": (
            "SECUCODE,SECURITY_CODE,NOTICE_DATE,REPORT_DATE,TOTAL_ASSETS,"
            "TOTAL_LIABILITIES,TOTAL_EQUITY"
        ),
    },
    "cashflow": {
        "report_name": "RPT_DMSK_FN_CASHFLOW",
        "columns": (
            "SECUCODE,SECURITY_CODE,NOTICE_DATE,REPORT_DATE,NETCASH_OPERATE"
        ),
    },
}


def _eastmoney_code(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if len(normalized) != 8 or normalized[:2] not in {"SH", "SZ", "BJ"}:
        raise ValueError(f"invalid canonical A-share symbol: {symbol}")
    return f"{normalized[2:]}.{normalized[:2]}"


def batch_cache_key(
    *,
    statement: str,
    symbols: Sequence[str],
    start_report_date: str,
    end_report_date: str,
    source_contract_version: int = SOURCE_CONTRACT_VERSION,
) -> str:
    payload = {
        "statement": statement,
        "symbols": sorted(str(value).upper() for value in symbols),
        "start_report_date": str(start_report_date),
        "end_report_date": str(end_report_date),
        "source_contract_version": int(source_contract_version),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def fetch_statement_batch(
    session: Any,
    *,
    statement: str,
    symbols: Sequence[str],
    start_report_date: str,
    end_report_date: str,
    timeout: float,
) -> pd.DataFrame:
    if statement not in STATEMENT_QUERIES:
        raise ValueError(f"unknown fundamental statement: {statement}")
    if not symbols:
        return pd.DataFrame()
    query = STATEMENT_QUERIES[statement]
    codes = ",".join(f'"{_eastmoney_code(symbol)}"' for symbol in symbols)
    filters = f"(SECUCODE in ({codes}))"

    def fetch(page: int) -> dict[str, Any]:
        params = {
            "type": query["type"],
            "sty": query["columns"],
            "filter": filters,
            "p": str(page),
            "ps": str(PAGE_SIZE),
            "sr": "1,1",
            "st": "SECUCODE,REPORT_DATE",
            "source": "HSF10",
            "client": "PC",
        }
        response = session.get(EASTMONEY_URL, params=params, timeout=float(timeout))
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is not True:
            raise RuntimeError(
                f"EastMoney {statement} query failed: {payload.get('message') or 'unknown error'}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"EastMoney {statement} query returned no result")
        return result

    first = fetch(1)
    pages = int(first.get("pages") or 1)
    records = list(first.get("data") or [])
    for page in range(2, pages + 1):
        records.extend(fetch(page).get("data") or [])
    expected = int(first.get("count") or len(records))
    if len(records) != expected:
        raise RuntimeError(
            f"EastMoney {statement} pagination mismatch: expected {expected}, got {len(records)}"
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    report_dates = pd.to_datetime(frame.get("REPORT_DATE"), errors="coerce")
    start = pd.Timestamp(start_report_date)
    end = pd.Timestamp(end_report_date)
    return frame.loc[report_dates.between(start, end, inclusive="both")].reset_index(
        drop=True
    )


def fetch_summary_period(
    session: Any,
    *,
    statement: str,
    report_date: str,
    timeout: float,
) -> pd.DataFrame:
    """Fetch one all-A-share summary period used for revision-safe values."""
    if statement not in SUMMARY_QUERIES:
        raise ValueError(f"unsupported summary statement: {statement}")
    query = SUMMARY_QUERIES[statement]
    filters = (
        '(SECURITY_TYPE_CODE="058001001")'
        f"(REPORT_DATE='{pd.Timestamp(report_date).date()}')"
    )

    def fetch(page: int) -> dict[str, Any]:
        params = {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": "1",
            "pageSize": str(PAGE_SIZE),
            "pageNumber": str(page),
            "reportName": query["report_name"],
            "columns": query["columns"],
            "filter": filters,
        }
        response = session.get(
            EASTMONEY_SUMMARY_URL, params=params, timeout=float(timeout)
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is not True:
            raise RuntimeError(
                f"EastMoney {statement} summary failed: {payload.get('message') or 'unknown error'}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"EastMoney {statement} summary returned no result")
        return result

    first = fetch(1)
    records = list(first.get("data") or [])
    for page in range(2, int(first.get("pages") or 1) + 1):
        records.extend(fetch(page).get("data") or [])
    expected = int(first.get("count") or len(records))
    if len(records) != expected:
        raise RuntimeError(
            f"EastMoney {statement} summary pagination mismatch: expected {expected}, got {len(records)}"
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["UPDATE_DATE"] = frame["NOTICE_DATE"]
    if statement == "balance":
        frame["TOTAL_PARENT_EQUITY"] = frame["TOTAL_EQUITY"]
        frame["SHARE_CAPITAL"] = pd.NA
    return frame


def fetch_income_period(
    session: Any,
    *,
    report_date: str,
    timeout: float,
) -> pd.DataFrame:
    """Fetch one period with original notice and last-update timestamps."""
    query = STATEMENT_QUERIES["income"]
    filters = (
        '(SECURITY_TYPE_CODE="058001001")'
        f"(REPORT_DATE='{pd.Timestamp(report_date).date()}')"
    )

    def fetch(page: int) -> dict[str, Any]:
        params = {
            "type": query["type"],
            "sty": query["columns"],
            "filter": filters,
            "p": str(page),
            "ps": str(PAGE_SIZE),
            "sr": "1",
            "st": "SECUCODE",
            "source": "HSF10",
            "client": "PC",
        }
        response = session.get(EASTMONEY_URL, params=params, timeout=float(timeout))
        response.raise_for_status()
        payload = response.json()
        if payload.get("success") is not True:
            raise RuntimeError(
                f"EastMoney income history failed: {payload.get('message') or 'unknown error'}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("EastMoney income history returned no result")
        return result

    first = fetch(1)
    records = list(first.get("data") or [])
    for page in range(2, int(first.get("pages") or 1) + 1):
        records.extend(fetch(page).get("data") or [])
    expected = int(first.get("count") or len(records))
    if len(records) != expected:
        raise RuntimeError(
            f"EastMoney income pagination mismatch: expected {expected}, got {len(records)}"
        )
    return pd.DataFrame.from_records(records)


def load_member_market_rows(
    membership: pd.DataFrame, *, price_dir: Any
) -> pd.DataFrame:
    """Read each price file once and retain only historical membership dates."""
    from pathlib import Path

    required = {"date", "symbol"}
    missing = sorted(required - set(membership.columns))
    if missing:
        raise ValueError(f"membership is missing columns: {', '.join(missing)}")
    members = membership.loc[:, ["date", "symbol"]].copy()
    members["date"] = pd.to_datetime(members["date"], errors="coerce").dt.normalize()
    members["symbol"] = members["symbol"].astype(str).str.upper()
    members = members.dropna(subset=["date", "symbol"]).drop_duplicates(
        ["date", "symbol"]
    )
    root = Path(price_dir)
    blocks: list[pd.DataFrame] = []
    for symbol, dates in members.groupby("symbol", sort=True):
        path = root / f"{symbol[2:]}.parquet"
        if not path.is_file():
            continue
        prices = pd.read_parquet(
            path, columns=["date", "close", "volume", "amount", "turn"]
        )
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
        prices = prices.dropna(subset=["date"]).drop_duplicates("date", keep="last")
        selected = dates.loc[:, ["date"]].merge(prices, on="date", how="inner")
        selected["symbol"] = symbol
        blocks.append(selected)
    if not blocks:
        return pd.DataFrame(
            columns=["date", "close", "volume", "amount", "turn", "symbol"]
        )
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["date", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def report_periods(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("fundamental report start date is after end date")
    periods = pd.date_range(start, end, freq="QE-DEC")
    return [str(value.date()) for value in periods]


def validate_goal1_dates(
    *, data_start: str, data_end: str, report_start: str, report_end: str
) -> None:
    values = {
        "data_start": pd.Timestamp(data_start).normalize(),
        "data_end": pd.Timestamp(data_end).normalize(),
        "report_start": pd.Timestamp(report_start).normalize(),
        "report_end": pd.Timestamp(report_end).normalize(),
    }
    if values["data_start"] > values["data_end"]:
        raise ValueError("fundamental data start is after end")
    if values["report_start"] > values["report_end"]:
        raise ValueError("fundamental report start is after end")
    if values["data_end"] >= LOCKED_CONFIRMATION_START:
        raise ValueError("Goal 1 fundamental data must not read 2026")
    if values["report_end"] >= LOCKED_CONFIRMATION_START:
        raise ValueError("Goal 1 fundamental reports must not read 2026")


def load_or_fetch_period(
    *,
    cache_dir: Path,
    statement: str,
    report_date: str,
    fetcher: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    path = (
        Path(cache_dir)
        / "eastmoney_hsf10"
        / statement
        / f"{report_date}.contract-v{PERIOD_CACHE_CONTRACT_VERSION}.parquet"
    )
    if path.is_file():
        return pd.read_parquet(path)
    frame = fetcher()
    _atomic_parquet(frame, path)
    return frame


def load_or_fetch_symbol_batch(
    *,
    cache_dir: Path,
    statement: str,
    symbols: Sequence[str],
    start_report_date: str,
    end_report_date: str,
    fetcher: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    key = batch_cache_key(
        statement=statement,
        symbols=symbols,
        start_report_date=start_report_date,
        end_report_date=end_report_date,
    )
    path = (
        Path(cache_dir)
        / "eastmoney_hsf10_symbol_history"
        / statement
        / f"{key}.parquet"
    )
    if path.is_file():
        return pd.read_parquet(path)
    frame = fetcher()
    _atomic_parquet(frame, path)
    return frame


def _canonical_symbol(value: Any) -> str | None:
    parts = str(value).strip().upper().split(".")
    if len(parts) != 2 or len(parts[0]) != 6 or parts[1] not in {"SH", "SZ", "BJ"}:
        return None
    return f"{parts[1]}{parts[0]}"


def supplement_missing_symbols(
    frame: pd.DataFrame,
    *,
    expected_symbols: Sequence[str],
    fetcher: Callable[[Sequence[str]], pd.DataFrame],
    batch_size: int = SYMBOL_BATCH_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    if int(batch_size) <= 0:
        raise ValueError("symbol supplement batch_size must be positive")
    present = {
        symbol
        for symbol in (
            _canonical_symbol(value)
            for value in frame.get("SECUCODE", pd.Series(dtype="object"))
        )
        if symbol is not None
    }
    missing = sorted(
        set(str(value).strip().upper() for value in expected_symbols) - present
    )
    blocks = [frame.reset_index(drop=True)]
    for offset in range(0, len(missing), int(batch_size)):
        batch = missing[offset : offset + int(batch_size)]
        supplement = fetcher(batch)
        if not supplement.empty:
            blocks.append(supplement.reset_index(drop=True))
    result = pd.concat(blocks, ignore_index=True, sort=False)
    return result.drop_duplicates().reset_index(drop=True), missing


def load_symbol_history(
    symbols: Sequence[str],
    *,
    fetcher: Callable[[Sequence[str]], pd.DataFrame],
    batch_size: int = SYMBOL_BATCH_SIZE,
) -> pd.DataFrame:
    if int(batch_size) <= 0:
        raise ValueError("symbol history batch_size must be positive")
    normalized = sorted(set(str(value).strip().upper() for value in symbols))
    blocks: list[pd.DataFrame] = []
    for offset in range(0, len(normalized), int(batch_size)):
        batch = normalized[offset : offset + int(batch_size)]
        frame = fetcher(batch)
        if not frame.empty:
            blocks.append(frame.reset_index(drop=True))
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True, sort=False).drop_duplicates().reset_index(
        drop=True
    )


def merge_statement_history(
    summary: pd.DataFrame, detail: pd.DataFrame
) -> pd.DataFrame:
    """Prefer non-null historical detail while retaining summary-only rows."""
    keys = ["SECUCODE", "REPORT_DATE"]
    for frame_name, frame in (("summary", summary), ("detail", detail)):
        if not set(keys).issubset(frame.columns):
            raise ValueError(f"{frame_name} statement is missing merge keys")

    def indexed(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop_duplicates(keys, keep="last").set_index(keys)

    base = indexed(summary)
    enriched = indexed(detail)
    union = base.index.union(enriched.index)
    result = base.reindex(union)
    for column in enriched.columns:
        values = enriched[column].reindex(union)
        result[column] = values.combine_first(result[column]) if column in result else values
    return result.reset_index()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fundamental_product(
    *,
    raw_statements: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    membership: pd.DataFrame,
    price_dir: Path,
    output_dir: Path,
    coverage_minimum: float,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the official event table and PIT daily feature partitions."""
    prepared = {
        statement: prepare_statement_frame(raw_statements[statement], statement=statement)
        for statement in ("balance", "income", "cashflow")
    }
    events = build_fundamental_events(
        balance=prepared["balance"],
        income=prepared["income"],
        cashflow=prepared["cashflow"],
        calendar=calendar,
    )
    market_rows = load_member_market_rows(membership, price_dir=price_dir)
    features = build_daily_fundamental_features(events, market_rows)
    coverage = fundamental_coverage_report(features, minimum=coverage_minimum)
    output = Path(output_dir)
    events_path = output / "fundamental_events.parquet"
    _atomic_parquet(events, events_path)
    daily_paths: list[Path] = []
    years = pd.to_datetime(features["datetime"], errors="coerce").dt.year
    for year in sorted(int(value) for value in years.dropna().unique()):
        path = output / "daily" / f"{year}.parquet"
        _atomic_parquet(features.loc[years == year].reset_index(drop=True), path)
        daily_paths.append(path)
    status = "pass" if coverage["passed"] else "blocked_data"
    expected_member_rows = int(
        membership.loc[:, ["date", "symbol"]].drop_duplicates().shape[0]
    )
    source = {
        "schema_version": 1,
        **dict(source_metadata),
        "status": status,
        "reason_code": None if status == "pass" else coverage["reason_code"],
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "effective_date_rule": "max_notice_or_update_plus_one_trading_day",
        "revision_policy": (
            "current statement values are not available before the last source update date"
        ),
        "derived_field_rules": {
            "market_cap": (
                "vwap=(amount/volume) else close; "
                "vwap*reported_share_capital; else amount/(turn/100); "
                "else vwap*(annualized_parent_netprofit/(basic_eps*annualization_factor))"
            )
        },
        "coverage": coverage,
        "row_counts": {
            "events": int(len(events)),
            "member_market_rows": int(len(market_rows)),
            "expected_member_rows": expected_member_rows,
            "daily_features": int(len(features)),
        },
        "artifacts": {
            "fundamental_events.parquet": _sha256_file(events_path),
            **{
                f"daily/{path.name}": _sha256_file(path) for path in daily_paths
            },
        },
    }
    _atomic_json(source, output / "FUNDAMENTAL_SOURCE.json")
    return source


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            )
        }
    )
    return session


def _retry_source_call(call: Callable[[], pd.DataFrame], *, attempts: int = 4) -> pd.DataFrame:
    error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
            error = exc
            if attempt < attempts:
                time.sleep(min(8.0, 0.75 * (2 ** (attempt - 1))))
    assert error is not None
    raise error


def _load_membership(
    membership_dir: Path, *, start_date: str, end_date: str
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    blocks: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        path = Path(membership_dir) / f"{year}.parquet"
        if not path.is_file():
            raise ValueError(f"missing PIT CSI800 membership file: {path}")
        frame = pd.read_parquet(path, columns=["date", "symbol"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        blocks.append(frame.loc[frame["date"].between(start, end, inclusive="both")])
    membership = pd.concat(blocks, ignore_index=True)
    if membership.empty:
        raise ValueError("PIT CSI800 membership is empty for fundamental data range")
    if membership.duplicated(["date", "symbol"]).any():
        raise ValueError("PIT CSI800 membership contains duplicate date-symbol rows")
    return membership.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _manifest_paths(quant_data_dir: Path) -> tuple[Path, Path]:
    root = Path(quant_data_dir).expanduser().resolve()
    manifest_path = root / "DATA_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    product = payload["official_entries"]["robust_price_product"]

    def resolve_relative(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    return resolve_relative(product["cleaned_parquet"]), resolve_relative(
        product["qlib_provider"]
    )


def _load_calendar(provider_dir: Path, *, end_date: str) -> pd.DatetimeIndex:
    path = Path(provider_dir) / "calendars" / "day.txt"
    values = pd.to_datetime(path.read_text(encoding="utf-8").splitlines(), errors="coerce")
    calendar = pd.DatetimeIndex(values).dropna().normalize().sort_values().unique()
    calendar = calendar[calendar <= pd.Timestamp(end_date).normalize()]
    if calendar.empty:
        raise ValueError("robust provider calendar is empty for Goal 1")
    return calendar


def _download_raw_statements(
    *,
    periods: Sequence[str],
    symbols: set[str],
    cache_dir: Path,
    start_report_date: str,
    end_report_date: str,
    timeout: float,
    progress: Callable[[int, int, str, str], None],
) -> dict[str, pd.DataFrame]:
    allowed = {f"{symbol[2:]}.{symbol[:2]}" for symbol in symbols}
    session = _session()
    blocks: dict[str, list[pd.DataFrame]] = {
        "balance": [],
        "income": [],
        "cashflow": [],
    }
    total = len(periods) * len(blocks)
    completed = 0
    for period in periods:
        for statement in ("income", "balance", "cashflow"):
            if statement == "income":
                def fetcher(period=period):
                    return _retry_source_call(
                        lambda: fetch_income_period(
                            session, report_date=period, timeout=timeout
                        )
                    )
            else:
                def fetcher(statement=statement, period=period):
                    return _retry_source_call(
                        lambda: fetch_summary_period(
                            session,
                            statement=statement,
                            report_date=period,
                            timeout=timeout,
                        )
                    )
            frame = load_or_fetch_period(
                cache_dir=cache_dir,
                statement=statement,
                report_date=period,
                fetcher=fetcher,
            )
            if not frame.empty and "SECUCODE" in frame.columns:
                frame = frame.loc[frame["SECUCODE"].astype(str).str.upper().isin(allowed)]
            blocks[statement].append(frame.reset_index(drop=True))
            completed += 1
            progress(completed, total, statement, period)
    combined = {
        statement: pd.concat(frames, ignore_index=True)
        for statement, frames in blocks.items()
    }
    for statement in ("income",):
        def fetch_batch(
            batch: Sequence[str], *, statement: str = statement
        ) -> pd.DataFrame:
            return load_or_fetch_symbol_batch(
                cache_dir=cache_dir,
                statement=statement,
                symbols=batch,
                start_report_date=start_report_date,
                end_report_date=end_report_date,
                fetcher=lambda: _retry_source_call(
                    lambda: fetch_statement_batch(
                        session,
                        statement=statement,
                        symbols=batch,
                        start_report_date=start_report_date,
                        end_report_date=end_report_date,
                        timeout=timeout,
                    )
                ),
            )

        combined[statement], supplemented = supplement_missing_symbols(
            combined[statement],
            expected_symbols=sorted(symbols),
            fetcher=fetch_batch,
        )
        if supplemented:
            print(
                f"[supplement] {statement}: requested {len(supplemented)} historical symbols",
                flush=True,
            )
    for statement in ("balance", "cashflow"):
        def fetch_history_batch(
            batch: Sequence[str], *, statement: str = statement
        ) -> pd.DataFrame:
            return load_or_fetch_symbol_batch(
                cache_dir=cache_dir,
                statement=statement,
                symbols=batch,
                start_report_date=start_report_date,
                end_report_date=end_report_date,
                fetcher=lambda: _retry_source_call(
                    lambda: fetch_statement_batch(
                        session,
                        statement=statement,
                        symbols=batch,
                        start_report_date=start_report_date,
                        end_report_date=end_report_date,
                        timeout=timeout,
                    )
                ),
            )

        history = load_symbol_history(
            sorted(symbols),
            fetcher=fetch_history_batch,
        )
        if history.empty:
            raise RuntimeError(f"EastMoney {statement} symbol history returned no data")
        combined[statement] = merge_statement_history(combined[statement], history)
        print(
            f"[symbol-history] {statement}: {history['SECUCODE'].nunique()} symbols, "
            f"{len(history)} rows",
            flush=True,
        )
    return combined


def run_download(
    *,
    quant_data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    membership_dir: Path,
    price_dir: Path,
    provider_dir: Path,
    data_start: str,
    data_end: str,
    report_start: str,
    report_end: str,
    coverage_minimum: float,
    timeout: float,
    limit_symbols: int,
    status_path: Path,
) -> dict[str, Any]:
    validate_goal1_dates(
        data_start=data_start,
        data_end=data_end,
        report_start=report_start,
        report_end=report_end,
    )
    membership = _load_membership(
        membership_dir, start_date=data_start, end_date=data_end
    )
    symbols = sorted(membership["symbol"].astype(str).str.upper().unique())
    if limit_symbols > 0:
        symbols = symbols[: int(limit_symbols)]
        membership = membership.loc[membership["symbol"].isin(symbols)].reset_index(drop=True)
    periods = report_periods(report_start, report_end)
    run_id = f"robust_fundamentals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    status: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "current_stage": "download",
        "completed_count": 0,
        "expected_count": len(periods) * 3,
        "symbol_count": len(symbols),
        "data_range": {"start": data_start, "end": data_end},
        "report_range": {"start": report_start, "end": report_end},
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_json(status, status_path)

    def progress(completed: int, total: int, statement: str, period: str) -> None:
        status.update(
            {
                "completed_count": completed,
                "expected_count": total,
                "current_statement": statement,
                "current_report_period": period,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        _atomic_json(status, status_path)
        if completed <= 3 or completed % 12 == 0 or completed == total:
            print(f"[{completed}/{total}] {period} {statement}", flush=True)

    raw = _download_raw_statements(
        periods=periods,
        symbols=set(symbols),
        cache_dir=cache_dir,
        start_report_date=report_start,
        end_report_date=report_end,
        timeout=timeout,
        progress=progress,
    )
    status.update(
        {
            "current_stage": "build_daily_pit_features",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    _atomic_json(status, status_path)
    calendar = _load_calendar(provider_dir, end_date=data_end)
    source = build_fundamental_product(
        raw_statements=raw,
        calendar=calendar,
        membership=membership,
        price_dir=price_dir,
        output_dir=output_dir,
        coverage_minimum=coverage_minimum,
        source_metadata={
            "requested_source": "jqdata_then_eastmoney",
            "selected_source": "eastmoney_hsf10",
            "fallback_reason": "jqdata_historical_coverage_insufficient",
            "source_endpoints": [EASTMONEY_URL, EASTMONEY_SUMMARY_URL],
            "source_routes": [
                "income_period_history_with_missing_symbol_supplement",
                "balance_and_cashflow_historical_symbol_batches",
            ],
            "data_range": {"start": data_start, "end": data_end},
            "report_range": {"start": report_start, "end": report_end},
            "symbol_count": len(symbols),
            "mixed_sources": False,
        },
    )
    status.update(
        {
            "status": source["status"],
            "reason_code": source.get("reason_code"),
            "current_stage": "completed",
            "coverage": source["coverage"],
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    _atomic_json(status, status_path)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    quant_data = quant_data_root().expanduser().resolve()
    default_price, default_provider = _manifest_paths(quant_data)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=quant_data / "cache" / "robust_fundamentals" / "v1",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=quant_data / "reference" / "fundamentals"
    )
    parser.add_argument(
        "--membership-dir",
        type=Path,
        default=quant_data / "reference" / "pit_csi800" / "daily",
    )
    parser.add_argument("--price-dir", type=Path, default=default_price)
    parser.add_argument("--provider-dir", type=Path, default=default_provider)
    parser.add_argument("--data-start", default="2014-01-02")
    parser.add_argument("--data-end", default="2025-12-31")
    parser.add_argument("--report-start", default="2010-01-01")
    parser.add_argument("--report-end", default="2025-12-31")
    parser.add_argument("--coverage-minimum", type=float, default=0.98)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=(
            reports_root()
            / "robust_signal"
            / "fundamentals"
            / "download_robust_fundamentals_latest.json"
        ),
    )
    args = parser.parse_args(argv)
    try:
        status = run_download(
            quant_data_dir=args.quant_data_dir.expanduser().resolve(),
            cache_dir=args.cache_dir.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            membership_dir=args.membership_dir.expanduser().resolve(),
            price_dir=args.price_dir.expanduser().resolve(),
            provider_dir=args.provider_dir.expanduser().resolve(),
            data_start=args.data_start,
            data_end=args.data_end,
            report_start=args.report_start,
            report_end=args.report_end,
            coverage_minimum=args.coverage_minimum,
            timeout=args.timeout,
            limit_symbols=args.limit_symbols,
            status_path=args.status_path.expanduser().resolve(),
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if status["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
