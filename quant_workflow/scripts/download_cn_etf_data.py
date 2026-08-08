#!/usr/bin/env python3
"""Build a governed, resumable mainland-listed ETF raw/qfq dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file  # noqa: E402
from _paths import quant_data_root  # noqa: E402
from download_etf_momentum_data import (  # noqa: E402
    KNOWN_CORPORATE_ACTIONS,
    TENCENT_ADJUSTED_SOURCE_NAME,
    TENCENT_KLINE_URL,
    TENCENT_RAW_KLINE_URL,
    TENCENT_RAW_SOURCE_NAME,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    finalize_paired_price_snapshot,
    publish_snapshot_group,
    resolve_dataset,
    sha256_file,
)


DATASET_ID = "cn_etf_daily"
DEFAULT_START_DATE = "2004-01-01"
DEFAULT_PAGE_SIZE = 640
DEFAULT_LIMIT = 51
DEFAULT_REQUEST_INTERVAL = 0.3
DEFAULT_REQUEST_BUDGET = 5_000
DEFAULT_INCREMENTAL_REQUEST_BUDGET = 250
DEFAULT_INCREMENTAL_PAGE_SIZE = 64
DEFAULT_REQUIRED_CODES = ("512890", "513010", "510720", "159915", "513100")
DEFAULT_SNAPSHOT_RETENTION = 2
DEFAULT_INCREMENTAL_OVERLAP_DAYS = 14
MIN_REQUEST_INTERVAL = 0.2
MAX_CONSECUTIVE_FAILURES = 3
PRODUCER_VERSION = "2"
AFFINE_RESIDUAL_TOLERANCE = 0.0015
AFFINE_NEAR_TIE_TOLERANCE = 0.00015
AFFINE_FIT_WINDOWS = (10, 7, 5, 3, 2, 1)


class CnEtfDownloadError(RuntimeError):
    """The common ETF dataset could not satisfy its publication contract."""


class HistoricalRevisionDetected(CnEtfDownloadError):
    """A bounded overlap proves one symbol needs a targeted history refresh."""


@dataclass
class RequestGate:
    interval: float
    budget: int
    calls: int = 0
    last_call: float = 0.0

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.calls >= self.budget:
            raise CnEtfDownloadError("request budget exhausted")
        wait = self.interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.calls += 1
        try:
            return function(*args, **kwargs)
        finally:
            self.last_call = time.monotonic()


def _as_date(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value).normalize()
    if pd.isna(result):
        raise ValueError(f"invalid date: {value}")
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_sina_symbol(value: object) -> tuple[str, str, str] | None:
    text = str(value or "").strip().lower()
    if len(text) != 8 or text[:2] not in {"sh", "sz"} or not text[2:].isdigit():
        return None
    market = text[:2]
    code = text[2:]
    return f"{market.upper()}{code}", code, market


def _market_for_code(code: str) -> str:
    if code.startswith("1"):
        return "sz"
    if code.startswith("5"):
        return "sh"
    raise CnEtfDownloadError(f"cannot infer ETF market for code {code}")


def build_etf_universe(
    sina: pd.DataFrame,
    ths: pd.DataFrame,
    *,
    through: str,
    recent_listing_days: int = 10,
) -> pd.DataFrame:
    """Reconcile current Sina ETFs with recently traded THS-only listings."""

    if recent_listing_days < 0:
        raise ValueError("recent_listing_days must be nonnegative")
    required_sina = {"代码", "名称"}
    required_ths = {"基金代码", "基金名称", "最新-交易日"}
    if not required_sina.issubset(sina.columns):
        raise CnEtfDownloadError("Sina ETF catalog columns are incomplete")
    if not required_ths.issubset(ths.columns):
        raise CnEtfDownloadError("THS ETF catalog columns are incomplete")

    records: dict[str, dict[str, Any]] = {}
    for row in sina.to_dict(orient="records"):
        parsed = _canonical_sina_symbol(row.get("代码"))
        if parsed is None:
            continue
        symbol, code, market = parsed
        amount = pd.to_numeric(row.get("成交额"), errors="coerce")
        records[symbol] = {
            "symbol": symbol,
            "code": code,
            "market": market,
            "name": str(row.get("名称") or code).strip(),
            "sina_code": f"{market}{code}",
            "in_sina": True,
            "in_ths": False,
            "ths_latest_trade_date": None,
            "sina_amount": float(amount) if pd.notna(amount) else 0.0,
        }

    through_date = _as_date(through)
    recent_cutoff = through_date - pd.Timedelta(days=recent_listing_days)
    for row in ths.to_dict(orient="records"):
        code = str(row.get("基金代码") or "").strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        latest = pd.to_datetime(row.get("最新-交易日"), errors="coerce")
        market = _market_for_code(code)
        symbol = f"{market.upper()}{code}"
        if symbol in records:
            records[symbol]["in_ths"] = True
            records[symbol]["ths_latest_trade_date"] = (
                latest.date().isoformat() if pd.notna(latest) else None
            )
            continue
        if pd.isna(latest) or latest < recent_cutoff or latest > through_date:
            continue
        records[symbol] = {
            "symbol": symbol,
            "code": code,
            "market": market,
            "name": str(row.get("基金名称") or code).strip(),
            "sina_code": None,
            "in_sina": False,
            "in_ths": True,
            "ths_latest_trade_date": latest.date().isoformat(),
            "sina_amount": 0.0,
        }

    if not records:
        raise CnEtfDownloadError("ETF catalog reconciliation returned no symbols")
    return pd.DataFrame(records.values()).sort_values("symbol").reset_index(drop=True)


def select_trial_universe(
    universe: pd.DataFrame,
    *,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
) -> pd.DataFrame:
    """Select a liquid trial set while retaining user-required ETF codes."""

    if limit < 1 or limit > 2_000:
        raise ValueError("ETF universe limit must be between 1 and 2000")
    required = tuple(dict.fromkeys(str(code).zfill(6) for code in required_codes))
    if len(required) > limit:
        raise ValueError("required ETF count exceeds the universe limit")
    by_code = universe.drop_duplicates("code", keep="first").set_index("code")
    missing = [code for code in required if code not in by_code.index]
    if missing:
        raise CnEtfDownloadError(
            "required ETFs are missing from the current catalog: " + ", ".join(missing)
        )

    ranked = universe.copy()
    ranked["sina_amount"] = pd.to_numeric(
        ranked.get("sina_amount"), errors="coerce"
    ).fillna(0.0)
    ranked = ranked.sort_values(
        ["sina_amount", "symbol"], ascending=[False, True], kind="stable"
    )
    ordinary = ranked.loc[~ranked["code"].isin(required)].head(limit - len(required))
    mandatory = ranked.loc[ranked["code"].isin(required)]
    selected = pd.concat([ordinary, mandatory], ignore_index=True)
    if len(selected) != limit or selected["symbol"].duplicated().any():
        raise CnEtfDownloadError("ETF trial universe could not reach the requested size")
    selected["selection_reason"] = "top_turnover"
    selected.loc[selected["code"].isin(required), "selection_reason"] = "required"
    selected = selected.sort_values(
        ["sina_amount", "symbol"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    return selected


def fetch_etf_universe(
    *,
    through: str,
    request_gate: RequestGate,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak

    sina = request_gate.call(ak.fund_etf_category_sina, symbol="ETF基金")
    ths = request_gate.call(ak.fund_etf_category_ths, symbol="ETF")
    if sina is None or sina.empty or ths is None or ths.empty:
        raise CnEtfDownloadError("ETF catalog source returned no rows")
    reconciled = build_etf_universe(sina, ths, through=through)
    selected = select_trial_universe(
        reconciled,
        limit=limit,
        required_codes=required_codes,
    )
    return selected, {
        "sina_count": int(len(sina)),
        "ths_count": int(len(ths)),
        "reconciled_count": int(len(reconciled)),
        "selected_count": int(len(selected)),
        "selection_method": "sina_turnover_top_n_plus_required_codes",
        "required_codes": list(required_codes),
    }


def _request_tencent_page(
    *,
    session: Any,
    request_gate: RequestGate,
    url: str,
    ticker: str,
    response_key: str,
    param: str,
    timeout: float,
    retries: int,
) -> tuple[list[list[object]], str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = request_gate.call(
                session.get,
                url,
                params={"param": param},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            ticker_data = data.get(ticker) if isinstance(data, dict) else None
            response_keys = (
                ("qfqday", "day") if response_key == "qfqday" else ("day",)
            )
            selected_key = next(
                (
                    key
                    for key in response_keys
                    if isinstance(ticker_data, dict)
                    and isinstance(ticker_data.get(key), list)
                ),
                None,
            )
            rows = ticker_data.get(selected_key) if selected_key else None
            if payload.get("code") != 0 or not isinstance(rows, list) or not selected_key:
                raise CnEtfDownloadError(
                    f"Tencent response is invalid: {payload.get('msg') if isinstance(payload, dict) else 'not an object'}"
                )
            return rows, selected_key
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(float(attempt + 1))
    raise CnEtfDownloadError(
        f"Tencent request failed: {type(last_error).__name__}: {last_error}"
    ) from last_error


def fetch_tencent_history(
    symbol: str,
    metadata: dict[str, str],
    *,
    adjustment: str,
    start_date: str,
    end_date: str,
    session: Any,
    request_gate: RequestGate,
    page_size: int = DEFAULT_PAGE_SIZE,
    retries: int = 2,
    timeout: float = 20.0,
) -> pd.DataFrame:
    if adjustment not in {"raw", "qfq"}:
        raise ValueError("Tencent adjustment must be raw or qfq")
    if page_size < 1 or page_size > DEFAULT_PAGE_SIZE:
        raise ValueError(f"Tencent page_size must be between 1 and {DEFAULT_PAGE_SIZE}")
    if retries < 1 or retries > 2:
        raise ValueError("Tencent retries must be between 1 and 2")
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")

    ticker = f"{metadata['market']}{metadata['code']}"
    response_key = "day" if adjustment == "raw" else "qfqday"
    suffix = "" if adjustment == "raw" else ",qfq"
    url = TENCENT_RAW_KLINE_URL if adjustment == "raw" else TENCENT_KLINE_URL
    page_end = end
    collected: list[list[object]] = []
    response_keys_seen: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for _page in range(32):
        param = f"{ticker},day,,{page_end.date().isoformat()},{page_size}{suffix}"
        rows, response_key_used = _request_tencent_page(
            session=session,
            request_gate=request_gate,
            url=url,
            ticker=ticker,
            response_key=response_key,
            param=param,
            timeout=timeout,
            retries=retries,
        )
        response_keys_seen.add(response_key_used)
        normalized_rows = [row[:6] for row in rows if isinstance(row, list) and len(row) >= 6]
        if not normalized_rows:
            if collected:
                break
            raise CnEtfDownloadError(f"Tencent returned no {adjustment} rows for {symbol}")
        dates = pd.to_datetime([row[0] for row in normalized_rows], errors="coerce")
        if dates.isna().any():
            raise CnEtfDownloadError(f"Tencent returned invalid dates for {symbol}")
        oldest = pd.Timestamp(dates.min()).normalize()
        if previous_oldest is not None and oldest >= previous_oldest:
            raise CnEtfDownloadError(f"Tencent pagination did not move backwards for {symbol}")
        collected.extend(normalized_rows)
        if oldest <= start or len(normalized_rows) < page_size:
            break
        previous_oldest = oldest
        page_end = oldest - pd.Timedelta(days=1)
    else:
        raise CnEtfDownloadError(f"Tencent pagination limit exceeded for {symbol}")

    frame = pd.DataFrame(
        collected,
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame = frame.loc[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.sort_values("date", kind="stable").drop_duplicates(
        "date", keep="last"
    ).reset_index(drop=True)
    if frame.empty:
        raise CnEtfDownloadError(f"no Tencent {adjustment} rows remain for {symbol}")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = (
        TENCENT_RAW_SOURCE_NAME
        if adjustment == "raw"
        else TENCENT_ADJUSTED_SOURCE_NAME
    )
    if adjustment == "qfq":
        frame["adjustment"] = "qfq"
        frame["provider_response_key"] = ",".join(sorted(response_keys_seen))
    return frame


def fetch_tencent_pair(
    symbol: str,
    metadata: dict[str, str],
    *,
    start_date: str,
    end_date: str,
    session: Any,
    request_gate: RequestGate,
    page_size: int = DEFAULT_PAGE_SIZE,
    retries: int = 2,
    timeout: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_tencent_history(
        symbol,
        metadata,
        adjustment="raw",
        start_date=start_date,
        end_date=end_date,
        session=session,
        request_gate=request_gate,
        page_size=page_size,
        retries=retries,
        timeout=timeout,
    )
    qfq = fetch_tencent_history(
        symbol,
        metadata,
        adjustment="qfq",
        start_date=start_date,
        end_date=end_date,
        session=session,
        request_gate=request_gate,
        page_size=page_size,
        retries=retries,
        timeout=timeout,
    )
    build_paired_adjustments(symbol, raw, qfq)
    return raw, qfq


def build_paired_adjustments(
    symbol: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    """Derive and validate one affine OHLC transform for every trade date.

    Tencent represents share splits with a multiplicative scale and some cash
    distributions with an additive offset. Both are needed to reproduce all
    four adjusted OHLC fields from the paired raw response.
    """

    raw_frame = raw.copy()
    qfq_frame = adjusted.copy()
    raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
    qfq_frame["date"] = pd.to_datetime(qfq_frame["date"], errors="coerce").dt.normalize()
    if (
        raw_frame["date"].isna().any()
        or qfq_frame["date"].isna().any()
        or raw_frame["date"].duplicated().any()
        or qfq_frame["date"].duplicated().any()
        or set(raw_frame["date"]) != set(qfq_frame["date"])
    ):
        raise DataContractError(f"raw/qfq dates differ for {symbol}")
    fields = ("open", "high", "low", "close")
    for field in fields:
        if field not in raw_frame or field not in qfq_frame:
            raise DataContractError(f"raw/qfq {field} field is missing for {symbol}")
    paired = raw_frame[["date", *fields]].merge(
        qfq_frame[["date", *fields]],
        on="date",
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", "_qfq"),
    ).sort_values("date", kind="stable")

    raw_values = np.column_stack(
        [pd.to_numeric(paired[f"{field}_raw"], errors="coerce") for field in fields]
    )
    qfq_values = np.column_stack(
        [pd.to_numeric(paired[f"{field}_qfq"], errors="coerce") for field in fields]
    )
    if (
        not np.isfinite(raw_values).all()
        or not np.isfinite(qfq_values).all()
        or (raw_values <= 0).any()
        or (qfq_values <= 0).any()
    ):
        raise DataContractError(f"raw/qfq OHLC values are invalid for {symbol}")

    def residual(values: np.ndarray, targets: np.ndarray, scale: float, offset: float) -> float:
        return float(np.max(np.abs(targets - (values * scale + offset))))

    def fit_window(start: int, stop: int) -> tuple[float, float, str] | None:
        values = raw_values[start:stop].reshape(-1)
        targets = qfq_values[start:stop].reshape(-1)
        candidates: list[tuple[float, int, float, float, str]] = []

        def add_candidate(scale: float, offset: float, priority: int, method: str) -> None:
            if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
                return
            candidates.append(
                (
                    residual(values, targets, float(scale), float(offset)),
                    priority,
                    float(scale),
                    float(offset),
                    method,
                )
            )

        add_candidate(1.0, 0.0, 0, "identity")
        add_candidate(1.0, float(np.median(targets - values)), 1, "offset")
        add_candidate(float(np.median(targets / values)), 0.0, 2, "scale")
        simple = [
            candidate
            for candidate in candidates
            if candidate[0] <= AFFINE_RESIDUAL_TOLERANCE
        ]
        if simple:
            best_residual = min(candidate[0] for candidate in simple)
            near_best = [
                candidate
                for candidate in simple
                if candidate[0] <= best_residual + AFFINE_NEAR_TIE_TOLERANCE
            ]
            _error, _priority, scale, offset, method = min(
                near_best, key=lambda candidate: candidate[1]
            )
            return scale, offset, method

        if float(np.ptp(values)) <= 1e-9:
            return None
        design = np.column_stack([values, np.ones(len(values))])
        scale, offset = np.linalg.lstsq(design, targets, rcond=None)[0]
        affine_residual = residual(values, targets, float(scale), float(offset))
        if (
            not np.isfinite(scale)
            or not np.isfinite(offset)
            or scale <= 0
            or affine_residual > AFFINE_RESIDUAL_TOLERANCE
        ):
            return None
        return float(scale), float(offset), "affine"

    latest_residual = residual(raw_values[-1], qfq_values[-1], 1.0, 0.0)
    if latest_residual > AFFINE_RESIDUAL_TOLERANCE:
        raise DataContractError(f"raw/qfq latest transform is not normalized for {symbol}")

    row_count = len(paired)
    assigned_scales = np.empty(row_count, dtype=float)
    assigned_offsets = np.empty(row_count, dtype=float)
    assigned_residuals = np.empty(row_count, dtype=float)
    reverse_segment_ids = np.empty(row_count, dtype=int)
    assigned_methods: list[str] = [""] * row_count
    scale = 1.0
    offset = 0.0
    method = "identity"
    reverse_segment_id = 0

    for position in range(row_count - 1, -1, -1):
        row_residual = residual(raw_values[position], qfq_values[position], scale, offset)
        if row_residual > AFFINE_RESIDUAL_TOLERANCE:
            fitted: tuple[float, float, str] | None = None
            attempted_starts: set[int] = set()
            for window in AFFINE_FIT_WINDOWS:
                start = max(0, position - window + 1)
                if start in attempted_starts:
                    continue
                attempted_starts.add(start)
                fitted = fit_window(start, position + 1)
                if fitted is not None:
                    break
            if fitted is None:
                raise DataContractError(
                    f"raw/qfq affine OHLC relationship is invalid for {symbol}"
                )
            scale, offset, method = fitted
            reverse_segment_id += 1
            row_residual = residual(
                raw_values[position], qfq_values[position], scale, offset
            )
        assigned_scales[position] = scale
        assigned_offsets[position] = offset
        assigned_residuals[position] = row_residual
        reverse_segment_ids[position] = reverse_segment_id
        assigned_methods[position] = method

    if float(np.max(assigned_residuals)) > AFFINE_RESIDUAL_TOLERANCE:
        raise DataContractError(f"raw/qfq affine OHLC relationship is invalid for {symbol}")
    segment_ids = int(reverse_segment_ids.max()) - reverse_segment_ids
    result = pd.DataFrame(
        {
            "date": paired["date"].to_numpy(),
            "factor": assigned_scales,
            "offset": assigned_offsets,
            "segment_id": segment_ids,
            "segment_method": assigned_methods,
            "max_ohlc_residual": assigned_residuals,
            "factor_source": "derived_same_provider_raw_qfq_piecewise_affine_ohlc",
            "adjustment_formula": "qfq_ohlc=raw_ohlc*factor+offset",
        }
    )
    return result


def _corporate_actions_from_adjustments(
    adjustments_by_symbol: dict[str, pd.DataFrame],
    *,
    provider: str,
) -> pd.DataFrame:
    known_dates = {
        (item["symbol"], pd.Timestamp(item["resumption_date"]))
        for item in KNOWN_CORPORATE_ACTIONS
    }
    rows: list[dict[str, Any]] = []
    for symbol, adjustments in adjustments_by_symbol.items():
        ordered = adjustments.sort_values("date", kind="stable").reset_index(drop=True)
        scales = pd.to_numeric(ordered["factor"], errors="coerce")
        offsets = pd.to_numeric(ordered["offset"], errors="coerce")
        scale_ratio = scales / scales.shift(1)
        offset_change = offsets - offsets.shift(1)
        if "segment_id" in ordered.columns:
            segments = pd.to_numeric(ordered["segment_id"], errors="coerce")
            changed = segments.ne(segments.shift(1)) & segments.shift(1).notna()
        else:
            changed = ((scale_ratio - 1.0).abs() >= 0.005) | (
                offset_change.abs() >= 0.005
            )
        for position in ordered.index[changed.fillna(False)]:
            effective = pd.Timestamp(ordered.loc[position, "date"])
            if (symbol, effective) in known_dates:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "record_date": pd.NaT,
                    "ex_date": effective,
                    "effective_trade_date": effective,
                    "resumption_date": effective,
                    "action_type": "provider_adjustment_event",
                    "ratio": float(scale_ratio.loc[position]),
                    "offset_change": float(offset_change.loc[position]),
                    "factor_before": float(scales.loc[position - 1]),
                    "factor_after": float(scales.loc[position]),
                    "offset_before": float(offsets.loc[position - 1]),
                    "offset_after": float(offsets.loc[position]),
                    "evidence_method": "derived_from_same_provider_raw_qfq_affine_pair",
                    "source": f"{provider}.paired_raw_qfq",
                    "source_url": None,
                }
            )
    for item in KNOWN_CORPORATE_ACTIONS:
        symbol = str(item["symbol"])
        if symbol not in adjustments_by_symbol:
            continue
        adjustments = adjustments_by_symbol[symbol].sort_values("date", kind="stable")
        effective = pd.Timestamp(item["resumption_date"])
        before = adjustments.loc[adjustments["date"] < effective]
        after = adjustments.loc[adjustments["date"] >= effective]
        rows.append(
            {
                "symbol": symbol,
                "record_date": pd.Timestamp(item["record_date"]),
                "ex_date": pd.Timestamp(item["split_date"]),
                "effective_trade_date": effective,
                "resumption_date": effective,
                "action_type": item["type"],
                "ratio": 5.0,
                "offset_change": (
                    float(after.iloc[0]["offset"] - before.iloc[-1]["offset"])
                    if not before.empty and not after.empty
                    else None
                ),
                "factor_before": (
                    float(before.iloc[-1]["factor"]) if not before.empty else None
                ),
                "factor_after": (
                    float(after.iloc[0]["factor"]) if not after.empty else None
                ),
                "offset_before": (
                    float(before.iloc[-1]["offset"]) if not before.empty else None
                ),
                "offset_after": (
                    float(after.iloc[0]["offset"]) if not after.empty else None
                ),
                "evidence_method": "announcement_and_same_provider_raw_qfq_affine_pair",
                "source": item["source"],
                "source_url": item["source_url"],
            }
        )
    columns = [
        "symbol",
        "record_date",
        "ex_date",
        "effective_trade_date",
        "resumption_date",
        "action_type",
        "ratio",
        "offset_change",
        "factor_before",
        "factor_after",
        "offset_before",
        "offset_after",
        "evidence_method",
        "source",
        "source_url",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for field in ("record_date", "ex_date", "effective_trade_date", "resumption_date"):
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    return frame.sort_values(
        ["symbol", "effective_trade_date"], kind="stable"
    ).reset_index(drop=True)


def _universe_hash(universe: pd.DataFrame) -> str:
    stable = universe.sort_values("symbol").fillna("").to_dict(orient="records")
    encoded = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        try:
            lock_file(handle, exclusive=True, blocking=False)
            acquired = True
        except BlockingIOError as exc:
            raise CnEtfDownloadError("another common ETF update is already running") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if acquired:
                unlock_file(handle)
        finally:
            handle.close()


def _cache_symbol_complete(cache: Path, symbol: str) -> bool:
    paths = {
        key: cache / key / f"{symbol}.parquet"
        for key in ("raw", "qfq", "factors")
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        raw = pd.read_parquet(paths["raw"])
        qfq = pd.read_parquet(paths["qfq"])
        recorded = pd.read_parquet(paths["factors"])
        if "provider_response_key" not in qfq.columns:
            return False
        required_factor_columns = {
            "date",
            "factor",
            "offset",
            "segment_id",
            "segment_method",
            "factor_source",
            "adjustment_formula",
        }
        if not required_factor_columns.issubset(recorded.columns):
            return False
        rebuilt = build_paired_adjustments(symbol, raw, qfq)
    except Exception:
        return False
    return (
        not raw.empty
        and len(recorded) == len(rebuilt)
        and recorded["date"].tolist() == rebuilt["date"].tolist()
        and pd.Series(recorded["factor"]).round(10).tolist()
        == pd.Series(rebuilt["factor"]).round(10).tolist()
        and pd.Series(recorded["offset"]).round(10).tolist()
        == pd.Series(rebuilt["offset"]).round(10).tolist()
        and recorded["segment_id"].tolist() == rebuilt["segment_id"].tolist()
        and recorded["segment_method"].tolist()
        == rebuilt["segment_method"].tolist()
        and recorded["factor_source"].tolist()
        == rebuilt["factor_source"].tolist()
        and recorded["adjustment_formula"].tolist()
        == rebuilt["adjustment_formula"].tolist()
    )


def _upgrade_cached_adjustment_evidence(cache: Path, symbol: str) -> bool:
    """Rebuild obsolete factor evidence from an already downloaded raw/qfq pair."""

    raw_path = cache / "raw" / f"{symbol}.parquet"
    qfq_path = cache / "qfq" / f"{symbol}.parquet"
    if not raw_path.is_file() or not qfq_path.is_file():
        return False
    try:
        raw = pd.read_parquet(raw_path)
        qfq = pd.read_parquet(qfq_path)
        if "provider_response_key" not in qfq.columns:
            return False
        rebuilt = build_paired_adjustments(symbol, raw, qfq)
        _atomic_parquet(rebuilt, cache / "factors" / f"{symbol}.parquet")
    except Exception:
        return False
    return True


def _normalize_component_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    component: str,
) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataContractError(f"{component} history is incomplete for {symbol}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="coerce"
    ).dt.normalize()
    if normalized["date"].isna().any() or normalized["date"].duplicated().any():
        raise DataContractError(f"{component} dates are invalid for {symbol}")
    if set(normalized["symbol"].astype(str)) != {symbol}:
        raise DataContractError(f"{component} symbol is inconsistent for {symbol}")
    return normalized.sort_values("date", kind="stable").reset_index(drop=True)


def _overlap_values_match(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> bool:
    fields = ("open", "high", "low", "close", "volume")
    left = current.set_index("date").loc[dates, fields]
    right = incoming.set_index("date").loc[dates, fields]
    left_values = left.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right_values = right.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(
        np.allclose(
            left_values,
            right_values,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
    )


def merge_incremental_pair(
    symbol: str,
    current_raw: pd.DataFrame,
    current_qfq: pd.DataFrame,
    incoming_raw: pd.DataFrame,
    incoming_qfq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Append a bounded provider pair after proving its overlap is unchanged."""

    old_raw = _normalize_component_frame(
        current_raw, symbol=symbol, component="current raw"
    )
    old_qfq = _normalize_component_frame(
        current_qfq, symbol=symbol, component="current qfq"
    )
    new_raw = _normalize_component_frame(
        incoming_raw, symbol=symbol, component="incremental raw"
    )
    new_qfq = _normalize_component_frame(
        incoming_qfq, symbol=symbol, component="incremental qfq"
    )
    if old_raw["date"].tolist() != old_qfq["date"].tolist():
        raise DataContractError(f"current raw/qfq dates differ for {symbol}")
    if new_raw["date"].tolist() != new_qfq["date"].tolist():
        raise DataContractError(f"incremental raw/qfq dates differ for {symbol}")

    old_dates = pd.DatetimeIndex(old_raw["date"])
    incoming_dates = pd.DatetimeIndex(new_raw["date"])
    overlap = old_dates.intersection(incoming_dates)
    if overlap.empty:
        raise HistoricalRevisionDetected(
            f"incremental response has no overlap for {symbol}"
        )
    if not _overlap_values_match(old_raw, new_raw, overlap):
        raise HistoricalRevisionDetected(
            f"provider revised raw history for {symbol}"
        )
    if not _overlap_values_match(old_qfq, new_qfq, overlap):
        raise HistoricalRevisionDetected(
            f"provider revised qfq history for {symbol}"
        )

    old_end = old_dates.max()
    append_raw = new_raw.loc[new_raw["date"] > old_end]
    append_qfq = new_qfq.loc[new_qfq["date"] > old_end]
    if append_raw["date"].tolist() != append_qfq["date"].tolist():
        raise DataContractError(f"incremental raw/qfq additions differ for {symbol}")
    if append_raw.empty:
        return old_raw, old_qfq, 0
    combined_raw = pd.concat([old_raw, append_raw], ignore_index=True)
    combined_qfq = pd.concat([old_qfq, append_qfq], ignore_index=True)
    return combined_raw, combined_qfq, int(len(append_raw))


def _link_or_copy(source: Path, destination: Path) -> None:
    """Copy a snapshot file without sharing its inode with another snapshot.

    Snapshot retention relies on the previous snapshot being an independent
    rollback point. Hard links make a later accidental in-place write mutate
    both the current snapshot and its backup, so ordinary updates always use a
    real copy here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_incremental_status(root: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(
        root / "runtime" / "status" / "cn_etf_daily_update_latest.json",
        payload,
    )


def _sync_root_manifest(root: Path) -> None:
    from sync_data_manifest import sync_data_manifest

    sync_data_manifest(quant_data_root=root)


def prune_snapshot_history(
    dataset_root: Path,
    *,
    keep: int = DEFAULT_SNAPSHOT_RETENTION,
) -> list[str]:
    """Keep the current complete snapshot and at most one previous snapshot."""

    if keep < 1 or keep > 2:
        raise ValueError("ETF snapshot retention must be 1 or 2")
    dataset_root = Path(dataset_root).expanduser()
    if dataset_root.is_symlink():
        raise CnEtfDownloadError("ETF dataset root cannot be a symlink")
    dataset_root = dataset_root.resolve()
    pointer_path = dataset_root / "LATEST.json"
    snapshots_root = dataset_root / "snapshots"
    if pointer_path.is_symlink() or snapshots_root.is_symlink():
        raise CnEtfDownloadError("ETF snapshot retention paths cannot be symlinks")
    if not pointer_path.is_file() or not snapshots_root.is_dir():
        raise CnEtfDownloadError("ETF snapshot retention state is incomplete")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CnEtfDownloadError("ETF latest pointer is unreadable") from exc
    current = str(pointer.get("snapshot_id") or "")
    records: list[tuple[str, str, Path]] = []
    for path in sorted(snapshots_root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise CnEtfDownloadError("ETF snapshots directory contains an unsafe entry")
        manifest_path = path / "DATASET_MANIFEST.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CnEtfDownloadError(
                f"ETF snapshot manifest is unreadable: {path.name}"
            ) from exc
        if str(manifest.get("snapshot_id") or "") != path.name:
            raise CnEtfDownloadError(
                f"ETF snapshot identity is inconsistent: {path.name}"
            )
        records.append(
            (str(manifest.get("source_retrieved_at") or ""), path.name, path)
        )
    if current not in {snapshot_id for _, snapshot_id, _ in records}:
        raise CnEtfDownloadError("ETF latest pointer target is missing")

    ordered = sorted(records, key=lambda item: (item[0], item[1]), reverse=True)
    retained = {current}
    for _, snapshot_id, _ in ordered:
        if len(retained) >= keep:
            break
        if snapshot_id != current:
            retained.add(snapshot_id)
    removed: list[str] = []
    for _, snapshot_id, path in records:
        if snapshot_id in retained:
            continue
        if path.is_symlink():
            raise CnEtfDownloadError("ETF snapshot retention path became a symlink")
        try:
            path.resolve().relative_to(snapshots_root.resolve())
        except ValueError as exc:
            raise CnEtfDownloadError(
                "ETF snapshot retention path escapes the dataset"
            ) from exc
        shutil.rmtree(path)
        removed.append(snapshot_id)
    return sorted(removed)


def _post_publish_housekeeping(
    root: Path,
    dataset_root: Path,
) -> tuple[list[str], list[str]]:
    """Run non-transactional housekeeping without hiding a committed publish."""

    removed: list[str] = []
    warnings: list[str] = []
    try:
        removed = prune_snapshot_history(
            dataset_root,
            keep=DEFAULT_SNAPSHOT_RETENTION,
        )
    except Exception as exc:
        warnings.append(f"retention cleanup failed: {type(exc).__name__}: {exc}")
    try:
        _sync_root_manifest(root)
    except Exception as exc:
        warnings.append(f"manifest sync failed: {type(exc).__name__}: {exc}")
    for warning in warnings:
        print(f"ETF snapshot was published; {warning}", file=sys.stderr)
    return removed, warnings


def update_cn_etf_snapshot(
    *,
    quant_data_dir: Path | None = None,
    end_date: str | None = None,
    overlap_days: int = DEFAULT_INCREMENTAL_OVERLAP_DAYS,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_INCREMENTAL_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    generated_at: str | None = None,
) -> Path:
    """Reuse the frozen ETF snapshot and fetch only its missing date window."""

    if overlap_days < 1 or overlap_days > 60:
        raise ValueError("overlap_days must be between 1 and 60")
    if request_interval < MIN_REQUEST_INTERVAL:
        raise ValueError(f"request_interval must be at least {MIN_REQUEST_INTERVAL}")
    if request_budget < 1 or request_budget > 500:
        raise ValueError("incremental request_budget must be between 1 and 500")
    if retries < 1 or retries > 2:
        raise ValueError("retries must be between 1 and 2")

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    requested_through = _as_date(
        end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat()
    )
    # The updater is commonly invoked by a daily job on Saturday or Sunday.
    # Those dates cannot have new A-share ETF bars; normalize them before
    # calculating missing symbols so a weekend does not trigger 51 requests.
    through = requested_through
    if through.weekday() >= 5:
        through = through - pd.Timedelta(days=through.weekday() - 4)
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    started = time.monotonic()

    with _run_lock(lock_path):
        resolved = resolve_dataset(
            DATASET_ID,
            quant_data_root=root,
            require_basis="qfq",
        )
        current = resolved.snapshot_path
        manifest = resolved.manifest
        if manifest.get("data_kind") != "paired_daily_bars":
            raise DataContractError("cn_etf_daily is not a paired raw/qfq snapshot")

        universe_path = current / "ETF_UNIVERSE.parquet"
        if not universe_path.is_file():
            raise DataContractError("current ETF universe is unavailable")
        universe = pd.read_parquet(universe_path)
        required_universe_columns = {"symbol", "code", "market", "name"}
        if universe.empty or not required_universe_columns.issubset(universe.columns):
            raise DataContractError("current ETF universe is invalid")
        if universe["symbol"].astype(str).duplicated().any():
            raise DataContractError("current ETF universe contains duplicate symbols")

        qfq_index = pd.read_parquet(current / str(manifest["file_index_path"]))
        indexed_symbols = set(qfq_index["symbol"].astype(str))
        universe_symbols = set(universe["symbol"].astype(str))
        if indexed_symbols != universe_symbols:
            raise DataContractError("current ETF universe and snapshot symbols differ")
        last_by_symbol = {
            str(row.symbol): _as_date(str(row.last_date))
            for row in qfq_index.itertuples(index=False)
        }
        missing_symbols = sorted(
            symbol for symbol, last_date in last_by_symbol.items() if last_date < through
        )
        if not missing_symbols:
            _write_incremental_status(
                root,
                {
                    "status": "already_current",
                    "dataset_id": DATASET_ID,
                    "snapshot_id": current.name,
                    "requested_through": requested_through.date().isoformat(),
                    "through": through.date().isoformat(),
                    "requests_used": 0,
                    "symbols_requested": 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "generated_at": timestamp,
                },
            )
            return current

        gate = RequestGate(interval=request_interval, budget=request_budget)
        import requests

        session = requests.Session()
        session.trust_env = False
        run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        snapshot_id = f"cn-etf-tencent-incremental-{run_id}"
        staging_run = root / "staging" / f"cn-etf-incremental-{run_id}"
        stage = staging_run / snapshot_id
        changed_symbols: list[str] = []
        targeted_full_refresh_symbols: list[str] = []
        rows_added_by_symbol: dict[str, int] = {}
        factors_by_symbol: dict[str, pd.DataFrame] = {}
        metadata_by_symbol = {
            str(row.symbol): row
            for row in universe.itertuples(index=False)
        }
        try:
            for symbol in sorted(universe_symbols):
                current_raw_path = current / "raw" / f"{symbol}.parquet"
                current_qfq_path = current / "qfq" / f"{symbol}.parquet"
                current_factor_path = current / "factors" / f"{symbol}.parquet"
                if not all(
                    path.is_file()
                    for path in (current_raw_path, current_qfq_path, current_factor_path)
                ):
                    raise DataContractError(
                        f"current ETF component is missing for {symbol}"
                    )

                if symbol not in missing_symbols:
                    _link_or_copy(current_raw_path, stage / "raw" / current_raw_path.name)
                    _link_or_copy(current_qfq_path, stage / "qfq" / current_qfq_path.name)
                    _link_or_copy(
                        current_factor_path,
                        stage / "factors" / current_factor_path.name,
                    )
                    factors_by_symbol[symbol] = pd.read_parquet(current_factor_path)
                    continue

                current_raw = pd.read_parquet(current_raw_path)
                current_qfq = pd.read_parquet(current_qfq_path)
                old_start = _as_date(current_raw["date"].min())
                old_end = _as_date(current_raw["date"].max())
                overlap_start = max(
                    old_start,
                    old_end - pd.Timedelta(days=overlap_days),
                )
                row = metadata_by_symbol[symbol]
                metadata = {
                    "code": str(row.code).zfill(6),
                    "market": str(row.market).lower(),
                    "name": str(row.name),
                }
                incoming_raw, incoming_qfq = fetch_tencent_pair(
                    symbol,
                    metadata,
                    start_date=overlap_start.date().isoformat(),
                    end_date=through.date().isoformat(),
                    session=session,
                    request_gate=gate,
                    page_size=DEFAULT_INCREMENTAL_PAGE_SIZE,
                    retries=retries,
                    timeout=timeout,
                )
                try:
                    updated_raw, updated_qfq, rows_added = merge_incremental_pair(
                        symbol,
                        current_raw,
                        current_qfq,
                        incoming_raw,
                        incoming_qfq,
                    )
                except HistoricalRevisionDetected:
                    updated_raw, updated_qfq = fetch_tencent_pair(
                        symbol,
                        metadata,
                        start_date=old_start.date().isoformat(),
                        end_date=through.date().isoformat(),
                        session=session,
                        request_gate=gate,
                        retries=retries,
                        timeout=timeout,
                    )
                    updated_raw = _normalize_component_frame(
                        updated_raw, symbol=symbol, component="targeted full raw"
                    )
                    updated_qfq = _normalize_component_frame(
                        updated_qfq, symbol=symbol, component="targeted full qfq"
                    )
                    if updated_raw["date"].tolist() != updated_qfq["date"].tolist():
                        raise DataContractError(
                            f"targeted raw/qfq dates differ for {symbol}"
                        )
                    if _as_date(updated_raw["date"].min()) > old_start:
                        raise DataContractError(
                            f"targeted history does not cover the old start for {symbol}"
                        )
                    rows_added = int(
                        (updated_raw["date"] > old_end).sum()
                    )
                    targeted_full_refresh_symbols.append(symbol)

                if "source_id" not in updated_raw.columns:
                    updated_raw["source_id"] = updated_raw["source"].astype(str)
                if "source_id" not in updated_qfq.columns:
                    updated_qfq["source_id"] = updated_qfq["source"].astype(str)
                factors = build_paired_adjustments(symbol, updated_raw, updated_qfq)
                if rows_added == 0 and symbol not in targeted_full_refresh_symbols:
                    _link_or_copy(current_raw_path, stage / "raw" / current_raw_path.name)
                    _link_or_copy(current_qfq_path, stage / "qfq" / current_qfq_path.name)
                    _link_or_copy(
                        current_factor_path,
                        stage / "factors" / current_factor_path.name,
                    )
                    factors_by_symbol[symbol] = pd.read_parquet(current_factor_path)
                    continue

                _atomic_parquet(updated_raw, stage / "raw" / f"{symbol}.parquet")
                _atomic_parquet(updated_qfq, stage / "qfq" / f"{symbol}.parquet")
                _atomic_parquet(factors, stage / "factors" / f"{symbol}.parquet")
                factors_by_symbol[symbol] = factors
                changed_symbols.append(symbol)
                rows_added_by_symbol[symbol] = rows_added

            if not changed_symbols:
                _write_incremental_status(
                    root,
                    {
                        "status": "no_new_trade_rows",
                        "dataset_id": DATASET_ID,
                        "snapshot_id": current.name,
                        "through": through.date().isoformat(),
                        "requests_used": gate.calls,
                        "symbols_requested": len(missing_symbols),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "generated_at": timestamp,
                    },
                )
                return current

            actions = _corporate_actions_from_adjustments(
                factors_by_symbol,
                provider="tencent",
            )
            _atomic_parquet(actions, stage / "corporate_actions.parquet")
            _link_or_copy(universe_path, stage / "ETF_UNIVERSE.parquet")
            current_universe_manifest_path = current / "UNIVERSE_MANIFEST.json"
            if current_universe_manifest_path.is_file():
                universe_manifest = json.loads(
                    current_universe_manifest_path.read_text(encoding="utf-8")
                )
            else:
                universe_manifest = {}
            universe_manifest.update(
                {
                    "inherited_from_snapshot_id": current.name,
                    "used_for_update_through": through.date().isoformat(),
                    "universe_file_sha256": sha256_file(stage / "ETF_UNIVERSE.parquet"),
                }
            )
            atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)

            qfq_response_keys = sorted(
                {
                    str(value)
                    for path in sorted((stage / "qfq").glob("*.parquet"))
                    for value in pd.read_parquet(
                        path, columns=["provider_response_key"]
                    )["provider_response_key"].dropna().unique()
                }
            )
            source = dict(manifest.get("source") or {})
            source.update(
                {
                    "provider": "tencent",
                    "interface": "ifzq_day_and_qfqday_bounded_increment",
                    "update_mode": "missing_dates_only",
                    "parent_snapshot_id": current.name,
                    "selection_method": "frozen_previous_snapshot",
                    "universe_path": "ETF_UNIVERSE.parquet",
                    "universe_manifest_path": "UNIVERSE_MANIFEST.json",
                    "requested_symbol_count": len(missing_symbols),
                    "reused_symbol_count": int(len(universe) - len(changed_symbols)),
                    "overlap_calendar_days": int(overlap_days),
                    "rows_added": int(sum(rows_added_by_symbol.values())),
                    "targeted_full_refresh_symbols": sorted(
                        targeted_full_refresh_symbols
                    ),
                    "qfq_response_keys": qfq_response_keys,
                    "snapshot_retention": {
                        "policy": "current_plus_one_previous",
                        "max_complete_snapshots": DEFAULT_SNAPSHOT_RETENTION,
                    },
                }
            )
            latest_date = max(
                _as_date(pd.read_parquet(path, columns=["date"])["date"].max())
                for path in (stage / "qfq").glob("*.parquet")
            ).date().isoformat()
            finalize_paired_price_snapshot(
                stage,
                dataset_id=DATASET_ID,
                snapshot_id=snapshot_id,
                role="canonical",
                status="ready",
                source=source,
                adjustment={
                    "method": "provider_native_paired",
                    "adjusted_at": timestamp,
                    "adjustment_as_of": latest_date,
                    "adjusted_fields": ["open", "high", "low", "close"],
                    "normalization_rule": (
                        "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
                    ),
                    "algorithm_version": "provider-native-affine-paired-v1",
                    "paired_raw_component_path": "raw",
                    "volume_adjusted": False,
                },
                producer={
                    "script": "download_cn_etf_data.py",
                    "version": PRODUCER_VERSION,
                },
                source_retrieved_at=timestamp,
            )
            dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
            published = publish_snapshot_group(
                [(stage, dataset_root)],
                quant_data_root=root,
                update_root_catalog=True,
            )[0]
            removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
                root,
                dataset_root,
            )
            published_status = {
                "status": (
                    "published_with_warnings"
                    if post_publish_warnings
                    else "published"
                ),
                "dataset_id": DATASET_ID,
                "snapshot_id": published.name,
                "parent_snapshot_id": current.name,
                "requested_through": requested_through.date().isoformat(),
                "through": through.date().isoformat(),
                "requests_used": gate.calls,
                "symbols_requested": len(missing_symbols),
                "symbols_changed": len(changed_symbols),
                "rows_added": int(sum(rows_added_by_symbol.values())),
                "targeted_full_refresh_symbols": sorted(
                    targeted_full_refresh_symbols
                ),
                "removed_snapshots": removed_snapshots,
                "post_publish_warnings": post_publish_warnings,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "generated_at": timestamp,
            }
            try:
                _write_incremental_status(root, published_status)
            except Exception as exc:
                print(
                    "ETF snapshot was published; status write failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return published
        finally:
            shutil.rmtree(staging_run, ignore_errors=True)


def _publish_cn_etf_snapshot_locked(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    generated_at: str | None = None,
    resume: bool = True,
    confirm_severe_current_corruption: bool = False,
    confirm_backup_unrecoverable: bool = False,
    confirm_user_approved: bool = False,
) -> Path:
    """Fully rebuild ETF history after the separate approval gate is satisfied."""

    if not all(
        (
            confirm_severe_current_corruption,
            confirm_backup_unrecoverable,
            confirm_user_approved,
        )
    ):
        raise CnEtfDownloadError(
            "full ETF rebuild requires proven severe current corruption, "
            "an unrecoverable backup, and explicit user approval"
        )
    if request_interval < MIN_REQUEST_INTERVAL:
        raise ValueError(f"request_interval must be at least {MIN_REQUEST_INTERVAL}")
    if request_budget < 1:
        raise ValueError("request_budget must be positive")
    if retries < 1 or retries > 2:
        raise ValueError("retries must be between 1 and 2")
    if max_consecutive_failures < 1 or max_consecutive_failures > 3:
        raise ValueError("max_consecutive_failures must be between 1 and 3")

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    through = (end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat())
    if _as_date(start_date) > _as_date(through):
        raise ValueError("start_date must not be after end_date")
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    gate = RequestGate(interval=request_interval, budget=request_budget)
    normalized_required_codes = tuple(
        dict.fromkeys(str(code).zfill(6) for code in required_codes)
    )
    selection_contract = {
        "start_date": _as_date(start_date).date().isoformat(),
        "through": _as_date(through).date().isoformat(),
        "limit": int(limit),
        "required_codes": list(normalized_required_codes),
        "provider": "tencent",
        "producer_version": PRODUCER_VERSION,
    }
    selection_digest = hashlib.sha256(
        json.dumps(selection_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache = (
        root
        / "cache"
        / "cn_etf_daily"
        / "v1"
        / through
        / f"trial-{limit}-{selection_digest[:12]}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    contract_path = cache / "RUN_CONTRACT.json"
    universe_path = cache / "ETF_UNIVERSE.parquet"
    summary_path = cache / "UNIVERSE_SUMMARY.json"
    if resume and contract_path.is_file() and universe_path.is_file() and summary_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if any(
            existing_contract.get(key) != value
            for key, value in selection_contract.items()
        ):
            raise CnEtfDownloadError("resume cache belongs to another ETF contract")
        universe = pd.read_parquet(universe_path)
        catalog_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        universe_digest = _universe_hash(universe)
        if (
            existing_contract.get("universe_sha256") != universe_digest
            or int(existing_contract.get("symbol_count") or 0) != len(universe)
        ):
            raise CnEtfDownloadError("resume ETF universe hash does not match its contract")
    else:
        universe, catalog_summary = fetch_etf_universe(
            through=through,
            request_gate=gate,
            limit=limit,
            required_codes=normalized_required_codes,
        )
        universe_digest = _universe_hash(universe)
        contract = {
            "schema_version": 1,
            **selection_contract,
            "universe_sha256": universe_digest,
            "symbol_count": int(len(universe)),
        }
        atomic_write_json(contract_path, contract)
        _atomic_parquet(universe, universe_path)
        atomic_write_json(summary_path, catalog_summary)

    import requests

    session = requests.Session()
    session.trust_env = False
    failures: dict[str, str] = {}
    failures_path = cache / "failures.json"
    if failures_path.is_file():
        try:
            payload = json.loads(failures_path.read_text(encoding="utf-8"))
            failures = {
                str(item["symbol"]): str(item["error"])
                for item in payload.get("failures", [])
                if isinstance(item, dict) and item.get("symbol")
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            failures = {}

    completed = 0
    downloaded = 0
    upgraded_cache = 0
    consecutive_failures = 0
    started = time.monotonic()
    for position, row in enumerate(universe.itertuples(index=False), 1):
        symbol = str(row.symbol)
        progress = {
            "status": "running",
            "current": position,
            "total": int(len(universe)),
            "symbol": symbol,
            "completed": completed,
            "downloaded_this_run": downloaded,
            "upgraded_cached_factor_files": upgraded_cache,
            "requests_used": gate.calls,
            "through": through,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        atomic_write_json(cache / "progress.json", progress)
        if resume:
            cache_complete = _cache_symbol_complete(cache, symbol)
            if not cache_complete and _upgrade_cached_adjustment_evidence(cache, symbol):
                upgraded_cache += 1
                cache_complete = _cache_symbol_complete(cache, symbol)
            if cache_complete:
                completed += 1
                failures.pop(symbol, None)
                continue
        metadata = {
            "code": str(row.code),
            "market": str(row.market),
            "name": str(row.name),
        }
        try:
            raw, qfq = fetch_tencent_pair(
                symbol,
                metadata,
                start_date=start_date,
                end_date=through,
                session=session,
                request_gate=gate,
                retries=retries,
                timeout=timeout,
            )
            factors = build_paired_adjustments(symbol, raw, qfq)
            raw["source_id"] = raw["source"].astype(str)
            qfq["source_id"] = qfq["source"].astype(str)
            _atomic_parquet(raw, cache / "raw" / f"{symbol}.parquet")
            _atomic_parquet(qfq, cache / "qfq" / f"{symbol}.parquet")
            _atomic_parquet(factors, cache / "factors" / f"{symbol}.parquet")
            failures.pop(symbol, None)
            completed += 1
            downloaded += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            failures[symbol] = f"{type(exc).__name__}: {str(exc)[:300]}"
            atomic_write_json(
                failures_path,
                {
                    "failures": [
                        {"symbol": key, "error": value}
                        for key, value in sorted(failures.items())
                    ]
                },
            )
            if consecutive_failures >= max_consecutive_failures:
                break
    atomic_write_json(
        failures_path,
        {
            "failures": [
                {"symbol": key, "error": value}
                for key, value in sorted(failures.items())
            ]
        },
    )

    incomplete = [
        str(symbol)
        for symbol in universe["symbol"]
        if not _cache_symbol_complete(cache, str(symbol))
    ]
    if incomplete:
        atomic_write_json(
            cache / "progress.json",
            {
                "status": "incomplete",
                "completed": int(len(universe) - len(incomplete)),
                "total": int(len(universe)),
                "incomplete": incomplete,
                "requests_used": gate.calls,
                "upgraded_cached_factor_files": upgraded_cache,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "through": through,
            },
        )
        raise CnEtfDownloadError(
            f"common ETF download is incomplete: {len(incomplete)} of {len(universe)} symbols failed; cache retained for resume"
        )

    run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    snapshot_id = f"cn-etf-tencent-{run_id}"
    staging_run = root / "staging" / f"cn-etf-{run_id}"
    stage = staging_run / snapshot_id
    try:
        factors_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in universe["symbol"].astype(str):
            raw = pd.read_parquet(cache / "raw" / f"{symbol}.parquet")
            qfq = pd.read_parquet(cache / "qfq" / f"{symbol}.parquet")
            factors = pd.read_parquet(cache / "factors" / f"{symbol}.parquet")
            _atomic_parquet(raw, stage / "raw" / f"{symbol}.parquet")
            _atomic_parquet(qfq, stage / "qfq" / f"{symbol}.parquet")
            _atomic_parquet(factors, stage / "factors" / f"{symbol}.parquet")
            factors_by_symbol[symbol] = factors
        actions = _corporate_actions_from_adjustments(
            factors_by_symbol,
            provider="tencent",
        )
        _atomic_parquet(actions, stage / "corporate_actions.parquet")
        _atomic_parquet(universe, stage / "ETF_UNIVERSE.parquet")
        universe_manifest = {
            **catalog_summary,
            "retrieved_at": timestamp,
            "through": through,
            "universe_sha256": universe_digest,
            "universe_file_sha256": sha256_file(stage / "ETF_UNIVERSE.parquet"),
        }
        atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)
        qfq_response_keys = sorted(
            {
                str(value)
                for symbol in universe["symbol"].astype(str)
                for value in pd.read_parquet(
                    cache / "qfq" / f"{symbol}.parquet",
                    columns=["provider_response_key"],
                )["provider_response_key"].dropna().unique()
            }
        )
        source = {
            "provider": "tencent",
            "interface": "ifzq_day_and_qfqday_backward_pagination",
            "universe_providers": ["sina", "tonghuashun"],
            "universe_path": "ETF_UNIVERSE.parquet",
            "universe_manifest_path": "UNIVERSE_MANIFEST.json",
            "universe_sha256": universe_digest,
            "selection_method": catalog_summary.get("selection_method"),
            "requested_symbol_count": int(len(universe)),
            "qfq_response_keys": qfq_response_keys,
            "snapshot_retention": {
                "policy": "current_plus_one_previous",
                "max_complete_snapshots": DEFAULT_SNAPSHOT_RETENTION,
            },
        }
        latest_date = max(
            pd.read_parquet(cache / "qfq" / f"{symbol}.parquet", columns=["date"])[
                "date"
            ].max().date().isoformat()
            for symbol in universe["symbol"].astype(str)
        )
        finalize_paired_price_snapshot(
            stage,
            dataset_id=DATASET_ID,
            snapshot_id=snapshot_id,
            role="canonical",
            status="ready",
            source=source,
            adjustment={
                "method": "provider_native_paired",
                "adjusted_at": timestamp,
                "adjustment_as_of": latest_date,
                "adjusted_fields": ["open", "high", "low", "close"],
                "normalization_rule": (
                    "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
                ),
                "algorithm_version": "provider-native-affine-paired-v1",
                "paired_raw_component_path": "raw",
                "volume_adjusted": False,
            },
            producer={"script": "download_cn_etf_data.py", "version": PRODUCER_VERSION},
            source_retrieved_at=timestamp,
        )
        dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
        published = publish_snapshot_group(
            [(stage, dataset_root)],
            quant_data_root=root,
            update_root_catalog=True,
        )
        removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
            root,
            dataset_root,
        )
        published_status = {
            "status": (
                "published_with_warnings"
                if post_publish_warnings
                else "published"
            ),
            "completed": int(len(universe)),
            "total": int(len(universe)),
            "downloaded_this_run": downloaded,
            "upgraded_cached_factor_files": upgraded_cache,
            "requests_used": gate.calls,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "snapshot_id": snapshot_id,
            "removed_snapshots": removed_snapshots,
            "post_publish_warnings": post_publish_warnings,
            "through": through,
        }
        try:
            atomic_write_json(cache / "progress.json", published_status)
        except Exception as exc:
            print(
                "ETF snapshot was published; progress write failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return published[0]
    finally:
        shutil.rmtree(staging_run, ignore_errors=True)


def publish_cn_etf_snapshot(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    generated_at: str | None = None,
    resume: bool = True,
    confirm_severe_current_corruption: bool = False,
    confirm_backup_unrecoverable: bool = False,
    confirm_user_approved: bool = False,
) -> Path:
    """Hold the shared ETF lock through full-history publication and cleanup."""

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    with _run_lock(lock_path):
        return _publish_cn_etf_snapshot_locked(
            quant_data_dir=root,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            required_codes=required_codes,
            request_interval=request_interval,
            request_budget=request_budget,
            retries=retries,
            timeout=timeout,
            max_consecutive_failures=max_consecutive_failures,
            generated_at=generated_at,
            resume=resume,
            confirm_severe_current_corruption=confirm_severe_current_corruption,
            confirm_backup_unrecoverable=confirm_backup_unrecoverable,
            confirm_user_approved=confirm_user_approved,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-root", type=Path, default=quant_data_root())
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--confirm-severe-current-corruption", action="store_true")
    parser.add_argument("--confirm-backup-unrecoverable", action="store_true")
    parser.add_argument("--confirm-user-approved", action="store_true")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--through", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--required-code", action="append", dest="required_codes")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL)
    parser.add_argument("--request-budget", type=int, default=DEFAULT_REQUEST_BUDGET)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    confirmations = (
        args.full_rebuild,
        args.confirm_severe_current_corruption,
        args.confirm_backup_unrecoverable,
        args.confirm_user_approved,
    )
    if not all(confirmations):
        print(
            "拒绝全量重建：必须同时确认当前快照严重损坏、备份无法恢复，"
            "并且用户已明确批准本次全量重建。普通更新请使用 "
            "quant_data_cli.py update cn_etf_daily。",
            file=sys.stderr,
        )
        return 2
    required_codes = tuple(args.required_codes or DEFAULT_REQUIRED_CODES)
    published = publish_cn_etf_snapshot(
        quant_data_dir=args.quant_data_root,
        start_date=args.start,
        end_date=args.through,
        limit=args.limit,
        required_codes=required_codes,
        request_interval=args.request_interval,
        request_budget=args.request_budget,
        retries=args.retries,
        timeout=args.timeout,
        resume=not args.no_resume,
        confirm_severe_current_corruption=args.confirm_severe_current_corruption,
        confirm_backup_unrecoverable=args.confirm_backup_unrecoverable,
        confirm_user_approved=args.confirm_user_approved,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
