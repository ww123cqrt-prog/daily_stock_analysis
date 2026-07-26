"""Download the two ETFs used by the isolated momentum-rotation candidate.

Raw prices come from AkShare's Sina ETF history endpoint. Adjusted prices are
stored separately from Eastmoney and Tencent qfq endpoints. This script writes
only under ``quant_data/parquet/etf_1d`` and never writes to the A-share stock
or Qlib providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from market_data_clients import proxy_bypass_env  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_price_snapshot,
    publish_snapshot,
)


SOURCE_NAME = "akshare.fund_etf_hist_sina"
EASTMONEY_RAW_SOURCE_NAME = "akshare.fund_etf_hist_em_raw"
ADJUSTED_SOURCE_NAME = "akshare.fund_etf_hist_em_qfq"
TENCENT_RAW_SOURCE_NAME = "tencent.ifzq_raw"
TENCENT_ADJUSTED_SOURCE_NAME = "tencent.ifzq_qfq"
TENCENT_RAW_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
KNOWN_CORPORATE_ACTIONS = [
    {
        "symbol": "SH513100",
        "type": "share_split",
        "record_date": "2022-01-12",
        "split_date": "2022-01-13",
        "resumption_date": "2022-01-14",
        "ratio": "1:5",
        "source": "China Securities Journal fund announcement",
        "source_url": "https://epaper.cs.com.cn/zgzqb/html/2022-01/14/nw.D110000zgzqb_20220114_7-B019.htm",
    }
]
DEFAULT_SYMBOLS = {
    "SZ159915": {"code": "159915", "market": "sz", "name": "创业板ETF"},
    "SH513100": {"code": "513100", "market": "sh", "name": "纳指ETF"},
}
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_OVERLAP_DAYS = 5
DEFAULT_OUTPUT_DIR = (
    quant_data_root() / "parquet" / "etf_1d" / "etf_momentum_rotation" / "v1"
)
DEFAULT_ADJUSTED_OUTPUT_DIR = (
    quant_data_root()
    / "parquet"
    / "etf_1d"
    / "etf_momentum_rotation"
    / "eastmoney_qfq"
)
DEFAULT_TENCENT_ADJUSTED_OUTPUT_DIR = (
    quant_data_root()
    / "parquet"
    / "etf_1d"
    / "etf_momentum_rotation"
    / "tencent_qfq"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d"))
    return pd.Timestamp(value).normalize()


def fetch_one(symbol: str, *, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and normalize one ETF without changing the global AkShare client."""

    import akshare as ak

    metadata = DEFAULT_SYMBOLS[symbol]
    raw = ak.fund_etf_hist_sina(symbol=f"{metadata['market']}{metadata['code']}")
    if raw is None or raw.empty:
        raise RuntimeError(f"Sina returned no rows for {symbol}")
    frame = raw.rename(columns={"date": "date"}).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        (frame["date"] >= _as_timestamp(start_date))
        & (frame["date"] <= _as_timestamp(end_date))
    ].copy()
    frame = frame.dropna(subset=["date", "open", "close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"Sina returned duplicate dates for {symbol}")
    if frame.empty:
        raise RuntimeError(f"no rows remain for {symbol} in requested range")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = SOURCE_NAME
    return frame


def fetch_one_adjusted(
    symbol: str, *, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch one Eastmoney ETF history series with forward adjustment."""

    import akshare as ak

    metadata = DEFAULT_SYMBOLS[symbol]
    with proxy_bypass_env(False):
        raw = ak.fund_etf_hist_em(
            symbol=metadata["code"],
            period="daily",
            start_date=_as_timestamp(start_date).strftime("%Y%m%d"),
            end_date=_as_timestamp(end_date).strftime("%Y%m%d"),
            adjust="qfq",
        )
    if raw is None or raw.empty:
        raise RuntimeError(f"Eastmoney returned no qfq rows for {symbol}")
    frame = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    required = {"date", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Eastmoney qfq rows for {symbol} are missing columns: "
            + ", ".join(sorted(missing))
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"Eastmoney returned duplicate qfq dates for {symbol}")
    if frame.empty:
        raise RuntimeError(f"no Eastmoney qfq rows remain for {symbol}")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = ADJUSTED_SOURCE_NAME
    frame["adjustment"] = "qfq"
    return frame


def fetch_one_eastmoney_raw(
    symbol: str, *, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch the raw-price peer of the EastMoney qfq series."""

    import akshare as ak

    metadata = DEFAULT_SYMBOLS[symbol]
    with proxy_bypass_env(False):
        raw = ak.fund_etf_hist_em(
            symbol=metadata["code"],
            period="daily",
            start_date=_as_timestamp(start_date).strftime("%Y%m%d"),
            end_date=_as_timestamp(end_date).strftime("%Y%m%d"),
            adjust="",
        )
    if raw is None or raw.empty:
        raise RuntimeError(f"Eastmoney returned no raw rows for {symbol}")
    frame = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Eastmoney raw rows for {symbol} are missing columns: "
            + ", ".join(sorted(missing))
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if frame["date"].duplicated().any() or frame.empty:
        raise RuntimeError(f"Eastmoney raw rows are invalid for {symbol}")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = EASTMONEY_RAW_SOURCE_NAME
    return frame


def _fetch_one_tencent_history(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    adjustment: str,
    session=None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    import requests

    if adjustment not in {"raw", "qfq"}:
        raise ValueError("Tencent adjustment must be raw or qfq")
    metadata = DEFAULT_SYMBOLS[symbol]
    ticker = f"{metadata['market']}{metadata['code']}"
    start_ts = _as_timestamp(start_date)
    end_ts = _as_timestamp(end_date)
    client = session or requests.Session()
    if session is None:
        client.trust_env = False
    rows: list[list[object]] = []
    response_key = "day" if adjustment == "raw" else "qfqday"
    suffix = "" if adjustment == "raw" else ",qfq"
    for year in range(start_ts.year, end_ts.year + 1):
        chunk_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        response = client.get(
            TENCENT_RAW_KLINE_URL if adjustment == "raw" else TENCENT_KLINE_URL,
            params={
                "param": (
                    f"{ticker},day,{chunk_start.date().isoformat()},"
                    f"{chunk_end.date().isoformat()},400{suffix}"
                )
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(
                f"Tencent {adjustment} request failed for {symbol}: "
                f"{payload.get('msg')}"
            )
        ticker_data = payload.get("data", {}).get(ticker, {})
        chunk_rows = ticker_data.get(response_key)
        if not isinstance(chunk_rows, list):
            raise RuntimeError(
                f"Tencent returned no {adjustment} rows for {symbol}"
            )
        rows.extend(chunk_rows)
    if not rows:
        raise RuntimeError(f"Tencent returned no {adjustment} rows for {symbol}")
    normalized_rows = [
        row[:6] for row in rows if isinstance(row, list) and len(row) >= 6
    ]
    frame = pd.DataFrame(
        normalized_rows,
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.loc[
        (frame["date"] >= start_ts) & (frame["date"] <= end_ts)
    ].copy()
    frame = frame.sort_values("date", kind="stable").drop_duplicates(
        subset=["date"], keep="last"
    ).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"no Tencent {adjustment} rows remain for {symbol}")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = (
        TENCENT_RAW_SOURCE_NAME
        if adjustment == "raw"
        else TENCENT_ADJUSTED_SOURCE_NAME
    )
    if adjustment == "qfq":
        frame["adjustment"] = "qfq"
    return frame


def fetch_one_tencent_adjusted(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    session=None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch one Tencent qfq ETF series in bounded calendar-year chunks."""
    return _fetch_one_tencent_history(
        symbol,
        start_date=start_date,
        end_date=end_date,
        adjustment="qfq",
        session=session,
        timeout=timeout,
    )


def fetch_one_tencent_raw(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    session=None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    return _fetch_one_tencent_history(
        symbol,
        start_date=start_date,
        end_date=end_date,
        adjustment="raw",
        session=session,
        timeout=timeout,
    )


def fetch_provider_pair(
    provider: str,
    symbol: str,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if provider == "tencent":
        return (
            fetch_one_tencent_raw(
                symbol, start_date=start_date, end_date=end_date
            ),
            fetch_one_tencent_adjusted(
                symbol, start_date=start_date, end_date=end_date
            ),
        )
    if provider == "eastmoney":
        return (
            fetch_one_eastmoney_raw(
                symbol, start_date=start_date, end_date=end_date
            ),
            fetch_one_adjusted(symbol, start_date=start_date, end_date=end_date),
        )
    raise ValueError(f"unsupported ETF paired provider: {provider}")


def build_paired_factors(
    symbol: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    raw_frame = raw.copy()
    adjusted_frame = adjusted.copy()
    raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
    adjusted_frame["date"] = pd.to_datetime(
        adjusted_frame["date"], errors="coerce"
    ).dt.normalize()
    if (
        raw_frame["date"].isna().any()
        or adjusted_frame["date"].isna().any()
        or raw_frame["date"].duplicated().any()
        or adjusted_frame["date"].duplicated().any()
        or set(raw_frame["date"]) != set(adjusted_frame["date"])
    ):
        raise DataContractError(f"raw/qfq dates differ for {symbol}")
    paired = raw_frame[["date", "close"]].merge(
        adjusted_frame[["date", "close"]],
        on="date",
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", "_qfq"),
    ).sort_values("date")
    raw_close = pd.to_numeric(paired["close_raw"], errors="coerce")
    qfq_close = pd.to_numeric(paired["close_qfq"], errors="coerce")
    factor = qfq_close / raw_close
    if (
        factor.isna().any()
        or (factor <= 0).any()
        or not np.isfinite(float(factor.iloc[-1]))
        or not abs(float(factor.iloc[-1]) - 1.0) <= 1e-8
    ):
        raise DataContractError(f"raw/qfq factor curve is invalid for {symbol}")
    for field in ("open", "high", "low", "close"):
        if field not in raw_frame or field not in adjusted_frame:
            raise DataContractError(f"raw/qfq {field} field is missing for {symbol}")
        raw_values = pd.to_numeric(raw_frame.set_index("date")[field], errors="coerce")
        qfq_values = pd.to_numeric(
            adjusted_frame.set_index("date")[field], errors="coerce"
        )
        expected = raw_values.loc[paired["date"]].to_numpy(dtype=float) * factor.to_numpy(
            dtype=float
        )
        observed = qfq_values.loc[paired["date"]].to_numpy(dtype=float)
        if not bool(
            pd.Series(observed).notna().all()
            and pd.Series(expected).notna().all()
            and np.allclose(
                observed,
                expected,
                rtol=0.002,
                atol=0.005,
            )
        ):
            raise DataContractError(
                f"raw/qfq {field} relationship is invalid for {symbol}"
            )
    result = pd.DataFrame(
        {
            "date": paired["date"].reset_index(drop=True),
            "factor": factor.reset_index(drop=True),
            "factor_source": "derived_same_provider_raw_qfq_close_ratio",
        }
    )
    if symbol == "SH513100":
        split = pd.Timestamp("2022-01-13")
        before = result.loc[result["date"] < split, "factor"]
        after = result.loc[result["date"] > split, "factor"]
        if not before.empty and not after.empty:
            observed_ratio = float(after.iloc[0] / before.iloc[-1])
            if not 4.5 <= observed_ratio <= 5.5:
                raise DataContractError(
                    "SH513100 split factor is inconsistent with the 1:5 action"
                )
    return result


def _corporate_actions_frame(
    factors_by_symbol: dict[str, pd.DataFrame] | None = None,
    *,
    provider: str = "unknown",
) -> pd.DataFrame:
    factors_by_symbol = factors_by_symbol or {}
    known_effective_dates = {
        (item["symbol"], pd.Timestamp(item["resumption_date"]))
        for item in KNOWN_CORPORATE_ACTIONS
    }
    rows: list[dict[str, object]] = []
    for symbol, factors in factors_by_symbol.items():
        ordered = factors.sort_values("date", kind="stable").reset_index(drop=True)
        ratios = pd.to_numeric(ordered["factor"], errors="coerce") / pd.to_numeric(
            ordered["factor"], errors="coerce"
        ).shift(1)
        for position in ratios.index[(ratios - 1.0).abs() >= 0.005]:
            effective = pd.Timestamp(ordered.loc[position, "date"])
            if (symbol, effective) in known_effective_dates:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "record_date": pd.NaT,
                    "ex_date": effective,
                    "effective_trade_date": effective,
                    "resumption_date": effective,
                    "action_type": "provider_adjustment_event",
                    "ratio": float(ratios.loc[position]),
                    "factor_before": float(ordered.loc[position - 1, "factor"]),
                    "factor_after": float(ordered.loc[position, "factor"]),
                    "evidence_method": "derived_from_same_provider_raw_qfq_pair",
                    "source": f"{provider}.paired_raw_qfq",
                    "source_url": None,
                }
            )
    for item in KNOWN_CORPORATE_ACTIONS:
        factors = factors_by_symbol.get(item["symbol"])
        factor_before = None
        factor_after = None
        effective = pd.Timestamp(item["resumption_date"])
        if factors is not None and not factors.empty:
            before = factors.loc[factors["date"] < effective, "factor"]
            after = factors.loc[factors["date"] >= effective, "factor"]
            if not before.empty and not after.empty:
                factor_before = float(before.iloc[-1])
                factor_after = float(after.iloc[0])
        rows.append(
            {
                "symbol": item["symbol"],
                "record_date": pd.Timestamp(item["record_date"]),
                "ex_date": pd.Timestamp(item["split_date"]),
                "effective_trade_date": effective,
                "resumption_date": effective,
                "action_type": item["type"],
                "ratio": 5.0,
                "factor_before": factor_before,
                "factor_after": factor_after,
                "evidence_method": "announcement_and_same_provider_raw_qfq_pair",
                "source": item["source"],
                "source_url": item["source_url"],
            }
        )
    frame = pd.DataFrame(rows)
    for field in ("record_date", "ex_date", "effective_trade_date", "resumption_date"):
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    return frame.sort_values(
        ["symbol", "effective_trade_date"], kind="stable"
    ).reset_index(drop=True)


def _sync_root_manifest(root: Path) -> None:
    from sync_data_manifest import sync_data_manifest

    sync_data_manifest(quant_data_root=root)


def publish_canonical_snapshot(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    provider_order: tuple[str, ...] = ("tencent", "eastmoney"),
    generated_at: str | None = None,
) -> Path:
    """Publish a full-history, same-provider raw/qfq ETF snapshot pair."""

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    end = end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat()
    timestamp = generated_at or pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    errors: list[str] = []
    raw_by_symbol: dict[str, pd.DataFrame] = {}
    qfq_by_symbol: dict[str, pd.DataFrame] = {}
    selected_provider = ""
    for provider in provider_order:
        try:
            candidate_raw: dict[str, pd.DataFrame] = {}
            candidate_qfq: dict[str, pd.DataFrame] = {}
            for symbol in DEFAULT_SYMBOLS:
                raw_frame, qfq_frame = fetch_provider_pair(
                    provider,
                    symbol,
                    start_date=start_date,
                    end_date=end,
                )
                build_paired_factors(symbol, raw_frame, qfq_frame)
                candidate_raw[symbol] = raw_frame
                candidate_qfq[symbol] = qfq_frame
            latest_dates = {
                pd.to_datetime(frame["date"], errors="coerce").max().date().isoformat()
                for frame in [*candidate_raw.values(), *candidate_qfq.values()]
            }
            if len(latest_dates) != 1:
                raise DataContractError(
                    "the two ETFs and their raw/qfq peers do not share one latest date"
                )
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
            continue
        raw_by_symbol = candidate_raw
        qfq_by_symbol = candidate_qfq
        selected_provider = provider
        break
    if not selected_provider:
        raise RuntimeError("all ETF paired providers failed: " + "; ".join(errors))

    run_id = (
        pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    raw_snapshot_id = f"etf-momentum-{selected_provider}-raw-{run_id}"
    qfq_snapshot_id = f"etf-momentum-{selected_provider}-qfq-{run_id}"
    staging_run = root / "staging" / run_id
    raw_stage = staging_run / raw_snapshot_id
    qfq_stage = staging_run / qfq_snapshot_id
    try:
        factors_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in DEFAULT_SYMBOLS:
            raw_frame = raw_by_symbol[symbol].copy()
            qfq_frame = qfq_by_symbol[symbol].copy()
            raw_frame["source_id"] = raw_frame["source"].astype(str)
            qfq_frame["source_id"] = qfq_frame["source"].astype(str)
            _atomic_parquet(raw_frame, raw_stage / "bars" / f"{symbol}.parquet")
            _atomic_parquet(qfq_frame, qfq_stage / "bars" / f"{symbol}.parquet")
            factors = build_paired_factors(symbol, raw_frame, qfq_frame)
            factors_by_symbol[symbol] = factors
            _atomic_parquet(factors, qfq_stage / "factors" / f"{symbol}.parquet")
        actions = _corporate_actions_frame(
            factors_by_symbol,
            provider=selected_provider,
        )
        _atomic_parquet(actions, raw_stage / "corporate_actions.parquet")
        _atomic_parquet(actions, qfq_stage / "corporate_actions.parquet")
        source = {
            "provider": selected_provider,
            "interface": (
                "ifzq_day_and_qfqday"
                if selected_provider == "tencent"
                else "fund_etf_hist_em_raw_and_qfq"
            ),
            "raw_security_codes": {
                symbol: f"{metadata['market']}{metadata['code']}"
                for symbol, metadata in DEFAULT_SYMBOLS.items()
            },
        }
        finalize_price_snapshot(
            raw_stage,
            dataset_id="etf_momentum_daily_raw",
            snapshot_id=raw_snapshot_id,
            role="audit",
            status="ready",
            source=source,
            price_basis="raw",
            adjustment={"method": "none"},
            producer={"script": "download_etf_momentum_data.py", "version": "2"},
            source_retrieved_at=timestamp,
        )
        latest_date = max(
            pd.to_datetime(frame["date"]).max().date().isoformat()
            for frame in qfq_by_symbol.values()
        )
        finalize_price_snapshot(
            qfq_stage,
            dataset_id="etf_momentum_daily_qfq",
            snapshot_id=qfq_snapshot_id,
            role="canonical",
            status="ready",
            source=source,
            price_basis="qfq",
            adjustment={
                "method": "provider_native_paired",
                "adjusted_at": timestamp,
                "adjustment_as_of": latest_date,
                "adjusted_fields": ["open", "high", "low", "close"],
                "normalization_rule": "latest_valid_trade_date_factor_equals_1",
                "algorithm_version": "provider-native-paired-v1",
                "paired_raw_snapshot_id": raw_snapshot_id,
                "volume_adjusted": False,
            },
            producer={"script": "download_etf_momentum_data.py", "version": "2"},
            source_retrieved_at=timestamp,
            upstream_snapshots=[
                {"dataset_id": "etf_momentum_daily_raw", "snapshot_id": raw_snapshot_id}
            ],
        )
        raw_root = root / "parquet" / "etf_1d" / "etf_momentum_daily_raw"
        qfq_root = root / "parquet" / "etf_1d" / "etf_momentum_daily_qfq"
        publish_snapshot(
            raw_stage,
            dataset_root=raw_root,
            quant_data_root=root,
        )
        published = publish_snapshot(
            qfq_stage,
            dataset_root=qfq_root,
            quant_data_root=root,
        )
        _sync_root_manifest(root)
        return published
    finally:
        shutil.rmtree(staging_run, ignore_errors=True)


def _merge_existing(
    existing: pd.DataFrame, refreshed: pd.DataFrame, *, symbol: str
) -> pd.DataFrame:
    """Merge a refreshed source slice without creating duplicate dates."""

    combined = pd.concat([existing, refreshed], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["date"]).sort_values("date", kind="stable")
    combined = combined.drop_duplicates(subset=["date"], keep="last").reset_index(
        drop=True
    )
    combined["symbol"] = symbol
    combined["name"] = DEFAULT_SYMBOLS[symbol]["name"]
    combined["source"] = SOURCE_NAME
    return combined


def _merge_adjusted_existing(
    existing: pd.DataFrame,
    refreshed: pd.DataFrame,
    *,
    symbol: str,
    source_name: str = ADJUSTED_SOURCE_NAME,
) -> pd.DataFrame:
    combined = pd.concat([existing, refreshed], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["date"]).sort_values("date", kind="stable")
    combined = combined.drop_duplicates(subset=["date"], keep="last").reset_index(
        drop=True
    )
    combined["symbol"] = symbol
    combined["name"] = DEFAULT_SYMBOLS[symbol]["name"]
    combined["source"] = source_name
    combined["adjustment"] = "qfq"
    return combined


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    incremental: bool = False,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> Path:
    end = end_date or date.today().isoformat()
    start_ts = _as_timestamp(start_date)
    end_ts = _as_timestamp(end)
    if overlap_days < 0:
        raise ValueError("overlap_days must be nonnegative")
    if start_ts > end_ts:
        raise ValueError("start_date must not be after end_date")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for symbol in DEFAULT_SYMBOLS:
        path = output_dir / f"{symbol}.parquet"
        existing = pd.read_parquet(path) if incremental and path.exists() else None
        existing_end = None
        if existing is not None and not existing.empty:
            existing_end = pd.to_datetime(existing["date"], errors="coerce").max()
        if (
            existing_end is not None
            and not pd.isna(existing_end)
            and existing_end >= end_ts
        ):
            frame = existing.sort_values("date", kind="stable").reset_index(drop=True)
        else:
            refresh_start = start_ts
            if existing_end is not None and not pd.isna(existing_end):
                refresh_start = max(
                    start_ts, existing_end.normalize() - timedelta(days=overlap_days)
                )
            refreshed = fetch_one(
                symbol,
                start_date=refresh_start.strftime("%Y-%m-%d"),
                end_date=end,
            )
            frame = (
                _merge_existing(existing, refreshed, symbol=symbol)
                if existing is not None
                else refreshed
            )
        _atomic_parquet(frame, path)
        entries[symbol] = {
            "name": DEFAULT_SYMBOLS[symbol]["name"],
            "code": DEFAULT_SYMBOLS[symbol]["code"],
            "market": DEFAULT_SYMBOLS[symbol]["market"],
            "source": SOURCE_NAME,
            "adjustment": "raw_close_from_sina",
            "incremental": incremental,
            "overlap_days": overlap_days,
            "requested_start": start_date,
            "requested_end": end,
            "actual_start": str(frame["date"].min().date()),
            "actual_end": str(frame["date"].max().date()),
            "row_count": int(len(frame)),
            "sha256": _sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "dataset": "etf_momentum_rotation_v1",
        "strategy_scope": "research_only_two_etf_rotation",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": SOURCE_NAME,
        "raw_price_note": "Sina endpoint supplies raw close; this is not a dividend-adjusted total-return series.",
        "symbols": entries,
    }
    manifest_path = output_dir / "SOURCE_MANIFEST.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def _download_adjusted_snapshot(
    *,
    output_dir: Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    incremental: bool = False,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    fetcher,
    source_name: str,
    dataset_name: str,
) -> Path:
    """Download a coherent two-file qfq snapshot.

    Fetching is completed for both symbols before any file is published, so a
    source failure cannot replace half of an existing pair.
    """

    end = end_date or date.today().isoformat()
    start_ts = _as_timestamp(start_date)
    end_ts = _as_timestamp(end)
    if overlap_days < 0:
        raise ValueError("overlap_days must be nonnegative")
    if start_ts > end_ts:
        raise ValueError("start_date must not be after end_date")

    output_dir = Path(output_dir).expanduser().resolve()
    staged: dict[str, pd.DataFrame] = {}
    for symbol in DEFAULT_SYMBOLS:
        path = output_dir / f"{symbol}.parquet"
        existing = pd.read_parquet(path) if incremental and path.exists() else None
        existing_end = None
        if existing is not None and not existing.empty:
            existing_end = pd.to_datetime(existing["date"], errors="coerce").max()
        if (
            existing_end is not None
            and not pd.isna(existing_end)
            and existing_end >= end_ts
        ):
            frame = existing.sort_values("date", kind="stable").reset_index(drop=True)
        else:
            refresh_start = start_ts
            if existing_end is not None and not pd.isna(existing_end):
                refresh_start = max(
                    start_ts, existing_end.normalize() - timedelta(days=overlap_days)
                )
            refreshed = fetcher(
                symbol,
                start_date=refresh_start.strftime("%Y-%m-%d"),
                end_date=end,
            )
            frame = (
                _merge_adjusted_existing(
                    existing,
                    refreshed,
                    symbol=symbol,
                    source_name=source_name,
                )
                if existing is not None
                else refreshed
            )
        staged[symbol] = frame

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for symbol, frame in staged.items():
        path = output_dir / f"{symbol}.parquet"
        _atomic_parquet(frame, path)
        entries[symbol] = {
            "name": DEFAULT_SYMBOLS[symbol]["name"],
            "code": DEFAULT_SYMBOLS[symbol]["code"],
            "market": DEFAULT_SYMBOLS[symbol]["market"],
            "source": source_name,
            "adjustment": "qfq",
            "incremental": incremental,
            "overlap_days": overlap_days,
            "requested_start": start_date,
            "requested_end": end,
            "actual_start": str(frame["date"].min().date()),
            "actual_end": str(frame["date"].max().date()),
            "row_count": int(len(frame)),
            "sha256": _sha256(path),
        }

    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "strategy_scope": "research_only_two_etf_rotation",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": source_name,
        "adjustment": "qfq",
        "corporate_actions": KNOWN_CORPORATE_ACTIONS,
        "symbols": entries,
    }
    manifest_path = output_dir / "SOURCE_MANIFEST.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def download_adjusted(
    *,
    output_dir: Path = DEFAULT_ADJUSTED_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    incremental: bool = False,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> Path:
    return _download_adjusted_snapshot(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        incremental=incremental,
        overlap_days=overlap_days,
        fetcher=fetch_one_adjusted,
        source_name=ADJUSTED_SOURCE_NAME,
        dataset_name="etf_momentum_rotation_eastmoney_qfq",
    )


def download_tencent_adjusted(
    *,
    output_dir: Path = DEFAULT_TENCENT_ADJUSTED_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    incremental: bool = False,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
) -> Path:
    return _download_adjusted_snapshot(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        incremental=incremental,
        overlap_days=overlap_days,
        fetcher=fetch_one_tencent_adjusted,
        source_name=TENCENT_ADJUSTED_SOURCE_NAME,
        dataset_name="etf_momentum_rotation_tencent_qfq",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download isolated ETF momentum-rotation data."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--adjusted-output-dir",
        type=Path,
        default=DEFAULT_ADJUSTED_OUTPUT_DIR,
        help="Isolated Eastmoney qfq output directory.",
    )
    parser.add_argument(
        "--tencent-adjusted-output-dir",
        type=Path,
        default=DEFAULT_TENCENT_ADJUSTED_OUTPUT_DIR,
        help="Isolated Tencent qfq output directory.",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument(
        "--end-date",
        "--end",
        dest="end_date",
        default=None,
        help="Download end date; --end is retained for daily-pipeline compatibility.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse local rows and refresh only the latest overlap window.",
    )
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS)
    parser.add_argument(
        "--skip-adjusted",
        action="store_true",
        help="Skip the best-effort Eastmoney and Tencent qfq refreshes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = download(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        incremental=args.incremental,
        overlap_days=args.overlap_days,
    )
    print(f"ETF momentum raw data manifest: {raw_path}")
    if not args.skip_adjusted:
        try:
            adjusted_path = download_adjusted(
                output_dir=args.adjusted_output_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                incremental=args.incremental,
                overlap_days=args.overlap_days,
            )
        except Exception as exc:
            print(
                "Eastmoney qfq refresh unavailable; existing adjusted cache was "
                f"preserved: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"ETF momentum adjusted data manifest: {adjusted_path}")
        try:
            tencent_path = download_tencent_adjusted(
                output_dir=args.tencent_adjusted_output_dir,
                start_date=args.start_date,
                end_date=args.end_date,
                incremental=args.incremental,
                overlap_days=args.overlap_days,
            )
        except Exception as exc:
            print(
                "Tencent qfq refresh unavailable; existing adjusted cache was "
                f"preserved: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"ETF momentum Tencent qfq data manifest: {tencent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
