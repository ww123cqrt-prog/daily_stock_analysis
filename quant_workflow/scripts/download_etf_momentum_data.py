"""ETF provider helpers and a compatibility entry for ``cn_etf_daily``.

The command-line entry always publishes the single governed ETF dataset. The
two-symbol functions remain calculation/source helpers for older tests and are
not registered as dataset publishers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from market_data_clients import proxy_bypass_env  # noqa: E402
from quant_data_contract import DataContractError  # noqa: E402


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


def publish_canonical_snapshot(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    provider_order: tuple[str, ...] = ("tencent", "eastmoney"),
    generated_at: str | None = None,
) -> Path:
    """Compatibility wrapper for the unified dataset's incremental updater."""

    if "tencent" not in provider_order:
        raise DataContractError(
            "the unified ETF producer currently requires Tencent paired history"
        )
    from download_cn_etf_data import update_cn_etf_snapshot

    del start_date
    return update_cn_etf_snapshot(
        quant_data_dir=quant_data_dir,
        end_date=end_date,
        generated_at=generated_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility entry point for the unified ETF dataset."
    )
    parser.add_argument("--quant-data-root", type=Path, default=quant_data_root())
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Compatibility-only; ignored because ordinary updates never rewind history.",
    )
    parser.add_argument(
        "--end-date",
        "--end",
        dest="end_date",
        default=None,
        help="Incremental update end date; --end is retained for compatibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    published = publish_canonical_snapshot(
        quant_data_dir=args.quant_data_root,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Unified ETF dataset snapshot: {published}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
