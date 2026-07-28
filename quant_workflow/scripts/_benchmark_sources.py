"""Shared, bounded download clients for Qlib benchmark indexes."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd


TENCENT_BENCHMARKS = {
    "SH000300": "sh000300",
    "SH000905": "sh000905",
    "SH000906": "sh000906",
}


def download_tencent_benchmarks(
    start_date: str,
    end_date: str,
    *,
    use_system_proxy: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_interval_seconds: float = 0.5,
    retry_backoff_seconds: float = 1.0,
    max_attempts: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download three index OHLCV series with bounded retries and pagination."""

    import requests

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if request_interval_seconds < 0 or retry_backoff_seconds < 0:
        raise ValueError("Tencent request timing values must be non-negative")

    session = requests.Session()
    session.trust_env = use_system_proxy

    def fetch(params: dict[str, str]):
        for attempt in range(max_attempts):
            sleep_fn(request_interval_seconds)
            try:
                response = session.get(
                    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempt + 1 >= max_attempts:
                    raise
                sleep_fn(retry_backoff_seconds * (2**attempt))
        raise RuntimeError("Tencent request exhausted all attempts")

    result: dict[str, pd.DataFrame] = {}
    try:
        for symbol, provider_code in TENCENT_BENCHMARKS.items():
            values: dict[str, dict[str, float]] = {}
            cursor_end = end_date
            for _ in range(64):
                response = fetch(
                    {"param": f"{provider_code},day,,{cursor_end},640,qfq"}
                )
                payload = response.json()
                instrument = (
                    ((payload or {}).get("data") or {}).get(provider_code) or {}
                )
                rows = instrument.get("day") or instrument.get("qfqday") or []
                if not rows:
                    break
                page_dates: list[str] = []
                for row in rows:
                    if len(row) < 6:
                        continue
                    day = str(row[0])[:10]
                    page_dates.append(day)
                    if not (start_date <= day <= end_date):
                        continue
                    try:
                        values[day] = {
                            "open": float(row[1]),
                            "close": float(row[2]),
                            "high": float(row[3]),
                            "low": float(row[4]),
                            "volume": float(row[5]),
                        }
                    except (TypeError, ValueError):
                        continue
                if not page_dates:
                    break
                earliest = min(page_dates)
                if earliest <= start_date:
                    break
                next_end = (
                    datetime.strptime(earliest, "%Y-%m-%d").date()
                    - timedelta(days=1)
                ).isoformat()
                if next_end >= cursor_end:
                    break
                cursor_end = next_end
            frame = pd.DataFrame.from_dict(values, orient="index")
            if not frame.empty:
                frame = frame[["open", "high", "low", "close", "volume"]]
                frame.index = frame.index.astype(str)
                frame = frame.sort_index()
            result[symbol] = frame
    finally:
        session.close()
    return result
