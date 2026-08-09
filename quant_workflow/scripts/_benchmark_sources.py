"""Shared, bounded download clients for Qlib benchmark indexes."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timedelta
from collections.abc import Sequence
from typing import Callable

import pandas as pd


TENCENT_BENCHMARKS = {
    "SH000300": "sh000300",
    "SH000905": "sh000905",
    "SH000906": "sh000906",
    "SZ399006": "sz399006",
    "SH000688": "sh000688",
}
EASTMONEY_BENCHMARKS = {
    "SH000300": "1.000300",
    "SH000905": "1.000905",
    "SH000906": "1.000906",
    "SZ399006": "0.399006",
    "SH000688": "1.000688",
}
DEFAULT_BENCHMARK_SYMBOLS = ("SH000300", "SH000905", "SH000906")


def _selected_benchmarks(
    provider_codes: dict[str, str], symbols: Sequence[str] | None
) -> dict[str, str]:
    requested = DEFAULT_BENCHMARK_SYMBOLS if symbols is None else tuple(symbols)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("benchmark symbols must be unique and non-empty")
    unknown = set(requested).difference(provider_codes)
    if unknown:
        raise ValueError("unsupported benchmark symbols: " + ", ".join(sorted(unknown)))
    return {symbol: provider_codes[symbol] for symbol in requested}


def download_eastmoney_benchmarks(
    start_date: str,
    end_date: str,
    *,
    use_system_proxy: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_interval_seconds: float = 0.5,
    retry_backoff_seconds: float = 1.0,
    max_attempts: int = 3,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download only the requested index OHLCV series with bounded retries."""

    import requests

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if request_interval_seconds < 0 or retry_backoff_seconds < 0:
        raise ValueError("Eastmoney request timing values must be non-negative")

    session = requests.Session()
    session.trust_env = use_system_proxy
    endpoint = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    )

    def fetch(params: dict[str, str]):
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            sleep_fn(request_interval_seconds)
            try:
                response = session.get(
                    endpoint,
                    params=params,
                    headers={
                        "User-Agent": user_agent,
                        "Referer": "https://quote.eastmoney.com/",
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable:
                    break
                if attempt + 1 >= max_attempts:
                    break
                sleep_fn(retry_backoff_seconds * (2**attempt))

        command = [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "10",
            "--max-time",
            "30",
            "--doh-url",
            "https://cloudflare-dns.com/dns-query",
            "--user-agent",
            user_agent,
            "--referer",
            "https://quote.eastmoney.com/",
            "--get",
        ]
        if not use_system_proxy:
            command.extend(["--noproxy", "*"])
        for key, value in params.items():
            command.extend(["--data-urlencode", f"{key}={value}"])
        command.append(endpoint)
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=35,
            )
            return json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise RuntimeError("Eastmoney request exhausted all transports") from (
                last_error or exc
            )

    result: dict[str, pd.DataFrame] = {}
    try:
        for symbol, secid in _selected_benchmarks(
            EASTMONEY_BENCHMARKS, symbols
        ).items():
            payload = fetch(
                {
                    "secid": secid,
                    "klt": "101",
                    "fqt": "0",
                    "beg": start_date.replace("-", ""),
                    "end": end_date.replace("-", ""),
                    "fields1": "f1,f2,f3,f4,f5,f6",
                    "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                }
            )
            rows = (((payload or {}).get("data") or {}).get("klines") or [])
            values: dict[str, dict[str, float]] = {}
            for raw_row in rows:
                row = raw_row.split(",") if isinstance(raw_row, str) else raw_row
                if len(row) < 6:
                    continue
                day = str(row[0])[:10]
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
            frame = pd.DataFrame.from_dict(values, orient="index")
            if not frame.empty:
                frame = frame[["open", "high", "low", "close", "volume"]]
                frame.index = frame.index.astype(str)
                frame = frame.sort_index()
                frame.index.name = "date"
            result[symbol] = frame
    finally:
        session.close()
    return result


def download_tencent_benchmarks(
    start_date: str,
    end_date: str,
    *,
    use_system_proxy: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_interval_seconds: float = 0.5,
    retry_backoff_seconds: float = 1.0,
    max_attempts: int = 3,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download only the requested index OHLCV series with bounded retries."""

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
        for symbol, provider_code in _selected_benchmarks(
            TENCENT_BENCHMARKS, symbols
        ).items():
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
                frame.index.name = "date"
            result[symbol] = frame
    finally:
        session.close()
    return result
