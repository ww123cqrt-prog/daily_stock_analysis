"""Reusable anonymous market-data clients for local quant workflows.

This module intentionally has no knowledge of parquet output, Qlib providers,
reports, or DSA web jobs.  It only owns source-specific request details and
response parsing so multiple tools can share one implementation.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Any

import pandas as pd
import requests


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_SNAPSHOT_URL = "https://push2.eastmoney.com/api/qt/stock/get"
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={codes}"
EASTMONEY_KLINE_FIELDS1 = "f1,f2,f3,f4,f5,f6"
EASTMONEY_KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
EASTMONEY_KLINE_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://quote.eastmoney.com/",
    "User-Agent": DEFAULT_USER_AGENT,
}
EASTMONEY_SNAPSHOT_FIELDS = "f57,f58,f43,f170,f116,f117,f168,f10,f9,f23"
SINA_QUOTE_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": DEFAULT_USER_AGENT,
}
SINA_QUOTE_RE = re.compile(r'var hq_str_(?P<code>[a-z]{2}\d{6})="(?P<body>.*?)";')


@contextmanager
def proxy_bypass_env(use_system_proxy: bool):
    """Bypass macOS/system proxies for anonymous quote endpoints by default."""
    if use_system_proxy:
        yield
        return

    keys = ("NO_PROXY", "no_proxy")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def normalize_date(value: str) -> str:
    raw = str(value).strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw


def compact_date(value: str) -> str:
    return normalize_date(value).replace("-", "")


def normalize_symbol(value: str) -> str:
    raw = str(value).split("#", 1)[0].strip().upper()
    if not raw:
        return ""
    raw = raw.replace(",", " ").split()[0]
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    if raw.isdigit() and len(raw) < 6:
        raw = raw.zfill(6)
    return raw if raw.isdigit() and len(raw) == 6 else ""


def eastmoney_secid(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    market = "1" if normalized.startswith(("5", "6", "9")) else "0"
    return f"{market}.{normalized}"


def sina_quote_code(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    prefix = "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{normalized}"


def _open_session(session: requests.Session | None, use_system_proxy: bool) -> tuple[requests.Session, bool]:
    if session is not None:
        return session, False
    session = requests.Session()
    session.trust_env = use_system_proxy
    return session, True


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _scaled(value: Any, scale: float) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    return number / scale


def parse_eastmoney_kline_payload(payload: dict[str, Any] | None) -> pd.DataFrame:
    """Parse EastMoney kline JSON into provider-neutral daily bar columns."""
    klines = ((payload or {}).get("data") or {}).get("klines") or []
    rows: list[dict[str, object]] = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 11:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6],
                "pctChg": parts[8],
                "turn": parts[10],
            }
        )
    return pd.DataFrame(rows)


def fetch_eastmoney_qfq_daily(
    symbol: str,
    start: str,
    end: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30,
    use_system_proxy: bool = False,
) -> pd.DataFrame:
    """Fetch EastMoney qfq daily bars. Volume is returned in hands."""
    session, close_session = _open_session(session, use_system_proxy)
    try:
        with proxy_bypass_env(use_system_proxy):
            response = session.get(
                EASTMONEY_KLINE_URL,
                params={
                    "secid": eastmoney_secid(symbol),
                    "fields1": EASTMONEY_KLINE_FIELDS1,
                    "fields2": EASTMONEY_KLINE_FIELDS2,
                    "klt": "101",
                    "fqt": "1",
                    "beg": compact_date(start),
                    "end": compact_date(end),
                },
                headers=EASTMONEY_KLINE_HEADERS,
                timeout=timeout,
            )
        response.raise_for_status()
        return parse_eastmoney_kline_payload(response.json())
    finally:
        if close_session:
            session.close()


def fetch_akshare_qfq_daily(
    symbol: str,
    start: str,
    end: str,
    *,
    timeout: float = 30,
    use_system_proxy: bool = False,
) -> pd.DataFrame:
    """Fetch AkShare qfq daily bars. Volume is returned in hands."""
    import akshare as ak

    with proxy_bypass_env(use_system_proxy):
        try:
            return ak.stock_zh_a_hist(
                symbol=normalize_symbol(symbol),
                period="daily",
                start_date=compact_date(start),
                end_date=compact_date(end),
                adjust="qfq",
                timeout=timeout,
            )
        except TypeError:
            return ak.stock_zh_a_hist(
                symbol=normalize_symbol(symbol),
                period="daily",
                start_date=compact_date(start),
                end_date=compact_date(end),
                adjust="qfq",
            )


def parse_sina_quote_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in SINA_QUOTE_RE.finditer(text):
        code = match.group("code")
        symbol = normalize_symbol(code)
        parts = match.group("body").split(",")
        if len(parts) < 32 or not symbol:
            continue
        prev_close = _to_float(parts[2])
        close = _to_float(parts[3])
        rows.append(
            {
                "symbol": symbol,
                "name": parts[0],
                "open": _to_float(parts[1]),
                "preclose": prev_close,
                "close": close,
                "pct": ((close / prev_close) - 1) * 100 if close is not None and prev_close else None,
                "high": _to_float(parts[4]),
                "low": _to_float(parts[5]),
                "volume": _to_float(parts[8]),
                "amount": _to_float(parts[9]),
                "trade_date": parts[30],
                "trade_time": parts[31],
                "source": "sina_snapshot",
            }
        )
    return rows


def fetch_sina_a_snapshots(
    symbols: list[str],
    *,
    session: requests.Session | None = None,
    timeout: float = 10,
    use_system_proxy: bool = False,
) -> list[dict[str, Any]]:
    if not symbols:
        return []
    session, close_session = _open_session(session, use_system_proxy)
    try:
        with proxy_bypass_env(use_system_proxy):
            response = session.get(
                SINA_QUOTE_URL.format(codes=",".join(sina_quote_code(symbol) for symbol in symbols)),
                headers=SINA_QUOTE_HEADERS,
                timeout=timeout,
            )
        response.encoding = "gb18030"
        response.raise_for_status()
        return parse_sina_quote_text(response.text)
    finally:
        if close_session:
            session.close()


def fetch_eastmoney_a_snapshot(
    symbol: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 10,
    use_system_proxy: bool = False,
) -> dict[str, Any]:
    """Fetch one A-share quote snapshot from EastMoney's lightweight endpoint."""
    session, close_session = _open_session(session, use_system_proxy)
    try:
        with proxy_bypass_env(use_system_proxy):
            response = session.get(
                EASTMONEY_SNAPSHOT_URL,
                params={"secid": eastmoney_secid(symbol), "fields": EASTMONEY_SNAPSHOT_FIELDS},
                headers=EASTMONEY_KLINE_HEADERS,
                timeout=timeout,
            )
        response.raise_for_status()
        data = (response.json() or {}).get("data") or {}
    finally:
        if close_session:
            session.close()

    if not data:
        return {}
    return {
        "symbol": normalize_symbol(str(data.get("f57") or symbol)),
        "name": data.get("f58") or "",
        "price": _scaled(data.get("f43"), 100),
        "pct": _scaled(data.get("f170"), 100),
        "market_cap": _to_float(data.get("f116")),
        "float_market_cap": _to_float(data.get("f117")),
        "turnover_rate": _scaled(data.get("f168"), 100),
        "volume_ratio": _scaled(data.get("f10"), 100),
        "pe": _scaled(data.get("f9"), 100),
        "pb": _scaled(data.get("f23"), 100),
        "source": "eastmoney_snapshot",
    }
