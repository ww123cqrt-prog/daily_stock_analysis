"""Reusable anonymous market-data clients for local quant workflows.

This module intentionally has no knowledge of parquet output, Qlib providers,
reports, or DSA web jobs.  It only owns source-specific request details and
response parsing so multiple tools can share one implementation.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_SNAPSHOT_URL = "https://push2.eastmoney.com/api/qt/stock/get"
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={codes}"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
EASTMONEY_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"
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
EASTMONEY_NEWS_JSONP_RE = re.compile(r"^[^(]*\((?P<payload>.*)\);?\s*$", flags=re.S)


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


def _open_yahoo_session(session: Any | None, use_system_proxy: bool) -> tuple[Any, bool]:
    """Use curl_cffi's browser-like TLS session when it is installed."""
    if session is not None:
        return session, False
    try:
        from curl_cffi import requests as curl_requests  # type: ignore

        session = curl_requests.Session(impersonate="chrome")
    except ImportError:
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
    text = fetch_sina_a_snapshot_text(
        symbols,
        session=session,
        timeout=timeout,
        use_system_proxy=use_system_proxy,
    )
    return parse_sina_quote_text(text)


def fetch_sina_a_snapshot_text(
    symbols: list[str],
    *,
    session: requests.Session | None = None,
    timeout: float = 10,
    use_system_proxy: bool = False,
) -> str:
    """Fetch and return Sina's original quote response text unchanged."""
    if not symbols:
        return ""
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
        return response.text
    finally:
        if close_session:
            session.close()


def parse_yahoo_chart_payload(payload: dict[str, Any] | None) -> pd.DataFrame:
    """Parse a Yahoo Chart response into normalized daily OHLCV rows."""
    chart = (payload or {}).get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume", "pct"])
    result = results[0] or {}
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    quote_values = quotes[0] or {}
    adjusted = indicators.get("adjclose") or [{}]
    adjusted_values = adjusted[0] or {}
    timezone_name = (result.get("meta") or {}).get("exchangeTimezoneName") or "UTC"

    dates = pd.to_datetime(pd.Series(timestamps, dtype="int64"), unit="s", utc=True)
    try:
        dates = dates.dt.tz_convert(timezone_name)
    except (KeyError, TypeError, ValueError):
        dates = dates.dt.tz_convert("UTC")

    def values(name: str, source: dict[str, Any] = quote_values) -> list[Any]:
        raw = source.get(name) or []
        return [raw[index] if index < len(raw) else None for index in range(len(timestamps))]

    frame = pd.DataFrame(
        {
            "date": dates.dt.tz_localize(None).dt.normalize(),
            "open": values("open"),
            "high": values("high"),
            "low": values("low"),
            "close": values("close"),
            "adj_close": values("adjclose", adjusted_values),
            "volume": values("volume"),
        }
    )
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame["pct"] = frame["close"].pct_change() * 100
    return frame.reset_index(drop=True)


def fetch_yahoo_chart_payload(
    symbol: str,
    start: str,
    end: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30,
    use_system_proxy: bool = False,
) -> dict[str, Any]:
    """Fetch one Yahoo daily-chart JSON payload without changing its contents."""
    start_ts = pd.Timestamp(normalize_date(start), tz="UTC")
    end_ts = pd.Timestamp(normalize_date(end), tz="UTC") + pd.Timedelta(days=1)
    session, close_session = _open_yahoo_session(session, use_system_proxy)
    try:
        with proxy_bypass_env(use_system_proxy):
            response = session.get(
                YAHOO_CHART_URL.format(symbol=quote(str(symbol).upper(), safe="-._^=")),
                params={
                    "period1": int(start_ts.timestamp()),
                    "period2": int(end_ts.timestamp()),
                    "interval": "1d",
                    "events": "history",
                    "includeAdjustedClose": "true",
                },
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Yahoo Chart returned a non-object payload")
        return payload
    finally:
        if close_session:
            session.close()


def parse_eastmoney_news_jsonp(text: str) -> dict[str, Any]:
    match = EASTMONEY_NEWS_JSONP_RE.match(str(text).strip())
    if not match:
        raise ValueError("EastMoney news endpoint returned invalid JSONP")
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict):
        raise ValueError("EastMoney news endpoint returned a non-object payload")
    return payload


def fetch_eastmoney_news_text(
    symbol: str,
    *,
    page_size: int = 10,
    session: requests.Session | None = None,
    timeout: float = 20,
    use_system_proxy: bool = False,
) -> str:
    """Fetch EastMoney news search JSONP while preserving the original text."""
    normalized = normalize_symbol(symbol)
    if not normalized:
        raise ValueError(f"Invalid A-share symbol: {symbol}")
    callback = "jQuery_quant_workflow"
    inner_param = {
        "uid": "",
        "keyword": normalized,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": max(1, min(int(page_size), 100)),
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    session, close_session = _open_session(session, use_system_proxy)
    try:
        with proxy_bypass_env(use_system_proxy):
            response = session.get(
                EASTMONEY_NEWS_URL,
                params={
                    "cb": callback,
                    "param": json.dumps(inner_param, ensure_ascii=False, separators=(",", ":")),
                    "_": str(int(time.time() * 1000)),
                },
                headers={
                    "Referer": f"https://so.eastmoney.com/news/s?keyword={normalized}",
                    "User-Agent": DEFAULT_USER_AGENT,
                },
                timeout=timeout,
            )
        response.raise_for_status()
        return response.text
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
