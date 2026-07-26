#!/usr/bin/env python3
"""Download qfq A-share daily bars from EastMoney's public kline endpoint."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ashare_sources import (  # noqa: E402
    QUANT_DATA,
    load_symbols,
    normalize_daily_frame,
    normalize_date,
    run_daily_download,
)
from market_data_clients import eastmoney_secid, fetch_eastmoney_qfq_daily  # noqa: E402


PARQUET_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "eastmoney_qfq"


def secid_for_symbol(symbol: str) -> str:
    return eastmoney_secid(symbol)


def fetch_one(
    symbol: str,
    start: str,
    end: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30,
    use_system_proxy: bool = False,
) -> "pd.DataFrame":
    raw = fetch_eastmoney_qfq_daily(
        symbol,
        start,
        end,
        session=session,
        timeout=timeout,
        use_system_proxy=use_system_proxy,
    )
    # EastMoney reports volume in hands; Baostock schema stores shares.
    return normalize_daily_frame(raw, volume_multiplier=100)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download EastMoney qfq A-share daily bars.")
    parser.add_argument("--start", default="19900101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true", help="Only download symbols not yet cached.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30, help="HTTP timeout in seconds.")
    parser.add_argument("--max-consecutive-failures", type=int, default=50)
    parser.add_argument("--symbols-file", type=Path, help="Optional text file of six-digit symbols to download.")
    parser.add_argument("--incremental", action="store_true", help="Merge fresh rows into existing parquet files.")
    parser.add_argument("--overlap-days", type=int, default=30, help="Calendar-day overlap used in incremental mode.")
    parser.add_argument("--use-system-proxy", action="store_true", help="Opt into macOS/system proxy settings.")
    parser.add_argument("--out", "--output-dir", dest="out", type=Path, default=PARQUET_DIR)
    args = parser.parse_args()

    symbols = load_symbols(args.symbols_file)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    session = requests.Session()
    session.trust_env = args.use_system_proxy

    def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        last_error: Exception | None = None
        retry_count = max(args.retries, 1)
        for attempt in range(1, retry_count + 1):
            try:
                return fetch_one(
                    symbol,
                    start,
                    end,
                    session=session,
                    timeout=args.timeout,
                    use_system_proxy=args.use_system_proxy,
                )
            except Exception as exc:  # pragma: no cover - exercised by integration runs
                last_error = exc
                if attempt < retry_count:
                    time.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError(str(last_error or "EastMoney request failed"))

    try:
        summary = run_daily_download(
            source_name="eastmoney",
            output_dir=args.out,
            symbols=symbols,
            fetch_one=fetch,
            start=normalize_date(args.start),
            end=normalize_date(args.end),
            incremental=args.incremental,
            overlap_days=args.overlap_days,
            sleep_seconds=args.sleep,
            retry_failed=args.retry_failed,
            max_consecutive_failures=args.max_consecutive_failures,
            price_basis="qfq",
        )
    finally:
        session.close()

    return 0 if summary["ok"] > 0 or summary["no_new"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
