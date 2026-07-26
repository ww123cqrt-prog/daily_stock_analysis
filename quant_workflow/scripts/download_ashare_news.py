#!/usr/bin/env python3
"""Download bounded A-share news responses as traceable raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ashare_sources import is_source_denial_error, load_symbols, run_source_entrypoint  # noqa: E402
from _daily_update_runtime import write_json_atomic  # noqa: E402
from _paths import quant_data_root, reports_root  # noqa: E402
from market_data_clients import fetch_eastmoney_news_text, parse_eastmoney_news_jsonp  # noqa: E402


RAW_ROOT = quant_data_root() / "raw" / "ashare_news"
SUMMARY_ROOT = reports_root() / "download_state"


def validate_bounded_request(*, symbols_file: Path | None, limit: int) -> None:
    if symbols_file is None and limit <= 0:
        raise ValueError("A-share news download must use a bounded --limit or an explicit --symbols-file")


def run_download(
    *,
    symbols: list[str],
    raw_root: Path,
    summary_dir: Path,
    limit: int,
    sleep_seconds: float,
    fetch_text: Callable[[str, int], str],
    page_size: int = 10,
    runtime_status_dir: Path | None = None,
) -> dict[str, Any]:
    selected = symbols[:limit] if limit > 0 else symbols
    ok = fail = 0
    failures: list[dict[str, str]] = []
    consecutive_failures = 0
    stopped_reason = "completed"
    started = time.time()
    for symbol in selected:
        try:
            raw_text = fetch_text(symbol, page_size)
            payload = parse_eastmoney_news_jsonp(raw_text)
            captured_at = datetime.now().astimezone()
            path = (
                raw_root
                / f"captured_date={captured_at.date().isoformat()}"
                / symbol
                / f"{captured_at.strftime('%Y%m%dT%H%M%S_%f%z')}.json"
            )
            write_json_atomic(
                path,
                {
                    "schema_version": 1,
                    "source": "eastmoney_news_search",
                    "symbol": symbol,
                    "captured_at": captured_at.isoformat(timespec="microseconds"),
                    "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                    "request": {"page_index": 1, "page_size": page_size},
                    "raw_text": raw_text,
                    "payload": payload,
                },
            )
            ok += 1
            consecutive_failures = 0
        except Exception as exc:
            fail += 1
            consecutive_failures += 1
            failures.append({"symbol": symbol, "error": str(exc)[:300]})
            if is_source_denial_error(exc):
                stopped_reason = "source_denied"
                break
            if consecutive_failures >= 3:
                stopped_reason = "consecutive_failures"
                break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    result = {
        "source": "ashare_news",
        "provider": "eastmoney_news_search",
        "raw_dir": str(raw_root),
        "total": len(selected),
        "ok": ok,
        "fail": fail,
        "stopped_reason": stopped_reason,
        "seconds": round(time.time() - started, 2),
        "failures": failures,
    }
    write_json_atomic(summary_dir / "ashare_news_download_summary_latest.json", result)
    if runtime_status_dir is not None:
        write_json_atomic(runtime_status_dir / "ashare_news_download_summary_latest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download bounded EastMoney A-share news raw responses.")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_ROOT)
    args = parser.parse_args()
    validate_bounded_request(symbols_file=args.symbols_file, limit=args.limit)
    if args.sleep < 2:
        parser.error("sleep must be at least 2 seconds")
    if args.retries != 1:
        parser.error("retries must be exactly 1")
    if args.limit > 6000:
        parser.error("limit must be at most 6000")
    symbols = load_symbols(args.symbols_file)

    session = requests.Session()
    session.trust_env = args.use_system_proxy

    def fetch(symbol: str, page_size: int) -> str:
        last_error: Exception | None = None
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                return fetch_eastmoney_news_text(
                    symbol,
                    page_size=page_size,
                    session=session,
                    timeout=args.timeout,
                    use_system_proxy=args.use_system_proxy,
                )
            except Exception as exc:
                last_error = exc
                if attempt < max(args.retries, 1):
                    time.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError(str(last_error or "EastMoney news request failed"))

    try:
        result = run_download(
            symbols=symbols,
            raw_root=args.raw_dir,
            summary_dir=args.summary_dir,
            runtime_status_dir=reports_root() / "runtime" / "status",
            limit=args.limit,
            sleep_seconds=args.sleep,
            page_size=max(1, min(args.page_size, 100)),
            fetch_text=fetch,
        )
    finally:
        session.close()

    print(f"A-share news download: OK={result['ok']} FAIL={result['fail']}")
    return 0 if result["ok"] > 0 and result["stopped_reason"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run_source_entrypoint("eastmoney_news", main))
