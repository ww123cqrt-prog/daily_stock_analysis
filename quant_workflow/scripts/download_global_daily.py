#!/usr/bin/env python3
"""Download canonical US/Japan/Korea daily bars with raw Yahoo evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import write_json_atomic  # noqa: E402
from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402
from market_data_clients import fetch_yahoo_chart_payload, normalize_date, parse_yahoo_chart_payload  # noqa: E402


QUANT_DATA = quant_data_root()
DEFAULT_SYMBOLS = workflow_root() / "configs" / "global_daily_symbols.json"
OUTPUT_ROOT = QUANT_DATA / "parquet" / "global_1d"
RAW_ROOT = QUANT_DATA / "raw" / "global_1d" / "yahoo"
SUMMARY_ROOT = reports_root() / "download_state"
SUPPORTED_MARKETS = ("us", "jp", "kr")
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.^=-]*$")


def load_universe(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    markets = payload.get("markets") if isinstance(payload, dict) else None
    if not isinstance(markets, dict):
        raise ValueError("global symbol config must contain a markets object")
    result: dict[str, list[str]] = {}
    for market in SUPPORTED_MARKETS:
        raw = markets.get(market, [])
        if not isinstance(raw, list):
            raise ValueError(f"market {market} must be a list")
        values: list[str] = []
        for item in raw:
            symbol = str(item).strip().upper()
            if not SYMBOL_RE.fullmatch(symbol) or ".." in symbol:
                raise ValueError(f"invalid global symbol: {item}")
            if symbol not in values:
                values.append(symbol)
        result[market] = values
    return result


def _flatten(universe: dict[str, list[str]], limit: int) -> list[tuple[str, str]]:
    values = [(market, symbol) for market in SUPPORTED_MARKETS for symbol in universe.get(market, [])]
    return values[:limit] if limit > 0 else values


def _incremental_start(path: Path, default_start: str, overlap_days: int) -> str:
    baseline = pd.Timestamp(normalize_date(default_start))
    if not path.exists():
        return str(baseline.date())
    try:
        dates = pd.read_parquet(path, columns=["date"])
        latest = pd.to_datetime(dates["date"], errors="coerce").dropna().max()
    except Exception:
        return str(baseline.date())
    if pd.isna(latest):
        return str(baseline.date())
    return str(max(latest - pd.Timedelta(days=max(overlap_days, 0)), baseline).date())


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_raw_capture(
    *,
    raw_root: Path,
    market: str,
    symbol: str,
    start: str,
    end: str,
    payload: dict[str, Any],
    captured_at: datetime,
) -> Path:
    stamp = captured_at.strftime("%Y%m%dT%H%M%S_%f%z")
    canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = (
        raw_root
        / f"captured_date={captured_at.date().isoformat()}"
        / market
        / symbol
        / f"{stamp}.json"
    )
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "source": "yahoo_chart",
            "market": market,
            "symbol": symbol,
            "captured_at": captured_at.isoformat(timespec="microseconds"),
            "payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
            "request": {"start": normalize_date(start), "end": normalize_date(end), "interval": "1d"},
            "payload": payload,
        },
    )
    return path


def run_download(
    *,
    universe: dict[str, list[str]],
    output_root: Path,
    raw_root: Path,
    summary_dir: Path,
    start: str,
    end: str,
    incremental: bool,
    overlap_days: int,
    limit: int,
    sleep_seconds: float,
    fetch_payload: Callable[[str, str, str], dict[str, Any]],
    runtime_status_dir: Path | None = None,
) -> dict[str, Any]:
    items = _flatten(universe, limit)
    ok = no_new = empty = fail = 0
    failures: list[dict[str, str]] = []
    started = time.time()

    for market, symbol in items:
        path = output_root / market / f"{symbol}.parquet"
        symbol_start = _incremental_start(path, start, overlap_days) if incremental else normalize_date(start)
        if pd.Timestamp(symbol_start) > pd.Timestamp(normalize_date(end)):
            no_new += 1
            continue
        try:
            payload = fetch_payload(symbol, symbol_start, normalize_date(end))
            captured_at = datetime.now().astimezone()
            _write_raw_capture(
                raw_root=raw_root,
                market=market,
                symbol=symbol,
                start=symbol_start,
                end=end,
                payload=payload,
                captured_at=captured_at,
            )
            fresh = parse_yahoo_chart_payload(payload)
            if fresh.empty:
                if path.exists():
                    no_new += 1
                else:
                    empty += 1
                continue
            if incremental and path.exists():
                existing = pd.read_parquet(path)
                fresh = pd.concat([existing, fresh], ignore_index=True)
                fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
                fresh = fresh.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
            _write_parquet_atomic(path, fresh)
            ok += 1
        except Exception as exc:
            fail += 1
            failures.append({"market": market, "symbol": symbol, "error": str(exc)[:300]})
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    result = {
        "source": "global_daily",
        "provider": "yahoo_chart",
        "output_dir": str(output_root),
        "raw_dir": str(raw_root),
        "start": normalize_date(start),
        "end": normalize_date(end),
        "total": len(items),
        "ok": ok,
        "no_new": no_new,
        "empty": empty,
        "fail": fail,
        "seconds": round(time.time() - started, 2),
        "failures": failures,
    }
    write_json_atomic(summary_dir / "global_daily_download_summary_latest.json", result)
    if runtime_status_dir is not None:
        write_json_atomic(runtime_status_dir / "global_daily_download_summary_latest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Download canonical global daily bars from Yahoo Chart.")
    parser.add_argument("--symbols-file", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_ROOT)
    args = parser.parse_args()

    try:
        from curl_cffi import requests as curl_requests  # type: ignore

        session = curl_requests.Session(impersonate="chrome")
    except ImportError:
        session = requests.Session()
    session.trust_env = args.use_system_proxy

    def fetch(symbol: str, start: str, end: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, max(args.retries, 1) + 1):
            try:
                return fetch_yahoo_chart_payload(
                    symbol,
                    start,
                    end,
                    session=session,
                    timeout=args.timeout,
                    use_system_proxy=args.use_system_proxy,
                )
            except Exception as exc:
                last_error = exc
                if attempt < max(args.retries, 1):
                    time.sleep(min(2 ** (attempt - 1), 10))
        raise RuntimeError(str(last_error or "Yahoo Chart request failed"))

    try:
        result = run_download(
            universe=load_universe(args.symbols_file),
            output_root=args.out,
            raw_root=args.raw_dir,
            summary_dir=args.summary_dir,
            runtime_status_dir=reports_root() / "runtime" / "status",
            start=args.start,
            end=args.end,
            incremental=args.incremental,
            overlap_days=args.overlap_days,
            limit=args.limit,
            sleep_seconds=args.sleep,
            fetch_payload=fetch,
        )
    finally:
        session.close()

    print(
        f"Global daily download: OK={result['ok']} NO_NEW={result['no_new']} "
        f"EMPTY={result['empty']} FAIL={result['fail']}"
    )
    return 0 if result["ok"] > 0 or result["no_new"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
