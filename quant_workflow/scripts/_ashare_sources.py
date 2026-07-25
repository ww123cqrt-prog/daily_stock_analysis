"""Shared helpers for anonymous A-share daily bar sources.

The helpers deliberately mirror the Baostock downloader schema so additional
sources can be merged without changing the canonical provider directories.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from market_data_clients import compact_date, normalize_date, normalize_symbol, proxy_bypass_env  # noqa: E402


QUANT_DATA = quant_data_root()
STOCK_LIST = QUANT_DATA / "stock_lists.json"
KEEP_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turn",
    "pctChg",
]
REQUIRED_COLUMNS = ["date", "open", "close", "high", "low", "volume"]

COLUMN_ALIASES = {
    "date": ["date", "日期", "交易日期"],
    "open": ["open", "开盘", "开盘价"],
    "close": ["close", "收盘", "收盘价", "最新价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "volume": ["volume", "成交量"],
    "amount": ["amount", "成交额"],
    "turn": ["turn", "换手率"],
    "pctChg": ["pctChg", "涨跌幅", "涨跌幅%"],
}


class EmptyResult(RuntimeError):
    """The source returned no usable daily bars for the requested symbol."""


def load_symbols_file(path: Path) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = normalize_symbol(line)
        if not symbol or symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def jq_code_to_symbol(code: str) -> str:
    return normalize_symbol(code)


def load_symbols(symbols_file: Path | None = None) -> list[str]:
    if symbols_file is not None:
        # A symbols file is an explicit operator-provided download universe. Keep
        # its order and allow symbols that are not yet present in stock_lists.
        return load_symbols_file(symbols_file)

    if STOCK_LIST.exists():
        data = json.loads(STOCK_LIST.read_text(encoding="utf-8"))
        raw_codes = data.get("all", []) if isinstance(data, dict) else []
        symbols = [jq_code_to_symbol(str(code)) for code in raw_codes]
        symbols = [symbol for symbol in symbols if symbol]
    else:
        symbols = []

    return symbols


def _first_present(columns: Iterable[str], aliases: list[str]) -> str | None:
    available = {str(column): str(column) for column in columns}
    for alias in aliases:
        if alias in available:
            return available[alias]
    return None


def normalize_daily_frame(df: pd.DataFrame, *, volume_multiplier: float = 1.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    renamed: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        source = _first_present(df.columns, aliases)
        if source is not None:
            renamed[source] = target

    normalized = df.rename(columns=renamed).copy()
    keep = [column for column in KEEP_COLUMNS if column in normalized.columns]
    if "date" not in keep:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    normalized = normalized[keep]
    normalized["date"] = pd.to_datetime(normalized["date"].map(normalize_date), errors="coerce")
    for column in keep:
        if column == "date":
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "volume" in normalized.columns and volume_multiplier != 1:
        normalized["volume"] = normalized["volume"] * volume_multiplier

    missing_required = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_required:
        return pd.DataFrame(columns=KEEP_COLUMNS)

    normalized = normalized.dropna(subset=REQUIRED_COLUMNS)
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return normalized[[column for column in KEEP_COLUMNS if column in normalized.columns]]


def incremental_start_for_file(path: Path, default_start: str, overlap_days: int) -> str:
    default_ts = pd.Timestamp(normalize_date(default_start))
    if not path.exists():
        return str(default_ts.date())
    try:
        dates = pd.read_parquet(path, columns=["date"])
    except Exception:
        return str(default_ts.date())
    series = pd.to_datetime(dates["date"], errors="coerce").dropna()
    if series.empty:
        return str(default_ts.date())
    start = max(series.max() - pd.Timedelta(days=overlap_days), default_ts)
    return str(start.date())


def range_starts_after_end(start: str, end: str) -> bool:
    start_ts = pd.to_datetime(normalize_date(start), errors="coerce")
    end_ts = pd.to_datetime(normalize_date(end), errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return False
    return start_ts > end_ts


def merge_existing_frame(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return combined


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_daily_download(
    *,
    source_name: str,
    output_dir: Path,
    symbols: list[str],
    fetch_one: Callable[[str, str, str], pd.DataFrame],
    start: str,
    end: str,
    incremental: bool,
    overlap_days: int,
    sleep_seconds: float,
    summary_dir: Path | None = None,
    runtime_status_dir: Path | None = None,
    retry_failed: bool = False,
    max_consecutive_failures: int = 50,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = reports_root()
    summary_root = summary_dir or (report_root / "download_state")
    runtime_root = runtime_status_dir or (report_root / "runtime" / "status")
    if retry_failed:
        existing = {path.stem for path in output_dir.glob("*.parquet")}
        symbols = [symbol for symbol in symbols if symbol not in existing]

    ok = 0
    empty = 0
    fail = 0
    no_new = 0
    failures: list[dict[str, str]] = []
    consecutive_failures = 0
    total = len(symbols)
    t0 = time.time()

    print(
        f"Downloading {total} stocks from {source_name}, {normalize_date(start)} -> {normalize_date(end)}, out={output_dir}",
        flush=True,
    )

    for index, symbol in enumerate(symbols, 1):
        out_path = output_dir / f"{symbol}.parquet"
        symbol_start = (
            incremental_start_for_file(out_path, start, overlap_days)
            if incremental
            else normalize_date(start)
        )
        current_path = summary_root / f"{source_name}_current.json"
        write_json(
            current_path,
            {
                "index": index,
                "total": total,
                "symbol": symbol,
                "source": source_name,
                "start": symbol_start,
                "end": normalize_date(end),
                "incremental": incremental,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        if incremental and out_path.exists() and range_starts_after_end(symbol_start, end):
            no_new += 1
            consecutive_failures = 0
            if no_new <= 5 or no_new % 50 == 0:
                print(
                    f"NO_NEW {symbol} from {source_name}: cached through {symbol_start}; end={normalize_date(end)}",
                    flush=True,
                )
            time.sleep(sleep_seconds)
            continue

        try:
            fresh = normalize_daily_frame(fetch_one(symbol, symbol_start, normalize_date(end)))
            if fresh.empty:
                if incremental and out_path.exists():
                    no_new += 1
                    consecutive_failures = 0
                    continue
                raise EmptyResult("empty result")

            if incremental and out_path.exists():
                fresh = merge_existing_frame(pd.read_parquet(out_path), fresh)

            fresh.to_parquet(out_path, index=False)
            ok += 1
            consecutive_failures = 0
            if index <= 3 or index % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(index, 1) * (total - index)
                print(
                    f"[{index}/{total}] {symbol}: {len(fresh)} rows "
                    f"({fresh['date'].min().date()} ~ {fresh['date'].max().date()}) "
                    f"start={symbol_start} ETA {eta / 60:.0f}m",
                    flush=True,
                )
        except EmptyResult as exc:
            empty += 1
            consecutive_failures = 0
            if empty <= 5 or empty % 50 == 0:
                print(f"EMPTY {symbol}: {exc}", file=sys.stderr, flush=True)
        except Exception as exc:
            fail += 1
            consecutive_failures += 1
            failures.append({"symbol": symbol, "error": str(exc)[:200]})
            if fail <= 5 or fail % 50 == 0:
                print(f"FAIL {symbol}: {exc}", file=sys.stderr, flush=True)
            if consecutive_failures >= max_consecutive_failures:
                print(
                    f"Stopping after {consecutive_failures} consecutive failures.",
                    file=sys.stderr,
                    flush=True,
                )
                break
        time.sleep(sleep_seconds)

    elapsed = time.time() - t0
    summary = {
        "source": source_name,
        "output_dir": str(output_dir),
        "start": normalize_date(start),
        "end": normalize_date(end),
        "total": total,
        "ok": ok,
        "no_new": no_new,
        "empty": empty,
        "fail": fail,
        "seconds": round(elapsed, 2),
        "failures": failures[:100],
    }
    write_json(summary_root / f"{source_name}_download_summary_latest.json", summary)
    write_json(runtime_root / f"{source_name}_download_summary_latest.json", summary)
    if failures:
        write_json(summary_root / f"{source_name}_failures.json", failures)
    write_json(runtime_root / f"{source_name}_failures_latest.json", failures)
    print(
        f"Done {source_name} in {elapsed / 60:.1f} min. OK={ok}, NO_NEW={no_new}, EMPTY={empty}, FAIL={fail}",
        flush=True,
    )
    return summary
