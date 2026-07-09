"""
Download A-share daily bars via Baostock.

This is the non-EastMoney fallback for long-history OHLCV. It reads the same
JQData-generated stock list used by the akshare downloader and writes one
Parquet file per 6-digit stock code.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402


QUANT_DATA = quant_data_root()
STOCK_LIST = QUANT_DATA / "stock_lists.json"
PARQUET_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "baostock_qfq"
EMPTY_CODES_FILE = QUANT_DATA / "baostock_empty_codes.json"
TIMEOUT_CODES_FILE = QUANT_DATA / "baostock_timeout_codes.json"
CURRENT_FILE = QUANT_DATA / "baostock_current.json"

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
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


class EmptyResult(RuntimeError):
    """Baostock returned successfully but has no rows for the symbol/date range."""


class QueryTimeout(TimeoutError):
    """A single Baostock symbol query exceeded the configured timeout."""


def _timeout_handler(signum, frame):
    raise QueryTimeout("single-symbol query timed out")


def normalize_date(value: str) -> str:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def jq_code_to_baostock(code: str) -> tuple[str, str]:
    """Return (six_digit_symbol, baostock_symbol)."""
    raw = code.strip()
    if "." in raw:
        sym, exchange = raw.split(".", 1)
        exchange = exchange.upper()
        if exchange == "XSHG":
            return sym, f"sh.{sym}"
        if exchange == "XSHE":
            return sym, f"sz.{sym}"
    else:
        sym = raw

    if sym.startswith(("600", "601", "603", "605", "688", "689")):
        return sym, f"sh.{sym}"
    return sym, f"sz.{sym}"


def normalize_symbol(value: str) -> str:
    raw = value.split("#", 1)[0].strip().upper()
    if not raw:
        return ""
    raw = raw.replace(",", " ").split()[0]
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    if raw.isdigit() and len(raw) < 6:
        raw = raw.zfill(6)
    return raw


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


def filter_codes_by_symbols(
    codes: list[tuple[str, str]],
    symbols: list[str],
) -> list[tuple[str, str]]:
    code_map = {sym: (sym, bs_code) for sym, bs_code in codes}
    filtered: list[tuple[str, str]] = []
    for symbol in symbols:
        filtered.append(code_map.get(symbol, jq_code_to_baostock(symbol)))
    return filtered


def load_codes(symbols_file: Path | None = None) -> list[tuple[str, str]]:
    stocks_data = json.loads(STOCK_LIST.read_text(encoding="utf-8"))
    codes = [jq_code_to_baostock(code) for code in stocks_data["all"]]
    if symbols_file is None:
        return codes
    symbols = load_symbols_file(symbols_file)
    return filter_codes_by_symbols(codes, symbols)


def load_empty_codes() -> set[str]:
    if not EMPTY_CODES_FILE.exists():
        return set()
    try:
        data = json.loads(EMPTY_CODES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(data if isinstance(data, list) else [])


def load_timeout_codes() -> set[str]:
    if not TIMEOUT_CODES_FILE.exists():
        return set()
    try:
        data = json.loads(TIMEOUT_CODES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(data if isinstance(data, list) else [])


def save_empty_codes(codes: set[str]) -> None:
    EMPTY_CODES_FILE.write_text(
        json.dumps(sorted(codes), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def frame_from_result(rs) -> pd.DataFrame:
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def query_one(
    bs,
    bs_code: str,
    start: str,
    end: str,
    adjustflag: str,
    retries: int,
    timeout: int,
) -> pd.DataFrame:
    last_error = ""
    for attempt in range(1, retries + 1):
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields=FIELDS,
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag=adjustflag,
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

        if rs.error_code == "0":
            return frame_from_result(rs)

        last_error = f"{rs.error_code}: {rs.error_msg}"
        time.sleep(min(2 * attempt, 10))

    raise RuntimeError(last_error or "Baostock query failed")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in KEEP_COLUMNS:
        if col != "date" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    df = df[[col for col in KEEP_COLUMNS if col in df.columns]]
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    return df


def incremental_start_for_file(path: Path, default_start: str, overlap_days: int) -> str:
    default_ts = pd.Timestamp(default_start)
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
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return False
    return start_ts > end_ts


def merge_existing_frame(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20250611")
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true", help="Only download stocks not yet cached")
    parser.add_argument("--retry-empty", action="store_true", help="Retry symbols previously recorded as empty")
    parser.add_argument("--retry-timeout", action="store_true", help="Retry symbols previously recorded as timed out")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=45, help="Seconds before aborting one symbol query")
    parser.add_argument("--max-consecutive-failures", type=int, default=50)
    parser.add_argument("--symbols-file", type=Path, help="Optional text file of six-digit symbols to download")
    parser.add_argument("--incremental", action="store_true", help="Merge fresh rows into existing parquet files")
    parser.add_argument("--overlap-days", type=int, default=30, help="Calendar-day overlap used in incremental mode")
    parser.add_argument(
        "--adjustflag",
        default="2",
        choices=["1", "2", "3"],
        help="Baostock adjustflag: 1=hfq, 2=qfq, 3=raw",
    )
    parser.add_argument("--out", type=Path, default=PARQUET_DIR)
    args = parser.parse_args()

    start = normalize_date(args.start)
    end = normalize_date(args.end)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_codes(args.symbols_file)
    if args.symbols_file:
        print(f"Symbols file: {args.symbols_file} ({len(codes)} symbols)", flush=True)
    empty_codes = load_empty_codes()
    timeout_codes = load_timeout_codes()
    if args.retry_failed:
        existing = {path.stem for path in out_dir.glob("*.parquet")}
        codes = [(sym, bs_code) for sym, bs_code in codes if sym not in existing]
        if not args.retry_empty and empty_codes:
            codes = [(sym, bs_code) for sym, bs_code in codes if sym not in empty_codes]
        if not args.retry_timeout and timeout_codes:
            codes = [(sym, bs_code) for sym, bs_code in codes if sym not in timeout_codes]
        print(f"Retry mode: {len(codes)} stocks remaining", flush=True)

    if args.limit > 0:
        codes = codes[: args.limit]

    import baostock as bs

    login_result = bs.login()
    print(f"Baostock login: {login_result.error_code} {login_result.error_msg}", flush=True)
    if login_result.error_code != "0":
        return 2

    ok = 0
    empty = 0
    fail = 0
    no_new = 0
    consecutive_failures = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    try:
        total = len(codes)
        print(
            f"Downloading {total} stocks, {start} -> {end}, adjustflag={args.adjustflag}, out={out_dir}",
            flush=True,
        )
        for i, (sym, bs_code) in enumerate(codes, 1):
            try:
                out_path = out_dir / f"{sym}.parquet"
                symbol_start = (
                    incremental_start_for_file(out_path, start, args.overlap_days)
                    if args.incremental
                    else start
                )
                CURRENT_FILE.write_text(
                    json.dumps(
                        {
                            "index": i,
                            "total": total,
                            "symbol": sym,
                            "baostock_code": bs_code,
                            "start": symbol_start,
                            "end": end,
                            "incremental": args.incremental,
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if args.incremental and out_path.exists() and range_starts_after_end(symbol_start, end):
                    no_new += 1
                    consecutive_failures = 0
                    if no_new <= 5 or no_new % 50 == 0:
                        print(
                            f"NO_NEW {sym} ({bs_code}) cached through {symbol_start}; end={end}",
                            flush=True,
                        )
                    continue

                df = query_one(bs, bs_code, symbol_start, end, args.adjustflag, args.retries, args.timeout)
                df = normalize_frame(df)
                if df.empty:
                    if args.incremental and out_path.exists():
                        no_new += 1
                        consecutive_failures = 0
                        if no_new <= 5 or no_new % 50 == 0:
                            print(f"NO_NEW {sym} ({bs_code}) from {symbol_start} to {end}", flush=True)
                        continue
                    raise EmptyResult("empty result")

                if args.incremental and out_path.exists():
                    existing = pd.read_parquet(out_path)
                    df = merge_existing_frame(existing, df)

                df.to_parquet(out_path, index=False)
                ok += 1
                consecutive_failures = 0

                if i <= 3 or i % 100 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / max(i, 1) * (total - i)
                    print(
                        f"[{i}/{total}] {sym}: {len(df)} rows "
                        f"({df['date'].min().date()} ~ {df['date'].max().date()}) "
                        f"start={symbol_start} "
                        f"ETA {eta / 60:.0f}m",
                        flush=True,
                    )
            except EmptyResult as exc:
                empty += 1
                empty_codes.add(sym)
                save_empty_codes(empty_codes)
                consecutive_failures = 0
                if empty <= 5 or empty % 50 == 0:
                    print(f"EMPTY {sym} ({bs_code}): {exc}", file=sys.stderr, flush=True)
            except Exception as exc:
                fail += 1
                consecutive_failures += 1
                failures.append((sym, bs_code, str(exc)[:160]))
                if fail <= 5 or fail % 50 == 0:
                    print(f"FAIL {sym} ({bs_code}): {exc}", file=sys.stderr, flush=True)
                if consecutive_failures >= args.max_consecutive_failures:
                    print(
                        f"Stopping after {consecutive_failures} consecutive failures.",
                        file=sys.stderr,
                        flush=True,
                    )
                    break

            time.sleep(args.sleep)
    finally:
        bs.logout()

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed / 60:.1f} min. OK={ok}, NO_NEW={no_new}, EMPTY={empty}, FAIL={fail}",
        flush=True,
    )
    if failures:
        fail_log = QUANT_DATA / "baostock_failures.json"
        fail_log.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Failure details saved to {fail_log}", flush=True)

    return 0 if ok > 0 or no_new > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
