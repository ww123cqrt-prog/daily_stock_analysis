#!/usr/bin/env python3
"""Download CSI300/CSI500/CSI800 data without polluting stock pools."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
DATA_DIR = qlib_provider_uri()
BENCHMARKS = {
    "SH000300": "sh.000300",
    "SH000905": "sh.000905",
    "SH000906": "sh.000906",
}
BENCHMARK_SYMBOL = "SH000300"


class BenchmarkPaths:
    def __init__(self, provider_dir: Path):
        self.provider_dir = provider_dir.expanduser()
        self.calendar_file = self.provider_dir / "calendars" / "day.txt"
        self.instruments_file = self.provider_dir / "instruments" / "all.txt"
        self.instruments_dir = self.provider_dir / "instruments"

    def feature_dir(self, symbol: str) -> Path:
        return self.provider_dir / "features" / symbol.lower()


def _load_baostock():
    import baostock as bs

    return bs


def _write_bin(field_path: Path, start_idx: int, values: np.ndarray) -> None:
    field_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = field_path.with_name(f".{field_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.array([start_idx], dtype="<f4").tofile(handle)
        values.astype("<f4").tofile(handle)
    os.replace(temporary, field_path)


def _remove_symbol_if_present(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(f"{symbol}\t")
    ]
    path.write_text("\n".join(lines) + "\n")


def _read_calendar(path: Path) -> list[str]:
    calendar = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calendar:
        raise ValueError(f"empty Qlib calendar: {path}")
    return calendar


def _download_index_data(
    baostock_module, code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    rs = baostock_module.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )

    def raise_for_error() -> None:
        error_code = str(getattr(rs, "error_code", "0"))
        if error_code != "0":
            raise RuntimeError(
                "baostock benchmark query failed: "
                f"{error_code} {getattr(rs, 'error_msg', '')}"
            )

    raise_for_error()
    rows: list[list[str]] = []
    while True:
        has_row = rs.next()
        raise_for_error()
        if not has_row:
            break
        rows.append(rs.get_row_data())
        raise_for_error()
    frame = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    for field in ("open", "high", "low", "close", "volume"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )


def _download_benchmarks(
    baostock_module, start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    lg = baostock_module.login()
    if getattr(lg, "error_code", "0") not in ("0", 0):
        raise RuntimeError(f"baostock login failed: {getattr(lg, 'error_msg', '')}")
    try:
        return {
            symbol: _download_index_data(baostock_module, code, start_date, end_date)
            for symbol, code in BENCHMARKS.items()
        }
    finally:
        baostock_module.logout()


def add_benchmark(
    provider_dir: Path = DATA_DIR,
    *,
    dry_run: bool = False,
    baostock_module=None,
    required_start_date: str = "2014-01-02",
) -> dict[str, str | int]:
    paths = BenchmarkPaths(provider_dir)
    try:
        calendar = _read_calendar(paths.calendar_file)
    except (FileNotFoundError, ValueError) as exc:
        if not dry_run:
            raise
        calendar = []
        print(f"DRY RUN: Qlib calendar unavailable: {exc}")
    if calendar:
        print(
            f"Calendar: {len(calendar)} trading days ({calendar[0]} ~ {calendar[-1]})"
        )
    else:
        print("Calendar: 0 trading days (unavailable)")
    print(f"Provider: {paths.provider_dir}")
    print(
        f"Benchmark feature dirs: {', '.join(str(paths.feature_dir(symbol)) for symbol in BENCHMARKS)}"
    )

    if dry_run:
        print(
            "Dry run OK. No baostock request, bin write, or instruments rewrite will be performed."
        )
        return {
            "status": "dry-run",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "benchmark_symbols": list(BENCHMARKS),
        }

    bs = _load_baostock() if baostock_module is None else baostock_module
    downloaded = _download_benchmarks(bs, calendar[0], calendar[-1])
    downloaded_days: dict[str, int] = {}
    calendar_index = pd.Index(calendar)
    required_start = max(pd.Timestamp(calendar[0]), pd.Timestamp(required_start_date))
    required_calendar = calendar_index[
        pd.to_datetime(calendar_index, errors="coerce") >= required_start
    ]
    if required_calendar.empty:
        raise RuntimeError("Qlib calendar has no dates in the required benchmark period")
    aligned_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, frame in downloaded.items():
        if frame.empty:
            raise RuntimeError(f"no benchmark data returned for {symbol}")
        aligned = frame.reindex(calendar_index)
        price_fields = ["open", "high", "low", "close"]
        required = aligned.loc[required_calendar]
        missing_dates = required.index[
            required[price_fields + ["volume"]].isna().any(axis=1)
        ].tolist()
        if missing_dates:
            raise RuntimeError(
                f"benchmark {symbol} is missing required trading days: "
                f"{missing_dates[:10]}"
            )
        if (required[price_fields] <= 0).any(axis=None) or (
            required["volume"] < 0
        ).any(axis=None):
            raise RuntimeError(f"benchmark OHLCV is invalid for {symbol}")
        aligned_by_symbol[symbol] = aligned
        downloaded_days[symbol] = int(len(frame))

    features_root = paths.provider_dir / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".benchmark-stage-", dir=features_root))
    backup_root = Path(tempfile.mkdtemp(prefix=".benchmark-backup-", dir=features_root))
    installed: list[tuple[Path, Path | None]] = []
    try:
        for symbol, aligned in aligned_by_symbol.items():
            feature_dir = staging_root / symbol.lower()
            for field in ["open", "high", "low", "close", "volume"]:
                _write_bin(
                    feature_dir / f"{field}.day.bin",
                    0,
                    aligned[field].to_numpy(dtype="<f4"),
                )
            _write_bin(
                feature_dir / "factor.day.bin",
                0,
                np.ones(len(calendar), dtype="<f4"),
            )
        for symbol in aligned_by_symbol:
            target = paths.feature_dir(symbol)
            backup = backup_root / symbol.lower()
            previous = backup if target.exists() else None
            if previous is not None:
                os.replace(target, previous)
            try:
                os.replace(staging_root / symbol.lower(), target)
            except Exception:
                if previous is not None and previous.exists():
                    os.replace(previous, target)
                raise
            installed.append((target, previous))
    except Exception:
        for target, previous in reversed(installed):
            shutil.rmtree(target, ignore_errors=True)
            if previous is not None and previous.exists():
                os.replace(previous, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)

    for symbol, frame in downloaded.items():
        print(
            f"Written {symbol}: {len(frame)} downloaded days -> "
            f"{paths.feature_dir(symbol)}"
        )

    # Benchmarks are features only and must never enter any stock training pool.
    for instrument_file in paths.instruments_dir.glob("*.txt"):
        for symbol in BENCHMARKS:
            _remove_symbol_if_present(instrument_file, symbol)
    print("Ensured benchmark symbols are absent from stock instrument pools")

    print("Done!")
    return {
        "status": "completed",
        "provider_dir": str(paths.provider_dir),
        "calendar_days": len(calendar),
        "downloaded_days": downloaded_days,
        "benchmark_symbols": list(BENCHMARKS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add CSI300/CSI500/CSI800 benchmark features to a Qlib provider."
    )
    parser.add_argument("--provider-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--required-start-date", default="2014-01-02")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and calendar without network or writes.",
    )
    args = parser.parse_args(argv)
    add_benchmark(
        args.provider_dir,
        dry_run=args.dry_run,
        required_start_date=args.required_start_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
