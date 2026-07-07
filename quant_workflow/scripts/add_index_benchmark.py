#!/usr/bin/env python3
"""Download CSI300 index data and add benchmark features without polluting stock pools."""

from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
DATA_DIR = qlib_provider_uri()
BENCHMARK_SYMBOL = "SH000300"


class BenchmarkPaths:
    def __init__(self, provider_dir: Path):
        self.provider_dir = provider_dir.expanduser()
        self.calendar_file = self.provider_dir / "calendars" / "day.txt"
        self.instruments_file = self.provider_dir / "instruments" / "all.txt"
        self.csi300_file = self.provider_dir / "instruments" / "csi300.txt"
        self.feature_dir = self.provider_dir / "features" / "sh000300"


def _load_baostock():
    import baostock as bs

    return bs


def _write_bin(field_path: Path, start_idx: int, values: np.ndarray) -> None:
    with field_path.open("wb") as handle:
        np.array([start_idx], dtype="<f4").tofile(handle)
        values.astype("<f4").tofile(handle)


def _remove_symbol_if_present(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    lines = [line for line in path.read_text().splitlines() if not line.startswith(f"{symbol}\t")]
    path.write_text("\n".join(lines) + "\n")


def _read_calendar(path: Path) -> list[str]:
    calendar = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not calendar:
        raise ValueError(f"empty Qlib calendar: {path}")
    return calendar


def _download_index_data(baostock_module, start_date: str, end_date: str) -> dict[str, float]:
    lg = baostock_module.login()
    if getattr(lg, "error_code", "0") not in ("0", 0):
        raise RuntimeError(f"baostock login failed: {getattr(lg, 'error_msg', '')}")
    try:
        rs = baostock_module.query_history_k_data_plus(
            "sh.000300",
            "date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        data: dict[str, float] = {}
        while rs.next():
            row = rs.get_row_data()
            if row[1]:
                data[row[0]] = float(row[1])
        return data
    finally:
        baostock_module.logout()


def add_benchmark(
    provider_dir: Path = DATA_DIR,
    *,
    dry_run: bool = False,
    baostock_module=None,
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
        print(f"Calendar: {len(calendar)} trading days ({calendar[0]} ~ {calendar[-1]})")
    else:
        print("Calendar: 0 trading days (unavailable)")
    print(f"Provider: {paths.provider_dir}")
    print(f"Benchmark feature dir: {paths.feature_dir}")

    if dry_run:
        print("Dry run OK. No baostock request, bin write, or instruments rewrite will be performed.")
        return {
            "status": "dry-run",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "benchmark_symbol": BENCHMARK_SYMBOL,
        }

    bs = _load_baostock() if baostock_module is None else baostock_module
    data = _download_index_data(bs, calendar[0], calendar[-1])
    print(f"Downloaded {len(data)} days of SH000300 data")

    close_arr = np.zeros(len(calendar), dtype="<f4")
    for i, day in enumerate(calendar):
        if day in data:
            close_arr[i] = data[day]
        else:
            # forward fill for missing days
            close_arr[i] = close_arr[i - 1] if i > 0 else 0.0

    positive = close_arr[close_arr > 0]
    if len(positive):
        print(f"Close range: {positive.min():.2f} ~ {close_arr.max():.2f}")
    else:
        print("Close range: no positive close values")

    paths.feature_dir.mkdir(parents=True, exist_ok=True)

    # Qlib bin format: first float = start index, followed by per-day values.
    start_idx = 0
    _write_bin(paths.feature_dir / "close.day.bin", start_idx, close_arr)
    _write_bin(paths.feature_dir / "volume.day.bin", start_idx, np.zeros(len(calendar), dtype="<f4"))
    _write_bin(paths.feature_dir / "factor.day.bin", start_idx, np.ones(len(calendar), dtype="<f4"))
    _write_bin(paths.feature_dir / "open.day.bin", start_idx, close_arr)
    _write_bin(paths.feature_dir / "high.day.bin", start_idx, close_arr)
    _write_bin(paths.feature_dir / "low.day.bin", start_idx, close_arr)
    print(f"Written bin files to {paths.feature_dir}")

    # Keep benchmark data in features, but never add it to training stock pools.
    _remove_symbol_if_present(paths.instruments_file, BENCHMARK_SYMBOL)
    _remove_symbol_if_present(paths.csi300_file, BENCHMARK_SYMBOL)
    print("Ensured SH000300 is not present in all.txt or csi300.txt")

    print("Done!")
    return {
        "status": "completed",
        "provider_dir": str(paths.provider_dir),
        "calendar_days": len(calendar),
        "downloaded_days": len(data),
        "benchmark_symbol": BENCHMARK_SYMBOL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add SH000300 benchmark features to a Qlib provider.")
    parser.add_argument("--provider-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and calendar without network or writes.")
    args = parser.parse_args(argv)
    add_benchmark(args.provider_dir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
