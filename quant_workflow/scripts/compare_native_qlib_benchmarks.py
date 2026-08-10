#!/usr/bin/env python3
"""Compare native Qlib V2 benchmark features with the retained legacy provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (  # noqa: E402
    qlib_compatibility_provider_uri,
    qlib_native_snapshot_uri,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    sha256_file,
)


BENCHMARK_SYMBOLS = ("SH000300", "SH000905", "SH000906")
REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "factor")


def _calendar(provider: Path) -> tuple[str, ...]:
    path = provider / "calendars" / "day.txt"
    try:
        values = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as exc:
        raise DataContractError(f"Qlib calendar is unavailable: {provider}") from exc
    if not values or values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise DataContractError(f"Qlib calendar is invalid: {provider}")
    return values


def _feature_series(
    provider: Path,
    calendar: tuple[str, ...],
    symbol: str,
    field: str,
) -> tuple[dict[str, np.float32], Path]:
    path = provider / "features" / symbol.lower() / f"{field}.day.bin"
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"benchmark feature is unavailable: {symbol}/{field}")
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2 or not np.isfinite(values[0]) or values[0] != int(values[0]):
        raise DataContractError(f"benchmark feature header is invalid: {symbol}/{field}")
    start = int(values[0])
    data = values[1:]
    if start < 0 or start + len(data) > len(calendar):
        raise DataContractError(f"benchmark feature range is invalid: {symbol}/{field}")
    return {
        calendar[start + offset]: value
        for offset, value in enumerate(data)
        if np.isfinite(value)
    }, path


def compare_benchmark_providers(
    native_provider: Path,
    compatibility_provider: Path,
) -> dict[str, Any]:
    native = Path(native_provider).expanduser().resolve()
    compatibility = Path(compatibility_provider).expanduser().resolve()
    native_calendar = _calendar(native)
    compatibility_calendar = _calendar(compatibility)
    symbols: dict[str, Any] = {}

    for symbol in BENCHMARK_SYMBOLS:
        field_results: dict[str, Any] = {}
        close_dates: list[str] = []
        for field in REQUIRED_FIELDS:
            native_values, native_path = _feature_series(
                native, native_calendar, symbol, field
            )
            compatibility_values, compatibility_path = _feature_series(
                compatibility, compatibility_calendar, symbol, field
            )
            native_dates = sorted(native_values)
            compatibility_dates = sorted(compatibility_values)
            if native_dates != compatibility_dates:
                raise DataContractError(
                    f"{symbol}/{field} date coverage differs between native and compatibility providers"
                )
            left = np.asarray(
                [native_values[day] for day in native_dates], dtype="<f4"
            )
            right = np.asarray(
                [compatibility_values[day] for day in native_dates], dtype="<f4"
            )
            if not np.array_equal(left, right):
                raise DataContractError(
                    f"{symbol}/{field} value differs between native and compatibility providers"
                )
            field_results[field] = {
                "observations": len(native_dates),
                "native_sha256": sha256_file(native_path),
                "compatibility_sha256": sha256_file(compatibility_path),
                "identical": True,
            }
            if field == "close":
                close_dates = native_dates
        if not close_dates:
            raise DataContractError(f"{symbol}/close has no comparable observations")
        symbols[symbol] = {
            "coverage": {
                "start": close_dates[0],
                "end": close_dates[-1],
                "observations": len(close_dates),
            },
            "fields": field_results,
        }

    return {
        "status": "pass",
        "native_provider": str(native),
        "compatibility_provider": str(compatibility),
        "calendar_relation": "joined_by_date",
        "comparison": "exact_float32_equality",
        "symbols": symbols,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-provider", type=Path, default=qlib_native_snapshot_uri())
    parser.add_argument(
        "--compatibility-provider",
        type=Path,
        default=qlib_compatibility_provider_uri(),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_benchmark_providers(
            args.native_provider,
            args.compatibility_provider,
        )
        if args.out is not None:
            atomic_write_json(args.out, result)
    except (DataContractError, OSError, ValueError) as exc:
        print(f"Qlib benchmark comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
