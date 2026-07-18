#!/usr/bin/env python3
"""Validate and fingerprint an isolated PIT price parquet bundle.

The validator is used both on the remote download directory and after the
bundle is transferred locally.  It fails closed: a manifest is published only
after the expected symbol set, parquet schema, date range, OHLCV values, and
file hashes all pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
MANIFEST_VERSION = 1


class PriceBundleError(ValueError):
    def __init__(self, reason_code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _atomic_write(
        Path(path),
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_symbols(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise PriceBundleError("symbols_file_missing", f"symbols file is missing: {path}")
    symbols: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.split("#", 1)[0].strip().upper()
        if not value:
            continue
        if not value.isdigit() or len(value) != 6:
            raise PriceBundleError(
                "symbols_file_invalid",
                f"invalid six-digit symbol at line {line_number}",
                line_number=line_number,
                value=value,
            )
        if value in seen:
            raise PriceBundleError(
                "symbols_file_duplicate",
                f"duplicate symbol in symbols file: {value}",
                symbol=value,
            )
        symbols.append(value)
        seen.add(value)
    if not symbols:
        raise PriceBundleError("symbols_file_empty", "symbols file has no symbols")
    return sorted(symbols)


def _actual_price_files(price_dir: Path) -> dict[str, Path]:
    if price_dir.is_symlink() or not price_dir.is_dir():
        raise PriceBundleError("price_dir_missing", f"price directory is missing: {price_dir}")
    files: dict[str, Path] = {}
    for path in sorted(price_dir.iterdir()):
        if path.is_symlink():
            raise PriceBundleError("symlink_forbidden", f"price bundle contains symlink: {path.name}")
        if not path.is_file():
            raise PriceBundleError("non_regular_file", f"price bundle contains non-file: {path.name}")
        if path.suffix.lower() != ".parquet":
            raise PriceBundleError("unexpected_file", f"price bundle contains non-parquet file: {path.name}")
        symbol = path.stem.upper()
        if not symbol.isdigit() or len(symbol) != 6:
            raise PriceBundleError("invalid_price_filename", f"invalid parquet filename: {path.name}")
        if symbol in files:
            raise PriceBundleError("duplicate_price_filename", f"duplicate parquet filename: {path.name}")
        files[symbol] = path
    return files


def _date_bound(value: str, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value).normalize()
    if pd.isna(parsed):
        raise PriceBundleError("invalid_date_range", f"invalid {label}: {value}")
    return parsed


def _validate_frame(
    path: Path,
    symbol: str,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    except Exception as exc:
        raise PriceBundleError(
            "unreadable_parquet",
            f"cannot read parquet for {symbol}: {type(exc).__name__}",
            symbol=symbol,
        ) from exc
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PriceBundleError(
            "missing_required_columns",
            f"{symbol} is missing required columns",
            symbol=symbol,
            missing_columns=missing,
        )
    if frame.empty:
        raise PriceBundleError("empty_price_file", f"{symbol} has no rows", symbol=symbol)

    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise PriceBundleError("invalid_dates", f"{symbol} contains invalid dates", symbol=symbol)
    if not dates.is_unique:
        raise PriceBundleError("duplicate_dates", f"{symbol} contains duplicate dates", symbol=symbol)
    if not dates.is_monotonic_increasing:
        raise PriceBundleError("dates_not_sorted", f"{symbol} dates are not sorted", symbol=symbol)
    outside = dates[(dates < start_date) | (dates > end_date)]
    if not outside.empty:
        raise PriceBundleError(
            "date_out_of_range",
            f"{symbol} contains dates outside requested range",
            symbol=symbol,
            first=str(outside.min().date()),
            last=str(outside.max().date()),
        )

    numeric = frame.loc[:, REQUIRED_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PriceBundleError("non_finite_ohlcv", f"{symbol} contains non-finite OHLCV values", symbol=symbol)
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise PriceBundleError("non_positive_price", f"{symbol} contains non-positive prices", symbol=symbol)
    if (numeric["volume"] < 0).any():
        raise PriceBundleError("negative_volume", f"{symbol} contains negative volume", symbol=symbol)
    bad_ohlc = (
        (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        | (numeric["high"] < numeric["low"])
    )
    if bad_ohlc.any():
        raise PriceBundleError("bad_ohlc", f"{symbol} contains inconsistent OHLC values", symbol=symbol)

    return {
        "symbol": symbol,
        "path": f"{symbol}.parquet",
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "first_date": str(dates.iloc[0].date()),
        "last_date": str(dates.iloc[-1].date()),
    }


def _assert_symbol_set(expected: list[str], actual: dict[str, Path]) -> None:
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise PriceBundleError(
            "symbol_set_mismatch",
            "price bundle symbols do not exactly match the expected PIT symbols",
            missing_symbols=missing,
            unexpected_symbols=unexpected,
        )


def build_bundle_manifest(
    *,
    price_dir: Path,
    symbols_file: Path,
    start_date: str,
    end_date: str,
    expected_symbol_count: int,
    source: str,
    adjustment: str,
) -> dict[str, Any]:
    start = _date_bound(start_date, "start_date")
    end = _date_bound(end_date, "end_date")
    if start > end:
        raise PriceBundleError("invalid_date_range", "start_date is after end_date")
    symbols = _read_symbols(Path(symbols_file))
    if expected_symbol_count > 0 and len(symbols) != expected_symbol_count:
        raise PriceBundleError(
            "symbols_file_count_mismatch",
            "symbols file count does not match expected_symbol_count",
            expected_symbol_count=expected_symbol_count,
            actual_symbol_count=len(symbols),
        )
    files = _actual_price_files(Path(price_dir))
    _assert_symbol_set(symbols, files)
    entries = [
        _validate_frame(files[symbol], symbol, start_date=start, end_date=end)
        for symbol in symbols
    ]
    return {
        "schema_version": MANIFEST_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source),
        "adjustment": str(adjustment),
        "requested_range": {"start_date": str(start.date()), "end_date": str(end.date())},
        "expected_symbol_count": len(symbols),
        "file_count": len(entries),
        "row_count": sum(int(entry["rows"]) for entry in entries),
        "files": entries,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceBundleError("manifest_invalid", "cannot read price bundle manifest") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_VERSION:
        raise PriceBundleError("manifest_invalid", "unsupported price bundle manifest")
    return value


def verify_bundle_manifest(
    *,
    price_dir: Path,
    manifest_path: Path,
    expected_source: str | None = None,
    expected_adjustment: str | None = None,
) -> dict[str, Any]:
    manifest = _read_manifest(Path(manifest_path))
    if expected_source is not None and manifest.get("source") != expected_source:
        raise PriceBundleError("manifest_source_mismatch", "price bundle source mismatch")
    if expected_adjustment is not None and manifest.get("adjustment") != expected_adjustment:
        raise PriceBundleError("manifest_adjustment_mismatch", "price bundle adjustment mismatch")
    requested = manifest.get("requested_range") or {}
    start = _date_bound(str(requested.get("start_date", "")), "manifest start_date")
    end = _date_bound(str(requested.get("end_date", "")), "manifest end_date")
    files = _actual_price_files(Path(price_dir))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise PriceBundleError("manifest_invalid", "manifest files must be a list")
    if any(not isinstance(entry, dict) for entry in entries):
        raise PriceBundleError("manifest_invalid", "manifest contains a non-object file entry")
    entry_symbols = [str(entry.get("symbol", "")).upper() for entry in entries]
    if any(not symbol.isdigit() or len(symbol) != 6 for symbol in entry_symbols):
        raise PriceBundleError("manifest_invalid", "manifest contains an invalid symbol")
    if len(set(entry_symbols)) != len(entry_symbols):
        raise PriceBundleError("manifest_invalid", "manifest contains duplicate symbols")
    expected = set(entry_symbols)
    if int(manifest.get("expected_symbol_count", -1)) != len(expected):
        raise PriceBundleError("manifest_invalid", "manifest expected_symbol_count is inconsistent")
    _assert_symbol_set(sorted(expected), files)
    if int(manifest.get("file_count", -1)) != len(entries):
        raise PriceBundleError("manifest_invalid", "manifest file_count does not match entries")
    verified_entries: list[dict[str, Any]] = []
    for entry in entries:
        symbol = str(entry.get("symbol", "")).upper()
        path = files.get(symbol)
        if path is None:
            raise PriceBundleError("symbol_set_mismatch", f"manifest symbol is missing: {symbol}")
        if entry.get("path") != f"{symbol}.parquet":
            raise PriceBundleError("manifest_invalid", f"manifest path does not match {symbol}")
        actual = _validate_frame(path, symbol, start_date=start, end_date=end)
        for key in ("bytes", "sha256", "rows", "first_date", "last_date"):
            if actual[key] != entry.get(key):
                raise PriceBundleError(
                    "file_hash_mismatch" if key in {"bytes", "sha256"} else "file_metadata_mismatch",
                    f"manifest does not match {symbol} for {key}",
                    symbol=symbol,
                    field=key,
                )
        verified_entries.append(actual)
    if int(manifest.get("row_count", -1)) != sum(int(entry["rows"]) for entry in verified_entries):
        raise PriceBundleError("manifest_invalid", "manifest row_count does not match files")
    return {
        "manifest_sha256": file_sha256(Path(manifest_path)),
        "file_count": len(verified_entries),
        "row_count": sum(int(entry["rows"]) for entry in verified_entries),
    }


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an isolated PIT price parquet bundle.")
    parser.add_argument("--price-dir", type=Path, required=True)
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--expected-symbol-count", type=int, default=0)
    parser.add_argument("--source", default="baostock")
    parser.add_argument("--adjustment", default="qfq")
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--status-out", type=Path, required=True)
    parser.add_argument("--verify-manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_manifest is not None:
            result = verify_bundle_manifest(
                price_dir=args.price_dir,
                manifest_path=args.verify_manifest,
                expected_source=args.source,
                expected_adjustment=args.adjustment,
            )
            payload = {"status": "completed", "reason_code": None, **result}
        else:
            if args.symbols_file is None or args.start_date is None or args.end_date is None:
                raise PriceBundleError("invalid_arguments", "symbols and date range are required to build a manifest")
            manifest = build_bundle_manifest(
                price_dir=args.price_dir,
                symbols_file=args.symbols_file,
                start_date=args.start_date,
                end_date=args.end_date,
                expected_symbol_count=args.expected_symbol_count,
                source=args.source,
                adjustment=args.adjustment,
            )
            write_manifest(args.manifest_out, manifest)
            payload = {
                "status": "completed",
                "reason_code": None,
                "manifest_sha256": file_sha256(args.manifest_out),
                "file_count": manifest["file_count"],
                "row_count": manifest["row_count"],
            }
        _write_status(args.status_out, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (PriceBundleError, OSError, ValueError) as exc:
        payload = {
            "status": "failed",
            "reason_code": getattr(exc, "reason_code", "validation_error"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        }
        _write_status(args.status_out, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
