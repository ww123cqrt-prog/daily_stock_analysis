#!/usr/bin/env python3
"""Generate csi300.txt for Qlib."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (  # noqa: E402
    _env_path,
    china_astock_root,
    qlib_compatibility_provider_uri,
    quant_data_root,
    workflow_root,
)
from _symbols import normalize_symbol as _normalize_symbol, to_qlib_name as _to_qlib_name  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
ASTOCK_PROJECT = china_astock_root()
PROVIDER_DIR = qlib_compatibility_provider_uri()
OUTPUT_PATH = PROVIDER_DIR / "instruments" / "csi300.txt"
CALENDAR_PATH = PROVIDER_DIR / "calendars" / "day.txt"
STOCK_LIST_PATH = _env_path("QUANT_STOCK_LIST_PATH", QUANT_DATA / "stock_lists.json")


def read_calendar_range(path: Path) -> tuple[str, str]:
    days = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not days:
        raise ValueError(f"empty Qlib calendar: {path}")
    return days[0], days[-1]


def normalize_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol, strip_any_suffix=True, strip_prefix=True)


def _load_csi300_from_project(project_root: Path = ASTOCK_PROJECT) -> list[str]:
    root = project_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Missing project for CSI300 universe: {root}")
    sys.path.insert(0, str(root))
    try:
        from data_fetcher.universe import csi300

        return [normalize_symbol(symbol) for symbol in csi300()]
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def load_csi300_symbols(
    stock_list_path: Path = STOCK_LIST_PATH,
    astock_project: Path = ASTOCK_PROJECT,
    *,
    allow_network_fallback: bool = True,
) -> list[str]:
    if stock_list_path.exists():
        data = json.loads(stock_list_path.read_text(encoding="utf-8"))
        cached = data.get("csi300", [])
        if cached:
            return [normalize_symbol(symbol) for symbol in cached]

    if not allow_network_fallback:
        raise ValueError(
            f"Missing cached CSI300 symbols in {stock_list_path}; dry-run will not use china-astock/AkShare fallback."
        )

    return _load_csi300_from_project(astock_project)


def to_qlib_name(sym: str) -> str:
    return _to_qlib_name(sym, default_exchange="SZ", strip_any_suffix=True, strip_prefix=True)


def build_instrument_lines(symbols: list[str], start_date: str, end_date: str) -> list[str]:
    qlib_symbols = [to_qlib_name(sym) for sym in symbols]
    return [f"{sym}\t{start_date}\t{end_date}" for sym in qlib_symbols]


def write_csi300(
    output_path: Path = OUTPUT_PATH,
    calendar_path: Path = CALENDAR_PATH,
    stock_list_path: Path = STOCK_LIST_PATH,
    astock_project: Path = ASTOCK_PROJECT,
    *,
    dry_run: bool = False,
) -> dict[str, str | int]:
    try:
        symbols = load_csi300_symbols(stock_list_path, astock_project, allow_network_fallback=not dry_run)
    except ValueError as exc:
        if not dry_run:
            raise
        print(f"DRY RUN: {exc}")
        symbols = []
    print(f"Found {len(symbols)} CSI300 constituents")
    try:
        start_date, end_date = read_calendar_range(calendar_path)
    except (FileNotFoundError, ValueError) as exc:
        if not dry_run:
            raise
        print(f"DRY RUN: Qlib calendar unavailable: {exc}")
        start_date, end_date = "", ""

    lines = build_instrument_lines(symbols, start_date, end_date)
    qlib_symbols = [line.split("\t", 1)[0] for line in lines]

    if dry_run:
        print(f"Dry run OK. Would write {len(lines)} lines to {output_path}")
        print(f"First 5 symbols: {qlib_symbols[:5]}")
        return {
            "status": "dry-run",
            "output_path": str(output_path),
            "symbols": len(symbols),
            "start_date": start_date,
            "end_date": end_date,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written to {output_path}")
    print(f"First 5 symbols: {qlib_symbols[:5]}")
    return {
        "status": "completed",
        "output_path": str(output_path),
        "symbols": len(symbols),
        "start_date": start_date,
        "end_date": end_date,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Qlib csi300.txt instrument file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--calendar", type=Path, default=CALENDAR_PATH)
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--astock-root", type=Path, default=ASTOCK_PROJECT)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned output without writing csi300.txt.")
    args = parser.parse_args(argv)
    write_csi300(
        args.output,
        args.calendar,
        args.stock_list,
        args.astock_root,
        dry_run=args.dry_run,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
