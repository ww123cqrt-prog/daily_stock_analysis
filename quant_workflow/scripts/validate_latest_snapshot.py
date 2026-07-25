#!/usr/bin/env python3
"""Validate latest merged daily bars against Sina's anonymous quote snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import write_json_atomic  # noqa: E402
from _paths import quant_data_root, reports_root  # noqa: E402
from market_data_clients import (  # noqa: E402
    fetch_sina_a_snapshot_text,
    fetch_sina_a_snapshots,
    parse_sina_quote_text,
    sina_quote_code,
)


QUANT_DATA = quant_data_root()
MERGED_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
REPORT_DIR = reports_root()
RAW_DIR = QUANT_DATA / "raw" / "ashare_snapshots"


def sina_code(symbol: str) -> str:
    return sina_quote_code(symbol)


def fetch_sina_snapshots(
    symbols: list[str],
    *,
    session=None,
    timeout: float = 10,
    use_system_proxy: bool = False,
) -> list[dict[str, object]]:
    return fetch_sina_a_snapshots(
        symbols,
        session=session,
        timeout=timeout,
        use_system_proxy=use_system_proxy,
    )


def fetch_sina_snapshot_capture(
    symbols: list[str],
    *,
    session=None,
    timeout: float = 10,
    use_system_proxy: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    raw_text = fetch_sina_a_snapshot_text(
        symbols,
        session=session,
        timeout=timeout,
        use_system_proxy=use_system_proxy,
    )
    return raw_text, parse_sina_quote_text(raw_text)


def write_raw_snapshot_capture(
    *,
    raw_text: str,
    symbols: list[str],
    raw_root: Path,
    captured_at: str | None = None,
) -> Path:
    captured = datetime.fromisoformat(captured_at) if captured_at else datetime.now().astimezone()
    if captured.tzinfo is None:
        captured = captured.astimezone()
    path = (
        raw_root
        / f"captured_date={captured.date().isoformat()}"
        / f"sina_snapshot_{captured.strftime('%Y%m%dT%H%M%S_%f%z')}.json"
    )
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "source": "sina_snapshot",
            "captured_at": captured.isoformat(timespec="microseconds"),
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "symbols": symbols,
            "raw_text": raw_text,
        },
    )
    return path


def _latest_merged_close(path: Path) -> tuple[str, float] | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["date", "close"])
    if frame.empty:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    if frame.empty:
        return None
    row = frame.iloc[-1]
    return str(row["date"].date()), float(row["close"])


def build_validation_report(
    *,
    snapshots: list[dict[str, Any]],
    merged_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in snapshots:
        latest = _latest_merged_close(merged_dir / f"{item['symbol']}.parquet")
        if latest is None:
            continue
        merged_date, merged_close = latest
        raw_close = item.get("close")
        if raw_close in (None, ""):
            continue
        try:
            snapshot_close = float(raw_close)
        except (TypeError, ValueError):
            continue
        rel_diff = abs(snapshot_close - merged_close) / abs(merged_close) if merged_close else None
        rows.append(
            {
                "symbol": item["symbol"],
                "name": item.get("name", ""),
                "snapshot_date": item.get("trade_date", ""),
                "snapshot_close": snapshot_close,
                "merged_date": merged_date,
                "merged_close": merged_close,
                "close_rel_diff": rel_diff,
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "sina_snapshot",
        "merged_dir": str(merged_dir),
        "snapshots": len(snapshots),
        "compared": len(rows),
        "rows": rows,
    }
    dated = out_dir / f"latest_snapshot_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest = out_dir / "latest_snapshot_validation_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    dated.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")

    md_rows = "\n".join(
        f"| {row['symbol']} | {row['name']} | {row['snapshot_date']} | "
        f"{row['snapshot_close']:.4f} | {row['merged_date']} | {row['merged_close']:.4f} | "
        f"{row['close_rel_diff']:.6f} |"
        if row['close_rel_diff'] is not None else
        f"| {row['symbol']} | {row['name']} | {row['snapshot_date']} | "
        f"{row['snapshot_close']:.4f} | {row['merged_date']} | {row['merged_close']:.4f} | n/a |"
        for row in rows
    )
    report = f"""# Latest Snapshot Validation

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

- Source: Sina quote snapshot
- Merged parquet: `{merged_dir}`
- Snapshots: {len(snapshots)}
- Compared: {len(rows)}

| symbol | name | snapshot date | snapshot close | merged date | merged close | relative diff |
| --- | --- | --- | --- | --- | --- | --- |
{md_rows}
"""
    (out_dir / "latest_snapshot_validation_latest.md").write_text(report, encoding="utf-8")
    (out_dir / f"latest_snapshot_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md").write_text(
        report,
        encoding="utf-8",
    )
    return payload


def load_symbols_from_merged(merged_dir: Path, limit: int) -> list[str]:
    symbols = [path.stem for path in sorted(merged_dir.glob("*.parquet"))]
    return symbols[:limit] if limit > 0 else symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate merged latest close against Sina snapshot.")
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--use-system-proxy", action="store_true", help="Opt into macOS/system proxy settings.")
    args = parser.parse_args()

    symbols = load_symbols_from_merged(args.merged_dir, args.limit)
    raw_text, snapshots = fetch_sina_snapshot_capture(
        symbols,
        timeout=args.timeout,
        use_system_proxy=args.use_system_proxy,
    )
    write_raw_snapshot_capture(raw_text=raw_text, symbols=symbols, raw_root=args.raw_dir)
    payload = build_validation_report(snapshots=snapshots, merged_dir=args.merged_dir, out_dir=args.out_dir)
    print(f"Snapshot validation compared {payload['compared']} of {payload['snapshots']} snapshots -> {args.out_dir}")
    return 0 if payload["compared"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
