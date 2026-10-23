#!/usr/bin/env python3
"""Freeze anomaly-derived cleaning rules for the direct Qlib Provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


VOLUME_ACTIONS = {"keep_but_mask_or_impute_volume"}
RETURN_ACTIONS = {
    "review_and_winsorize_or_mask_event_dates",
    "winsorize_returns_and_mark_event_dates",
}
RETURN_EVENT_TYPES = {"abs_return_gt_50pct", "abs_return_gt_30pct"}
VOLUME_EVENT_TYPES = {"non_positive_volume"}
EVENT_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "pctChg",
]


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _sample_dates(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    parsed = pd.to_datetime(
        [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()],
        errors="coerce",
    )
    if parsed.isna().any():
        return set()
    return {item.normalize().date().isoformat() for item in parsed}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cleaning_plan(
    *,
    stocks_csv: Path,
    events_csv: Path,
    output_path: Path,
    source_snapshot: Mapping[str, Any],
) -> Path:
    stocks = pd.read_csv(stocks_csv, dtype={"symbol": str})
    events = pd.read_csv(events_csv, dtype={"symbol": str})
    excluded: set[str] = set()
    volume_symbols: set[str] = set()
    return_symbols: set[str] = set()
    event_dates: dict[str, set[str]] = {}

    for _, row in stocks.iterrows():
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        action = str(row.get("cleaning_action") or "").strip()
        severity = str(row.get("severity") or "").strip().lower()
        if action == "exclude_symbol_from_training_until_repaired" or severity == "blocker":
            excluded.add(symbol)
        if action in VOLUME_ACTIONS:
            volume_symbols.add(symbol)
        if action in RETURN_ACTIONS:
            return_symbols.add(symbol)

    for _, row in events.iterrows():
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        issue_type = str(row.get("issue_type") or "").strip()
        if issue_type in VOLUME_EVENT_TYPES:
            volume_symbols.add(symbol)
        if issue_type in RETURN_EVENT_TYPES:
            return_symbols.add(symbol)
            event_dates.setdefault(symbol, set()).update(
                _sample_dates(row.get("sample_dates"))
            )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "stocks_csv_sha256": _sha256(stocks_csv),
            "events_csv_sha256": _sha256(events_csv),
            "snapshot": dict(source_snapshot),
        },
        "excluded_symbols": sorted(excluded),
        "volume_mask_symbols": sorted(volume_symbols),
        "return_mask_symbols": sorted(return_symbols),
        "event_dates": {
            symbol: sorted(values) for symbol, values in sorted(event_dates.items())
        },
        "event_mask_fields": EVENT_FIELDS,
    }
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("r+b" if os.name == "nt" else "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stocks-csv", type=Path, required=True)
    parser.add_argument("--events-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-snapshot", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_snapshot = json.loads(args.source_snapshot)
    path = build_cleaning_plan(
        stocks_csv=args.stocks_csv,
        events_csv=args.events_csv,
        output_path=args.output,
        source_snapshot=source_snapshot,
    )
    print(json.dumps({"status": "ready", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
