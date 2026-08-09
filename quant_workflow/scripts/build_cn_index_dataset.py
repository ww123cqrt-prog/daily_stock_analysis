#!/usr/bin/env python3
"""Build the formal five-index Parquet dataset from existing Qlib evidence.

This is a bounded migration utility.  It reads only the five benchmark series
from one immutable native Qlib snapshot; it never downloads stock data and it
never creates another Qlib ``features/*.bin`` market-data copy.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_price_snapshot,
    publish_snapshot_group,
    sha256_file,
)


REQUIRED_SYMBOLS = (
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
)
FIELDS = ("open", "high", "low", "close", "volume")
DATASET_ID = "cn_index_daily"
DATASET_ROOT = Path("parquet/index_1d/cn_index_daily")


class IndexDatasetError(DataContractError):
    """Raised when the existing index evidence cannot form a valid dataset."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexDatasetError(f"invalid native provider manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise IndexDatasetError("native provider manifest is not an object")
    return payload


def _read_calendar(native_snapshot: Path) -> pd.DatetimeIndex:
    path = native_snapshot / "calendars" / "day.txt"
    try:
        values = [
            value.strip()
            for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    except OSError as exc:
        raise IndexDatasetError("native Qlib calendar is unavailable") from exc
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).normalize()
    if (
        parsed.empty
        or parsed.hasnans
        or parsed.has_duplicates
        or not parsed.is_monotonic_increasing
    ):
        raise IndexDatasetError("native Qlib calendar is invalid")
    return parsed


def _read_feature(
    native_snapshot: Path,
    symbol: str,
    field: str,
    calendar: pd.DatetimeIndex,
) -> np.ndarray:
    path = native_snapshot / "features" / symbol.lower() / f"{field}.day.bin"
    try:
        raw = np.fromfile(path, dtype="<f4")
    except OSError as exc:
        raise IndexDatasetError(f"missing native index feature: {symbol} {field}") from exc
    if len(raw) < 2 or not np.isfinite(raw[0]) or raw[0] != int(raw[0]):
        raise IndexDatasetError(f"invalid native index feature header: {symbol} {field}")
    start = int(raw[0])
    values = raw[1:]
    if start < 0 or start + len(values) > len(calendar):
        raise IndexDatasetError(f"native index feature range is invalid: {symbol} {field}")
    aligned = np.full(len(calendar), np.nan, dtype=float)
    aligned[start : start + len(values)] = values
    return aligned


def _source_metadata(native_snapshot: Path, provider_manifest: Mapping[str, Any]) -> dict[str, Any]:
    benchmarks = provider_manifest.get("benchmarks")
    source = benchmarks.get("source") if isinstance(benchmarks, Mapping) else None
    source = dict(source) if isinstance(source, Mapping) else {}
    source.setdefault("provider", "native_qlib_snapshot")
    source["interface"] = "native_qlib_bin_migration"
    source["native_snapshot_name"] = native_snapshot.name
    manifest_path = native_snapshot / "PROVIDER_MANIFEST.json"
    if manifest_path.is_file():
        source["native_provider_manifest_sha256"] = sha256_file(manifest_path)
    return source


def _native_snapshot_id(native_snapshot: Path) -> str:
    manifest_path = native_snapshot / "DATASET_MANIFEST.json"
    if manifest_path.is_file():
        payload = _read_json(manifest_path)
        value = str(payload.get("snapshot_id") or "").strip()
        if value:
            return value
    return native_snapshot.name


def _build_frame(
    native_snapshot: Path,
    symbol: str,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    values = {
        field: _read_feature(native_snapshot, symbol, field, calendar)
        for field in FIELDS
    }
    matrix = np.column_stack([values[field] for field in FIELDS])
    valid = np.isfinite(matrix).all(axis=1)
    positions = np.flatnonzero(valid)
    if not len(positions):
        raise IndexDatasetError(f"native index has no complete rows: {symbol}")
    first, last = int(positions[0]), int(positions[-1])
    if not valid[first : last + 1].all():
        raise IndexDatasetError(f"native index has a gap inside coverage: {symbol}")
    frame = pd.DataFrame(
        {field: values[field][first : last + 1] for field in FIELDS},
        index=calendar[first : last + 1],
    )
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["source_id"] = f"native_qlib:{_native_snapshot_id(native_snapshot)}"
    return frame


def stage_index_dataset_from_native(
    *,
    quant_data_root: Path,
    native_snapshot: Path,
    run_id: str | None = None,
) -> Path:
    root = Path(quant_data_root).expanduser().resolve()
    native = Path(native_snapshot).expanduser().resolve()
    if not native.is_dir() or native.is_symlink():
        raise IndexDatasetError("native Qlib snapshot is unavailable")
    provider_manifest = _read_json(native / "PROVIDER_MANIFEST.json")
    benchmarks = provider_manifest.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or benchmarks.get("required_symbols") != list(
        REQUIRED_SYMBOLS
    ):
        raise IndexDatasetError("native Qlib benchmark evidence is incomplete")

    run_id = run_id or (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    snapshot_id = f"cn-index-daily-{run_id}"
    staging = root / "staging" / f"cn-index-dataset-{run_id}" / snapshot_id
    bars = staging / "bars"
    bars.mkdir(parents=True, exist_ok=False)
    calendar = _read_calendar(native)
    for symbol in REQUIRED_SYMBOLS:
        _build_frame(native, symbol, calendar).to_parquet(
            bars / f"{symbol}.parquet",
            index=False,
        )

    source = _source_metadata(native, provider_manifest)
    source["native_snapshot_path"] = native.name
    retrieved_at = str(source.get("retrieved_at") or "")
    if not retrieved_at:
        retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    finalize_price_snapshot(
        staging,
        dataset_id=DATASET_ID,
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source=source,
        price_basis="derived",
        adjustment={"method": "none"},
        producer={
            "script": "build_cn_index_dataset.py",
            "version": "1",
            "source_snapshot_id": _native_snapshot_id(native),
        },
        source_retrieved_at=retrieved_at,
    )
    return staging


def build_cn_index_dataset(
    *,
    quant_data_root: Path,
    native_snapshot: Path,
    publish: bool = False,
    run_id: str | None = None,
) -> Path:
    root = Path(quant_data_root).expanduser().resolve()
    staged = stage_index_dataset_from_native(
        quant_data_root=root,
        native_snapshot=native_snapshot,
        run_id=run_id,
    )
    if not publish:
        return staged
    return publish_snapshot_group(
        [(staged, root / DATASET_ROOT)],
        quant_data_root=root,
        update_root_catalog=True,
    )[0]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-root", type=Path, required=True)
    parser.add_argument("--native-snapshot", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        path = build_cn_index_dataset(
            quant_data_root=args.quant_data_root,
            native_snapshot=args.native_snapshot,
            publish=args.publish,
            run_id=args.run_id,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "published" if args.publish else "staged", "path": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
