"""Strict read-only adapter for an unaccepted native V2 ETF snapshot.

The generic quant-data validator remains the authority for formally accepted
datasets.  This module only permits bounded research against the locked
three-ETF universe and always labels the result ``research_preview``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant_data_contract import (  # noqa: E402
    DataContractError,
    sha256_file,
    validate_latest_pointer,
)


DATASET_ID = "cn_etf_daily"
LOCKED_SYMBOLS = ("SZ159915", "SH513100", "SH512890")
EXPECTED_DATA_KIND = "paired_daily_bars"
EXPECTED_QFQ_INDEX = "QFQ_FILE_INDEX.parquet"
SYMBOL_RE = re.compile(r"^(?:SH|SZ)\d{6}$")
REQUIRED_BAR_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
INDEX_COLUMNS = {
    "symbol",
    "relative_path",
    "row_count",
    "first_date",
    "last_date",
    "source_ids",
    "sha256",
}


class EtfResearchPreview(NamedTuple):
    status: str
    dataset_id: str
    snapshot_id: str
    snapshot_path: Path
    manifest_sha256: str
    content_sha256: str
    date_range: dict[str, str]
    bars: dict[str, pd.DataFrame]
    formally_accepted: bool
    usable_for_trading: bool


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _contained_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DataContractError(f"{label} path is invalid")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or "." in raw.parts:
        raise DataContractError(f"{label} path is invalid")
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise DataContractError(f"{label} path contains a symlink")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise DataContractError(f"{label} path is unavailable or escapes its root") from exc
    return resolved


def _reject_snapshot_symlinks(snapshot: Path) -> None:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DataContractError("ETF snapshot directory is unavailable")
    for path in snapshot.rglob("*"):
        if path.is_symlink():
            raise DataContractError(
                f"ETF snapshot contains a symlink: {path.relative_to(snapshot)}"
            )


def _content_hash(snapshot: Path, manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "DATASET_MANIFEST.json"
    )
    for path in files:
        relative = path.relative_to(snapshot).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    stable = dict(manifest)
    stable.pop("content_sha256", None)
    digest.update(_json_bytes(stable))
    return digest.hexdigest()


def _require_identity(
    entry: Mapping[str, Any],
    pointer: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected = {
        "dataset_id": DATASET_ID,
        "data_kind": EXPECTED_DATA_KIND,
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
    }
    for field, value in expected.items():
        if entry.get(field) != value or manifest.get(field) != value:
            raise DataContractError(f"ETF preview identity mismatch: {field}")
    if pointer.get("dataset_id") != DATASET_ID:
        raise DataContractError("latest pointer belongs to another dataset")
    if entry.get("snapshot_id") != pointer.get("snapshot_id"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    if manifest.get("snapshot_id") != pointer.get("snapshot_id"):
        raise DataContractError("latest pointer snapshot_id mismatch")
    if entry.get("snapshot_manifest_sha256") != pointer.get("manifest_sha256"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    if entry.get("content_sha256") != manifest.get("content_sha256"):
        raise DataContractError("dataset catalog and snapshot disagree on content hash")


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> tuple[str, str]:
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise DataContractError("paired ETF components are missing")
    qfq = components.get("qfq")
    if not isinstance(qfq, Mapping):
        raise DataContractError("paired ETF qfq component is missing")
    qfq_path = qfq.get("path")
    index_path = qfq.get("file_index_path")
    if (
        qfq_path != "qfq"
        or index_path != EXPECTED_QFQ_INDEX
        or manifest.get("file_index_path") != EXPECTED_QFQ_INDEX
        or qfq.get("price_basis") != "qfq"
    ):
        raise DataContractError("paired ETF qfq component contract is invalid")
    adjustment = manifest.get("adjustment")
    if (
        not isinstance(adjustment, Mapping)
        or adjustment.get("method") != "provider_native_paired"
        or adjustment.get("paired_raw_component_path") != "raw"
    ):
        raise DataContractError("paired ETF adjustment contract is invalid")
    date_range = manifest.get("date_range")
    if (
        not isinstance(date_range, Mapping)
        or not str(date_range.get("start") or "")
        or not str(date_range.get("end") or "")
    ):
        raise DataContractError("paired ETF date range is invalid")
    return str(qfq_path), str(index_path)


def _validate_index(index: pd.DataFrame, manifest: Mapping[str, Any]) -> None:
    if index.empty or not INDEX_COLUMNS.issubset(index.columns):
        raise DataContractError("QFQ ETF file index is invalid")
    if index["symbol"].astype(str).duplicated().any():
        raise DataContractError("QFQ ETF file index contains duplicate symbols")
    if index["relative_path"].astype(str).duplicated().any():
        raise DataContractError("QFQ ETF file index contains duplicate paths")
    expected_count = manifest.get("components", {}).get("qfq", {}).get("symbol_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(index)
        or manifest.get("symbol_count") != len(index)
    ):
        raise DataContractError("QFQ ETF file index symbol count is stale")
    for row in index.itertuples(index=False):
        symbol = str(row.symbol)
        if not SYMBOL_RE.fullmatch(symbol):
            raise DataContractError("QFQ ETF file index contains an invalid symbol")
        if str(row.relative_path) != f"qfq/{symbol}.parquet":
            raise DataContractError(f"QFQ ETF path is invalid: {symbol}")


def _validate_required_bar(
    path: Path,
    row: object,
    *,
    symbol: str,
) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"required ETF file is unavailable: {symbol}")
    expected_hash = str(getattr(row, "sha256"))
    if sha256_file(path) != expected_hash:
        raise DataContractError(f"ETF file hash mismatch: {symbol}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DataContractError(f"required ETF parquet is unreadable: {symbol}") from exc
    missing = REQUIRED_BAR_COLUMNS - set(frame.columns)
    if missing:
        raise DataContractError(f"required ETF columns are missing: {symbol}")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    positive_prices = (numeric[["open", "high", "low", "close"]] > 0).all().all()
    finite_values = np.isfinite(numeric.to_numpy(dtype=float)).all()
    valid_ohlc = (
        (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)).all()
        and (numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)).all()
    )
    if (
        frame.empty
        or dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or not positive_prices
        or not finite_values
        or not valid_ohlc
    ):
        raise DataContractError(f"required ETF bars are invalid: {symbol}")
    if "symbol" in frame and set(frame["symbol"].astype(str)) != {symbol}:
        raise DataContractError(f"required ETF symbol column is invalid: {symbol}")
    if "adjustment" in frame and set(frame["adjustment"].astype(str)) != {"qfq"}:
        raise DataContractError(f"required ETF adjustment column is invalid: {symbol}")
    if (
        int(getattr(row, "row_count")) != len(frame)
        or str(getattr(row, "first_date")) != dates.iloc[0].date().isoformat()
        or str(getattr(row, "last_date")) != dates.iloc[-1].date().isoformat()
    ):
        raise DataContractError(f"required ETF index metadata is stale: {symbol}")
    return frame.copy()


def resolve_etf_research_preview(
    quant_data_root: Path,
    *,
    required_symbols: Sequence[str] = LOCKED_SYMBOLS,
) -> EtfResearchPreview:
    """Resolve the locked ETF inputs without claiming formal data acceptance."""

    if tuple(required_symbols) != LOCKED_SYMBOLS:
        raise ValueError("required_symbols must match the locked three-ETF contract")
    raw_root = Path(quant_data_root).expanduser().absolute()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise DataContractError("quant_data root is unavailable")
    root = raw_root.resolve(strict=True)
    catalog = _read_json(root / "DATA_MANIFEST.json", label="quant_data catalog")
    if (
        catalog.get("schema_version") != 2
        or catalog.get("path_format") != "relative_to_quant_data_root"
        or not isinstance(catalog.get("datasets"), Mapping)
    ):
        raise DataContractError("quant_data catalog contract is invalid")
    entry = catalog["datasets"].get(DATASET_ID)
    if not isinstance(entry, Mapping):
        raise DataContractError(f"dataset is not registered: {DATASET_ID}")
    if entry.get("dataset_root") != "parquet/etf_1d/cn_etf_daily":
        raise DataContractError("ETF dataset root identity is invalid")
    pointer_path = _contained_path(
        root,
        entry.get("latest_pointer"),
        label="latest pointer",
    )
    pointer, snapshot = validate_latest_pointer(
        pointer_path,
        quant_data_root=root,
        expected_dataset_id=DATASET_ID,
    )
    _reject_snapshot_symlinks(snapshot)
    manifest = _read_json(
        snapshot / "DATASET_MANIFEST.json", label="ETF snapshot manifest"
    )
    _require_identity(entry, pointer, manifest)
    qfq_path, index_path = _validate_manifest_contract(manifest)
    actual_content_hash = _content_hash(snapshot, manifest)
    if actual_content_hash != manifest.get("content_sha256"):
        raise DataContractError("ETF snapshot content hash mismatch")
    qfq_dir = _contained_path(snapshot, qfq_path, label="qfq component")
    if not qfq_dir.is_dir():
        raise DataContractError("qfq component is unavailable")
    index_file = _contained_path(snapshot, index_path, label="QFQ file index")
    try:
        index = pd.read_parquet(index_file)
    except Exception as exc:
        raise DataContractError("QFQ ETF file index is unavailable") from exc
    _validate_index(index, manifest)
    indexed = index.assign(symbol=index["symbol"].astype(str)).set_index("symbol")
    missing = [symbol for symbol in LOCKED_SYMBOLS if symbol not in indexed.index]
    if missing:
        raise DataContractError(
            f"QFQ file index is missing required ETF: {', '.join(missing)}"
        )
    bars: dict[str, pd.DataFrame] = {}
    for symbol in LOCKED_SYMBOLS:
        row = next(indexed.loc[[symbol]].itertuples(index=False))
        path = _contained_path(
            snapshot,
            getattr(row, "relative_path"),
            label=f"required ETF {symbol}",
        )
        bars[symbol] = _validate_required_bar(path, row, symbol=symbol)
    date_range = manifest["date_range"]
    return EtfResearchPreview(
        status="research_preview",
        dataset_id=DATASET_ID,
        snapshot_id=str(manifest["snapshot_id"]),
        snapshot_path=snapshot,
        manifest_sha256=str(pointer["manifest_sha256"]),
        content_sha256=str(manifest["content_sha256"]),
        date_range={
            "start": str(date_range["start"]),
            "end": str(date_range["end"]),
        },
        bars=bars,
        formally_accepted=False,
        usable_for_trading=False,
    )
