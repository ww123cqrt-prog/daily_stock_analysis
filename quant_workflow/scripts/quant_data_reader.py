#!/usr/bin/env python3
"""Read daily bars through the governed quant_data snapshot contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant_data_contract import (  # noqa: E402
    DataContractError,
    ResolvedDataset,
    ValidatedResolutionCache,
    resolve_dataset,
    resolve_dataset_component,
    sha256_file,
)


_CANONICAL_SYMBOL = re.compile(r"^(?:SH|SZ|BJ)\d{6}$")
_SUFFIX_SYMBOL = re.compile(r"^(\d{6})\.(XSHG|SH|XSHE|SZ|BJ)$")
_SUFFIX_EXCHANGE = {
    "XSHG": "SH",
    "SH": "SH",
    "XSHE": "SZ",
    "SZ": "SZ",
    "BJ": "BJ",
}
_RESOLUTION_CACHE: ValidatedResolutionCache = {}


def _component_and_index(
    dataset_id: str,
    *,
    quant_data_root: Path,
    price_basis: str | None,
    validated_cache: ValidatedResolutionCache,
) -> tuple[ResolvedDataset, Path, Path]:
    resolved = resolve_dataset(
        dataset_id,
        quant_data_root=quant_data_root,
        require_basis=price_basis,
        validated_cache=validated_cache,
    )
    manifest = resolved.manifest
    data_kind = manifest.get("data_kind")
    selected_basis = price_basis or str(manifest.get("price_basis") or "")
    if data_kind == "paired_daily_bars":
        component = resolve_dataset_component(
            dataset_id,
            quant_data_root=quant_data_root,
            price_basis=selected_basis,
            validated_cache=validated_cache,
        )
        components = manifest.get("components")
        details = (
            components.get(selected_basis)
            if isinstance(components, dict)
            else None
        )
        if not isinstance(details, dict) or not details.get("file_index_path"):
            raise DataContractError("bar component file index is unavailable")
        component_path = component.component_path
        index_relative = str(details["file_index_path"])
    elif data_kind == "daily_bars":
        component_path = resolved.snapshot_path / "bars"
        index_relative = str(manifest.get("file_index_path") or "")
    else:
        raise DataContractError("dataset does not contain daily bars")
    if not index_relative:
        raise DataContractError("bar file index is unavailable")
    index_path = (resolved.snapshot_path / index_relative).resolve()
    try:
        index_path.relative_to(resolved.snapshot_path)
        component_path.resolve().relative_to(resolved.snapshot_path)
    except ValueError as exc:
        raise DataContractError("bar dataset path escapes its snapshot") from exc
    if (
        index_path.is_symlink()
        or not index_path.is_file()
        or component_path.is_symlink()
        or not component_path.is_dir()
    ):
        raise DataContractError("bar dataset component is unavailable")
    return resolved, component_path.resolve(), index_path


def _normalize_symbol(value: object, available: Sequence[str]) -> str:
    text = str(value).strip().upper()
    normalized = text if _CANONICAL_SYMBOL.fullmatch(text) else None
    suffix = _SUFFIX_SYMBOL.fullmatch(text)
    if suffix is not None:
        normalized = f"{_SUFFIX_EXCHANGE[suffix.group(2)]}{suffix.group(1)}"
    if normalized is None and re.fullmatch(r"\d{6}", text):
        matches = [symbol for symbol in available if symbol.endswith(text)]
        if len(matches) == 1:
            normalized = matches[0]
    if normalized is None or normalized not in available:
        raise DataContractError("bar symbol is unavailable")
    return normalized


def _parse_date(value: str | None, field: str) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be an ISO date") from exc
    if pd.isna(parsed) or parsed.time() != pd.Timestamp(parsed.date()).time():
        raise DataContractError(f"{field} must be an ISO date")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _indexed_bar_path(
    *,
    snapshot: Path,
    component_path: Path,
    index: pd.DataFrame,
    symbol: str,
) -> tuple[Path, str]:
    rows = index.loc[index["symbol"].astype(str) == symbol]
    if len(rows) != 1:
        raise DataContractError("bar symbol is unavailable")
    row = rows.iloc[0]
    relative = Path(str(row["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise DataContractError("bar file index path is invalid")
    path = (snapshot / relative).resolve()
    try:
        path.relative_to(component_path)
    except ValueError as exc:
        raise DataContractError("bar file index path is invalid") from exc
    if path.is_symlink() or not path.is_file():
        raise DataContractError("indexed bar file is unavailable")
    expected_hash = str(row.get("sha256") or "")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise DataContractError("indexed bar file hash mismatch")
    return path, expected_hash


def build_update_universe(
    dataset_id: str,
    *,
    quant_data_root: Path,
    price_basis: str | None = None,
    validated_cache: ValidatedResolutionCache | None = None,
) -> dict[str, object]:
    """Build a downloader universe bound to one resolved formal snapshot."""

    cache = _RESOLUTION_CACHE if validated_cache is None else validated_cache
    resolved, _component_path, index_path = _component_and_index(
        dataset_id,
        quant_data_root=Path(quant_data_root),
        price_basis=price_basis,
        validated_cache=cache,
    )
    try:
        index = pd.read_parquet(index_path, columns=["symbol"])
    except Exception as exc:
        raise DataContractError("bar file index is unavailable") from exc
    symbols = index["symbol"].astype(str)
    if (
        symbols.empty
        or symbols.duplicated().any()
        or not symbols.map(lambda value: bool(_CANONICAL_SYMBOL.fullmatch(value))).all()
    ):
        raise DataContractError("bar file index symbols are invalid")
    ordered = sorted(symbols.tolist())
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "snapshot_id": str(resolved.manifest["snapshot_id"]),
        "manifest_sha256": sha256_file(
            resolved.snapshot_path / "DATASET_MANIFEST.json"
        ),
        "latest_trade_date": str(resolved.manifest["latest_trade_date"]),
        "symbol_count": len(ordered),
        "all": ordered,
    }


def read_bars(
    dataset_id: str,
    symbol: object,
    *,
    quant_data_root: Path,
    price_basis: str | None = None,
    fields: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    validated_cache: ValidatedResolutionCache | None = None,
) -> pd.DataFrame:
    """Return one symbol's governed daily bars for an inclusive date range."""

    cache = _RESOLUTION_CACHE if validated_cache is None else validated_cache
    resolved, component_path, index_path = _component_and_index(
        dataset_id,
        quant_data_root=Path(quant_data_root),
        price_basis=price_basis,
        validated_cache=cache,
    )
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        raise DataContractError("bar file index is unavailable") from exc
    required_index = {"symbol", "relative_path", "sha256"}
    if index.empty or not required_index.issubset(index.columns):
        raise DataContractError("bar file index is invalid")
    available = sorted(index["symbol"].astype(str).unique())
    normalized_symbol = _normalize_symbol(symbol, available)
    path, _ = _indexed_bar_path(
        snapshot=resolved.snapshot_path,
        component_path=component_path,
        index=index,
        symbol=normalized_symbol,
    )
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DataContractError("indexed bar file is unreadable") from exc
    if "date" not in frame.columns:
        raise DataContractError("indexed bar file has no date column")
    selected_fields = list(fields) if fields is not None else [
        str(column) for column in frame.columns if column != "date"
    ]
    if (
        any(not isinstance(field, str) or not field for field in selected_fields)
        or len(selected_fields) != len(set(selected_fields))
    ):
        raise DataContractError("bar fields are invalid")
    unavailable = [field for field in selected_fields if field not in frame.columns]
    if unavailable:
        raise DataContractError("bar field is unavailable: " + ", ".join(unavailable))
    start_date = _parse_date(start, "start")
    end_date = _parse_date(end, "end")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise DataContractError("bar date range is invalid")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise DataContractError("indexed bar dates are invalid")
    mask = pd.Series(True, index=frame.index)
    if start_date is not None:
        mask &= dates >= start_date
    if end_date is not None:
        mask &= dates <= end_date
    result = frame.loc[mask, ["date", *selected_fields]].copy()
    result["date"] = dates.loc[mask].to_numpy()
    return result.reset_index(drop=True)
