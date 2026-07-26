# -*- coding: utf-8 -*-
"""Read verified daily bars from canonical quant_data snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.core.strategy_backtest import PriceBar


def normalize_market_symbol(value: str) -> str:
    symbol = str(value or "").strip().upper().replace(".", "")
    if symbol.startswith(("SH", "SZ", "BJ")) and len(symbol) == 8:
        return symbol
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"invalid A-share or ETF symbol: {value}")
    if symbol.startswith(("5", "6", "9")):
        return f"SH{symbol}"
    if symbol.startswith(("0", "1", "2", "3")):
        return f"SZ{symbol}"
    if symbol.startswith(("4", "8")):
        return f"BJ{symbol}"
    raise ValueError(f"cannot infer exchange for symbol: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"governed data metadata is unavailable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"governed data metadata must be an object: {path.name}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("governed data metadata contains an unsafe path") from exc
    return candidate


@dataclass(frozen=True)
class QuantPriceLoad:
    symbols: tuple[str, ...]
    bars: tuple[PriceBar, ...]
    dataset_id: str
    snapshot_id: str
    price_basis: str
    provider: str
    latest_trade_date: str
    manifest_sha256: str


class QuantPriceRepository:
    """Resolve one ready canonical qfq snapshot containing all requested symbols."""

    def __init__(self, quant_data_root: Path | None = None):
        if quant_data_root is None:
            configured = os.getenv("QUANT_DATA_ROOT")
            quant_data_root = (
                Path(configured).expanduser()
                if configured
                else Path(__file__).resolve().parents[2].parent / "quant_data"
            )
        self.root = Path(quant_data_root).expanduser().resolve()

    def load(
        self,
        symbols: Sequence[str],
        *,
        end_date: date | None = None,
    ) -> QuantPriceLoad:
        normalized = tuple(normalize_market_symbol(symbol) for symbol in symbols)
        if not normalized:
            raise ValueError("at least one symbol is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must not contain duplicates")

        catalog = _read_json(self.root / "DATA_MANIFEST.json")
        if catalog.get("schema_version") != 2 or not isinstance(catalog.get("datasets"), dict):
            raise ValueError("quant_data DATA_MANIFEST schema version 2 is required")

        candidates = []
        for dataset_id, entry in catalog["datasets"].items():
            if not isinstance(entry, dict):
                continue
            if (
                entry.get("data_kind") == "daily_bars"
                and entry.get("role") == "canonical"
                and entry.get("status") == "ready"
                and entry.get("price_basis") == "qfq"
                and entry.get("latest_pointer")
            ):
                candidates.append((int(entry.get("symbol_count") or 10**9), str(dataset_id), entry))
        candidates.sort(key=lambda item: (item[0], item[1]))

        for _, dataset_id, entry in candidates:
            resolved = self._try_dataset(dataset_id, entry, normalized, end_date=end_date)
            if resolved is not None:
                return resolved
        raise ValueError(
            "No ready canonical qfq dataset contains all requested symbols: "
            + ", ".join(normalized)
        )

    def _try_dataset(
        self,
        dataset_id: str,
        entry: dict[str, Any],
        symbols: tuple[str, ...],
        *,
        end_date: date | None,
    ) -> QuantPriceLoad | None:
        latest_path = _contained_path(self.root, str(entry["latest_pointer"]))
        if not latest_path.is_file():
            return None
        latest = _read_json(latest_path)
        if latest.get("dataset_id") != dataset_id or not latest.get("snapshot_path"):
            raise ValueError(f"dataset pointer does not match catalog: {dataset_id}")
        snapshot = _contained_path(self.root, str(latest["snapshot_path"]))
        manifest_path = _contained_path(snapshot, "DATASET_MANIFEST.json")
        if not manifest_path.is_file():
            raise ValueError(f"dataset snapshot manifest is missing: {dataset_id}")
        manifest_hash = _sha256(manifest_path)
        expected_hashes = {
            str(value)
            for value in (latest.get("manifest_sha256"), entry.get("snapshot_manifest_sha256"))
            if value
        }
        if expected_hashes and manifest_hash not in expected_hashes:
            raise ValueError(f"dataset snapshot manifest hash mismatch: {dataset_id}")
        manifest = _read_json(manifest_path)
        if (
            manifest.get("dataset_id") != dataset_id
            or manifest.get("snapshot_id") != latest.get("snapshot_id")
            or manifest.get("status") != "ready"
            or manifest.get("role") != "canonical"
            or manifest.get("price_basis") != "qfq"
        ):
            raise ValueError(f"dataset snapshot contract is invalid: {dataset_id}")

        index_path = _contained_path(snapshot, str(manifest.get("file_index_path") or "FILE_INDEX.parquet"))
        try:
            file_index = pd.read_parquet(index_path)
        except Exception as exc:
            raise ValueError(f"dataset file index is unreadable: {dataset_id}") from exc
        if not {"symbol", "relative_path", "sha256"}.issubset(file_index.columns):
            raise ValueError(f"dataset file index is incomplete: {dataset_id}")
        by_symbol = {
            str(row["symbol"]).upper(): row
            for _, row in file_index.iterrows()
        }
        if any(symbol not in by_symbol for symbol in symbols):
            return None

        bars: list[PriceBar] = []
        for symbol in symbols:
            row = by_symbol[symbol]
            path = _contained_path(snapshot, str(row["relative_path"]))
            if not path.is_file() or _sha256(path) != str(row["sha256"]):
                raise ValueError(f"price file hash mismatch: {symbol}")
            try:
                frame = pd.read_parquet(path, columns=["date", "open", "close"])
            except Exception as exc:
                raise ValueError(f"price file is unreadable: {symbol}") from exc
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
            frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.dropna(subset=["date", "open", "close"])
            frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
            if end_date is not None:
                frame = frame[frame["date"] <= end_date]
            if frame["date"].duplicated().any():
                raise ValueError(f"price file contains duplicate dates: {symbol}")
            for row_value in frame.sort_values("date").itertuples(index=False):
                bars.append(
                    PriceBar(
                        symbol=symbol,
                        date=row_value.date,
                        open=float(row_value.open),
                        close=float(row_value.close),
                    )
                )

        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        return QuantPriceLoad(
            symbols=symbols,
            bars=tuple(bars),
            dataset_id=dataset_id,
            snapshot_id=str(manifest.get("snapshot_id") or ""),
            price_basis="qfq",
            provider=str(source.get("provider") or "unknown"),
            latest_trade_date=str(manifest.get("latest_trade_date") or ""),
            manifest_sha256=manifest_hash,
        )
