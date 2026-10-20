# -*- coding: utf-8 -*-
"""Read local snapshot prices; retain explicit clients for historical callers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pathlib import Path

from data_provider.base import is_bse_code, normalize_stock_code

from quant_workflow.scripts.quant_data_service_client import (
    DEFAULT_QUANT_DATA_SERVICE_URL,
    QuantDataServiceClient,
    NativeDatasetIdentity,
    NativeReleaseIdentity,
)
from src.core.strategy_backtest import PriceBar


FORMAL_QFQ_DATASETS = ("cn_etf_daily", "ashare_daily_raw")


def normalize_market_symbol(value: str) -> str:
    original = str(value or "").strip().upper()
    symbol = normalize_stock_code(original)
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"invalid A-share or ETF symbol: {value}")
    if original.startswith(("SH", "SZ", "BJ")):
        return f"{original[:2]}{symbol}"
    if original.endswith((".SH", ".SZ", ".BJ")):
        return f"{original[-2:]}{symbol}"
    if is_bse_code(symbol):
        return f"BJ{symbol}"
    if symbol.startswith(("5", "6", "9")):
        return f"SH{symbol}"
    if symbol.startswith(("0", "1", "2", "3")):
        return f"SZ{symbol}"
    if symbol.startswith(("4", "8")):
        return f"BJ{symbol}"
    raise ValueError(f"cannot infer exchange for symbol: {value}")


@dataclass(frozen=True)
class QuantPriceLoad:
    symbols: tuple[str, ...]
    bars: tuple[PriceBar, ...]
    dataset_id: str
    snapshot_id: str | None
    price_basis: str
    provider: str
    latest_trade_date: str | None
    manifest_sha256: str | None
    content_sha256: str | None
    data_kind: str
    dataset_set_version: str | None = None
    generation: int | None = None
    release_set_sha256: str | None = None
    module_identities: Mapping[str, Mapping[str, Any]] | None = None
    dataset_identities: Mapping[str, Mapping[str, Any]] | None = None


class QuantPriceRepository:
    """Load requested symbols from pinned local qfq snapshots by default."""

    def __init__(self, *, client: QuantDataServiceClient | None = None, quant_data_root: Path | None = None):
        self._client = client
        self._root = quant_data_root

    def load(
        self,
        symbols: Sequence[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> QuantPriceLoad:
        normalized = tuple(normalize_market_symbol(symbol) for symbol in symbols)
        if not normalized:
            raise ValueError("at least one symbol is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("symbols must not contain duplicates")
        if start_date is None or end_date is None or start_date > end_date:
            raise ValueError("price requests require a valid start and end date")

        if self._client is None:
            return self._load_local(normalized, start_date, end_date)

        owns_client = self._client is None
        client = self._client or QuantDataServiceClient(
            base_url=os.getenv(
                "QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL
            )
        )
        try:
            release = client.native_ready()
            grouped: dict[str, tuple[NativeDatasetIdentity, list[str]]] = {}
            remaining = list(normalized)
            for dataset_id in FORMAL_QFQ_DATASETS:
                dataset = client.native_dataset(dataset_id, expected_release=release)
                published = set(dataset.symbols)
                selected = [symbol for symbol in remaining if symbol in published]
                if selected:
                    grouped[dataset_id] = (dataset.identity, selected)
                    remaining = [symbol for symbol in remaining if symbol not in selected]
                if not remaining:
                    break
            if remaining:
                raise ValueError(
                    "No active production.v2 qfq module contains: " + ", ".join(remaining)
                )
            frames: list[pd.DataFrame] = []
            for identity, selected in grouped.values():
                for offset in range(0, len(selected), 50):
                    result = client.native_bars_batch(
                        identity,
                        selected[offset : offset + 50],
                        basis="qfq",
                        fields=("open", "close"),
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                    )
                    frames.append(result.frame)
            return self._normalize_result(
                normalized, pd.concat(frames, ignore_index=True), release,
                {name: identity for name, (identity, _) in grouped.items()},
            )
        finally:
            if owns_client:
                client.close()

    @staticmethod
    def _price_bars(normalized: tuple[str, ...], frame: pd.DataFrame) -> tuple[PriceBar, ...]:
        required = {"symbol", "date", "open", "close"}
        if not required.issubset(frame.columns):
            raise ValueError("quant_data data backend returned incomplete price bars")
        prepared = frame.loc[:, ["symbol", "date", "open", "close"]].copy()
        prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date
        prepared["open"] = pd.to_numeric(prepared["open"], errors="coerce")
        prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
        prices = prepared[["open", "close"]].to_numpy(dtype=float)
        if (
            prepared[["symbol", "date", "open", "close"]].isna().any().any()
            or not np.isfinite(prices).all()
            or (prices <= 0).any()
        ):
            raise ValueError("quant_data data backend returned invalid price bars")
        if not set(prepared["symbol"]).issubset(normalized):
            raise ValueError("quant_data data backend returned an unexpected symbol")
        if prepared.duplicated(["symbol", "date"]).any():
            raise ValueError("quant_data data backend returned duplicate price dates")

        return tuple(
            PriceBar(
                symbol=str(row.symbol),
                date=row.date,
                open=float(row.open),
                close=float(row.close),
            )
            for row in prepared.sort_values(["symbol", "date"]).itertuples(index=False)
        )

    def _load_local(self, normalized, start_date, end_date) -> QuantPriceLoad:
        from src.services.script_universe_loader import load_market_view

        view, meta = load_market_view(
            "custom", custom_symbols=normalized,
            start=start_date.isoformat(), end=end_date.isoformat(),
            quant_data_root=self._root,
        )
        rows, columns = np.where(~np.isnan(view.open) & ~np.isnan(view.close))
        frame = pd.DataFrame({
            "symbol": view.symbols[columns],
            "date": [view.calendar[int(row)] for row in rows],
            "open": view.open[rows, columns], "close": view.close[rows, columns],
        })
        return QuantPriceLoad(
            symbols=normalized, bars=self._price_bars(normalized, frame),
            dataset_id=meta["dataset_id"], snapshot_id=meta["snapshot_id"],
            price_basis="qfq", provider="quant_qmt_data",
            latest_trade_date=meta.get("latest_trade_date"),
            manifest_sha256=meta.get("manifest_sha256"),
            content_sha256=meta.get("content_sha256"), data_kind="daily_bars",
            dataset_set_version="local_snapshots", dataset_identities=meta["datasets"],
        )

    @staticmethod
    def _normalize_result(
        normalized: tuple[str, ...], frame: pd.DataFrame,
        release: NativeReleaseIdentity, identities: Mapping[str, NativeDatasetIdentity],
    ) -> QuantPriceLoad:
        return QuantPriceLoad(
            symbols=tuple(normalized),
            bars=QuantPriceRepository._price_bars(normalized, frame),
            dataset_id="+".join(identities),
            snapshot_id=None,
            price_basis="qfq",
            provider="quant_data",
            latest_trade_date=None,
            manifest_sha256=None,
            content_sha256=None,
            data_kind="daily_bars",
            dataset_set_version="production.v2",
            generation=release.generation,
            release_set_sha256=release.release_set_sha256,
            module_identities={name: identity.as_dict() for name, identity in identities.items()},
        )
