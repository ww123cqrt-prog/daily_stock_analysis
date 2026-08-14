# -*- coding: utf-8 -*-
"""Read governed daily bars through the local quant_data service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Sequence

import pandas as pd

from quant_workflow.scripts.quant_data_service_client import (
    DEFAULT_QUANT_DATA_SERVICE_URL,
    QuantDataClientError,
    QuantDataServiceClient,
)
from src.core.strategy_backtest import PriceBar


FORMAL_QFQ_DATASETS = ("cn_etf_daily", "ashare_daily_qfq")


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
    content_sha256: str
    data_kind: str


class QuantPriceRepository:
    """Load all requested symbols from one formal qfq dataset."""

    def __init__(self, *, client: QuantDataServiceClient | None = None):
        self._client = client

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

        owns_client = self._client is None
        client = self._client or QuantDataServiceClient(
            base_url=os.getenv(
                "QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL
            )
        )
        try:
            for dataset_id in FORMAL_QFQ_DATASETS:
                try:
                    result = client.query_bars_batch(
                        dataset_id,
                        normalized,
                        basis="qfq",
                        fields=["open", "close"],
                        end=end_date.isoformat() if end_date is not None else None,
                    )
                except QuantDataClientError as exc:
                    if exc.code == "symbol_unavailable":
                        continue
                    raise ValueError(
                        f"quant_data data backend rejected the price request ({exc.code})"
                    ) from exc
                return self._normalize_result(normalized, result.frame, result.identity)
        finally:
            if owns_client:
                client.close()
        raise ValueError(
            "No ready canonical qfq dataset contains all requested symbols: "
            + ", ".join(normalized)
        )

    @staticmethod
    def _normalize_result(normalized, frame, identity) -> QuantPriceLoad:
        required = {"symbol", "date", "open", "close"}
        if not required.issubset(frame.columns):
            raise ValueError("quant_data data backend returned incomplete price bars")
        prepared = frame.loc[:, ["symbol", "date", "open", "close"]].copy()
        prepared["symbol"] = prepared["symbol"].astype(str).str.upper()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.date
        prepared["open"] = pd.to_numeric(prepared["open"], errors="coerce")
        prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
        prepared = prepared.dropna(subset=["symbol", "date", "open", "close"])
        prepared = prepared[(prepared["open"] > 0) & (prepared["close"] > 0)]
        if not set(prepared["symbol"]).issubset(normalized):
            raise ValueError("quant_data data backend returned an unexpected symbol")
        if prepared.duplicated(["symbol", "date"]).any():
            raise ValueError("quant_data data backend returned duplicate price dates")

        bars = tuple(
            PriceBar(
                symbol=str(row.symbol),
                date=row.date,
                open=float(row.open),
                close=float(row.close),
            )
            for row in prepared.sort_values(["symbol", "date"]).itertuples(index=False)
        )
        return QuantPriceLoad(
            symbols=tuple(normalized),
            bars=bars,
            dataset_id=identity.dataset_id,
            snapshot_id=identity.snapshot_id,
            price_basis=identity.price_basis,
            provider="quant_data",
            latest_trade_date=identity.latest_trade_date,
            manifest_sha256=identity.manifest_sha256,
            content_sha256=identity.content_sha256,
            data_kind=identity.data_kind,
        )
