# -*- coding: utf-8 -*-
"""Vectorized market bundle shared by the sandbox runner and the backtest engine.

A ``MarketBundle`` is a directory containing:

- ``market.npz``  numpy arrays (dates / symbols / open / close / membership mask)
- ``universe.json`` pool metadata + quant_data snapshot identity

Both the sandbox child process and the DSA backend load it through
``MarketView``, so the strategy script and the portfolio backtest share one
vectorized representation instead of millions of ``PriceBar`` objects.

Arrays use ``NaN`` for missing prices (symbol not listed on a date).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np


UNIVERSE_VERSION = "universe-v1"


def _dates_to_ints(values: Sequence[date]) -> np.ndarray:
    return np.array(
        [value.toordinal() for value in values], dtype=np.int32
    )


@dataclass(frozen=True)
class MarketBundle:
    """On-disk bundle produced by the DSA backend for the sandbox run."""

    dates: np.ndarray  # int32 ordinal dates, ascending
    symbols: np.ndarray  # object array of uppercase symbols
    open: np.ndarray  # float64 (dates, symbols), NaN when missing
    close: np.ndarray  # float64 (dates, symbols), NaN when missing
    members: np.ndarray  # bool (dates, symbols), True when tradable on that date
    meta: dict[str, Any]
    volume: np.ndarray | None = None
    index_close: np.ndarray | None = None

    def save(self, bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bundle_dir / "market.npz",
            dates=self.dates,
            symbols=self.symbols,
            open=self.open,
            close=self.close,
            members=self.members,
            volume=(self.volume if self.volume is not None else np.full_like(self.close, np.nan, dtype=np.float64)),
            index_close=(self.index_close if self.index_close is not None else np.full(len(self.dates), np.nan, dtype=np.float64)),
        )
        (bundle_dir / "universe.json").write_text(
            json.dumps(
                {
                    "version": UNIVERSE_VERSION,
                    "meta": self.meta,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class MarketView:
    """Vectorized read-only view used by the script data API."""

    def __init__(
        self,
        dates: np.ndarray,
        symbols: np.ndarray,
        open_matrix: np.ndarray,
        close_matrix: np.ndarray,
        members: np.ndarray,
        meta: dict[str, Any],
        volume_matrix: np.ndarray | None = None,
        index_close: np.ndarray | None = None,
    ) -> None:
        self.dates = dates
        self.symbols = symbols
        self.open = open_matrix
        self.close = close_matrix
        self.members = members
        self.meta = meta
        self.volume = volume_matrix if volume_matrix is not None else np.full_like(close_matrix, np.nan, dtype=np.float64)
        self.index_close = index_close if index_close is not None else np.full(len(dates), np.nan, dtype=np.float64)
        self._symbol_index = {str(symbol): i for i, symbol in enumerate(symbols)}

    @classmethod
    def load(cls, bundle_dir: Path) -> "MarketView":
        with np.load(bundle_dir / "market.npz", allow_pickle=True) as archive:
            dates = archive["dates"].astype(np.int32)
            symbols = archive["symbols"]
            open_matrix = archive["open"]
            close_matrix = archive["close"]
            members = archive["members"]
            volume_matrix = archive["volume"] if "volume" in archive.files else np.full_like(close_matrix, np.nan, dtype=np.float64)
            index_close = archive["index_close"] if "index_close" in archive.files else np.full(len(dates), np.nan, dtype=np.float64)
        payload = json.loads(
            (bundle_dir / "universe.json").read_text(encoding="utf-8")
        )
        return cls(dates, symbols, open_matrix, close_matrix, members, payload.get("meta", {}), volume_matrix, index_close)

    @classmethod
    def from_bundle(cls, bundle: MarketBundle) -> "MarketView":
        return cls(
            bundle.dates,
            bundle.symbols,
            bundle.open,
            bundle.close,
            bundle.members,
            bundle.meta,
            bundle.volume,
            bundle.index_close,
        )

    @property
    def calendar(self) -> tuple[date, ...]:
        return tuple(date.fromordinal(int(value)) for value in self.dates)

    @property
    def calendar_ordinals(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.dates)

    def symbol_index(self, symbol: str) -> int:
        normalized = str(symbol).strip().upper()
        if normalized not in self._symbol_index:
            raise ValueError(f"symbol is outside the universe: {symbol}")
        return self._symbol_index[normalized]

    def close_at(self, symbol: str, day_index: int) -> float | None:
        value = self.close[day_index, self.symbol_index(symbol)]
        return None if bool(np.isnan(value)) else float(value)

    def open_at(self, symbol: str, day_index: int) -> float | None:
        value = self.open[day_index, self.symbol_index(symbol)]
        return None if bool(np.isnan(value)) else float(value)

    def volume_at(self, symbol: str, day_index: int) -> float | None:
        value = self.volume[day_index, self.symbol_index(symbol)]
        return None if bool(np.isnan(value)) else float(value)

    def members_on(self, day_index: int) -> list[str]:
        row = self.members[day_index]
        return [str(self.symbols[i]) for i in np.flatnonzero(row)]


def build_market_bundle(
    *,
    dates: Sequence[date],
    symbols: Sequence[str],
    open_frame: Any,
    close_frame: Any,
    members_by_date: Any,
    meta: dict[str, Any],
    volume_frame: Any | None = None,
    index_close: Any | None = None,
) -> MarketBundle:
    """Assemble a bundle from pandas frames.

    ``open_frame`` / ``close_frame`` are wide pandas DataFrames indexed by
    date with one column per symbol. ``members_by_date`` is either a bool
    array-like of shape (len(dates), len(symbols)) or a pandas DataFrame with
    the same layout (True = tradable).
    """
    normalized_symbols = [str(symbol).strip().upper() for symbol in symbols]
    date_ords = _dates_to_ints(dates)
    open_matrix = open_frame[normalized_symbols].to_numpy(dtype=np.float64)
    close_matrix = close_frame[normalized_symbols].to_numpy(dtype=np.float64)
    if volume_frame is None:
        volume_matrix = np.full_like(close_matrix, np.nan, dtype=np.float64)
    else:
        volume_matrix = volume_frame[normalized_symbols].to_numpy(dtype=np.float64)
    if index_close is None:
        index_values = np.full(len(dates), np.nan, dtype=np.float64)
    else:
        index_values = np.asarray(index_close, dtype=np.float64)
        if index_values.shape != (len(dates),):
            raise ValueError("index_close must have one value per date")
    members = np.asarray(members_by_date, dtype=bool)
    if members.shape != (len(dates), len(normalized_symbols)):
        raise ValueError("members matrix shape must match (dates, symbols)")
    return MarketBundle(
        dates=date_ords,
        symbols=np.array(normalized_symbols, dtype=object),
        open=open_matrix,
        close=close_matrix,
        members=members,
        meta=meta,
        volume=volume_matrix,
        index_close=index_values,
    )
