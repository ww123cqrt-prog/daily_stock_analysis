"""Build script-backtest inputs from the active quant_data read service."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from quant_workflow.scripts.quant_data_service_client import (
    DEFAULT_QUANT_DATA_SERVICE_URL,
    NativeDatasetIdentity,
    QuantDataServiceClient,
)
from src.core.script_market import MarketBundle, MarketView
from src.services.quant_workflow_paths import reports_root


def _resolve_symbol(value: str, known: set[str]) -> str | None:
    raw = str(value).strip().upper()
    candidates = [raw]
    if len(raw) == 6 and raw.isdigit():
        candidates.extend((f"SH{raw}", f"SZ{raw}"))
    return next((candidate for candidate in candidates if candidate in known), None)


def _read_bars(
    client: QuantDataServiceClient,
    dataset: NativeDatasetIdentity,
    symbols: Sequence[str],
    *,
    fields: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    frames = []
    for offset in range(0, len(symbols), 50):
        chunk = symbols[offset : offset + 50]
        result = client.native_bars_batch(
            dataset, chunk, basis="raw" if dataset.dataset_id == "cn_index_daily" else "qfq",
            fields=fields, start=start, end=end,
        )
        frames.append(result.frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _wide_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        raise ValueError("quant_data returned no verified bars for this period")
    expected = {"date", "symbol", "open", "close", "volume"}
    if not expected.issubset(frame.columns):
        raise ValueError("quant_data bar columns are incomplete")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("quant_data returned duplicate symbol/date bars")
    dates = np.array(sorted(frame["date"].unique()), dtype=object)
    symbols = np.array(sorted(frame["symbol"].unique()), dtype=object)
    date_index = {value: index for index, value in enumerate(dates)}
    symbol_index = {value: index for index, value in enumerate(symbols)}
    open_matrix = np.full((len(dates), len(symbols)), np.nan)
    close_matrix = np.full_like(open_matrix, np.nan)
    volume_matrix = np.full_like(open_matrix, np.nan)
    for row in frame.itertuples(index=False):
        day, symbol = date_index[row.date], symbol_index[row.symbol]
        open_matrix[day, symbol] = float(row.open)
        close_matrix[day, symbol] = float(row.close)
        volume_matrix[day, symbol] = float(row.volume)
    return (
        np.array([day.toordinal() for day in dates], dtype=np.int32),
        symbols, open_matrix, close_matrix, volume_matrix,
    )


def _dataset_meta(dataset: NativeDatasetIdentity) -> dict[str, object]:
    return {
        "module_id": dataset.dataset_id,
        "module_release_sha256": dataset.module_release_sha256,
        "content_manifest_sha256": dataset.content_manifest_sha256,
        "component_identities": dict(dataset.component_identities),
        "requested_through": dataset.requested_through,
    }


def load_native_market_view(
    pool: str,
    *,
    custom_symbols: Sequence[str] | None,
    start: str | None,
    end: str | None,
    cache_dir: Path | None,
) -> tuple[MarketView, dict[str, object]]:
    if not start or not end or start > end:
        raise ValueError("production.v2 backtests require a start and end date")
    if pool == "all_a":
        raise ValueError("production.v2 service cannot return the complete all-A PIT membership")
    if pool not in {"custom", "etf", "csi800"}:
        raise ValueError(f"unknown pool: {pool}")
    with QuantDataServiceClient(
        base_url=os.getenv("QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL)
    ) as client:
        release = client.native_ready()
        equity = client.native_dataset("ashare_daily_raw", expected_release=release) if pool != "etf" else None
        etf = client.native_dataset("cn_etf_daily", expected_release=release) if pool in {"etf", "custom"} else None
        requested = list(custom_symbols or ())
        ashare_symbols: list[str] = []
        etf_symbols: list[str] = []
        if pool == "etf":
            assert etf is not None
            etf_symbols = list(etf.symbols)
        elif pool == "custom":
            if not requested:
                raise ValueError("custom pool requires custom_symbols")
            assert equity is not None and etf is not None
            ashare_known, etf_known = set(equity.symbols), set(etf.symbols)
            missing = []
            for value in requested:
                stock = _resolve_symbol(value, ashare_known)
                fund = _resolve_symbol(value, etf_known)
                if stock:
                    ashare_symbols.append(stock)
                elif fund:
                    etf_symbols.append(fund)
                else:
                    missing.append(str(value))
            if missing:
                raise ValueError("custom pool symbols not published: " + ", ".join(missing))

        pit_members: dict[str, set[str]] = {}
        pit_identity: NativeDatasetIdentity | None = None
        index_identity: NativeDatasetIdentity | None = None
        if pool == "csi800":
            pit = client.native_dataset("pit_csi800", expected_release=release)
            pit_identity = pit.identity
            index = client.native_dataset("cn_index_daily", expected_release=release)
            index_identity = index.identity
            calendar = _read_bars(
                client, index.identity, ["SH000906"], fields=("close",), start=start, end=end,
            )
            for day in sorted(pd.to_datetime(calendar["date"]).dt.date.unique()):
                day_text = day.isoformat()
                result = client.native_named_component(
                    pit.identity, "daily", start=day_text, end=day_text,
                )
                pit_members[day_text] = set(result.frame["symbol"].astype(str))
            ashare_symbols = sorted(set().union(*pit_members.values())) if pit_members else []

        frames = []
        fields = ("open", "close", "volume")
        if ashare_symbols:
            assert equity is not None
            frames.append(_read_bars(client, equity.identity, sorted(set(ashare_symbols)), fields=fields, start=start, end=end))
        if etf_symbols:
            assert etf is not None
            frames.append(_read_bars(client, etf.identity, sorted(set(etf_symbols)), fields=fields, start=start, end=end))
        bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dates, symbols, opens, closes, volumes = _wide_matrices(bars)
        if pool == "csi800":
            members = np.array(
                [[symbol in pit_members.get(date.fromordinal(int(day)).isoformat(), set()) for symbol in symbols] for day in dates],
                dtype=bool,
            )
        else:
            members = ~np.isnan(closes)
        index_close = np.full(len(dates), np.nan)
        if pool == "custom" and ashare_symbols:
            index = client.native_dataset("cn_index_daily", expected_release=release)
            index_identity = index.identity
            benchmark = _read_bars(
                client, index.identity, ["SZ399006"], fields=("close",), start=start, end=end,
            )
            lookup = dict(zip(
                pd.to_datetime(benchmark["date"]).dt.date,
                benchmark["close"],
            ))
            index_close = np.array(
                [float(lookup.get(date.fromordinal(int(day)), np.nan)) for day in dates]
            )
        # The issue ledger can change while the release SHA stays fixed, so
        # native bundles are validated afresh on every run.
        meta: dict[str, object] = {
            "pool": pool,
            "dataset_set_version": "production.v2",
            "release": release.as_dict(),
            "release_set_sha256": release.release_set_sha256,
            "generation": release.generation,
            "dataset_id": "+".join(
                name for name, selected in (("ashare_daily_raw", ashare_symbols), ("cn_etf_daily", etf_symbols)) if selected
            ),
            "datasets": {
                item.dataset_id: _dataset_meta(item)
                for item in (equity.identity if ashare_symbols and equity else None,
                             etf.identity if etf_symbols and etf else None,
                             pit_identity, index_identity)
                if item is not None
            },
            "price_basis": "qfq",
            "symbol_count": int(len(symbols)),
            "trading_days": int(len(dates)),
            "start": start,
            "end": end,
        }
        if requested:
            meta["custom_symbols"] = requested
        bundle = MarketBundle(
            dates=dates, symbols=symbols, open=opens, close=closes,
            members=members, meta=meta, volume=volumes, index_close=index_close,
        )
        destination = (cache_dir or reports_root() / "cache" / "universe") / (
            f"production_v2_{release.release_set_sha256[:12]}_{uuid4().hex[:12]}"
        )
        bundle.save(destination)
        meta["bundle_dir"] = str(destination)
        return MarketView.from_bundle(bundle), meta


def load_native_benchmark_series(
    symbols: Sequence[str],
    *,
    start: str | None,
    end: str | None,
    expected_release: Mapping[str, object] | None,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    if not start or not end:
        raise ValueError("production.v2 benchmarks require a start and end date")
    with QuantDataServiceClient(
        base_url=os.getenv("QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL)
    ) as client:
        release = client.native_ready()
        if expected_release is not None and release.as_dict() != dict(expected_release):
            raise ValueError("quant_data release changed during backtest")
        datasets = {
            name: client.native_dataset(name, expected_release=release)
            for name in ("cn_index_daily", "ashare_daily_raw", "cn_etf_daily")
        }
        result: dict[str, pd.DataFrame] = {}
        for value in symbols:
            for name in ("cn_index_daily", "ashare_daily_raw", "cn_etf_daily"):
                dataset = datasets[name]
                resolved = _resolve_symbol(value, set(dataset.symbols))
                if resolved is None:
                    continue
                frame = _read_bars(
                    client, dataset.identity, [resolved], fields=("close",), start=start, end=end,
                )
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
                result[resolved] = frame[["date", "close"]]
                break
        return result
