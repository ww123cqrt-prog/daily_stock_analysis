# -*- coding: utf-8 -*-
"""Adapt one frozen QMT experiment input to DSA's existing sandbox format."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Optional, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from quant_qmt_data import QMTClient, load_dataset, save_dataset
from src.core.script_market import MarketBundle, MarketView
from src.repositories.quant_price_repo import normalize_market_symbol
from src.services.quant_workflow_paths import reports_root

VALID_POOLS = ("etf", "csi800", "all_a", "custom")
INDEX_BENCHMARK_LABELS = {
    "SH000300": "沪深300", "SH000905": "中证500", "SH000906": "中证800",
    "SZ399006": "创业板指", "SH000688": "科创50",
}
INDEX_BENCHMARK_SYMBOLS = tuple(INDEX_BENCHMARK_LABELS)


class MarketDataUnavailable(RuntimeError):
    """The requested experiment input cannot supply this backtest."""


def _normalize_ashare_symbol(value: str) -> str:
    return normalize_market_symbol(value)


def _research_adjustment(client, selected, *, benchmarks=()):
    """Keep deployment stock/ETF conventions without mixing bases in one input."""
    from quant_qmt_data.reader import ADJUSTMENTS
    default = os.environ.get('QMT_ADJUSTMENT', 'front')
    stock = os.environ.get('QMT_STOCK_ADJUSTMENT', default)
    etf = os.environ.get('QMT_ETF_ADJUSTMENT', default)
    if stock not in ADJUSTMENTS or etf not in ADJUSTMENTS:
        raise ValueError('Unsupported configured QMT research adjustment')
    if stock == etf:
        return stock
    types = client.instrument_types(list(dict.fromkeys([*selected, *benchmarks])))
    modes = {etf if types.get(symbol, {}).get('etf') else stock
             for symbol in selected}
    # Index benchmarks are neutral; a stock/ETF benchmark retains its own basis.
    modes.update(etf if types[symbol].get('etf') else stock
                 for symbol in benchmarks
                 if types.get(symbol, {}).get('etf') or types.get(symbol, {}).get('stock'))
    if len(modes) != 1:
        raise ValueError('Different stock/ETF bases require separate inputs; mixed bases are not allowed')
    return modes.pop()


def load_market_input(*, quant_data_root=None, pool="custom", custom_symbols=None,
                      start=None, end=None, benchmark_symbols=()):
    """Load a selected input, or freeze one live request when none is configured.

    A configured but incomplete/damaged input never falls back to the SDK.
    This is the file/SDK boundary shared by backtests and historical fetchers.
    """
    configured = quant_data_root or os.environ.get("QMT_DATASET_DIR")
    try:
        if configured:
            directory = Path(configured).expanduser().resolve()
            return load_dataset(directory), directory
        if pool == "csi800":
            raise MarketDataUnavailable("CSI800 requires frozen historical membership; 当前历史成分不可用。")
        if not start or not end:
            raise ValueError("Live QMT reads require start and end dates, or QMT_DATASET_DIR")
        client = QMTClient.from_env()
        if pool == "custom":
            selected = [normalize_market_symbol(s) for s in custom_symbols or ()]
        elif pool == "etf":
            selected = client.universe("沪深ETF")
        else:
            selected = list(dict.fromkeys(client.universe("沪深A股") + client.universe("京市A股")))
        if not selected:
            raise MarketDataUnavailable("The requested QMT universe is empty")
        benchmarks = list(dict.fromkeys(normalize_market_symbol(s) for s in benchmark_symbols))
        required_symbols = set(selected + benchmarks)
        if pool in {"custom", "all_a"} and "SZ399006" not in selected + benchmarks:
            benchmarks.append("SZ399006")
        symbols = list(dict.fromkeys(selected + benchmarks))
        adjustment = _research_adjustment(client, selected, benchmarks=benchmarks)
        bars = client.read_bars(symbols, start=start, end=end, adjustment=adjustment, volume_unit="base")
        calendar = sorted({pd.Timestamp(day).date().isoformat()
                           for market in sorted({symbol[:2] for symbol in symbols})
                           for day in client.trading_dates(market, start=start, end=end)})
        completed = bars.attrs["completed_through"]
        calendar = [day for day in calendar if day <= completed]
        if not calendar:
            raise MarketDataUnavailable("QMT returned no completed trading calendar for this input")
        missing = required_symbols - set(bars.symbol)
        if missing:
            raise MarketDataUnavailable("QMT has no local bars for: " + ", ".join(sorted(missing)))
        bars.attrs["benchmark_symbols"] = benchmarks
        bars.attrs["universe_read_at"] = datetime.now(timezone.utc).isoformat()
        directory = reports_root() / "inputs" / uuid4().hex
        save_dataset(bars, directory, calendar=calendar, universe=selected,
                     universe_kind="custom" if pool == "custom" else "current")
        return load_dataset(directory), directory.resolve()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise MarketDataUnavailable("QMT experiment input unavailable; 请检查输入或在 QMT 中维护所需行情。") from exc


def _interval_membership(frame, *, symbol_col, start_col, end_col, dates, symbols):
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    for row in frame.to_dict("records"):
        symbol = normalize_market_symbol(row[symbol_col])
        if symbol not in symbols:
            continue
        first = pd.Timestamp(row[start_col]).date().toordinal()
        last = pd.Timestamp(row[end_col]).date().toordinal() if row.get(end_col) else date.max.toordinal()
        mask[:, list(symbols).index(symbol)] |= (dates >= first) & (dates <= last)
    return mask


def _build_wide_matrices(long_frame, *, calendar, symbols):
    dates = pd.DatetimeIndex(calendar)
    columns = pd.Index(symbols)
    indexed = long_frame.set_index(["date", "symbol"])
    matrices = []
    for field in ("open", "close", "volume"):
        wide = indexed[field].unstack("symbol").reindex(index=dates, columns=columns)
        matrices.append(wide.to_numpy(dtype=np.float64))
    return (np.array([day.date().toordinal() for day in dates], dtype=np.int32),
            np.array(symbols, dtype=object), *matrices)


def load_market_view(pool: str, *, custom_symbols: Optional[Sequence[str]] = None,
                     start: Optional[str] = None, end: Optional[str] = None,
                     quant_data_root: Optional[Path] = None, cache_dir: Optional[Path] = None,
                     use_cache: bool = True, benchmark_symbols: Sequence[str] = ()):
    """Return a full-calendar MarketView and the truthful experiment identity."""
    if pool not in VALID_POOLS:
        raise ValueError(f"unknown pool: {pool}")
    if pool == "custom" and not custom_symbols:
        raise ValueError("custom pool requires custom_symbols")
    if start and end and pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("start cannot be after end")
    bars, directory = load_market_input(quant_data_root=quant_data_root, pool=pool,
        custom_symbols=custom_symbols, start=start, end=end, benchmark_symbols=benchmark_symbols)
    metadata = bars.attrs
    universe = metadata["universe"]
    units = metadata["volume_units"]
    if pool == "custom":
        selected = list(dict.fromkeys(normalize_market_symbol(s) for s in custom_symbols))
    elif pool == "etf":
        selected = [s for s in universe if units.get(s) == "units"]
    elif pool == "all_a":
        selected = [s for s in universe if units.get(s) == "shares"]
    else:
        if metadata["universe_kind"] != "historical" or not metadata.get("membership_intervals"):
            raise MarketDataUnavailable("CSI800 requires actual frozen historical membership; 历史成分不可用。")
        selected = list(universe)
    selected = sorted(selected)
    missing = set(selected) - set(universe) | (set(selected) - set(bars.symbol))
    if missing or not selected:
        raise MarketDataUnavailable("Requested input universe unavailable: " + ", ".join(sorted(missing)))
    if any(units.get(symbol) not in {"shares", "units"} for symbol in selected):
        raise MarketDataUnavailable("Backtest input requires base shares/units, not native lots")
    calendar = pd.DatetimeIndex(metadata["calendar"])
    if start:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    if calendar.empty:
        raise MarketDataUnavailable("The selected input has no trading dates in this range")
    frame = bars.loc[bars.symbol.isin(selected)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    suspended = frame["suspend_flag"].eq(1)
    frame.loc[suspended, ["open", "close", "volume"]] = np.nan
    dates, symbols, opens, closes, volumes = _build_wide_matrices(frame, calendar=calendar, symbols=selected)
    members = np.isfinite(closes)
    if metadata["universe_kind"] == "historical":
        members &= _interval_membership(pd.DataFrame(metadata["membership_intervals"]),
            symbol_col="symbol", start_col="start_date", end_col="end_date", dates=dates, symbols=symbols)
    cyb = bars.loc[bars.symbol.eq("SZ399006")].set_index("date")["close"]
    cyb.index = pd.to_datetime(cyb.index)
    index_close = cyb.reindex(calendar).to_numpy(dtype=np.float64)
    meta = {key: metadata.get(key) for key in (
        "experiment_id", "bars_sha256", "inputs_sha256", "price_basis", "volume_units", "read_at",
        "completed_through", "universe_kind", "universe_read_at", "universe_recorded_at",
        "benchmark_symbols", "missing_symbols", "actual_start", "actual_end")}
    meta.update(input_dir=str(directory), provider="qmt", pool=pool, volume_unit="base",
        dataset_id=None, snapshot_id=None, dataset_set_version=None, manifest_sha256=None,
        content_sha256=None, datasets=None, latest_trade_date=metadata["actual_end"],
        symbol_count=len(symbols), trading_days=len(dates), start=start, end=end)
    identity = json.dumps(dict(input=metadata, pool=pool, symbols=selected, start=start, end=end),
                          sort_keys=True, ensure_ascii=False)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    bundle_dir = Path(cache_dir) if cache_dir is not None else reports_root() / "cache" / "universe"
    bundle_dir = bundle_dir / key
    meta["bundle_dir"] = str(bundle_dir)
    if use_cache and (bundle_dir / "market.npz").is_file():
        return MarketView.load(bundle_dir), meta
    bundle = MarketBundle(dates=dates, symbols=symbols, open=opens, close=closes,
        members=members, meta=meta, volume=volumes, index_close=index_close)
    bundle.save(bundle_dir)
    return MarketView.from_bundle(bundle), meta


def load_view_for_run(bundle_dir: Path) -> MarketView:
    return MarketView.load(bundle_dir)


def load_benchmark_series(symbols: Sequence[str], *, input_dir=None, quant_data_root=None,
                          start=None, end=None):
    """Read comparison series from the same frozen input, never from the SDK."""
    if not symbols:
        return {}
    directory = input_dir or quant_data_root or os.environ.get("QMT_DATASET_DIR")
    if not directory:
        raise MarketDataUnavailable("Benchmarks require the backtest's frozen input_dir")
    bars, _ = load_market_input(quant_data_root=directory)
    metadata = bars.attrs
    allowed = set(metadata["universe"]) | set(metadata.get("benchmark_symbols", []))
    result = {}
    for value in symbols:
        symbol = normalize_market_symbol(value)
        if symbol not in allowed:
            raise MarketDataUnavailable(f"Benchmark is outside this frozen input: {symbol}")
        frame = bars.loc[bars.symbol.eq(symbol)].copy()
        if start:
            frame = frame.loc[pd.to_datetime(frame.date) >= pd.Timestamp(start)]
        if end:
            frame = frame.loc[pd.to_datetime(frame.date) <= pd.Timestamp(end)]
        if frame.empty:
            raise MarketDataUnavailable(f"Benchmark has no bars in this frozen input: {symbol}")
        frame["date"] = pd.to_datetime(frame.date).dt.date
        frame.attrs = metadata
        result[symbol] = frame
    return result
