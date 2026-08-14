# -*- coding: utf-8 -*-
"""Universe pool loading for the script research workbench.

Loads canonical quant_data snapshots into a vectorized :class:`MarketView`
with a PIT (point-in-time) tradable-membership mask, and caches the assembled
bundle keyed by the immutable snapshot id so a pool is assembled only once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.core.script_market import MarketBundle, MarketView, build_market_bundle
from src.services.quant_workflow_paths import reports_root

logger = logging.getLogger(__name__)

VALID_POOLS = ("etf", "csi800", "all_a", "custom")

_ASHARE_DATASET_ID = "ashare_daily_qfq"
_ETF_DATASET_ID = "cn_etf_daily"

_UNIVERSE_CACHE_VERSION = "universe-cache-v1"


def _normalize_ashare_symbol(value: str) -> str:
    """Normalize a bare A-share code to the SH/SZ-prefixed canonical form."""
    symbol = str(value).strip().upper()
    if symbol.startswith(("SH", "SZ")):
        return symbol
    if symbol.startswith("6"):
        return f"SH{symbol}"
    if symbol.startswith(("0", "3")):
        return f"SZ{symbol}"
    return symbol


def _file_sha256(path: Path) -> str:
    """Content hash of a reference file, used to key universe caches."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_snapshot_latest(quant_data_root: Path, family: str, dataset_id: str) -> dict[str, Any]:
    latest_path = quant_data_root / "parquet" / family / dataset_id / "LATEST.json"
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    snapshot_id = payload.get("snapshot_id")
    snapshot_path = payload.get("snapshot_path")
    if not isinstance(snapshot_id, str) or not isinstance(snapshot_path, str):
        raise ValueError(f"{dataset_id} LATEST.json is invalid")
    manifest_sha256 = str(payload.get("manifest_sha256") or "")
    content_sha256 = ""
    latest_trade_date = ""
    if manifest_sha256:
        manifest_path = quant_data_root / snapshot_path / "DATASET_MANIFEST.json"
        if manifest_path.is_file():
            actual = _file_sha256(manifest_path)
            if actual != manifest_sha256:
                raise ValueError(
                    f"{dataset_id} snapshot manifest does not match LATEST.json: "
                    f"declared {manifest_sha256}, actual {actual}"
                )
            try:
                manifest_payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                content_sha256 = str(manifest_payload.get("content_sha256") or "")
                latest_trade_date = str(
                    manifest_payload.get("latest_trade_date") or ""
                )
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "snapshot manifest unreadable, content identity unavailable: %s",
                    manifest_path,
                )
                latest_trade_date = ""
    return {
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha256,
        "content_sha256": content_sha256,
        "latest_trade_date": latest_trade_date,
        "bars_dir": quant_data_root / snapshot_path / "bars",
    }


def _load_bars_long(
    bars_dir: Path,
    *,
    start: Optional[str],
    end: Optional[str],
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read date/symbol/open/close for per-symbol parquet files in one scan."""
    dataset = ds.dataset(str(bars_dir), format="parquet")
    # Snapshot date columns are timestamps (ns or us depending on the
    # producer); compare against timestamp scalars so string literals never
    # hit the missing pyarrow kernel.
    date_field = ds.field("date").cast("timestamp[ns]")
    expression = None
    if start is not None:
        expression = date_field >= pd.Timestamp(start)
    if end is not None:
        bound = date_field <= pd.Timestamp(end)
        expression = bound if expression is None else expression & bound
    if symbols is not None:
        wanted = [str(symbol).strip().upper() for symbol in symbols]
        bound = ds.field("symbol").isin(wanted)
        expression = bound if expression is None else expression & bound
    table = dataset.to_table(
        columns=["date", "symbol", "open", "close"],
        filter=expression,
    )
    return table.to_pandas()


def _interval_membership(
    frame: pd.DataFrame,
    *,
    symbol_col: str,
    start_col: str,
    end_col: str,
    dates: np.ndarray,
    symbols: np.ndarray,
) -> np.ndarray:
    """Build a (dates, symbols) bool mask from symbol membership intervals."""
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    symbol_index = {str(symbol): i for i, symbol in enumerate(symbols)}
    dates_ord = {int(value): row for row, value in enumerate(dates)}
    for _, row in frame.iterrows():
        symbol = str(row[symbol_col]).strip().upper()
        col = symbol_index.get(symbol)
        if col is None:
            continue
        start_value = row[start_col]
        end_value = row[end_col]
        start_ord = pd.Timestamp(start_value).toordinal() if pd.notna(start_value) else None
        end_ord = pd.Timestamp(end_value).toordinal() if pd.notna(end_value) else None
        for ord_value, row_index in dates_ord.items():
            if start_ord is not None and ord_value < start_ord:
                continue
            if end_ord is not None and ord_value > end_ord:
                continue
            mask[row_index, col] = True
    return mask


def _build_wide_matrices(
    long_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates_series = pd.to_datetime(long_frame["date"]).dt.date
    unique_dates = sorted(dates_series.unique())
    unique_symbols = sorted(str(symbol) for symbol in long_frame["symbol"].unique())
    date_index = {value: i for i, value in enumerate(unique_dates)}
    symbol_index = {symbol: i for i, symbol in enumerate(unique_symbols)}
    dates = np.array([value.toordinal() for value in unique_dates], dtype=np.int32)
    open_matrix = np.full((len(unique_dates), len(unique_symbols)), np.nan, dtype=np.float64)
    close_matrix = np.full_like(open_matrix, np.nan)
    date_codes = dates_series.map(date_index).to_numpy(dtype=np.int32)
    symbol_codes = long_frame["symbol"].astype(str).str.strip().str.upper().map(symbol_index).to_numpy(dtype=np.int32)
    open_values = long_frame["open"].to_numpy(dtype=np.float64)
    close_values = long_frame["close"].to_numpy(dtype=np.float64)
    open_matrix[date_codes, symbol_codes] = open_values
    close_matrix[date_codes, symbol_codes] = close_values
    return dates, np.array(unique_symbols, dtype=object), open_matrix, close_matrix


def _pool_cache_key(
    pool: str,
    snapshot_id: str,
    start: Optional[str],
    end: Optional[str],
    *,
    custom_symbols: Optional[Sequence[str]] = None,
    membership_sha256: str = "",
    secondary_snapshot_id: str = "",
) -> str:
    """Cache identity must cover every input that changes the assembled view.

    This includes the custom symbol list (a different list must never reuse
    another pool's cached bundle), the content hash of the PIT membership
    reference file (membership edits must invalidate csi800/all_a caches even
    when the price snapshot id is unchanged), and the secondary (ETF) snapshot
    id for mixed custom pools.
    """
    identity = f"{pool}|{snapshot_id}|{start or ''}|{end or ''}"
    if custom_symbols:
        ordered = sorted(str(symbol).strip().upper() for symbol in custom_symbols)
        identity += "|symbols:" + ",".join(ordered)
    if membership_sha256:
        identity += "|members:" + membership_sha256
    if secondary_snapshot_id:
        identity += "|secondary:" + secondary_snapshot_id
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _etf_code_to_symbol(etf_snapshot: dict[str, Any]) -> dict[str, str]:
    """Map bare ETF codes (159915) and SH/SZ forms to canonical ETF symbols."""
    mapping: dict[str, str] = {}
    universe_path = etf_snapshot["bars_dir"].parent / "ETF_UNIVERSE.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(universe_path, columns=["symbol", "code"])
        for row in universe.itertuples(index=False):
            code = str(row.code).strip().upper()
            symbol = str(row.symbol).strip().upper()
            mapping[code] = symbol
            mapping[symbol] = symbol
    return mapping


def load_market_view(
    pool: str,
    *,
    custom_symbols: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    quant_data_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[MarketView, dict[str, Any]]:
    """Assemble a PIT market view for a universe pool.

    Returns ``(view, meta)`` where ``meta`` carries the quant_data snapshot
    identity for provenance reporting.
    """
    if pool not in VALID_POOLS:
        raise ValueError(f"unknown pool: {pool}")
    if pool == "custom":
        if not custom_symbols:
            raise ValueError("custom pool requires custom_symbols")
    if quant_data_root is None:
        from src.services.quant_workflow_paths import quant_data_root as resolve_root

        quant_data_root = resolve_root()
    quant_data_root = Path(quant_data_root)
    if cache_dir is None:
        cache_dir = reports_root() / "cache" / "universe"

    family = "etf_1d" if pool == "etf" else "ashare_1d"
    dataset_id = _ETF_DATASET_ID if pool == "etf" else _ASHARE_DATASET_ID
    snapshot = _read_snapshot_latest(quant_data_root, family, dataset_id)
    snapshot_id = snapshot["snapshot_id"]
    if pool == "etf":
        # paired raw+qfq layout: bars live under the qfq basis subdirectory.
        snapshot["bars_dir"] = snapshot["bars_dir"].parent / "qfq"

    # Resolve the PIT membership reference before the cache key so membership
    # edits invalidate the cache even when the price snapshot is unchanged.
    membership_sha256 = ""
    membership_path: Optional[Path] = None
    if pool == "csi800":
        membership_path = (
            quant_data_root / "reference" / "pit_csi800" / "membership_intervals.parquet"
        )
    elif pool == "all_a":
        reference_dir = quant_data_root / "reference" / "snapshots"
        membership_path = next(reference_dir.glob("*/security_master.parquet"), None)
    if membership_path is not None and membership_path.is_file():
        membership_sha256 = _file_sha256(membership_path)

    # Mixed custom pools read the ETF snapshot too: resolve it before the
    # cache key so an ETF republish invalidates cached mixed bundles even
    # when the A-share snapshot is unchanged.
    etf_snapshot: Optional[dict[str, Any]] = None
    if pool == "custom":
        etf_snapshot = _read_snapshot_latest(quant_data_root, "etf_1d", _ETF_DATASET_ID)

    cache_key = _pool_cache_key(
        pool,
        snapshot_id,
        start,
        end,
        custom_symbols=custom_symbols if pool == "custom" else None,
        membership_sha256=membership_sha256,
        secondary_snapshot_id=(
            etf_snapshot["snapshot_id"] if etf_snapshot is not None else ""
        ),
    )
    cache_bundle_dir = cache_dir / f"{dataset_id}_{pool}_{cache_key}"

    if use_cache and (cache_bundle_dir / "market.npz").exists():
        try:
            meta_path = cache_bundle_dir / "universe.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")).get("meta", {})
            if meta.get("snapshot_id") == snapshot_id:
                logger.info("universe cache hit for %s/%s", pool, snapshot_id)
                meta.setdefault("content_sha256", snapshot["content_sha256"])
                meta.setdefault("latest_trade_date", snapshot["latest_trade_date"])
                meta.setdefault("membership_sha256", membership_sha256)
                if etf_snapshot is not None:
                    meta.setdefault(
                        "etf_latest_trade_date", etf_snapshot["latest_trade_date"]
                    )
                meta["bundle_dir"] = str(cache_bundle_dir)
                return MarketView.load(cache_bundle_dir), meta
        except Exception:
            logger.warning("universe cache read failed, rebuilding", exc_info=True)

    etf_wanted: list[str] = []
    if pool == "csi800":
        intervals = pd.read_parquet(membership_path)
        member_symbols = sorted(
            {str(symbol).strip().upper() for symbol in intervals["symbol"]}
        )
        bars = _load_bars_long(
            snapshot["bars_dir"], start=start, end=end, symbols=member_symbols
        )
    elif pool == "custom":
        # A custom pool may mix A-shares and ETFs (e.g. 600519, 159915):
        # A-share codes resolve in the ashare snapshot, ETF codes in the
        # cn_etf_daily snapshot, and the two are merged on the union calendar.
        raw_wanted = [str(symbol).strip().upper() for symbol in custom_symbols if str(symbol).strip()]
        ashare_wanted: list[str] = []
        unknown: list[str] = []
        etf_bars_dir = etf_snapshot["bars_dir"].parent / "qfq"
        code_to_symbol = _etf_code_to_symbol(etf_snapshot)
        for raw in raw_wanted:
            normalized = _normalize_ashare_symbol(raw)
            if (snapshot["bars_dir"] / f"{normalized}.parquet").exists():
                ashare_wanted.append(normalized)
                continue
            etf_symbol = code_to_symbol.get(raw) or code_to_symbol.get(normalized)
            if etf_symbol and (etf_bars_dir / f"{etf_symbol}.parquet").exists():
                etf_wanted.append(etf_symbol)
                continue
            unknown.append(raw)
        if unknown:
            # Silently shrinking the user's requested pool is misleading:
            # an invalid code is an input error, not a smaller universe.
            raise ValueError(
                "custom pool symbols not found in any dataset: "
                + ", ".join(unknown)
            )
        frames = []
        if ashare_wanted:
            frames.append(
                _load_bars_long(
                    snapshot["bars_dir"], start=start, end=end, symbols=ashare_wanted
                )
            )
        if etf_wanted:
            frames.append(
                _load_bars_long(etf_bars_dir, start=start, end=end, symbols=etf_wanted)
            )
        bars = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if bars.empty:
            raise ValueError(
                "none of the custom symbols exist in the canonical A-share or ETF datasets"
            )
    else:
        bars = _load_bars_long(snapshot["bars_dir"], start=start, end=end)
    dates, symbols, open_matrix, close_matrix = _build_wide_matrices(bars)

    if pool == "csi800":
        membership_path = quant_data_root / "reference" / "pit_csi800" / "membership_intervals.parquet"
        intervals = pd.read_parquet(membership_path)
        members = _interval_membership(
            intervals,
            symbol_col="symbol",
            start_col="start_date",
            end_col="end_date",
            dates=dates,
            symbols=symbols,
        )
    elif pool == "all_a":
        if membership_path is None:
            raise ValueError("security_master reference snapshot is unavailable")
        master = pd.read_parquet(membership_path)
        members = _interval_membership(
            master,
            symbol_col="symbol",
            start_col="list_date",
            end_col="delist_date",
            dates=dates,
            symbols=symbols,
        )
    else:
        members = ~np.isnan(close_matrix)

    dataset_ids = {dataset_id}
    if etf_wanted:
        dataset_ids.add(_ETF_DATASET_ID)
    meta = {
        "pool": pool,
        "dataset_id": "+".join(sorted(dataset_ids)),
        "snapshot_id": snapshot_id,
        "manifest_sha256": snapshot["manifest_sha256"],
        "content_sha256": snapshot["content_sha256"],
        "latest_trade_date": snapshot["latest_trade_date"],
        "membership_sha256": membership_sha256,
        "symbol_count": int(len(symbols)),
        "trading_days": int(len(dates)),
        "start": start,
        "end": end,
    }
    if etf_wanted:
        meta["etf_snapshot_id"] = etf_snapshot["snapshot_id"]
        meta["etf_manifest_sha256"] = etf_snapshot["manifest_sha256"]
        meta["etf_content_sha256"] = etf_snapshot["content_sha256"]
        meta["etf_latest_trade_date"] = etf_snapshot["latest_trade_date"]
    if pool == "custom":
        meta["custom_symbols"] = [str(symbol).strip().upper() for symbol in custom_symbols]
    bundle = MarketBundle(
        dates=dates,
        symbols=symbols,
        open=open_matrix,
        close=close_matrix,
        members=members,
        meta=meta,
    )
    # The bundle directory is also the sandbox input, so it is always
    # materialised on disk (the cache is the transport).
    try:
        bundle.save(cache_bundle_dir)
    except Exception:
        logger.warning("universe cache write failed", exc_info=True)
    meta["bundle_dir"] = str(cache_bundle_dir)
    return MarketView.from_bundle(bundle), meta


def load_view_for_run(
    bundle_dir: Path,
) -> MarketView:
    """Load a previously saved bundle directory (sandbox child process)."""
    return MarketView.load(bundle_dir)


INDEX_BENCHMARK_LABELS = {
    "SH000300": "沪深300",
    "SH000905": "中证500",
    "SH000906": "中证800",
    "SZ399006": "创业板指",
    "SH000688": "科创50",
}

INDEX_BENCHMARK_SYMBOLS = tuple(INDEX_BENCHMARK_LABELS)


def _read_benchmark_file(symbol: str, bars_dir: Path) -> np.ndarray:
    frame = pd.read_parquet(bars_dir / f"{symbol}.parquet", columns=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def load_benchmark_series(
    symbols: Sequence[str],
    *,
    quant_data_root: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """Load close series for comparison benchmarks.

    Resolution order per requested code:
    1. index codes (SH000300 etc.) from the cn_index_daily snapshot;
    2. A-share codes from the ashare_daily_qfq snapshot (bare codes are
       normalized to SH/SZ form);
    3. ETF codes (5-digit, e.g. 159915/510300) from the cn_etf_daily
       snapshot universe.
    """
    if quant_data_root is None:
        from src.services.quant_workflow_paths import quant_data_root as resolve_root

        quant_data_root = resolve_root()
    quant_data_root = Path(quant_data_root)
    wanted = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if not wanted:
        return {}

    index_snapshot = _read_snapshot_latest(quant_data_root, "index_1d", "cn_index_daily")
    ashare_snapshot = _read_snapshot_latest(quant_data_root, "ashare_1d", _ASHARE_DATASET_ID)
    etf_snapshot = _read_snapshot_latest(quant_data_root, "etf_1d", _ETF_DATASET_ID)
    etf_bars_dir = etf_snapshot["bars_dir"].parent / "qfq"

    etf_code_to_symbol = _etf_code_to_symbol(etf_snapshot)

    result: dict[str, pd.DataFrame] = {}
    for symbol in sorted(wanted):
        resolved: Optional[tuple[str, Path]] = None
        if symbol in INDEX_BENCHMARK_LABELS:
            resolved = (symbol, index_snapshot["bars_dir"])
        else:
            normalized = _normalize_ashare_symbol(symbol)
            ashare_path = ashare_snapshot["bars_dir"] / f"{normalized}.parquet"
            if ashare_path.exists():
                resolved = (normalized, ashare_snapshot["bars_dir"])
            else:
                etf_symbol = etf_code_to_symbol.get(symbol) or etf_code_to_symbol.get(normalized)
                if etf_symbol:
                    etf_path = etf_bars_dir / f"{etf_symbol}.parquet"
                    if etf_path.exists():
                        resolved = (etf_symbol, etf_bars_dir)
        if resolved is None:
            continue
        resolved_symbol, bars_dir = resolved
        path = bars_dir / f"{resolved_symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        result[resolved_symbol] = frame
    return result
