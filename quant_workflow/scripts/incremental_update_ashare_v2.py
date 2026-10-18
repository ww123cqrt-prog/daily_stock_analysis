#!/usr/bin/env python3
"""Build staged A-share raw/qfq snapshots from a bounded incremental bundle."""

from __future__ import annotations

import json
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_baostock_v2 import (  # noqa: E402
    BaostockV2Error,
    FACTOR_COLUMNS,
    PRICE_COLUMNS,
    atomic_parquet,
    build_local_qfq,
    expand_daily_factors,
    resolve_bundle_file,
    sha256_file,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_price_snapshot,
    resolve_dataset,
)


RAW_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_raw")
QFQ_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_qfq")
_EVENT_VALUE_COLUMNS = (
    "foreAdjustFactor",
    "backAdjustFactor",
    "adjustFactor",
)
_PROVIDER_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


@dataclass(frozen=True)
class IncrementalPriceResult:
    raw_stage: Path
    qfq_stage: Path
    base_latest_trade_date: str
    latest_trade_date: str
    downloaded_rows: int
    reused_symbols: tuple[str, ...]
    factor_changed_symbols: tuple[str, ...]
    qfq_recalculated_symbols: tuple[str, ...]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return value


def _bundle_file(bundle_dir: Path, relative: object, *, label: str) -> Path:
    try:
        return resolve_bundle_file(bundle_dir, relative, label=label)
    except BaostockV2Error as exc:
        raise DataContractError(str(exc)) from exc


def _validate_bundle(bundle_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = _bundle_file(
        bundle_dir,
        "BUNDLE_MANIFEST.json",
        label="incremental bundle manifest",
    )
    manifest = _read_json(manifest_path, label="incremental bundle manifest")
    if manifest.get("status") != "ready":
        raise DataContractError("incremental bundle is not ready")
    if manifest.get("bundle_mode") != "incremental":
        raise DataContractError("incremental update requires an incremental bundle")
    bundle_kind = str(manifest.get("bundle_kind") or "")
    provider_paired = bundle_kind == "provider_paired_raw_qfq"
    if not provider_paired and manifest.get("factor_requested_start") != "1990-01-01":
        raise DataContractError("incremental bundle does not contain full factor history")
    if provider_paired:
        source = manifest.get("source")
        provider = source.get("provider") if isinstance(source, dict) else None
        if not isinstance(provider, str) or not _PROVIDER_ID_RE.fullmatch(provider):
            raise DataContractError("provider-paired bundle source is incomplete")
        snapshot_id = manifest.get("base_qfq_snapshot_id")
        manifest_sha256 = str(manifest.get("base_qfq_manifest_sha256") or "")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise DataContractError("provider-paired base qfq identity is missing")
        if len(manifest_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in manifest_sha256
        ):
            raise DataContractError("provider-paired base qfq identity is invalid")
    try:
        requested_start = pd.Timestamp(manifest["requested_start"]).normalize()
        requested_through = pd.Timestamp(manifest["requested_through"]).normalize()
        base_latest = pd.Timestamp(manifest["base_latest_trade_date"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("incremental bundle date contract is invalid") from exc
    if not requested_start <= base_latest < requested_through:
        raise DataContractError("incremental bundle does not overlap and extend the base")
    if (base_latest - requested_start).days > 62:
        raise DataContractError("incremental bundle overlap exceeds 62 calendar days")
    index_path = _bundle_file(
        bundle_dir,
        manifest.get("index_path"),
        label="incremental bundle index",
    )
    try:
        index = pd.read_parquet(index_path)
    except (OSError, ValueError) as exc:
        raise DataContractError("incremental bundle index is unavailable") from exc
    required = {
        "symbol",
        "raw_path",
        "raw_sha256",
        "raw_rows",
    }
    if provider_paired:
        required.update({"qfq_provider_path", "qfq_provider_sha256"})
    else:
        required.update({"factor_events_path", "factor_events_sha256"})
    if required.difference(index.columns):
        raise DataContractError("incremental bundle index is incomplete")
    if index.empty or index["symbol"].duplicated().any():
        raise DataContractError("incremental bundle symbol index is invalid")
    if len(index) != int(manifest.get("completed_symbols", -1)):
        raise DataContractError("incremental bundle index row count is invalid")
    for row in index.itertuples(index=False):
        symbol = str(row.symbol)
        _bundle_file(
            bundle_dir,
            row.raw_path,
            label=f"incremental raw file for {symbol}",
        )
        if provider_paired:
            qfq_path = getattr(row, "qfq_provider_path", None)
            qfq_hash = getattr(row, "qfq_provider_sha256", None)
            if not isinstance(qfq_path, str) or not qfq_path:
                raise DataContractError(
                    f"qfq audit file is required for provider-paired symbol {symbol}"
                )
            if not isinstance(qfq_hash, str) or len(qfq_hash) != 64:
                raise DataContractError(
                    f"qfq audit hash is required for provider-paired symbol {symbol}"
                )
            _bundle_file(
                bundle_dir,
                qfq_path,
                label=f"incremental qfq audit file for {symbol}",
            )
        else:
            _bundle_file(
                bundle_dir,
                row.factor_events_path,
                label=f"incremental factor file for {symbol}",
            )
    return manifest, index.sort_values("symbol", kind="stable").reset_index(drop=True)


def _normalize_bars(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {"date", *PRICE_COLUMNS, "volume", "symbol", "source_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"bar schema is incomplete for {symbol}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    if normalized["date"].isna().any():
        raise DataContractError(f"bar dates are invalid for {symbol}")
    if normalized["date"].duplicated().any():
        raise DataContractError(f"bar dates are duplicated for {symbol}")
    if set(normalized["symbol"].astype(str)) != {symbol}:
        raise DataContractError(f"bar symbol does not match index for {symbol}")
    return normalized.sort_values("date", kind="stable").reset_index(drop=True)


def _merge_by_date(base: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    if update.empty:
        return base.copy().reset_index(drop=True)
    columns = list(dict.fromkeys([*base.columns, *update.columns]))
    merged = pd.concat(
        [base.reindex(columns=columns), update.reindex(columns=columns)],
        ignore_index=True,
    )
    return (
        merged.sort_values("date", kind="stable")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def restore_tickflow_factor_metadata(
    raw: pd.DataFrame,
    qfq: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Restore omitted cells in a frozen TickFlow multiplicative factor table.

    This explicit offline repair is not part of reading or ordinary updating.
    The caller must bind all three files to their frozen manifest/index. Only
    missing cells are candidates; existing prices and known metadata stay intact.
    """
    restored = factors.copy(deep=True)
    missing_symbol = (
        restored["symbol"].isna() if "symbol" in restored
        else pd.Series(False, index=restored.index)
    )
    missing_offset = (
        restored["offset"].isna() if "offset" in restored
        else pd.Series(False, index=restored.index)
    )
    affected = missing_symbol | missing_offset
    if not affected.any():
        return restored
    try:
        dates = pd.to_datetime(restored["date"], errors="raise").reset_index(drop=True)
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("factor dates are invalid")
        for frame in (raw, qfq, restored):
            actual_dates = pd.to_datetime(frame["date"], errors="raise").reset_index(drop=True)
            if not actual_dates.equals(dates):
                raise ValueError("raw/QFQ/factor keys differ")
            if "symbol" in frame and not frame["symbol"].dropna().eq(symbol).all():
                raise ValueError("known symbol differs from frozen index")
        if not restored.loc[affected, "factor_source"].eq("tickflow_adj_factor").all():
            raise ValueError("missing metadata is not a declared TickFlow factor")
        scale = pd.to_numeric(restored.loc[affected, "factor"], errors="raise").to_numpy(float)
        offsets = (
            pd.to_numeric(restored["offset"], errors="raise").fillna(0.0)
            if "offset" in restored else pd.Series(0.0, index=restored.index)
        ).loc[affected].to_numpy(float)
        if not np.isfinite(scale).all() or (scale <= 0).any() or not np.isfinite(offsets).all():
            raise ValueError("factor or offset is invalid")
        positions = np.flatnonzero(affected.to_numpy())
        for field in PRICE_COLUMNS:
            left = pd.to_numeric(raw.iloc[positions][field], errors="raise").to_numpy(float)
            right = pd.to_numeric(qfq.iloc[positions][field], errors="raise").to_numpy(float)
            if (
                not np.isfinite(left).all() or not np.isfinite(right).all()
                or (left <= 0).any() or (right <= 0).any()
                or not np.isclose(left * scale + offsets, right, rtol=1e-12, atol=1e-12).all()
            ):
                raise ValueError("saved OHLC do not prove the proposed metadata")
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"cannot restore factor metadata for {symbol}: {exc}") from exc
    if "symbol" in restored:
        restored.loc[missing_symbol, "symbol"] = symbol
    if "offset" in restored:
        restored.loc[missing_offset, "offset"] = 0.0
    return restored


def _canonical_events(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    events = frame.copy()
    if "action_type" in events.columns:
        events = events.loc[events["action_type"] == "adjustment_factor_event"]
    if "symbol" in events.columns:
        events = events.loc[events["symbol"].astype(str) == symbol]
    date_column = "event_date" if "event_date" in events.columns else "dividOperateDate"
    if date_column not in events.columns:
        if events.empty:
            return pd.DataFrame(columns=["event_date", *_EVENT_VALUE_COLUMNS])
        raise DataContractError(f"factor event dates are missing for {symbol}")
    missing = set(_EVENT_VALUE_COLUMNS).difference(events.columns)
    if missing:
        raise DataContractError(f"factor event values are missing for {symbol}")
    normalized = pd.DataFrame(
        {
            "event_date": pd.to_datetime(
                events[date_column], errors="coerce"
            )
            .dt.normalize()
            .astype("datetime64[ns]")
        }
    )
    for column in _EVENT_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(events[column], errors="coerce")
    if normalized.isna().any().any() or (normalized["foreAdjustFactor"] <= 0).any():
        raise DataContractError(f"factor event history is invalid for {symbol}")
    return (
        normalized.sort_values("event_date", kind="stable")
        .drop_duplicates("event_date", keep="last")
        .reset_index(drop=True)
    )


def _events_changed(
    old_actions: pd.DataFrame,
    new_events: pd.DataFrame,
    *,
    symbol: str,
) -> bool:
    old = _canonical_events(old_actions, symbol=symbol)
    new = _canonical_events(new_events, symbol=symbol)
    if len(old) != len(new):
        return True
    if old.empty:
        return False
    if not old["event_date"].equals(new["event_date"]):
        return True
    return not bool(
        np.isclose(
            old[list(_EVENT_VALUE_COLUMNS)].to_numpy(dtype=float),
            new[list(_EVENT_VALUE_COLUMNS)].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ).all()
    )


def _event_output(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    output = frame.copy()
    if "dividOperateDate" in output.columns:
        output = output.rename(columns={"dividOperateDate": "event_date"})
    output["event_date"] = pd.to_datetime(output["event_date"], errors="raise").dt.normalize()
    output["symbol"] = symbol
    output["action_type"] = "adjustment_factor_event"
    return output


def _identity_extension(
    base_factors: pd.DataFrame,
    update_bars: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    if update_bars.empty:
        return base_factors.copy()
    latest = float(pd.to_numeric(base_factors["factor"], errors="coerce").iloc[-1])
    if not np.isclose(latest, 1.0, atol=1e-12, rtol=1e-12):
        raise DataContractError(f"empty factor history cannot extend {symbol}")
    extension = pd.DataFrame({"date": update_bars["date"].copy()})
    extension["source_event_date"] = pd.NaT
    extension["factor"] = 1.0
    extension["factor_source"] = "baostock_no_adjustment_events"
    extension["factor_backfilled_before_first_event"] = False
    extension["factor_paired_fallback"] = False
    extension["factor_terminal_normalized"] = False
    extension["provider_qfq_terminal_factor"] = np.nan
    if "symbol" in base_factors:
        extension["symbol"] = symbol
    if "offset" in base_factors:
        extension["offset"] = 0.0
    return _merge_by_date(base_factors, extension)


def _provider_alignment_scale(
    base_qfq: pd.DataFrame,
    provider_qfq: pd.DataFrame,
    *,
    symbol: str,
) -> tuple[float, float]:
    left = base_qfq[["date", *PRICE_COLUMNS]].copy()
    right = provider_qfq[["date", *PRICE_COLUMNS]].copy()
    for frame in (left, right):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for column in PRICE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    merged = left.merge(right, on="date", suffixes=("_base", "_provider"))
    if len(merged) != len(left) or len(merged) != len(right) or merged.empty:
        raise DataContractError(
            f"provider-paired qfq dates do not match the base overlap for {symbol}"
        )
    ratios: list[float] = []
    for column in PRICE_COLUMNS:
        base_values = merged[f"{column}_base"].to_numpy(dtype=float)
        provider_values = merged[f"{column}_provider"].to_numpy(dtype=float)
        if (
            not np.isfinite(base_values).all()
            or not np.isfinite(provider_values).all()
            or (base_values <= 0).any()
            or (provider_values <= 0).any()
        ):
            raise DataContractError(
                f"provider-paired qfq prices are invalid for {symbol}"
            )
        ratios.extend((base_values / provider_values).tolist())
    scale = float(np.median(np.asarray(ratios, dtype=float)))
    # EastMoney A-share prices are cent-rounded. Scaling a provider series can
    # magnify that half-cent source rounding, so validate in price space.
    absolute_tolerance = 0.011 * max(1.0, abs(scale))
    for column in PRICE_COLUMNS:
        errors = (
            merged[f"{column}_base"]
            - merged[f"{column}_provider"] * scale
        ).abs()
        if (errors > absolute_tolerance).any():
            raise DataContractError(
                f"provider-paired qfq scale is inconsistent for {symbol}"
            )
    return scale, absolute_tolerance


def _validate_provider_pair_relation(
    raw_bars: pd.DataFrame,
    provider_qfq: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    absolute_tolerance: float,
    symbol: str,
) -> None:
    raw = raw_bars[["date", *PRICE_COLUMNS]].copy()
    qfq = provider_qfq[["date", *PRICE_COLUMNS]].copy()
    factor_frame = factors[["date", "factor"]].copy()
    for frame in (raw, qfq, factor_frame):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    merged = raw.merge(qfq, on="date", suffixes=("_raw", "_qfq"), validate="one_to_one")
    merged = merged.merge(factor_frame, on="date", how="inner", validate="one_to_one")
    if len(merged) != len(raw):
        raise DataContractError(
            f"provider-paired OHLC reconstruction is incomplete for {symbol}"
        )
    factor_values = pd.to_numeric(merged["factor"], errors="coerce")
    for column in PRICE_COLUMNS:
        expected = (
            pd.to_numeric(merged[f"{column}_raw"], errors="coerce")
            * factor_values
        )
        actual = pd.to_numeric(merged[f"{column}_qfq"], errors="coerce")
        if (expected - actual).abs().gt(absolute_tolerance).any():
            raise DataContractError(
                f"provider-paired OHLC reconstruction failed for {symbol}"
            )


def _provider_pair_update(
    *,
    base_raw: pd.DataFrame,
    base_qfq: pd.DataFrame,
    base_factors: pd.DataFrame,
    raw_update: pd.DataFrame,
    provider_qfq: pd.DataFrame,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align a bounded provider raw/qfq pair to the frozen qfq scale.

    The overlap with the base snapshot is the evidence that the provider's
    current qfq scale is compatible with the frozen history.  A changing
    factor inside the overlap is rejected instead of silently stitching two
    incompatible histories together.
    """

    raw_dates = set(pd.to_datetime(raw_update["date"]).dt.normalize())
    qfq_dates = set(pd.to_datetime(provider_qfq["date"]).dt.normalize())
    if raw_dates != qfq_dates:
        raise DataContractError(f"paired raw/qfq dates do not match for {symbol}")
    if not raw_dates:
        return base_qfq.copy(), base_factors.copy(), raw_update.copy()

    base_qfq_dates = pd.to_datetime(base_qfq["date"]).dt.normalize()
    overlap_dates = raw_dates.intersection(set(base_qfq_dates))
    if not overlap_dates:
        raise DataContractError(
            f"provider-paired update has no overlap with base qfq for {symbol}"
        )
    overlap = pd.Series(sorted(overlap_dates), name="date")
    base_overlap = base_qfq.loc[base_qfq_dates.isin(overlap)].copy()
    provider_dates = pd.to_datetime(provider_qfq["date"]).dt.normalize()
    provider_overlap = provider_qfq.loc[provider_dates.isin(overlap)].copy()
    scale, alignment_tolerance = _provider_alignment_scale(
        base_overlap,
        provider_overlap,
        symbol=symbol,
    )
    aligned_qfq = provider_qfq.copy()
    for column in PRICE_COLUMNS:
        aligned_qfq[column] = (
            pd.to_numeric(aligned_qfq[column], errors="coerce") * scale
        ).round(4)
    if "preclose" in aligned_qfq.columns:
        aligned_qfq["preclose"] = (
            pd.to_numeric(aligned_qfq["preclose"], errors="coerce") * scale
        ).round(4)
    if "source_id" in aligned_qfq.columns:
        aligned_qfq["source_id"] = aligned_qfq["source_id"].astype(str).map(
            lambda value: f"{value}_aligned_to_base_qfq"
        )
    else:
        aligned_qfq["source_id"] = "provider_qfq_aligned_to_base_qfq"

    from download_baostock_v2 import derive_paired_daily_factors

    try:
        provider_native_factors = derive_paired_daily_factors(
            raw_update,
            provider_qfq,
            ratio_tolerance=0.02,
        )
        paired_factors = derive_paired_daily_factors(
            raw_update,
            aligned_qfq,
            ratio_tolerance=0.02,
        )
    except Exception as exc:
        raise DataContractError(
            f"provider-paired OHLC adjustment ratios disagree for {symbol}"
        ) from exc
    provider_terminal_factor = float(
        provider_native_factors.sort_values("date")["factor"].iloc[-1]
    )
    _validate_provider_pair_relation(
        raw_update,
        provider_qfq,
        provider_native_factors,
        absolute_tolerance=0.011,
        symbol=symbol,
    )
    _validate_provider_pair_relation(
        raw_update,
        aligned_qfq,
        paired_factors,
        absolute_tolerance=alignment_tolerance,
        symbol=symbol,
    )
    minimum_raw_price = float(
        pd.to_numeric(
            raw_update[list(PRICE_COLUMNS)].stack(), errors="coerce"
        ).min()
    )
    if not np.isfinite(minimum_raw_price) or minimum_raw_price <= 0:
        raise DataContractError(f"provider-paired raw prices are invalid for {symbol}")
    factor_tolerance = max(1e-4, alignment_tolerance / minimum_raw_price)
    base_raw_dates = pd.to_datetime(base_raw["date"]).dt.normalize()
    base_qfq_dates = pd.to_datetime(base_qfq["date"]).dt.normalize()
    base_pair_dates = sorted(set(base_raw_dates).intersection(set(base_qfq_dates)))
    base_pair_raw = base_raw.loc[base_raw_dates.isin(base_pair_dates)]
    base_pair_qfq = base_qfq.loc[base_qfq_dates.isin(base_pair_dates)]
    base_paired_factors = derive_paired_daily_factors(base_pair_raw, base_pair_qfq)
    base_factor_map = base_paired_factors.set_index("date")
    overlap_factors = paired_factors.loc[
        paired_factors["date"].isin(overlap_dates)
    ]
    if not overlap_factors.empty:
        expected = overlap_factors["date"].map(base_factor_map["factor"])
        if expected.isna().any() or not np.isclose(
            overlap_factors["factor"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=factor_tolerance,
        ).all():
            raise DataContractError(
                f"provider-paired factor continuity failed for {symbol}"
            )
    latest_base_factor = float(base_paired_factors.sort_values("date")["factor"].iloc[-1])
    future_factors = paired_factors.loc[
        paired_factors["date"] > max(overlap_dates)
    ]
    if not future_factors.empty and not np.isclose(
        future_factors["factor"].to_numpy(dtype=float),
        latest_base_factor,
        rtol=0.0,
        atol=factor_tolerance,
    ).all():
        raise DataContractError(
            f"provider-paired factor changed for {symbol}; targeted full-history qfq is required"
        )
    update_factors = paired_factors.copy()
    update_factors["source_event_date"] = pd.NaT
    update_factors["factor_source"] = "provider_native_paired"
    update_factors["factor_backfilled_before_first_event"] = False
    update_factors["factor_paired_fallback"] = True
    update_factors["factor_terminal_normalized"] = False
    update_factors["provider_qfq_terminal_factor"] = provider_terminal_factor
    update_factors["provider_qfq_alignment_scale"] = scale
    merged_factors = _merge_by_date(base_factors, update_factors)
    merged_qfq = _merge_by_date(base_qfq, aligned_qfq)
    return merged_qfq, merged_factors, raw_update


def merge_miniqmt_ratio_increment(
    *, base_raw: pd.DataFrame, base_qfq: pd.DataFrame,
    base_factors: pd.DataFrame, raw_update: pd.DataFrame,
    ratio_qfq: pd.DataFrame, actions: list[dict], symbol: str,
    repair_event_dates: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extend a proven multiplicative stock history using SDK ``front_ratio``.

    Cash ``front`` and ratio-adjusted history are different price bases. Never
    subtract a new cash dividend from the old ratio-adjusted prices. A newly
    effective event supplies ``dr``; the SDK ratio series independently checks
    its use on the bounded window. Earlier unexplained revisions are rejected.
    An explicitly selected omitted event additionally needs an unchanged old
    factor boundary and a matching SDK overlap after applying that event.
    The caller verifies receipt identity, calendar and suspension evidence.
    """
    frames = [base_raw.copy(), base_qfq.copy(), base_factors.copy(),
              raw_update.copy(), ratio_qfq.copy()]
    for frame in frames:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        if frame.empty or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
            raise DataContractError(f"invalid MiniQMT ratio dates for {symbol}")
        if "symbol" in frame and not frame.symbol.eq(symbol).all():
            raise DataContractError(f"different MiniQMT ratio symbol for {symbol}")
    old_raw, old_qfq, factors, raw, observed = frames
    if (old_raw.date.tolist() != old_qfq.date.tolist()
            or old_raw.date.tolist() != factors.date.tolist()
            or raw.date.tolist() != observed.date.tolist()):
        raise DataContractError(f"MiniQMT ratio keys differ for {symbol}")
    offsets = pd.to_numeric(factors.get("offset", pd.Series(0., index=factors.index)), errors="coerce")
    if offsets.isna().any() or offsets.ne(0).any():
        raise DataContractError(f"existing history is not a proven ratio basis for {symbol}")
    factor = pd.to_numeric(factors.factor, errors="coerce").to_numpy(float)
    old_prices = old_raw[list(PRICE_COLUMNS)].to_numpy(float)
    qfq_prices = old_qfq[list(PRICE_COLUMNS)].to_numpy(float)
    reconstructed = old_prices * factor[:, None]
    exact_history = np.allclose(reconstructed, qfq_prices, rtol=1e-9, atol=1e-8)
    # Some frozen TickFlow snapshots stored prices rounded to four decimals.
    # Prove that exact representation, rather than admitting arbitrary errors.
    rounded_history = np.allclose(np.round(reconstructed, 4), qfq_prices, rtol=0, atol=1e-8)
    if (not np.isfinite(factor).all() or (factor <= 0).any()
            or not np.isfinite(old_prices).all() or not np.isfinite(qfq_prices).all()
            or (old_prices <= 0).any() or (qfq_prices <= 0).any()
            or not (exact_history or rounded_history)):
        raise DataContractError(f"old raw/factor/QFQ relation differs for {symbol}")
    end = old_raw.date.max()
    applicable = []
    for action in actions:
        day = pd.Timestamp(action["date"]).normalize()
        if day <= raw.date.min() or day > raw.date.max():
            continue
        dr = float(action["dr"])
        if not np.isfinite(dr) or dr <= 0:
            raise DataContractError(f"invalid MiniQMT ratio event for {symbol}")
        if day <= raw.date.max():
            applicable.append((day, dr))
    if len({day for day, _ in applicable}) != len(applicable):
        raise DataContractError(f"duplicate MiniQMT ratio event for {symbol}")
    scales = np.ones(len(raw))
    history_scales = np.ones(len(old_raw))
    repairs = {pd.Timestamp(day).normalize() for day in repair_event_dates}
    if len(repairs) != len(repair_event_dates) or not repairs.issubset({day for day, _ in applicable}):
        raise DataContractError(f"unproven ratio repair event for {symbol}")
    for day, dr in sorted(applicable):
        scales[np.asarray(raw.date < day)] /= dr
        if day > end:
            history_scales /= dr
        elif day in repairs:
            before = factors.loc[factors.date < day, "factor"]
            after = factors.loc[factors.date >= day, "factor"]
            if (before.empty or after.empty or day not in set(raw.date)
                    or not np.isclose(float(before.iloc[-1]), float(after.iloc[0]), rtol=1e-10, atol=1e-12)):
                raise DataContractError(f"ratio event is not omitted at frozen boundary for {symbol}")
            history_scales[np.asarray(old_raw.date < day)] /= dr
    native_expected = raw[list(PRICE_COLUMNS)].to_numpy(float) * scales[:, None]
    if not np.allclose(native_expected, observed[list(PRICE_COLUMNS)].to_numpy(float), rtol=1e-9, atol=1e-8):
        raise DataContractError(f"MiniQMT raw/events/front_ratio relation differs for {symbol}")
    overlap = old_raw.merge(raw, on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        raise DataContractError(f"MiniQMT ratio update has no overlap for {symbol}")
    for field in (*PRICE_COLUMNS, "volume"):
        if not np.allclose(overlap[field + "_old"], overlap[field + "_new"], rtol=0, atol=1e-8):
            raise DataContractError(f"MiniQMT raw overlap differs for {symbol}: {field}")
    anchor = float(factor[-1])
    aligned = observed.copy()
    for field in PRICE_COLUMNS:
        aligned[field] *= anchor
    check = old_qfq.merge(aligned, on="date", suffixes=("_old", "_new"))
    overlap_scales = check.date.map(dict(zip(old_raw.date, history_scales))).to_numpy(float)[:, None]
    overlap_tolerance = 1e-8 if exact_history else 0.00005 * overlap_scales + 1e-8
    if not np.allclose(check[[c + "_old" for c in PRICE_COLUMNS]].to_numpy(float) * overlap_scales,
                       check[[c + "_new" for c in PRICE_COLUMNS]].to_numpy(float), rtol=1e-9, atol=overlap_tolerance):
        raise DataContractError(f"unexplained historical ratio revision for {symbol}")
    historical = old_qfq.copy()
    for field in (*PRICE_COLUMNS, "preclose"):
        if field in historical:
            historical[field] *= history_scales
    if "factor" in historical:
        historical["factor"] *= history_scales
    factors["factor"] = factor * history_scales
    changed = history_scales != 1.
    if "source_id" in historical:
        historical.loc[changed, "source_id"] = historical.loc[changed, "source_id"].astype(str) + "|miniqmt-explicit-dr"
    if "factor_source" in factors:
        factors.loc[changed, "factor_source"] = factors.loc[changed, "factor_source"].astype(str) + "|miniqmt-explicit-dr"
    tail = np.asarray(raw.date > end)
    extension = pd.DataFrame({"date": raw.loc[tail, "date"], "symbol": symbol,
                              "factor": scales[tail] * anchor, "offset": 0.,
                              "factor_source": "miniqmt_explicit_dr_ratio",
                              "factor_terminal_normalized": anchor == 1.})
    aligned["factor"] = scales * anchor
    aligned["source_id"] = aligned.get("source_id", pd.Series("miniqmt", index=aligned.index)).astype(str) + "|front_ratio"
    out_raw = pd.concat([old_raw, raw.loc[tail]], ignore_index=True)
    out_qfq = pd.concat([historical, aligned.loc[tail]], ignore_index=True)
    for frame in (out_raw, out_qfq):
        begin = 1 if frame is out_qfq and repairs else len(old_raw)
        if "preclose" in frame:
            frame.loc[begin:, "preclose"] = frame.close.shift(1).iloc[begin:].to_numpy()
        if "pctChg" in frame:
            frame.loc[begin:, "pctChg"] = frame.close.pct_change(fill_method=None).iloc[begin:].to_numpy() * 100
    return out_raw, out_qfq, pd.concat([factors, extension], ignore_index=True)


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_incremental_price_snapshots(
    *,
    bundle_dir: Path,
    quant_data_root: Path,
) -> IncrementalPriceResult:
    """Create complete staged raw/qfq snapshots without moving formal pointers."""

    root = Path(quant_data_root).expanduser().resolve()
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest, index = _validate_bundle(bundle)
    provider_paired = str(manifest.get("bundle_kind") or "") == "provider_paired_raw_qfq"
    raw_base = resolve_dataset("ashare_daily_raw", quant_data_root=root, require_basis="raw")
    qfq_base = resolve_dataset("ashare_daily_qfq", quant_data_root=root, require_basis="qfq")
    base_latest = str(manifest["base_latest_trade_date"])
    if raw_base.manifest.get("latest_trade_date") != base_latest:
        raise DataContractError("incremental bundle does not match current raw snapshot")
    if qfq_base.manifest.get("latest_trade_date") != base_latest:
        raise DataContractError("incremental bundle does not match current qfq snapshot")
    if provider_paired:
        qfq_manifest_hash = sha256_file(
            qfq_base.snapshot_path / "DATASET_MANIFEST.json"
        )
        if (
            manifest.get("base_qfq_snapshot_id")
            != qfq_base.manifest.get("snapshot_id")
            or manifest.get("base_qfq_manifest_sha256") != qfq_manifest_hash
        ):
            raise DataContractError(
                "provider-paired bundle does not match current qfq snapshot identity"
            )
        raw_upstreams = [
            item
            for item in qfq_base.manifest.get("upstream_snapshots", [])
            if isinstance(item, dict) and item.get("dataset_id") == "ashare_daily_raw"
        ]
        if len(raw_upstreams) != 1 or raw_upstreams[0].get(
            "snapshot_id"
        ) != raw_base.manifest.get("snapshot_id"):
            raise DataContractError("current raw/qfq snapshot identities disagree")

    raw_symbols = {path.stem for path in (raw_base.snapshot_path / "bars").glob("*.parquet")}
    qfq_symbols = {path.stem for path in (qfq_base.snapshot_path / "bars").glob("*.parquet")}
    factor_symbols = {path.stem for path in (qfq_base.snapshot_path / "factors").glob("*.parquet")}
    bundle_symbols = set(index["symbol"].astype(str))
    if raw_symbols != qfq_symbols or raw_symbols != factor_symbols:
        raise DataContractError("current raw/qfq/factor symbol sets disagree")
    if bundle_symbols != raw_symbols:
        raise DataContractError("incremental bundle does not cover the current symbol set")

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = root / "staging" / f"ashare-incremental-v2-{run_id}"
    source_provider = (
        str((manifest.get("source") or {}).get("provider"))
        if provider_paired
        else "baostock"
    )
    raw_stage = run_root / f"ashare-{source_provider}-raw-{run_id}"
    qfq_stage = run_root / f"ashare-{source_provider}-qfq-{run_id}"
    actions_path = qfq_base.snapshot_path / "corporate_actions.parquet"
    try:
        old_actions = pd.read_parquet(actions_path)
    except (OSError, ValueError) as exc:
        raise DataContractError("current factor event evidence is unavailable") from exc

    reused: list[str] = []
    changed: list[str] = []
    actions: list[pd.DataFrame] = []
    downloaded_rows = 0
    latest_observed_trade_date: pd.Timestamp | None = None
    requested_start = pd.Timestamp(manifest["requested_start"]).normalize()
    requested_through = pd.Timestamp(manifest["requested_through"]).normalize()

    try:
        for row in index.itertuples(index=False):
            symbol = str(row.symbol)
            raw_update_path = _bundle_file(
                bundle,
                row.raw_path,
                label=f"incremental raw file for {symbol}",
            )
            events_path = None
            if not provider_paired:
                events_path = _bundle_file(
                    bundle,
                    row.factor_events_path,
                    label=f"incremental factor file for {symbol}",
                )
            if sha256_file(raw_update_path) != str(row.raw_sha256):
                raise DataContractError(f"incremental raw hash mismatch for {symbol}")
            if not provider_paired and sha256_file(events_path) != str(row.factor_events_sha256):
                raise DataContractError(f"incremental factor hash mismatch for {symbol}")
            try:
                raw_update = _normalize_bars(pd.read_parquet(raw_update_path), symbol=symbol)
                new_events = (
                    pd.read_parquet(events_path)
                    if events_path is not None
                    else pd.DataFrame()
                )
                base_raw = _normalize_bars(
                    pd.read_parquet(raw_base.snapshot_path / "bars" / f"{symbol}.parquet"),
                    symbol=symbol,
                )
                base_qfq = _normalize_bars(
                    pd.read_parquet(qfq_base.snapshot_path / "bars" / f"{symbol}.parquet"),
                    symbol=symbol,
                )
                base_factors = pd.read_parquet(
                    qfq_base.snapshot_path / "factors" / f"{symbol}.parquet"
                )
                provider_qfq = None
                provider_relative = getattr(row, "qfq_provider_path", None)
                if provider_paired or (isinstance(provider_relative, str) and provider_relative):
                    provider_path = _bundle_file(
                        bundle,
                        provider_relative,
                        label=f"incremental qfq audit file for {symbol}",
                    )
                    provider_hash = getattr(row, "qfq_provider_sha256", None)
                    if not isinstance(provider_hash, str) or not provider_hash:
                        raise DataContractError(
                            f"incremental qfq audit hash is missing for {symbol}"
                        )
                    if sha256_file(provider_path) != provider_hash:
                        raise DataContractError(
                            f"incremental qfq audit hash mismatch for {symbol}"
                        )
                    provider_qfq = _normalize_bars(
                        pd.read_parquet(provider_path),
                        symbol=symbol,
                    )
            except (OSError, ValueError) as exc:
                raise DataContractError(f"incremental inputs are unreadable for {symbol}") from exc
            downloaded_rows += len(raw_update)
            if len(raw_update) != int(row.raw_rows):
                raise DataContractError(f"incremental raw row count mismatch for {symbol}")
            if not raw_update.empty and (
                raw_update["date"].min() < requested_start
                or raw_update["date"].max() > requested_through
            ):
                raise DataContractError(f"incremental raw dates are out of range for {symbol}")
            merged_raw = _merge_by_date(base_raw, raw_update)
            symbol_latest_trade_date = pd.to_datetime(merged_raw["date"]).max()
            if latest_observed_trade_date is None or symbol_latest_trade_date > latest_observed_trade_date:
                latest_observed_trade_date = symbol_latest_trade_date
            if provider_paired:
                if provider_qfq is None:
                    raise DataContractError(f"qfq audit file is required for {symbol}")
                merged_qfq, daily_factors, _ = _provider_pair_update(
                    base_raw=base_raw,
                    base_qfq=base_qfq,
                    base_factors=base_factors,
                    raw_update=raw_update,
                    provider_qfq=provider_qfq,
                    symbol=symbol,
                )
                if raw_update.empty:
                    _copy_or_link(
                        qfq_base.snapshot_path / "bars" / f"{symbol}.parquet",
                        qfq_stage / "bars" / f"{symbol}.parquet",
                    )
                    _copy_or_link(
                        qfq_base.snapshot_path / "factors" / f"{symbol}.parquet",
                        qfq_stage / "factors" / f"{symbol}.parquet",
                    )
                    merged_qfq = None
                    daily_factors = None
                reused.append(symbol)
            else:
                factor_changed = _events_changed(old_actions, new_events, symbol=symbol)
                canonical_new_events = _canonical_events(new_events, symbol=symbol)
            if not provider_paired and factor_changed:
                if canonical_new_events.empty:
                    raise DataContractError(f"factor history disappeared for {symbol}")
                daily_factors = expand_daily_factors(
                    merged_raw,
                    new_events,
                    provider_qfq=provider_qfq,
                )
                merged_qfq = build_local_qfq(merged_raw, daily_factors)
                changed.append(symbol)
            elif not provider_paired and raw_update.empty:
                _copy_or_link(
                    qfq_base.snapshot_path / "bars" / f"{symbol}.parquet",
                    qfq_stage / "bars" / f"{symbol}.parquet",
                )
                _copy_or_link(
                    qfq_base.snapshot_path / "factors" / f"{symbol}.parquet",
                    qfq_stage / "factors" / f"{symbol}.parquet",
                )
                merged_qfq = None
                daily_factors = None
                reused.append(symbol)
            elif not provider_paired:
                if canonical_new_events.empty:
                    daily_factors = _identity_extension(
                        base_factors, raw_update, symbol=symbol
                    )
                    update_factors = daily_factors.loc[
                        daily_factors["date"].isin(raw_update["date"])
                    ]
                else:
                    update_factors = expand_daily_factors(raw_update, new_events)
                    daily_factors = _merge_by_date(base_factors, update_factors)
                update_qfq = build_local_qfq(raw_update, update_factors)
                merged_qfq = _merge_by_date(base_qfq, update_qfq)
                reused.append(symbol)

            atomic_parquet(merged_raw, raw_stage / "bars" / f"{symbol}.parquet")
            if merged_qfq is not None and daily_factors is not None:
                atomic_parquet(merged_qfq, qfq_stage / "bars" / f"{symbol}.parquet")
                atomic_parquet(daily_factors, qfq_stage / "factors" / f"{symbol}.parquet")
            if not provider_paired:
                actions.append(_event_output(new_events, symbol=symbol))

        corporate_actions = (
            old_actions.copy()
            if provider_paired
            else (
                pd.concat(actions, ignore_index=True)
                if actions
                else pd.DataFrame(columns=["event_date", *FACTOR_COLUMNS, "symbol", "action_type"])
            )
        )
        atomic_parquet(corporate_actions, raw_stage / "corporate_actions.parquet")
        atomic_parquet(corporate_actions, qfq_stage / "corporate_actions.parquet")
        source_retrieved_at = str(manifest.get("source_retrieved_at") or "")
        try:
            datetime.fromisoformat(source_retrieved_at)
        except ValueError as exc:
            raise DataContractError("incremental bundle retrieval timestamp is invalid") from exc
        bundle_source = manifest.get("source") if provider_paired else None
        source = {
            "provider": source_provider,
            "interface": str(
                (bundle_source or {}).get("interface")
                or "incremental_raw_plus_complete_factor_history"
            ),
            "source_bundle_sha256": sha256_file(
                _bundle_file(
                    bundle,
                    "BUNDLE_MANIFEST.json",
                    label="incremental bundle manifest",
                )
            ),
            "execution_environment": manifest.get("source", {}).get(
                "execution_environment", "private_remote_windows"
            ),
            "bundle_kind": manifest.get("bundle_kind"),
            "base_snapshot_id": raw_base.manifest["snapshot_id"],
            "requested_start": str(manifest["requested_start"]),
            "requested_through": str(manifest["requested_through"]),
        }
        if latest_observed_trade_date is None or pd.isna(latest_observed_trade_date):
            raise DataContractError("incremental bundle produced no trade dates")
        actual_latest_trade_date = str(latest_observed_trade_date.date())
        finalize_price_snapshot(
            raw_stage,
            dataset_id="ashare_daily_raw",
            snapshot_id=raw_stage.name,
            role="canonical",
            status="ready",
            source=source,
            price_basis="raw",
            adjustment={"method": "none"},
            producer={"script": "incremental_update_ashare_v2.py", "version": "1"},
            source_retrieved_at=source_retrieved_at,
            upstream_snapshots=[
                {
                    "dataset_id": "ashare_daily_raw",
                    "snapshot_id": str(raw_base.manifest["snapshot_id"]),
                    "content_sha256": str(raw_base.manifest["content_sha256"]),
                    "price_basis": "raw",
                }
            ],
        )
        finalize_price_snapshot(
            qfq_stage,
            dataset_id="ashare_daily_qfq",
            snapshot_id=qfq_stage.name,
            role="canonical",
            status="ready",
            source=source,
            price_basis="qfq",
            adjustment={
                "method": "provider_native_paired" if provider_paired else "local_factor",
                "adjusted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "adjustment_as_of": actual_latest_trade_date,
                "adjusted_fields": list(PRICE_COLUMNS),
                "normalization_rule": (
                    "provider qfq aligned to frozen base qfq scale"
                    if provider_paired
                    else "latest_valid_trade_date_factor_equals_1"
                ),
                "algorithm_version": (
                    "provider-native-paired-incremental-v1"
                    if provider_paired
                    else "baostock-factor-incremental-v1"
                ),
                "factor_changed_symbols": len(changed),
                "volume_adjusted": False,
                **(
                    {
                        "paired_raw_snapshot_id": raw_stage.name,
                        "paired_raw_component_path": None,
                    }
                    if provider_paired
                    else {}
                ),
            },
            producer={"script": "incremental_update_ashare_v2.py", "version": "1"},
            source_retrieved_at=source_retrieved_at,
            upstream_snapshots=[
                {
                    "dataset_id": "ashare_daily_raw",
                    "snapshot_id": raw_stage.name,
                }
            ],
        )
    except Exception:
        shutil.rmtree(run_root, ignore_errors=True)
        raise

    finalized_raw = _read_json(
        raw_stage / "DATASET_MANIFEST.json",
        label="finalized incremental raw manifest",
    )
    latest = str(finalized_raw["latest_trade_date"])
    return IncrementalPriceResult(
        raw_stage=raw_stage,
        qfq_stage=qfq_stage,
        base_latest_trade_date=base_latest,
        latest_trade_date=latest,
        downloaded_rows=downloaded_rows,
        reused_symbols=tuple(sorted(reused)),
        factor_changed_symbols=tuple(sorted(changed)),
        qfq_recalculated_symbols=tuple(sorted(changed)),
    )
