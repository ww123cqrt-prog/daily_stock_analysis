#!/usr/bin/env python3
"""Download and publish traceable Baostock raw/factor/qfq A-share snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


HISTORY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,"
    "tradestatus,isST"
)
PRICE_COLUMNS = ("open", "high", "low", "close")
NUMERIC_BAR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
    "tradestatus",
    "isST",
)
FACTOR_COLUMNS = (
    "code",
    "dividOperateDate",
    "foreAdjustFactor",
    "backAdjustFactor",
    "adjustFactor",
)
DENIAL_CODES = {"10001011"}
NETWORK_CODES = {
    "10002001",
    "10002002",
    "10002003",
    "10002004",
    "10002005",
    "10002006",
    "10002007",
    "10002008",
}
MAX_FACTOR_START_GAP_DAYS = 7


class BaostockV2Error(RuntimeError):
    """The source bundle cannot satisfy the v2 raw/factor/qfq contract."""


class ProviderDenied(BaostockV2Error):
    """Baostock explicitly denied this client; the run must stop immediately."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_retrieval_window(paths: Sequence[Path]) -> dict[str, str]:
    timestamps = [Path(path).stat().st_mtime for path in paths if Path(path).is_file()]
    if not timestamps:
        raise BaostockV2Error("source retrieval window has no files")
    return {
        "started_at": datetime.fromtimestamp(min(timestamps))
        .astimezone()
        .isoformat(timespec="seconds"),
        "completed_at": datetime.fromtimestamp(max(timestamps))
        .astimezone()
        .isoformat(timespec="seconds"),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_symbol(value: str) -> tuple[str, str]:
    raw = str(value).strip().upper()
    if raw.endswith(".XSHG"):
        code = raw.split(".", 1)[0]
        return f"SH{code}", f"sh.{code}"
    if raw.endswith(".XSHE"):
        code = raw.split(".", 1)[0]
        return f"SZ{code}", f"sz.{code}"
    code = raw.removeprefix("SH").removeprefix("SZ")
    if not (code.isdigit() and len(code) == 6):
        raise BaostockV2Error(f"unsupported A-share symbol: {value}")
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"SH{code}", f"sh.{code}"
    return f"SZ{code}", f"sz.{code}"


def load_universe(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("all") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise BaostockV2Error("universe must contain a non-empty all list")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        symbol, provider_code = canonical_symbol(str(value))
        if symbol not in seen:
            result.append((symbol, provider_code))
            seen.add(symbol)
    return result


def collect_result(result: Any) -> pd.DataFrame:
    error_code = str(getattr(result, "error_code", ""))
    if error_code in DENIAL_CODES:
        raise ProviderDenied("Baostock access is temporarily restricted")
    if error_code != "0":
        raise BaostockV2Error(
            f"Baostock query failed: {error_code} {getattr(result, 'error_msg', '')}"
        )
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def normalize_bars(
    frame: pd.DataFrame,
    symbol: str,
    *,
    source_id: str,
) -> pd.DataFrame:
    required = {"date", *PRICE_COLUMNS, "volume"}
    if frame.empty:
        return pd.DataFrame(columns=["date", *PRICE_COLUMNS, "volume"])
    missing = required - set(frame.columns)
    if missing:
        raise BaostockV2Error(f"bar fields are missing for {symbol}: {sorted(missing)}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    for column in NUMERIC_BAR_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", *PRICE_COLUMNS, "volume"])
    normalized = normalized.sort_values("date", kind="stable").drop_duplicates(
        subset=["date"], keep="last"
    )
    if normalized.empty:
        return normalized
    if (normalized[[*PRICE_COLUMNS, "volume"]] < 0).any().any():
        raise BaostockV2Error(f"negative bar values for {symbol}")
    normalized["symbol"] = symbol
    normalized["source_id"] = source_id
    keep = [
        "date",
        *PRICE_COLUMNS,
        *[
            column
            for column in (
                "preclose",
                "volume",
                "amount",
                "turn",
                "pctChg",
                "tradestatus",
                "isST",
            )
            if column in normalized.columns
        ],
        "symbol",
        "source_id",
    ]
    return normalized[keep].reset_index(drop=True)


def normalize_factor_events(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    missing = set(FACTOR_COLUMNS) - set(frame.columns)
    if missing:
        raise BaostockV2Error(
            f"factor fields are missing for {symbol}: {sorted(missing)}"
        )
    events = frame[list(FACTOR_COLUMNS)].copy()
    events["dividOperateDate"] = pd.to_datetime(
        events["dividOperateDate"], errors="coerce"
    ).dt.normalize()
    for column in ("foreAdjustFactor", "backAdjustFactor", "adjustFactor"):
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events = events.dropna(
        subset=[
            "dividOperateDate",
            "foreAdjustFactor",
            "backAdjustFactor",
            "adjustFactor",
        ]
    )
    events = events.sort_values("dividOperateDate", kind="stable").drop_duplicates(
        subset=["dividOperateDate"], keep="last"
    )
    if (events["foreAdjustFactor"] <= 0).any():
        raise BaostockV2Error(f"factor history is invalid for {symbol}")
    events["symbol"] = symbol
    events["source_id"] = "baostock_query_adjust_factor"
    return events.reset_index(drop=True)


def derive_paired_daily_factors(
    raw_bars: pd.DataFrame,
    provider_qfq: pd.DataFrame,
    *,
    ratio_tolerance: float = 1e-5,
) -> pd.DataFrame:
    raw = raw_bars[["date", *PRICE_COLUMNS]].copy()
    qfq = provider_qfq[["date", *PRICE_COLUMNS]].copy()
    for frame in (raw, qfq):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for column in PRICE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.dropna(subset=["date", *PRICE_COLUMNS], inplace=True)
        frame.sort_values("date", inplace=True)
        frame.drop_duplicates("date", keep="last", inplace=True)
    paired = raw.merge(
        qfq,
        on="date",
        how="inner",
        suffixes=("_raw", "_qfq"),
        validate="one_to_one",
    )
    if len(paired) != len(raw) or len(paired) != len(qfq):
        raise BaostockV2Error("paired raw/qfq dates do not match")
    ratios = pd.DataFrame(index=paired.index)
    for column in PRICE_COLUMNS:
        denominator = paired[f"{column}_raw"]
        if (denominator <= 0).any():
            raise BaostockV2Error("paired raw prices must be positive")
        ratios[column] = paired[f"{column}_qfq"] / denominator
    factors = ratios.median(axis=1)
    deviations = ratios.div(factors, axis=0).sub(1.0).abs()
    if (
        factors.isna().any()
        or (factors <= 0).any()
        or float(deviations.max().max()) > ratio_tolerance
    ):
        raise BaostockV2Error("paired qfq factors disagree across OHLC")
    return pd.DataFrame({"date": paired["date"], "factor": factors})


def expand_daily_factors(
    raw_bars: pd.DataFrame,
    factor_events: pd.DataFrame,
    *,
    provider_qfq: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw_bars.empty:
        raise BaostockV2Error("raw bars are empty")
    raw_dates = pd.DataFrame(
        {"date": pd.to_datetime(raw_bars["date"], errors="coerce").dt.normalize()}
    ).dropna()
    raw_dates = raw_dates.sort_values("date").drop_duplicates("date")
    events = factor_events.copy()
    if "dividOperateDate" not in events.columns or "foreAdjustFactor" not in events.columns:
        raise BaostockV2Error("factor event fields are incomplete")
    events["source_event_date"] = pd.to_datetime(
        events["dividOperateDate"], errors="coerce"
    ).dt.normalize()
    events["factor"] = pd.to_numeric(events["foreAdjustFactor"], errors="coerce")
    events = events.dropna(subset=["source_event_date", "factor"])
    events = events.sort_values("source_event_date").drop_duplicates(
        "source_event_date", keep="last"
    )
    first_raw_date = raw_dates["date"].iloc[0]
    if events.empty:
        if provider_qfq is None:
            raise BaostockV2Error("factor events do not cover earliest raw bar")
        paired = derive_paired_daily_factors(raw_bars, provider_qfq)
        daily = raw_dates.merge(paired, on="date", validate="one_to_one")
        daily["source_event_date"] = pd.NaT
        backfilled = pd.Series(False, index=daily.index)
        paired_fallback = pd.Series(True, index=daily.index)
    else:
        first_event_date = events["source_event_date"].iloc[0]
        start_gap_days = max(int((first_event_date - first_raw_date).days), 0)
        daily = pd.merge_asof(
            raw_dates,
            events[["source_event_date", "factor"]],
            left_on="date",
            right_on="source_event_date",
            direction="backward",
        )
        missing = daily["factor"].isna()
        backfilled = pd.Series(False, index=daily.index)
        paired_fallback = pd.Series(False, index=daily.index)
        if missing.any() and start_gap_days <= MAX_FACTOR_START_GAP_DAYS:
            daily.loc[missing, "factor"] = 1.0
            backfilled.loc[missing] = True
        elif missing.any():
            if provider_qfq is None:
                raise BaostockV2Error("factor events do not cover earliest raw bar")
            paired = derive_paired_daily_factors(raw_bars, provider_qfq).set_index(
                "date"
            )
            missing_dates = daily.loc[missing, "date"]
            daily.loc[missing, "factor"] = missing_dates.map(paired["factor"])
            paired_fallback.loc[missing] = True
    if daily["factor"].isna().any() or (daily["factor"] <= 0).any():
        raise BaostockV2Error("factor events do not cover every raw bar")
    latest_factor = (
        float(events["factor"].iloc[-1])
        if not events.empty
        else float(daily["factor"].iloc[-1])
    )
    if not np.isclose(latest_factor, 1.0, atol=1e-6):
        if provider_qfq is None:
            raise BaostockV2Error(
                "non-unit terminal foreAdjustFactor requires provider qfq verification"
            )
        paired = derive_paired_daily_factors(raw_bars, provider_qfq)
        paired_latest = float(paired.sort_values("date")["factor"].iloc[-1])
        if not np.isclose(latest_factor, paired_latest, atol=1e-6, rtol=1e-6):
            raise BaostockV2Error(
                "terminal foreAdjustFactor does not match provider qfq"
            )
    daily["factor_source"] = "baostock_foreAdjustFactor"
    daily.loc[backfilled, "factor_source"] = "baostock_listing_gap_identity"
    daily.loc[paired_fallback, "factor_source"] = (
        "baostock_raw_qfq_pair_fallback"
    )
    daily["factor_backfilled_before_first_event"] = backfilled.astype(bool)
    daily["factor_paired_fallback"] = paired_fallback.astype(bool)
    return daily.reset_index(drop=True)


def build_local_qfq(
    raw_bars: pd.DataFrame,
    daily_factors: pd.DataFrame,
) -> pd.DataFrame:
    raw = raw_bars.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    factors = daily_factors[["date", "factor"]].copy()
    result = raw.merge(factors, on="date", how="left", validate="one_to_one")
    if result["factor"].isna().any():
        raise BaostockV2Error("daily factors do not cover raw bars")
    for column in PRICE_COLUMNS:
        result[column] = (pd.to_numeric(result[column]) * result["factor"]).round(4)
    if "preclose" in result.columns:
        first_preclose = float(result.loc[0, "preclose"]) * float(
            result.loc[0, "factor"]
        )
        result["preclose"] = result["close"].shift(1)
        result.loc[0, "preclose"] = round(first_preclose, 4)
    if "pctChg" in result.columns and "preclose" in result.columns:
        result["pctChg"] = (
            (result["close"] / result["preclose"] - 1.0) * 100.0
        ).replace([np.inf, -np.inf], np.nan)
    result["source_id"] = "baostock_raw_x_foreAdjustFactor"
    return result


def compare_provider_qfq(
    local_qfq: pd.DataFrame,
    provider_qfq: pd.DataFrame,
    *,
    absolute_tolerance: float = 0.005,
) -> dict[str, Any]:
    left = local_qfq[["date", *PRICE_COLUMNS]].copy()
    right = provider_qfq[["date", *PRICE_COLUMNS]].copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    merged = left.merge(right, on="date", suffixes=("_local", "_provider"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise BaostockV2Error("provider qfq dates do not match raw dates")
    maximum = 0.0
    for column in PRICE_COLUMNS:
        errors = (
            pd.to_numeric(merged[f"{column}_local"])
            - pd.to_numeric(merged[f"{column}_provider"])
        ).abs()
        maximum = max(maximum, float(errors.max()))
        if (errors > absolute_tolerance).any():
            raise BaostockV2Error(
                f"provider qfq mismatch for {column}: max error {float(errors.max()):.6f}"
            )
    return {"rows": len(merged), "max_absolute_error": maximum}


def compare_legacy_qfq_directory(
    *,
    qfq_bars_dir: Path,
    legacy_qfq_dir: Path,
    minimum_symbol_coverage: float = 0.98,
    minimum_date_coverage: float = 0.999,
    minimum_explained_consistency: float = 0.99,
    ratio_tolerance: float = 0.005,
    return_tolerance_pct_points: float = 0.1,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare overlapping history while allowing explainable constant rescaling."""

    qfq_bars_dir = Path(qfq_bars_dir)
    legacy_qfq_dir = Path(legacy_qfq_dir)
    legacy_files = sorted(legacy_qfq_dir.glob("*.parquet"))
    if not legacy_files:
        raise BaostockV2Error("legacy Baostock qfq comparison input is empty")
    details: list[dict[str, Any]] = []
    legacy_rows_total = 0
    overlapping_rows = 0
    compared_symbols = 0
    stable_symbols = 0
    scale_shift_symbols = 0
    explained_legacy_drift_symbols = 0
    consistent_symbols = 0
    for legacy_path in legacy_files:
        symbol, _ = canonical_symbol(legacy_path.stem)
        new_path = qfq_bars_dir / f"{symbol}.parquet"
        row: dict[str, Any] = {
            "symbol": symbol,
            "legacy_file": legacy_path.name,
            "new_file": new_path.name,
            "legacy_rows": 0,
            "overlap_rows": 0,
            "date_coverage": 0.0,
            "median_price_ratio": np.nan,
            "max_ratio_deviation": np.nan,
            "ratio_stable": False,
            "constant_scale_shift": False,
            "new_return_match_rate": np.nan,
            "legacy_return_match_rate": np.nan,
            "legacy_splice_days": 0,
            "legacy_scale_drift_explained": False,
            "price_path_consistent": False,
            "reason": "new_symbol_missing",
        }
        try:
            legacy = pd.read_parquet(legacy_path)
            legacy["date"] = pd.to_datetime(
                legacy["date"], errors="coerce"
            ).dt.normalize()
            legacy["close"] = pd.to_numeric(legacy["close"], errors="coerce")
            legacy = legacy.dropna(subset=["date", "close"])
            legacy = legacy.loc[legacy["close"] > 0].drop_duplicates(
                "date", keep="last"
            )
        except Exception:
            row["reason"] = "legacy_file_invalid"
            details.append(row)
            continue
        legacy_rows = len(legacy)
        row["legacy_rows"] = legacy_rows
        legacy_rows_total += legacy_rows
        if not new_path.is_file() or legacy.empty:
            if legacy.empty:
                row["reason"] = "legacy_file_empty"
            details.append(row)
            continue
        try:
            current = pd.read_parquet(new_path, columns=["date", "close"])
            current["date"] = pd.to_datetime(
                current["date"], errors="coerce"
            ).dt.normalize()
            current["close"] = pd.to_numeric(current["close"], errors="coerce")
            current = current.dropna(subset=["date", "close"])
            current = current.loc[current["close"] > 0].drop_duplicates(
                "date", keep="last"
            )
        except Exception:
            row["reason"] = "new_file_invalid"
            details.append(row)
            continue
        overlap = legacy.merge(
            current,
            on="date",
            how="inner",
            suffixes=("_legacy", "_new"),
        )
        overlap_rows = len(overlap)
        row["overlap_rows"] = overlap_rows
        row["date_coverage"] = (
            overlap_rows / legacy_rows if legacy_rows else 0.0
        )
        overlapping_rows += overlap_rows
        if overlap.empty:
            row["reason"] = "no_overlapping_dates"
            details.append(row)
            continue
        compared_symbols += 1
        ratios = overlap["close_new"] / overlap["close_legacy"]
        median_ratio = float(ratios.median())
        deviations = (ratios / median_ratio - 1.0).abs()
        maximum_deviation = float(deviations.max())
        stable = bool(np.isfinite(maximum_deviation)) and (
            maximum_deviation <= ratio_tolerance
        )
        scale_shift = stable and abs(median_ratio - 1.0) > ratio_tolerance
        explained_legacy_drift = False
        new_match_rate = np.nan
        legacy_match_rate = np.nan
        legacy_splice_days = 0
        if "pctChg" in overlap.columns:
            reference_returns = pd.to_numeric(
                overlap["pctChg"], errors="coerce"
            )
            new_returns = overlap["close_new"].pct_change(fill_method=None) * 100.0
            legacy_returns = (
                overlap["close_legacy"].pct_change(fill_method=None) * 100.0
            )
            valid = (
                reference_returns.notna()
                & new_returns.notna()
                & legacy_returns.notna()
            )
            if valid.any():
                new_errors = (new_returns[valid] - reference_returns[valid]).abs()
                legacy_errors = (
                    legacy_returns[valid] - reference_returns[valid]
                ).abs()
                new_match_rate = float(
                    (new_errors <= return_tolerance_pct_points).mean()
                )
                legacy_match_rate = float(
                    (legacy_errors <= return_tolerance_pct_points).mean()
                )
                legacy_splice_days = int(
                    (
                        (legacy_errors > max(0.5, return_tolerance_pct_points))
                        & (new_errors <= return_tolerance_pct_points)
                    ).sum()
                )
                explained_legacy_drift = (
                    not stable
                    and new_match_rate >= 0.995
                    and legacy_splice_days >= 1
                )
        price_path_consistent = stable or explained_legacy_drift
        row.update(
            {
                "median_price_ratio": median_ratio,
                "max_ratio_deviation": maximum_deviation,
                "ratio_stable": stable,
                "constant_scale_shift": scale_shift,
                "new_return_match_rate": new_match_rate,
                "legacy_return_match_rate": legacy_match_rate,
                "legacy_splice_days": legacy_splice_days,
                "legacy_scale_drift_explained": explained_legacy_drift,
                "price_path_consistent": price_path_consistent,
                "reason": (
                    "ok"
                    if stable
                    else (
                        "legacy_incremental_scale_drift"
                        if explained_legacy_drift
                        else "unexplained_mid_history_scale_drift"
                    )
                ),
            }
        )
        stable_symbols += int(stable)
        scale_shift_symbols += int(scale_shift)
        explained_legacy_drift_symbols += int(explained_legacy_drift)
        consistent_symbols += int(price_path_consistent)
        details.append(row)

    detail_frame = pd.DataFrame(details)
    symbol_coverage = compared_symbols / len(legacy_files)
    date_coverage = (
        overlapping_rows / legacy_rows_total if legacy_rows_total else 0.0
    )
    ratio_stability = (
        stable_symbols / compared_symbols if compared_symbols else 0.0
    )
    explained_consistency = (
        consistent_symbols / compared_symbols if compared_symbols else 0.0
    )
    failed_gates: list[str] = []
    if symbol_coverage < minimum_symbol_coverage:
        failed_gates.append("symbol_coverage")
    if date_coverage < minimum_date_coverage:
        failed_gates.append("date_coverage")
    if explained_consistency < minimum_explained_consistency:
        failed_gates.append("unexplained_price_path")
    worst = (
        detail_frame.sort_values(
            "max_ratio_deviation", ascending=False, na_position="last"
        )
        .head(20)[
            [
                "symbol",
                "date_coverage",
                "median_price_ratio",
                "max_ratio_deviation",
                "reason",
            ]
        ]
        .replace({np.nan: None})
        .to_dict("records")
    )
    summary = {
        "status": "pass" if not failed_gates else "failed",
        "failed_gates": failed_gates,
        "legacy_symbols": len(legacy_files),
        "new_symbols": len(list(qfq_bars_dir.glob("*.parquet"))),
        "compared_symbols": compared_symbols,
        "symbol_coverage": symbol_coverage,
        "legacy_rows": legacy_rows_total,
        "overlap_rows": overlapping_rows,
        "date_coverage": date_coverage,
        "ratio_stable_symbols": stable_symbols,
        "ratio_stability_rate": ratio_stability,
        "constant_scale_shift_symbols": scale_shift_symbols,
        "explained_legacy_scale_drift_symbols": explained_legacy_drift_symbols,
        "unexplained_price_path_symbols": compared_symbols - consistent_symbols,
        "explained_consistency_rate": explained_consistency,
        "thresholds": {
            "minimum_symbol_coverage": minimum_symbol_coverage,
            "minimum_date_coverage": minimum_date_coverage,
            "minimum_explained_consistency": minimum_explained_consistency,
            "ratio_tolerance": ratio_tolerance,
            "return_tolerance_pct_points": return_tolerance_pct_points,
        },
        "interpretation": (
            "A stable non-1 price ratio is an explainable qfq renormalization; "
            "legacy scale drift is accepted only when new returns match the "
            "legacy provider pctChg while legacy close-to-close returns do not."
        ),
        "worst_examples": worst,
    }
    return summary, detail_frame


def needs_provider_qfq_audit(
    symbol: str,
    factor_events: pd.DataFrame,
    *,
    audit_since: str,
    sample_modulus: int = 100,
    factor_backfill_used: bool = False,
) -> bool:
    if sample_modulus < 1:
        raise BaostockV2Error("qfq sample modulus must be positive")
    if factor_backfill_used:
        return True
    ordered_events = factor_events.copy()
    ordered_events["_event_date"] = pd.to_datetime(
        ordered_events["dividOperateDate"], errors="coerce"
    )
    ordered_events["_fore_factor"] = pd.to_numeric(
        ordered_events["foreAdjustFactor"], errors="coerce"
    )
    ordered_events = ordered_events.dropna(
        subset=["_event_date", "_fore_factor"]
    ).sort_values("_event_date")
    if (
        not ordered_events.empty
        and not np.isclose(
            float(ordered_events["_fore_factor"].iloc[-1]),
            1.0,
            atol=1e-6,
        )
    ):
        return True
    event_dates = pd.to_datetime(
        factor_events["dividOperateDate"], errors="coerce"
    ).dropna()
    if not event_dates.empty and event_dates.max() >= pd.Timestamp(audit_since):
        return True
    sample_key = int(hashlib.sha256(symbol.encode("ascii")).hexdigest()[:8], 16)
    return sample_key % sample_modulus == 0


@dataclass
class RequestGate:
    interval: float
    budget: int
    calls: int = 0
    last_call: float = 0.0

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.calls >= self.budget:
            raise BaostockV2Error("request budget exhausted")
        wait = self.interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        result = function(*args, **kwargs)
        self.calls += 1
        self.last_call = time.monotonic()
        return result


def _configure_socket_timeout(timeout: float) -> None:
    try:
        import baostock.common.context as context

        if getattr(context, "default_socket", None) is not None:
            context.default_socket.settimeout(float(timeout))
    except Exception:
        pass


def _query_with_retry(
    gate: RequestGate,
    function: Callable[..., Any],
    *,
    retries: int,
    relogin: Callable[[], None],
    **kwargs: Any,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(max(int(retries), 1)):
        try:
            return collect_result(gate.call(function, **kwargs))
        except ProviderDenied:
            raise
        except (BaostockV2Error, OSError, socket.timeout) as exc:
            last_error = exc
            code = str(exc).split(maxsplit=1)[0]
            if attempt + 1 >= max(int(retries), 1):
                break
            if code in NETWORK_CODES or isinstance(exc, (OSError, socket.timeout)):
                relogin()
            time.sleep(min(5 * (attempt + 1), 15))
    raise BaostockV2Error(str(last_error or "Baostock query failed"))


def _symbol_source_files_complete(root: Path, symbol: str) -> bool:
    raw_path = root / "raw" / f"{symbol}.parquet"
    factor_path = root / "factor_events" / f"{symbol}.parquet"
    try:
        if not raw_path.is_file() or not factor_path.is_file():
            return False
        raw = pd.read_parquet(raw_path)
        factors = pd.read_parquet(factor_path)
        return not raw.empty and set(FACTOR_COLUMNS).issubset(factors.columns)
    except Exception:
        return False


def download_bundle(
    *,
    universe_file: Path,
    output_dir: Path,
    through: str,
    start: str = "1990-01-01",
    request_interval: float = 0.25,
    request_budget: int = 18000,
    retries: int = 1,
    timeout: int = 60,
    max_consecutive_failures: int = 3,
    qfq_audit_since: str | None = None,
    qfq_sample_modulus: int = 100,
    resume: bool = False,
    provider: Any | None = None,
) -> Path:
    if request_interval < 0.2:
        raise BaostockV2Error("request interval must be at least 0.2 seconds")
    if retries < 1 or retries > 2:
        raise BaostockV2Error("retries must be between 1 and 2")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "RUNNING.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise BaostockV2Error("another Baostock v2 run owns this output") from exc
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)

    symbols = load_universe(universe_file)
    audit_since = qfq_audit_since or (
        pd.Timestamp(through) - pd.Timedelta(days=45)
    ).date().isoformat()
    gate = RequestGate(request_interval, request_budget)
    bs = provider
    if bs is None:
        import baostock as bs

    source_retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    empty_symbols: list[str] = []
    consecutive_failures = 0
    status = "failed"
    started = time.monotonic()

    def login() -> None:
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(timeout))
        try:
            result = bs.login()
        finally:
            socket.setdefaulttimeout(previous)
        code = str(getattr(result, "error_code", ""))
        if code in DENIAL_CODES:
            raise ProviderDenied("Baostock access is temporarily restricted")
        if code != "0":
            raise BaostockV2Error(
                f"Baostock login failed: {code} {getattr(result, 'error_msg', '')}"
            )
        _configure_socket_timeout(timeout)

    def relogin() -> None:
        try:
            bs.logout()
        except Exception:
            pass
        login()

    try:
        login()
        for index, (symbol, provider_code) in enumerate(symbols, 1):
            progress = {
                "status": "running",
                "index": index,
                "total": len(symbols),
                "symbol": symbol,
                "through": through,
                "requests_used": gate.calls,
                "source_retrieved_at": source_retrieved_at,
            }
            atomic_json(output_dir / "progress.json", progress)
            if resume and _symbol_source_files_complete(output_dir, symbol):
                raw = pd.read_parquet(output_dir / "raw" / f"{symbol}.parquet")
                events = pd.read_parquet(
                    output_dir / "factor_events" / f"{symbol}.parquet"
                )
            else:
                try:
                    raw_result = _query_with_retry(
                        gate,
                        bs.query_history_k_data_plus,
                        retries=retries,
                        relogin=relogin,
                        code=provider_code,
                        fields=HISTORY_FIELDS,
                        start_date=start,
                        end_date=through,
                        frequency="d",
                        adjustflag="3",
                    )
                    raw = normalize_bars(raw_result, symbol, source_id="baostock_raw")
                    if raw.empty:
                        empty_symbols.append(symbol)
                        consecutive_failures = 0
                        continue
                    factor_result = _query_with_retry(
                        gate,
                        bs.query_adjust_factor,
                        retries=retries,
                        relogin=relogin,
                        code=provider_code,
                        start_date=start,
                        end_date=through,
                    )
                    events = normalize_factor_events(factor_result, symbol)
                    atomic_parquet(raw, output_dir / "raw" / f"{symbol}.parquet")
                    atomic_parquet(
                        events, output_dir / "factor_events" / f"{symbol}.parquet"
                    )
                except ProviderDenied:
                    raise
                except Exception as exc:
                    consecutive_failures += 1
                    failures.append(
                        {
                            "symbol": symbol,
                            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                        }
                    )
                    atomic_json(output_dir / "failures.json", {"failures": failures})
                    if consecutive_failures >= max_consecutive_failures:
                        raise BaostockV2Error(
                            f"stopped after {consecutive_failures} consecutive failures"
                        ) from exc
                    continue
            qfq_path = output_dir / "qfq_provider" / f"{symbol}.parquet"
            raw_first = pd.to_datetime(raw["date"], errors="coerce").min()
            event_first = (
                pd.to_datetime(
                    events["dividOperateDate"], errors="coerce"
                ).min()
                if not events.empty
                else pd.NaT
            )
            start_gap_days = (
                int((event_first - raw_first).days)
                if pd.notna(event_first) and pd.notna(raw_first)
                else None
            )
            paired_fallback_required = (
                start_gap_days is None
                or start_gap_days > MAX_FACTOR_START_GAP_DAYS
            )
            bounded_identity_required = (
                start_gap_days is not None
                and 0 < start_gap_days <= MAX_FACTOR_START_GAP_DAYS
            )
            audit_required = needs_provider_qfq_audit(
                symbol,
                events,
                audit_since=audit_since,
                sample_modulus=qfq_sample_modulus,
                factor_backfill_used=(
                    bounded_identity_required or paired_fallback_required
                ),
            )
            qfq: pd.DataFrame | None = None
            if qfq_path.is_file():
                qfq = pd.read_parquet(qfq_path)
            elif audit_required:
                qfq_result = _query_with_retry(
                    gate,
                    bs.query_history_k_data_plus,
                    retries=retries,
                    relogin=relogin,
                    code=provider_code,
                    fields=HISTORY_FIELDS,
                    start_date=start,
                    end_date=through,
                    frequency="d",
                    adjustflag="2",
                )
                qfq = normalize_bars(
                    qfq_result, symbol, source_id="baostock_qfq_provider"
                )
                atomic_parquet(qfq, qfq_path)
            factors = expand_daily_factors(
                raw,
                events,
                provider_qfq=qfq,
            )
            local_qfq = build_local_qfq(raw, factors)
            comparison = (
                compare_provider_qfq(local_qfq, qfq) if qfq is not None else None
            )
            rows.append(
                {
                    "symbol": symbol,
                    "provider_code": provider_code,
                    "raw_path": f"raw/{symbol}.parquet",
                    "qfq_provider_path": (
                        f"qfq_provider/{symbol}.parquet" if qfq is not None else None
                    ),
                    "factor_events_path": f"factor_events/{symbol}.parquet",
                    "raw_rows": len(raw),
                    "factor_event_rows": len(events),
                    "factor_start_backfill_days": int(
                        factors["factor_backfilled_before_first_event"].sum()
                    ),
                    "factor_paired_fallback_days": int(
                        factors["factor_paired_fallback"].sum()
                    ),
                    "first_date": pd.Timestamp(raw["date"].min()).date().isoformat(),
                    "last_date": pd.Timestamp(raw["date"].max()).date().isoformat(),
                    "raw_sha256": sha256_file(output_dir / "raw" / f"{symbol}.parquet"),
                    "qfq_provider_sha256": (
                        sha256_file(qfq_path) if qfq is not None else None
                    ),
                    "factor_events_sha256": sha256_file(
                        output_dir / "factor_events" / f"{symbol}.parquet"
                    ),
                    "qfq_max_absolute_error": (
                        comparison["max_absolute_error"] if comparison else None
                    ),
                }
            )
            consecutive_failures = 0
            if index <= 3 or index % 100 == 0:
                elapsed = time.monotonic() - started
                eta = elapsed / index * (len(symbols) - index)
                print(
                    f"[{index}/{len(symbols)}] {symbol} rows={len(raw)} "
                    f"requests={gate.calls} eta_min={eta / 60:.1f}",
                    flush=True,
                )
        nonempty_coverage = len(rows) / len(symbols)
        if failures:
            raise BaostockV2Error(f"{len(failures)} symbols failed")
        if nonempty_coverage < 0.98:
            raise BaostockV2Error(
                f"non-empty symbol coverage {nonempty_coverage:.4%} is below 98%"
            )
        status = "ready"
    except ProviderDenied:
        status = "provider_denied"
        raise
    finally:
        try:
            bs.logout()
        except Exception:
            pass
        index_frame = pd.DataFrame(rows)
        if not index_frame.empty:
            atomic_parquet(index_frame, output_dir / "BUNDLE_FILE_INDEX.parquet")
        manifest = {
            "schema_version": 2,
            "bundle_kind": "baostock_raw_factor_qfq",
            "status": status,
            "source": {
                "provider": "baostock",
                "interfaces": [
                    "query_history_k_data_plus_adjustflag_3",
                    "query_history_k_data_plus_adjustflag_2",
                    "query_adjust_factor",
                ],
                "execution_environment": "private_remote_windows",
            },
            "source_retrieved_at": source_retrieved_at,
            "requested_start": start,
            "requested_through": through,
            "requested_symbols": len(symbols),
            "completed_symbols": len(rows),
            "empty_symbols": len(empty_symbols),
            "failed_symbols": len(failures),
            "nonempty_coverage": len(rows) / len(symbols),
            "requests_used": gate.calls,
            "request_interval_seconds": request_interval,
            "qfq_audit_since": audit_since,
            "qfq_sample_modulus": qfq_sample_modulus,
            "qfq_audited_symbols": sum(
                1 for row in rows if row.get("qfq_provider_path")
            ),
            "index_path": "BUNDLE_FILE_INDEX.parquet",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(output_dir / "BUNDLE_MANIFEST.json", manifest)
        atomic_json(
            output_dir / "progress.json",
            {
                "status": status,
                "completed": len(rows),
                "total": len(symbols),
                "through": through,
                "requests_used": gate.calls,
            },
        )
        lock_path.unlink(missing_ok=True)
    return output_dir / "BUNDLE_MANIFEST.json"


def build_snapshots_from_bundle(
    *,
    bundle_dir: Path,
    quant_data_root: Path,
    publish: bool = False,
    legacy_qfq_dir: Path | None = None,
) -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from quant_data_contract import (
        finalize_price_snapshot,
        publish_snapshot_group,
    )

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    root = Path(quant_data_root).expanduser().resolve()
    manifest = json.loads(
        (bundle_dir / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "ready":
        raise BaostockV2Error("source bundle is not ready")
    index = pd.read_parquet(bundle_dir / str(manifest["index_path"]))
    if index.empty or len(index) != int(manifest["completed_symbols"]):
        raise BaostockV2Error("source bundle index is incomplete")
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    raw_snapshot_id = f"ashare-baostock-raw-{run_id}"
    qfq_snapshot_id = f"ashare-baostock-qfq-{run_id}"
    staging_run = root / "staging" / f"ashare-v2-{run_id}"
    raw_stage = staging_run / raw_snapshot_id
    qfq_stage = staging_run / qfq_snapshot_id
    actions: list[pd.DataFrame] = []
    source_files: list[Path] = [bundle_dir / "BUNDLE_MANIFEST.json"]
    maximum_error = 0.0
    audited_symbols = 0
    bounded_factor_backfill_symbols = 0
    paired_factor_fallback_symbols = 0
    for row in index.itertuples(index=False):
        symbol = str(row.symbol)
        raw_path = bundle_dir / str(row.raw_path)
        events_path = bundle_dir / str(row.factor_events_path)
        source_files.extend([raw_path, events_path])
        if sha256_file(raw_path) != str(row.raw_sha256):
            raise BaostockV2Error(f"raw bundle hash mismatch: {symbol}")
        if sha256_file(events_path) != str(row.factor_events_sha256):
            raise BaostockV2Error(f"factor bundle hash mismatch: {symbol}")
        raw = pd.read_parquet(raw_path)
        events = pd.read_parquet(events_path)
        provider_relative = getattr(row, "qfq_provider_path", None)
        provider_qfq: pd.DataFrame | None = None
        if isinstance(provider_relative, str) and provider_relative:
            provider_path = bundle_dir / provider_relative
            source_files.append(provider_path)
            if sha256_file(provider_path) != str(row.qfq_provider_sha256):
                raise BaostockV2Error(f"qfq bundle hash mismatch: {symbol}")
            provider_qfq = pd.read_parquet(provider_path)
        daily_factors = expand_daily_factors(
            raw,
            events,
            provider_qfq=provider_qfq,
        )
        bounded_factor_backfill_symbols += int(
            daily_factors["factor_backfilled_before_first_event"].any()
        )
        paired_factor_fallback_symbols += int(
            daily_factors["factor_paired_fallback"].any()
        )
        local_qfq = build_local_qfq(raw, daily_factors)
        if provider_qfq is not None:
            comparison = compare_provider_qfq(local_qfq, provider_qfq)
            maximum_error = max(maximum_error, comparison["max_absolute_error"])
            audited_symbols += 1
        atomic_parquet(raw, raw_stage / "bars" / f"{symbol}.parquet")
        atomic_parquet(local_qfq, qfq_stage / "bars" / f"{symbol}.parquet")
        atomic_parquet(daily_factors, qfq_stage / "factors" / f"{symbol}.parquet")
        action_frame = events.rename(columns={"dividOperateDate": "event_date"}).copy()
        action_frame["action_type"] = "adjustment_factor_event"
        actions.append(action_frame)
        paired_rows = daily_factors.loc[
            daily_factors["factor_paired_fallback"]
        ].copy()
        if not paired_rows.empty:
            previous = daily_factors["factor"].shift(1)
            transitions = paired_rows.loc[
                previous.loc[paired_rows.index].isna()
                | ~np.isclose(
                    paired_rows["factor"],
                    previous.loc[paired_rows.index],
                    atol=1e-10,
                    rtol=1e-10,
                )
            ]
            derived_actions = transitions[["date", "factor"]].rename(
                columns={
                    "date": "event_date",
                    "factor": "foreAdjustFactor",
                }
            )
            derived_actions["symbol"] = symbol
            derived_actions["source_id"] = "baostock_raw_qfq_pair_derived"
            derived_actions["action_type"] = "paired_factor_transition"
            actions.append(derived_actions)
    corporate_actions = pd.concat(actions, ignore_index=True)
    atomic_parquet(corporate_actions, raw_stage / "corporate_actions.parquet")
    atomic_parquet(corporate_actions, qfq_stage / "corporate_actions.parquet")
    legacy_dir = Path(legacy_qfq_dir) if legacy_qfq_dir else (
        root / "parquet" / "ashare_1d" / "baostock_qfq"
    )
    legacy_comparison: dict[str, Any] | None = None
    if legacy_dir.is_dir():
        legacy_comparison, comparison_details = compare_legacy_qfq_directory(
            qfq_bars_dir=qfq_stage / "bars",
            legacy_qfq_dir=legacy_dir,
        )
        atomic_json(qfq_stage / "LEGACY_COMPARISON.json", legacy_comparison)
        atomic_parquet(
            comparison_details,
            qfq_stage / "LEGACY_COMPARISON.parquet",
        )
        if legacy_comparison["status"] != "pass":
            raise BaostockV2Error(
                "new-versus-legacy qfq comparison failed: "
                + ", ".join(legacy_comparison["failed_gates"])
            )
    elif publish:
        raise BaostockV2Error("legacy Baostock qfq comparison input is missing")
    retrieval_window = source_retrieval_window(source_files)
    source = {
        "provider": "baostock",
        "interface": "raw_plus_factor_with_same_provider_qfq_evidence",
        "source_bundle_sha256": sha256_file(bundle_dir / "BUNDLE_MANIFEST.json"),
        "execution_environment": "private_remote_windows",
        "retrieval_window": retrieval_window,
        "remote_run_started_at": manifest.get("source_retrieved_at"),
    }
    timestamp = retrieval_window["completed_at"]
    through = str(manifest["requested_through"])
    finalize_price_snapshot(
        raw_stage,
        dataset_id="ashare_daily_raw",
        snapshot_id=raw_snapshot_id,
        role="canonical",
        status="ready",
        source=source,
        price_basis="raw",
        adjustment={"method": "none"},
        producer={"script": "download_baostock_v2.py", "version": "2"},
        source_retrieved_at=timestamp,
    )
    finalize_price_snapshot(
        qfq_stage,
        dataset_id="ashare_daily_qfq",
        snapshot_id=qfq_snapshot_id,
        role="canonical",
        status="ready",
        source=source,
        price_basis="qfq",
        adjustment={
            "method": "local_factor",
            "adjusted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "adjustment_as_of": through,
            "adjusted_fields": list(PRICE_COLUMNS),
            "normalization_rule": "latest_daily_factor_equals_1",
            "algorithm_version": "baostock-factor-evidence-v2",
            "provider_qfq_cross_check_max_absolute_error": maximum_error,
            "provider_qfq_cross_check_symbols": audited_symbols,
            "bounded_factor_start_backfill_symbols": bounded_factor_backfill_symbols,
            "paired_qfq_factor_fallback_symbols": paired_factor_fallback_symbols,
            "maximum_factor_start_gap_days": MAX_FACTOR_START_GAP_DAYS,
            "legacy_comparison_path": (
                "LEGACY_COMPARISON.json" if legacy_comparison else None
            ),
            "legacy_comparison_status": (
                legacy_comparison["status"] if legacy_comparison else "not_run"
            ),
            "volume_adjusted": False,
        },
        producer={"script": "download_baostock_v2.py", "version": "2"},
        source_retrieved_at=timestamp,
        upstream_snapshots=[
            {"dataset_id": "ashare_daily_raw", "snapshot_id": raw_snapshot_id}
        ],
    )
    if not publish:
        return raw_stage, qfq_stage
    raw_root = root / "parquet" / "ashare_1d" / "ashare_daily_raw"
    qfq_root = root / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    published_raw, published_qfq = publish_snapshot_group(
        [(raw_stage, raw_root), (qfq_stage, qfq_root)],
        quant_data_root=root,
        update_root_catalog=True,
    )
    return published_raw, published_qfq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--universe-file", type=Path, required=True)
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--through", required=True)
    download.add_argument("--start", default="1990-01-01")
    download.add_argument("--request-interval", type=float, default=0.25)
    download.add_argument("--request-budget", type=int, default=18000)
    download.add_argument("--retries", type=int, default=1)
    download.add_argument("--timeout", type=int, default=60)
    download.add_argument("--max-consecutive-failures", type=int, default=3)
    download.add_argument("--qfq-audit-since")
    download.add_argument("--qfq-sample-modulus", type=int, default=100)
    download.add_argument("--resume", action="store_true")
    build = commands.add_parser("build")
    build.add_argument("--bundle-dir", type=Path, required=True)
    build.add_argument("--quant-data-root", type=Path, required=True)
    build.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            path = download_bundle(
                universe_file=args.universe_file,
                output_dir=args.output_dir,
                through=args.through,
                start=args.start,
                request_interval=args.request_interval,
                request_budget=args.request_budget,
                retries=args.retries,
                timeout=args.timeout,
                max_consecutive_failures=args.max_consecutive_failures,
                qfq_audit_since=args.qfq_audit_since,
                qfq_sample_modulus=args.qfq_sample_modulus,
                resume=args.resume,
            )
            print(path)
            return 0
        raw, qfq = build_snapshots_from_bundle(
            bundle_dir=args.bundle_dir,
            quant_data_root=args.quant_data_root,
            publish=args.publish,
        )
        print(json.dumps({"raw": str(raw), "qfq": str(qfq)}, ensure_ascii=False))
        return 0
    except (BaostockV2Error, OSError, ValueError, KeyError) as exc:
        print(f"A-share v2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
