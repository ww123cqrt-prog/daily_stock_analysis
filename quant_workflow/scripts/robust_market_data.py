"""Load PIT member-day market data and build no-lookahead market regimes."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from report_market_regime import classify_regime  # noqa: E402
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)


def _symbol_file(price_dir: Path, symbol: str) -> Path | None:
    normalized = str(symbol).strip().upper()
    for name in (normalized, normalized.lower(), normalized[-6:]):
        path = price_dir / f"{name}.parquet"
        if path.is_file():
            return path
    return None


def _read_membership(
    pit_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    files = sorted((pit_dir / "daily").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"PIT membership partitions are missing: {pit_dir / 'daily'}"
        )
    frame = pd.concat(
        [pd.read_parquet(path, columns=["date", "symbol"]) for path in files],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("PIT membership contains duplicate member-day keys")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _read_prices(price_dir: Path, symbols: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {"date", "open", "close", "high", "low", "volume"}
    for symbol in sorted(symbols):
        path = _symbol_file(price_dir, symbol)
        if path is None:
            continue
        frame = pd.read_parquet(path)
        if not required.issubset(frame.columns):
            continue
        frame = frame.loc[:, ["date", "open", "close", "high", "low", "volume"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for column in ["open", "close", "high", "low", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.sort_values("date").drop_duplicates("date", keep=False)
        frame["previous_close"] = frame["close"].shift(1)
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "previous_close",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _industry_for_members(members: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
    intervals = intervals.copy()
    intervals["symbol"] = intervals["symbol"].astype(str).str.upper()
    intervals["start_date"] = pd.to_datetime(
        intervals["start_date"], errors="coerce"
    ).dt.normalize()
    intervals["end_date"] = pd.to_datetime(
        intervals["end_date"], errors="coerce"
    ).dt.normalize()
    result = pd.Series(index=members.index, dtype="object")
    for symbol, indices in members.groupby("symbol").groups.items():
        ranges = intervals[intervals["symbol"] == symbol].sort_values(
            ["start_date", "end_date"]
        )
        if ranges.empty:
            continue
        starts = ranges["start_date"].reset_index(drop=True)
        ends = ranges["end_date"].reset_index(drop=True)
        if (
            len(ranges) > 1
            and (
                starts.iloc[1:].reset_index(drop=True)
                <= ends.iloc[:-1].reset_index(drop=True)
            ).any()
        ):
            raise ValueError(f"overlapping industry intervals for {symbol}")
        interval_index = pd.IntervalIndex.from_arrays(starts, ends, closed="both")
        member_indices = pd.Index(indices)
        locations = interval_index.get_indexer(members.loc[member_indices, "date"])
        matched = locations >= 0
        if matched.any():
            industry_values = (
                ranges["industry_code"].astype(str).reset_index(drop=True).to_numpy()
            )
            result.loc[member_indices[matched]] = industry_values[locations[matched]]
    return result


def load_market_panel(
    *,
    pit_dir: Path,
    price_dir: Path,
    reference_dir: Path,
    start_date: str,
    end_date: str,
    ohlcv_coverage_min: float = 0.98,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")
    if not 0 < float(ohlcv_coverage_min) <= 1:
        raise ValueError("ohlcv_coverage_min must be in (0, 1]")
    pit_dir = Path(pit_dir)
    with membership_read_lock(pit_dir) as membership_snapshot:
        membership_snapshot.ensure_transaction_clean()
        membership_source_metadata = membership_snapshot.validate_source(
            start_date=str(start.date()),
            end_date=str(end.date()),
        )
        membership_snapshot_hash = membership_snapshot.fingerprint()
        reference_dir = Path(reference_dir)
        with reference_read_lock(reference_dir) as reference_precheck:
            reference_precheck.ensure_transaction_clean()
            reference_precheck.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
                expected_membership_snapshot_hash=membership_snapshot_hash,
            )
        members = _read_membership(pit_dir, start, end)
    prices = _read_prices(Path(price_dir), set(members["symbol"]))
    with reference_read_lock(reference_dir) as reference_snapshot:
        reference_snapshot.ensure_transaction_clean()
        reference_source_metadata = reference_snapshot.validate_source(
            start_date=str(start.date()),
            end_date=str(end.date()),
            expected_membership_snapshot_hash=membership_snapshot_hash,
        )
        security = pd.read_parquet(reference_dir / "security_master.parquet")
        industries = pd.read_parquet(reference_dir / "industry_intervals.parquet")
        st = pd.read_parquet(reference_dir / "st_status.parquet")
        suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
        reference_snapshot_hash = reference_snapshot.fingerprint()

    panel = members.merge(
        prices, on=["date", "symbol"], how="left", validate="one_to_one"
    )
    security["symbol"] = security["symbol"].astype(str).str.upper()
    if security.duplicated("symbol").any():
        raise ValueError("security master contains duplicate symbols")
    panel = panel.merge(
        security.loc[:, ["symbol", "board"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    panel["industry"] = _industry_for_members(panel, industries)

    for frame in (st, suspension):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError("daily reference table contains duplicate member-day keys")
    panel = panel.merge(
        st.loc[:, ["date", "symbol", "is_st"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        suspension.loc[:, ["date", "symbol", "is_suspended"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"is_suspended": "suspended"})
    panel["is_st"] = panel["is_st"].astype("boolean")
    panel["suspended"] = panel["suspended"].astype("boolean")
    metadata_required = ["board", "industry", "is_st", "suspended"]
    if panel[metadata_required].isna().any(axis=None):
        missing = {
            column: int(panel[column].isna().sum())
            for column in metadata_required
            if panel[column].isna().any()
        }
        raise ValueError(f"market panel metadata is incomplete: {missing}")
    price_columns = ["open", "close", "previous_close"]
    panel["price_available"] = panel[price_columns].notna().all(axis=1)
    covered = panel["price_available"] | panel["suspended"].astype(bool)
    price_coverage = float(covered.mean()) if len(panel) else 0.0
    if price_coverage < float(ohlcv_coverage_min):
        missing = {
            column: int(panel.loc[~covered, column].isna().sum())
            for column in price_columns
            if panel.loc[~covered, column].isna().any()
        }
        raise ValueError(
            "market panel price coverage is below the configured gate: "
            f"{price_coverage:.4f} < {float(ohlcv_coverage_min):.4f}; {missing}"
        )
    with membership_read_lock(pit_dir) as final_membership:
        final_membership.ensure_transaction_clean()
        final_membership.validate_source(
            start_date=str(start.date()),
            end_date=str(end.date()),
        )
        final_membership_hash = final_membership.fingerprint()
        with reference_read_lock(reference_dir) as final_reference:
            final_reference.ensure_transaction_clean()
            if final_membership_hash != membership_snapshot_hash:
                raise MembershipIntegrityError(
                    "membership_snapshot_changed",
                    "membership snapshot changed while loading market panel",
                )
            final_reference.validate_source(
                start_date=str(start.date()),
                end_date=str(end.date()),
                expected_membership_snapshot_hash=final_membership_hash,
            )
            final_reference_hash = final_reference.fingerprint()
            if final_reference_hash != reference_snapshot_hash:
                raise ReferenceIntegrityError(
                    "reference_snapshot_changed",
                    "reference snapshot changed while loading market panel",
                )
    result = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    result.attrs["reference_source_metadata"] = reference_source_metadata
    result.attrs["reference_snapshot_hash"] = reference_snapshot_hash
    result.attrs["membership_source_metadata"] = membership_source_metadata
    result.attrs["membership_snapshot_hash"] = membership_snapshot_hash
    result.attrs["ohlcv_coverage"] = price_coverage
    return result


def read_qlib_feature(provider: Path, symbol: str, field: str) -> pd.Series:
    provider = Path(provider)
    calendar_path = provider / "calendars" / "day.txt"
    feature_path = provider / "features" / str(symbol).lower() / f"{field}.day.bin"
    if not calendar_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError(f"Qlib feature is missing: {feature_path}")
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            calendar_path.read_text(encoding="utf-8").splitlines(), errors="coerce"
        )
    )
    raw = np.fromfile(feature_path, dtype="<f4")
    if len(raw) < 2 or not math.isfinite(float(raw[0])):
        raise ValueError(f"invalid Qlib binary feature: {feature_path}")
    start = int(raw[0])
    values = raw[1:]
    end = start + len(values)
    if start < 0 or end > len(calendar):
        raise ValueError(f"Qlib feature offset is outside calendar: {feature_path}")
    return pd.Series(values.astype(float), index=calendar[start:end], name=field)


def load_benchmark_frame(
    provider: Path, *, start_date: str, end_date: str, symbol: str = "SH000300"
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    fields = {
        field: read_qlib_feature(provider, symbol, field) for field in ["open", "close"]
    }
    frame = pd.concat(fields, axis=1).reset_index(names="date")
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(
        drop=True
    )


def build_market_regimes(panel: pd.DataFrame) -> pd.Series:
    if not {"date", "symbol", "close"}.issubset(panel.columns):
        raise ValueError("market panel must contain date, symbol, and close")
    frame = panel.loc[:, ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    matrix = frame.pivot(index="date", columns="symbol", values="close").sort_index()
    stock_returns = matrix.pct_change(fill_method=None)
    market_returns = stock_returns.mean(axis=1, skipna=True).fillna(0.0)
    market_index = (1.0 + market_returns).cumprod()
    metrics = pd.DataFrame(index=matrix.index)
    metrics["return_20d"] = market_index / market_index.shift(20) - 1.0
    metrics["return_60d"] = market_index / market_index.shift(60) - 1.0
    metrics["breadth_20d"] = (matrix / matrix.shift(20) - 1.0).gt(0).mean(axis=1)
    metrics["breadth_60d"] = (matrix / matrix.shift(60) - 1.0).gt(0).mean(axis=1)
    metrics["drawdown"] = market_index / market_index.cummax() - 1.0
    metrics["volatility_20d"] = market_returns.rolling(20, min_periods=2).std()
    regimes: list[str] = []
    for _, row in metrics.iterrows():
        payload = {
            key: (None if pd.isna(value) else float(value))
            for key, value in row.items()
        }
        regime, _, _ = classify_regime(payload)
        regimes.append(regime)
    return pd.Series(regimes, index=metrics.index, name="regime")
