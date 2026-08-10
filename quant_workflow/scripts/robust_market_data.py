"""Load PIT member-day market data and build no-lookahead market regimes."""

from __future__ import annotations

import sys
from pathlib import Path
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
from _paths import init_qlib_workflow, provider_calendar_days  # noqa: E402


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
        traded = frame[["open", "close"]].notna().all(axis=1)
        frame["listing_trading_day"] = traded.cumsum().where(traded).astype("Int64")
        valid_trade_dates = frame.loc[traded, "date"]
        frame["price_history_start"] = (
            valid_trade_dates.iloc[0] if not valid_trade_dates.empty else pd.NaT
        )
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
                "listing_trading_day",
                "price_history_start",
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
    security["list_date"] = pd.to_datetime(
        security["list_date"], errors="coerce"
    ).dt.normalize()
    if security.duplicated("symbol").any():
        raise ValueError("security master contains duplicate symbols")
    panel = panel.merge(
        security.loc[:, ["symbol", "list_date", "board"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    panel["industry"] = _industry_for_members(panel, industries)
    panel["price_history_start"] = pd.to_datetime(
        panel["price_history_start"], errors="coerce"
    ).dt.normalize()
    history_gap_days = (panel["price_history_start"] - panel["list_date"]).dt.days
    invalid_history_order = history_gap_days < 0
    incomplete_near_listing = history_gap_days.between(1, 29, inclusive="both")
    if invalid_history_order.any() or incomplete_near_listing.any():
        raise ValueError("price history start does not match the security listing date")
    panel["listing_history_complete"] = history_gap_days.eq(0)
    mature_truncated_history = history_gap_days >= 30
    panel.loc[mature_truncated_history, "listing_trading_day"] = (
        panel.loc[mature_truncated_history, "listing_trading_day"]
        .clip(lower=6)
        .astype("Int64")
    )

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
    metadata_required = ["list_date", "board", "industry", "is_st", "suspended"]
    if panel[metadata_required].isna().any(axis=None):
        missing = {
            column: int(panel[column].isna().sum())
            for column in metadata_required
            if panel[column].isna().any()
        }
        raise ValueError(f"market panel metadata is incomplete: {missing}")
    price_columns = ["open", "close", "previous_close"]
    panel["price_available"] = panel[price_columns].notna().all(axis=1)
    missing_listing_age = panel["price_available"] & panel["listing_trading_day"].isna()
    if missing_listing_age.any():
        raise ValueError(
            "market panel listing trading-day age is missing for available prices"
        )
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
    calendar = provider_calendar_days(provider)
    if not calendar:
        raise FileNotFoundError("Qlib Provider calendar is unavailable")
    init_qlib_workflow(provider_uri=provider)

    from qlib.data import D

    qlib_field = f"${str(field).lstrip('$')}"
    frame = D.features(
        [str(symbol).upper()],
        [qlib_field],
        start_time=calendar[0],
        end_time=calendar[-1],
        freq="day",
    )
    if not isinstance(frame, pd.DataFrame) or qlib_field not in frame.columns:
        raise FileNotFoundError(
            f"Qlib Provider feature is unavailable: {symbol} {qlib_field}"
        )
    if isinstance(frame.index, pd.MultiIndex):
        level = "datetime" if "datetime" in frame.index.names else frame.index.nlevels - 1
        dates = pd.DatetimeIndex(
            pd.to_datetime(frame.index.get_level_values(level), errors="coerce")
        )
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    if dates.hasnans or dates.duplicated().any():
        raise ValueError(f"Qlib Provider feature dates are invalid: {symbol} {qlib_field}")
    values = pd.to_numeric(frame[qlib_field], errors="coerce").to_numpy(dtype=float)
    return pd.Series(values, index=dates, name=str(field).lstrip("$"))


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
    returns_20d = matrix / matrix.shift(20) - 1.0
    returns_60d = matrix / matrix.shift(60) - 1.0
    metrics["breadth_20d"] = (
        returns_20d.gt(0).astype(float).where(returns_20d.notna()).mean(axis=1)
    )
    metrics["breadth_60d"] = (
        returns_60d.gt(0).astype(float).where(returns_60d.notna()).mean(axis=1)
    )
    trailing_peak = market_index.rolling(252, min_periods=1).max()
    metrics["drawdown"] = market_index / trailing_peak - 1.0
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
