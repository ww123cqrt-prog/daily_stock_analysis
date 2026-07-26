"""Pure implementation of the two-ETF 20-day momentum rotation rule.

The strategy is intentionally separate from CSI800 model research.  A signal is
computed from the close of day T and becomes a position at the next trading
day's open.  The module contains no broker, network, or order-submission code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


CASH = "CASH"
DEFAULT_SYMBOLS: tuple[str, ...] = ("SZ159915", "SH513100")
REQUIRED_COLUMNS = {"date", "open", "close"}
# This is the research-candidate gate for the isolated ETF rule.  It does not
# relax the separate CSI800 robust-strategy gates.
ETF_RESEARCH_MAX_DRAWDOWN_ABS = 0.35


def normalize_price_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Normalize a source frame and reject duplicate or unusable bars."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} is missing columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in ("open", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "close"])
    result = result[(result["open"] > 0) & (result["close"] > 0)]
    result = result.sort_values("date", kind="stable").reset_index(drop=True)
    if result["date"].duplicated().any():
        duplicates = (
            result.loc[result["date"].duplicated(), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(f"{symbol} contains duplicate dates: {duplicates[:5]}")
    if result.empty:
        raise ValueError(f"{symbol} has no usable price rows")
    return result


def build_signal_table(
    data: Mapping[str, pd.DataFrame],
    *,
    lookback_days: int = 20,
    moving_average_days: int = 20,
) -> pd.DataFrame:
    """Build close-time ranks and the next-open target signal.

    ``signal`` is the position selected using information available at that
    row's close. ``target_position`` is shifted by one common trading day and
    is the position held after the next open. Missing or incomplete windows
    deliberately produce CASH.
    """

    if lookback_days <= 0 or moving_average_days <= 0:
        raise ValueError("lookback_days and moving_average_days must be positive")
    if len(data) != 2:
        raise ValueError("the strategy requires exactly two ETF price series")

    normalized = {
        symbol: normalize_price_frame(frame, symbol=symbol)
        for symbol, frame in sorted(data.items())
    }
    # Preserve the caller's explicit order so equal 20-day returns have a
    # deterministic, documented tie-break rather than depending on sorting.
    symbols = tuple(data)
    merged: pd.DataFrame | None = None
    for symbol in symbols:
        frame = normalized[symbol][["date", "open", "close"]].rename(
            columns={"open": f"{symbol}_open", "close": f"{symbol}_close"}
        )
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on="date", how="inner", validate="one_to_one")
        )
    assert merged is not None
    merged = merged.sort_values("date", kind="stable").reset_index(drop=True)

    returns: dict[str, pd.Series] = {}
    moving_averages: dict[str, pd.Series] = {}
    for symbol in symbols:
        close = merged[f"{symbol}_close"]
        returns[symbol] = close / close.shift(lookback_days) - 1.0
        moving_averages[symbol] = close.rolling(
            moving_average_days, min_periods=moving_average_days
        ).mean()
        merged[f"{symbol}_return_{lookback_days}d"] = returns[symbol]
        merged[f"{symbol}_ma_{moving_average_days}"] = moving_averages[symbol]
        merged[f"{symbol}_above_ma"] = close > moving_averages[symbol]

    return_frame = pd.DataFrame(returns)
    rank_frame = return_frame.rank(axis=1, ascending=False, method="first")
    for symbol in symbols:
        merged[f"{symbol}_rank"] = rank_frame[symbol]

    has_return = return_frame.notna().any(axis=1)
    top_symbol = return_frame.fillna(-np.inf).idxmax(axis=1).where(has_return)
    eligible = pd.Series(False, index=merged.index)
    for symbol in symbols:
        eligible |= (top_symbol == symbol) & merged[f"{symbol}_above_ma"].fillna(False)
    signal = top_symbol.where(eligible, CASH).fillna(CASH).astype(str)
    merged["top_symbol"] = top_symbol.fillna(CASH).astype(str)
    merged["signal"] = signal
    merged["target_position"] = signal.shift(1).fillna(CASH).astype(str)
    return merged


def with_execution_lag(signal_table: pd.DataFrame, *, lag_days: int) -> pd.DataFrame:
    """Return a copy whose position is executed ``lag_days`` sessions later.

    ``signal`` is generated at the close of day T.  The normal rule uses
    ``lag_days=1`` (the next open); ``lag_days=2`` is the one-session delay
    stress test.  The helper intentionally shifts the signal, rather than an
    already-shifted target, so the delay remains explicit and auditable.
    """

    if lag_days < 1:
        raise ValueError("lag_days must be at least one trading day")
    if "signal" not in signal_table.columns:
        raise ValueError("signal_table must contain signal")
    result = signal_table.copy()
    result["target_position"] = (
        result["signal"].shift(lag_days).fillna(CASH).astype(str)
    )
    return result


def summarize_periods(
    periods: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, object]:
    """Summarize complete open-to-open periods inside an inclusive date range.

    A row dated T represents the return from T's open to ``next_date``'s open.
    The next date must also remain inside the requested range, which prevents a
    quarter report from borrowing the first return of the following quarter.
    """

    required = {"date", "next_date", "net_return", "wealth"}
    if not required.issubset(periods.columns):
        raise ValueError("periods must contain date, next_date, net_return, and wealth")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must not be after end_date")

    frame = periods.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["next_date"] = pd.to_datetime(
        frame["next_date"], errors="coerce"
    ).dt.normalize()
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
    frame["wealth"] = pd.to_numeric(frame["wealth"], errors="coerce")
    frame = frame.loc[
        (frame["date"] >= start)
        & (frame["next_date"] <= end)
        & frame["net_return"].notna()
        & frame["wealth"].notna()
    ].copy()
    if frame.empty:
        return {
            "periods": 0,
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "max_drawdown": float("nan"),
        }

    returns = frame["net_return"].astype(float)
    wealth = frame["wealth"].astype(float)
    normalized_wealth = wealth / float(wealth.iloc[0] / (1.0 + float(returns.iloc[0])))
    drawdown = normalized_wealth / normalized_wealth.cummax() - 1.0
    total_return = float((1.0 + returns).prod() - 1.0)
    return {
        "periods": int(len(frame)),
        "start_date": str(pd.Timestamp(frame.iloc[0]["date"]).date()),
        "end_date": str(pd.Timestamp(frame.iloc[-1]["next_date"]).date()),
        "total_return": total_return,
        "annualized_return": _annualized(total_return, len(frame)),
        "max_drawdown": float(drawdown.min()),
    }


def _period_return(
    frame: pd.DataFrame, position: str, index: int, next_index: int
) -> float:
    if position == CASH:
        return 0.0
    open_now = float(frame.iloc[index][f"{position}_open"])
    open_next = float(frame.iloc[next_index][f"{position}_open"])
    return open_next / open_now - 1.0


def _annualized(total_return: float, periods: int) -> float:
    if periods <= 0:
        return float("nan")
    wealth = 1.0 + total_return
    if wealth <= 0:
        return -1.0
    return wealth ** (252.0 / periods) - 1.0


def simulate_open_to_open(
    signal_table: pd.DataFrame,
    *,
    side_cost_bps: float = 0.0,
) -> dict[str, object]:
    """Simulate next-open execution with optional per-side cost.

    The cost is a combined commission and slippage allowance per buy or sell.
    ETF stamp duty is not charged.  A switch from ETF A to ETF B therefore
    pays two sides; entering or leaving cash pays one side.
    """

    if side_cost_bps < 0:
        raise ValueError("side_cost_bps must be nonnegative")
    required = {"date", "target_position"}
    if not required.issubset(signal_table.columns):
        raise ValueError("signal_table must contain date and target_position")
    frame = signal_table.sort_values("date", kind="stable").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("at least two common trading days are required")

    wealth = 1.0
    previous_position = CASH
    curve: list[dict[str, object]] = [{"date": frame.iloc[0]["date"], "wealth": wealth}]
    total_cost = 0.0
    total_turnover = 0.0
    switches = 0
    invested_periods = 0
    period_rows: list[dict[str, object]] = []
    side_rate = side_cost_bps / 10000.0

    for index in range(len(frame) - 1):
        row = frame.iloc[index]
        next_row = frame.iloc[index + 1]
        position = str(row["target_position"])
        sides = (
            (int(previous_position != CASH) + int(position != CASH))
            if position != previous_position
            else 0
        )
        if position != previous_position:
            switches += 1
        turnover = float(sides)
        cost_rate = turnover * side_rate
        gross_return = _period_return(frame, position, index, index + 1)
        wealth_before = wealth
        wealth = wealth * (1.0 - cost_rate) * (1.0 + gross_return)
        net_return = wealth / wealth_before - 1.0
        total_cost += cost_rate
        total_turnover += turnover
        if position != CASH:
            invested_periods += 1
        period_rows.append(
            {
                "date": row["date"],
                "next_date": next_row["date"],
                "position": position,
                "previous_position": previous_position,
                "gross_return": gross_return,
                "transaction_cost_rate": cost_rate,
                "net_return": net_return,
                "wealth": wealth,
            }
        )
        curve.append({"date": next_row["date"], "wealth": wealth})
        previous_position = position

    curve_frame = pd.DataFrame(curve)
    running_max = curve_frame["wealth"].cummax()
    drawdown = curve_frame["wealth"] / running_max - 1.0
    period_frame = pd.DataFrame(period_rows)
    total_return = float(wealth - 1.0)
    periods = len(period_frame)
    return {
        "metrics": {
            "periods": periods,
            "start_date": str(pd.Timestamp(curve_frame.iloc[0]["date"]).date()),
            "end_date": str(pd.Timestamp(curve_frame.iloc[-1]["date"]).date()),
            "total_return": total_return,
            "annualized_return": _annualized(total_return, periods),
            "max_drawdown": float(drawdown.min()),
            "invested_fraction": invested_periods / periods
            if periods
            else float("nan"),
            "switch_count": switches,
            "total_turnover_sides": total_turnover,
            "total_transaction_cost_fraction": total_cost,
            "side_cost_bps": side_cost_bps,
        },
        "periods": period_frame,
        "equity_curve": curve_frame,
    }


def compare_claim(
    actual: Mapping[str, object],
    *,
    claimed_total_return: float = 5.228,
    claimed_annualized_return: float = 0.3153,
    claimed_max_drawdown: float = -0.2053,
) -> dict[str, object]:
    """Compare a result with the user's unverified public claim."""

    metrics = actual["metrics"]
    assert isinstance(metrics, Mapping)
    return {
        "claim_status": "unverified_user_supplied_claim",
        "claimed": {
            "total_return": claimed_total_return,
            "annualized_return": claimed_annualized_return,
            "max_drawdown": claimed_max_drawdown,
        },
        "actual": {
            "total_return": float(metrics["total_return"]),
            "annualized_return": float(metrics["annualized_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
        },
        "difference": {
            "total_return": float(metrics["total_return"]) - claimed_total_return,
            "annualized_return": float(metrics["annualized_return"])
            - claimed_annualized_return,
            "max_drawdown": float(metrics["max_drawdown"]) - claimed_max_drawdown,
        },
    }


def load_price_data(
    data_dir: Path, symbols: tuple[str, ...] = DEFAULT_SYMBOLS
) -> dict[str, pd.DataFrame]:
    """Load the isolated ETF parquet files written by the downloader."""

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"ETF data is missing: {path}")
        result[symbol] = normalize_price_frame(pd.read_parquet(path), symbol=symbol)
    return result


def sanitize_json_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value
