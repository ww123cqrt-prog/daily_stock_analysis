"""Point-in-time primitives for offline A-share limit-up research.

The module is intentionally broker-free.  It reconstructs 14:00 and 14:30
research snapshots from immutable minute bars and keeps all post-decision
information outside the feature payload.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from limit_up_intraday_contract import (  # noqa: E402
    PRICE_TOLERANCE,
    normalize_minute_bars,
)
from limit_up_intraday_data import (  # noqa: E402
    FORBIDDEN_MODEL_FEATURES,
    resolve_snapshot_member,
    validate_intraday_snapshot,
)


DECISION_TIMES = frozenset({"14:00", "14:30"})
LOCKED_CONFIRMATION_START = pd.Timestamp("2026-01-01")
MODEL_FEATURE_COLUMNS = (
    "first_limit_minute_index",
    "break_count",
    "minutes_at_limit",
    "cumulative_volume_log",
    "cumulative_money_log",
    "turnover_rate_at_decision",
    "momentum_5d",
    "momentum_20d",
    "consecutive_limit_days",
    "industry_heat",
    "sealed_count",
    "broken_board_rate",
    "max_consecutive_limit_days",
    "index_trend_20d",
    "advance_decline_ratio",
    "turnover_activity",
)


def _cutoff(frame: pd.DataFrame, decision_time: str) -> pd.Timestamp:
    if decision_time not in DECISION_TIMES:
        raise ValueError("decision_time must be 14:00 or 14:30")
    day = pd.Timestamp(frame.iloc[0]["datetime"]).normalize()
    return day + pd.Timedelta(
        hours=int(decision_time[:2]), minutes=int(decision_time[3:])
    )


def _at_limit(values: pd.Series, limits: pd.Series) -> pd.Series:
    return pd.Series(
        np.isclose(
            values.to_numpy(dtype=float),
            limits.to_numpy(dtype=float),
            rtol=0.0,
            atol=PRICE_TOLERANCE,
        ),
        index=values.index,
    )


def build_decision_snapshot(
    bars: pd.DataFrame, *, decision_time: str
) -> dict[str, object]:
    """Build features from rows timestamped no later than the decision minute."""

    frame = normalize_minute_bars(bars)
    cutoff = _cutoff(frame, decision_time)
    visible = frame.loc[frame["datetime"] <= cutoff].copy()
    if visible.empty or pd.Timestamp(visible.iloc[-1]["datetime"]) != cutoff:
        raise ValueError(f"minute bars do not contain the {decision_time} decision row")

    close_at_limit = _at_limit(visible["close"], visible["high_limit"])
    high_touched = visible["high"] >= visible["high_limit"] - PRICE_TOLERANCE
    touched_rows = visible.loc[high_touched, "datetime"]
    breaks = close_at_limit.shift(1, fill_value=False) & ~close_at_limit
    last = visible.iloc[-1]
    return {
        "decision_time": decision_time,
        "decision_timestamp": cutoff,
        "is_candidate": bool(close_at_limit.iloc[-1]),
        "decision_close": float(last["close"]),
        "limit_price": float(last["high_limit"]),
        "pre_close": float(last["pre_close"]),
        "first_limit_time": (
            pd.Timestamp(touched_rows.iloc[0]) if not touched_rows.empty else None
        ),
        "break_count": int(breaks.sum()),
        "minutes_at_limit": int(close_at_limit.sum()),
        "cumulative_volume": float(visible["volume"].sum()),
        "cumulative_money": float(visible["money"].sum()),
        "decision_return": float(last["close"] / last["pre_close"] - 1.0),
        "visible_minute_count": int(len(visible)),
    }


def post_decision_board_opened(
    bars: pd.DataFrame, *, decision_time: str
) -> bool:
    """Return whether a price below the limit was observable after decision."""

    frame = normalize_minute_bars(bars)
    cutoff = _cutoff(frame, decision_time)
    future = frame.loc[frame["datetime"] > cutoff]
    if future.empty:
        return False
    return bool(
        (future["low"] < future["high_limit"] - PRICE_TOLERANCE).any()
    )


def audit_candidate_recall(
    *, expected: Iterable[str], recovered: Iterable[str]
) -> dict[str, object]:
    """Require the retrieval index to contain every point-in-time candidate."""

    expected_set = {str(symbol) for symbol in expected}
    recovered_set = {str(symbol) for symbol in recovered}
    missing = sorted(expected_set - recovered_set)
    matched = len(expected_set & recovered_set)
    recall = matched / len(expected_set) if expected_set else 1.0
    return {
        "status": "blocked_data" if missing else "passed",
        "expected_count": len(expected_set),
        "recovered_count": matched,
        "recall": float(recall),
        "missing_symbols": missing,
    }


def assert_research_date_unlocked(value: object) -> None:
    """Reject the locked 2026 confirmation period from model research."""

    date = pd.Timestamp(value).normalize()
    if pd.isna(date):
        raise ValueError("research date is invalid")
    if date >= LOCKED_CONFIRMATION_START:
        raise ValueError("2026 locked confirmation data cannot enter research")


def _transfer_fee_bps(value: object) -> float:
    date = pd.Timestamp(value).normalize()
    if date < pd.Timestamp("2018-01-01"):
        raise ValueError("transfer-fee schedule is registered from 2018 onward")
    return 0.1 if date >= pd.Timestamp("2022-04-29") else 0.2


def _stamp_duty_bps(value: object) -> float:
    date = pd.Timestamp(value).normalize()
    if date < pd.Timestamp("2018-01-01"):
        raise ValueError("stamp-duty schedule is registered from 2018 onward")
    return 5.0 if date >= pd.Timestamp("2023-08-28") else 10.0


def stock_round_trip_costs(
    *,
    buy_date: object,
    sell_date: object,
    commission_bps_each_side: float = 0.754,
    sell_slippage_bps: float = 0.0,
) -> dict[str, float]:
    """Return registered A-share costs without a minimum commission."""

    commission = float(commission_bps_each_side)
    slippage = float(sell_slippage_bps)
    if commission < 0 or slippage < 0:
        raise ValueError("commission and slippage must be nonnegative")
    buy_transfer = _transfer_fee_bps(buy_date)
    sell_transfer = _transfer_fee_bps(sell_date)
    stamp = _stamp_duty_bps(sell_date)
    buy_bps = commission + buy_transfer
    sell_bps = commission + sell_transfer + stamp + slippage
    return {
        "commission_bps_each_side": commission,
        "buy_transfer_bps": buy_transfer,
        "sell_transfer_bps": sell_transfer,
        "stamp_duty_bps": stamp,
        "sell_slippage_bps": slippage,
        "buy_cost_rate": buy_bps / 10_000.0,
        "sell_cost_rate": sell_bps / 10_000.0,
        "round_trip_cost_bps": buy_bps + sell_bps,
    }


def build_fill_layers(
    bars: pd.DataFrame,
    *,
    decision_time: str,
    order_value: float,
    capacity_participation_rates: tuple[float, ...] = (0.001, 0.005, 0.01),
) -> dict[str, dict[str, object]]:
    """Separate signal, observable board-open, and turnover-capacity layers."""

    value = float(order_value)
    if value <= 0:
        raise ValueError("order_value must be positive")
    frame = normalize_minute_bars(bars)
    cutoff = _cutoff(frame, decision_time)
    snapshot = build_decision_snapshot(frame, decision_time=decision_time)
    candidate = bool(snapshot["is_candidate"])
    future_turnover = float(frame.loc[frame["datetime"] > cutoff, "money"].sum())
    result: dict[str, dict[str, object]] = {
        "theoretical_signal": {
            "filled": candidate,
            "definition": "assume_limit_price_fill",
        },
        "board_open_verified": {
            "filled": candidate
            and post_decision_board_opened(frame, decision_time=decision_time),
            "definition": "post_decision_price_below_limit_observed",
        },
    }
    for rate in capacity_participation_rates:
        participation = float(rate)
        if participation <= 0 or participation > 1:
            raise ValueError("capacity participation rates must be in (0, 1]")
        capacity = future_turnover * participation
        label = f"capacity_proxy_{participation * 100:g}pct"
        result[label] = {
            "filled": candidate and capacity >= value,
            "definition": "post_decision_turnover_capacity_proxy",
            "participation_rate": participation,
            "post_decision_turnover": future_turnover,
            "capacity_value": capacity,
            "is_real_fill_probability": False,
        }
    return result


def trade_metrics(returns: Iterable[float]) -> dict[str, float | int]:
    """Calculate overnight trade metrics without annualizing sparse events."""

    values = pd.Series(list(returns), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "trade_count": 0,
            "expectancy": float("nan"),
            "profit_factor": float("nan"),
            "realized_win_rate": float("nan"),
            "average_win": float("nan"),
            "average_loss": float("nan"),
            "average_win_loss_ratio": float("nan"),
            "tail_loss_5pct": float("nan"),
            "max_consecutive_losses": 0,
            "max_drawdown": float("nan"),
        }
    wins = values.loc[values > 0]
    losses = values.loc[values < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0
    loss_abs = abs(average_loss)
    losing_streak = 0
    max_losing_streak = 0
    for value in values:
        losing_streak = losing_streak + 1 if value < 0 else 0
        max_losing_streak = max(max_losing_streak, losing_streak)
    wealth = pd.concat([pd.Series([1.0]), (1.0 + values).cumprod()], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "trade_count": int(len(values)),
        "expectancy": float(values.mean()),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else float("nan"))
        ),
        "realized_win_rate": float((values > 0).mean()),
        "average_win": average_win,
        "average_loss": average_loss,
        "average_win_loss_ratio": (
            average_win / loss_abs
            if loss_abs > 0
            else (float("inf") if average_win > 0 else float("nan"))
        ),
        "tail_loss_5pct": float(values.quantile(0.05)),
        "max_consecutive_losses": int(max_losing_streak),
        "max_drawdown": float(drawdown.min()),
    }


def grade_signal(
    *,
    expectancy: float,
    expectancy_bootstrap_lower: float,
    profit_factor: float,
    max_drawdown: float,
    positive_quarter_fraction: float,
    regime_positive_count: int,
    regime_count: int,
    realized_win_rate: float,
) -> dict[str, object]:
    """Apply the preregistered evidence grades and retain all diagnostics."""

    base_pass = (
        float(expectancy) > 0
        and float(profit_factor) > 1.0
        and abs(float(max_drawdown)) <= 0.35
    )
    strong = (
        base_pass
        and float(expectancy_bootstrap_lower) > 0
        and float(profit_factor) > 1.2
        and float(positive_quarter_fraction) >= 0.6333
    )
    status = (
        "strong_signal"
        if strong
        else "potential_signal"
        if base_pass
        else "signal_not_found"
    )
    regime_concentrated = int(regime_count) > 1 and int(regime_positive_count) == 1
    return {
        "status": status,
        "regime_concentrated": regime_concentrated,
        "checks": {
            "expectancy_positive": float(expectancy) > 0,
            "bootstrap_lower_positive": float(expectancy_bootstrap_lower) > 0,
            "profit_factor_above_1": float(profit_factor) > 1.0,
            "profit_factor_above_1_2": float(profit_factor) > 1.2,
            "max_drawdown_within_35pct": abs(float(max_drawdown)) <= 0.35,
            "positive_quarter_fraction_at_least_63_33pct": float(
                positive_quarter_fraction
            )
            >= 0.6333,
        },
        "diagnostics": {"realized_win_rate": float(realized_win_rate)},
    }


def model_feature_columns() -> list[str]:
    """Return the frozen numeric feature contract in model input order."""

    forbidden = set(FORBIDDEN_MODEL_FEATURES)
    if forbidden.intersection(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("forbidden retrieval or label fields entered model features")
    return list(MODEL_FEATURE_COLUMNS)


def preregistered_experiment_specs() -> list[dict[str, str]]:
    """Return the four tracks in their frozen reporting order."""

    return [
        {
            "experiment_id": f"limit_up_{decision.replace(':', '')}_{exit_at}",
            "decision_time": decision,
            "exit_at": exit_at,
        }
        for decision in ("14:00", "14:30")
        for exit_at in ("next_open", "next_close")
    ]


def _net_trade_return(
    *,
    entry_price: float,
    exit_price: float,
    buy_date: object,
    sell_date: object,
    commission_bps_each_side: float,
    sell_slippage_bps: float,
) -> float:
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("entry and exit prices must be positive")
    costs = stock_round_trip_costs(
        buy_date=buy_date,
        sell_date=sell_date,
        commission_bps_each_side=commission_bps_each_side,
        sell_slippage_bps=sell_slippage_bps,
    )
    return (
        float(exit_price)
        * (1.0 - float(costs["sell_cost_rate"]))
        / (float(entry_price) * (1.0 + float(costs["buy_cost_rate"])))
        - 1.0
    )


def _first_limit_index(bars: pd.DataFrame, timestamp: object) -> int:
    if timestamp is None or pd.isna(timestamp):
        return -1
    target = pd.Timestamp(timestamp)
    matches = bars.index[bars["datetime"] == target]
    return int(matches[0]) + 1 if len(matches) else -1


def build_research_rows(
    snapshot: Path,
    *,
    decision_time: str,
    commission_bps_each_side: float = 0.754,
    sell_slippage_bps: tuple[float, ...] = (0.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Build one point-in-time candidate table from a validated snapshot."""

    if decision_time not in DECISION_TIMES:
        raise ValueError("decision_time must be 14:00 or 14:30")
    root = Path(snapshot).resolve()
    manifest = validate_intraday_snapshot(root, verify_hashes=True)
    forbidden = set(str(item) for item in manifest.get("forbidden_model_features", []))
    if not set(FORBIDDEN_MODEL_FEATURES).issubset(forbidden):
        raise ValueError("snapshot does not freeze the forbidden model fields")

    events = pd.read_parquet(resolve_snapshot_member(root, manifest["events_path"]))
    required_event_fields = {
        "symbol",
        "date",
        "relative_path",
        "listing_age_sessions",
        "is_st",
        "board",
        "free_float_shares",
        "momentum_5d",
        "momentum_20d",
        "consecutive_limit_days",
        "industry_heat",
        "next_date",
        "next_open",
        "next_close",
    }
    missing = required_event_fields - set(events.columns)
    if missing:
        raise ValueError(f"events are missing research fields: {', '.join(sorted(missing))}")
    states = pd.read_parquet(resolve_snapshot_member(root, manifest["market_state_path"]))
    states["date"] = pd.to_datetime(states["date"], errors="coerce").dt.normalize()
    state_rows = states.loc[states["decision_time"].astype(str) == decision_time]
    state_by_date = state_rows.set_index("date", verify_integrity=True)

    rows: list[dict[str, object]] = []
    for event in events.sort_values(["date", "symbol"], kind="stable").itertuples(
        index=False
    ):
        date = pd.Timestamp(event.date).normalize()
        assert_research_date_unlocked(date)
        if bool(event.is_st):
            continue
        if str(event.board) not in {"main", "chinext", "star"}:
            continue
        if int(event.listing_age_sessions) < 20:
            continue
        if date not in state_by_date.index:
            raise ValueError(f"market state is missing for {date.date()} {decision_time}")
        bars = normalize_minute_bars(
            pd.read_parquet(resolve_snapshot_member(root, event.relative_path))
        )
        point = build_decision_snapshot(bars, decision_time=decision_time)
        if not bool(point["is_candidate"]):
            continue
        state = state_by_date.loc[date]
        free_float_shares = float(event.free_float_shares)
        if free_float_shares <= 0:
            raise ValueError("free_float_shares must be positive")
        entry_price = float(point["limit_price"])
        next_date = pd.Timestamp(event.next_date).normalize()
        row: dict[str, object] = {
            "symbol": str(event.symbol),
            "date": date,
            "decision_time": decision_time,
            "entry_price": entry_price,
            "next_date": next_date,
            "regime": str(state["regime"]),
            "first_limit_minute_index": float(
                _first_limit_index(bars, point["first_limit_time"])
            ),
            "break_count": float(point["break_count"]),
            "minutes_at_limit": float(point["minutes_at_limit"]),
            "cumulative_volume_log": float(np.log1p(point["cumulative_volume"])),
            "cumulative_money_log": float(np.log1p(point["cumulative_money"])),
            "turnover_rate_at_decision": float(point["cumulative_volume"])
            / free_float_shares,
            "momentum_5d": float(event.momentum_5d),
            "momentum_20d": float(event.momentum_20d),
            "consecutive_limit_days": float(event.consecutive_limit_days),
            "industry_heat": float(event.industry_heat),
            "sealed_count": float(state["sealed_count"]),
            "broken_board_rate": float(state["broken_board_rate"]),
            "max_consecutive_limit_days": float(
                state["max_consecutive_limit_days"]
            ),
            "index_trend_20d": float(state["index_trend_20d"]),
            "advance_decline_ratio": float(state["advance_decline_ratio"]),
            "turnover_activity": float(state["turnover_activity"]),
            "board_open_verified": post_decision_board_opened(
                bars, decision_time=decision_time
            ),
            "post_decision_turnover": float(
                bars.loc[
                    bars["datetime"] > pd.Timestamp(point["decision_timestamp"]),
                    "money",
                ].sum()
            ),
            "gross_next_open_return": float(event.next_open) / entry_price - 1.0,
            "gross_next_close_return": float(event.next_close) / entry_price - 1.0,
        }
        for exit_name in ("open", "close"):
            exit_price = float(getattr(event, f"next_{exit_name}"))
            for slippage in sell_slippage_bps:
                key = f"net_next_{exit_name}_return_slippage_{float(slippage):g}"
                row[key] = _net_trade_return(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    buy_date=date,
                    sell_date=next_date,
                    commission_bps_each_side=commission_bps_each_side,
                    sell_slippage_bps=float(slippage),
                )
        rows.append(row)

    columns = [
        "symbol",
        "date",
        "decision_time",
        "entry_price",
        "next_date",
        "regime",
        *MODEL_FEATURE_COLUMNS,
        "board_open_verified",
        "post_decision_turnover",
        "gross_next_open_return",
        "gross_next_close_return",
    ]
    for exit_name in ("open", "close"):
        columns.extend(
            f"net_next_{exit_name}_return_slippage_{float(value):g}"
            for value in sell_slippage_bps
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result = result.loc[:, columns].sort_values(["date", "symbol"], kind="stable")
    if result.loc[:, model_feature_columns()].isna().any().any():
        raise ValueError("research features contain missing values")
    return result.reset_index(drop=True)


def size_topk_orders(
    predictions: pd.DataFrame,
    *,
    capital: float = 300_000.0,
    topk: int = 3,
    lot_size: int = 100,
    probability_threshold: float = 0.55,
) -> pd.DataFrame:
    """Select eligible predictions and size equal-weight integer-lot orders."""

    required = {
        "symbol",
        "calibrated_probability",
        "predicted_net_return",
        "entry_price",
    }
    if not required.issubset(predictions.columns):
        raise ValueError("predictions are missing order-sizing fields")
    if capital <= 0 or topk <= 0 or lot_size <= 0:
        raise ValueError("capital, topk, and lot_size must be positive")
    eligible = predictions.loc[
        (pd.to_numeric(predictions["calibrated_probability"], errors="coerce") >= probability_threshold)
        & (pd.to_numeric(predictions["predicted_net_return"], errors="coerce") > 0)
        & (pd.to_numeric(predictions["entry_price"], errors="coerce") > 0)
    ].copy()
    eligible = eligible.sort_values(
        ["calibrated_probability", "predicted_net_return", "symbol"],
        ascending=[False, False, True],
        kind="stable",
    ).head(int(topk))
    if eligible.empty:
        return eligible.assign(
            rank=pd.Series(dtype=int),
            target_weight=pd.Series(dtype=float),
            shares=pd.Series(dtype=int),
            order_value=pd.Series(dtype=float),
        )
    count = len(eligible)
    target_weight = 1.0 / count
    budget = float(capital) * target_weight
    prices = pd.to_numeric(eligible["entry_price"], errors="raise").astype(float)
    shares = np.floor(budget / prices / int(lot_size)).astype(int) * int(lot_size)
    eligible["rank"] = np.arange(1, count + 1, dtype=int)
    eligible["target_weight"] = target_weight
    eligible["shares"] = shares
    eligible["order_value"] = shares * prices
    return eligible.loc[eligible["shares"] > 0].reset_index(drop=True)
