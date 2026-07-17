"""Pure, testable rules shared by robust training and portfolio simulation."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd


class RollingSplit(NamedTuple):
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _clean_calendar(calendar: Sequence[Any]) -> pd.DatetimeIndex:
    values = (
        pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )
    if values.empty:
        raise ValueError("calendar must contain trading days")
    return values


def make_rolling_split(
    calendar: Sequence[Any],
    test_quarter: str,
    *,
    train_days: int,
    validation_days: int = 126,
    label_horizon_days: int = 5,
) -> RollingSplit:
    days = _clean_calendar(calendar)
    if train_days <= 0 or validation_days <= 0 or label_horizon_days < 0:
        raise ValueError(
            "train_days and validation_days must be positive; label_horizon_days must be nonnegative"
        )
    try:
        quarter = pd.Period(str(test_quarter), freq="Q")
    except ValueError as exc:
        raise ValueError(f"invalid test quarter: {test_quarter}") from exc
    test_days = days[
        (days >= quarter.start_time.normalize())
        & (days <= quarter.end_time.normalize())
    ]
    if test_days.empty:
        raise ValueError(f"calendar contains no trading days for {test_quarter}")
    test_start_index = int(days.get_loc(test_days[0]))
    validation_end_index = test_start_index - label_horizon_days - 1
    validation_start_index = validation_end_index - validation_days + 1
    train_end_index = validation_start_index - label_horizon_days - 1
    train_start_index = train_end_index - train_days + 1
    if train_start_index < 0:
        raise ValueError(
            f"calendar history is insufficient for {test_quarter} and a {train_days}-day window"
        )
    return RollingSplit(
        train_start=days[train_start_index],
        train_end=days[train_end_index],
        validation_start=days[validation_start_index],
        validation_end=days[validation_end_index],
        test_start=test_days[0],
        test_end=test_days[-1],
    )


def make_confirmation_split(
    calendar: Sequence[Any],
    *,
    train_days: int,
    validation_days: int,
    label_horizon_days: int,
    test_start: str,
    test_end: str,
) -> RollingSplit:
    days = _clean_calendar(calendar)
    requested_start = pd.Timestamp(test_start).normalize()
    requested_end = pd.Timestamp(test_end).normalize()
    if requested_start > requested_end:
        raise ValueError("confirmation test_start must not be after test_end")
    selected = days[(days >= requested_start) & (days <= requested_end)]
    if selected.empty:
        raise ValueError("calendar contains no confirmation trading days")
    test_start_index = int(days.get_loc(selected[0]))
    validation_end_index = test_start_index - label_horizon_days - 1
    validation_start_index = validation_end_index - validation_days + 1
    train_end_index = validation_start_index - label_horizon_days - 1
    train_start_index = train_end_index - train_days + 1
    if train_start_index < 0:
        raise ValueError("calendar history is insufficient for confirmation split")
    return RollingSplit(
        train_start=days[train_start_index],
        train_end=days[train_end_index],
        validation_start=days[validation_start_index],
        validation_end=days[validation_end_index],
        test_start=selected[0],
        test_end=selected[-1],
    )


def add_forward_open_label(
    frame: pd.DataFrame,
    *,
    symbol_column: str = "symbol",
    date_column: str = "date",
    open_column: str = "open",
    output_column: str = "label_5d_open",
) -> pd.DataFrame:
    required = {symbol_column, date_column, open_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"price frame must contain: {', '.join(sorted(required))}")
    result = frame.copy()
    result[date_column] = pd.to_datetime(
        result[date_column], errors="coerce"
    ).dt.normalize()
    result[open_column] = pd.to_numeric(result[open_column], errors="coerce")
    result = result.sort_values([symbol_column, date_column], kind="stable")
    grouped = result.groupby(symbol_column, sort=False)[open_column]
    result[output_column] = grouped.shift(-6) / grouped.shift(-1) - 1.0
    return result


def execution_positions_from_regime(
    regimes: pd.Series, positions: Mapping[str, float]
) -> pd.Series:
    mapped = regimes.map({str(key): float(value) for key, value in positions.items()})
    unknown = sorted(set(regimes.dropna().astype(str)) - set(positions))
    if unknown:
        raise ValueError(f"unknown market regimes: {', '.join(unknown)}")
    return mapped.shift(1).rename("target_position")


def _remaining_group_capacity(
    selected: pd.DataFrame,
    weights: pd.Series,
    *,
    column: str,
    maximum: float,
) -> dict[str, float]:
    used = weights.groupby(selected[column]).sum()
    return {
        str(group): max(0.0, maximum - float(used.get(group, 0.0)))
        for group in selected[column].unique()
    }


def build_constrained_portfolio(
    candidates: pd.DataFrame,
    *,
    target_position: float,
    topk: int = 50,
    single_stock_max: float = 0.03,
    industry_max: float = 0.25,
    growth_star_max: float = 0.30,
) -> pd.DataFrame:
    required = {"symbol", "score", "industry", "board", "tradable"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates must contain: {', '.join(sorted(required))}")
    if not 0 <= target_position <= 1 or topk <= 0:
        raise ValueError("target_position must be in [0, 1] and topk must be positive")
    for value, name in (
        (single_stock_max, "single_stock_max"),
        (industry_max, "industry_max"),
        (growth_star_max, "growth_star_max"),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")

    selected = candidates.loc[candidates["tradable"].fillna(False).astype(bool)].copy()
    selected["score"] = pd.to_numeric(selected["score"], errors="coerce")
    selected = selected.dropna(subset=["symbol", "score", "industry", "board"])
    selected["symbol"] = selected["symbol"].astype(str).str.upper()
    selected["industry"] = selected["industry"].astype(str)
    selected["board"] = selected["board"].astype(str)
    selected = (
        selected.sort_values(
            ["score", "symbol"], ascending=[False, True], kind="stable"
        )
        .head(topk)
        .reset_index(drop=True)
    )
    selected["weight"] = 0.0
    if selected.empty or target_position == 0:
        return selected

    weights = selected["weight"].copy()
    growth_mask = selected["board"].isin(["growth", "star"])
    tolerance = 1e-12
    for _ in range(len(selected) * 4 + 20):
        residual = float(target_position - weights.sum())
        if residual <= tolerance:
            break
        individual_capacity = single_stock_max - weights
        industry_capacity = _remaining_group_capacity(
            selected, weights, column="industry", maximum=industry_max
        )
        growth_remaining = max(0.0, growth_star_max - float(weights[growth_mask].sum()))
        eligible = individual_capacity > tolerance
        eligible &= selected["industry"].map(industry_capacity).fillna(0.0) > tolerance
        if growth_remaining <= tolerance:
            eligible &= ~growth_mask
        eligible_indices = selected.index[eligible]
        if eligible_indices.empty:
            break
        equal_increment = residual / len(eligible_indices)
        increments = pd.Series(0.0, index=selected.index)
        increments.loc[eligible_indices] = np.minimum(
            individual_capacity.loc[eligible_indices], equal_increment
        )
        for industry, indices in (
            selected.loc[eligible].groupby("industry").groups.items()
        ):
            group_indices = pd.Index(indices)
            group_total = float(increments.loc[group_indices].sum())
            remaining = industry_capacity[str(industry)]
            if group_total > remaining and group_total > 0:
                increments.loc[group_indices] *= remaining / group_total
        growth_indices = selected.index[eligible & growth_mask]
        growth_total = float(increments.loc[growth_indices].sum())
        if growth_total > growth_remaining and growth_total > 0:
            increments.loc[growth_indices] *= growth_remaining / growth_total
        added = float(increments.sum())
        if added <= tolerance:
            break
        weights += increments
    selected["weight"] = weights.clip(lower=0.0)
    return selected


def price_limit_ratio(
    symbol: str, trade_date: str | pd.Timestamp, *, is_st: bool
) -> float:
    symbol = str(symbol).upper()
    day = pd.Timestamp(trade_date).normalize()
    if is_st:
        return 0.05
    if symbol.startswith("SH68"):
        return 0.20
    if symbol.startswith("SZ30"):
        return 0.20 if day >= pd.Timestamp("2020-08-24") else 0.10
    if symbol.startswith(("BJ", "NEEQ")):
        return 0.30
    return 0.10


def can_execute_open(
    *,
    side: str,
    previous_close: float,
    open_price: float,
    limit_ratio: float,
    suspended: bool,
) -> bool:
    if suspended or side not in {"buy", "sell"}:
        return False
    values = (previous_close, open_price, limit_ratio)
    if (
        not all(math.isfinite(float(value)) for value in values)
        or previous_close <= 0
        or open_price <= 0
        or limit_ratio <= 0
    ):
        return False
    close = Decimal(str(previous_close))
    ratio = Decimal(str(limit_ratio))
    upper_limit = float(
        (close * (Decimal("1") + ratio)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )
    lower_limit = float(
        (close * (Decimal("1") - ratio)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    )
    tolerance = 1e-8
    if side == "buy":
        return open_price < upper_limit - tolerance
    return open_price > lower_limit + tolerance


def calculate_order_costs(
    orders: pd.DataFrame,
    *,
    trade_date: str | pd.Timestamp,
    commission_rate: float,
    minimum_commission: float,
    slippage_bps: float,
) -> dict[str, float]:
    required = {"side", "notional"}
    if not required.issubset(orders.columns):
        raise ValueError("orders must contain side and notional")
    if commission_rate < 0 or minimum_commission < 0 or slippage_bps < 0:
        raise ValueError("cost parameters must be nonnegative")
    frame = orders.copy()
    frame["side"] = frame["side"].astype(str).str.lower()
    if not frame["side"].isin(["buy", "sell"]).all():
        raise ValueError("order side must be buy or sell")
    frame["notional"] = pd.to_numeric(frame["notional"], errors="coerce")
    if frame["notional"].isna().any() or (frame["notional"] < 0).any():
        raise ValueError("order notional must be finite and nonnegative")
    active = frame["notional"] > 0
    commission = (
        frame.loc[active, "notional"]
        .map(lambda value: max(float(value) * commission_rate, minimum_commission))
        .sum()
    )
    stamp_rate = (
        0.0005 if pd.Timestamp(trade_date) >= pd.Timestamp("2023-08-28") else 0.001
    )
    stamp_tax = (
        float(frame.loc[active & frame["side"].eq("sell"), "notional"].sum())
        * stamp_rate
    )
    slippage = float(frame.loc[active, "notional"].sum()) * slippage_bps / 10000.0
    return {
        "commission": float(commission),
        "stamp_tax": float(stamp_tax),
        "slippage": float(slippage),
        "total": float(commission + stamp_tax + slippage),
    }


def _finite(values: Sequence[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append(numeric)
    return result


def evaluate_development_acceptance(
    *,
    quarter_metrics: Sequence[Mapping[str, Any]],
    aggregate_annual_excess_return: float,
    aggregate_max_drawdown: float,
    window_rank_correlations: Sequence[float],
    top50_overlaps: Sequence[float],
    stress_excess_return: float,
    delayed_excess_return: float,
    regime_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    quarter_count = len(quarter_metrics)
    positive_excess = sum(
        float(item.get("annual_excess_return", float("nan"))) > 0
        for item in quarter_metrics
    )
    information_ratios = _finite(
        [item.get("information_ratio") for item in quarter_metrics]
    )
    rank_ics = _finite([item.get("rank_ic") for item in quarter_metrics])
    drawdowns = _finite([item.get("max_drawdown") for item in quarter_metrics])
    correlations = _finite(window_rank_correlations)
    overlaps = _finite(top50_overlaps)
    regime_positive = sum(
        float(item.get("annual_excess_return", float("nan"))) > 0
        for item in regime_metrics.values()
    )
    regime_floor_ok = all(
        int(item.get("days", 0)) < 60
        or float(item.get("annual_excess_return", -math.inf)) >= -0.05
        for item in regime_metrics.values()
    )
    values = {
        "positive_excess_quarter_ratio": positive_excess / quarter_count
        if quarter_count
        else 0.0,
        "aggregate_annual_excess_return": float(aggregate_annual_excess_return),
        "median_quarter_information_ratio": float(np.median(information_ratios))
        if information_ratios
        else -math.inf,
        "positive_rank_ic_interval_ratio": sum(value > 0 for value in rank_ics)
        / quarter_count
        if quarter_count
        else 0.0,
        "window_rank_correlation_median": float(np.median(correlations))
        if correlations
        else -math.inf,
        "top50_overlap_mean": float(np.mean(overlaps)) if overlaps else -math.inf,
        "worst_max_drawdown_abs": max(
            (abs(value) for value in drawdowns), default=math.inf
        ),
        "aggregate_max_drawdown_abs": abs(float(aggregate_max_drawdown)),
        "stress_excess_return": float(stress_excess_return),
        "delayed_excess_return": float(delayed_excess_return),
        "positive_regime_count": regime_positive,
        "regime_floor_ok": regime_floor_ok,
    }
    gates = {
        "positive_excess_quarters": values["positive_excess_quarter_ratio"] >= 0.70,
        "aggregate_annual_excess": values["aggregate_annual_excess_return"] >= 0.05,
        "median_quarter_information_ratio": values["median_quarter_information_ratio"]
        >= 0.5,
        "positive_rank_ic_intervals": values["positive_rank_ic_interval_ratio"] >= 0.70,
        "window_rank_stability": values["window_rank_correlation_median"] >= 0.70,
        "top50_stability": values["top50_overlap_mean"] >= 0.60,
        "maximum_drawdown": max(
            values["worst_max_drawdown_abs"],
            values["aggregate_max_drawdown_abs"],
        )
        <= 0.20,
        "stress_cost_positive_excess": values["stress_excess_return"] > 0,
        "delayed_execution_positive_excess": values["delayed_excess_return"] > 0,
        "market_regime_coverage": values["positive_regime_count"] >= 3
        and regime_floor_ok,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "values": values,
    }


def select_ensemble_models(
    candidates: Sequence[Mapping[str, Any]], *, count: int
) -> list[Mapping[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")

    def development_score(item: Mapping[str, Any]) -> float:
        try:
            value = float(item.get("development_score", -math.inf))
        except (TypeError, ValueError):
            return -math.inf
        return value if math.isfinite(value) else -math.inf

    return sorted(
        candidates,
        key=lambda item: (development_score(item), str(item.get("id", ""))),
        reverse=True,
    )[:count]
