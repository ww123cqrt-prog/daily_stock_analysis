"""Pure diagnostics and acceptance rules for robust signal research."""

from __future__ import annotations

import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def feature_diagnostics(
    features: pd.DataFrame,
    *,
    correlation_threshold: float = 0.98,
    low_variance_threshold: float = 1.0e-8,
) -> dict[str, Any]:
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    numeric = features.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    missing = numeric.isna().mean()
    variances = numeric.var(ddof=0, skipna=True)
    constants = [str(column) for column in variances.index if variances[column] == 0]
    low_variance = [
        str(column)
        for column in variances.index
        if math.isfinite(float(variances[column]))
        and 0 < float(variances[column]) <= low_variance_threshold
    ]
    correlation = numeric.corr(method="spearman", min_periods=3)
    pairs: list[dict[str, Any]] = []
    for left, right in itertools.combinations(correlation.columns, 2):
        value = _finite(correlation.loc[left, right])
        if value is not None and abs(value) >= correlation_threshold:
            pairs.append({"left": str(left), "right": str(right), "correlation": value})
    return {
        "rows": len(features),
        "feature_count": len(features.columns),
        "missing_rate_by_feature": {str(key): float(value) for key, value in missing.items()},
        "constant_features": constants,
        "low_variance_features": low_variance,
        "nonfinite_value_count": int((~np.isfinite(features.apply(pd.to_numeric, errors="coerce"))).sum().sum()),
        "high_correlation_pairs": pairs,
    }


def _validated_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"score", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("prediction frame requires score and label")
    if not isinstance(frame.index, pd.MultiIndex) or not {
        "datetime",
        "instrument",
    }.issubset(frame.index.names):
        raise ValueError("prediction frame requires datetime/instrument MultiIndex")
    result = frame.loc[:, ["score", "label"]].apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def topk_sensitivity(
    frame: pd.DataFrame, *, topks: Sequence[int], primary: int
) -> dict[str, Any]:
    data = _validated_predictions(frame).dropna(subset=["score"])
    normalized = sorted({int(value) for value in topks})
    if primary not in normalized or any(value <= 0 for value in normalized):
        raise ValueError("primary TopK must be a positive member of topks")
    results: dict[str, dict[str, Any]] = {}
    for topk in normalized:
        selected = data.sort_values("score", ascending=False).groupby(
            level="datetime", sort=True
        ).head(topk)
        results[str(topk)] = {
            "selected_rows": int(len(selected)),
            "date_count": int(selected.index.get_level_values("datetime").nunique()),
            "mean_label": _finite(selected["label"].mean()),
        }
    return {"primary_topk": int(primary), "results": results}


def quarterly_rank_ic(frame: pd.DataFrame) -> dict[str, float | None]:
    data = _validated_predictions(frame).dropna()
    daily = data.groupby(level="datetime", sort=True).apply(
        lambda group: group["score"].corr(group["label"], method="spearman")
    )
    daily = pd.to_numeric(daily, errors="coerce").replace([np.inf, -np.inf], np.nan)
    quarters = daily.index.to_period("Q")
    return {
        str(quarter): _finite(values.mean())
        for quarter, values in daily.groupby(quarters, sort=True)
    }


def prediction_dispersion(frame: pd.DataFrame, *, topk: int) -> dict[str, Any]:
    data = _validated_predictions(frame).dropna(subset=["score"])
    rows: list[dict[str, Any]] = []
    for date_value, group in data.groupby(level="datetime", sort=True):
        ordered = group["score"].sort_values(ascending=False)
        boundary_gap = None
        if len(ordered) > topk:
            boundary_gap = _finite(ordered.iloc[topk - 1] - ordered.iloc[topk])
        rows.append(
            {
                "date": str(pd.Timestamp(date_value).date()),
                "score_std": _finite(ordered.std(ddof=0)),
                "topk_boundary_gap": boundary_gap,
            }
        )
    gaps = [row["topk_boundary_gap"] for row in rows if row["topk_boundary_gap"] is not None]
    return {
        "dates": rows,
        "score_std_median": _finite(np.median([row["score_std"] for row in rows if row["score_std"] is not None])) if rows else None,
        "topk_boundary_gap_median": _finite(np.median(gaps)) if gaps else None,
    }


def decile_monotonicity(
    frame: pd.DataFrame, *, deciles: int = 10, spearman_min: float = 0.8
) -> dict[str, Any]:
    data = _validated_predictions(frame).dropna()
    quarter_results: list[dict[str, Any]] = []
    dates = data.index.get_level_values("datetime")
    for quarter, quarter_frame in data.groupby(dates.to_period("Q"), sort=True):
        assigned: list[pd.DataFrame] = []
        for _, day in quarter_frame.groupby(level="datetime", sort=True):
            if len(day) < deciles:
                continue
            ranks = day["score"].rank(method="first", pct=True)
            bucket = np.minimum((ranks * deciles).astype(int), deciles - 1)
            current = day.copy()
            current["decile"] = bucket.values
            assigned.append(current)
        if not assigned:
            continue
        grouped = pd.concat(assigned).groupby("decile")["label"].mean().sort_index()
        correlation = _finite(pd.Series(grouped.index).corr(pd.Series(grouped.values), method="spearman"))
        passed = bool(
            correlation is not None
            and correlation >= spearman_min
            and grouped.iloc[-1] > grouped.iloc[0]
        )
        quarter_results.append(
            {
                "quarter": str(quarter),
                "spearman": correlation,
                "top_minus_bottom": _finite(grouped.iloc[-1] - grouped.iloc[0]),
                "passed": passed,
            }
        )
    ratio = (
        sum(item["passed"] for item in quarter_results) / len(quarter_results)
        if quarter_results
        else 0.0
    )
    return {"quarter_ratio": ratio, "quarters": quarter_results}


def horizon_aligned_excess(
    frame: pd.DataFrame,
    *,
    topk: int,
    horizon_days: int,
    total_cost_bps: float,
    labels_are_excess: bool = False,
    evaluation_dates: Sequence[Any] | None = None,
) -> dict[str, Any]:
    data = _validated_predictions(frame).dropna()
    if not labels_are_excess:
        if "benchmark_return" not in frame.columns:
            raise ValueError("raw-return evaluation requires benchmark_return")
        data["benchmark_return"] = pd.to_numeric(
            frame.loc[data.index, "benchmark_return"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    dates = pd.DatetimeIndex(data.index.get_level_values("datetime").unique()).sort_values()
    selected_dates = (
        dates[:: int(horizon_days)]
        if evaluation_dates is None
        else pd.DatetimeIndex(pd.to_datetime(list(evaluation_dates))).normalize().unique().sort_values()
    )
    available_dates = set(dates)
    period_returns: list[tuple[pd.Timestamp, float]] = []
    cost = float(total_cost_bps) / 10000.0
    for date_value in selected_dates:
        if date_value not in available_dates:
            continue
        day = data.xs(date_value, level="datetime")
        if len(day) < topk:
            continue
        selected = day.nlargest(topk, "score")
        if labels_are_excess:
            benchmark = 0.0
        else:
            benchmark_values = day["benchmark_return"].dropna().unique()
            if len(benchmark_values) != 1:
                raise ValueError(
                    "benchmark_return must contain one finite value per evaluation date"
                )
            benchmark = float(benchmark_values[0])
        excess = float(selected["label"].mean() - benchmark - cost)
        if math.isfinite(excess):
            period_returns.append((pd.Timestamp(date_value), excess))
    by_quarter: dict[str, list[float]] = {}
    for date_value, value in period_returns:
        by_quarter.setdefault(str(date_value.to_period("Q")), []).append(value)
    quarter_returns = {
        quarter: float(np.prod([1.0 + item for item in values]) - 1.0)
        for quarter, values in by_quarter.items()
    }
    cumulative = float(np.prod([1.0 + value for _, value in period_returns]) - 1.0) if period_returns else -1.0
    periods_per_year = 252.0 / float(horizon_days)
    annual = (
        float((1.0 + cumulative) ** (periods_per_year / len(period_returns)) - 1.0)
        if period_returns and cumulative > -1.0
        else None
    )
    return {
        "period_count": len(period_returns),
        "quarter_returns": quarter_returns,
        "positive_quarter_ratio": (
            sum(value > 0 for value in quarter_returns.values()) / len(quarter_returns)
            if quarter_returns
            else 0.0
        ),
        "cumulative_excess_return": cumulative,
        "annual_excess_return": annual,
    }


def market_state_attribution(
    frame: pd.DataFrame,
    *,
    topk: int,
    horizon_days: int,
    total_cost_bps: float,
    labels_are_excess: bool = False,
) -> dict[str, Any]:
    states = ("risk_on", "neutral", "defensive", "risk_off")
    if "market_state" not in frame.columns:
        raise ValueError("market-state attribution requires market_state")
    data = frame.copy()
    data["market_state"] = data["market_state"].astype("string")
    unknown = sorted(set(data["market_state"].dropna().astype(str)) - set(states))
    if unknown:
        raise ValueError(f"unknown market_state values: {', '.join(unknown)}")

    all_dates = pd.DatetimeIndex(
        data.index.get_level_values("datetime").unique()
    ).sort_values()
    evaluation_dates = all_dates[:: int(horizon_days)]
    result: dict[str, Any] = {}
    for state in states:
        state_frame = data[data["market_state"] == state]
        metrics = horizon_aligned_excess(
            state_frame,
            topk=topk,
            horizon_days=horizon_days,
            total_cost_bps=total_cost_bps,
            labels_are_excess=labels_are_excess,
            evaluation_dates=evaluation_dates,
        )
        result[state] = metrics
    return result


def acceptance_verdict(
    values: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    gates = {
        "feature_coverage": ("feature_coverage", "feature_coverage_min"),
        "positive_rank_ic_quarters": (
            "positive_rank_ic_quarter_ratio",
            "positive_rank_ic_quarter_ratio_min",
        ),
        "median_quarter_rank_ic": (
            "median_quarter_rank_ic",
            "median_quarter_rank_ic_min",
        ),
        "window_rank_correlation": (
            "window_rank_correlation_median",
            "window_rank_correlation_median_min",
        ),
        "top50_overlap": ("top50_overlap_mean", "top50_overlap_mean_min"),
        "positive_excess_quarters": (
            "positive_excess_quarter_ratio",
            "positive_excess_quarter_ratio_min",
        ),
        "annual_excess_return": (
            "annual_excess_return",
            "annual_excess_return_min",
        ),
        "monotonic_quarters": (
            "monotonic_quarter_ratio",
            "monotonic_quarter_ratio_min",
        ),
    }
    results: dict[str, bool] = {}
    for gate, (value_key, threshold_key) in gates.items():
        value = _finite(values.get(value_key))
        threshold = _finite(thresholds.get(threshold_key))
        if threshold is None:
            raise ValueError(f"missing or invalid acceptance threshold: {threshold_key}")
        results[gate] = value is not None and value >= threshold
    return {
        "passed": all(results.values()),
        "gates": results,
        "failed_gates": [name for name, passed in results.items() if not passed],
        "values": dict(values),
        "thresholds": dict(thresholds),
    }
