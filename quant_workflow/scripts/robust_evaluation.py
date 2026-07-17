"""Aggregation and fail-closed acceptance helpers for robust strategy research."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robust_backtest import performance_metrics, simulate_strategy
from robust_strategy_core import evaluate_development_acceptance


def _validate_prediction_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not {"score", "label"}.issubset(frame.columns):
        raise ValueError(f"prediction frame {name} must contain score and label")
    if not isinstance(frame.index, pd.MultiIndex) or not {
        "datetime",
        "instrument",
    }.issubset(frame.index.names):
        raise ValueError(f"prediction frame {name} must use datetime/instrument index")
    result = frame.loc[:, ["score", "label"]].copy().sort_index()
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    return result


def aggregate_prediction_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one prediction frame is required")
    normalized = {
        name: _validate_prediction_frame(frame, name) for name, frame in frames.items()
    }
    first_name = next(iter(normalized))
    first = normalized[first_name]
    ranked: list[pd.Series] = []
    for name, frame in normalized.items():
        if not frame.index.equals(first.index):
            raise ValueError("prediction indexes differ across seeds or windows")
        if not np.allclose(
            frame["label"].to_numpy(), first["label"].to_numpy(), equal_nan=True
        ):
            raise ValueError("label values differ across seeds or windows")
        ranked.append(
            frame["score"]
            .groupby(level="datetime")
            .rank(method="average", pct=True)
            .rename(name)
        )
    return pd.DataFrame(
        {
            "score": pd.concat(ranked, axis=1).mean(axis=1),
            "label": first["label"],
        },
        index=first.index,
    )


def window_stability(
    frames: Mapping[int, pd.DataFrame], *, topk: int = 50
) -> dict[str, Any]:
    if len(frames) < 2:
        return {
            "pair_count": 0,
            "rank_correlation_median": float("nan"),
            "topk_overlap_mean": float("nan"),
            "pairs": [],
        }
    pair_reports: list[dict[str, Any]] = []
    correlations: list[float] = []
    overlaps: list[float] = []
    for (left_window, left_raw), (right_window, right_raw) in itertools.combinations(
        sorted(frames.items()), 2
    ):
        left = _validate_prediction_frame(left_raw, str(left_window))
        right = _validate_prediction_frame(right_raw, str(right_window))
        joined = (
            left[["score"]]
            .rename(columns={"score": "left"})
            .join(right[["score"]].rename(columns={"score": "right"}), how="inner")
            .dropna()
        )
        daily_corr = joined.groupby(level="datetime").apply(
            lambda group: group["left"].corr(group["right"], method="spearman")
        )
        daily_corr = (
            pd.to_numeric(daily_corr, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        pair_corr = float(daily_corr.median()) if not daily_corr.empty else float("nan")
        pair_overlaps: list[float] = []
        for _, group in joined.groupby(level="datetime", sort=True):
            left_top = set(
                group.nlargest(min(topk, len(group)), "left").index.get_level_values(
                    "instrument"
                )
            )
            right_top = set(
                group.nlargest(min(topk, len(group)), "right").index.get_level_values(
                    "instrument"
                )
            )
            denominator = max(1, min(len(left_top), len(right_top)))
            pair_overlaps.append(len(left_top & right_top) / denominator)
        pair_overlap = float(np.mean(pair_overlaps)) if pair_overlaps else float("nan")
        if math.isfinite(pair_corr):
            correlations.append(pair_corr)
        if math.isfinite(pair_overlap):
            overlaps.append(pair_overlap)
        pair_reports.append(
            {
                "left_window": int(left_window),
                "right_window": int(right_window),
                "rank_correlation_median": pair_corr,
                "topk_overlap_mean": pair_overlap,
            }
        )
    return {
        "pair_count": len(pair_reports),
        "rank_correlation_median": float(np.median(correlations))
        if correlations
        else float("nan"),
        "topk_overlap_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
        "pairs": pair_reports,
    }


def model_signal_report(
    *,
    model_id: str,
    quarter_metrics: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    rank_ics: list[float] = []
    for item in quarter_metrics:
        try:
            value = float(item.get("rank_ic"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            rank_ics.append(value)
    denominator = len(quarter_metrics)
    positive_ratio = (
        sum(value > 0 for value in rank_ics) / denominator if denominator else 0.0
    )
    rank_corr = float(stability.get("rank_correlation_median", float("nan")))
    overlap = float(stability.get("topk_overlap_mean", float("nan")))
    gates = {
        "positive_rank_ic_intervals": positive_ratio >= 0.70,
        "window_rank_correlation": math.isfinite(rank_corr) and rank_corr >= 0.70,
        "top50_overlap": math.isfinite(overlap) and overlap >= 0.60,
    }
    mean_rank_ic = float(np.mean(rank_ics)) if rank_ics else -math.inf
    development_score = (
        mean_rank_ic
        + (rank_corr if math.isfinite(rank_corr) else -1.0) * 0.01
        + (overlap if math.isfinite(overlap) else -1.0) * 0.01
    )
    return {
        "model_id": model_id,
        "eligible": all(gates.values()),
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "gates": gates,
        "positive_rank_ic_interval_ratio": positive_ratio,
        "rank_ic_mean": mean_rank_ic,
        "stability": dict(stability),
        "quarter_metrics": [dict(item) for item in quarter_metrics],
        "development_score": development_score,
    }


def _load_prediction_artifact(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"datetime", "instrument", "score", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction artifact is missing columns: {path}")
    frame["datetime"] = pd.to_datetime(
        frame["datetime"], errors="coerce"
    ).dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    if frame.duplicated(["datetime", "instrument"]).any():
        raise ValueError(f"prediction artifact contains duplicate keys: {path}")
    return frame.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()


def _rank_ic(frame: pd.DataFrame) -> float:
    daily = (
        frame.dropna()
        .groupby(level="datetime")
        .apply(lambda group: group["score"].corr(group["label"], method="spearman"))
    )
    clean = (
        pd.to_numeric(daily, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return float(clean.mean()) if not clean.empty else float("nan")


def aggregate_training_artifacts(
    records: Sequence[Mapping[str, Any]], *, topk: int = 50
) -> dict[str, Any]:
    if not records:
        raise ValueError("training artifact records are empty")
    grouped: dict[tuple[str, int, str], dict[str, pd.DataFrame]] = {}
    for record in records:
        key = (
            str(record["model_id"]),
            int(record["window_days"]),
            str(record["quarter"]),
        )
        seed = "fixed" if record.get("seed") is None else str(record.get("seed"))
        path = Path(str(record.get("prediction_path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"prediction artifact is missing: {path}")
        grouped.setdefault(key, {})[seed] = _load_prediction_artifact(path)

    quarter_windows: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
    model_windows: dict[str, dict[int, list[pd.DataFrame]]] = {}
    for (model_id, window, quarter), seed_frames in grouped.items():
        aggregate = aggregate_prediction_frames(seed_frames)
        quarter_windows.setdefault((model_id, quarter), {})[window] = aggregate
        model_windows.setdefault(model_id, {}).setdefault(window, []).append(aggregate)

    window_predictions: dict[str, dict[int, pd.DataFrame]] = {}
    for model_id, windows in model_windows.items():
        window_predictions[model_id] = {}
        for window, frames in windows.items():
            combined = pd.concat(frames).sort_index()
            if combined.index.duplicated().any():
                raise ValueError(
                    f"quarter prediction intervals overlap for {model_id} window {window}"
                )
            window_predictions[model_id][window] = combined

    model_reports: list[dict[str, Any]] = []
    model_predictions: dict[str, pd.DataFrame] = {}
    for model_id, windows in sorted(window_predictions.items()):
        stability = window_stability(windows, topk=topk)
        quarter_metrics: list[dict[str, Any]] = []
        quarter_aggregates: list[pd.DataFrame] = []
        for (candidate_model, quarter), frames in sorted(quarter_windows.items()):
            if candidate_model != model_id:
                continue
            aggregate = aggregate_prediction_frames(
                {str(window): frame for window, frame in frames.items()}
            )
            quarter_metrics.append({"quarter": quarter, "rank_ic": _rank_ic(aggregate)})
            quarter_aggregates.append(aggregate)
        report = model_signal_report(
            model_id=model_id, quarter_metrics=quarter_metrics, stability=stability
        )
        model_reports.append(report)
        combined = pd.concat(quarter_aggregates).sort_index()
        if combined.index.duplicated().any():
            raise ValueError(f"model quarter aggregates overlap for {model_id}")
        model_predictions[model_id] = combined
    return {
        "model_reports": model_reports,
        "model_predictions": model_predictions,
        "window_predictions": window_predictions,
    }


def _period_metrics(
    daily: pd.DataFrame, predictions: pd.DataFrame, quarter: str
) -> dict[str, Any]:
    period = pd.Period(quarter, freq="Q")
    start = period.start_time.normalize()
    end = period.end_time.normalize()
    dates = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    metrics = dict(performance_metrics(daily[(dates >= start) & (dates <= end)]))
    prediction_dates = predictions.index.get_level_values("datetime")
    quarter_predictions = predictions[
        (prediction_dates >= start) & (prediction_dates <= end)
    ]
    metrics["quarter"] = quarter
    metrics["rank_ic"] = _rank_ic(quarter_predictions)
    return metrics


def _regime_metrics(
    daily: pd.DataFrame, regimes: pd.Series
) -> dict[str, dict[str, Any]]:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    regime_series = regimes.copy()
    regime_series.index = pd.to_datetime(regime_series.index).normalize()
    regime_for_next_open = regime_series.sort_index().shift(1)
    frame["regime"] = frame["date"].map(regime_for_next_open)
    result: dict[str, dict[str, Any]] = {}
    for regime in ("risk_on", "neutral", "defensive", "risk_off"):
        selected = frame[frame["regime"] == regime]
        metrics = dict(performance_metrics(selected))
        metrics["days"] = int(len(selected))
        result[regime] = metrics
    return result


def _board_attribution(daily: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for board in ("main", "growth", "star"):
        column = f"board_exposure_{board}"
        if column not in daily:
            values = pd.Series(dtype=float)
        else:
            values = (
                pd.to_numeric(daily[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
        result[board] = {
            "average_exposure": float(values.mean()) if not values.empty else 0.0,
            "max_exposure": float(values.max()) if not values.empty else 0.0,
            "days": int(len(values)),
        }
    return result


def evaluate_strategy_candidates(
    *,
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    regimes: pd.Series,
    config: Mapping[str, Any],
    selected_model_reports: Sequence[Mapping[str, Any]],
    simulator: Any = simulate_strategy,
) -> dict[str, Any]:
    prediction_frame = _validate_prediction_frame(predictions, "ensemble")
    simulator_predictions = prediction_frame.reset_index().loc[
        :, ["datetime", "instrument", "score"]
    ]
    portfolio = config["portfolio"]
    costs = config["costs"]
    positions = config["market_regime"]["positions"]
    common = {
        "predictions": simulator_predictions,
        "market": market,
        "benchmark": benchmark,
        "regimes": regimes,
        "regime_positions": positions,
        "initial_cash": 10_000_000.0,
        "topk": int(portfolio["topk"]),
        "fixed_position": float(positions["risk_on"]),
        "single_stock_max": float(portfolio["single_stock_weight_max"]),
        "industry_max": float(portfolio["industry_weight_max"]),
        "growth_star_max": float(portfolio["growth_and_star_weight_max"]),
        "commission_rate": float(costs["commission_rate_each_side"]),
        "minimum_commission": float(costs["minimum_commission_cny"]),
        "slippage_bps": 0.0,
        "defer_unfilled_open_orders": bool(
            config["execution_rules"]["defer_unfilled_open_orders"]
        ),
    }
    rank_correlations = [
        float(item.get("stability", {}).get("rank_correlation_median", float("nan")))
        for item in selected_model_reports
    ]
    topk_overlaps = [
        float(item.get("stability", {}).get("topk_overlap_mean", float("nan")))
        for item in selected_model_reports
    ]
    reports: list[dict[str, Any]] = []
    for strategy in config["strategies"]:
        strategy_id = str(strategy["id"])
        low_cost = simulator(
            **common,
            strategy_id=strategy_id,
            all_in_round_trip_bps=float(costs["scenarios"]["low_total_bps"]),
            additional_execution_delay_days=0,
        )
        baseline = simulator(
            **common,
            strategy_id=strategy_id,
            all_in_round_trip_bps=float(costs["scenarios"]["baseline_total_bps"]),
            additional_execution_delay_days=0,
        )
        stress = simulator(
            **common,
            strategy_id=strategy_id,
            all_in_round_trip_bps=float(costs["scenarios"]["stress_total_bps"]),
            additional_execution_delay_days=0,
        )
        delayed = simulator(
            **common,
            strategy_id=strategy_id,
            all_in_round_trip_bps=float(costs["scenarios"]["baseline_total_bps"]),
            additional_execution_delay_days=int(costs["delayed_execution_days_stress"]),
        )
        quarter_metrics = [
            _period_metrics(baseline["daily"], prediction_frame, quarter)
            for quarter in config["rolling_validation"]["test_quarters"]
        ]
        regime_metrics = _regime_metrics(baseline["daily"], regimes)
        board_attribution = _board_attribution(baseline["daily"])
        acceptance = evaluate_development_acceptance(
            quarter_metrics=quarter_metrics,
            aggregate_annual_excess_return=float(
                baseline["metrics"]["annual_excess_return"]
            ),
            aggregate_max_drawdown=float(baseline["metrics"]["max_drawdown"]),
            window_rank_correlations=rank_correlations,
            top50_overlaps=topk_overlaps,
            stress_excess_return=float(stress["metrics"]["annual_excess_return"]),
            delayed_excess_return=float(delayed["metrics"]["annual_excess_return"]),
            regime_metrics=regime_metrics,
        )
        reports.append(
            {
                "strategy_id": strategy_id,
                "low_cost": low_cost,
                "baseline": baseline,
                "stress": stress,
                "delayed": delayed,
                "quarter_metrics": quarter_metrics,
                "regime_metrics": regime_metrics,
                "board_attribution": board_attribution,
                "acceptance": acceptance,
            }
        )
    passing = [item for item in reports if item["acceptance"]["passed"]]
    passing.sort(
        key=lambda item: (
            float(item["baseline"]["metrics"]["annual_excess_return"]),
            float(item["baseline"]["metrics"]["information_ratio"]),
            item["strategy_id"],
        ),
        reverse=True,
    )
    return {
        "status": "research_candidate_ready" if passing else "development_failed",
        "reason_code": None if passing else "no_strategy_passed_all_development_gates",
        "selected_strategy": passing[0] if passing else None,
        "strategies": reports,
    }


def select_development_models(
    reports: Sequence[Mapping[str, Any]],
    *,
    minimum: int = 2,
    maximum: int = 3,
) -> list[Mapping[str, Any]]:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid ensemble model count bounds")
    eligible = [item for item in reports if item.get("eligible") is True]
    eligible.sort(
        key=lambda item: (
            float(item.get("development_score", -math.inf)),
            str(item.get("model_id", "")),
        ),
        reverse=True,
    )
    if len(eligible) < minimum:
        return []
    return eligible[:maximum]


def confirmation_verdict(
    *,
    annual_excess_return: float,
    information_ratio: float,
    max_drawdown: float,
) -> dict[str, Any]:
    gates = {
        "positive_net_excess": float(annual_excess_return) > 0,
        "positive_information_ratio": float(information_ratio) > 0,
        "maximum_drawdown": abs(float(max_drawdown)) <= 0.20,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def shadow_verdict(
    *,
    trading_days: int,
    net_excess_return: float,
    max_drawdown: float,
    task_success_ratio: float,
) -> dict[str, Any]:
    if int(trading_days) < 60:
        return {
            "status": "shadow_observation",
            "passed": False,
            "remaining_days": 60 - int(trading_days),
            "gates": {},
        }
    gates = {
        "positive_net_excess": float(net_excess_return) > 0,
        "maximum_drawdown": abs(float(max_drawdown)) <= 0.12,
        "task_success_ratio": float(task_success_ratio) >= 0.95,
    }
    return {
        "status": "robust_strategy_qualified"
        if all(gates.values())
        else "development_failed",
        "passed": all(gates.values()),
        "remaining_days": 0,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }
