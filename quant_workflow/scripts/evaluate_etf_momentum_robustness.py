"""Run a fixed, independent robustness acceptance for the two-ETF rule.

This is deliberately separate from CSI800 Goal 1.  The primary rule is never
selected from the parameter grid: neighboring lookbacks and moving averages
are diagnostics for sensitivity only.  No broker, network, or order code is
used here.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from etf_momentum_rotation import (  # noqa: E402
    CASH,
    DEFAULT_SYMBOLS,
    build_signal_table,
    simulate_open_to_open,
    summarize_periods,
    with_execution_lag,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "etf_momentum_robustness.yaml"
)
DEFAULT_DATA_DIR = (
    quant_data_root() / "parquet" / "etf_1d" / "cn_etf_daily"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_momentum_rotation" / "robustness"


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics_without_warmup(
    signal_table: pd.DataFrame, *, side_cost_bps: float, lag_days: int = 1
) -> dict[str, object]:
    table = with_execution_lag(signal_table, lag_days=lag_days)
    result = simulate_open_to_open(table, side_cost_bps=side_cost_bps)
    return {
        "metrics": result["metrics"],
        "periods": result["periods"],
        "equity_curve": result["equity_curve"],
    }


def complete_quarter_labels(signal_table: pd.DataFrame) -> list[str]:
    dates = (
        pd.to_datetime(signal_table["date"], errors="coerce").dropna().dt.normalize()
    )
    if dates.empty:
        return []
    periods = dates.dt.to_period("Q")
    labels: list[str] = []
    for period in sorted(periods.unique()):
        # Require observations on both sides of the quarter.  This excludes
        # the indicator warm-up quarter and the current partial quarter without
        # assuming that a calendar quarter end itself is a trading day.
        if (
            dates.min() < period.start_time.normalize()
            and dates.max() > period.end_time.normalize()
        ):
            labels.append(str(period))
    return labels


def _available_finished_quarter_labels(signal_table: pd.DataFrame) -> list[str]:
    dates = (
        pd.to_datetime(signal_table["date"], errors="coerce").dropna().dt.normalize()
    )
    if dates.empty:
        return []
    return [
        str(period)
        for period in sorted(dates.dt.to_period("Q").unique())
        if dates.max() > period.end_time.normalize()
    ]


def price_discontinuity_report(
    data: Mapping[str, pd.DataFrame],
    *,
    absolute_return_threshold: float = 0.30,
) -> dict[str, object]:
    """Flag price jumps that require a split/corporate-action explanation."""

    if not 0 < absolute_return_threshold < 1:
        raise ValueError("absolute_return_threshold must be between zero and one")
    events: list[dict[str, object]] = []
    for symbol, raw in data.items():
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values(
            "date", kind="stable"
        )
        frame["previous_close"] = frame["close"].shift(1)
        frame["close_return"] = frame["close"] / frame["previous_close"] - 1.0
        flagged = frame.loc[frame["close_return"].abs() > absolute_return_threshold]
        for _, row in flagged.iterrows():
            events.append(
                {
                    "symbol": str(symbol),
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "previous_close": float(row["previous_close"]),
                    "close": float(row["close"]),
                    "close_return": float(row["close_return"]),
                    "status": "unresolved_possible_split_or_bad_data",
                }
            )
    return {
        "passed": not events,
        "absolute_return_threshold": absolute_return_threshold,
        "events": events,
    }


def _quarter_report(
    periods: pd.DataFrame, labels: list[str]
) -> list[dict[str, object]]:
    if periods.empty:
        return []
    dates = pd.to_datetime(periods["date"], errors="coerce").dt.normalize()
    next_dates = pd.to_datetime(periods["next_date"], errors="coerce").dt.normalize()
    result: list[dict[str, object]] = []
    for label in labels:
        period = pd.Period(label, freq="Q")
        summary = summarize_periods(
            periods,
            start_date=period.start_time.normalize(),
            end_date=period.end_time.normalize(),
        )
        mask = (dates.dt.to_period("Q") == period) & (
            next_dates <= period.end_time.normalize()
        )
        scoped = periods.loc[mask]
        if int(summary["periods"]) == 0:
            continue
        result.append(
            {
                "quarter": label,
                **summary,
                "positive": bool(float(summary["total_return"]) > 0),
                "cash_fraction": float((scoped["position"].astype(str) == CASH).mean()),
                "switch_count": int(
                    (
                        scoped["position"].astype(str)
                        != scoped["previous_position"].astype(str)
                    ).sum()
                ),
            }
        )
    return result


def _cash_streaks(values: pd.Series) -> dict[str, object]:
    flags = values.astype(str).eq(CASH).tolist()
    streaks: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return {
        "cash_days": int(sum(streaks)),
        "cash_fraction": float(sum(streaks) / len(flags)) if flags else None,
        "cash_streak_count": len(streaks),
        "longest_cash_streak_days": max(streaks, default=0),
    }


def cash_metrics(signal_table: pd.DataFrame) -> dict[str, object]:
    """Return explicit no-position metrics after the indicator warm-up."""

    valid = signal_table.loc[
        signal_table[
            [
                column
                for column in signal_table
                if "return_" in column or "_ma_" in column
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()
    if valid.empty:
        valid = signal_table.iloc[0:0].copy()
    return {
        "warmup_rows_excluded": int(len(signal_table) - len(valid)),
        "signal_cash": _cash_streaks(valid["signal"]),
        "next_open_target_cash": _cash_streaks(valid["target_position"]),
        "definition": "After warm-up, CASH means the 20-day-return leader was not above its own 20-day moving average at the signal close.",
    }


def _parameter_metrics(
    data: Mapping[str, pd.DataFrame],
    *,
    lookback_days: int,
    moving_average_days: int,
    side_cost_bps: float,
) -> dict[str, object]:
    table = build_signal_table(
        data, lookback_days=lookback_days, moving_average_days=moving_average_days
    )
    valid_columns = [
        column for column in table if "return_" in column or "_ma_" in column
    ]
    table = table.loc[table[valid_columns].notna().all(axis=1)].reset_index(drop=True)
    result = _metrics_without_warmup(table, side_cost_bps=side_cost_bps)
    return {
        "lookback_days": lookback_days,
        "moving_average_days": moving_average_days,
        "metrics": result["metrics"],
    }


def evaluate_parameter_grid(
    data: Mapping[str, pd.DataFrame],
    *,
    lookbacks: list[int],
    moving_averages: list[int],
    side_cost_bps: float,
) -> dict[str, object]:
    """Evaluate a pre-declared grid without permitting winner selection."""

    variants = [
        _parameter_metrics(
            data,
            lookback_days=lookback,
            moving_average_days=moving_average,
            side_cost_bps=side_cost_bps,
        )
        for lookback in lookbacks
        for moving_average in moving_averages
    ]
    return {
        "primary": {"lookback_days": 20, "moving_average_days": 20},
        "selection_allowed": False,
        "variants": variants,
    }


def _reference_metrics(signal_table: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for symbol in DEFAULT_SYMBOLS:
        current = signal_table[f"{symbol}_open"]
        next_open = current.shift(-1)
        returns = (next_open / current - 1.0).dropna()
        if returns.empty:
            continue
        wealth = (1.0 + returns).cumprod()
        rows.append(
            {
                "symbol": symbol,
                "annualized_return": float(
                    wealth.iloc[-1] ** (252.0 / len(returns)) - 1.0
                ),
                "total_return": float(wealth.iloc[-1] - 1.0),
                "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
            }
        )
    return {"open_to_open_buy_and_hold": rows}


def _failure_diagnostics(periods: pd.DataFrame) -> dict[str, object]:
    if periods.empty:
        return {"switches": {}, "loss_after_switch": {}, "worst_periods": []}
    frame = periods.copy()
    switched = frame["position"].astype(str) != frame["previous_position"].astype(str)
    switch_rows = frame.loc[switched].copy()
    losses = pd.to_numeric(switch_rows["net_return"], errors="coerce") < 0
    worst = frame.nsmallest(min(10, len(frame)), "net_return")
    return {
        "switches": {
            "count": int(switched.sum()),
            "cash_entries_or_exits": int(
                (
                    (switch_rows["position"].astype(str) == CASH)
                    | (switch_rows["previous_position"].astype(str) == CASH)
                ).sum()
            ),
            "asset_to_asset_switches": int(
                (
                    (switch_rows["position"].astype(str) != CASH)
                    & (switch_rows["previous_position"].astype(str) != CASH)
                ).sum()
            ),
        },
        "loss_after_switch": {
            "switch_period_count": int(len(switch_rows)),
            "loss_count": int(losses.sum()),
            "loss_fraction": float(losses.mean()) if len(losses) else None,
            "median_net_return": _finite_or_none(
                pd.to_numeric(switch_rows["net_return"], errors="coerce").median()
            ),
        },
        "worst_periods": [
            {
                "date": str(pd.Timestamp(row["date"]).date()),
                "next_date": str(pd.Timestamp(row["next_date"]).date()),
                "position": str(row["position"]),
                "net_return": float(row["net_return"]),
            }
            for _, row in worst.iterrows()
        ],
        "known_failure_conditions": [
            "两只ETF同时下跌或同时跌破各自20日均线时，策略会空仓，但卖出仍可能遭遇隔夜跳空。",
            "价格在20日均线附近反复穿越时，策略可能频繁进出，手续费和滑点会吞掉小幅收益。",
            "强势标的快速反转时，20日动量属于滞后信号，可能在高位买入、在下跌后才退出。",
            "159915和513100不是分散到全市场的防守组合；两个标的同时受成长股、汇率、海外市场或溢价折价影响时，分散效果会下降。",
            "513100受美国市场开闭市、汇率、QDII额度、跟踪误差和场内溢价折价影响，A股收盘价不等于纳指当日最终表现。",
        ],
    }


def _acceptance(
    *,
    baseline: Mapping[str, object],
    stress: Mapping[str, object],
    delayed: Mapping[str, object],
    quarters: list[Mapping[str, object]],
    parameter_grid: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    limits = config["reference_lines"]
    baseline_metrics = baseline["metrics"]
    stress_metrics = stress["metrics"]
    delayed_metrics = delayed["metrics"]
    positive_ratio = (
        float(sum(bool(row["positive"]) for row in quarters) / len(quarters))
        if quarters
        else 0.0
    )
    variants = list(parameter_grid["variants"])
    positive_variants = sum(
        float(item["metrics"]["annualized_return"]) > 0 for item in variants
    )
    drawdown_variants = sum(
        abs(float(item["metrics"]["max_drawdown"]))
        <= float(limits["baseline_max_drawdown_abs_max"])
        for item in variants
    )
    parameter_positive_ratio = positive_variants / len(variants) if variants else 0.0
    parameter_drawdown_ratio = drawdown_variants / len(variants) if variants else 0.0
    checks = {
        "positive_quarter_ratio": positive_ratio
        >= float(limits["positive_quarter_ratio_min"]),
        "baseline_annualized_return": float(baseline_metrics["annualized_return"])
        >= float(limits["baseline_annualized_return_min"]),
        "baseline_max_drawdown": abs(float(baseline_metrics["max_drawdown"]))
        <= float(limits["baseline_max_drawdown_abs_max"]),
        "stress_annualized_return": float(stress_metrics["annualized_return"])
        > float(limits["stress_annualized_return_min_exclusive"]),
        "delayed_annualized_return": float(delayed_metrics["annualized_return"])
        > float(limits["delayed_annualized_return_min_exclusive"]),
        "parameter_positive_ratio": parameter_positive_ratio
        >= float(limits["parameter_positive_ratio_min"]),
        "parameter_max_drawdown_pass_ratio": parameter_drawdown_ratio
        >= float(limits["parameter_max_drawdown_pass_ratio_min"]),
    }
    values = {
        "positive_quarter_ratio": positive_ratio,
        "baseline_annualized_return": float(baseline_metrics["annualized_return"]),
        "baseline_max_drawdown_abs": abs(float(baseline_metrics["max_drawdown"])),
        "stress_annualized_return": float(stress_metrics["annualized_return"]),
        "delayed_annualized_return": float(delayed_metrics["annualized_return"]),
        "parameter_positive_ratio": parameter_positive_ratio,
        "parameter_max_drawdown_pass_ratio": parameter_drawdown_ratio,
    }
    reference_values = {
        "positive_quarter_ratio": float(limits["positive_quarter_ratio_min"]),
        "baseline_annualized_return": float(limits["baseline_annualized_return_min"]),
        "baseline_max_drawdown_abs": float(limits["baseline_max_drawdown_abs_max"]),
        "stress_annualized_return": float(
            limits["stress_annualized_return_min_exclusive"]
        ),
        "delayed_annualized_return": float(
            limits["delayed_annualized_return_min_exclusive"]
        ),
        "parameter_positive_ratio": float(limits["parameter_positive_ratio_min"]),
        "parameter_max_drawdown_pass_ratio": float(
            limits["parameter_max_drawdown_pass_ratio_min"]
        ),
    }
    return {
        "status": "manual_review_required",
        "decision_owner": "user",
        "automatic_rejection_disabled": True,
        "reference_lines_are_not_vetoes": True,
        "advisory_checks": checks,
        "values": values,
        "reference_values": reference_values,
        "advisory_not_met": [name for name, passed in checks.items() if not passed],
        "important_boundary": "Parameter neighbors are diagnostics only; this report never chooses a better-looking neighbor as the strategy.",
    }


def evaluate(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    config_path: Path = DEFAULT_CONFIG,
) -> Path:
    """Backward-compatible wrapper around the canonical report generator."""

    from generate_etf_momentum_strategy_report import run as generate_report

    resolved_report_dir = Path(report_dir)
    if resolved_report_dir.expanduser().resolve() == DEFAULT_REPORT_DIR.resolve():
        resolved_report_dir = DEFAULT_REPORT_DIR.parent
    return generate_report(
        data_dir=Path(data_dir),
        report_dir=resolved_report_dir,
        config_path=Path(config_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of the fixed two-ETF momentum rule."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = evaluate(
        data_dir=args.data_dir, report_dir=args.report_dir, config_path=args.config
    )
    print(f"ETF momentum robustness report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
