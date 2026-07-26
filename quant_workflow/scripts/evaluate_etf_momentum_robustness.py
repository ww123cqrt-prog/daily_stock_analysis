"""Run a fixed, independent robustness acceptance for the two-ETF rule.

This is deliberately separate from CSI800 Goal 1.  The primary rule is never
selected from the parameter grid: neighboring lookbacks and moving averages
are diagnostics for sensitivity only.  No broker, network, or order code is
used here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from etf_momentum_rotation import (  # noqa: E402
    CASH,
    DEFAULT_SYMBOLS,
    build_signal_table,
    load_price_data,
    sanitize_json_value,
    simulate_open_to_open,
    summarize_periods,
    with_execution_lag,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "etf_momentum_robustness.yaml"
)
DEFAULT_DATA_DIR = (
    quant_data_root() / "parquet" / "etf_1d" / "etf_momentum_rotation" / "v1"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_momentum_rotation" / "robustness"


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            sanitize_json_value(dict(payload)),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _markdown(report: Mapping[str, object]) -> str:
    acceptance = report["acceptance"]
    baseline_total_bps = report["config"]["evaluation"]["costs_total_bps"]["baseline"]
    stress_total_bps = report["config"]["evaluation"]["costs_total_bps"]["stress"]
    baseline = report["variants"][f"baseline_total_{baseline_total_bps}bps"]["metrics"]
    stress = report["variants"][f"stress_total_{stress_total_bps}bps"]["metrics"]
    delayed = report["variants"]["delayed_one_session"]["metrics"]
    cash = report["cash_metrics"]
    advisory_names = {
        "positive_quarter_ratio": "严格完整季度正收益比例",
        "baseline_annualized_return": "基准成本年化收益",
        "baseline_max_drawdown": "基准成本最大回撤绝对值",
        "stress_annualized_return": "40bp总成本下年化收益",
        "delayed_annualized_return": "额外延迟一个交易日后年化收益",
        "parameter_positive_ratio": "参数邻域正收益比例",
        "parameter_max_drawdown_pass_ratio": "参数邻域回撤合格比例",
    }
    advisory_value_keys = {
        "positive_quarter_ratio": "positive_quarter_ratio",
        "baseline_annualized_return": "baseline_annualized_return",
        "baseline_max_drawdown": "baseline_max_drawdown_abs",
        "stress_annualized_return": "stress_annualized_return",
        "delayed_annualized_return": "delayed_annualized_return",
        "parameter_positive_ratio": "parameter_positive_ratio",
        "parameter_max_drawdown_pass_ratio": "parameter_max_drawdown_pass_ratio",
    }
    advisory_operators = {
        "positive_quarter_ratio": ">=",
        "baseline_annualized_return": ">=",
        "baseline_max_drawdown": "<=",
        "stress_annualized_return": ">",
        "delayed_annualized_return": ">",
        "parameter_positive_ratio": ">=",
        "parameter_max_drawdown_pass_ratio": ">=",
    }
    evidence_names = {
        "unresolved_price_discontinuity": "513100价格序列存在尚未解释的约80%断点",
        "independent_source_cross_check_not_done": "尚未用第二个独立数据源交叉核验",
        "dividend_and_split_adjustment_not_verified": "分红、份额拆分和复权口径尚未核验",
        "sixty_trading_day_paper_tracking_not_done": "尚未完成60个交易日纸面跟踪",
    }
    lines = [
        "# ETF 动量轮动鲁棒性验收",
        "",
        "> 这是固定规则的历史验收，不是收益承诺、投资建议或自动交易信号。",
        "",
        f"## 状态：`{acceptance['status']}`",
        "",
        "主规则固定为：20日涨幅排名 + 收盘价高于20日均线，下一交易日开盘执行。自动一票否决已关闭；通过项、参考门槛未达到项和证据缺口全部保留，由用户决定是否接受。",
        "",
        "## 主规则验收摘要",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
        f"| 严格完整季度正收益比例 | {acceptance['values']['positive_quarter_ratio']:.2%} |",
        f"| 含首个热身期季度的比例 | {report['quarter_summary']['all_available_positive_ratio']:.2%} |",
        f"| 早期可比口径（总成本{report['quarter_summary']['earlier_comparison_total_bps']}bp，含首个不完整季度） | {report['quarter_summary']['earlier_comparison_positive_ratio']:.2%} |",
        f"| 基准成本年化收益（总成本{baseline_total_bps}bp） | {baseline['annualized_return']:.2%} |",
        f"| 基准成本最大回撤 | {baseline['max_drawdown']:.2%} |",
        f"| 压力成本年化收益（总成本{stress_total_bps}bp） | {stress['annualized_return']:.2%} |",
        f"| 延迟一个交易日后的年化收益 | {delayed['annualized_return']:.2%} |",
        "",
        f"早期可比口径的{report['quarter_summary']['earlier_comparison_positive_ratio']:.2%}和当前严格口径不是同一个数字：早期使用{report['quarter_summary']['earlier_comparison_total_bps']}bp总成本并包含首个不完整季度；当前主口径使用{baseline_total_bps}bp总成本并只统计完整季度。两者都会保留，不互相替代。",
        "",
        "## 全部参考线检查",
        "",
        "| 检查 | 实际值 | 参考线 | 结果 |",
        "|---|---:|---:|---|",
    ]
    for name, passed in acceptance["advisory_checks"].items():
        value_key = advisory_value_keys[name]
        lines.append(
            f"| {advisory_names[name]} | {acceptance['values'][value_key]:.2%} | "
            f"{advisory_operators[name]} {acceptance['reference_values'][value_key]:.2%} | "
            f"{'达到参考线' if passed else '未达到，交由用户判断'} |"
        )
    lines.extend(
        [
            "",
            "## 参考门槛未达到项",
            "",
        ]
    )
    if acceptance["advisory_not_met"]:
        lines.extend(
            f"- {advisory_names.get(name, name)}未达到原参考线；不会自动淘汰，由你判断。"
            for name in acceptance["advisory_not_met"]
        )
    else:
        lines.append("- 没有；但这不等于证据已经完整。")
    lines.extend(
        [
            "",
            "## 尚未完成的关键证据",
            "",
        ]
    )
    lines.extend(
        f"- {evidence_names.get(name, name)}。"
        for name in report["data_quality"]["evidence_gaps"]
    )
    lines.extend(
        [
            "",
            "## 空仓指标",
            "",
            "- 指标已经存在，代码状态是 `CASH`。热身结束后，它表示20日涨幅第一名没有站在自己的20日均线上方；并不要求两只ETF都跌破均线。",
            f"- 指标热身期排除了 `{cash['warmup_rows_excluded']}` 行；这部分只是指标还没形成，不能算策略主动空仓。",
            f"- 热身期之后，收盘信号为空仓 `{cash['signal_cash']['cash_days']}` 天，占 `{cash['signal_cash']['cash_fraction']:.2%}`；最长连续空仓 `{cash['signal_cash']['longest_cash_streak_days']}` 个交易日。",
            f"- 下一开盘目标为空仓 `{cash['next_open_target_cash']['cash_days']}` 天，占 `{cash['next_open_target_cash']['cash_fraction']:.2%}`。",
            "",
            "## 参数稳定性",
            "",
            f"- 预先声明的邻域：lookback={report['config']['evaluation']['parameter_lookbacks']}，均线={report['config']['evaluation']['parameter_moving_averages']}。",
            f"- 通过成本和回撤的邻域比例：{acceptance['values']['parameter_max_drawdown_pass_ratio']:.2%}。",
            "- 邻域只用于发现敏感性，不能据此改成新的20日策略，也不能事后挑选冠军。",
            "",
            "## 已知可能失效的情况",
            "",
        ]
    )
    for condition in report["failure_diagnostics"]["known_failure_conditions"]:
        lines.append(f"- {condition}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这套规则没有训练模型，因此没有训练集/测试集意义上的模型泛化证明；它只能通过固定规则的时间分段、成本、执行延迟和参数敏感性检查。",
            "- 当前数据是新浪原始收盘价，不是已核验的分红再投资总收益序列；这会影响与原帖数字比较。",
            "- 513100在2022-01-14出现约80%的单位价格断点，疑似份额拆分或未复权企业行为；在用独立来源和复权因子确认前，收益、回撤和信号都不能视为最终结果。",
            "- 513100是A股场内上市的QDII ETF，仍有汇率、海外市场时差、跟踪误差和场内溢价折价风险。",
        ]
    )
    return "\n".join(lines) + "\n"


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

    # The implementation below is retained temporarily because the canonical
    # generator imports its bounded calculation helpers from this module. It is
    # no longer reachable through this compatibility entry point.
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data = load_price_data(Path(data_dir), DEFAULT_SYMBOLS)
    primary_full = build_signal_table(
        data,
        lookback_days=int(config["primary_rule"]["lookback_days"]),
        moving_average_days=int(config["primary_rule"]["moving_average_days"]),
    )
    valid_columns = [
        column for column in primary_full if "return_" in column or "_ma_" in column
    ]
    valid = primary_full.loc[primary_full[valid_columns].notna().all(axis=1)]
    if valid.empty:
        raise ValueError("primary rule has no rows after indicator warm-up")
    primary = primary_full.loc[primary_full["date"] >= valid["date"].min()].reset_index(
        drop=True
    )

    total_costs = config["evaluation"]["costs_total_bps"]
    variants: dict[str, object] = {}
    for name, total_bps in total_costs.items():
        result = _metrics_without_warmup(primary, side_cost_bps=float(total_bps) / 2.0)
        variants[f"{name}_total_{total_bps}bps"] = {
            "metrics": result["metrics"],
            "total_cost_bps": float(total_bps),
            "periods": result["periods"],
        }
    delayed = _metrics_without_warmup(
        primary,
        side_cost_bps=float(total_costs["baseline"]) / 2.0,
        lag_days=int(config["evaluation"]["delayed_execution_lag_days"]),
    )
    variants["delayed_one_session"] = {
        "metrics": delayed["metrics"],
        "total_cost_bps": float(total_costs["baseline"]),
        "additional_execution_lag_days": int(
            config["evaluation"]["delayed_execution_lag_days"]
        )
        - 1,
        "periods": delayed["periods"],
    }

    baseline_key = f"baseline_total_{total_costs['baseline']}bps"
    stress_key = f"stress_total_{total_costs['stress']}bps"
    baseline_periods = variants[baseline_key]["periods"]
    strict_quarter_labels = complete_quarter_labels(primary)
    all_available_quarter_labels = _available_finished_quarter_labels(primary)
    quarters = _quarter_report(baseline_periods, strict_quarter_labels)
    all_available_quarters = _quarter_report(
        baseline_periods, all_available_quarter_labels
    )
    earlier_comparison_total_bps = float(
        config["evaluation"]["earlier_comparison_total_bps"]
    )
    earlier_comparison = _metrics_without_warmup(
        primary, side_cost_bps=earlier_comparison_total_bps / 2.0
    )
    earlier_comparison_quarters = _quarter_report(
        earlier_comparison["periods"],
        all_available_quarter_labels,
    )
    parameter_grid = evaluate_parameter_grid(
        data,
        lookbacks=[int(value) for value in config["evaluation"]["parameter_lookbacks"]],
        moving_averages=[
            int(value) for value in config["evaluation"]["parameter_moving_averages"]
        ],
        side_cost_bps=float(total_costs["baseline"]) / 2.0,
    )
    discontinuities = price_discontinuity_report(data)
    evidence_gaps = [
        "independent_source_cross_check_not_done",
        "dividend_and_split_adjustment_not_verified",
        "sixty_trading_day_paper_tracking_not_done",
    ]
    if not discontinuities["passed"]:
        evidence_gaps.insert(0, "unresolved_price_discontinuity")
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "robustness_evaluated",
        "strategy_id": config["strategy_id"],
        "config": config,
        "data": {
            "source": config["data"]["source"],
            "adjustment": config["data"]["adjustment"],
            "symbols": list(DEFAULT_SYMBOLS),
            "start_date": str(primary["date"].min().date()),
            "end_date": str(primary["date"].max().date()),
            "common_trading_days_after_warmup": int(len(primary)),
            "independent_source_cross_check": False,
            "dividend_total_return_verified": False,
        },
        "primary_rule": {
            "lookback_days": int(config["primary_rule"]["lookback_days"]),
            "moving_average_days": int(config["primary_rule"]["moving_average_days"]),
            "signal_at": "close",
            "execution_at": "next_open",
            "selection_from_parameter_grid": False,
        },
        "variants": {
            name: {key: value for key, value in item.items() if key != "periods"}
            for name, item in variants.items()
        },
        "quarterly_baseline": quarters,
        "quarterly_all_available": all_available_quarters,
        "quarter_summary": {
            "strict_complete_count": len(quarters),
            "strict_complete_positive_count": int(
                sum(bool(row["positive"]) for row in quarters)
            ),
            "strict_complete_positive_ratio": (
                float(sum(bool(row["positive"]) for row in quarters) / len(quarters))
                if quarters
                else None
            ),
            "all_available_count": len(all_available_quarters),
            "all_available_positive_count": int(
                sum(bool(row["positive"]) for row in all_available_quarters)
            ),
            "all_available_positive_ratio": (
                float(
                    sum(bool(row["positive"]) for row in all_available_quarters)
                    / len(all_available_quarters)
                )
                if all_available_quarters
                else None
            ),
            "earlier_comparison_total_bps": earlier_comparison_total_bps,
            "earlier_comparison_count": len(earlier_comparison_quarters),
            "earlier_comparison_positive_count": int(
                sum(bool(row["positive"]) for row in earlier_comparison_quarters)
            ),
            "earlier_comparison_positive_ratio": (
                float(
                    sum(bool(row["positive"]) for row in earlier_comparison_quarters)
                    / len(earlier_comparison_quarters)
                )
                if earlier_comparison_quarters
                else None
            ),
        },
        "cash_metrics": cash_metrics(primary_full),
        "parameter_grid": {key: value for key, value in parameter_grid.items()},
        "reference_metrics": _reference_metrics(primary),
        "data_quality": {
            "price_discontinuities": discontinuities,
            "independent_source_cross_check": False,
            "dividend_and_split_adjustment_verified": False,
            "paper_tracking_days_completed": 0,
            "evidence_gaps": evidence_gaps,
        },
        "failure_diagnostics": _failure_diagnostics(variants[baseline_key]["periods"]),
    }
    report["acceptance"] = _acceptance(
        baseline=variants[baseline_key],
        stress=variants[stress_key],
        delayed=variants["delayed_one_session"],
        quarters=quarters,
        parameter_grid=parameter_grid,
        config=config,
    )
    report_dir = Path(report_dir).expanduser().resolve()
    run_id = f"etf_momentum_robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = report_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "result.json", report)
    (run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_json(report_dir / "etf_momentum_robustness_latest.json", report)
    (report_dir / "etf_momentum_robustness_latest.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return run_dir


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
