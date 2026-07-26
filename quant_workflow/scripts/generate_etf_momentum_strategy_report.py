"""Generate the canonical, chart-ready ETF momentum strategy report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
    compare_claim,
    load_price_data,
    normalize_price_frame,
    sanitize_json_value,
    simulate_open_to_open,
    summarize_periods,
    with_execution_lag,
)
from evaluate_etf_momentum_robustness import (  # noqa: E402
    _acceptance,
    _available_finished_quarter_labels,
    _failure_diagnostics,
    _quarter_report,
    cash_metrics,
    complete_quarter_labels,
    evaluate_parameter_grid,
    price_discontinuity_report,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    resolve_dataset,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "etf_momentum_robustness.yaml"
)
DEFAULT_DATA_DIR = (
    quant_data_root() / "parquet" / "etf_1d" / "etf_momentum_rotation" / "v1"
)
DEFAULT_ADJUSTED_DATA_DIR = (
    quant_data_root()
    / "parquet"
    / "etf_1d"
    / "etf_momentum_rotation"
    / "eastmoney_qfq"
)
DEFAULT_TENCENT_ADJUSTED_DATA_DIR = (
    quant_data_root()
    / "parquet"
    / "etf_1d"
    / "etf_momentum_rotation"
    / "tencent_qfq"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_momentum_rotation"
RAW_SOURCE = "akshare.fund_etf_hist_sina"
ADJUSTED_SOURCE = "akshare.fund_etf_hist_em_qfq"
TENCENT_ADJUSTED_SOURCE = "tencent.ifzq_qfq"
MINIMUM_ADJUSTED_COVERAGE = 0.98
RETURN_AGREEMENT_TOLERANCE = 0.005
RETURN_AGREEMENT_MINIMUM = 0.95
START_DATE_SENSITIVITY = (
    "2019-01-02",
    "2020-07-27",
    "2021-01-04",
    "2022-01-04",
    "2023-01-03",
    "2024-01-02",
    "2025-01-02",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    content = (
        json.dumps(
            sanitize_json_value(dict(payload)),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n"
    )
    _atomic_text(path, content)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_data(data: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    missing = set(DEFAULT_SYMBOLS) - set(data)
    if missing:
        raise ValueError(f"ETF data is missing symbols: {', '.join(sorted(missing))}")
    return {
        symbol: normalize_price_frame(data[symbol], symbol=symbol)
        for symbol in DEFAULT_SYMBOLS
    }


def _common_dates(data: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    common: pd.DatetimeIndex | None = None
    for symbol in DEFAULT_SYMBOLS:
        dates = pd.DatetimeIndex(pd.to_datetime(data[symbol]["date"])).normalize()
        common = dates if common is None else common.intersection(dates)
    return common.sort_values() if common is not None else pd.DatetimeIndex([])


def _return_agreement(
    raw_data: Mapping[str, pd.DataFrame],
    adjusted_data: Mapping[str, pd.DataFrame],
) -> float | None:
    comparisons: list[pd.Series] = []
    for symbol in DEFAULT_SYMBOLS:
        raw = raw_data[symbol][["date", "close"]].rename(columns={"close": "raw"})
        adjusted = adjusted_data[symbol][["date", "close"]].rename(
            columns={"close": "adjusted"}
        )
        merged = raw.merge(adjusted, on="date", how="inner")
        raw_return = merged["raw"].pct_change()
        adjusted_return = merged["adjusted"].pct_change()
        usable = raw_return.notna() & adjusted_return.notna()
        usable &= raw_return.abs() <= 0.30
        usable &= adjusted_return.abs() <= 0.30
        if usable.any():
            comparisons.append(
                (raw_return.loc[usable] - adjusted_return.loc[usable]).abs()
                <= RETURN_AGREEMENT_TOLERANCE
            )
    if not comparisons:
        return None
    combined = pd.concat(comparisons, ignore_index=True)
    return float(combined.mean()) if len(combined) else None


def select_price_source(
    raw_data: Mapping[str, pd.DataFrame],
    adjusted_data: Mapping[str, pd.DataFrame] | None,
    *,
    minimum_coverage: float = MINIMUM_ADJUSTED_COVERAGE,
    adjusted_source: str = ADJUSTED_SOURCE,
    adjusted_manifest: Mapping[str, object] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Select qfq data only after bounded coverage and anomaly checks pass."""

    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    raw = _normalize_data(raw_data)
    raw_discontinuities = price_discontinuity_report(raw)
    raw_common = _common_dates(raw)
    selected = raw
    status = "unverified_data"
    selected_source = RAW_SOURCE
    adjusted_coverage = 0.0
    agreement: float | None = None
    adjusted_discontinuities: dict[str, object] | None = None
    adjusted_available = adjusted_data is not None and set(DEFAULT_SYMBOLS).issubset(
        adjusted_data
    )

    if adjusted_available:
        adjusted = _normalize_data(adjusted_data or {})
        adjusted_common = _common_dates(adjusted)
        adjusted_coverage = (
            len(adjusted_common.intersection(raw_common)) / len(raw_common)
            if len(raw_common)
            else 0.0
        )
        adjusted_discontinuities = price_discontinuity_report(adjusted)
        agreement = _return_agreement(raw, adjusted)
        if (
            adjusted_coverage >= minimum_coverage
            and adjusted_discontinuities["passed"]
            and agreement is not None
            and agreement >= RETURN_AGREEMENT_MINIMUM
        ):
            selected = adjusted
            status = "verified_qfq"
            selected_source = adjusted_source

    evidence_gaps = ["sixty_trading_day_paper_tracking_not_done"]
    if status != "verified_qfq":
        if not raw_discontinuities["passed"]:
            evidence_gaps.insert(0, "unresolved_price_discontinuity")
        evidence_gaps.extend(
            [
                "independent_source_cross_check_not_done",
                "dividend_and_split_adjustment_not_verified",
            ]
        )
    else:
        evidence_gaps.insert(0, "dividend_total_return_not_verified")

    selected_discontinuities = price_discontinuity_report(selected)
    resolved_actions = []
    if status == "verified_qfq" and adjusted_manifest is not None:
        manifest_actions = adjusted_manifest.get("corporate_actions", [])
        if isinstance(manifest_actions, list):
            resolved_actions = manifest_actions
    quality = {
        "status": status,
        "selected_source": selected_source,
        "selected_adjustment": "qfq" if status == "verified_qfq" else "raw_close",
        "raw_source": RAW_SOURCE,
        "adjusted_source": adjusted_source,
        "adjusted_source_available": bool(adjusted_available),
        "coverage_ratio": adjusted_coverage,
        "minimum_coverage": minimum_coverage,
        "return_agreement_ratio": agreement,
        "return_agreement_minimum": RETURN_AGREEMENT_MINIMUM,
        "price_discontinuities": selected_discontinuities,
        "raw_price_discontinuities": raw_discontinuities,
        "adjusted_price_discontinuities": adjusted_discontinuities,
        "independent_source_cross_check": status == "verified_qfq",
        "split_adjustment_verified": status == "verified_qfq",
        "dividend_total_return_verified": False,
        "paper_tracking_days_completed": 0,
        "resolved_corporate_actions": resolved_actions,
        "evidence_gaps": evidence_gaps,
    }
    return selected, quality


def _annualized(total_return: float, periods: int) -> float:
    if periods <= 0:
        return float("nan")
    wealth = 1.0 + total_return
    if wealth <= 0:
        return -1.0
    return wealth ** (252.0 / periods) - 1.0


def _curve_metrics(curve: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(curve, errors="coerce").dropna()
    if values.empty:
        return {
            "periods": 0,
            "total_return": None,
            "annualized_return": None,
            "max_drawdown": None,
        }
    drawdown = values / values.cummax() - 1.0
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    return {
        "periods": max(len(values) - 1, 0),
        "total_return": total_return,
        "annualized_return": _annualized(total_return, max(len(values) - 1, 0)),
        "max_drawdown": float(drawdown.min()),
    }


def _benchmark_curves(
    signal_table: pd.DataFrame, strategy_curve: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = signal_table[["date", *[f"{symbol}_open" for symbol in DEFAULT_SYMBOLS]]].copy()
    for symbol in DEFAULT_SYMBOLS:
        frame[symbol] = frame[f"{symbol}_open"] / float(frame[f"{symbol}_open"].iloc[0])
    frame["equal_weight"] = 0.5 * frame[DEFAULT_SYMBOLS[0]] + 0.5 * frame[DEFAULT_SYMBOLS[1]]
    strategy = strategy_curve.rename(columns={"wealth": "strategy"})
    frame = frame.merge(strategy[["date", "strategy"]], on="date", how="left")
    frame["strategy"] = frame["strategy"].ffill().fillna(1.0)
    metrics = {
        symbol: _curve_metrics(frame[symbol]) for symbol in DEFAULT_SYMBOLS
    }
    metrics["equal_weight"] = _curve_metrics(frame["equal_weight"])
    return frame, metrics


def drawdown_episodes(curve: pd.DataFrame) -> list[dict[str, object]]:
    """Return completed and active underwater episodes ordered by severity."""

    if curve.empty:
        return []
    frame = curve[["date", "wealth"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["wealth"] = pd.to_numeric(frame["wealth"], errors="coerce")
    frame = frame.dropna().sort_values("date", kind="stable").reset_index(drop=True)
    if frame.empty:
        return []
    running_peak = frame["wealth"].cummax()
    frame["drawdown"] = frame["wealth"] / running_peak - 1.0
    episodes: list[dict[str, object]] = []
    start_index: int | None = None
    for index, row in frame.iterrows():
        if row["drawdown"] < 0 and start_index is None:
            start_index = max(index - 1, 0)
        recovered = start_index is not None and row["drawdown"] >= 0 and index > start_index
        last_row = index == len(frame) - 1
        if start_index is None or (not recovered and not last_row):
            continue
        end_index = index if recovered else len(frame) - 1
        scoped = frame.iloc[start_index : end_index + 1]
        trough_index = int(scoped["drawdown"].idxmin())
        episodes.append(
            {
                "start_date": str(frame.iloc[start_index]["date"].date()),
                "trough_date": str(frame.loc[trough_index, "date"].date()),
                "recovery_date": str(row["date"].date()) if recovered else None,
                "max_drawdown": float(frame.loc[trough_index, "drawdown"]),
                "trading_days": int(end_index - start_index),
                "recovered": recovered,
            }
        )
        start_index = None
    return sorted(episodes, key=lambda item: float(item["max_drawdown"]))


def holding_episode_summary(periods: pd.DataFrame) -> dict[str, object]:
    if periods.empty:
        return {
            "episode_count": 0,
            "invested_episode_count": 0,
            "cash_episode_count": 0,
            "average_trading_days": None,
            "median_trading_days": None,
            "invested_win_rate": None,
            "episodes": [],
        }
    frame = periods.copy().reset_index(drop=True)
    frame["position"] = frame["position"].astype(str)
    group = frame["position"].ne(frame["position"].shift()).cumsum()
    episodes: list[dict[str, object]] = []
    for _, scoped in frame.groupby(group, sort=False):
        returns = pd.to_numeric(scoped["net_return"], errors="coerce").fillna(0.0)
        total_return = float((1.0 + returns).prod() - 1.0)
        position = str(scoped.iloc[0]["position"])
        episodes.append(
            {
                "position": position,
                "start_date": str(pd.Timestamp(scoped.iloc[0]["date"]).date()),
                "end_date": str(pd.Timestamp(scoped.iloc[-1]["next_date"]).date()),
                "trading_days": int(len(scoped)),
                "total_return": total_return,
                "positive": total_return > 0,
            }
        )
    invested = [item for item in episodes if item["position"] != CASH]
    lengths = [int(item["trading_days"]) for item in episodes]
    return {
        "episode_count": len(episodes),
        "invested_episode_count": len(invested),
        "cash_episode_count": len(episodes) - len(invested),
        "average_trading_days": float(pd.Series(lengths).mean()) if lengths else None,
        "median_trading_days": float(pd.Series(lengths).median()) if lengths else None,
        "invested_win_rate": (
            float(sum(bool(item["positive"]) for item in invested) / len(invested))
            if invested
            else None
        ),
        "episodes": episodes,
    }


def _period_report(periods: pd.DataFrame, *, frequency: str) -> list[dict[str, object]]:
    if periods.empty:
        return []
    dates = pd.to_datetime(periods["date"], errors="coerce").dt.normalize()
    labels = sorted(dates.dt.to_period(frequency).dropna().unique())
    result: list[dict[str, object]] = []
    for label in labels:
        summary = summarize_periods(
            periods,
            start_date=label.start_time.normalize(),
            end_date=label.end_time.normalize(),
        )
        if int(summary["periods"]) == 0:
            continue
        result.append(
            {
                "period": str(label),
                **summary,
                "positive": bool(float(summary["total_return"]) > 0),
            }
        )
    return result


def _monthly_allocations(signal_table: pd.DataFrame) -> list[dict[str, object]]:
    frame = signal_table[["date", "target_position"]].copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M")
    rows: list[dict[str, object]] = []
    for month, scoped in frame.groupby("month", sort=True):
        total = len(scoped)
        rows.append(
            {
                "month": str(month),
                "SZ159915": float((scoped["target_position"] == "SZ159915").sum() / total),
                "SH513100": float((scoped["target_position"] == "SH513100").sum() / total),
                "CASH": float((scoped["target_position"] == CASH).sum() / total),
            }
        )
    return rows


def _latest_signal(signal_table: pd.DataFrame, data_status: str) -> dict[str, object]:
    row = signal_table.iloc[-1]
    details: dict[str, object] = {}
    for symbol in DEFAULT_SYMBOLS:
        return_columns = [column for column in signal_table if column.startswith(f"{symbol}_return_")]
        ma_columns = [column for column in signal_table if column.startswith(f"{symbol}_ma_")]
        details[symbol] = {
            "close": float(row[f"{symbol}_close"]),
            "lookback_return": float(row[return_columns[0]]) if return_columns and pd.notna(row[return_columns[0]]) else None,
            "moving_average": float(row[ma_columns[0]]) if ma_columns and pd.notna(row[ma_columns[0]]) else None,
            "above_moving_average": bool(row[f"{symbol}_above_ma"]),
            "rank": int(row[f"{symbol}_rank"]) if pd.notna(row[f"{symbol}_rank"]) else None,
        }
    return {
        "as_of_date": str(pd.Timestamp(row["date"]).date()),
        "signal": str(row["signal"]),
        "position_at_latest_open": str(row["target_position"]),
        "leader": str(row["top_symbol"]),
        "next_open_research_target": str(row["signal"]),
        "reliability": data_status,
        "usable_for_trading": False,
        "details": details,
    }


def _reference_checks(acceptance: Mapping[str, object]) -> list[dict[str, object]]:
    value_keys = {
        "positive_quarter_ratio": "positive_quarter_ratio",
        "baseline_annualized_return": "baseline_annualized_return",
        "baseline_max_drawdown": "baseline_max_drawdown_abs",
        "stress_annualized_return": "stress_annualized_return",
        "delayed_annualized_return": "delayed_annualized_return",
        "parameter_positive_ratio": "parameter_positive_ratio",
        "parameter_max_drawdown_pass_ratio": "parameter_max_drawdown_pass_ratio",
    }
    operators = {
        "positive_quarter_ratio": ">=",
        "baseline_annualized_return": ">=",
        "baseline_max_drawdown": "<=",
        "stress_annualized_return": ">",
        "delayed_annualized_return": ">",
        "parameter_positive_ratio": ">=",
        "parameter_max_drawdown_pass_ratio": ">=",
    }
    checks = acceptance["advisory_checks"]
    values = acceptance["values"]
    thresholds = acceptance["reference_values"]
    return [
        {
            "id": name,
            "value": values[value_keys[name]],
            "threshold": thresholds[value_keys[name]],
            "operator": operators[name],
            "met": bool(met),
            "decision": "reference_met" if met else "manual_review",
        }
        for name, met in checks.items()
    ]


def _git_metadata(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def build_strategy_report(
    data: Mapping[str, pd.DataFrame],
    *,
    config: Mapping[str, object],
    data_quality: Mapping[str, object],
    generated_at: str,
    data_manifest_hash: str | None,
    git_commit: str | None,
    git_dirty: bool | None,
) -> dict[str, object]:
    lookback = int(config["primary_rule"]["lookback_days"])
    moving_average = int(config["primary_rule"]["moving_average_days"])
    full = build_signal_table(
        data,
        lookback_days=lookback,
        moving_average_days=moving_average,
    )
    valid_columns = [column for column in full if "return_" in column or "_ma_" in column]
    valid = full.loc[full[valid_columns].notna().all(axis=1)]
    if valid.empty:
        raise ValueError("primary rule has no rows after indicator warm-up")
    primary = full.loc[full["date"] >= valid["date"].min()].reset_index(drop=True)

    total_costs = config["evaluation"]["costs_total_bps"]
    variants: dict[str, dict[str, object]] = {}
    for name, total_bps in total_costs.items():
        result = simulate_open_to_open(
            with_execution_lag(primary, lag_days=1),
            side_cost_bps=float(total_bps) / 2.0,
        )
        variants[name] = {
            "total_cost_bps": float(total_bps),
            "metrics": result["metrics"],
            "periods": result["periods"],
            "equity_curve": result["equity_curve"],
        }
    delayed_result = simulate_open_to_open(
        with_execution_lag(
            primary,
            lag_days=int(config["evaluation"]["delayed_execution_lag_days"]),
        ),
        side_cost_bps=float(total_costs["baseline"]) / 2.0,
    )
    delayed = {
        "metrics": delayed_result["metrics"],
        "periods": delayed_result["periods"],
        "equity_curve": delayed_result["equity_curve"],
        "additional_execution_lag_days": int(
            config["evaluation"]["delayed_execution_lag_days"]
        )
        - 1,
    }
    baseline = variants["baseline"]
    stress = variants["stress"]

    strict_labels = complete_quarter_labels(primary)
    available_labels = _available_finished_quarter_labels(primary)
    strict_quarters = _quarter_report(baseline["periods"], strict_labels)
    available_quarters = _quarter_report(baseline["periods"], available_labels)
    earlier_total_bps = float(config["evaluation"]["earlier_comparison_total_bps"])
    earlier = simulate_open_to_open(
        with_execution_lag(primary, lag_days=1),
        side_cost_bps=earlier_total_bps / 2.0,
    )
    earlier_quarters = _quarter_report(earlier["periods"], available_labels)
    parameter_grid = evaluate_parameter_grid(
        data,
        lookbacks=[int(value) for value in config["evaluation"]["parameter_lookbacks"]],
        moving_averages=[
            int(value) for value in config["evaluation"]["parameter_moving_averages"]
        ],
        side_cost_bps=float(total_costs["baseline"]) / 2.0,
    )
    acceptance = _acceptance(
        baseline=baseline,
        stress=stress,
        delayed=delayed,
        quarters=strict_quarters,
        parameter_grid=parameter_grid,
        config=config,
    )

    benchmark_curve, benchmark_metrics = _benchmark_curves(
        primary, baseline["equity_curve"]
    )
    strategy_drawdown = (
        baseline["equity_curve"]["wealth"]
        / baseline["equity_curve"]["wealth"].cummax()
        - 1.0
    )
    quarterly = _period_report(baseline["periods"], frequency="Q")
    yearly = _period_report(baseline["periods"], frequency="Y")
    sensitivity: list[dict[str, object]] = []
    for start_date in START_DATE_SENSITIVITY:
        window = primary.loc[primary["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
        if len(window) < 2:
            continue
        result = simulate_open_to_open(
            with_execution_lag(window, lag_days=1),
            side_cost_bps=float(total_costs["baseline"]) / 2.0,
        )
        sensitivity.append(
            {"start_date": start_date, "metrics": result["metrics"]}
        )

    quarter_consistency = {
        "strict_complete_count": len(strict_quarters),
        "strict_complete_positive_count": int(
            sum(bool(row["positive"]) for row in strict_quarters)
        ),
        "strict_complete_positive_ratio": (
            float(
                sum(bool(row["positive"]) for row in strict_quarters)
                / len(strict_quarters)
            )
            if strict_quarters
            else None
        ),
        "all_available_count": len(available_quarters),
        "all_available_positive_count": int(
            sum(bool(row["positive"]) for row in available_quarters)
        ),
        "all_available_positive_ratio": (
            float(
                sum(bool(row["positive"]) for row in available_quarters)
                / len(available_quarters)
            )
            if available_quarters
            else None
        ),
        "earlier_comparison_total_bps": earlier_total_bps,
        "earlier_comparison_count": len(earlier_quarters),
        "earlier_comparison_positive_count": int(
            sum(bool(row["positive"]) for row in earlier_quarters)
        ),
        "earlier_comparison_positive_ratio": (
            float(
                sum(bool(row["positive"]) for row in earlier_quarters)
                / len(earlier_quarters)
            )
            if earlier_quarters
            else None
        ),
    }

    data_status = str(data_quality["status"])
    failure_diagnostics = _failure_diagnostics(baseline["periods"])
    holding_summary = holding_episode_summary(baseline["periods"])
    episodes = drawdown_episodes(baseline["equity_curve"])
    baseline_metrics = baseline["metrics"]
    report = {
        "schema_version": 1,
        "status": "manual_review_required",
        "generated_at": generated_at,
        "strategy_id": str(config["strategy_id"]),
        "summary": {
            "status": "manual_review_required",
            "verdict_code": (
                "continue_research_data_unverified"
                if data_status != "verified_qfq"
                else "continue_research_manual_review"
            ),
            "data_status": data_status,
            "usable_for_trading": False,
            "baseline_total_return": baseline_metrics["total_return"],
            "baseline_annualized_return": baseline_metrics["annualized_return"],
            "baseline_max_drawdown": baseline_metrics["max_drawdown"],
            "stress_annualized_return": stress["metrics"]["annualized_return"],
            "delayed_annualized_return": delayed["metrics"]["annualized_return"],
            "unmet_reference_count": len(acceptance["advisory_not_met"]),
            "evidence_gap_count": len(data_quality["evidence_gaps"]),
        },
        "strategy": {
            "symbols": list(DEFAULT_SYMBOLS),
            "lookback_days": lookback,
            "moving_average_days": moving_average,
            "signal_time": "close",
            "execution_time": "next_open",
            "buy_rule": "20-day return leader and close above its own 20-day moving average",
            "sell_rule": "held ETF is no longer the leader or closes below its own moving average",
            "cash_rule": "the return leader is not above its own moving average",
            "automatic_orders": False,
            "broker_connection": False,
        },
        "latest_signal": _latest_signal(primary, data_status),
        "performance": {
            "data_range": {
                "start_date": str(pd.Timestamp(primary["date"].min()).date()),
                "end_date": str(pd.Timestamp(primary["date"].max()).date()),
                "trading_days": int(len(primary)),
            },
            "variants": {
                name: {
                    "total_cost_bps": item["total_cost_bps"],
                    "metrics": item["metrics"],
                }
                for name, item in variants.items()
            },
            "delayed_execution": {
                "additional_execution_lag_days": delayed[
                    "additional_execution_lag_days"
                ],
                "metrics": delayed["metrics"],
            },
            "benchmarks": benchmark_metrics,
            "quarterly": quarterly,
            "yearly": yearly,
            "start_date_sensitivity": sensitivity,
            "claim_comparison": compare_claim(variants["zero"]),
        },
        "stability": {
            "decision_mode": "manual_review",
            "automatic_rejection_disabled": True,
            "reference_checks": _reference_checks(acceptance),
            "advisory_not_met": acceptance["advisory_not_met"],
            "quarter_consistency": quarter_consistency,
            "parameter_grid": parameter_grid,
            "cost_stress": [
                {
                    "name": name,
                    "total_cost_bps": item["total_cost_bps"],
                    "annualized_return": item["metrics"]["annualized_return"],
                    "max_drawdown": item["metrics"]["max_drawdown"],
                }
                for name, item in variants.items()
            ],
        },
        "risk": {
            "cash_metrics": cash_metrics(full),
            "holding_summary": holding_summary,
            "drawdown_episodes": episodes[:5],
            "longest_underwater_trading_days": max(
                (int(item["trading_days"]) for item in episodes), default=0
            ),
            "failure_diagnostics": failure_diagnostics,
            "known_failure_conditions": failure_diagnostics.get(
                "known_failure_conditions", []
            ),
        },
        "chart_data": {
            "wealth": [
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "strategy": float(row["strategy"]),
                    "SZ159915": float(row["SZ159915"]),
                    "SH513100": float(row["SH513100"]),
                    "equal_weight": float(row["equal_weight"]),
                }
                for _, row in benchmark_curve.iterrows()
            ],
            "drawdown": [
                {
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "drawdown": float(value),
                }
                for (_, row), value in zip(
                    baseline["equity_curve"].iterrows(), strategy_drawdown
                )
            ],
            "quarterly_returns": [
                {
                    "period": item["period"],
                    "return": item["total_return"],
                    "positive": item["positive"],
                }
                for item in quarterly
            ],
            "yearly_returns": [
                {
                    "period": item["period"],
                    "return": item["total_return"],
                    "positive": item["positive"],
                }
                for item in yearly
            ],
            "monthly_allocations": _monthly_allocations(primary),
            "cost_stress": [
                {
                    "label": f"{item['total_cost_bps']:g}bp",
                    "annualized_return": item["metrics"]["annualized_return"],
                    "max_drawdown": item["metrics"]["max_drawdown"],
                }
                for item in variants.values()
            ],
            "start_date_sensitivity": [
                {
                    "start_date": item["start_date"],
                    "annualized_return": item["metrics"]["annualized_return"],
                    "max_drawdown": item["metrics"]["max_drawdown"],
                }
                for item in sensitivity
            ],
            "anomaly_dates": [
                item["date"]
                for item in data_quality["price_discontinuities"].get(
                    "events", []
                )
            ],
        },
        "data_quality": dict(data_quality),
        "reproducibility": {
            "config_hash": hashlib.sha256(
                yaml.safe_dump(dict(config), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "data_manifest_hash": data_manifest_hash,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "generated_at": generated_at,
        },
    }
    return sanitize_json_value(report)


def _summary_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    quality = report["data_quality"]
    quarter = report["stability"]["quarter_consistency"]
    return "\n".join(
        [
            "# ETF动量轮动完整策略报告摘要",
            "",
            "> 完整图表请在 DSA `/backtest/etf-momentum` 页面查看。本文件只用于审查和兼容。",
            "",
            f"- 状态：`{report['status']}`。",
            f"- 数据状态：`{summary['data_status']}`，不可用于自动或人工直接下单。",
            f"- 20bp成本年化收益：{summary['baseline_annualized_return']:.2%}。",
            f"- 20bp成本总收益：{summary['baseline_total_return']:.2%}。",
            f"- 最大回撤：{summary['baseline_max_drawdown']:.2%}。",
            f"- 严格完整季度正收益比例：{quarter['strict_complete_positive_ratio']:.2%}。",
            f"- 含首个不完整季度的20bp口径：{quarter['all_available_positive_ratio']:.2%}。",
            f"- 早期15bp可比口径：{quarter['earlier_comparison_positive_ratio']:.2%}。",
            "",
            "## 尚未完成的证据",
            "",
            *[f"- `{item}`" for item in quality["evidence_gaps"]],
            "",
        ]
    )


def _load_optional_adjusted_data(path: Path) -> dict[str, pd.DataFrame] | None:
    if not all((path / f"{symbol}.parquet").exists() for symbol in DEFAULT_SYMBOLS):
        return None
    return load_price_data(path, DEFAULT_SYMBOLS)


def resolve_canonical_etf_bars(
    quant_data_dir: Path,
) -> tuple[Path, dict[str, object]]:
    resolved = resolve_dataset(
        "etf_momentum_daily_qfq",
        quant_data_root=Path(quant_data_dir),
        require_basis="qfq",
    )
    bars = resolved.snapshot_path / "bars"
    if not bars.is_dir():
        raise DataContractError("canonical ETF snapshot bars directory is missing")
    return bars, resolved.manifest


def _canonical_manifest_for_quality(
    snapshot_dir: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(manifest)
    actions_path = snapshot_dir / str(
        manifest.get("corporate_actions_path") or "corporate_actions.parquet"
    )
    if actions_path.is_file():
        actions = pd.read_parquet(actions_path)
        records: list[dict[str, object]] = []
        for record in actions.to_dict(orient="records"):
            normalized = {
                key: (
                    value.date().isoformat()
                    if isinstance(value, pd.Timestamp)
                    else value
                )
                for key, value in record.items()
            }
            if "ex_date" in normalized and "split_date" not in normalized:
                normalized["split_date"] = normalized["ex_date"]
            if normalized.get("action_type") == "share_split":
                normalized["type"] = "share_split"
            ratio = normalized.get("ratio")
            if isinstance(ratio, (int, float)):
                normalized["ratio"] = f"1:{ratio:g}"
            records.append(normalized)
        payload["corporate_actions"] = records
    return payload


def _load_optional_manifest(path: Path) -> dict[str, object] | None:
    manifest_path = path / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def run(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    adjusted_data_dir: Path = DEFAULT_ADJUSTED_DATA_DIR,
    tencent_adjusted_data_dir: Path = DEFAULT_TENCENT_ADJUSTED_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    config_path: Path = DEFAULT_CONFIG,
    generated_at: str | None = None,
    quant_data_dir: Path | None = None,
) -> Path:
    raw_data = load_price_data(Path(data_dir), DEFAULT_SYMBOLS)
    selected_data, quality = select_price_source(raw_data, None)
    selected_adjusted_manifest: Path | None = None
    adjusted_candidates: list[
        tuple[str, Path, Mapping[str, object] | None, Path | None]
    ] = []
    catalog_root = Path(quant_data_dir).expanduser().resolve() if quant_data_dir else None
    if catalog_root is None and Path(data_dir).resolve() == DEFAULT_DATA_DIR.resolve():
        catalog_root = quant_data_root().resolve()
    if catalog_root is not None:
        try:
            canonical_dir, canonical_manifest = resolve_canonical_etf_bars(
                catalog_root
            )
        except DataContractError:
            pass
        else:
            provider = str(
                (canonical_manifest.get("source") or {}).get("provider", "catalog")
                if isinstance(canonical_manifest.get("source"), dict)
                else "catalog"
            )
            source_name = (
                TENCENT_ADJUSTED_SOURCE
                if provider == "tencent"
                else ADJUSTED_SOURCE
                if provider == "eastmoney"
                else f"{provider}.catalog_qfq"
            )
            adjusted_candidates.append(
                (
                    source_name,
                    canonical_dir,
                    _canonical_manifest_for_quality(
                        canonical_dir.parent, canonical_manifest
                    ),
                    canonical_dir.parent / "DATASET_MANIFEST.json",
                )
            )
    adjusted_candidates.extend(
        [
            (ADJUSTED_SOURCE, Path(adjusted_data_dir), None, None),
            (TENCENT_ADJUSTED_SOURCE, Path(tencent_adjusted_data_dir), None, None),
        ]
    )
    for source_name, candidate_dir, supplied_manifest, supplied_manifest_path in adjusted_candidates:
        adjusted_data = _load_optional_adjusted_data(candidate_dir)
        if adjusted_data is None:
            continue
        manifest = supplied_manifest or _load_optional_manifest(candidate_dir)
        candidate_data, candidate_quality = select_price_source(
            raw_data,
            adjusted_data,
            adjusted_source=source_name,
            adjusted_manifest=manifest,
        )
        selected_data, quality = candidate_data, candidate_quality
        if candidate_quality["status"] == "verified_qfq":
            selected_adjusted_manifest = supplied_manifest_path or (
                candidate_dir / "SOURCE_MANIFEST.json"
            )
            break
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    timestamp = generated_at or pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    manifest_path = Path(data_dir) / "SOURCE_MANIFEST.json"
    manifest_hashes = [
        _sha256(path)
        for path in (manifest_path, selected_adjusted_manifest)
        if path is not None and path.exists()
    ]
    manifest_hash = (
        hashlib.sha256("|".join(manifest_hashes).encode("ascii")).hexdigest()
        if manifest_hashes
        else None
    )
    git_commit, git_dirty = _git_metadata(Path(__file__).resolve().parents[2])
    report = build_strategy_report(
        selected_data,
        config=config,
        data_quality=quality,
        generated_at=timestamp,
        data_manifest_hash=manifest_hash,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    signal_table = build_signal_table(
        selected_data,
        lookback_days=int(config["primary_rule"]["lookback_days"]),
        moving_average_days=int(config["primary_rule"]["moving_average_days"]),
    )

    report_dir = Path(report_dir).expanduser().resolve()
    run_stamp = pd.Timestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    run_dir = report_dir / "runs" / f"etf_momentum_strategy_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    markdown = _summary_markdown(report)
    _atomic_json(run_dir / "strategy_report.json", report)
    _atomic_text(run_dir / "strategy_report.md", markdown)
    _atomic_parquet(signal_table, run_dir / "signals.parquet")
    _atomic_json(report_dir / "strategy_report_latest.json", report)
    _atomic_text(report_dir / "strategy_report_latest.md", markdown)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the complete two-ETF momentum strategy report."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--adjusted-data-dir", type=Path, default=DEFAULT_ADJUSTED_DATA_DIR
    )
    parser.add_argument(
        "--tencent-adjusted-data-dir",
        type=Path,
        default=DEFAULT_TENCENT_ADJUSTED_DATA_DIR,
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run(
        data_dir=args.data_dir,
        adjusted_data_dir=args.adjusted_data_dir,
        tencent_adjusted_data_dir=args.tencent_adjusted_data_dir,
        report_dir=args.report_dir,
        config_path=args.config,
    )
    print(f"ETF momentum strategy report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
