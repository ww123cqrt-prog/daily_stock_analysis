#!/usr/bin/env python3
"""Run the fixed three-ETF defensive rotation research preview."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from etf_momentum_rotation import (  # noqa: E402
    CASH,
    build_signal_table,
    normalize_price_frame,
    sanitize_json_value,
    simulate_open_to_open,
    summarize_periods,
)
from etf_research_preview_data import (  # noqa: E402
    LOCKED_SYMBOLS,
    EtfResearchPreview,
    resolve_etf_research_preview,
)
from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "etf_defensive_rotation.yaml"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_defensive_rotation"
INDEX_NAMES = {
    "SH000300": "沪深300",
    "SH000905": "中证500",
    "SZ399006": "创业板指",
}
VARIANT_SPECS = {
    "three_etf_defensive": {
        "symbols": LOCKED_SYMBOLS,
        "eligibility_mode": "filter_then_rank",
        "name": "三ETF防守轮动",
    },
    "two_etf_legacy_exact": {
        "symbols": LOCKED_SYMBOLS[:2],
        "eligibility_mode": "leader_then_filter",
        "name": "原两ETF精确旧规则",
    },
    "two_etf_eligible_first": {
        "symbols": LOCKED_SYMBOLS[:2],
        "eligibility_mode": "filter_then_rank",
        "name": "两ETF先过滤再排名",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            sanitize_json_value(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_text(path, content)


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(project_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "git_dirty": bool(run("status", "--porcelain")),
    }


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ETF defensive rotation config must be a mapping")
    strategy = payload.get("strategy")
    costs = payload.get("costs")
    data = payload.get("data")
    if not all(isinstance(value, Mapping) for value in (strategy, costs, data)):
        raise ValueError("ETF defensive rotation config is incomplete")
    if tuple(strategy.get("symbols", ())) != LOCKED_SYMBOLS:
        raise ValueError("strategy symbols must match the locked three-ETF contract")
    expected_rule = {
        "lookback_days": 20,
        "moving_average_days": 20,
        "eligibility_mode": "filter_then_rank",
        "signal_at": "close",
        "execution_at": "next_common_open",
        "all_ineligible_position": "CASH",
        "parameter_search": False,
    }
    for field, expected in expected_rule.items():
        actual = strategy.get(field)
        if (isinstance(expected, bool) and actual is not expected) or (
            not isinstance(expected, bool) and actual != expected
        ):
            raise ValueError(f"locked strategy config mismatch: {field}")
    if data.get("required_runtime_status") != "research_preview":
        raise ValueError("temporary ETF research must remain research_preview")
    if data.get("formal_acceptance_proven") is not False:
        raise ValueError("temporary ETF snapshot cannot claim formal acceptance")
    slippage = tuple(float(value) for value in costs.get("slippage_bps_each_side", ()))
    if slippage != (0.0, 1.0, 2.0):
        raise ValueError("slippage stress must remain locked at 0/1/2bp")
    if float(costs.get("commission_bps_each_side", -1)) != 0.5:
        raise ValueError("ETF commission must remain 0.5bp each side")
    for field in (
        "minimum_commission_applied",
        "stamp_duty_applied",
        "transfer_fee_applied",
    ):
        if costs.get(field) is not False:
            raise ValueError(f"locked costs config mismatch: {field}")
    return payload


def _common_data(data: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    missing = set(LOCKED_SYMBOLS) - set(data)
    if missing:
        raise ValueError(f"ETF data is missing symbols: {', '.join(sorted(missing))}")
    normalized = {
        symbol: normalize_price_frame(data[symbol], symbol=symbol)
        for symbol in LOCKED_SYMBOLS
    }
    common: pd.DatetimeIndex | None = None
    for symbol in LOCKED_SYMBOLS:
        dates = pd.DatetimeIndex(normalized[symbol]["date"]).normalize()
        common = dates if common is None else common.intersection(dates)
    if common is None or len(common) < 22:
        raise ValueError("three ETFs have fewer than 22 common trading days")
    common = common.sort_values()
    return {
        symbol: normalized[symbol]
        .loc[normalized[symbol]["date"].isin(common)]
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
        for symbol in LOCKED_SYMBOLS
    }


def _slippage_label(value: float) -> str:
    return f"{value:g}bp_slippage"


def evaluate_locked_variants(
    data: Mapping[str, pd.DataFrame],
    *,
    commission_bps_each_side: float,
    slippage_bps_each_side: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Evaluate only the three preregistered rules on one common calendar."""

    common = _common_data(data)
    result: dict[str, dict[str, Any]] = {}
    for variant_id, spec in VARIANT_SPECS.items():
        symbols = tuple(spec["symbols"])
        selected = {symbol: common[symbol] for symbol in symbols}
        table = build_signal_table(
            selected,
            lookback_days=20,
            moving_average_days=20,
            eligibility_mode=str(spec["eligibility_mode"]),
        )
        stress: dict[str, dict[str, Any]] = {}
        for slippage in slippage_bps_each_side:
            slippage_value = float(slippage)
            if not math.isfinite(slippage_value) or slippage_value < 0:
                raise ValueError("slippage must be finite and nonnegative")
            stress[_slippage_label(slippage_value)] = simulate_open_to_open(
                table,
                side_cost_bps=float(commission_bps_each_side) + slippage_value,
            )
        result[variant_id] = {
            "name": spec["name"],
            "rule": {
                "symbols": list(symbols),
                "lookback_days": 20,
                "moving_average_days": 20,
                "eligibility_mode": spec["eligibility_mode"],
                "signal_at": "close",
                "execution_at": "next_common_open",
            },
            "signal_table": table,
            "cost_stress": stress,
        }
    return result


def _annualized_return(returns: np.ndarray) -> float:
    if not len(returns):
        return float("nan")
    wealth = float(np.prod(1.0 + returns))
    if wealth <= 0:
        return -1.0
    return wealth ** (252.0 / len(returns)) - 1.0


def _return_metrics(returns: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "periods": 0,
            "total_return": None,
            "annualized_return": None,
            "max_drawdown": None,
        }
    wealth = np.cumprod(1.0 + values)
    curve = np.concatenate(([1.0], wealth))
    drawdown = curve / np.maximum.accumulate(curve) - 1.0
    return {
        "periods": int(len(values)),
        "total_return": float(wealth[-1] - 1.0),
        "annualized_return": _annualized_return(values),
        "max_drawdown": float(drawdown.min()),
    }


def block_bootstrap_incremental_return(
    strategy_periods: pd.DataFrame,
    baseline_periods: pd.DataFrame,
    *,
    block_days: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if block_days <= 0 or samples <= 0:
        raise ValueError("bootstrap block_days and samples must be positive")
    left = strategy_periods[["date", "net_return"]].rename(
        columns={"net_return": "strategy"}
    )
    right = baseline_periods[["date", "net_return"]].rename(
        columns={"net_return": "baseline"}
    )
    merged = left.merge(right, on="date", how="inner", validate="one_to_one")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
    if len(merged) < block_days:
        raise ValueError("bootstrap sample is shorter than one block")
    strategy = merged["strategy"].to_numpy(dtype=float)
    baseline = merged["baseline"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(samples), dtype=float)
    block_count = math.ceil(len(merged) / block_days)
    offsets = np.arange(block_days)
    for sample in range(int(samples)):
        starts = rng.integers(0, len(merged), size=block_count)
        indexes = np.concatenate(
            [((start + offsets) % len(merged)) for start in starts]
        )[: len(merged)]
        deltas[sample] = _annualized_return(strategy[indexes]) - _annualized_return(
            baseline[indexes]
        )
    return {
        "method": "circular_block_bootstrap",
        "block_days": int(block_days),
        "samples": int(samples),
        "seed": int(seed),
        "observations": int(len(merged)),
        "mean_annualized_return_delta": float(deltas.mean()),
        "median_annualized_return_delta": float(np.median(deltas)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
        "positive_probability": float((deltas > 0).mean()),
    }


def classify_incremental_value(
    evidence: Mapping[str, float],
    *,
    formally_accepted: bool,
    max_drawdown_abs_max: float,
    leave_one_year_positive_ratio_min: float,
) -> dict[str, Any]:
    if not formally_accepted:
        return {
            "status": "research_preview",
            "formal_candidate_status": None,
            "classification_enabled": False,
            "reason": "ETF snapshot has not completed formal quant_data acceptance",
        }
    hard_drawdown_pass = abs(float(evidence["max_drawdown"])) <= float(
        max_drawdown_abs_max
    )
    if (
        not hard_drawdown_pass
        or float(evidence["full_annualized_delta"]) <= 0
        or float(evidence["validation_annualized_delta"]) <= 0
    ):
        label = "no_incremental_value"
    elif (
        float(evidence["observation_annualized_delta"]) > 0
        and float(evidence["bootstrap_ci95_lower"]) > 0
        and float(evidence["leave_one_year_positive_ratio"])
        >= float(leave_one_year_positive_ratio_min)
    ):
        label = "strong_candidate"
    else:
        label = "potential_candidate"
    return {
        "status": label,
        "formal_candidate_status": label,
        "classification_enabled": True,
        "reason": "formal classification policy applied",
    }


def _qlib_calendar(provider: Path) -> tuple[str, ...]:
    path = provider / "calendars/day.txt"
    if path.is_symlink() or not path.is_file():
        raise DataContractError("Qlib benchmark calendar is unavailable")
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not values or values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise DataContractError("Qlib benchmark calendar is invalid")
    return values


def _qlib_open_series(
    provider: Path, calendar: tuple[str, ...], symbol: str
) -> pd.DataFrame | None:
    path = provider / "features" / symbol.lower() / "open.day.bin"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"Qlib benchmark feature is invalid: {symbol}")
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2 or not np.isfinite(values[0]) or values[0] != int(values[0]):
        raise DataContractError(f"Qlib benchmark feature header is invalid: {symbol}")
    start = int(values[0])
    data = values[1:]
    if start < 0 or start + len(data) > len(calendar):
        raise DataContractError(f"Qlib benchmark feature range is invalid: {symbol}")
    dates = pd.to_datetime(calendar[start : start + len(data)], errors="coerce")
    frame = pd.DataFrame({"date": dates, "open": data.astype(float)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty or frame["date"].isna().any() or frame["date"].duplicated().any():
        raise DataContractError(f"Qlib benchmark feature values are invalid: {symbol}")
    return frame.reset_index(drop=True)


def load_index_benchmarks_from_provider(provider: Path) -> dict[str, Any]:
    raw = Path(provider).expanduser().absolute()
    if raw.is_symlink() or not raw.is_dir():
        raise DataContractError("Qlib benchmark provider is unavailable")
    resolved = raw.resolve(strict=True)
    calendar = _qlib_calendar(resolved)
    available: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in INDEX_NAMES:
        frame = _qlib_open_series(resolved, calendar, symbol)
        if frame is None:
            missing.append(symbol)
        else:
            available[symbol] = frame
    return {"available": available, "missing": missing}


def resolve_index_benchmarks(root: Path) -> dict[str, Any]:
    try:
        resolved = resolve_dataset(
            "qlib_ashare_v2",
            quant_data_root=root,
            require_basis="qfq",
        )
        result = load_index_benchmarks_from_provider(resolved.snapshot_path)
        result["data_identity"] = {
            "dataset_id": resolved.dataset_id,
            "snapshot_id": resolved.manifest["snapshot_id"],
            "content_sha256": resolved.manifest["content_sha256"],
        }
        result["error"] = None
        return result
    except (DataContractError, OSError, ValueError) as exc:
        return {
            "available": {},
            "missing": list(INDEX_NAMES),
            "data_identity": None,
            "error": str(exc),
        }


def _curve_metrics(curve: pd.DataFrame, column: str) -> dict[str, Any]:
    frame = curve[["date", column]].copy().dropna()
    if frame.empty:
        return _return_metrics([])
    returns = frame[column].pct_change().dropna().to_numpy(dtype=float)
    result = _return_metrics(returns)
    result.update(
        {
            "start_date": str(pd.Timestamp(frame.iloc[0]["date"]).date()),
            "end_date": str(pd.Timestamp(frame.iloc[-1]["date"]).date()),
        }
    )
    return result


def _price_comparator_curves(common: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    first_symbol = LOCKED_SYMBOLS[0]
    frame = common[first_symbol][["date", "open"]].rename(
        columns={"open": first_symbol}
    )
    for symbol in LOCKED_SYMBOLS[1:]:
        current = common[symbol][["date", "open"]].rename(columns={"open": symbol})
        frame = frame.merge(current, on="date", how="inner", validate="one_to_one")
    for symbol in LOCKED_SYMBOLS:
        frame[symbol] = frame[symbol] / float(frame[symbol].iloc[0])
    frame["three_etf_equal_weight"] = frame[list(LOCKED_SYMBOLS)].mean(axis=1)
    return frame


def _append_index_curves(
    curve: pd.DataFrame, index_result: Mapping[str, Any]
) -> pd.DataFrame:
    result = curve.copy()
    available = index_result.get("available", {})
    for symbol, frame in available.items():
        scoped = frame[["date", "open"]].copy()
        scoped["date"] = pd.to_datetime(scoped["date"]).dt.normalize()
        scoped = scoped.loc[
            (scoped["date"] >= result["date"].min())
            & (scoped["date"] <= result["date"].max())
        ].sort_values("date", kind="stable")
        if scoped.empty:
            continue
        scoped[symbol] = scoped["open"] / float(scoped.iloc[0]["open"])
        result = result.merge(scoped[["date", symbol]], on="date", how="left")
    return result


def _period_summary(
    periods: pd.DataFrame, start: str, end: str
) -> dict[str, Any]:
    return dict(summarize_periods(periods, start_date=start, end_date=end))


def _yearly_returns(periods_by_name: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    years: set[int] = set()
    for periods in periods_by_name.values():
        years.update(pd.to_datetime(periods["date"]).dt.year.unique().tolist())
    rows: list[dict[str, Any]] = []
    for year in sorted(years):
        row: dict[str, Any] = {"year": int(year)}
        for name, periods in periods_by_name.items():
            summary = _period_summary(
                periods, f"{year}-01-01", f"{year}-12-31"
            )
            row[name] = summary["total_return"] if summary["periods"] else None
        rows.append(row)
    return rows


def _leave_one_year_out(
    strategy: pd.DataFrame, baseline: pd.DataFrame
) -> list[dict[str, Any]]:
    merged = strategy[["date", "net_return"]].rename(
        columns={"net_return": "strategy"}
    ).merge(
        baseline[["date", "net_return"]].rename(
            columns={"net_return": "baseline"}
        ),
        on="date",
        how="inner",
        validate="one_to_one",
    )
    years = pd.to_datetime(merged["date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year in sorted(years.unique()):
        scoped = merged.loc[years != year]
        strategy_return = _annualized_return(scoped["strategy"].to_numpy(dtype=float))
        baseline_return = _annualized_return(scoped["baseline"].to_numpy(dtype=float))
        rows.append(
            {
                "omitted_year": int(year),
                "strategy_annualized_return": strategy_return,
                "baseline_annualized_return": baseline_return,
                "annualized_return_delta": strategy_return - baseline_return,
                "positive_delta": strategy_return > baseline_return,
            }
        )
    return rows


def _start_date_sensitivity(
    strategy: pd.DataFrame,
    baseline: pd.DataFrame,
    start_dates: Sequence[str],
) -> list[dict[str, Any]]:
    left = strategy[["date", "net_return"]].rename(
        columns={"net_return": "strategy"}
    )
    right = baseline[["date", "net_return"]].rename(
        columns={"net_return": "baseline"}
    )
    merged = left.merge(right, on="date", how="inner", validate="one_to_one")
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    for raw_start in start_dates:
        start = pd.Timestamp(raw_start).normalize()
        scoped = merged.loc[merged["date"] >= start]
        if scoped.empty:
            continue
        strategy_metrics = _return_metrics(scoped["strategy"].to_numpy(dtype=float))
        baseline_metrics = _return_metrics(scoped["baseline"].to_numpy(dtype=float))
        rows.append(
            {
                "requested_start": str(start.date()),
                "actual_start": str(scoped.iloc[0]["date"].date()),
                "strategy": strategy_metrics,
                "legacy_two_etf": baseline_metrics,
                "annualized_return_delta": float(strategy_metrics["annualized_return"])
                - float(baseline_metrics["annualized_return"]),
            }
        )
    return rows


def _regime_attribution(
    strategy_periods: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    *,
    trend_days: int,
    momentum_days: int,
) -> dict[str, Any]:
    if benchmark is None:
        return {"status": "unavailable", "reason": "SH000300 benchmark is missing"}
    market = benchmark[["date", "open"]].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    market = market.sort_values("date", kind="stable")
    market["trend"] = market["open"].rolling(trend_days).mean()
    market["momentum"] = market["open"].pct_change(momentum_days)
    market["regime"] = "transition"
    market.loc[
        (market["open"] > market["trend"]) & (market["momentum"] > 0), "regime"
    ] = "risk_on"
    market.loc[
        (market["open"] < market["trend"]) & (market["momentum"] < 0), "regime"
    ] = "risk_off"
    periods = strategy_periods[["date", "net_return"]].copy()
    periods["date"] = pd.to_datetime(periods["date"]).dt.normalize()
    merged = periods.merge(market[["date", "regime"]], on="date", how="left")
    rows: dict[str, Any] = {}
    for regime in ("risk_on", "transition", "risk_off"):
        scoped = merged.loc[merged["regime"] == regime, "net_return"]
        rows[regime] = _return_metrics(scoped.to_numpy(dtype=float))
    return {
        "status": "available",
        "benchmark": "SH000300",
        "trend_days": int(trend_days),
        "momentum_days": int(momentum_days),
        "regimes": rows,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output = frame.copy()
    if "date" in output:
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    output = output.replace([np.inf, -np.inf], np.nan)
    return sanitize_json_value(output.to_dict(orient="records"))  # type: ignore[return-value]


def build_report(
    preview: EtfResearchPreview,
    config: Mapping[str, Any],
    index_result: Mapping[str, Any],
    *,
    generated_at: str,
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    costs = config["costs"]
    common = _common_data(preview.bars)
    variants = evaluate_locked_variants(
        common,
        commission_bps_each_side=float(costs["commission_bps_each_side"]),
        slippage_bps_each_side=tuple(costs["slippage_bps_each_side"]),
    )
    primary_variant = variants["three_etf_defensive"]
    legacy_variant = variants["two_etf_legacy_exact"]
    eligible_two_variant = variants["two_etf_eligible_first"]
    base_key = "0bp_slippage"
    primary = primary_variant["cost_stress"][base_key]
    legacy = legacy_variant["cost_stress"][base_key]
    eligible_two = eligible_two_variant["cost_stress"][base_key]

    price_curves = _price_comparator_curves(common)
    equity = price_curves.merge(
        primary["equity_curve"].rename(columns={"wealth": "strategy"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    equity = equity.merge(
        legacy["equity_curve"].rename(columns={"wealth": "two_etf_legacy_exact"}),
        on="date",
        how="left",
        validate="one_to_one",
    )
    equity = equity.merge(
        eligible_two["equity_curve"].rename(
            columns={"wealth": "two_etf_eligible_first"}
        ),
        on="date",
        how="left",
        validate="one_to_one",
    )
    equity = _append_index_curves(equity, index_result)
    equity = equity.sort_values("date", kind="stable").reset_index(drop=True)

    comparator_metrics: dict[str, Any] = {}
    for column in [
        *LOCKED_SYMBOLS,
        "three_etf_equal_weight",
        "two_etf_legacy_exact",
        "two_etf_eligible_first",
        *[symbol for symbol in INDEX_NAMES if symbol in equity],
    ]:
        comparator_metrics[column] = _curve_metrics(equity, column)

    common_start = str(pd.Timestamp(equity.iloc[0]["date"]).date())
    common_end = str(pd.Timestamp(equity.iloc[-1]["date"]).date())
    split_metrics: dict[str, Any] = {}
    for name, period in config["periods"].items():
        end = common_end if str(period["end"]) == "latest" else str(period["end"])
        split_metrics[name] = {
            "start": str(period["start"]),
            "end": end,
            "strategy": _period_summary(primary["periods"], str(period["start"]), end),
            "legacy_two_etf": _period_summary(
                legacy["periods"], str(period["start"]), end
            ),
        }

    bootstrap_config = config["stability"]["bootstrap"]
    bootstrap = block_bootstrap_incremental_return(
        primary["periods"],
        legacy["periods"],
        block_days=int(bootstrap_config["block_days"]),
        samples=int(bootstrap_config["samples"]),
        seed=int(bootstrap_config["seed"]),
    )
    leave_one_year = _leave_one_year_out(primary["periods"], legacy["periods"])
    positive_ratio = (
        sum(bool(row["positive_delta"]) for row in leave_one_year)
        / len(leave_one_year)
        if leave_one_year
        else 0.0
    )
    validation = split_metrics["validation"]
    observation = split_metrics["observation"]
    evidence = {
        "max_drawdown": float(primary["metrics"]["max_drawdown"]),
        "full_annualized_delta": float(primary["metrics"]["annualized_return"])
        - float(legacy["metrics"]["annualized_return"]),
        "validation_annualized_delta": float(
            validation["strategy"]["annualized_return"]
        )
        - float(validation["legacy_two_etf"]["annualized_return"]),
        "observation_annualized_delta": float(
            observation["strategy"]["annualized_return"]
        )
        - float(observation["legacy_two_etf"]["annualized_return"]),
        "bootstrap_ci95_lower": float(bootstrap["ci95_lower"]),
        "leave_one_year_positive_ratio": positive_ratio,
    }
    acceptance = config["acceptance"]
    classification = classify_incremental_value(
        evidence,
        formally_accepted=preview.formally_accepted,
        max_drawdown_abs_max=float(acceptance["max_drawdown_abs_max"]),
        leave_one_year_positive_ratio_min=float(
            acceptance["leave_one_year_positive_ratio_min"]
        ),
    )

    yearly = _yearly_returns(
        {
            "strategy": primary["periods"],
            "two_etf_legacy_exact": legacy["periods"],
            "two_etf_eligible_first": eligible_two["periods"],
        }
    )
    start_sensitivity = _start_date_sensitivity(
        primary["periods"],
        legacy["periods"],
        config["stability"]["start_dates"],
    )
    regime_config = config["stability"]["regime_attribution"]
    available_indexes = index_result.get("available", {})
    regime = _regime_attribution(
        primary["periods"],
        available_indexes.get(str(regime_config["benchmark"])),
        trend_days=int(regime_config["trend_days"]),
        momentum_days=int(regime_config["momentum_days"]),
    )

    drawdown = equity[["date", "strategy"]].copy()
    drawdown["strategy"] = (
        drawdown["strategy"] / drawdown["strategy"].cummax() - 1.0
    )
    latest = primary_variant["signal_table"].iloc[-1]
    position_fraction = (
        primary["periods"]["position"].value_counts(normalize=True).sort_index()
    )
    cost_stress = {
        key: value["metrics"]
        for key, value in primary_variant["cost_stress"].items()
    }

    return {
        "schema_version": 1,
        "status": "research_preview",
        "generated_at": generated_at,
        "research_id": config["research_id"],
        "summary": {
            "status": classification["status"],
            "formal_candidate_status": classification["formal_candidate_status"],
            "classification_enabled": classification["classification_enabled"],
            "classification_reason": classification["reason"],
            "usable_for_trading": False,
            "data_start": common_start,
            "data_end": common_end,
            "primary_metrics": primary["metrics"],
            "max_drawdown_limit": acceptance["max_drawdown_abs_max"],
            "history_warning": config["report"]["history_warning"],
        },
        "strategy": {
            **dict(config["strategy"]),
            "commission_bps_each_side": costs["commission_bps_each_side"],
            "base_slippage_bps_each_side": 0.0,
            "stamp_duty_applied": False,
            "transfer_fee_applied": False,
            "warmup_position": CASH,
        },
        "data_quality": {
            "runtime_status": preview.status,
            "formally_accepted": preview.formally_accepted,
            "usable_for_trading": preview.usable_for_trading,
            "dataset_id": preview.dataset_id,
            "snapshot_id": preview.snapshot_id,
            "manifest_sha256": preview.manifest_sha256,
            "content_sha256": preview.content_sha256,
            "required_symbols": list(LOCKED_SYMBOLS),
            "common_start": common_start,
            "common_end": common_end,
        },
        "performance": {
            "primary_metrics": primary["metrics"],
            "comparators": comparator_metrics,
            "periods": split_metrics,
            "incremental_evidence_vs_legacy_two_etf": evidence,
        },
        "stability": {
            "cost_stress": cost_stress,
            "block_bootstrap": bootstrap,
            "leave_one_year_out": leave_one_year,
            "leave_one_year_positive_ratio": positive_ratio,
            "start_date_sensitivity": start_sensitivity,
            "market_regime_attribution": regime,
        },
        "holding_behavior": {
            "position_fraction": {
                str(key): float(value) for key, value in position_fraction.items()
            },
            "switch_count": primary["metrics"]["switch_count"],
            "invested_fraction": primary["metrics"]["invested_fraction"],
        },
        "latest_signal": {
            "as_of_date": str(pd.Timestamp(latest["date"]).date()),
            "signal_for_next_common_open": str(latest["signal"]),
            "position_at_latest_open": str(latest["target_position"]),
            "usable_for_trading": False,
        },
        "benchmarks": {
            "available": [
                {"symbol": symbol, "name": INDEX_NAMES[symbol]}
                for symbol in index_result.get("available", {})
            ],
            "missing": list(index_result.get("missing", [])),
            "substitution_used": False,
            "data_identity": index_result.get("data_identity"),
            "load_error": index_result.get("error"),
        },
        "chart_data": {
            "equity": _records(equity),
            "drawdown": _records(drawdown),
            "yearly_returns": yearly,
        },
        "reproducibility": dict(reproducibility),
        "boundaries": {
            "parameter_search_performed": False,
            "data_download_performed_by_dsa": False,
            "data_cleaning_performed_by_dsa": False,
            "broker_connected": False,
            "automatic_orders": False,
            "formal_candidate_labels_suppressed": True,
        },
    }


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(number) else f"{number * 100:.2f}%"


def _line_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    series: Sequence[tuple[str, str, str]],
) -> str:
    width, height = 1000, 300
    left, right, top, bottom = 58, 18, 34, 34
    points_by_series: dict[str, list[tuple[int, float]]] = {}
    values: list[float] = []
    for key, _, _ in series:
        points: list[tuple[int, float]] = []
        for index, row in enumerate(rows):
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                points.append((index, value))
                values.append(value)
        points_by_series[key] = points
    if not rows or not values:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><text x="24" y="36">{html.escape(title)}：无数据</text></svg>'
    low, high = min(values), max(values)
    if math.isclose(low, high):
        low -= 0.05
        high += 0.05
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coord(index: int) -> float:
        return left + plot_width * index / max(len(rows) - 1, 1)

    def y_coord(value: float) -> float:
        return top + plot_height * (high - value) / (high - low)

    lines = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="grid"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="grid"/>',
        f'<text x="4" y="{top+4}" class="axis">{high:.2f}</text>',
        f'<text x="4" y="{height-bottom}" class="axis">{low:.2f}</text>',
    ]
    for key, label, color in series:
        points = points_by_series[key]
        if not points:
            continue
        polyline = " ".join(
            f"{x_coord(index):.1f},{y_coord(value):.1f}" for index, value in points
        )
        lines.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/>'
        )
    legend_x = left
    for _, label, color in series:
        lines.append(
            f'<line x1="{legend_x}" y1="{height-10}" x2="{legend_x+16}" y2="{height-10}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{legend_x+21}" y="{height-6}" class="axis">{html.escape(label)}</text>'
        )
        legend_x += max(115, len(label) * 14)
    lines.append("</svg>")
    return "".join(lines)


def _bar_svg(
    rows: Sequence[Mapping[str, Any]], *, title: str, key: str = "strategy"
) -> str:
    width, height = 1000, 300
    left, right, top, bottom = 58, 18, 34, 38
    numeric: list[tuple[str, float]] = []
    for row in rows:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            numeric.append((str(row.get("year", "")), value))
    if not numeric:
        return f'<svg viewBox="0 0 {width} {height}" role="img"><text x="24" y="36">{html.escape(title)}：无数据</text></svg>'
    bound = max(abs(value) for _, value in numeric) or 0.01
    plot_width = width - left - right
    plot_height = height - top - bottom
    zero = top + plot_height / 2
    slot = plot_width / len(numeric)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{zero}" x2="{width-right}" y2="{zero}" class="grid"/>',
    ]
    for index, (label, value) in enumerate(numeric):
        bar_height = abs(value) / bound * (plot_height / 2)
        x = left + index * slot + slot * 0.18
        y = zero - bar_height if value >= 0 else zero
        color = "#c62828" if value >= 0 else "#16865b"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*0.64:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x+slot*0.32:.1f}" y="{height-16}" text-anchor="middle" class="axis">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {})
    performance = report.get("performance", {})
    metrics = performance.get("primary_metrics") or summary.get("primary_metrics", {})
    chart_data = report.get("chart_data", {})
    missing = report.get("benchmarks", {}).get("missing", [])
    equity_chart = _line_svg(
        chart_data.get("equity", []),
        title="净值对比",
        series=(
            ("strategy", "三ETF策略", "#c62828"),
            ("SZ159915", "159915", "#2563eb"),
            ("SH513100", "513100", "#d97706"),
            ("SH512890", "512890", "#0f766e"),
            ("two_etf_legacy_exact", "原两ETF", "#7c3aed"),
        ),
    )
    drawdown_chart = _line_svg(
        chart_data.get("drawdown", []),
        title="策略回撤",
        series=(("strategy", "回撤", "#16865b"),),
    )
    yearly_chart = _bar_svg(
        chart_data.get("yearly_returns", []), title="逐年收益"
    )
    strategy = report.get("strategy", {})
    symbols = "、".join(strategy.get("symbols", []))
    missing_text = "、".join(missing) if missing else "无"
    commission_bps = float(strategy.get("commission_bps_each_side", 0.0))
    commission_pct = commission_bps * 0.01
    comparator_names = {
        "SZ159915": "159915 买入持有",
        "SH513100": "513100 买入持有",
        "SH512890": "512890 买入持有",
        "three_etf_equal_weight": "三只ETF等权买入持有",
        "two_etf_legacy_exact": "原两ETF精确旧规则",
        "two_etf_eligible_first": "两ETF先过滤再排名",
        "SH000300": "沪深300",
        "SH000905": "中证500",
        "SZ399006": "创业板指",
    }
    comparator_rows = "".join(
        "<tr>"
        f"<td>{html.escape(comparator_names.get(str(key), str(key)))}</td>"
        f"<td>{_fmt_pct(value.get('annualized_return'))}</td>"
        f"<td>{_fmt_pct(value.get('total_return'))}</td>"
        f"<td>{_fmt_pct(value.get('max_drawdown'))}</td>"
        "</tr>"
        for key, value in performance.get("comparators", {}).items()
        if isinstance(value, Mapping)
    )
    period_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(name))}</td>"
        f"<td>{_fmt_pct(value.get('strategy', {}).get('annualized_return'))}</td>"
        f"<td>{_fmt_pct(value.get('legacy_two_etf', {}).get('annualized_return'))}</td>"
        f"<td>{_fmt_pct(value.get('strategy', {}).get('max_drawdown'))}</td>"
        "</tr>"
        for name, value in performance.get("periods", {}).items()
        if isinstance(value, Mapping)
    )
    stability = report.get("stability", {})
    cost_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(name).replace('_slippage', '滑点'))}</td>"
        f"<td>{float(value.get('side_cost_bps', 0)):.2f}bp</td>"
        f"<td>{_fmt_pct(value.get('annualized_return'))}</td>"
        f"<td>{_fmt_pct(value.get('max_drawdown'))}</td>"
        "</tr>"
        for name, value in stability.get("cost_stress", {}).items()
        if isinstance(value, Mapping)
    )
    holding = report.get("holding_behavior", {})
    holding_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(position))}</td><td>{_fmt_pct(fraction)}</td>"
        "</tr>"
        for position, fraction in holding.get("position_fraction", {}).items()
    )
    bootstrap = stability.get("block_bootstrap", {})
    bootstrap_text = (
        "区块Bootstrap的年化增量95%区间："
        f"{_fmt_pct(bootstrap.get('ci95_lower'))} 至 "
        f"{_fmt_pct(bootstrap.get('ci95_upper'))}；"
        f"增量为正的模拟占比 {_fmt_pct(bootstrap.get('positive_probability'))}。"
        if bootstrap
        else "区块Bootstrap：暂无数据。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三ETF防守轮动研究预览</title>
<style>
:root{{--red:#c62828;--green:#16865b;--ink:#17212b;--muted:#5d6975;--line:#d8dee5;--paper:#fff;--wash:#f4f6f8}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;line-height:1.55}}
main{{max-width:1180px;margin:0 auto;background:var(--paper);min-height:100vh;padding:34px 40px 60px}}
h1{{font-size:28px;margin:0 0 8px}} h2{{font-size:19px;margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.warning{{border-left:4px solid #d97706;background:#fff8e8;padding:12px 14px;margin:18px 0;font-weight:600}}
.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} .metric{{border:1px solid var(--line);padding:14px;border-radius:6px}}
.metric b{{display:block;font-size:22px;margin-top:4px}} .muted{{color:var(--muted)}}
.chart{{border:1px solid var(--line);border-radius:6px;padding:8px;margin:12px 0;overflow-x:auto}} svg{{display:block;width:100%;min-width:720px;height:auto}}
.grid{{stroke:#cbd3dc;stroke-width:1}} .axis{{font-size:12px;fill:#65717d}} .chart-title{{font-size:15px;font-weight:600;fill:#17212b}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#f7f8fa}}
code{{background:#eef1f4;padding:2px 5px;border-radius:3px}} .status{{color:#9a5b00;font-weight:700}}
@media(max-width:720px){{main{{padding:22px 16px}}.metrics{{grid-template-columns:1fr}}h1{{font-size:23px}}}}
@media print{{body{{background:#fff}}main{{max-width:none;padding:12mm}}.chart{{break-inside:avoid}}}}
</style>
</head>
<body><main>
<h1>三ETF防守轮动研究预览</h1>
<p class="muted">状态：<span class="status">{html.escape(str(report.get('status', 'research_preview')))}</span> · 数据截止 {html.escape(str(summary.get('data_end', '--')))}</p>
<div class="warning">当前 ETF 快照尚未完成 quant_data 正式验收。本报告仅用于研究预览，不可用于交易，也不会生成正式候选评级。</div>
<h2>固定规则</h2>
<p>标的：{html.escape(symbols)}。只使用20日涨幅与20日均线；先排除未站上自身20日均线的ETF，再选择20日涨幅最高者；全部不合格时持有现金。T日收盘产生信号，T+1共同开盘执行，不搜索参数。</p>
<p><strong>费用定义：</strong>单边佣金 {commission_bps:g}bp，即每次买入或卖出收取成交金额的{commission_pct:g}%；基础回测不另加滑点。1bp = 0.01%，ETF不收印花税和过户费。ETF之间切换会发生一次卖出和一次买入，因此支付两边费用。</p>
<div class="metrics">
  <div class="metric"><span>年化收益</span><b>{_fmt_pct(metrics.get('annualized_return'))}</b></div>
  <div class="metric"><span>累计收益</span><b>{_fmt_pct(metrics.get('total_return'))}</b></div>
  <div class="metric"><span>最大回撤</span><b style="color:var(--green)">{_fmt_pct(metrics.get('max_drawdown'))}</b></div>
</div>
<h2>收益与风险</h2><div class="chart">{equity_chart}</div><div class="chart">{drawdown_chart}</div><div class="chart">{yearly_chart}</div>
<h2>策略与基准对比</h2>
<table><thead><tr><th>对象</th><th>年化收益</th><th>累计收益</th><th>最大回撤</th></tr></thead><tbody>{comparator_rows or '<tr><td colspan="4">暂无数据</td></tr>'}</tbody></table>
<h2>分阶段表现</h2>
<table><thead><tr><th>阶段</th><th>三ETF年化</th><th>原两ETF年化</th><th>三ETF最大回撤</th></tr></thead><tbody>{period_rows or '<tr><td colspan="4">暂无数据</td></tr>'}</tbody></table>
<h2>成本压力与稳定性</h2>
<table><thead><tr><th>滑点假设</th><th>单边总成本</th><th>年化收益</th><th>最大回撤</th></tr></thead><tbody>{cost_rows or '<tr><td colspan="4">暂无数据</td></tr>'}</tbody></table>
<p>{html.escape(bootstrap_text)}</p><p>逐年留一后，三ETF年化收益高于原两ETF的比例：{_fmt_pct(stability.get('leave_one_year_positive_ratio'))}。</p>
<h2>持仓行为</h2>
<p>换仓次数：{html.escape(str(holding.get('switch_count', '--')))}；持有ETF的交易日占比：{_fmt_pct(holding.get('invested_fraction'))}。</p>
<table><thead><tr><th>持仓</th><th>时间占比</th></tr></thead><tbody>{holding_rows or '<tr><td colspan="2">暂无数据</td></tr>'}</tbody></table>
<h2>指数证据</h2><p>缺失指数：<code>{html.escape(missing_text)}</code>。未使用ETF或其他数据替代缺失指数。</p>
<h2>结果边界</h2>
<table><tbody>
<tr><th>数据状态</th><td>{html.escape(str(report.get('status', 'research_preview')))}</td></tr>
<tr><th>正式候选评级</th><td>{html.escape(str(summary.get('formal_candidate_status') or '未启用'))}</td></tr>
<tr><th>可用于交易</th><td>否</td></tr>
<tr><th>参数搜索</th><td>未执行</td></tr>
</tbody></table>
</main></body></html>"""


def run(
    *,
    config_path: Path = DEFAULT_CONFIG,
    report_dir: Path = DEFAULT_REPORT_DIR,
    quant_data_dir: Path | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    preview = resolve_etf_research_preview(root)
    if preview.status != config["data"]["required_runtime_status"]:
        raise DataContractError("ETF runtime status does not match preregistration")
    indexes = resolve_index_benchmarks(root)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    project_root = Path(__file__).resolve().parents[2]
    reproducibility = {
        "config_file": config_path.name,
        "config_sha256": _sha256(config_path),
        **_git_metadata(project_root),
    }
    report = build_report(
        preview,
        config,
        indexes,
        generated_at=generated_at,
        reproducibility=reproducibility,
    )
    run_id = (
        datetime.now().strftime("%Y%m%dT%H%M%S")
        + "-"
        + preview.snapshot_id[-8:]
        + "-"
        + reproducibility["config_sha256"][:8]
    )
    report["run_id"] = run_id
    destination = Path(report_dir).expanduser().resolve()
    run_dir = destination / "runs" / run_id
    _atomic_json(run_dir / "strategy_report.json", report)
    _atomic_text(run_dir / "strategy_report.html", render_html(report))
    _atomic_json(destination / "strategy_report_latest.json", report)
    _atomic_text(destination / "strategy_report_latest.html", render_html(report))
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--quant-data-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_dir = run(
            config_path=args.config,
            report_dir=args.report_dir,
            quant_data_dir=args.quant_data_root,
        )
    except (DataContractError, OSError, ValueError, KeyError) as exc:
        print(f"ETF defensive rotation research failed: {exc}", file=sys.stderr)
        return 2
    print(f"ETF defensive rotation research preview: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
