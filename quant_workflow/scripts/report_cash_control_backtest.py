#!/usr/bin/env python3
"""Compare a Qlib portfolio backtest with a read-only cash-control overlay.

The overlay scales the existing Qlib daily strategy return by a market-regime
position ratio.  It is an advisory research report only: it does not connect to
broker software, does not submit orders, and does not mutate Qlib data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, projects_root, quant_data_root, reports_root, workflow_root  # noqa: E402
from report_market_regime import classify_regime  # noqa: E402


WORKFLOW = workflow_root()
ROOT = projects_root()
QUANT_DATA = quant_data_root()
REPORT_DIR = reports_root()
MLRUNS_DIR = mlruns_root()
LATEST_TRAINING_RESULT = REPORT_DIR / "latest_training_result.json"
MERGED_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
DEFAULT_TRADING_DAYS_PER_YEAR = 244


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_pct(value: float | int | None) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_run_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw and not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError("run_id may only contain letters, numbers, underscores, or hyphens.")
    return raw


def find_portfolio_report_path(
    latest_training_path: Path = LATEST_TRAINING_RESULT,
    mlruns_dir: Path = MLRUNS_DIR,
    run_id: str | None = None,
) -> Path:
    """Locate the Qlib daily portfolio report for the latest or requested run."""
    requested_run_id = _safe_run_id(run_id)
    if not requested_run_id:
        latest = _read_json_object(latest_training_path) or {}
        requested_run_id = _safe_run_id(latest.get("run_id"))

    if requested_run_id:
        matches = sorted(mlruns_dir.glob(f"*/{requested_run_id}/artifacts/portfolio_analysis/report_normal_1day.pkl"))
        if matches:
            return matches[-1]

    fallback = sorted(
        mlruns_dir.glob("*/*/artifacts/portfolio_analysis/report_normal_1day.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        return fallback[0]
    detail = f" for run {requested_run_id}" if requested_run_id else ""
    raise FileNotFoundError(f"Could not find Qlib report_normal_1day.pkl{detail} under {mlruns_dir}.")


def load_portfolio_returns(path: Path, return_column: str = "return") -> pd.Series:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Portfolio report must be a pandas DataFrame: {path}")
    if return_column not in frame.columns:
        raise ValueError(f"Portfolio report is missing return column {return_column!r}: {path}")

    data = frame.copy()
    if isinstance(data.index, pd.MultiIndex):
        raise ValueError("Portfolio report with MultiIndex is not supported yet.")
    if not isinstance(data.index, pd.DatetimeIndex):
        date_column = next((name for name in ("datetime", "date") if name in data.columns), None)
        if date_column is None:
            raise ValueError("Portfolio report must have a DatetimeIndex or a datetime/date column.")
        data.index = pd.to_datetime(data[date_column], errors="coerce")
    else:
        data.index = pd.to_datetime(data.index, errors="coerce")

    returns = pd.to_numeric(data[return_column], errors="coerce")
    returns = returns[returns.index.notna()].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    returns = returns[~returns.index.duplicated(keep="last")]
    returns.name = return_column
    if returns.empty:
        raise ValueError(f"Portfolio report has no usable returns: {path}")
    return returns.astype("float64")


def _read_close_series(path: Path) -> pd.Series | None:
    try:
        frame = pd.read_parquet(path, columns=["date", "close"])
    except (OSError, KeyError, ValueError):
        return None
    if frame.empty:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame = frame.sort_values("date").drop_duplicates(subset=["date"])
    return pd.Series(frame["close"].to_numpy(dtype="float64"), index=frame["date"], name=path.stem)


def load_close_matrix(data_dir: Path, limit: int = 0) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    series = []
    for path in files:
        item = _read_close_series(path)
        if item is not None:
            series.append(item)
    if not series:
        return pd.DataFrame()
    matrix = pd.concat(series, axis=1).sort_index()
    matrix = matrix.loc[~matrix.index.duplicated(keep="last")]
    return matrix


def _window_return(matrix: pd.DataFrame, days: int) -> pd.Series:
    if len(matrix) < days + 1:
        return pd.Series(dtype="float64")
    recent = matrix.iloc[-1]
    base = matrix.iloc[-days - 1]
    return (recent / base - 1).replace([np.inf, -np.inf], np.nan).dropna()


def _market_metrics_for_window(window: pd.DataFrame) -> dict[str, float | int | str | None]:
    market_index = window.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0).add(1).cumprod()
    daily_returns = market_index.pct_change(fill_method=None).dropna()
    ret20_days = min(20, len(market_index) - 1)
    ret60_days = min(60, len(market_index) - 1)
    ret20_by_symbol = _window_return(window, ret20_days) if ret20_days > 0 else pd.Series(dtype="float64")
    ret60_by_symbol = _window_return(window, ret60_days) if ret60_days > 0 else pd.Series(dtype="float64")
    rolling_peak = market_index.cummax()

    return {
        "symbols": int(window.shape[1]),
        "calendar_days": int(len(window)),
        "start_date": str(window.index.min().date()),
        "end_date": str(window.index.max().date()),
        "return_20d": _safe_float(market_index.iloc[-1] / market_index.iloc[-ret20_days - 1] - 1) if ret20_days > 0 else 0.0,
        "return_60d": _safe_float(market_index.iloc[-1] / market_index.iloc[-ret60_days - 1] - 1) if ret60_days > 0 else 0.0,
        "breadth_20d": _safe_float((ret20_by_symbol > 0).mean()) if not ret20_by_symbol.empty else 0.0,
        "breadth_60d": _safe_float((ret60_by_symbol > 0).mean()) if not ret60_by_symbol.empty else 0.0,
        "drawdown": _safe_float(market_index.iloc[-1] / rolling_peak.iloc[-1] - 1),
        "volatility_20d": _safe_float(daily_returns.tail(20).std()) if not daily_returns.empty else 0.0,
    }


def _neutral_signal(date: pd.Timestamp, reason: str) -> dict[str, Any]:
    return {
        "date": str(date.date()),
        "regime": "neutral",
        "cash_ratio": 0.30,
        "position_ratio": 0.70,
        "interpretation": reason,
    }


def build_rolling_market_signals(
    close_matrix: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    lookback_days: int = 60,
    min_symbols: int = 100,
) -> pd.DataFrame:
    signal_dates = pd.DatetimeIndex(signal_dates).sort_values()
    if close_matrix.empty:
        signals = [
            _neutral_signal(pd.Timestamp(raw_date), "Insufficient prior market data; default to neutral cash control.")
            for raw_date in signal_dates
        ]
        signal_frame = pd.DataFrame(signals)
        if signal_frame.empty:
            empty = pd.DataFrame(columns=["regime", "cash_ratio", "position_ratio"])
            empty.index = pd.DatetimeIndex([], name="datetime")
            return empty
        signal_frame.index = pd.to_datetime(signal_frame.pop("date"))
        signal_frame.index.name = "datetime"
        return signal_frame

    signals: list[dict[str, Any]] = []
    lookback_rows = max(lookback_days + 1, 2)
    min_history_rows = max(2, min(20, lookback_days // 2))
    for raw_date in signal_dates:
        date = pd.Timestamp(raw_date)
        history = close_matrix.loc[close_matrix.index < date].tail(lookback_rows)
        history = history.dropna(axis=1, thresh=min_history_rows)
        if len(history) < 2 or history.shape[1] < min_symbols:
            signals.append(_neutral_signal(date, "Insufficient prior market data; default to neutral cash control."))
            continue

        metrics = _market_metrics_for_window(history)
        regime, cash_ratio, interpretation = classify_regime(metrics)
        signals.append(
            {
                "date": str(date.date()),
                "regime": regime,
                "cash_ratio": cash_ratio,
                "position_ratio": round(1 - cash_ratio, 4),
                "interpretation": interpretation,
                "return_20d": metrics.get("return_20d"),
                "breadth_20d": metrics.get("breadth_20d"),
                "drawdown": metrics.get("drawdown"),
                "volatility_20d": metrics.get("volatility_20d"),
            }
        )

    signal_frame = pd.DataFrame(signals)
    signal_frame.index = pd.to_datetime(signal_frame.pop("date"))
    signal_frame.index.name = "datetime"
    return signal_frame


def apply_cash_control_overlay(returns: pd.Series, signals: pd.DataFrame) -> pd.Series:
    aligned = signals.reindex(returns.index)
    position_ratio = pd.to_numeric(aligned["position_ratio"], errors="coerce").fillna(0.70).clip(lower=0, upper=1)
    overlay = returns.astype("float64") * position_ratio.astype("float64")
    overlay.name = "cash_control_return"
    return overlay


def calculate_return_metrics(
    returns: pd.Series,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if clean.empty:
        return {
            "days": 0,
            "cumulative_return": None,
            "annualized_return": None,
            "mean_daily_return": None,
            "daily_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "win_rate": None,
            "best_day": None,
            "worst_day": None,
        }

    curve = (1 + clean).cumprod()
    cumulative_return = float(curve.iloc[-1] - 1)
    peak = curve.cummax()
    drawdown = curve / peak - 1
    mean = float(clean.mean())
    volatility = float(clean.std()) if len(clean) > 1 else 0.0
    sharpe_ratio = mean / volatility * math.sqrt(trading_days_per_year) if volatility > 0 else None
    annualized_return = None
    if cumulative_return > -1:
        annualized_return = float((1 + cumulative_return) ** (trading_days_per_year / len(clean)) - 1)

    return {
        "days": int(len(clean)),
        "cumulative_return": _safe_float(cumulative_return),
        "annualized_return": _safe_float(annualized_return),
        "mean_daily_return": _safe_float(mean),
        "daily_volatility": _safe_float(volatility),
        "sharpe_ratio": _safe_float(sharpe_ratio),
        "max_drawdown": _safe_float(float(drawdown.min())),
        "win_rate": _safe_float(float((clean > 0).mean())),
        "best_day": _safe_float(float(clean.max())),
        "worst_day": _safe_float(float(clean.min())),
    }


def _metric_delta(overlay: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = overlay.get(key)
    right = baseline.get(key)
    if left is None or right is None:
        return None
    return _safe_float(float(left) - float(right))


def compare_returns(
    baseline_returns: pd.Series,
    overlay_returns: pd.Series,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> dict[str, Any]:
    baseline = calculate_return_metrics(baseline_returns, trading_days_per_year=trading_days_per_year)
    cash_control = calculate_return_metrics(overlay_returns, trading_days_per_year=trading_days_per_year)
    return {
        "baseline": baseline,
        "cash_control": cash_control,
        "delta": {
            "cumulative_return": _metric_delta(cash_control, baseline, "cumulative_return"),
            "annualized_return": _metric_delta(cash_control, baseline, "annualized_return"),
            "sharpe_ratio": _metric_delta(cash_control, baseline, "sharpe_ratio"),
            "max_drawdown": _metric_delta(cash_control, baseline, "max_drawdown"),
        },
    }


def summarize_signals(signals: pd.DataFrame) -> dict[str, Any]:
    regime_counts = signals["regime"].value_counts().sort_index().to_dict() if "regime" in signals else {}
    cash_ratio = pd.to_numeric(signals["cash_ratio"], errors="coerce") if "cash_ratio" in signals else pd.Series(dtype="float64")
    position_ratio = pd.to_numeric(signals["position_ratio"], errors="coerce") if "position_ratio" in signals else pd.Series(dtype="float64")
    return {
        "days": int(len(signals)),
        "regime_counts": {str(key): int(value) for key, value in regime_counts.items()},
        "average_cash_ratio": _safe_float(cash_ratio.mean()),
        "average_position_ratio": _safe_float(position_ratio.mean()),
        "latest_regime": str(signals["regime"].iloc[-1]) if len(signals) and "regime" in signals else None,
        "latest_cash_ratio": _safe_float(cash_ratio.iloc[-1]) if len(cash_ratio) else None,
        "latest_position_ratio": _safe_float(position_ratio.iloc[-1]) if len(position_ratio) else None,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    baseline = payload.get("comparison", {}).get("baseline", {})
    cash_control = payload.get("comparison", {}).get("cash_control", {})
    delta = payload.get("comparison", {}).get("delta", {})
    signal_summary = payload.get("signal_summary", {})
    return f"""# Cash-Control Backtest

Generated: {payload.get("generated_at")}

## Scope

This is a read-only research report. It scales the existing Qlib portfolio daily
return by market-regime position ratios. It does not connect to broker software,
不连接券商、不读取密码、不自动下单，也不改写 Qlib provider。

## Inputs

- Portfolio report: `{payload.get("portfolio_report_path")}`
- Market data: `{payload.get("market_data_dir")}`
- Signal lookback days: `{payload.get("lookback_days")}`
- Annualization trading days: `{payload.get("trading_days_per_year")}`
- Signal lag: previous market close only, no same-day look-ahead

## Result

| Metric | Baseline | Cash-control | Delta |
| --- | ---: | ---: | ---: |
| Cumulative return | {_fmt_pct(baseline.get("cumulative_return"))} | {_fmt_pct(cash_control.get("cumulative_return"))} | {_fmt_pct(delta.get("cumulative_return"))} |
| Annualized return | {_fmt_pct(baseline.get("annualized_return"))} | {_fmt_pct(cash_control.get("annualized_return"))} | {_fmt_pct(delta.get("annualized_return"))} |
| Max drawdown | {_fmt_pct(baseline.get("max_drawdown"))} | {_fmt_pct(cash_control.get("max_drawdown"))} | {_fmt_pct(delta.get("max_drawdown"))} |
| Sharpe ratio | {baseline.get("sharpe_ratio") if baseline.get("sharpe_ratio") is not None else "n/a"} | {cash_control.get("sharpe_ratio") if cash_control.get("sharpe_ratio") is not None else "n/a"} | {delta.get("sharpe_ratio") if delta.get("sharpe_ratio") is not None else "n/a"} |

## Signal Summary

- Days: {signal_summary.get("days")}
- Regime counts: `{signal_summary.get("regime_counts")}`
- Average cash ratio: {_fmt_pct(signal_summary.get("average_cash_ratio"))}
- Latest regime: `{signal_summary.get("latest_regime")}`
- Latest cash ratio: {_fmt_pct(signal_summary.get("latest_cash_ratio"))}

## Interpretation

- If max drawdown improves but return drops sharply, the overlay is acting as a defensive risk budget, not an alpha source.
- If both drawdown and Sharpe ratio improve, the market state signal may be worth wiring into future Qlib strategy experiments.
- This report ignores extra transaction costs from changing gross exposure; treat it as a first-pass risk-control diagnostic.
"""


def generate_report(
    portfolio_report_path: Path,
    market_data_dir: Path,
    out_dir: Path,
    lookback_days: int = 60,
    min_symbols: int = 100,
    limit: int = 0,
    return_column: str = "return",
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    preflight_unavailable_reason: str = "",
) -> dict[str, Any]:
    unavailable_reason = preflight_unavailable_reason
    try:
        returns = load_portfolio_returns(portfolio_report_path, return_column=return_column)
    except (FileNotFoundError, ValueError) as exc:
        unavailable_reason = unavailable_reason or str(exc)
        returns = pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="datetime"), name=return_column)
        signals = pd.DataFrame(columns=["regime", "cash_ratio", "position_ratio"])
        signals.index = pd.DatetimeIndex([], name="datetime")
    else:
        close_matrix = load_close_matrix(market_data_dir, limit=limit)
        signals = build_rolling_market_signals(
            close_matrix,
            signal_dates=pd.DatetimeIndex(returns.index),
            lookback_days=lookback_days,
            min_symbols=min_symbols,
        )
    overlay = apply_cash_control_overlay(returns, signals)
    comparison = compare_returns(returns, overlay, trading_days_per_year=trading_days_per_year)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"cash_control_backtest_{ts}.json"
    md_path = out_dir / f"cash_control_backtest_{ts}.md"
    latest_json = out_dir / "cash_control_backtest_latest.json"
    latest_md = out_dir / "cash_control_backtest_latest.md"

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "read_only_cash_control_overlay",
        "portfolio_report_path": str(portfolio_report_path),
        "market_data_dir": str(market_data_dir),
        "lookback_days": int(lookback_days),
        "min_symbols": int(min_symbols),
        "limit": int(limit),
        "return_column": return_column,
        "trading_days_per_year": int(trading_days_per_year),
        "comparison": comparison,
        "signal_summary": summarize_signals(signals),
        "unavailable_reason": unavailable_reason,
        "sample_signals": signals.tail(10).reset_index().assign(datetime=lambda frame: frame["datetime"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        },
    }
    markdown = build_markdown(payload)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only cash-control overlay report for Qlib backtests.")
    parser.add_argument("--portfolio-report", type=Path, default=None, help="Path to Qlib report_normal_1day.pkl. Defaults to latest MLflow run.")
    parser.add_argument("--run-id", default="", help="Optional MLflow run id used to locate report_normal_1day.pkl.")
    parser.add_argument("--market-data-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--min-symbols", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Optional symbol limit for smoke runs; 0 means all symbols.")
    parser.add_argument("--return-column", default="return")
    parser.add_argument("--trading-days-per-year", type=int, default=DEFAULT_TRADING_DAYS_PER_YEAR)
    args = parser.parse_args()

    preflight_unavailable_reason = ""
    try:
        portfolio_report = args.portfolio_report or find_portfolio_report_path(run_id=args.run_id)
    except FileNotFoundError as exc:
        preflight_unavailable_reason = str(exc)
        portfolio_report = args.out_dir / "missing_report_normal_1day.pkl"
    payload = generate_report(
        portfolio_report_path=portfolio_report,
        market_data_dir=args.market_data_dir,
        out_dir=args.out_dir,
        lookback_days=args.lookback_days,
        min_symbols=args.min_symbols,
        limit=args.limit,
        return_column=args.return_column,
        trading_days_per_year=args.trading_days_per_year,
        preflight_unavailable_reason=preflight_unavailable_reason,
    )
    baseline = payload["comparison"]["baseline"]
    cash_control = payload["comparison"]["cash_control"]
    print(f"Baseline cumulative return: {_fmt_pct(baseline.get('cumulative_return'))}")
    print(f"Cash-control cumulative return: {_fmt_pct(cash_control.get('cumulative_return'))}")
    print(f"Latest markdown: {payload['artifacts']['latest_markdown']}")
    print(f"Latest JSON: {payload['artifacts']['latest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
