# -*- coding: utf-8 -*-
"""Rerun an allowlisted MLflow prediction with a selected portfolio strategy."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from quant_workflow.scripts._optimized_strategy_core import OptimizedStrategy
from src.services.quant_workflow_paths import (
    mlruns_root as resolve_mlruns_root,
    reports_root as resolve_reports_root,
    workflow_root as resolve_workflow_root,
)


RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXPERIMENT_ID_RE = re.compile(r"^[0-9]+$")
MAX_REPORT_BYTES = 10 * 1024 * 1024
MAX_PICKLE_BYTES = 512 * 1024 * 1024
ALLOWED_STRATEGY_KWARGS = {
    "n_drop",
    "rebalance_freq",
    "turnover_cap",
    "trend_filter",
    "trend_ma",
    "trend_reduce_pct",
    "vol_sizing",
    "vol_reduce_threshold",
    "max_drawdown_stop",
    "max_dd_cooldown_days",
    "max_single_weight",
    "max_sector_weight",
}

MarketLoader = Callable[[Path, str, str], pd.DataFrame]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required training report is missing: {path.name}")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError(f"Training report is too large: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training report must contain an object: {path.name}")
    return payload


def _single_column(frame: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    result = frame.to_frame(name) if isinstance(frame, pd.Series) else frame.copy()
    if result.shape[1] != 1:
        raise ValueError(f"Saved {name} artifact must contain exactly one column")
    result.columns = [name]
    return result


def _slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        raise ValueError("Saved prediction artifacts must use a datetime/instrument MultiIndex")
    dates = pd.to_datetime(frame.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("Saved prediction artifacts contain invalid dates")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return frame.loc[(dates >= start_ts) & (dates <= end_ts)]


def _default_market_loader(provider_uri: Path, start: str, end: str) -> pd.DataFrame:
    if not provider_uri.is_dir():
        raise FileNotFoundError("The Qlib data provider used by this training run is unavailable")

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region="cn")
    buffered_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).date().isoformat()
    raw = D.features(["SH000300"], ["$close"], start_time=start, end_time=buffered_end)
    if raw.empty:
        raise ValueError("SH000300 benchmark data is unavailable for the selected period")
    market = raw.droplevel("instrument").rename(columns={"$close": "close"}).sort_index()
    market.index = pd.to_datetime(market.index)
    close = pd.to_numeric(market["close"], errors="coerce")
    market["benchmark_return"] = close.shift(-2) / close.shift(-1) - 1.0
    for days in (20, 60, 120, 200):
        market[f"ma{days}"] = close.rolling(days).mean()
        market[f"trend_{days}"] = (close > market[f"ma{days}"]).astype(float)
    market["volatility_20"] = market["benchmark_return"].rolling(20).std() * math.sqrt(252)
    market["vol_percentile"] = market["volatility_20"].rolling(120).rank(pct=True)
    return market


def _annualized(series: pd.Series) -> float:
    total = float((1.0 + series).prod())
    return total ** (252.0 / len(series)) - 1.0 if total > 0 else -1.0


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class TrainedStrategyBacktestService:
    """Load only locally registered training artifacts and rerun their strategy."""

    def __init__(
        self,
        *,
        reports_root: Path | None = None,
        mlruns_root: Path | None = None,
        strategy_config_path: Path | None = None,
        market_loader: MarketLoader | None = None,
    ) -> None:
        self.reports_root = Path(reports_root or resolve_reports_root()).resolve()
        self.mlruns_root = Path(mlruns_root or resolve_mlruns_root()).resolve()
        self.strategy_config_path = Path(
            strategy_config_path
            or (resolve_workflow_root() / "configs" / "candidate_pipeline.yaml")
        ).resolve()
        self.market_loader = market_loader or _default_market_loader

    def run(
        self,
        *,
        run_id: str,
        model_id: str,
        strategy_id: str,
        start_date: str,
        end_date: str,
        topk: int,
        cost_bps: float,
    ) -> dict[str, Any]:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Run ID must contain exactly 32 lowercase hexadecimal characters")
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts > end_ts:
            raise ValueError("start_date cannot be after end_date")
        if not 1 <= int(topk) <= 500:
            raise ValueError("topk must be between 1 and 500")
        if not 0 <= float(cost_bps) <= 1000:
            raise ValueError("cost_bps must be between 0 and 1000")

        selection = self._resolve_selection(run_id, model_id, strategy_id)
        pred = self._read_artifact(selection["experiment_id"], run_id, "pred.pkl")
        label = self._read_artifact(selection["experiment_id"], run_id, "label.pkl")
        pred_slice = _single_column(_slice_frame(pred, start_date, end_date), "score")
        label_slice = _single_column(_slice_frame(label, start_date, end_date), "label")
        if pred_slice.empty or label_slice.empty:
            raise ValueError("The selected run has no prediction and label data in this date range")

        market = self.market_loader(selection["provider_uri"], start_date, end_date).copy()
        market.index = pd.to_datetime(market.index)
        if "benchmark_return" not in market.columns:
            raise ValueError("Market data is missing benchmark_return")

        strategy = OptimizedStrategy(
            topk=int(topk),
            cost_bps=float(cost_bps),
            **selection["strategy_kwargs"],
        )
        backtest = strategy.run(pred_slice, label_slice, market)
        if backtest.empty:
            raise ValueError("The selected TopK is larger than the available stocks in this period")

        returns = pd.to_numeric(backtest["return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if returns.isna().any():
            raise ValueError("The recalculated strategy contains invalid daily returns")
        benchmark = pd.to_numeric(market["benchmark_return"], errors="coerce").reindex(returns.index)
        if benchmark.isna().any():
            missing = int(benchmark.isna().sum())
            raise ValueError(f"SH000300 benchmark is missing {missing} selected trading days")

        strategy_nav = (1.0 + returns).cumprod()
        benchmark_nav = (1.0 + benchmark).cumprod()
        strategy_drawdown = strategy_nav / strategy_nav.cummax() - 1.0
        benchmark_drawdown = benchmark_nav / benchmark_nav.cummax() - 1.0
        excess = returns - benchmark
        return_std = float(returns.std(ddof=1))
        excess_std = float(excess.std(ddof=1))
        annual_return = _annualized(returns)
        benchmark_annual_return = _annualized(benchmark)

        equity_curve = []
        for day in returns.index:
            equity_curve.append(
                {
                    "date": pd.Timestamp(day).date().isoformat(),
                    "strategy_cumulative_return": float(strategy_nav.loc[day] - 1.0),
                    "benchmark_cumulative_return": float(benchmark_nav.loc[day] - 1.0),
                    "drawdown": float(strategy_drawdown.loc[day]),
                    "benchmark_drawdown": float(benchmark_drawdown.loc[day]),
                    "daily_return": float(returns.loc[day]),
                    "benchmark_return": float(benchmark.loc[day]),
                    "turnover": float(backtest.loc[day, "turnover"]),
                    "position_pct": float(backtest.loc[day, "position_pct"]),
                }
            )

        warnings: list[str] = []
        actual_start = equity_curve[0]["date"]
        actual_end = equity_curve[-1]["date"]
        if actual_start != start_date or actual_end != end_date:
            warnings.append(
                f"The saved run contains evaluable trading days from {actual_start} to {actual_end}."
            )

        return {
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": run_id,
            "model_id": model_id,
            "model_label": selection["model_label"],
            "strategy_id": selection["strategy_id"],
            "strategy_label": selection["strategy_label"],
            "provider": selection["provider"],
            "period": {"start_date": actual_start, "end_date": actual_end},
            "parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "topk": int(topk),
                "cost_bps": float(cost_bps),
            },
            "metrics": {
                "days": int(len(returns)),
                "total_return": float(strategy_nav.iloc[-1] - 1.0),
                "benchmark_total_return": float(benchmark_nav.iloc[-1] - 1.0),
                "annual_return": annual_return,
                "benchmark_annual_return": benchmark_annual_return,
                "annual_excess_return": annual_return - benchmark_annual_return,
                "information_ratio": _finite_or_none(
                    float(excess.mean()) / excess_std * math.sqrt(252) if excess_std > 0 else None
                ),
                "sharpe": _finite_or_none(
                    float(returns.mean()) / return_std * math.sqrt(252) if return_std > 0 else None
                ),
                "max_drawdown": float(strategy_drawdown.min()),
                "avg_turnover": float(backtest["turnover"].mean()),
                "avg_position": float(backtest["position_pct"].mean()),
            },
            "equity_curve": equity_curve,
            "warnings": warnings,
        }

    def _resolve_selection(self, run_id: str, model_id: str, strategy_id: str) -> dict[str, Any]:
        if model_id == "latest":
            report = _load_json(self.reports_root / "latest_training_result.json")
            if report.get("status") != "completed" or report.get("run_id") != run_id:
                raise ValueError("Run is not available in the current training results")
            experiment_id = str(report.get("experiment_id") or "")
            self._validate_experiment_id(experiment_id)
            normalized_strategy_id = "baseline_daily" if strategy_id == "qlib_default" else strategy_id
            strategy_label, strategy_kwargs = self._strategy_config(normalized_strategy_id)
            return {
                "experiment_id": experiment_id,
                "model_label": str(report.get("experiment_name") or "Latest training result"),
                "strategy_id": normalized_strategy_id,
                "strategy_label": strategy_label,
                "strategy_kwargs": strategy_kwargs,
                "provider": str(report.get("provider") or "unknown"),
                "provider_uri": Path(str(report.get("provider_uri") or "")).expanduser(),
            }

        report = _load_json(self.reports_root / "candidate_pipeline_latest.json")
        models = report.get("models") if isinstance(report.get("models"), list) else []
        model = next(
            (
                item
                for item in models
                if isinstance(item, dict)
                and item.get("id") == model_id
                and item.get("run_id") == run_id
                and item.get("status") == "completed"
            ),
            None,
        )
        if model is None:
            raise ValueError("Run is not available in the current training results")
        evaluations = (
            report.get("strategy_evaluations")
            if isinstance(report.get("strategy_evaluations"), list)
            else []
        )
        evaluation = next(
            (
                item
                for item in evaluations
                if isinstance(item, dict)
                and item.get("model_id") == model_id
                and item.get("strategy_id") == strategy_id
            ),
            None,
        )
        if evaluation is None:
            raise ValueError("Strategy is not available for the selected trained model")
        experiment_id = str(model.get("experiment_id") or "")
        self._validate_experiment_id(experiment_id)
        strategy_label, strategy_kwargs = self._strategy_config(strategy_id)
        return {
            "experiment_id": experiment_id,
            "model_label": str(model.get("label") or model_id),
            "strategy_id": strategy_id,
            "strategy_label": str(evaluation.get("strategy_label") or strategy_label),
            "strategy_kwargs": strategy_kwargs,
            "provider": str(report.get("provider") or "unknown"),
            "provider_uri": Path(str(report.get("provider_uri") or "")).expanduser(),
        }

    def _strategy_config(self, strategy_id: str) -> tuple[str, dict[str, Any]]:
        payload = yaml.safe_load(self.strategy_config_path.read_text(encoding="utf-8")) or {}
        strategies = payload.get("strategies") if isinstance(payload, dict) else None
        if not isinstance(strategies, list):
            raise ValueError("Candidate strategy configuration is invalid")
        row = next(
            (item for item in strategies if isinstance(item, dict) and item.get("id") == strategy_id),
            None,
        )
        if row is None:
            raise ValueError("Strategy configuration is missing")
        kwargs = row.get("kwargs") if isinstance(row.get("kwargs"), dict) else {}
        unsupported = set(kwargs) - ALLOWED_STRATEGY_KWARGS
        if unsupported:
            raise ValueError(f"Strategy configuration contains unsupported fields: {', '.join(sorted(unsupported))}")
        return str(row.get("label") or strategy_id), dict(kwargs)

    def _read_artifact(self, experiment_id: str, run_id: str, filename: str) -> pd.DataFrame:
        root = self.mlruns_root.resolve()
        path = (root / experiment_id / run_id / "artifacts" / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Resolved artifact path escapes the MLflow data root")
        if not path.is_file():
            raise FileNotFoundError(f"Saved training artifact is missing: {filename}")
        if path.stat().st_size > MAX_PICKLE_BYTES:
            raise ValueError(f"Saved training artifact is too large: {filename}")
        value = pd.read_pickle(path)
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise ValueError(f"Saved training artifact has an unsupported type: {filename}")
        return value.to_frame() if isinstance(value, pd.Series) else value

    @staticmethod
    def _validate_experiment_id(experiment_id: str) -> None:
        if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
            raise ValueError("Training result contains an invalid experiment ID")
