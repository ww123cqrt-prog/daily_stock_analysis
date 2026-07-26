#!/usr/bin/env python3
"""Generate read-only Qlib research reports for the DSA workbench.

The report sections turn existing Qlib/MLflow artifacts into browser-readable
diagnostics. They do not train models, mutate Qlib providers, connect to broker
software, or place orders.
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

from _paths import mlruns_root, qlib_provider_uri, reports_root, workflow_root  # noqa: E402


WORKFLOW = workflow_root()
REPORT_DIR = reports_root()
MLRUNS_DIR = mlruns_root()
CANONICAL_PROVIDER = qlib_provider_uri()
LATEST_TRAINING_RESULT = REPORT_DIR / "latest_training_result.json"
DEFAULT_MAX_RUNS = 8
DEFAULT_TOPK = 20
SECTION_TITLES = {
    "model_explainability": "模型效果深度解释",
    "backtest_detail": "回测详细分解",
    "model_comparison": "多选股模型/多实验对比",
    "factor_analysis": "因子挖掘辅助",
    "rolling_training": "滚动训练计划",
    "risk_model": "风险模型集成诊断",
}
SECTIONS = tuple(SECTION_TITLES)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_run_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw and not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError("run_id may only contain letters, numbers, underscores, or hyphens.")
    return raw


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return _safe_float(value)
    return value


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _metric_value(path: Path) -> float | None:
    text = _read_text(path)
    if not text:
        return None
    last = text.splitlines()[-1].split()
    if len(last) < 2:
        return None
    return _safe_float(last[1])


def _run_dirs(mlruns_dir: Path = MLRUNS_DIR) -> list[Path]:
    if not mlruns_dir.exists():
        return []
    dirs = []
    for meta in mlruns_dir.glob("*/*/meta.yaml"):
        run_dir = meta.parent
        if (run_dir / "artifacts").exists():
            dirs.append(run_dir)
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def _latest_training_run_id(latest_training_path: Path = LATEST_TRAINING_RESULT) -> str:
    latest = _read_json_object(latest_training_path) or {}
    return _safe_run_id(latest.get("run_id"))


def find_run_dir(
    run_id: str | None = None,
    *,
    latest_training_path: Path = LATEST_TRAINING_RESULT,
    mlruns_dir: Path = MLRUNS_DIR,
) -> Path:
    requested = _safe_run_id(run_id) or _latest_training_run_id(latest_training_path)
    if requested:
        matches = sorted(mlruns_dir.glob(f"*/{requested}"))
        matches = [path for path in matches if path.is_dir() and (path / "artifacts").exists()]
        if matches:
            return matches[-1]
    runs = _run_dirs(mlruns_dir)
    if runs:
        return runs[0]
    detail = f" for run {requested}" if requested else ""
    raise FileNotFoundError(f"Could not find a Qlib MLflow run{detail} under {mlruns_dir}.")


def _artifact(run_dir: Path, relative: str) -> Path:
    return run_dir / "artifacts" / relative


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _experiment_name(run_dir: Path) -> str:
    exp_meta = run_dir.parent / "meta.yaml"
    payload = _read_yaml_object(exp_meta) or {}
    return str(payload.get("name") or run_dir.parent.name)


def read_run_summary(run_dir: Path) -> dict[str, Any]:
    metrics_dir = run_dir / "metrics"
    params_dir = run_dir / "params"
    metrics: dict[str, float] = {}
    if metrics_dir.exists():
        for metric_file in metrics_dir.iterdir():
            if metric_file.is_file():
                value = _metric_value(metric_file)
                if value is not None:
                    metrics[metric_file.name] = value
    params: dict[str, str] = {}
    if params_dir.exists():
        for param_file in params_dir.iterdir():
            if param_file.is_file():
                params[param_file.name] = _read_text(param_file)
    meta = _read_yaml_object(run_dir / "meta.yaml") or {}
    return {
        "run_id": _run_id(run_dir),
        "experiment_name": _experiment_name(run_dir),
        "status": str(meta.get("status") or ""),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "model_class": params.get("model.class", ""),
        "dataset_train": params.get("dataset.kwargs.segments.train", ""),
        "dataset_valid": params.get("dataset.kwargs.segments.valid", ""),
        "dataset_test": params.get("dataset.kwargs.segments.test", ""),
        "metrics": metrics,
        "artifact_flags": {
            "pred": _artifact(run_dir, "pred.pkl").exists(),
            "label": _artifact(run_dir, "label.pkl").exists(),
            "portfolio": _artifact(run_dir, "portfolio_analysis/report_normal_1day.pkl").exists(),
            "positions": _artifact(run_dir, "portfolio_analysis/positions_normal_1day.pkl").exists(),
        },
    }


def load_prediction_label(run_dir: Path) -> pd.DataFrame:
    pred_path = _artifact(run_dir, "pred.pkl")
    label_path = _artifact(run_dir, "label.pkl")
    pred = pd.read_pickle(pred_path)
    label = pd.read_pickle(label_path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame("score")
    if isinstance(label, pd.Series):
        label = label.to_frame("label")
    pred_col = "score" if "score" in pred.columns else str(pred.columns[0])
    label_col = "LABEL0" if "LABEL0" in label.columns else str(label.columns[0])
    frame = pd.concat(
        [
            pd.to_numeric(pred[pred_col], errors="coerce").rename("score"),
            pd.to_numeric(label[label_col], errors="coerce").rename("label"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("Prediction/label artifacts have no aligned usable rows.")
    return frame.sort_index()


def load_portfolio_report(run_dir: Path) -> pd.DataFrame:
    path = _artifact(run_dir, "portfolio_analysis/report_normal_1day.pkl")
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Portfolio report must be a DataFrame: {path}")
    data = frame.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        date_col = next((name for name in ("datetime", "date") if name in data.columns), None)
        if date_col is None:
            raise ValueError("Portfolio report must have a DatetimeIndex or a date column.")
        data.index = pd.to_datetime(data[date_col], errors="coerce")
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.loc[data.index.notna()].sort_index()
    if data.empty:
        raise ValueError("Portfolio report has no usable rows.")
    return data


def load_indicators(run_dir: Path) -> pd.DataFrame | None:
    path = _artifact(run_dir, "portfolio_analysis/indicators_normal_1day.pkl")
    if not path.exists():
        return None
    payload = pd.read_pickle(path)
    if not isinstance(payload, pd.DataFrame):
        return None
    frame = payload.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame.loc[frame.index.notna()].sort_index()


def load_positions(run_dir: Path) -> dict[Any, Any] | None:
    path = _artifact(run_dir, "portfolio_analysis/positions_normal_1day.pkl")
    if not path.exists():
        return None
    payload = pd.read_pickle(path)
    return payload if isinstance(payload, dict) else None


def calculate_return_metrics(returns: pd.Series, trading_days_per_year: int = 244) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if clean.empty:
        return {"days": 0}
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1
    std = clean.std()
    return {
        "days": int(len(clean)),
        "cumulative_return": _safe_float(equity.iloc[-1] - 1),
        "annualized_return": _safe_float(equity.iloc[-1] ** (trading_days_per_year / len(clean)) - 1),
        "mean_daily_return": _safe_float(clean.mean()),
        "daily_volatility": _safe_float(std),
        "sharpe_ratio": _safe_float(clean.mean() / std * math.sqrt(trading_days_per_year)) if std and std > 0 else None,
        "max_drawdown": _safe_float(drawdown.min()),
        "win_rate": _safe_float((clean > 0).mean()),
        "best_day": _safe_float(clean.max()),
        "worst_day": _safe_float(clean.min()),
    }


def _feature_importance(run_dir: Path, topk: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_path = _artifact(run_dir, "params.pkl")
    if not model_path.exists():
        return [], {"status": "missing", "reason": "params.pkl is not available."}
    try:
        model = pd.read_pickle(model_path)
        if hasattr(model, "get_feature_importance"):
            importance = model.get_feature_importance()
            if isinstance(importance, pd.Series):
                series = importance.sort_values(ascending=False).head(topk)
                return (
                    [{"feature": str(name), "importance": _safe_float(value)} for name, value in series.items()],
                    {"status": "available", "source": "model.get_feature_importance"},
                )
        booster = getattr(model, "model", None)
        if booster is not None and hasattr(booster, "feature_importance"):
            names = booster.feature_name()
            values = booster.feature_importance(importance_type="gain")
            series = pd.Series(values, index=names).sort_values(ascending=False).head(topk)
            return (
                [{"feature": str(name), "importance": _safe_float(value)} for name, value in series.items()],
                {"status": "available", "source": "lightgbm_gain"},
            )
    except Exception as exc:
        return [], {"status": "unavailable", "reason": str(exc)}
    return [], {"status": "missing", "reason": "Model does not expose feature importance."}


def _prediction_contrast(frame: pd.DataFrame, topk: int) -> dict[str, Any]:
    latest_date = frame.index.get_level_values("datetime").max() if isinstance(frame.index, pd.MultiIndex) else frame.index.max()
    if isinstance(frame.index, pd.MultiIndex):
        latest = frame.loc[latest_date].copy()
    else:
        latest = frame.copy()
    latest = latest.sort_values("score", ascending=False)
    top = latest.head(topk)
    bottom = latest.tail(topk)
    return {
        "latest_date": str(pd.Timestamp(latest_date).date()),
        "top_mean_score": _safe_float(top["score"].mean()),
        "top_mean_label": _safe_float(top["label"].mean()),
        "bottom_mean_score": _safe_float(bottom["score"].mean()),
        "bottom_mean_label": _safe_float(bottom["label"].mean()),
        "score_spread": _safe_float(top["score"].mean() - bottom["score"].mean()),
        "label_spread": _safe_float(top["label"].mean() - bottom["label"].mean()),
    }


def model_explainability(run_dir: Path, *, topk: int = DEFAULT_TOPK) -> dict[str, Any]:
    summary = read_run_summary(run_dir)
    features, feature_meta = _feature_importance(run_dir, topk)
    metrics = summary.get("metrics", {})
    diagnostics = {
        "train_loss": metrics.get("l2.train"),
        "valid_loss": metrics.get("l2.valid"),
        "valid_minus_train_loss": (
            _safe_float(metrics.get("l2.valid") - metrics.get("l2.train"))
            if metrics.get("l2.train") is not None and metrics.get("l2.valid") is not None
            else None
        ),
        "overfit_flag": (
            bool(metrics.get("l2.valid") > metrics.get("l2.train") * 1.5)
            if metrics.get("l2.train") not in (None, 0) and metrics.get("l2.valid") is not None
            else None
        ),
    }
    try:
        contrast = _prediction_contrast(load_prediction_label(run_dir), topk=min(topk, 50))
        contrast_status = "available"
    except Exception as exc:
        contrast = {"error": str(exc)}
        contrast_status = "missing"
    status = "available" if features or contrast_status == "available" else "partial"
    return {
        "section": "model_explainability",
        "title": SECTION_TITLES["model_explainability"],
        "status": status,
        "run": summary,
        "feature_importance": {
            "status": feature_meta.get("status"),
            "source": feature_meta.get("source"),
            "top": features,
        },
        "prediction_explanation": {
            "status": contrast_status,
            "method": "latest-date top/bottom score contrast",
            **contrast,
        },
        "model_diagnostics": diagnostics,
        "limitations": [
            "SHAP is not computed in this browser-safe first version; feature importance comes from the trained model artifact.",
            "Single-stock factor contribution needs persisted feature rows for the selected prediction date.",
        ],
        "next_actions": [
            "Persist test-set feature matrix or sample rows during training.",
            "Add optional SHAP calculation with row and feature limits.",
        ],
    }


def _monthly_returns(returns: pd.Series) -> list[dict[str, Any]]:
    monthly = (1 + returns).resample("ME").prod() - 1
    return [
        {"month": str(index.date())[:7], "return": _safe_float(value)}
        for index, value in monthly.tail(24).items()
    ]


def _positions_summary(positions: dict[Any, Any] | None) -> dict[str, Any]:
    if not positions:
        return {"status": "missing"}
    dates = sorted(pd.Timestamp(key) for key in positions)
    latest_key = next((key for key in positions if pd.Timestamp(key) == dates[-1]), None)
    latest_payload = positions.get(latest_key, {}) if latest_key is not None else {}
    raw_position = latest_payload.get("position", {}) if isinstance(latest_payload, dict) else {}
    holdings = []
    for symbol, item in raw_position.items():
        if not isinstance(item, dict):
            continue
        holdings.append({
            "instrument": str(symbol),
            "weight": _safe_float(item.get("weight")),
            "amount": _safe_float(item.get("amount")),
            "price": _safe_float(item.get("price")),
            "holding_days": _safe_int(item.get("count_day")),
        })
    holdings = sorted(holdings, key=lambda item: item.get("weight") or 0, reverse=True)
    holding_days = [item["holding_days"] for item in holdings if item.get("holding_days") is not None]
    return {
        "status": "available",
        "latest_date": str(dates[-1].date()),
        "holding_count": len(holdings),
        "top_holdings": holdings[:20],
        "average_holding_days": _safe_float(np.mean(holding_days)) if holding_days else None,
        "max_holding_days": max(holding_days) if holding_days else None,
    }


def _trade_activity_proxy(positions: dict[Any, Any] | None) -> dict[str, Any]:
    if not positions:
        return {"status": "missing"}
    previous: set[str] = set()
    rows = []
    for key in sorted(positions, key=lambda item: pd.Timestamp(item)):
        payload = positions.get(key, {})
        raw_position = payload.get("position", {}) if isinstance(payload, dict) else {}
        current = {str(symbol) for symbol, item in raw_position.items() if isinstance(item, dict)}
        buys = sorted(current - previous)
        sells = sorted(previous - current)
        if buys or sells:
            rows.append({
                "date": str(pd.Timestamp(key).date()),
                "buy_count": len(buys),
                "sell_count": len(sells),
                "sample_buys": buys[:5],
                "sample_sells": sells[:5],
            })
        previous = current
    return {
        "status": "available",
        "event_days": len(rows),
        "recent_events": rows[-30:],
    }


def backtest_detail(run_dir: Path) -> dict[str, Any]:
    summary = read_run_summary(run_dir)
    report = load_portfolio_report(run_dir)
    returns = pd.to_numeric(report["return"], errors="coerce").dropna() if "return" in report else pd.Series(dtype="float64")
    bench = pd.to_numeric(report["bench"], errors="coerce").dropna() if "bench" in report else pd.Series(dtype="float64")
    turnover = pd.to_numeric(report.get("turnover"), errors="coerce").dropna() if "turnover" in report else pd.Series(dtype="float64")
    cost = pd.to_numeric(report.get("cost"), errors="coerce").dropna() if "cost" in report else pd.Series(dtype="float64")
    indicators = load_indicators(run_dir)
    positions = load_positions(run_dir)
    daily = report.copy()
    daily["return"] = pd.to_numeric(daily.get("return"), errors="coerce")
    best_days = daily.sort_values("return", ascending=False).head(10)
    worst_days = daily.sort_values("return", ascending=True).head(10)
    return {
        "section": "backtest_detail",
        "title": SECTION_TITLES["backtest_detail"],
        "status": "available",
        "run": summary,
        "metrics": {
            "strategy": calculate_return_metrics(returns),
            "benchmark": calculate_return_metrics(bench) if not bench.empty else None,
            "average_turnover": _safe_float(turnover.mean()) if not turnover.empty else None,
            "average_cost": _safe_float(cost.mean()) if not cost.empty else None,
            "total_cost": _safe_float(cost.sum()) if not cost.empty else None,
            "indicator_means": (
                {str(col): _safe_float(pd.to_numeric(indicators[col], errors="coerce").mean()) for col in indicators.columns}
                if indicators is not None
                else {}
            ),
        },
        "monthly_returns": _monthly_returns(returns) if not returns.empty else [],
        "best_days": [
            {"date": str(index.date()), "return": _safe_float(row.get("return")), "bench": _safe_float(row.get("bench"))}
            for index, row in best_days.iterrows()
        ],
        "worst_days": [
            {"date": str(index.date()), "return": _safe_float(row.get("return")), "bench": _safe_float(row.get("bench"))}
            for index, row in worst_days.iterrows()
        ],
        "positions": _positions_summary(positions),
        "trade_activity": _trade_activity_proxy(positions),
    }


def model_comparison(*, mlruns_dir: Path = MLRUNS_DIR, max_runs: int = DEFAULT_MAX_RUNS) -> dict[str, Any]:
    runs = _run_dirs(mlruns_dir)[:max(1, max_runs)]
    summaries = [read_run_summary(path) for path in runs]
    key_metrics = [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
        "l2.valid",
    ]
    table = []
    for summary in summaries:
        metrics = summary.get("metrics", {})
        table.append({
            "run_id": summary.get("run_id"),
            "experiment_name": summary.get("experiment_name"),
            "model_class": summary.get("model_class"),
            **{key: metrics.get(key) for key in key_metrics},
        })
    best = {}
    for key in key_metrics:
        rows = [row for row in table if row.get(key) is not None]
        if not rows:
            continue
        reverse = key != "l2.valid" and not key.endswith("max_drawdown")
        best[key] = sorted(rows, key=lambda row: row.get(key), reverse=reverse)[0]
    correlations = []
    prediction_series = []
    for run_dir in runs[: min(len(runs), 5)]:
        pred_path = _artifact(run_dir, "pred.pkl")
        if not pred_path.exists():
            continue
        try:
            pred = pd.read_pickle(pred_path)
            if isinstance(pred, pd.DataFrame):
                col = "score" if "score" in pred.columns else str(pred.columns[0])
                series = pred[col]
            else:
                series = pred
            prediction_series.append((_run_id(run_dir), pd.to_numeric(series, errors="coerce").dropna()))
        except Exception:
            continue
    for idx, (left_id, left) in enumerate(prediction_series):
        for right_id, right in prediction_series[idx + 1:]:
            aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
            if len(aligned) < 3:
                continue
            correlations.append({"left": left_id, "right": right_id, "corr": _safe_float(aligned["left"].corr(aligned["right"]))})
    return {
        "section": "model_comparison",
        "title": SECTION_TITLES["model_comparison"],
        "status": "available" if summaries else "missing",
        "run_count": len(summaries),
        "metric_keys": key_metrics,
        "runs": table,
        "best_by_metric": best,
        "prediction_correlations": correlations,
        "limitations": [
            "This compares completed local MLflow runs. Training CatBoost/Linear/Transformer models is a separate long-running job.",
        ],
    }


def factor_analysis(run_dir: Path, *, topk: int = DEFAULT_TOPK) -> dict[str, Any]:
    frame = load_prediction_label(run_dir)
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError("Prediction/label frame must use Qlib MultiIndex for factor analysis.")
    daily_rows = []
    for date, group in frame.groupby(level="datetime"):
        if len(group) < 5:
            continue
        daily_rows.append({
            "date": str(pd.Timestamp(date).date()),
            "ic": _safe_float(group["score"].corr(group["label"], method="pearson")),
            "rank_ic": _safe_float(group["score"].corr(group["label"], method="spearman")),
            "count": int(len(group)),
        })
    daily = pd.DataFrame(daily_rows)
    quantile_rows = []
    for date, group in frame.groupby(level="datetime"):
        if len(group) < 10:
            continue
        ranked = group.copy()
        try:
            ranked["bucket"] = pd.qcut(ranked["score"].rank(method="first"), 5, labels=False) + 1
        except ValueError:
            continue
        means = ranked.groupby("bucket")["label"].mean()
        quantile_rows.append({"date": str(pd.Timestamp(date).date()), **{f"q{int(k)}": _safe_float(v) for k, v in means.items()}})
    top_features, feature_meta = _feature_importance(run_dir, topk)
    return {
        "section": "factor_analysis",
        "title": SECTION_TITLES["factor_analysis"],
        "status": "available" if not daily.empty else "partial",
        "run": read_run_summary(run_dir),
        "score_factor": {
            "daily_count": int(len(daily)),
            "mean_ic": _safe_float(daily["ic"].mean()) if not daily.empty else None,
            "mean_rank_ic": _safe_float(daily["rank_ic"].mean()) if not daily.empty else None,
            "ic_ir": _safe_float(daily["ic"].mean() / daily["ic"].std()) if len(daily) > 1 and daily["ic"].std() else None,
            "rank_ic_ir": _safe_float(daily["rank_ic"].mean() / daily["rank_ic"].std()) if len(daily) > 1 and daily["rank_ic"].std() else None,
            "recent_daily_ic": daily.tail(30).to_dict(orient="records") if not daily.empty else [],
        },
        "quantile_returns": quantile_rows[-30:],
        "feature_importance_proxy": {
            "status": feature_meta.get("status"),
            "top": top_features,
            "note": "Feature importance is a proxy for factor screening until per-factor IC artifacts are persisted.",
        },
        "next_actions": [
            "Persist Alpha158 feature rows for sampled dates.",
            "Add formula-based custom factor IC tests after the factor registry is defined.",
        ],
    }


def _calendar(provider_uri: Path = CANONICAL_PROVIDER) -> pd.DatetimeIndex:
    path = provider_uri / "calendars" / "day.txt"
    if not path.exists():
        return pd.DatetimeIndex([])
    dates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).dropna().sort_values()


def _nearest_on_or_before(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    values = calendar[calendar <= date]
    return pd.Timestamp(values[-1]) if len(values) else None


def _nearest_on_or_after(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    values = calendar[calendar >= date]
    return pd.Timestamp(values[0]) if len(values) else None


def rolling_training_plan(
    *,
    provider_uri: Path = CANONICAL_PROVIDER,
    config_path: Path = WORKFLOW / "configs" / "baseline_lgb_alpha158.yaml",
    windows: int = 6,
    train_years: int = 3,
    valid_months: int = 12,
    test_months: int = 1,
    step_months: int = 1,
) -> dict[str, Any]:
    calendar = _calendar(provider_uri)
    config = _read_yaml_object(config_path) or {}
    if calendar.empty:
        return {
            "section": "rolling_training",
            "title": SECTION_TITLES["rolling_training"],
            "status": "missing",
            "provider_uri": str(provider_uri),
            "reason": "Qlib calendar is missing.",
            "config_segments": (((config.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments", {}),
        }
    plans = []
    anchor = pd.Timestamp(calendar[-1])
    for offset in range(max(1, windows)):
        test_end = _nearest_on_or_before(calendar, anchor - pd.DateOffset(months=offset * step_months))
        if test_end is None:
            continue
        test_start_raw = test_end - pd.DateOffset(months=test_months) + pd.Timedelta(days=1)
        test_start = _nearest_on_or_after(calendar, test_start_raw)
        valid_end = _nearest_on_or_before(calendar, test_start - pd.Timedelta(days=1)) if test_start is not None else None
        valid_start_raw = valid_end - pd.DateOffset(months=valid_months) + pd.Timedelta(days=1) if valid_end is not None else None
        valid_start = _nearest_on_or_after(calendar, valid_start_raw) if valid_start_raw is not None else None
        train_end = _nearest_on_or_before(calendar, valid_start - pd.Timedelta(days=1)) if valid_start is not None else None
        train_start_raw = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1) if train_end is not None else None
        train_start = _nearest_on_or_after(calendar, train_start_raw) if train_start_raw is not None else None
        if None in (train_start, train_end, valid_start, valid_end, test_start, test_end):
            continue
        plans.append({
            "window": len(plans) + 1,
            "train": [str(train_start.date()), str(train_end.date())],
            "valid": [str(valid_start.date()), str(valid_end.date())],
            "test": [str(test_start.date()), str(test_end.date())],
            "suggested_experiment": f"rolling_lgb_alpha158_{test_end.strftime('%Y%m')}",
        })
    return {
        "section": "rolling_training",
        "title": SECTION_TITLES["rolling_training"],
        "status": "available" if plans else "partial",
        "provider_uri": str(provider_uri),
        "calendar_range": f"{calendar[0].date()} ~ {calendar[-1].date()}",
        "calendar_days": int(len(calendar)),
        "config_path": str(config_path),
        "config_segments": (((config.get("task") or {}).get("dataset") or {}).get("kwargs") or {}).get("segments", {}),
        "policy": {
            "train_years": train_years,
            "valid_months": valid_months,
            "test_months": test_months,
            "step_months": step_months,
        },
        "windows": plans,
        "next_actions": [
            "Turn each planned window into a runtime config.",
            "Run monthly retraining as separate durable jobs and compare against the current champion model.",
        ],
    }


def risk_model(run_dir: Path) -> dict[str, Any]:
    report = load_portfolio_report(run_dir)
    returns = pd.to_numeric(report.get("return"), errors="coerce").dropna() if "return" in report else pd.Series(dtype="float64")
    bench = pd.to_numeric(report.get("bench"), errors="coerce").dropna() if "bench" in report else pd.Series(dtype="float64")
    turnover = pd.to_numeric(report.get("turnover"), errors="coerce").dropna() if "turnover" in report else pd.Series(dtype="float64")
    positions = _positions_summary(load_positions(run_dir))
    top_holdings = positions.get("top_holdings") or []
    weights = [item.get("weight") for item in top_holdings if item.get("weight") is not None]
    beta = None
    if not returns.empty and not bench.empty:
        aligned = pd.concat([returns.rename("strategy"), bench.rename("bench")], axis=1).dropna()
        if len(aligned) > 2 and aligned["bench"].var() > 0:
            beta = _safe_float(aligned["strategy"].cov(aligned["bench"]) / aligned["bench"].var())
    max_weight = max(weights) if weights else None
    hhi = sum(float(weight) ** 2 for weight in weights) if weights else None
    recommendations = []
    if max_weight is not None and max_weight > 0.05:
        recommendations.append("Set max_single_weight <= 5% or review concentrated names.")
    if not turnover.empty and turnover.mean() > 0.30:
        recommendations.append("Review turnover cap and transaction costs before live-like evaluation.")
    if beta is not None and abs(beta) > 1.2:
        recommendations.append("Add beta or benchmark exposure checks to the strategy report.")
    if not recommendations:
        recommendations.append("No immediate concentration breach detected from latest available artifacts.")
    return {
        "section": "risk_model",
        "title": SECTION_TITLES["risk_model"],
        "status": "available",
        "run": read_run_summary(run_dir),
        "risk_metrics": {
            "strategy": calculate_return_metrics(returns),
            "beta_to_benchmark": beta,
            "average_turnover": _safe_float(turnover.mean()) if not turnover.empty else None,
            "max_single_weight_latest": _safe_float(max_weight),
            "top5_weight_latest": _safe_float(sum(weights[:5])) if weights else None,
            "herfindahl_latest_top_holdings": _safe_float(hhi),
        },
        "constraint_checks": {
            "max_single_weight_5pct": "pass" if max_weight is None or max_weight <= 0.05 else "breach",
            "top5_weight_30pct": "pass" if not weights or sum(weights[:5]) <= 0.30 else "breach",
        },
        "latest_positions": positions,
        "recommendations": recommendations,
        "limitations": [
            "Industry/style neutralization requires industry and style exposure data not yet persisted in this workflow.",
        ],
    }


def build_section_payload(section: str, args: argparse.Namespace) -> dict[str, Any]:
    if section == "model_comparison":
        return model_comparison(mlruns_dir=args.mlruns_dir, max_runs=args.max_runs)
    if section == "rolling_training":
        return rolling_training_plan(
            provider_uri=args.provider_uri,
            config_path=args.config,
            windows=args.windows,
            train_years=args.train_years,
            valid_months=args.valid_months,
            test_months=args.test_months,
            step_months=args.step_months,
        )
    run_dir = find_run_dir(args.run_id, latest_training_path=args.latest_training, mlruns_dir=args.mlruns_dir)
    if section == "model_explainability":
        return model_explainability(run_dir, topk=args.topk)
    if section == "backtest_detail":
        return backtest_detail(run_dir)
    if section == "factor_analysis":
        return factor_analysis(run_dir, topk=args.topk)
    if section == "risk_model":
        return risk_model(run_dir)
    raise ValueError(f"Unsupported section: {section}")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_markdown(payload: dict[str, Any]) -> str:
    if payload.get("section") == "all":
        lines = [
            "# Qlib Research Suite",
            "",
            f"Generated: {payload.get('generated_at')}",
            "",
            "This is a read-only DSA workbench report. It does not connect to broker software, 不读取交易密码、不自动下单，也不改写 Qlib provider。",
            "",
        ]
        for section in payload.get("sections", []):
            lines.extend([f"## {section.get('title')}", "", f"- Status: `{section.get('status')}`"])
            for action in section.get("next_actions", [])[:3]:
                lines.append(f"- Next: {action}")
            lines.append("")
        return "\n".join(lines)

    lines = [
        f"# Qlib Research - {payload.get('title')}",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Status: `{payload.get('status')}`",
        "",
        "This report is read-only: it does not connect to broker software, 不读取交易密码、不自动下单，也不改写 Qlib provider。",
        "",
    ]
    if payload.get("run"):
        run = payload["run"]
        lines.extend([
            "## Run",
            "",
            f"- Run id: `{run.get('run_id')}`",
            f"- Experiment: `{run.get('experiment_name')}`",
            f"- Model: `{run.get('model_class') or 'n/a'}`",
            "",
        ])
    if payload.get("metrics"):
        lines.extend(["## Metrics", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in payload["metrics"].items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"| {key}.{sub_key} | {_fmt(sub_value)} |")
            else:
                lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")
    if payload.get("feature_importance", {}).get("top"):
        lines.extend(["## Feature Importance", "", "| Feature | Importance |", "| --- | ---: |"])
        for item in payload["feature_importance"]["top"][:20]:
            lines.append(f"| `{item.get('feature')}` | {_fmt(item.get('importance'))} |")
        lines.append("")
    if payload.get("score_factor"):
        score = payload["score_factor"]
        lines.extend([
            "## Score Factor IC",
            "",
            f"- Mean IC: {_fmt(score.get('mean_ic'))}",
            f"- Mean RankIC: {_fmt(score.get('mean_rank_ic'))}",
            f"- ICIR: {_fmt(score.get('ic_ir'))}",
            "",
        ])
    for action in payload.get("next_actions", []):
        lines.append(f"- Next: {action}")
    for limitation in payload.get("limitations", []):
        lines.append(f"- Limitation: {limitation}")
    return "\n".join(lines).rstrip() + "\n"


def write_report(section: str, payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"qlib_research_{section}_{ts}.json"
    md_path = out_dir / f"qlib_research_{section}_{ts}.md"
    latest_json = out_dir / f"qlib_research_{section}_latest.json"
    latest_md = out_dir / f"qlib_research_{section}_latest.md"
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "read_only_qlib_research",
        **payload,
    }
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    serialized = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    markdown = build_markdown(payload)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate read-only Qlib research reports.")
    parser.add_argument("--section", choices=("all", *SECTIONS), default="all")
    parser.add_argument("--run-id", default="", help="Optional MLflow run id for run-specific reports.")
    parser.add_argument("--latest-training", type=Path, default=LATEST_TRAINING_RESULT)
    parser.add_argument("--mlruns-dir", type=Path, default=MLRUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--provider-uri", type=Path, default=CANONICAL_PROVIDER)
    parser.add_argument("--config", type=Path, default=WORKFLOW / "configs" / "baseline_lgb_alpha158.yaml")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--valid-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    args = parser.parse_args()

    if args.topk < 1 or args.topk > 500:
        raise ValueError("topk must be between 1 and 500.")
    if args.max_runs < 1 or args.max_runs > 50:
        raise ValueError("max-runs must be between 1 and 50.")
    if args.windows < 1 or args.windows > 60:
        raise ValueError("windows must be between 1 and 60.")
    if args.train_years < 1 or args.train_years > 20:
        raise ValueError("train-years must be between 1 and 20.")
    for name in ("valid_months", "test_months", "step_months"):
        value = getattr(args, name)
        if value < 1 or value > 120:
            raise ValueError(f"{name.replace('_', '-')} must be between 1 and 120.")

    sections = list(SECTIONS) if args.section == "all" else [args.section]
    outputs = []
    for section in sections:
        payload = build_section_payload(section, args)
        outputs.append(write_report(section, payload, args.out_dir))

    if args.section == "all":
        suite_payload = {
            "section": "all",
            "title": "Qlib 六类研究能力总览",
            "status": "available" if outputs else "missing",
            "sections": [
                {
                    "section": item.get("section"),
                    "title": item.get("title"),
                    "status": item.get("status"),
                    "generated_at": item.get("generated_at"),
                    "artifacts": item.get("artifacts"),
                    "next_actions": item.get("next_actions", []),
                }
                for item in outputs
            ],
        }
        write_report("suite", suite_payload, args.out_dir)

    for item in outputs:
        print(f"[artifact] {item['section']}: {item['artifacts']['latest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
