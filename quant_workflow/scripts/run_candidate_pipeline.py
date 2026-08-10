#!/usr/bin/env python3
"""Train five Qlib models, select on 2024-2025, and verify on the 2026 holdout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (  # noqa: E402
    mlruns_root,
    provider_calendar_days,
    qlib_workflow_init_config,
    qlib_workflow_provider_uri,
    init_qlib_workflow,
    quant_data_root,
    reports_root,
    workflow_root,
)
from _mlflow_run_provenance import (  # noqa: E402
    RunProvenanceError,
    provider_lineage_is_complete,
    provider_lineages_match,
    provider_identity,
    validate_file_store_run_artifact,
)
from _optimized_strategy_core import OptimizedStrategy  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
CONFIGS_DIR = WORKFLOW_ROOT / "configs"
REPORT_DIR = reports_root()
MLRUNS_DIR = mlruns_root()
DEFAULT_SPEC_PATH = CONFIGS_DIR / "candidate_pipeline.yaml"
TRAIN_SCRIPT = SCRIPT_DIR / "train_baseline_lgb.py"
MAX_PICKLE_BYTES = 512 * 1024 * 1024
ALLOWED_MODEL_MODULES = {
    "qlib.contrib.model.linear",
    "qlib.contrib.model.gbdt",
    "qlib.contrib.model.xgboost",
    "qlib.contrib.model.catboost_model",
}


def resolve_pipeline_provider(
    provider: str,
    expected_provider_identity: Mapping[str, Any] | None = None,
) -> Path:
    return qlib_workflow_provider_uri(
        provider,
        expected_provider_identity=expected_provider_identity,
    )


REQUIRED_MODEL_IDS = ["linear_ols", "linear_ridge", "lightgbm", "xgboost", "catboost"]
MODEL_DEPENDENCIES = {
    "linear_ols": "sklearn",
    "linear_ridge": "sklearn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
}
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _date_pair(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two dates")
    result = [str(item) for item in value]
    if name != "holdout_period" or result[1] != "provider_latest_label_day":
        if pd.Timestamp(result[0]) > pd.Timestamp(result[1]):
            raise ValueError(f"{name} start must not be after end")
    return result


def load_pipeline_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Candidate pipeline spec must be a mapping")
    models = payload.get("models")
    strategies = payload.get("strategies")
    if not isinstance(models, list) or [item.get("id") for item in models if isinstance(item, dict)] != REQUIRED_MODEL_IDS:
        raise ValueError(f"Candidate pipeline must define models in this order: {', '.join(REQUIRED_MODEL_IDS)}")
    if not isinstance(strategies, list) or len(strategies) != 5:
        raise ValueError("Candidate pipeline must define exactly five strategies")
    for model in models:
        if model.get("module_path") not in ALLOWED_MODEL_MODULES:
            raise ValueError(f"Unsupported model module: {model.get('module_path')}")
        if not model.get("class") or not isinstance(model.get("kwargs", {}), dict):
            raise ValueError(f"Invalid model definition: {model.get('id')}")
    for field in ("training_period", "validation_period", "selection_period", "holdout_period"):
        payload[field] = _date_pair(payload.get(field), field)
    if pd.Timestamp(payload["selection_period"][1]) >= pd.Timestamp(payload["holdout_period"][0]):
        raise ValueError("selection_period must end before holdout_period starts")
    payload["top_models"] = int(payload.get("top_models", 2))
    payload["topk"] = int(payload.get("topk", 50))
    payload["cost_bps"] = float(payload.get("cost_bps", 15))
    if not 1 <= payload["top_models"] <= len(models):
        raise ValueError("top_models is outside the model candidate range")
    if not 1 <= payload["topk"] <= 500:
        raise ValueError("topk must be between 1 and 500")
    return payload


def build_candidate_config(base: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    task = payload.get("task")
    if not isinstance(task, dict):
        raise ValueError("Base Qlib config is missing task")
    task["model"] = {
        "class": str(model["class"]),
        "module_path": str(model["module_path"]),
        "kwargs": copy.deepcopy(model.get("kwargs") or {}),
    }
    if model.get("class") == "LinearModel":
        dataset = task.get("dataset")
        dataset_kwargs = dataset.get("kwargs") if isinstance(dataset, dict) else None
        handler = dataset_kwargs.get("handler") if isinstance(dataset_kwargs, dict) else None
        handler_kwargs = handler.get("kwargs") if isinstance(handler, dict) else None
        if isinstance(handler_kwargs, dict):
            handler_kwargs["infer_processors"] = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
            handler_kwargs["learn_processors"] = [{"class": "DropnaLabel"}]
    return payload


def _ensure_signal_analysis_record(payload: dict[str, Any]) -> None:
    records = payload["task"].get("record")
    if not isinstance(records, list):
        raise ValueError("Base Qlib config is missing task.record")
    if any(isinstance(item, dict) and item.get("class") == "SigAnaRecord" for item in records):
        return
    insert_at = next(
        (index for index, item in enumerate(records) if isinstance(item, dict) and item.get("class") == "PortAnaRecord"),
        len(records),
    )
    records.insert(
        insert_at,
        {
            "class": "SigAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"ana_long_short": False, "ann_scaler": 252},
        },
    )


def apply_common_periods(
    payload: dict[str, Any],
    spec: dict[str, Any],
    *,
    provider_latest_day: str,
    latest_prediction_day: str,
) -> None:
    dataset = payload["task"].get("dataset")
    dataset_kwargs = dataset.get("kwargs") if isinstance(dataset, dict) else None
    if not isinstance(dataset_kwargs, dict):
        raise ValueError("Base Qlib config is missing task.dataset.kwargs")
    handler = dataset_kwargs.get("handler")
    handler_kwargs = handler.get("kwargs") if isinstance(handler, dict) else None
    if not isinstance(handler_kwargs, dict):
        raise ValueError("Base Qlib config is missing handler kwargs")
    handler_kwargs.update(
        {
            "start_time": spec["training_period"][0],
            "end_time": provider_latest_day,
            "fit_start_time": spec["training_period"][0],
            "fit_end_time": spec["training_period"][1],
            "instruments": "csi300",
        }
    )
    dataset_kwargs["segments"] = {
        "train": list(spec["training_period"]),
        "valid": list(spec["validation_period"]),
        "test": [spec["selection_period"][0], latest_prediction_day],
    }
    _ensure_signal_analysis_record(payload)
    for record in payload["task"]["record"]:
        if not isinstance(record, dict) or record.get("class") != "PortAnaRecord":
            continue
        kwargs = record.setdefault("kwargs", {})
        config = kwargs.get("config")
        if not isinstance(config, dict):
            legacy = record.pop("config", None)
            config = legacy if isinstance(legacy, dict) else {}
            kwargs["config"] = config
        backtest = config.get("backtest")
        if isinstance(backtest, dict):
            backtest["start_time"] = spec["selection_period"][0]
            backtest["end_time"] = latest_prediction_day


def slice_frame(frame: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        raise ValueError("Prediction and label artifacts must use a datetime/instrument MultiIndex")
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame.loc[mask]


def _single_column(frame: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        return frame.rename(name).to_frame()
    if frame.shape[1] != 1:
        raise ValueError(f"{name} artifact must contain exactly one column")
    result = frame.copy()
    result.columns = [name]
    return result


def signal_metrics(pred: pd.DataFrame | pd.Series, label: pd.DataFrame | pd.Series) -> dict[str, float]:
    data = pd.concat([_single_column(pred, "score"), _single_column(label, "label")], axis=1).dropna()
    if data.empty:
        return {"ic": float("nan"), "icir": float("nan"), "rank_ic": float("nan"), "rank_icir": float("nan")}
    daily_ic = data.groupby(level="datetime", sort=True).apply(lambda group: group["score"].corr(group["label"]))
    daily_rank_ic = data.groupby(level="datetime", sort=True).apply(
        lambda group: group["score"].corr(group["label"], method="spearman")
    )

    def summarize(series: pd.Series) -> tuple[float, float]:
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return float("nan"), float("nan")
        mean = float(clean.mean())
        std = float(clean.std(ddof=1))
        return mean, mean / std if std > 0 else float("nan")

    ic, icir = summarize(daily_ic)
    rank_ic, rank_icir = summarize(daily_rank_ic)
    return {"ic": ic, "icir": icir, "rank_ic": rank_ic, "rank_icir": rank_icir}


def performance_metrics(backtest: pd.DataFrame, benchmark_returns: pd.Series) -> dict[str, float | int]:
    if backtest.empty:
        return {
            "days": 0,
            "annual_return": float("nan"),
            "benchmark_annual_return": float("nan"),
            "annual_excess_return": float("nan"),
            "information_ratio": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "avg_turnover": float("nan"),
            "avg_position": float("nan"),
        }
    returns = pd.to_numeric(backtest["return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").reindex(returns.index).fillna(0.0)
    if returns.empty:
        return performance_metrics(pd.DataFrame(), benchmark)

    def annualized(series: pd.Series) -> float:
        total = float((1.0 + series).prod())
        return total ** (252.0 / len(series)) - 1.0 if total > 0 else -1.0

    annual_return = annualized(returns)
    benchmark_annual = annualized(benchmark)
    excess = returns - benchmark
    excess_std = float(excess.std(ddof=1))
    return_std = float(returns.std(ddof=1))
    cumulative = (1.0 + returns).cumprod()
    return {
        "days": int(len(returns)),
        "annual_return": annual_return,
        "benchmark_annual_return": benchmark_annual,
        "annual_excess_return": annual_return - benchmark_annual,
        "information_ratio": float(excess.mean()) / excess_std * math.sqrt(252) if excess_std > 0 else float("nan"),
        "sharpe": float(returns.mean()) / return_std * math.sqrt(252) if return_std > 0 else float("nan"),
        "max_drawdown": float((cumulative / cumulative.cummax() - 1.0).min()),
        "avg_turnover": float(backtest["turnover"].mean()),
        "avg_position": float(backtest["position_pct"].mean()),
    }


def evaluate_period(
    pred: pd.DataFrame | pd.Series,
    label: pd.DataFrame | pd.Series,
    market: pd.DataFrame,
    period: Iterable[str],
    strategy_kwargs: dict[str, Any],
    *,
    topk: int,
    cost_bps: float,
) -> dict[str, Any]:
    start, end = list(period)
    pred_slice = slice_frame(pred, start, end)
    label_slice = slice_frame(label, start, end)
    strategy = OptimizedStrategy(topk=topk, cost_bps=cost_bps, **strategy_kwargs)
    backtest = strategy.run(_single_column(pred_slice, "score"), _single_column(label_slice, "label"), market)
    metrics = performance_metrics(backtest, market["benchmark_return"])
    metrics.update(signal_metrics(pred_slice, label_slice))
    return metrics


def rank_models(candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    completed = [item for item in candidates if item.get("status") == "completed"]

    def key(item: dict[str, Any]) -> tuple[float, float, float]:
        metrics = item.get("selection") if isinstance(item.get("selection"), dict) else {}
        return (
            _finite_float(metrics.get("information_ratio"), -math.inf),
            _finite_float(metrics.get("annual_excess_return"), -math.inf),
            _finite_float(metrics.get("rank_ic"), -math.inf),
        )

    return sorted(completed, key=key, reverse=True)[: int(top_n)]


def pipeline_status(candidates: list[dict[str, Any]]) -> str:
    return "completed" if len(candidates) == 5 and all(item.get("status") == "completed" for item in candidates) else "failed"


def provider_days(provider_uri: Path) -> tuple[str, str, str]:
    days = provider_calendar_days(provider_uri)
    if len(days) < 3:
        raise ValueError(f"Provider calendar needs at least three days: {provider_uri}")
    return days[-1], days[-2], days[-3]


def benchmark_label_returns(close: pd.Series) -> pd.Series:
    """Align SH000300 returns with Alpha158's Ref(close,-2)/Ref(close,-1)-1 label."""
    numeric = pd.to_numeric(close, errors="coerce")
    return numeric.shift(-2) / numeric.shift(-1) - 1.0


def load_market_data(provider_uri: Path, start: str, end: str) -> pd.DataFrame:
    from qlib.data import D

    init_qlib_workflow(provider_uri=provider_uri)
    raw = D.features(["SH000300"], ["$close"], start_time=start, end_time=end)
    if raw.empty:
        raise ValueError("SH000300 benchmark data is unavailable")
    market = raw.droplevel("instrument").rename(columns={"$close": "close"}).sort_index()
    market.index = pd.to_datetime(market.index)
    market["benchmark_return"] = benchmark_label_returns(market["close"])
    for days in (20, 60, 120, 200):
        market[f"ma{days}"] = market["close"].rolling(days).mean()
        market[f"trend_{days}"] = (market["close"] > market[f"ma{days}"]).astype(float)
    historical_return = pd.to_numeric(market["close"], errors="coerce").pct_change(fill_method=None)
    market["volatility_20"] = historical_return.rolling(20).std() * math.sqrt(252)
    market["vol_percentile"] = market["volatility_20"].rolling(120).rank(pct=True)
    return market


def _artifact_path(
    info: dict[str, Any],
    name: str,
    *,
    mlruns_dir: Path = MLRUNS_DIR,
) -> Path:
    direct = mlruns_dir / str(info["experiment_id"]) / str(info["run_id"]) / "artifacts" / name
    if direct.exists():
        return direct
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{mlruns_dir}")
    return Path(client.download_artifacts(str(info["run_id"]), name))


def _load_artifacts(
    info: dict[str, Any],
    *,
    mlruns_dir: Path = MLRUNS_DIR,
    quant_data_root_path: Path | None = None,
    expected_provider_identity: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    root = (
        Path(quant_data_root_path).expanduser().resolve()
        if quant_data_root_path is not None
        else quant_data_root().expanduser().resolve()
    )
    run_id = str(info["run_id"])
    paths = {
        name: _artifact_path(info, name, mlruns_dir=mlruns_dir)
        for name in ("pred.pkl", "label.pkl")
    }
    artifact_bytes: dict[str, bytes] = {}
    for name, path in paths.items():
        try:
            with path.open("rb") as handle:
                payload = handle.read(MAX_PICKLE_BYTES + 1)
        except OSError as exc:
            raise RuntimeError(f"Candidate artifact cannot be read: {name}") from exc
        if len(payload) > MAX_PICKLE_BYTES:
            raise RuntimeError(f"Candidate artifact is too large: {name}")
        artifact_bytes[name] = payload
    try:
        for name, path in paths.items():
            provenance = validate_file_store_run_artifact(
                artifact_path=path,
                artifact_relative_path=Path(name),
                run_id=run_id,
                mlruns_dir=mlruns_dir,
                quant_data_root=root,
            )
            if expected_provider_identity is not None:
                _require_provider_identity(
                    provenance.get("provider_identity"),
                    expected_provider_identity,
                )
            loaded_sha256 = hashlib.sha256(artifact_bytes[name]).hexdigest()
            if loaded_sha256 != str(provenance.get("artifact_sha256") or ""):
                raise RunProvenanceError(
                    f"{name} changed while its provenance was being verified"
                )
    except RunProvenanceError as exc:
        raise RuntimeError(f"Candidate artifact provenance is invalid: {exc}") from exc
    return (
        pd.read_pickle(io.BytesIO(artifact_bytes["pred.pkl"])),
        pd.read_pickle(io.BytesIO(artifact_bytes["label.pkl"])),
    )


def _tag_candidate_run(info: dict[str, Any], batch_id: str, model_id: str) -> None:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{MLRUNS_DIR}")
    run_id = str(info["run_id"])
    client.set_tag(run_id, "candidate_pipeline.batch_id", batch_id)
    client.set_tag(run_id, "candidate_pipeline.model_id", model_id)
    client.set_tag(run_id, "candidate_pipeline.selection_basis", "2024-2025_only")
    client.set_tag(run_id, "mlflow.runName", model_id)


def _tag_pipeline_outcome(
    candidates: list[dict[str, Any]],
    selected_models: list[dict[str, Any]],
    champion: dict[str, Any] | None,
    verdict: dict[str, Any],
) -> None:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{MLRUNS_DIR}")
    selected_ids = {str(item["id"]) for item in selected_models}
    champion_model = str(champion.get("model_id")) if champion else ""
    for candidate in candidates:
        run_id = str(candidate.get("run_id") or "")
        if not run_id:
            continue
        model_id = str(candidate["id"])
        client.set_tag(run_id, "candidate_pipeline.selected_model", str(model_id in selected_ids).lower())
        if model_id == champion_model:
            client.set_tag(run_id, "candidate_pipeline.selected_strategy", str(champion.get("strategy_id") or ""))
            client.set_tag(run_id, "candidate_pipeline.holdout_verdict", str(verdict.get("status") or ""))


def _log_candidate_metrics(info: dict[str, Any], selection: dict[str, Any], holdout: dict[str, Any]) -> None:
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{MLRUNS_DIR}")
    run_id = str(info["run_id"])
    for prefix, metrics in (("candidate.selection", selection), ("candidate.holdout", holdout)):
        for key, value in metrics.items():
            numeric = _finite_float(value)
            if math.isfinite(numeric):
                client.log_metric(run_id, f"{prefix}.{key}", numeric)


def _baseline_strategy_kwargs() -> dict[str, Any]:
    return {"rebalance_freq": 1, "turnover_cap": 1.0, "max_drawdown_stop": 0.0}


def _resolve_base_config(spec: dict[str, Any]) -> Path:
    name = Path(str(spec.get("base_config") or "")).name
    if not name.endswith((".yaml", ".yml")):
        raise ValueError("base_config must be a YAML file under quant_workflow/configs")
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Base config does not exist: {path}")
    return path


def _preflight_dependencies(models: list[dict[str, Any]]) -> None:
    missing = [MODEL_DEPENDENCIES[item["id"]] for item in models if importlib.util.find_spec(MODEL_DEPENDENCIES[item["id"]]) is None]
    if missing:
        raise RuntimeError(f"Missing candidate model dependencies: {', '.join(sorted(set(missing)))}")


def _training_command(
    *,
    config_path: Path,
    experiment_name: str,
    provider: str,
    expected_provider_identity: dict[str, Any],
    result_output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--experiment-name",
        experiment_name,
        "--provider",
        provider,
        "--no-publish-latest",
        "--result-scope",
        "candidate_child",
        "--result-output",
        str(result_output),
        "--provider-identity-json",
        json.dumps(
            expected_provider_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]


def _require_provider_identity(
    actual: Any,
    expected: dict[str, Any],
) -> None:
    if not provider_lineages_match(actual, expected):
        raise RuntimeError(
            "Candidate model used a different immutable provider than the batch snapshot"
        )


def _backtest_eligibility(
    info: Mapping[str, Any],
    provider: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    missing: list[str] = []
    if dry_run:
        missing.append("synthetic_dry_run")
    if not info.get("run_id"):
        missing.append("run_id")
    if not info.get("experiment_id"):
        missing.append("experiment_id")
    artifact_hashes = info.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or any(
        not SHA256_RE.fullmatch(str(artifact_hashes.get(name) or ""))
        for name in ("pred.pkl", "label.pkl")
    ):
        missing.append("replayable_predictions")
    if not provider_lineage_is_complete(provider):
        missing.append("provider_snapshot")
    return {"eligible": not missing, "missing": missing}


def _run_model_training(
    model: dict[str, Any],
    spec: dict[str, Any],
    *,
    provider: str,
    provider_uri: Path,
    provider_latest_day: str,
    latest_prediction_day: str,
    batch_id: str,
    config_dir: Path,
    experiment_name: str,
    expected_provider_identity: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    base = yaml.safe_load(_resolve_base_config(spec).read_text(encoding="utf-8")) or {}
    candidate_config = build_candidate_config(base, model)
    qlib_init = dict(candidate_config.get("qlib_init") or {})
    qlib_init.update(
        qlib_workflow_init_config(
            provider=provider,
            provider_uri=provider_uri,
        )
    )
    candidate_config["qlib_init"] = qlib_init
    apply_common_periods(
        candidate_config,
        spec,
        provider_latest_day=provider_latest_day,
        latest_prediction_day=latest_prediction_day,
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{model['id']}.yaml"
    result_path = config_dir / f"{model['id']}.training_result.json"
    _atomic_write(config_path, yaml.safe_dump(candidate_config, allow_unicode=True, sort_keys=False))
    result_path.unlink(missing_ok=True)
    command = _training_command(
        config_path=config_path,
        experiment_name=experiment_name,
        provider=provider,
        expected_provider_identity=expected_provider_identity,
        result_output=result_path,
    )
    print(f"[candidate] command model={model['id']} config={config_path}", flush=True)
    result = subprocess.run(command, cwd=str(WORKFLOW_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training command exited with code {result.returncode}")
    try:
        info = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Candidate training result is missing or invalid: {result_path.name}"
        ) from exc
    if not isinstance(info, dict):
        raise RuntimeError("Candidate training result must be a JSON object")
    if info.get("experiment_name") != experiment_name or not info.get("run_id"):
        raise RuntimeError("Training summary does not belong to the current candidate experiment")
    if Path(str(info.get("source_config") or "")).resolve() != config_path.resolve():
        raise RuntimeError("Training summary source config does not match the current candidate")
    _require_provider_identity(info.get("provider_identity"), expected_provider_identity)
    _tag_candidate_run(info, batch_id, str(model["id"]))
    pred, label = _load_artifacts(
        info,
        expected_provider_identity=expected_provider_identity,
    )
    return info, pred, label


def _periods(spec: dict[str, Any], latest_label_day: str) -> tuple[list[str], list[str]]:
    selection = list(spec["selection_period"])
    holdout = [spec["holdout_period"][0], latest_label_day]
    if pd.Timestamp(holdout[0]) > pd.Timestamp(holdout[1]):
        raise ValueError(f"Provider does not contain the configured holdout period: {holdout[0]} > {holdout[1]}")
    return selection, holdout


def _strategy_rows(
    selected: list[dict[str, Any]],
    artifacts: dict[str, tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]],
    spec: dict[str, Any],
    market: pd.DataFrame,
    selection_period: list[str],
    holdout_period: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in selected:
        pred, label = artifacts[str(model["id"])]
        for strategy in spec["strategies"]:
            kwargs = copy.deepcopy(strategy.get("kwargs") or {})
            rows.append(
                {
                    "model_id": model["id"],
                    "strategy_id": strategy["id"],
                    "strategy_label": strategy.get("label") or strategy["id"],
                    "strategy_config": kwargs,
                    "selection": evaluate_period(
                        pred,
                        label,
                        market,
                        selection_period,
                        kwargs,
                        topk=spec["topk"],
                        cost_bps=spec["cost_bps"],
                    ),
                    "holdout": evaluate_period(
                        pred,
                        label,
                        market,
                        holdout_period,
                        kwargs,
                        topk=spec["topk"],
                        cost_bps=spec["cost_bps"],
                    ),
                }
            )
    return rows


def rank_strategies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[float, float, float]:
        metrics = item.get("selection") if isinstance(item.get("selection"), dict) else {}
        return (
            _finite_float(metrics.get("information_ratio"), -math.inf),
            _finite_float(metrics.get("annual_excess_return"), -math.inf),
            _finite_float(metrics.get("max_drawdown"), -math.inf),
        )

    return sorted(rows, key=key, reverse=True)


def holdout_verdict(champion: dict[str, Any] | None, *, pipeline_complete: bool) -> dict[str, Any]:
    if not pipeline_complete:
        return {
            "status": "unavailable_incomplete_pipeline",
            "reason": "At least one of the five required model candidates failed.",
        }
    if not champion:
        return {"status": "unavailable", "reason": "No complete model-strategy candidate"}
    holdout = champion.get("holdout") or {}
    passed = (
        _finite_float(holdout.get("annual_excess_return"), -math.inf) > 0
        and _finite_float(holdout.get("information_ratio"), -math.inf) > 0
        and _finite_float(holdout.get("max_drawdown"), -math.inf) >= -0.20
    )
    return {
        "status": "passed_research_gate" if passed else "rejected_by_holdout",
        "criteria": {
            "annual_excess_return": "> 0",
            "information_ratio": "> 0",
            "max_drawdown": ">= -20%",
        },
        "note": "研究筛选结论，不代表实盘许可，不自动下单。",
    }


def _dry_run_market(dates: pd.DatetimeIndex) -> pd.DataFrame:
    market_return = pd.Series(0.0002 + np.sin(np.arange(len(dates)) / 17.0) * 0.001, index=dates)
    market = pd.DataFrame({"benchmark_return": market_return})
    market["close"] = 100.0 * (1.0 + market["benchmark_return"]).cumprod()
    for days in (20, 60, 120, 200):
        market[f"ma{days}"] = market["close"].rolling(days).mean()
        market[f"trend_{days}"] = (market["close"] > market[f"ma{days}"]).astype(float)
    market["volatility_20"] = market["benchmark_return"].rolling(20).std() * math.sqrt(252)
    market["vol_percentile"] = market["volatility_20"].rolling(120).rank(pct=True)
    return market


def _dry_run_artifacts(models: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    dates = pd.bdate_range("2024-01-02", periods=650)
    instruments = [f"SH600{index:03d}" for index in range(80)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rng = np.random.default_rng(42)
    instrument_alpha = np.tile(np.linspace(-0.002, 0.002, len(instruments)), len(dates))
    market_wave = np.repeat(0.0002 + np.sin(np.arange(len(dates)) / 17.0) * 0.001, len(instruments))
    label_values = instrument_alpha + market_wave + rng.normal(0.0, 0.002, len(index))
    label = pd.DataFrame({"label": label_values}, index=index)
    strengths = [0.55, 0.65, 0.85, 0.75, 0.70]
    artifacts: dict[str, Any] = {}
    for model, strength in zip(models, strengths):
        score = strength * label_values + rng.normal(0.0, 0.0015, len(index))
        artifacts[str(model["id"])] = (pd.DataFrame({"score": score}, index=index), label.copy())
    last_day = str(dates[-1].date())
    return artifacts, _dry_run_market(dates), {
        "provider_latest_day": last_day,
        "latest_prediction_day": last_day,
        "latest_label_day": last_day,
    }


def _report_namespace(output_dir: Path, *, dry_run: bool) -> tuple[Path, str]:
    if dry_run:
        return output_dir / "test_runs" / "candidate_pipeline", "candidate_pipeline_test"
    return output_dir, "candidate_pipeline"


def _progress(
    output_dir: Path,
    batch_id: str,
    stage: str,
    completed: int,
    total: int,
    current: str = "",
    *,
    dry_run: bool = False,
) -> None:
    target_dir, prefix = _report_namespace(output_dir, dry_run=dry_run)
    payload = {
        "batch_id": batch_id,
        "dry_run": dry_run,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": stage,
        "completed": completed,
        "total": total,
        "current": current,
    }
    _atomic_write(
        target_dir / f"{prefix}_progress_latest.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 五个选股模型候选流程报告",
        "",
    ]
    if payload.get("dry_run"):
        lines.extend(
            [
                "> **测试数据，不可用于研究或交易。** 本报告来自合成小样本预演，",
                "> 没有真实 Run ID、预测产物或 MLflow 训练记录。",
                "",
            ]
        )
    lines.extend(
        [
        f"- 状态：`{payload['status']}`",
        f"- 批次：`{payload['batch_id']}`",
        f"- 数据：`{payload['provider']}`，最新可验证标签日 `{payload['latest_label_day']}`",
        f"- 最新预测日：`{payload['latest_prediction_day']}`（可能晚于最新可验证标签日）",
        f"- 训练期：{' ~ '.join(payload['periods']['training'])}",
        f"- 验证期：{' ~ '.join(payload['periods']['validation'])}",
        f"- 模型和策略选择期：{' ~ '.join(payload['periods']['selection'])}",
        f"- 独立留出期：{' ~ '.join(payload['periods']['holdout'])}",
        "- 选择规则：只按 2024-2025 开发期排名；2026 留出期不参与挑选。",
        "- 实验记录：真实候选训练会追加到实验历史；日常 latest_training_result 不会被候选流程替换。",
        "",
        "## 模型比较",
        "",
        "| 模型 | 状态 | 开发期IR | 开发期超额年化 | 开发期RankIC | 留出期IR | 留出期超额年化 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in payload["models"]:
        selection = item.get("selection") or {}
        holdout = item.get("holdout") or {}
        lines.append(
            "| {id} | {status} | {sir:.2f} | {sexcess:.2%} | {rank_ic:.4f} | {hir:.2f} | {hexcess:.2%} |".format(
                id=item.get("id"),
                status=item.get("status"),
                sir=_finite_float(selection.get("information_ratio"), 0.0),
                sexcess=_finite_float(selection.get("annual_excess_return"), 0.0),
                rank_ic=_finite_float(selection.get("rank_ic"), 0.0),
                hir=_finite_float(holdout.get("information_ratio"), 0.0),
                hexcess=_finite_float(holdout.get("annual_excess_return"), 0.0),
            )
        )
        if item.get("error"):
            lines.append(f"\n- `{item.get('id')}` 失败原因：{item['error']}")
    lines.extend(
        [
            "",
            "## 选股模型与交易策略组合",
            "",
            "| 选股模型 | 交易策略 | 开发期IR | 开发期超额年化 | 开发期最大回撤 | 留出期IR | 留出期超额年化 | 留出期最大回撤 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["strategy_evaluations"]:
        selection = row["selection"]
        holdout = row["holdout"]
        lines.append(
            "| {model} | {strategy} | {sir:.2f} | {sexcess:.2%} | {sdd:.2%} | {hir:.2f} | {hexcess:.2%} | {hdd:.2%} |".format(
                model=row["model_id"],
                strategy=row["strategy_label"],
                sir=_finite_float(selection.get("information_ratio"), 0.0),
                sexcess=_finite_float(selection.get("annual_excess_return"), 0.0),
                sdd=_finite_float(selection.get("max_drawdown"), 0.0),
                hir=_finite_float(holdout.get("information_ratio"), 0.0),
                hexcess=_finite_float(holdout.get("annual_excess_return"), 0.0),
                hdd=_finite_float(holdout.get("max_drawdown"), 0.0),
            )
        )
    champion = payload.get("selected_combination")
    lines.extend(["", "## 结论", ""])
    if champion:
        lines.append(f"- 开发期选出的组合：`{champion['model_id']} + {champion['strategy_id']}`。")
    lines.append(f"- 留出期终审：`{payload['holdout_verdict']['status']}`。")
    lines.append("- 本流程只生成研究结论，不连接券商，不读取交易密码，不自动下单。")
    return "\n".join(lines) + "\n"


def write_reports(payload: dict[str, Any], output_dir: Path, timestamp: str) -> dict[str, str]:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    markdown = _report_markdown(payload)
    target_dir, prefix = _report_namespace(
        output_dir, dry_run=bool(payload.get("dry_run"))
    )
    timestamped_json = target_dir / f"{prefix}_{timestamp}.json"
    timestamped_md = target_dir / f"{prefix}_{timestamp}.md"
    latest_json = target_dir / f"{prefix}_latest.json"
    latest_md = target_dir / f"{prefix}_latest.md"
    for path, content in (
        (timestamped_json, json_text),
        (timestamped_md, markdown),
        (latest_json, json_text),
        (latest_md, markdown),
    ):
        _atomic_write(path, content)
    return {
        "json": str(timestamped_json.resolve()),
        "markdown": str(timestamped_md.resolve()),
        "latest_json": str(latest_json.resolve()),
        "latest_markdown": str(latest_md.resolve()),
    }


def run_pipeline(
    *,
    spec_path: Path = DEFAULT_SPEC_PATH,
    provider: str = "canonical",
    output_dir: Path = REPORT_DIR,
    timestamp: str | None = None,
    dry_run: bool = False,
    expected_provider_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not TIMESTAMP_RE.fullmatch(timestamp):
        raise ValueError("timestamp must use YYYYMMDD_HHMMSS")
    output_dir = Path(output_dir)
    spec = load_pipeline_spec(Path(spec_path))
    batch_id = f"candidate-alpha158-{timestamp}"
    candidates: list[dict[str, Any]] = []
    artifacts: dict[str, tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]] = {}

    if dry_run:
        artifacts, market, provider_state = _dry_run_artifacts(spec["models"])
        provider_uri = Path("dry-run-provider")
        locked_provider_identity = provider_identity(provider_uri)
    else:
        _preflight_dependencies(spec["models"])
        provider_uri = resolve_pipeline_provider(
            provider,
            expected_provider_identity=expected_provider_identity,
        )
        if not provider_uri.exists():
            raise FileNotFoundError(f"Provider does not exist: {provider_uri}")
        provider_latest_day, latest_prediction_day, latest_label_day = provider_days(
            provider_uri
        )
        locked_provider_identity = provider_identity(provider_uri)
        if expected_provider_identity is not None:
            _require_provider_identity(
                locked_provider_identity,
                dict(expected_provider_identity),
            )
        provider_state = {
            "provider_latest_day": provider_latest_day,
            "latest_prediction_day": latest_prediction_day,
            "latest_label_day": latest_label_day,
        }
        market = load_market_data(
            provider_uri, spec["selection_period"][0], provider_latest_day
        )
    selection_period, holdout_period = _periods(
        spec, provider_state["latest_label_day"]
    )
    experiment_name = f"candidate_models_alpha158_{timestamp}"
    config_dir = output_dir / "runtime_configs" / batch_id
    _progress(
        output_dir,
        batch_id,
        "models",
        0,
        len(spec["models"]),
        dry_run=dry_run,
    )

    for index, model in enumerate(spec["models"], start=1):
        model_id = str(model["id"])
        print(
            f"\n[{index}/{len(spec['models'])}] 训练候选模型 {model_id}",
            flush=True,
        )
        _progress(
            output_dir,
            batch_id,
            "models",
            index - 1,
            len(spec["models"]),
            model_id,
            dry_run=dry_run,
        )
        try:
            if dry_run:
                info = {
                    "run_id": None,
                    "experiment_id": None,
                    "experiment_name": experiment_name,
                    "metrics": {},
                }
                pred, label = artifacts[model_id]
            else:
                info, pred, label = _run_model_training(
                    model,
                    spec,
                    provider=provider,
                    provider_uri=provider_uri,
                    provider_latest_day=provider_state["provider_latest_day"],
                    latest_prediction_day=provider_state["latest_prediction_day"],
                    batch_id=batch_id,
                    config_dir=config_dir,
                    experiment_name=experiment_name,
                    expected_provider_identity=locked_provider_identity,
                )
                artifacts[model_id] = (pred, label)
            selection = evaluate_period(
                pred,
                label,
                market,
                selection_period,
                _baseline_strategy_kwargs(),
                topk=spec["topk"],
                cost_bps=spec["cost_bps"],
            )
            holdout = evaluate_period(
                pred,
                label,
                market,
                holdout_period,
                _baseline_strategy_kwargs(),
                topk=spec["topk"],
                cost_bps=spec["cost_bps"],
            )
            if not dry_run:
                _log_candidate_metrics(info, selection, holdout)
            candidates.append(
                {
                    "id": model_id,
                    "label": model.get("label") or model_id,
                    "class": model["class"],
                    "status": "completed",
                    "run_id": info.get("run_id"),
                    "experiment_id": info.get("experiment_id"),
                    "experiment_name": info.get("experiment_name"),
                    "artifact_sha256": info.get("artifact_sha256") or {},
                    "backtest_eligibility": _backtest_eligibility(
                        info,
                        locked_provider_identity,
                        dry_run=dry_run,
                    ),
                    "qlib_metrics": info.get("metrics") or {},
                    "selection": selection,
                    "holdout": holdout,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "id": model_id,
                    "label": model.get("label") or model_id,
                    "class": model["class"],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[candidate:error] {model_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        _progress(
            output_dir,
            batch_id,
            "models",
            index,
            len(spec["models"]),
            model_id,
            dry_run=dry_run,
        )

    selected_models = rank_models(candidates, spec["top_models"])
    strategy_rows = (
        _strategy_rows(
            selected_models,
            artifacts,
            spec,
            market,
            selection_period,
            holdout_period,
        )
        if len(selected_models) == spec["top_models"]
        else []
    )
    ranked_strategies = rank_strategies(strategy_rows)
    status = pipeline_status(candidates)
    champion = ranked_strategies[0] if ranked_strategies else None
    selected_combination = champion if status == "completed" else None
    verdict = holdout_verdict(champion, pipeline_complete=status == "completed")
    if not dry_run:
        _tag_pipeline_outcome(
            candidates, selected_models, selected_combination, verdict
        )
    payload = {
        "status": status,
        "dry_run": dry_run,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "experiment_name": experiment_name,
        "provider": provider if not dry_run else "dry_run",
        "provider_uri": str(provider_uri),
        "provider_identity": locked_provider_identity,
        "provider_latest_day": provider_state["provider_latest_day"],
        "latest_prediction_day": provider_state["latest_prediction_day"],
        "latest_label_day": provider_state["latest_label_day"],
        "periods": {
            "training": list(spec["training_period"]),
            "validation": list(spec["validation_period"]),
            "selection": selection_period,
            "holdout": holdout_period,
        },
        "selection_basis": "selection_period_only",
        "ranking_order": ["information_ratio", "annual_excess_return", "rank_ic"],
        "metric_basis": {
            "portfolio_return": "absolute_net_return_after_configured_cost",
            "benchmark": "SH000300",
            "annual_excess_return": (
                "portfolio_annual_return_minus_benchmark_annual_return"
            ),
            "directly_comparable": True,
        },
        "models": candidates,
        "selected_models": [item["id"] for item in selected_models],
        "strategy_evaluations": strategy_rows,
        "selected_combination": selected_combination,
        "partial_development_leader": champion if status != "completed" else None,
        "holdout_verdict": verdict,
        "operational_latest_training_changed": False,
        "experiment_notebook_appended": not dry_run,
        "experiment_notebook_policy": (
            "candidate MLflow runs are appended to the experiment notebook; "
            "candidate runs never publish the operational latest training pointer"
        ),
        "safety": {
            "broker_connected": False,
            "credentials_read": False,
            "auto_order": False,
        },
    }
    replay_contract = {
        "contract_version": 2,
        "replay_engine": "dsa_optimized_strategy",
        "replay_capability": "custom_only",
        "exact_strategy_ids": [],
        "source_config_sha256": hashlib.sha256(
            Path(spec_path).read_bytes()
        ).hexdigest(),
        "periods": dict(payload["periods"]),
        "original_parameters": {
            "start_date": selection_period[0],
            "end_date": selection_period[1],
            "topk": int(spec["topk"]),
            "cost_bps": float(spec["cost_bps"]),
        },
        "strategies": [
            {
                "id": str(item["id"]),
                "label": str(item.get("label") or item["id"]),
                "config": dict(item.get("kwargs") or {}),
            }
            for item in spec["strategies"]
        ],
    }
    replay_contract["config_hash"] = hashlib.sha256(
        json.dumps(
            replay_contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload["backtest_contract"] = replay_contract
    payload["artifacts"] = write_reports(payload, output_dir, timestamp)
    _progress(
        output_dir,
        batch_id,
        "completed" if status == "completed" else "failed",
        5,
        5,
        dry_run=dry_run,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the five-model Qlib candidate pipeline")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--provider", choices=("canonical", "cleaned"), default="canonical")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provider-identity-json",
        default=None,
        help="Trusted immutable provider identity injected by the DSA job API",
    )
    args = parser.parse_args()
    expected_provider_identity = None
    if args.provider_identity_json is not None:
        try:
            parsed_identity = json.loads(args.provider_identity_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--provider-identity-json must be valid JSON: {exc}")
        if not isinstance(parsed_identity, dict):
            parser.error("--provider-identity-json must contain a JSON object")
        expected_provider_identity = parsed_identity
    payload = run_pipeline(
        spec_path=args.spec,
        provider=args.provider,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
        dry_run=args.dry_run,
        expected_provider_identity=expected_provider_identity,
    )
    print(f"\n[完成] 五个选股模型候选流程状态: {payload['status']}")
    print(f"[报告] {payload['artifacts']['json']}")
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
