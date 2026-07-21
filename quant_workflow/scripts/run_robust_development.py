#!/usr/bin/env python3
"""Plan and execute rolling out-of-sample robust strategy development."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import cloudpickle
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robust_strategy_core import RollingSplit, make_rolling_split  # noqa: E402
from _paths import mlruns_root  # noqa: E402
from robust_evaluation import (  # noqa: E402
    aggregate_prediction_frames,
    aggregate_training_artifacts,
    cross_model_diversification_report,
    evaluate_strategy_candidates,
    select_development_models,
)
from robust_market_data import (  # noqa: E402
    build_market_regimes,
    load_benchmark_frame,
    load_market_panel,
)
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from prepare_robust_provider import runtime_provider_identity  # noqa: E402
from run_robust_strategy import (  # noqa: E402
    DEFAULT_CONFIG,
    file_sha256,
    git_worktree_state,
    load_config,
)


TREE_MODELS = {"lightgbm", "xgboost", "catboost"}
NEURAL_MODELS = {"mps_dnn"}
FROZEN_CONTEXT_KEYS = (
    "status",
    "as_of_date",
    "config_hash",
    "git_commit",
    "data_manifest_hash",
    "membership_source",
    "membership_source_metadata",
    "membership_snapshot_hash",
    "reference_source_metadata",
    "reference_snapshot_hash",
    "provider_uri",
    "provider_key",
    "provider_tree_sha256",
    "runtime_environment_hash",
    "launch_runtime_evidence_hash",
    "selected_models",
    "selected_model_configs",
    "selected_strategy",
    "selection_hash",
    "development_run_id",
    "development_run_dir",
    "development_context_hash",
    "development_artifacts_hash",
)
SEED_ARGUMENTS = {
    "lightgbm": "seed",
    "xgboost": "seed",
    "catboost": "random_seed",
    "mps_dnn": "seed",
}
MODEL_DISTRIBUTIONS = {
    "linear_ols": (),
    "linear_ridge": (),
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
    "catboost": ("catboost",),
    "mps_dnn": ("torch",),
}
BASE_TRAINING_DISTRIBUTIONS = (
    "cloudpickle",
    "mlflow",
    "numpy",
    "pandas",
    "pyarrow",
    "pyqlib",
    "pyyaml",
    "scikit-learn",
)


def collect_runtime_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return path-free package and device provenance for a formal run."""
    model_ids = {str(item["id"]) for item in config["models"]}
    distributions = set(BASE_TRAINING_DISTRIBUTIONS)
    for model_id in model_ids:
        distributions.update(MODEL_DISTRIBUTIONS.get(model_id, ()))
    packages: dict[str, str] = {}
    for distribution in sorted(distributions):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "missing"
    requests_mps = "mps_dnn" in model_ids
    mps_built = False
    mps_available = False
    if requests_mps:
        try:
            import torch

            mps_built = bool(torch.backends.mps.is_built())
            mps_available = bool(torch.backends.mps.is_available())
        except ImportError:
            pass
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "requested_device": "mps" if requests_mps else "cpu",
        "mps_built": mps_built,
        "mps_available": mps_available,
        "packages": packages,
    }
    serialized = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
    result["environment_hash"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return result


class TrainingJob(NamedTuple):
    quarter: str
    window_days: int
    model_id: str
    seed: int | None

    @property
    def key(self) -> str:
        seed = "fixed" if self.seed is None else str(self.seed)
        return f"{self.quarter}__w{self.window_days}__{self.model_id}__s{seed}"


class NeuralRepeatabilityError(RuntimeError):
    """Raised when identical neural training inputs produce different scores."""

    def __init__(self, message: str, *, max_abs_diff: float | None = None):
        super().__init__(message)
        self.max_abs_diff = max_abs_diff


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            cloudpickle.dump(value, stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_training_jobs(config: Mapping[str, Any]) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    quarters = list(config["rolling_validation"]["test_quarters"])
    if any(str(quarter).startswith("2026") for quarter in quarters):
        raise ValueError(
            "2026 confirmation periods must not enter development job selection"
        )
    for quarter in quarters:
        for window in config["training"]["windows_trading_days"]:
            for model in config["models"]:
                model_id = str(model["id"])
                if model_id in TREE_MODELS:
                    seeds = config["training"]["tree_seeds"]
                elif model_id in NEURAL_MODELS:
                    seeds = config["training"].get(
                        "neural_seeds", config["training"]["tree_seeds"]
                    )
                else:
                    seeds = [None]
                for seed in seeds:
                    jobs.append(
                        TrainingJob(
                            str(quarter),
                            int(window),
                            model_id,
                            None if seed is None else int(seed),
                        )
                    )
    return jobs


def generate_tuning_trials(
    model_id: str,
    base_kwargs: Mapping[str, Any],
    *,
    maximum: int = 20,
) -> list[dict[str, Any]]:
    if maximum <= 0:
        return []
    limit = min(int(maximum), 20)
    base = copy.deepcopy(dict(base_kwargs))
    variants: list[dict[str, Any]] = []
    if model_id == "linear_ols":
        return []
    if model_id == "linear_ridge":
        for alpha in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            trial = copy.deepcopy(base)
            trial["alpha"] = alpha
            variants.append(trial)
    elif model_id == "lightgbm":
        for learning_rate, max_depth in itertools.product(
            (0.02, 0.04, 0.06, 0.08, 0.10), (4, 6, 8, 10)
        ):
            trial = copy.deepcopy(base)
            trial.update(
                {
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "num_leaves": min(
                        int(base.get("num_leaves", 210)), 2**max_depth - 1
                    ),
                }
            )
            variants.append(trial)
    elif model_id == "xgboost":
        for eta, max_depth in itertools.product(
            (0.02, 0.04, 0.06, 0.08, 0.10), (4, 6, 8, 10)
        ):
            trial = copy.deepcopy(base)
            trial.update({"eta": eta, "max_depth": max_depth})
            variants.append(trial)
    elif model_id == "catboost":
        for learning_rate, depth in itertools.product(
            (0.02, 0.04, 0.06, 0.08, 0.10), (4, 6, 8, 10)
        ):
            trial = copy.deepcopy(base)
            trial.update({"learning_rate": learning_rate, "depth": depth})
            variants.append(trial)
    elif model_id == "mps_dnn":
        for learning_rate, hidden_layers in itertools.product(
            (0.0003, 0.001, 0.003, 0.01),
            ((128,), (128, 64), (256, 128), (256, 128, 64)),
        ):
            trial = copy.deepcopy(base)
            trial.update(
                {
                    "lr": learning_rate,
                    "hidden_layers": list(hidden_layers),
                }
            )
            variants.append(trial)
    else:
        raise ValueError(f"unknown model for bounded tuning: {model_id}")
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trial in variants:
        key = json.dumps(trial, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(trial)
        if len(unique) >= limit:
            break
    return unique


def build_tuning_screen_jobs(
    config: Mapping[str, Any], model_id: str
) -> list[TrainingJob]:
    if model_id not in {str(item["id"]) for item in config["models"]}:
        raise ValueError(f"unknown model: {model_id}")
    if model_id in TREE_MODELS:
        seed = int(config["training"]["tree_seeds"][0])
    elif model_id in NEURAL_MODELS:
        seed = int(
            (
                config["training"].get("neural_seeds")
                or config["training"]["tree_seeds"]
            )[0]
        )
    else:
        seed = None
    return [
        TrainingJob(str(quarter), 756, model_id, seed)
        for quarter in config["rolling_validation"]["test_quarters"]
    ]


def _config_with_model_kwargs(
    config: Mapping[str, Any], model_id: str, kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    found = False
    for model in result["models"]:
        if str(model["id"]) == model_id:
            model["kwargs"] = copy.deepcopy(dict(kwargs))
            found = True
            break
    if not found:
        raise ValueError(f"unknown model: {model_id}")
    return result


def _artifact_records(
    jobs: Sequence[TrainingJob], progress: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "model_id": job.model_id,
            "window_days": job.window_days,
            "quarter": job.quarter,
            "seed": job.seed,
            "prediction_path": progress[job.key]["prediction_path"],
        }
        for job in jobs
    ]


def run_bounded_tuning(
    config: Mapping[str, Any],
    model_ids: Sequence[str],
    *,
    provider_uri: Path,
    calendar: Sequence[Any],
    run_dir: Path,
    training_runner: Any = None,
    aggregator: Any = None,
    context_hash: str | None = None,
) -> dict[str, Any]:
    training_runner = training_runner or run_training_jobs
    aggregator = aggregator or aggregate_training_artifacts
    maximum = int(config["training"]["bounded_tuning_trials_per_failed_model_max"])
    reports: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    model_artifacts: dict[str, list[dict[str, Any]]] = {}
    tuning_summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        base_model = next(
            (item for item in config["models"] if str(item["id"]) == model_id), None
        )
        if base_model is None:
            continue
        trials = generate_tuning_trials(
            model_id, base_model.get("kwargs") or {}, maximum=maximum
        )
        screened: list[dict[str, Any]] = []
        for index, kwargs in enumerate(trials, start=1):
            trial_config = _config_with_model_kwargs(config, model_id, kwargs)
            jobs = build_tuning_screen_jobs(trial_config, model_id)
            trial_dir = Path(run_dir) / model_id / f"trial_{index:02d}"
            progress = training_runner(
                trial_config,
                jobs,
                provider_uri=provider_uri,
                calendar=calendar,
                run_dir=trial_dir,
                mlflow_enabled=True,
                resume=True,
                context_hash=payload_sha256(
                    {
                        "parent_context_hash": context_hash,
                        "trial": index,
                        "model_id": model_id,
                        "config": trial_config,
                    }
                ),
            )
            if any(
                progress.get(job.key, {}).get("status") != "completed" for job in jobs
            ):
                screened.append({"trial": index, "kwargs": kwargs, "status": "failed"})
                continue
            aggregate = aggregator(
                _artifact_records(jobs, progress),
                topk=int(config["portfolio"]["topk"]),
                thresholds=config["acceptance"],
            )
            report = aggregate["model_reports"][0]
            score = (
                float(report.get("positive_rank_ic_interval_ratio", 0.0)),
                float(report.get("rank_ic_mean", -math.inf)),
            )
            screened.append(
                {
                    "trial": index,
                    "kwargs": kwargs,
                    "status": "completed",
                    "score": score,
                    "model_report": report,
                }
            )
        completed = [item for item in screened if item["status"] == "completed"]
        if not completed:
            tuning_summaries.append(
                {
                    "model_id": model_id,
                    "screened_trials": len(screened),
                    "status": "all_trials_failed",
                }
            )
            continue
        finalist_count = int(
            config["training"].get("bounded_tuning_full_matrix_finalists", 1)
        )
        if not 1 <= finalist_count <= 3:
            raise ValueError(
                "training.bounded_tuning_full_matrix_finalists must be between 1 and 3"
            )
        finalists = sorted(
            completed,
            key=lambda item: (item["score"], -int(item["trial"])),
            reverse=True,
        )[:finalist_count]
        full_results: list[dict[str, Any]] = []
        for finalist in finalists:
            finalist_config = _config_with_model_kwargs(
                config, model_id, finalist["kwargs"]
            )
            full_jobs = [
                job
                for job in build_training_jobs(finalist_config)
                if job.model_id == model_id
            ]
            full_dir_name = (
                "best_full_matrix"
                if finalist_count == 1
                else f"finalist_{int(finalist['trial']):02d}_full_matrix"
            )
            full_dir = Path(run_dir) / model_id / full_dir_name
            progress = training_runner(
                finalist_config,
                full_jobs,
                provider_uri=provider_uri,
                calendar=calendar,
                run_dir=full_dir,
                mlflow_enabled=True,
                resume=True,
                context_hash=payload_sha256(
                    {
                        "parent_context_hash": context_hash,
                        "trial": finalist["trial"],
                        "model_id": model_id,
                        "config": finalist_config,
                        "phase": "best_full_matrix",
                    }
                ),
            )
            if any(
                progress.get(job.key, {}).get("status") != "completed"
                for job in full_jobs
            ):
                full_results.append(
                    {"trial": finalist["trial"], "status": "failed"}
                )
                continue
            aggregate = aggregator(
                _artifact_records(full_jobs, progress),
                topk=int(config["portfolio"]["topk"]),
                thresholds=config["acceptance"],
            )
            full_results.append(
                {
                    "trial": finalist["trial"],
                    "kwargs": finalist["kwargs"],
                    "screen_score": finalist["score"],
                    "status": "completed",
                    "report": copy.deepcopy(aggregate["model_reports"][0]),
                    "predictions": aggregate["model_predictions"][model_id],
                    "artifacts": _development_artifacts(
                        full_jobs, progress, [model_id]
                    ),
                }
            )
        completed_full = [
            item for item in full_results if item["status"] == "completed"
        ]
        if not completed_full:
            tuning_summaries.append(
                {
                    "model_id": model_id,
                    "screened_trials": len(screened),
                    "full_matrix_finalists": [item["trial"] for item in finalists],
                    "status": "best_full_matrix_failed",
                }
            )
            continue

        def full_result_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            report = item["report"]
            gates = report.get("gates") or {}
            try:
                development_score = float(report.get("development_score", -math.inf))
            except (TypeError, ValueError):
                development_score = -math.inf
            if not math.isfinite(development_score):
                development_score = -math.inf
            return (
                report.get("eligible") is True,
                sum(value is True for value in gates.values()),
                development_score,
                item["screen_score"],
                -int(item["trial"]),
            )

        best = max(completed_full, key=full_result_key)
        report = best["report"]
        report["tuning"] = {
            "screened_trials": len(screened),
            "selected_trial": best["trial"],
            "selected_kwargs": best["kwargs"],
            "full_matrix_finalists": [item["trial"] for item in finalists],
            "selection_basis": (
                "2023Q1-2025Q4_only_756d_screen_then_"
                f"top{len(finalists)}_full_matrix"
            ),
        }
        reports.append(report)
        predictions[model_id] = best["predictions"]
        model_artifacts[model_id] = best["artifacts"]
        tuning_summaries.append(
            {"model_id": model_id, "status": "completed", **report["tuning"]}
        )
    return {
        "model_reports": reports,
        "model_predictions": predictions,
        "model_artifacts": model_artifacts,
        "summaries": tuning_summaries,
    }


def model_config_for_job(config: Mapping[str, Any], job: TrainingJob) -> dict[str, Any]:
    try:
        model = next(item for item in config["models"] if item["id"] == job.model_id)
    except StopIteration as exc:
        raise ValueError(f"unknown model: {job.model_id}") from exc
    result = copy.deepcopy(model)
    result.setdefault("kwargs", {})
    if job.model_id in SEED_ARGUMENTS:
        if job.seed is None:
            raise ValueError(f"seeded model {job.model_id} requires a seed")
        result["kwargs"][SEED_ARGUMENTS[job.model_id]] = int(job.seed)
        if job.model_id == "lightgbm":
            result["kwargs"]["bagging_seed"] = int(job.seed)
            result["kwargs"]["feature_fraction_seed"] = int(job.seed)
    return result


def make_job_split(
    calendar: Sequence[Any],
    quarter: str,
    *,
    window_days: int,
    validation_days: int,
    label_horizon_days: int,
) -> RollingSplit:
    return make_rolling_split(
        calendar,
        quarter,
        train_days=window_days,
        validation_days=validation_days,
        label_horizon_days=label_horizon_days,
    )


def _date(value: pd.Timestamp) -> str:
    return str(pd.Timestamp(value).date())


def build_dataset_config(split: RollingSplit) -> dict[str, Any]:
    return {
        "class": "BoardAwareDataset",
        "module_path": "robust_dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": _date(split.train_start),
                    "end_time": _date(split.test_end),
                    "fit_start_time": _date(split.train_start),
                    "fit_end_time": _date(split.train_end),
                    "instruments": "csi800",
                    "label": [["Ref($open, -6) / Ref($open, -1) - 1"], ["LABEL0"]],
                    "infer_processors": [
                        {
                            "class": "RobustZScoreNorm",
                            "kwargs": {"fields_group": "feature", "clip_outlier": True},
                        },
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [{"class": "DropnaLabel"}],
                },
            },
            "segments": {
                "train": [_date(split.train_start), _date(split.train_end)],
                "valid": [_date(split.validation_start), _date(split.validation_end)],
                "test": [_date(split.test_start), _date(split.test_end)],
            },
            "normalize_within_board": True,
            "add_board_features": True,
            "add_market_state_features": True,
        },
    }


def percentile_rank_ensemble(predictions: Mapping[str, pd.Series]) -> pd.Series:
    if not predictions:
        raise ValueError("at least one prediction series is required")
    ranked: list[pd.Series] = []
    expected_index: pd.Index | None = None
    for name, series in predictions.items():
        if (
            not isinstance(series.index, pd.MultiIndex)
            or "datetime" not in series.index.names
        ):
            raise ValueError(
                f"prediction {name} must use a datetime/instrument MultiIndex"
            )
        numeric = pd.to_numeric(series, errors="coerce")
        rank = numeric.groupby(level="datetime").rank(method="average", pct=True)
        if expected_index is None:
            expected_index = rank.index
        elif not rank.index.equals(expected_index):
            raise ValueError("prediction indexes must match exactly")
        ranked.append(rank.rename(str(name)))
    return pd.concat(ranked, axis=1).mean(axis=1).rename("score")


def fit_predict_job(model: Any, dataset: Any, artifact_path: Path) -> dict[str, Any]:
    model.fit(dataset)
    prediction = model.predict(dataset, segment="test")
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("prediction must contain exactly one score column")
        prediction = prediction.iloc[:, 0]
    labels = dataset.prepare("test", col_set="label", data_key="learn")
    if isinstance(labels, pd.Series):
        label = labels
    elif isinstance(labels, pd.DataFrame) and labels.shape[1] == 1:
        label = labels.iloc[:, 0]
    else:
        raise ValueError("label artifact must contain exactly one column")
    for value, name in ((prediction, "prediction"), (label, "label")):
        if not isinstance(value.index, pd.MultiIndex) or not {
            "datetime",
            "instrument",
        }.issubset(value.index.names):
            raise ValueError(f"{name} index must contain datetime and instrument")
        if not value.index.is_unique:
            raise ValueError(f"{name} index must be unique")
    unexpected_labels = label.index.difference(prediction.index)
    if not unexpected_labels.empty:
        raise ValueError("label index contains rows outside prediction index")
    saved_frame = pd.to_numeric(prediction, errors="coerce").rename("score").to_frame()
    saved_frame = saved_frame.join(
        pd.to_numeric(label, errors="coerce").rename("label"), how="left"
    )
    saved_frame = saved_frame.replace([float("inf"), float("-inf")], pd.NA)
    saved_frame = saved_frame.dropna(subset=["score"])
    if saved_frame.empty:
        raise ValueError("prediction contains no valid scores")
    evaluable = saved_frame.dropna(subset=["label"])
    daily_rank_ic = evaluable.groupby(level="datetime", sort=True).apply(
        lambda group: group["score"].corr(group["label"], method="spearman")
    )
    clean_rank_ic = (
        pd.to_numeric(daily_rank_ic, errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )
    saved = saved_frame.reset_index().loc[
        :, ["datetime", "instrument", "score", "label"]
    ]
    _atomic_write_parquet(saved, Path(artifact_path))
    return {
        "rows": int(len(saved)),
        "evaluable_rows": int(len(evaluable)),
        "dates": int(saved["datetime"].nunique()),
        "rank_ic_mean": (
            float(clean_rank_ic.mean()) if not clean_rank_ic.empty else float("nan")
        ),
        "rank_ic_positive_ratio": (
            float((clean_rank_ic > 0).mean())
            if not clean_rank_ic.empty
            else float("nan")
        ),
        "prediction_path": str(Path(artifact_path).resolve()),
    }


def compare_neural_prediction_artifacts(
    primary_path: Path,
    repeat_path: Path,
    *,
    score_max_abs_diff: float,
) -> dict[str, Any]:
    """Compare ordered neural scores and return durable repeatability evidence."""
    if not math.isfinite(score_max_abs_diff) or score_max_abs_diff < 0.0:
        raise ValueError("score_max_abs_diff must be finite and non-negative")
    frames: list[pd.DataFrame] = []
    for path in (Path(primary_path), Path(repeat_path)):
        frame = pd.read_parquet(path, columns=["datetime", "instrument", "score"])
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["instrument"] = frame["instrument"].astype(str)
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        if (
            frame.empty
            or frame[["datetime", "instrument", "score"]].isna().any().any()
            or frame.duplicated(["datetime", "instrument"]).any()
        ):
            raise NeuralRepeatabilityError("neural repeatability artifact is invalid")
        frames.append(
            frame.sort_values(["datetime", "instrument"]).reset_index(drop=True)
        )
    primary, repeated = frames
    if not primary[["datetime", "instrument"]].equals(
        repeated[["datetime", "instrument"]]
    ):
        raise NeuralRepeatabilityError("neural repeatability prediction keys differ")
    primary_scores = np.ascontiguousarray(primary["score"].to_numpy(dtype=np.float64))
    repeat_scores = np.ascontiguousarray(repeated["score"].to_numpy(dtype=np.float64))
    if not np.isfinite(primary_scores).all() or not np.isfinite(repeat_scores).all():
        raise NeuralRepeatabilityError("neural repeatability scores are non-finite")
    max_abs_diff = float(np.max(np.abs(primary_scores - repeat_scores)))
    if max_abs_diff > score_max_abs_diff:
        raise NeuralRepeatabilityError(
            "neural repeatability score difference exceeds tolerance",
            max_abs_diff=max_abs_diff,
        )
    return {
        "neural_repeatability_passed": True,
        "neural_repeatability_max_abs_diff": max_abs_diff,
        "neural_repeatability_score_count": int(len(primary_scores)),
        "neural_repeatability_primary_score_sha256": hashlib.sha256(
            primary_scores.tobytes()
        ).hexdigest(),
        "neural_repeatability_repeat_score_sha256": hashlib.sha256(
            repeat_scores.tobytes()
        ).hexdigest(),
    }


def _split_record(split: RollingSplit) -> dict[str, str]:
    return {
        "train_start": _date(split.train_start),
        "train_end": _date(split.train_end),
        "validation_start": _date(split.validation_start),
        "validation_end": _date(split.validation_end),
        "test_start": _date(split.test_start),
        "test_end": _date(split.test_end),
    }


def completed_job_artifact_is_valid(
    job: TrainingJob,
    state: Mapping[str, Any],
    *,
    run_dir: Path | None = None,
    expected_split: Mapping[str, str] | None = None,
    context_hash: str | None = None,
) -> bool:
    if state.get("status") != "completed":
        return False
    if (
        state.get("model_id") != job.model_id
        or state.get("window_days") != job.window_days
        or state.get("quarter") != job.quarter
        or state.get("seed") != job.seed
    ):
        return False
    if job.model_id in NEURAL_MODELS:
        try:
            repeatability_diff = float(state["neural_repeatability_max_abs_diff"])
            repeatability_count = int(state["neural_repeatability_score_count"])
        except (KeyError, TypeError, ValueError):
            return False
        repeatability_hashes = (
            state.get("neural_repeatability_primary_score_sha256"),
            state.get("neural_repeatability_repeat_score_sha256"),
        )
        if (
            state.get("neural_repeatability_passed") is not True
            or not math.isfinite(repeatability_diff)
            or repeatability_diff < 0.0
            or repeatability_diff > 1e-7
            or repeatability_count <= 0
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in repeatability_hashes
            )
        ):
            return False
    split = state.get("split")
    split_keys = {
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    }
    if not isinstance(split, Mapping) or set(split) != split_keys:
        return False
    if expected_split is not None and dict(split) != dict(expected_split):
        return False
    if context_hash is not None and state.get("context_hash") != context_hash:
        return False
    try:
        if any(pd.isna(pd.Timestamp(split[key])) for key in split_keys):
            return False
    except (TypeError, ValueError):
        return False
    model_path = Path(str(state.get("model_path") or ""))
    prediction_path = Path(str(state.get("prediction_path") or ""))
    if not model_path.is_absolute() or not prediction_path.is_absolute():
        return False
    if model_path.is_symlink() or prediction_path.is_symlink():
        return False
    if run_dir is not None:
        expected_model = Path(run_dir).absolute() / "models" / f"{job.key}.pkl"
        expected_prediction = (
            Path(run_dir).absolute() / "predictions" / f"{job.key}.parquet"
        )
        if (
            model_path.absolute() != expected_model
            or prediction_path.absolute() != expected_prediction
        ):
            return False
    if not model_path.is_file() or not prediction_path.is_file():
        return False
    if not state.get("model_sha256") or not state.get("prediction_sha256"):
        return False
    try:
        if file_sha256(model_path) != state["model_sha256"]:
            return False
        if file_sha256(prediction_path) != state["prediction_sha256"]:
            return False
        frame = pd.read_parquet(prediction_path)
    except (OSError, ValueError, KeyError):
        return False
    if list(frame.columns) != ["datetime", "instrument", "score", "label"]:
        return False
    try:
        recorded_rows = int(state.get("rows", -1))
    except (TypeError, ValueError):
        return False
    if len(frame) <= 0 or recorded_rows != len(frame):
        return False
    prediction_dates = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    if prediction_dates.isna().any():
        return False
    if expected_split is not None:
        test_start = pd.Timestamp(expected_split["test_start"]).normalize()
        test_end = pd.Timestamp(expected_split["test_end"]).normalize()
        if (prediction_dates < test_start).any() or (prediction_dates > test_end).any():
            return False
    if (
        frame["instrument"].isna().any()
        or frame["instrument"].astype(str).str.len().eq(0).any()
    ):
        return False
    if pd.to_numeric(frame["score"], errors="coerce").isna().any():
        return False
    if pd.to_numeric(frame["label"], errors="coerce").isna().any():
        return False
    return True


def pending_jobs(
    jobs: Sequence[TrainingJob],
    progress: Mapping[str, Mapping[str, Any]],
    *,
    run_dir: Path | None = None,
    expected_splits: Mapping[str, Mapping[str, str]] | None = None,
    context_hash: str | None = None,
) -> list[TrainingJob]:
    return [
        job
        for job in jobs
        if not completed_job_artifact_is_valid(
            job,
            progress.get(job.key) or {},
            run_dir=run_dir,
            expected_split=(expected_splits or {}).get(job.key),
            context_hash=context_hash,
        )
    ]


def _default_dataset_factory(
    provider_uri: Path,
    dataset_config: dict[str, Any],
    split: RollingSplit,
    group_jobs: Sequence[TrainingJob],
):
    from qlib.utils import init_instance_by_config

    return init_instance_by_config(dataset_config)


def _default_model_factory(model_config: dict[str, Any], job: TrainingJob):
    from qlib.utils import init_instance_by_config

    return init_instance_by_config(
        {
            "class": model_config["class"],
            "module_path": model_config["module_path"],
            "kwargs": model_config.get("kwargs") or {},
        }
    )


def _load_progress(path: Path, *, resume: bool) -> dict[str, Any]:
    if resume and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict) and isinstance(payload.get("jobs"), dict):
            return payload
    return {"jobs": {}}


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(path, payload)


@contextmanager
def _mlflow_run(
    job: TrainingJob,
    provider_uri: Path,
    *,
    enabled: bool,
    phase: str,
    experiment_name: str,
):
    if not enabled:
        yield None
        return
    from qlib.workflow import R

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    with R.start(
        experiment_name=experiment_name,
        recorder_name=f"{job.key}-{uuid.uuid4().hex[:8]}",
        uri=f"file://{mlruns_root()}",
    ):
        yield R.get_recorder()


def _log_mlflow_result(
    active_run: Any,
    *,
    job: TrainingJob,
    model_config: Mapping[str, Any],
    split: RollingSplit,
    result: Mapping[str, Any],
    prediction_path: Path,
    model_path: Path,
    phase: str,
    runtime_environment: Mapping[str, Any] | None = None,
) -> str | None:
    if active_run is None:
        return None
    params = {
        "model_id": job.model_id,
        "quarter": job.quarter,
        "window_days": job.window_days,
        "seed": "fixed" if job.seed is None else job.seed,
        "train_start": _date(split.train_start),
        "train_end": _date(split.train_end),
        "validation_start": _date(split.validation_start),
        "validation_end": _date(split.validation_end),
        "test_start": _date(split.test_start),
        "test_end": _date(split.test_end),
        "label": "T+1_open_to_T+6_open",
    }
    for key, value in (model_config.get("kwargs") or {}).items():
        params[f"model.{key}"] = value
    for key in (
        "neural_repeatability_primary_score_sha256",
        "neural_repeatability_repeat_score_sha256",
    ):
        if result.get(key):
            params[key] = result[key]
    if runtime_environment:
        for key in (
            "environment_hash",
            "python_version",
            "platform",
            "machine",
            "requested_device",
            "mps_built",
            "mps_available",
        ):
            if key in runtime_environment:
                params[f"runtime.{key}"] = runtime_environment[key]
        for name, version in (runtime_environment.get("packages") or {}).items():
            params[f"runtime.package.{name}"] = version
    active_run.log_params(**{key: str(value) for key, value in params.items()})
    tags = {
        "mlflow.runName": job.key,
        "robust_strategy.phase": phase,
        "robust_strategy.quarter": job.quarter,
        "robust_strategy.window_days": str(job.window_days),
        "robust_strategy.model_id": job.model_id,
        "robust_strategy.seed": "fixed" if job.seed is None else str(job.seed),
        "robust_strategy.selection_period": (
            "2023Q1-2025Q4_only"
            if phase == "development"
            else "frozen_before_2026_confirmation"
        ),
    }
    if runtime_environment and runtime_environment.get("environment_hash"):
        tags["robust_strategy.runtime_environment_hash"] = str(
            runtime_environment["environment_hash"]
        )
    active_run.set_tags(**tags)
    for key in (
        "rows",
        "dates",
        "rank_ic_mean",
        "rank_ic_positive_ratio",
        "neural_repeatability_max_abs_diff",
        "neural_repeatability_score_count",
    ):
        try:
            numeric = float(result[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            active_run.log_metrics(**{f"development.{key}": numeric})
    active_run.log_artifact(str(prediction_path), artifact_path="predictions")
    active_run.log_artifact(str(model_path), artifact_path="models")
    return str(active_run.id)


def run_training_jobs(
    config: Mapping[str, Any],
    jobs: Sequence[TrainingJob],
    *,
    provider_uri: Path,
    calendar: Sequence[Any],
    run_dir: Path,
    dataset_factory: Any = None,
    model_factory: Any = None,
    mlflow_enabled: bool = True,
    resume: bool = False,
    split_factory: Any = None,
    dataset_config_factory: Any = build_dataset_config,
    phase: str = "development",
    experiment_name: str = "robust_csi800_development",
    context_hash: str | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    run_dir = Path(run_dir)
    runtime_environment = dict(
        runtime_environment or collect_runtime_environment(config)
    )
    if context_hash is None:
        context_hash = hashlib.sha256(
            json.dumps(
                {
                    "config": config,
                    "provider_uri": str(Path(provider_uri).resolve()),
                    "phase": phase,
                    "experiment_name": experiment_name,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
    progress_path = run_dir / "progress.json"
    progress = _load_progress(progress_path, resume=resume)
    if progress.get("context_hash") != context_hash:
        progress = {"context_hash": context_hash, "jobs": {}}
    else:
        progress["context_hash"] = context_hash
    progress["runtime_environment_hash"] = runtime_environment.get("environment_hash")
    job_progress: dict[str, dict[str, Any]] = progress["jobs"]
    expected_splits: dict[str, dict[str, str]] = {}
    split_objects: dict[tuple[str, int], RollingSplit] = {}
    for job in jobs:
        group_key = (job.quarter, job.window_days)
        if group_key not in split_objects:
            split_objects[group_key] = (
                split_factory(job.quarter, job.window_days)
                if split_factory is not None
                else make_job_split(
                    calendar,
                    job.quarter,
                    window_days=job.window_days,
                    validation_days=int(
                        config["rolling_validation"]["validation_days"]
                    ),
                    label_horizon_days=5,
                )
            )
        expected_splits[job.key] = _split_record(split_objects[group_key])
    pending = pending_jobs(
        jobs,
        job_progress,
        run_dir=run_dir,
        expected_splits=expected_splits,
        context_hash=context_hash,
    )
    if not pending:
        return job_progress
    dataset_factory = dataset_factory or _default_dataset_factory
    model_factory = model_factory or _default_model_factory
    if dataset_factory is _default_dataset_factory:
        import qlib

        qlib.init(provider_uri=str(provider_uri), region="cn", kernels=8)

    groups: dict[tuple[str, int], list[TrainingJob]] = {}
    for job in pending:
        groups.setdefault((job.quarter, job.window_days), []).append(job)
    for (quarter, window_days), group_jobs in groups.items():
        try:
            split = split_objects[(quarter, window_days)]
            dataset_config = dataset_config_factory(split)
            dataset = dataset_factory(provider_uri, dataset_config, split, group_jobs)
        except Exception as exc:
            for job in group_jobs:
                job_progress[job.key] = {
                    "status": "failed",
                    "reason_code": "dataset_build_failed",
                    "error_type": type(exc).__name__,
                }
            _write_progress(progress_path, progress)
            continue
        for job in group_jobs:
            prediction_path = run_dir / "predictions" / f"{job.key}.parquet"
            model_path = run_dir / "models" / f"{job.key}.pkl"
            model_config = model_config_for_job(config, job)
            job_progress[job.key] = {
                "status": "running",
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            _write_progress(progress_path, progress)
            try:
                model = model_factory(model_config, job)
                with _mlflow_run(
                    job,
                    provider_uri,
                    enabled=mlflow_enabled,
                    phase=phase,
                    experiment_name=experiment_name,
                ) as active_run:
                    result = fit_predict_job(model, dataset, prediction_path)
                    if job.model_id in NEURAL_MODELS:
                        repeatability_config = config["training"][
                            "neural_repeatability"
                        ]
                        repeat_prediction_path = (
                            run_dir / "repeatability" / f"{job.key}.parquet"
                        )
                        try:
                            repeat_model = model_factory(model_config, job)
                            fit_predict_job(
                                repeat_model, dataset, repeat_prediction_path
                            )
                            result.update(
                                compare_neural_prediction_artifacts(
                                    prediction_path,
                                    repeat_prediction_path,
                                    score_max_abs_diff=float(
                                        repeatability_config["score_max_abs_diff"]
                                    ),
                                )
                            )
                        finally:
                            repeat_prediction_path.unlink(missing_ok=True)
                    _atomic_write_pickle(model, model_path)
                    run_id = _log_mlflow_result(
                        active_run,
                        job=job,
                        model_config=model_config,
                        split=split,
                        result=result,
                        prediction_path=prediction_path,
                        model_path=model_path,
                        phase=phase,
                        runtime_environment=runtime_environment,
                    )
                job_progress[job.key] = {
                    "status": "completed",
                    "finished_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "mlflow_run_id": run_id,
                    "model_id": job.model_id,
                    "window_days": job.window_days,
                    "quarter": job.quarter,
                    "seed": job.seed,
                    "model_path": str(model_path.resolve()),
                    "model_sha256": file_sha256(model_path),
                    "prediction_sha256": file_sha256(prediction_path),
                    "context_hash": context_hash,
                    "runtime_environment_hash": runtime_environment.get(
                        "environment_hash"
                    ),
                    "split": _split_record(split),
                    **result,
                }
            except Exception as exc:
                reason_code = (
                    "neural_repeatability_failed"
                    if isinstance(exc, NeuralRepeatabilityError)
                    else "model_training_failed"
                )
                job_progress[job.key] = {
                    "status": "failed",
                    "finished_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                    "reason_code": reason_code,
                    "error_type": type(exc).__name__,
                }
                if (
                    isinstance(exc, NeuralRepeatabilityError)
                    and exc.max_abs_diff is not None
                ):
                    job_progress[job.key]["neural_repeatability_max_abs_diff"] = (
                        exc.max_abs_diff
                    )
            _write_progress(progress_path, progress)
        del dataset
    return job_progress


def _read_provider_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_membership_source_metadata(pit_dir: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            (Path(pit_dir) / "MEMBERSHIP_SOURCE.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("selected") not in {
        "jqdata",
        "baostock",
    }:
        return {}
    return {
        key: copy.deepcopy(raw[key])
        for key in (
            "requested",
            "selected",
            "fallback_reason",
            "date_range",
            "indices",
        )
        if key in raw
    }


def _failure(reason_code: str, *, as_of_date: str) -> dict[str, Any]:
    return {
        "status": "development_failed",
        "reason_code": reason_code,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "training_jobs_started": 0,
    }


def _standard_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _standard_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_standard_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(
            _standard_json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _failure_report(
    reason_code: str, *, as_of_date: str, started: int = 0, **details: Any
) -> dict[str, Any]:
    return {
        "status": "development_failed",
        "reason_code": reason_code,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "training_jobs_started": int(started),
        **details,
    }


def selection_contract_hash(
    selected_models: Sequence[str],
    selected_model_configs: Mapping[str, Any],
    selected_strategy: str,
) -> str:
    contract = {
        "selected_models": [str(value) for value in selected_models],
        "selected_model_configs": selected_model_configs,
        "selected_strategy": str(selected_strategy),
    }
    serialized = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def payload_sha256(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def launch_runtime_evidence(
    config: Mapping[str, Any],
    *,
    report_dir: Path,
    run_id: str,
    runtime_environment: Mapping[str, Any],
    runner_namespace: str = "runner",
) -> dict[str, Any]:
    requires_mps = "mps_dnn" in {
        str(item.get("id")) for item in config.get("models") or []
    }
    if not requires_mps:
        return {"required": False}
    record = _read_provider_state(
        Path(report_dir) / runner_namespace / run_id / "launch.json"
    )
    evidence = record.get("mps_preflight")
    expected_python = str(Path(sys.executable).absolute())
    expected_torch = str(
        (runtime_environment.get("packages") or {}).get("torch") or ""
    )
    if (
        record.get("run_id") != run_id
        or record.get("mps_required") is not True
        or not isinstance(evidence, Mapping)
        or evidence.get("python_executable") != expected_python
        or not expected_torch
        or str(evidence.get("torch_version") or "") != expected_torch
        or evidence.get("mps_built") is not True
        or evidence.get("mps_available") is not True
        or evidence.get("device") != "mps:0"
        or evidence.get("tensor_sum") != 4.0
        or not evidence.get("checked_at")
    ):
        raise ValueError("launchd MPS preflight evidence is missing or invalid")
    return {"required": True, "source": "launchd_worker", **dict(evidence)}


def frozen_context_hash(payload: Mapping[str, Any]) -> str:
    return payload_sha256({key: payload.get(key) for key in FROZEN_CONTEXT_KEYS})


def _development_artifacts(
    jobs: Sequence[TrainingJob],
    progress: Mapping[str, Mapping[str, Any]],
    selected_models: Sequence[str],
) -> list[dict[str, Any]]:
    selected = set(selected_models)
    artifacts = []
    for job in sorted(jobs, key=lambda item: item.key):
        if job.model_id not in selected:
            continue
        state = progress[job.key]
        artifacts.append(
            {
                "status": "completed",
                "job_key": job.key,
                "model_id": job.model_id,
                "window_days": job.window_days,
                "quarter": job.quarter,
                "seed": job.seed,
                "model_path": state["model_path"],
                "model_sha256": state["model_sha256"],
                "prediction_path": state["prediction_path"],
                "prediction_sha256": state["prediction_sha256"],
                "rows": state["rows"],
                "split": copy.deepcopy(state["split"]),
                "context_hash": state["context_hash"],
            }
        )
    return artifacts


def select_development_artifacts(
    selected_reports: Sequence[Mapping[str, Any]],
    *,
    fixed_artifacts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    tuned_artifacts_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Bind every selected model to the matrix that produced its selected report."""
    selected: list[dict[str, Any]] = []
    for report in selected_reports:
        model_id = str(report["model_id"])
        source = (
            tuned_artifacts_by_model
            if isinstance(report.get("tuning"), Mapping)
            else fixed_artifacts_by_model
        )
        artifacts = source.get(model_id)
        if not artifacts:
            raise ValueError(f"selected model artifacts are missing: {model_id}")
        selected.extend(copy.deepcopy([dict(item) for item in artifacts]))
    return sorted(selected, key=lambda item: str(item.get("job_key", "")))


def _externalize_frames(value: Any, output_dir: Path, prefix: str) -> Any:
    if isinstance(value, pd.DataFrame):
        path = output_dir / f"{prefix}.parquet"
        _atomic_write_parquet(
            (
                value.reset_index(drop=False)
                if not isinstance(value.index, pd.RangeIndex)
                else value
            ),
            path,
        )
        return {"artifact": str(path.resolve()), "rows": int(len(value))}
    if isinstance(value, dict):
        return {
            str(key): _externalize_frames(item, output_dir, f"{prefix}_{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _externalize_frames(item, output_dir, f"{prefix}_{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _development_markdown(payload: Mapping[str, Any]) -> str:
    selected_models = ", ".join(payload.get("selected_models") or []) or "none"
    selected_strategy = payload.get("selected_strategy") or "none"
    lines = [
        "# Robust CSI800 Development Report",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Reason: `{payload.get('reason_code') or 'none'}`",
        f"- Config hash: `{payload.get('config_hash')}`",
        f"- Git commit: `{payload.get('git_commit') or 'unavailable'}`",
        f"- Data manifest hash: `{payload.get('data_manifest_hash') or 'unavailable'}`",
        f"- Selected models: `{selected_models}`",
        f"- Selected strategy: `{selected_strategy}`",
        "",
        "This is read-only research. It does not connect to a broker or place orders.",
    ]
    return "\n".join(lines) + "\n"


def _run_development_impl(
    *,
    config_path: Path,
    quant_data_dir: Path,
    report_dir: Path,
    provider_state_path: Path,
    as_of_date: str,
    run_id: str,
    resume: bool = False,
    training_runner: Any = run_training_jobs,
    aggregator: Any = aggregate_training_artifacts,
    market_loader: Any = load_market_panel,
    benchmark_loader: Any = load_benchmark_frame,
    regime_builder: Any = build_market_regimes,
    strategy_evaluator: Any = evaluate_strategy_candidates,
    tuner: Any = run_bounded_tuning,
    git_state_provider: Any = git_worktree_state,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    quant_data_dir = Path(quant_data_dir).resolve()
    report_dir = Path(report_dir).resolve()
    provider_state_path = Path(provider_state_path).resolve()
    config = load_config(config_path)
    runtime_environment = collect_runtime_environment(config)
    try:
        launch_evidence = launch_runtime_evidence(
            config,
            report_dir=report_dir,
            run_id=run_id,
            runtime_environment=runtime_environment,
        )
    except ValueError:
        payload = _failure_report(
            "launch_mps_preflight_missing_or_invalid", as_of_date=as_of_date
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    launch_evidence_hash = payload_sha256(launch_evidence)
    paths = config["paths"]
    pit_dir = quant_data_dir / paths["pit_membership"]
    reference_dir = quant_data_dir / Path(paths["security_master"]).parent
    provider_state = _read_provider_state(provider_state_path)
    provider_uri = Path(str(provider_state.get("provider_uri") or ""))
    if provider_state.get("status") != "completed" or not provider_uri.is_dir():
        payload = _failure_report("provider_state_missing", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    provider_key = str(provider_state.get("provider_key") or "")
    if not provider_key:
        payload = _failure_report("provider_identity_missing", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    provider_identity = runtime_provider_identity(provider_uri, provider_key)
    if provider_identity is None:
        payload = _failure_report("provider_snapshot_invalid", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    provider_tree_sha256 = provider_identity["provider_tree_sha256"]
    calendar_path = provider_uri / "calendars" / "day.txt"
    if not calendar_path.is_file():
        payload = _failure_report("provider_calendar_missing", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    initial_git = dict(git_state_provider())
    if not initial_git.get("commit"):
        payload = _failure_report("git_commit_unavailable", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    if initial_git.get("clean") is not True:
        payload = _failure_report(
            "git_worktree_dirty",
            as_of_date=as_of_date,
            git_changes=list(initial_git.get("changes") or [])[:100],
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    manifest_path = quant_data_dir / "DATA_MANIFEST.json"
    if not manifest_path.is_file():
        payload = _failure_report("data_manifest_missing", as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    initial_config_hash = file_sha256(config_path)
    initial_manifest_hash = file_sha256(manifest_path)
    try:
        with membership_read_lock(pit_dir) as membership_snapshot:
            membership_snapshot.ensure_transaction_clean()
            membership_source_metadata = membership_snapshot.validate_source(
                start_date="2023-01-01",
                end_date="2025-12-31",
            )
            membership_snapshot_hash = membership_snapshot.fingerprint()
    except MembershipIntegrityError as exc:
        payload = _failure_report(exc.reason_code, as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    try:
        with reference_read_lock(reference_dir) as reference_snapshot:
            reference_snapshot.ensure_transaction_clean()
            reference_source_metadata = reference_snapshot.validate_source(
                start_date="2023-01-01",
                end_date="2025-12-31",
                expected_membership_snapshot_hash=membership_snapshot_hash,
            )
            reference_snapshot_hash = reference_snapshot.fingerprint()
    except ReferenceIntegrityError as exc:
        payload = _failure_report(exc.reason_code, as_of_date=as_of_date)
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            calendar_path.read_text(encoding="utf-8").splitlines(), errors="coerce"
        )
    ).dropna()
    jobs = build_training_jobs(config)
    run_dir = report_dir / "runs" / run_id / "development"
    context_path = run_dir / "development_context.json"
    expected_context = {
        "config_hash": initial_config_hash,
        "git_commit": initial_git["commit"],
        "data_manifest_hash": initial_manifest_hash,
        "provider_uri": str(provider_uri.resolve()),
        "provider_key": provider_key,
        "as_of_date": str(as_of_date),
        "membership_source_metadata": membership_source_metadata,
        "membership_snapshot_hash": membership_snapshot_hash,
        "reference_source_metadata": reference_source_metadata,
        "reference_snapshot_hash": reference_snapshot_hash,
        "provider_tree_sha256": provider_tree_sha256,
        "runtime_environment": runtime_environment,
        "runtime_environment_hash": runtime_environment["environment_hash"],
        "launch_runtime_evidence": launch_evidence,
        "launch_runtime_evidence_hash": launch_evidence_hash,
    }
    development_context_hash = payload_sha256(expected_context)
    expected_context["context_hash"] = development_context_hash
    if resume:
        stored_context = _read_provider_state(context_path)
        if not stored_context:
            payload = _failure_report("resume_context_missing", as_of_date=as_of_date)
            _write_json(report_dir / "development_latest.json", payload)
            return payload
        mismatches = [
            key
            for key, value in expected_context.items()
            if key != "context_hash" and stored_context.get(key) != value
        ]
        if (
            not mismatches
            and stored_context.get("context_hash") != development_context_hash
        ):
            mismatches.append("context_hash")
        if mismatches:
            payload = _failure_report(
                "resume_context_mismatch",
                as_of_date=as_of_date,
                context_mismatches=mismatches,
            )
            _write_json(report_dir / "development_latest.json", payload)
            return payload
    else:
        if context_path.exists():
            payload = _failure_report(
                "run_context_already_exists", as_of_date=as_of_date
            )
            _write_json(report_dir / "development_latest.json", payload)
            return payload
        _write_json(context_path, expected_context)
    if training_runner is run_training_jobs:
        dependencies = ["qlib", "sklearn", "lightgbm", "xgboost", "catboost", "mlflow"]
        if "mps_dnn" in {str(item["id"]) for item in config["models"]}:
            dependencies.append("torch")
        missing = [
            name for name in dependencies if importlib.util.find_spec(name) is None
        ]
        if missing:
            payload = _failure_report(
                "training_dependencies_missing",
                as_of_date=as_of_date,
                missing_dependencies=missing,
            )
            _write_json(report_dir / "development_latest.json", payload)
            return payload
        if "mps_dnn" in {str(item["id"]) for item in config["models"]}:
            try:
                from mps_dnn import require_mps

                require_mps()
            except Exception as exc:
                payload = _failure_report(
                    "mps_dnn_unavailable",
                    as_of_date=as_of_date,
                    error_type=type(exc).__name__,
                )
                _write_json(report_dir / "development_latest.json", payload)
                return payload
    progress = training_runner(
        config,
        jobs,
        provider_uri=provider_uri,
        calendar=calendar,
        run_dir=run_dir,
        mlflow_enabled=True,
        resume=resume,
        context_hash=development_context_hash,
        runtime_environment=runtime_environment,
    )
    incomplete = [
        job.key
        for job in jobs
        if progress.get(job.key, {}).get("status") != "completed"
    ]
    if incomplete:
        payload = _failure_report(
            "training_jobs_incomplete",
            as_of_date=as_of_date,
            started=sum(
                1
                for value in progress.values()
                if value.get("status") in {"running", "completed", "failed"}
            ),
            training_job_count=len(jobs),
            incomplete_job_count=len(incomplete),
            incomplete_job_sample=incomplete[:20],
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    records = [
        {
            "model_id": job.model_id,
            "window_days": job.window_days,
            "quarter": job.quarter,
            "seed": job.seed,
            "prediction_path": progress[job.key]["prediction_path"],
        }
        for job in jobs
    ]
    aggregated = aggregator(
        records,
        topk=int(config["portfolio"]["topk"]),
        thresholds=config["acceptance"],
    )
    selection_options = {
        "maximum": int(config["training"]["ensemble_models_max"]),
        "topk": int(config["portfolio"]["topk"]),
        "rank_correlation_abs_max": float(
            config["training"]["ensemble_model_rank_correlation_abs_max"]
        ),
        "topk_overlap_max": float(
            config["training"]["ensemble_model_top50_overlap_max"]
        ),
    }
    selected_reports = select_development_models(
        aggregated["model_reports"],
        minimum=int(config["training"]["ensemble_models_min"]),
        model_predictions=aggregated["model_predictions"],
        **selection_options,
    )
    tuning_summaries: list[dict[str, Any]] = []
    tuned_artifacts_by_model: dict[str, list[dict[str, Any]]] = {}
    if len(selected_reports) < int(config["training"]["ensemble_models_min"]):
        preserved_reports = select_development_models(
            aggregated["model_reports"],
            minimum=1,
            model_predictions=aggregated["model_predictions"],
            **selection_options,
        )
        preserved_ids = {str(item["model_id"]) for item in preserved_reports}
        failed_model_ids = [
            str(item["model_id"])
            for item in aggregated["model_reports"]
            if item.get("eligible") is not True
            or str(item["model_id"]) not in preserved_ids
        ]
        tuned = tuner(
            config,
            failed_model_ids,
            provider_uri=provider_uri,
            calendar=calendar,
            run_dir=run_dir / "tuning",
            context_hash=development_context_hash,
        )
        tuning_summaries = list(tuned.get("summaries") or [])
        tuned_artifacts_by_model = {
            str(model_id): [dict(item) for item in artifacts]
            for model_id, artifacts in (tuned.get("model_artifacts") or {}).items()
        }
        reports_by_model = {
            str(item["model_id"]): item for item in aggregated["model_reports"]
        }
        predictions_by_model = dict(aggregated["model_predictions"])
        for report in tuned.get("model_reports") or []:
            reports_by_model[str(report["model_id"])] = report
        predictions_by_model.update(tuned.get("model_predictions") or {})
        aggregated["model_reports"] = list(reports_by_model.values())
        aggregated["model_predictions"] = predictions_by_model
        selected_reports = select_development_models(
            aggregated["model_reports"],
            minimum=int(config["training"]["ensemble_models_min"]),
            model_predictions=aggregated["model_predictions"],
            **selection_options,
        )
        if len(selected_reports) < int(config["training"]["ensemble_models_min"]):
            eligible_ids = [
                str(item["model_id"])
                for item in aggregated["model_reports"]
                if item.get("eligible") is True
            ]
            payload = _failure_report(
                "insufficient_stable_diverse_models_after_bounded_tuning",
                as_of_date=as_of_date,
                started=len(jobs),
                model_reports=aggregated["model_reports"],
                model_diversification=cross_model_diversification_report(
                    aggregated["model_predictions"],
                    eligible_ids,
                    topk=int(config["portfolio"]["topk"]),
                    rank_correlation_abs_max=selection_options[
                        "rank_correlation_abs_max"
                    ],
                    topk_overlap_max=selection_options["topk_overlap_max"],
                ),
                tuning_summaries=tuning_summaries,
                bounded_tuning_trials_per_model=int(
                    config["training"]["bounded_tuning_trials_per_failed_model_max"]
                ),
            )
            _write_json(report_dir / "development_latest.json", payload)
            return payload
    selected_ids = [str(item["model_id"]) for item in selected_reports]
    model_diversification = cross_model_diversification_report(
        aggregated["model_predictions"],
        selected_ids,
        topk=int(config["portfolio"]["topk"]),
        rank_correlation_abs_max=selection_options["rank_correlation_abs_max"],
        topk_overlap_max=selection_options["topk_overlap_max"],
    )
    ensemble = aggregate_prediction_frames(
        {
            model_id: aggregated["model_predictions"][model_id]
            for model_id in selected_ids
        }
    )
    price_dir = quant_data_dir / paths["price_parquet"]
    market = market_loader(
        pit_dir=pit_dir,
        price_dir=price_dir,
        reference_dir=reference_dir,
        start_date="2023-01-01",
        end_date="2025-12-31",
        ohlcv_coverage_min=float(config["data_gates"]["ohlcv_coverage_min"]),
    )
    market_reference_source = market.attrs.get("reference_source_metadata")
    market_reference_hash = market.attrs.get("reference_snapshot_hash")
    if market_reference_source is None or market_reference_hash is None:
        try:
            with reference_read_lock(reference_dir) as reference_snapshot:
                reference_snapshot.ensure_transaction_clean()
                market_reference_source = reference_snapshot.validate_source(
                    start_date="2023-01-01",
                    end_date="2025-12-31",
                    expected_membership_snapshot_hash=membership_snapshot_hash,
                )
                market_reference_hash = reference_snapshot.fingerprint()
        except ReferenceIntegrityError:
            market_reference_source = None
            market_reference_hash = None
    if (
        market_reference_source != reference_source_metadata
        or market_reference_hash != reference_snapshot_hash
    ):
        payload = _failure_report(
            "reference_snapshot_changed",
            as_of_date=as_of_date,
            started=len(jobs),
            reference_source_metadata=reference_source_metadata,
            reference_snapshot_hash=reference_snapshot_hash,
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    market_membership_source = market.attrs.get("membership_source_metadata")
    market_membership_hash = market.attrs.get("membership_snapshot_hash")
    if market_membership_source is None or market_membership_hash is None:
        try:
            with membership_read_lock(pit_dir) as membership_snapshot:
                membership_snapshot.ensure_transaction_clean()
                market_membership_source = membership_snapshot.validate_source(
                    start_date="2023-01-01",
                    end_date="2025-12-31",
                )
                market_membership_hash = membership_snapshot.fingerprint()
        except MembershipIntegrityError:
            market_membership_source = None
            market_membership_hash = None
    if (
        market_membership_source != membership_source_metadata
        or market_membership_hash != membership_snapshot_hash
    ):
        payload = _failure_report(
            "membership_snapshot_changed",
            as_of_date=as_of_date,
            started=len(jobs),
            membership_source_metadata=membership_source_metadata,
            membership_snapshot_hash=membership_snapshot_hash,
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    benchmark = benchmark_loader(
        provider_uri, start_date="2023-01-01", end_date="2025-12-31"
    )
    regimes = regime_builder(market)
    evaluation = strategy_evaluator(
        predictions=ensemble,
        market=market,
        benchmark=benchmark,
        regimes=regimes,
        config=config,
        selected_model_reports=selected_reports,
    )
    final_git = dict(git_state_provider())
    reproducibility_failures: list[str] = []
    if final_git.get("clean") is not True:
        reproducibility_failures.append("git_worktree_dirty")
    if final_git.get("commit") != initial_git.get("commit"):
        reproducibility_failures.append("git_commit_changed")
    if file_sha256(config_path) != initial_config_hash:
        reproducibility_failures.append("config_changed")
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != initial_manifest_hash
    ):
        reproducibility_failures.append("data_manifest_changed")
    final_provider_state = _read_provider_state(provider_state_path)
    if final_provider_state.get("status") != "completed":
        reproducibility_failures.append("provider_status")
    final_provider_uri = str(final_provider_state.get("provider_uri") or "")
    if (
        not final_provider_uri
        or Path(final_provider_uri).resolve() != provider_uri.resolve()
    ):
        reproducibility_failures.append("provider_uri")
    if final_provider_state.get("provider_key") != provider_key:
        reproducibility_failures.append("provider_key")
    final_provider_identity = runtime_provider_identity(provider_uri, provider_key)
    if final_provider_identity is None:
        reproducibility_failures.append("provider_snapshot")
    elif final_provider_identity.get("provider_tree_sha256") != provider_tree_sha256:
        reproducibility_failures.append("provider_tree_sha256")
    if collect_runtime_environment(config).get(
        "environment_hash"
    ) != runtime_environment.get("environment_hash"):
        reproducibility_failures.append("runtime_environment_changed")
    if reproducibility_failures:
        payload = _failure_report(
            "reproducibility_state_changed_during_development",
            as_of_date=as_of_date,
            started=len(jobs),
            reproducibility_failures=reproducibility_failures,
            git_changes=list(final_git.get("changes") or [])[:100],
        )
        _write_json(report_dir / "development_latest.json", payload)
        return payload
    payload = {
        "status": evaluation["status"],
        "reason_code": evaluation.get("reason_code"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": as_of_date,
        "run_id": run_id,
        "config_path": str(config_path),
        "config_hash": initial_config_hash,
        "git_commit": initial_git["commit"],
        "data_manifest_path": str(manifest_path),
        "data_manifest_hash": initial_manifest_hash,
        "membership_source": str(pit_dir / "membership_intervals.parquet"),
        "membership_source_metadata": membership_source_metadata,
        "membership_snapshot_hash": membership_snapshot_hash,
        "reference_source_metadata": reference_source_metadata,
        "reference_snapshot_hash": reference_snapshot_hash,
        "provider_uri": str(provider_uri),
        "provider_key": provider_key,
        "provider_tree_sha256": provider_tree_sha256,
        "runtime_environment": runtime_environment,
        "runtime_environment_hash": runtime_environment["environment_hash"],
        "launch_runtime_evidence": launch_evidence,
        "launch_runtime_evidence_hash": launch_evidence_hash,
        "development_context_hash": development_context_hash,
        "training_job_count": len(jobs),
        "selected_models": selected_ids,
        "selected_strategy": (
            evaluation.get("selected_strategy", {}).get("strategy_id")
            if evaluation.get("selected_strategy")
            else None
        ),
        "model_reports": aggregated["model_reports"],
        "model_diversification": model_diversification,
        "tuning_summaries": tuning_summaries,
        "evaluation": _externalize_frames(
            evaluation, run_dir / "backtests", "development"
        ),
    }
    frozen: dict[str, Any] | None = None
    if payload["status"] == "research_candidate_ready":
        base_models = {str(item["id"]): item for item in config["models"]}
        selected_model_configs = {}
        for report in selected_reports:
            model_id = str(report["model_id"])
            selected_model_configs[model_id] = {
                "class": base_models[model_id]["class"],
                "module_path": base_models[model_id]["module_path"],
                "kwargs": copy.deepcopy(
                    report.get("tuning", {}).get("selected_kwargs")
                    or base_models[model_id].get("kwargs")
                    or {}
                ),
            }
        fixed_artifacts = _development_artifacts(jobs, progress, selected_ids)
        fixed_artifacts_by_model: dict[str, list[dict[str, Any]]] = {}
        for artifact in fixed_artifacts:
            fixed_artifacts_by_model.setdefault(str(artifact["model_id"]), []).append(
                artifact
            )
        development_artifacts = select_development_artifacts(
            selected_reports,
            fixed_artifacts_by_model=fixed_artifacts_by_model,
            tuned_artifacts_by_model=tuned_artifacts_by_model,
        )
        frozen = {
            "status": "research_candidate_ready",
            "frozen_at": payload["generated_at"],
            "as_of_date": payload["as_of_date"],
            "config_hash": payload["config_hash"],
            "git_commit": payload["git_commit"],
            "data_manifest_hash": payload["data_manifest_hash"],
            "membership_source": payload["membership_source"],
            "membership_source_metadata": payload["membership_source_metadata"],
            "membership_snapshot_hash": payload["membership_snapshot_hash"],
            "reference_source_metadata": payload["reference_source_metadata"],
            "reference_snapshot_hash": payload["reference_snapshot_hash"],
            "provider_uri": payload["provider_uri"],
            "provider_key": payload["provider_key"],
            "provider_tree_sha256": payload["provider_tree_sha256"],
            "runtime_environment": payload["runtime_environment"],
            "runtime_environment_hash": payload["runtime_environment_hash"],
            "launch_runtime_evidence": payload["launch_runtime_evidence"],
            "launch_runtime_evidence_hash": payload[
                "launch_runtime_evidence_hash"
            ],
            "selected_models": selected_ids,
            "selected_model_configs": selected_model_configs,
            "selected_strategy": payload["selected_strategy"],
            "selection_hash": selection_contract_hash(
                selected_ids,
                selected_model_configs,
                str(payload["selected_strategy"]),
            ),
            "development_run_id": run_id,
            "development_run_dir": str(run_dir.resolve()),
            "development_context_hash": development_context_hash,
            "development_artifacts": development_artifacts,
            "development_artifacts_hash": payload_sha256(development_artifacts),
            "confirmation_run_started": False,
            "shadow_tracking_started": False,
        }
        frozen["frozen_context_hash"] = frozen_context_hash(frozen)
    _write_json(run_dir / "development.json", payload)
    _write_json(report_dir / "development_latest.json", payload)
    _atomic_write(report_dir / "development_latest.md", _development_markdown(payload))
    if frozen is not None:
        _write_json(run_dir / "frozen_strategy.json", frozen)
        _write_json(report_dir / "frozen_strategy_latest.json", frozen)
    return payload


def _publish_development_result(
    payload: Mapping[str, Any], *, report_dir: Path, run_id: str | None
) -> None:
    """Persist every result both per-run and as the convenience latest view."""
    report_dir = Path(report_dir).resolve()
    if run_id:
        _write_json(
            report_dir / "runs" / str(run_id) / "development" / "development.json",
            payload,
        )
    _write_json(report_dir / "development_latest.json", payload)
    _atomic_write(report_dir / "development_latest.md", _development_markdown(payload))


def run_development(**kwargs: Any) -> dict[str, Any]:
    """Run development and persist normal and unexpected results consistently."""
    report_dir = Path(kwargs["report_dir"]).resolve()
    run_id = str(kwargs.get("run_id") or "") or None
    try:
        payload = _run_development_impl(**kwargs)
    except Exception as exc:
        payload = _failure_report(
            "development_execution_failed",
            as_of_date=str(kwargs.get("as_of_date") or ""),
            run_id=run_id or "",
            error_type=type(exc).__name__,
        )
    _publish_development_result(payload, report_dir=report_dir, run_id=run_id)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run 2023-2025 rolling robust strategy development."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--provider-state", type=Path, required=True)
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    jobs = build_training_jobs(config)
    if args.plan_only:
        provider_state = _read_provider_state(args.provider_state)
        provider_uri = Path(str(provider_state.get("provider_uri") or ""))
        payload = {
            "lifecycle_status": "planned",
            "official_status": None,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "as_of_date": args.as_of_date,
            "provider_uri": str(provider_uri.resolve()) if provider_uri else None,
            "training_job_count": len(jobs),
            "models": [item["id"] for item in config["models"]],
            "windows_trading_days": config["training"]["windows_trading_days"],
            "tree_seeds": config["training"]["tree_seeds"],
            "test_quarters": config["rolling_validation"]["test_quarters"],
        }
        _write_json(args.report_dir / "development_plan_latest.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    provider_state = _read_provider_state(args.provider_state)
    provider_uri = Path(str(provider_state.get("provider_uri") or ""))
    if provider_state.get("status") != "completed" or not provider_uri.is_dir():
        payload = _failure_report("provider_state_missing", as_of_date=args.as_of_date)
        _publish_development_result(
            payload, report_dir=args.report_dir, run_id=args.run_id
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    result = run_development(
        config_path=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        provider_state_path=args.provider_state,
        as_of_date=args.as_of_date,
        run_id=args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "research_candidate_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
