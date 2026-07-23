#!/usr/bin/env python3
"""Run the isolated, resumable CSI800 signal-research workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cloudpickle
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, quant_data_root, reports_root  # noqa: E402
from robust_dataset import board_from_symbol  # noqa: E402
from robust_evaluation import (  # noqa: E402
    aggregate_prediction_frames,
    window_stability,
)
from robust_market_data import build_market_regimes  # noqa: E402
from robust_signal_dataset import (  # noqa: E402
    build_signal_dataset_config,
    label_expression,
)
from robust_signal_evaluation import (  # noqa: E402
    acceptance_verdict,
    decile_monotonicity,
    feature_diagnostics,
    horizon_aligned_excess,
    market_state_attribution,
    prediction_dispersion,
    quarterly_rank_ic,
    topk_sensitivity,
)
from robust_signal_models import EqualWeightRankModel  # noqa: E402
from robust_strategy_core import RollingSplit, make_rolling_split  # noqa: E402


DEFAULT_CONFIG = SCRIPT_DIR.parent / "configs" / "robust_signal_research.yaml"
DEFAULT_REPORT_DIR = reports_root() / "robust_signal"
LOCKED_CONFIRMATION_START = pd.Timestamp("2026-01-01")
FINAL_STATUSES = {
    "blocked_data",
    "signal_diagnosis_complete",
    "signal_not_found",
    "signal_candidate_ready",
}


@dataclass(frozen=True)
class SignalJob:
    phase: str
    signal_id: str
    model_id: str
    quarter: str
    window_days: int

    @property
    def key(self) -> str:
        return "__".join(
            [self.phase, self.signal_id, self.model_id, self.quarter, f"w{self.window_days}"]
        )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            cloudpickle.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _git_clean(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    return not completed.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("signal research config must be a mapping")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    statuses = set(config.get("statuses") or [])
    if statuses != FINAL_STATUSES:
        raise ValueError("signal research statuses do not match the frozen contract")
    signals = config.get("signals") or []
    if len(signals) != 8 or len({str(item.get("id")) for item in signals}) != 8:
        raise ValueError("signal research requires exactly eight unique preregistered schemes")
    quarters = list(config["periods"]["screening_quarters"]) + list(
        config["periods"]["validation_quarters"]
    )
    if any(str(quarter).startswith("2026") for quarter in quarters):
        raise ValueError("Goal 1 must not use 2026 screening or validation results")
    if pd.Timestamp(config["safety"]["locked_confirmation_start"]) != LOCKED_CONFIRMATION_START:
        raise ValueError("the 2026 confirmation lock must remain 2026-01-01")
    if int(config["evaluation"]["primary_topk"]) != 50:
        raise ValueError("Top50 is the frozen primary acceptance scope")
    if str(config["evaluation"]["market"]) != "csi800":
        raise ValueError("Goal 1 market universe must remain csi800")
    if int(config["evaluation"].get("market_state_lookback_days", 0)) < 60:
        raise ValueError("market-state features require at least 60 trading days of lookback")


def _resolve_provider(config: Mapping[str, Any], report_root: Path) -> tuple[Path, dict[str, Any]]:
    status_path = report_root / str(config["paths"]["qlib_provider_status"])
    status = _read_json(status_path)
    provider = Path(str(status.get("provider_uri") or "")).expanduser()
    if not provider.is_dir():
        raise ValueError("verified robust provider is unavailable")
    if not (provider / "instruments" / "csi800.txt").is_file():
        raise ValueError("verified provider does not contain PIT csi800 instruments")
    return provider.resolve(), status


def _calendar(provider: Path) -> pd.DatetimeIndex:
    path = provider / "calendars" / "day.txt"
    values = pd.to_datetime(path.read_text(encoding="utf-8").splitlines(), errors="coerce")
    calendar = pd.DatetimeIndex(values).dropna().normalize().sort_values().unique()
    if calendar.empty:
        raise ValueError("Qlib provider calendar is empty")
    return calendar


def _lookback_start(
    calendar: pd.DatetimeIndex, start_date: Any, lookback_days: int
) -> pd.Timestamp:
    start = pd.Timestamp(start_date).normalize()
    position = int(calendar.searchsorted(start))
    if position >= len(calendar) or calendar[position] != start:
        raise ValueError(f"start date is not in provider calendar: {start.date()}")
    return pd.Timestamp(calendar[max(0, position - int(lookback_days))]).normalize()


def build_context(
    *, config_path: Path, config: Mapping[str, Any], quant_data_dir: Path, report_root: Path
) -> dict[str, Any]:
    provider, provider_status = _resolve_provider(config, report_root)
    manifest_path = quant_data_dir / str(config["paths"]["data_manifest"])
    if not manifest_path.is_file():
        raise ValueError("DATA_MANIFEST.json is unavailable")
    repo_root = SCRIPT_DIR.parents[1]
    payload = {
        "config_hash": _sha256_file(config_path),
        "git_commit": _git_commit(repo_root),
        "git_clean": _git_clean(repo_root),
        "data_manifest_hash": _sha256_file(manifest_path),
        "provider_uri": str(provider),
        "provider_tree_sha256": provider_status.get("provider_tree_sha256"),
        "provider_key": provider_status.get("provider_key"),
        "membership_snapshot_hash": provider_status.get("membership_snapshot_hash"),
        "reference_snapshot_hash": provider_status.get("reference_snapshot_hash"),
        "locked_confirmation_start": str(LOCKED_CONFIRMATION_START.date()),
    }
    payload["context_hash"] = _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    return payload


def build_jobs(
    config: Mapping[str, Any],
    *,
    phase: str,
    signal_ids: Sequence[str] | None = None,
    quarters: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
) -> list[SignalJob]:
    if phase not in {"screening", "validation"}:
        raise ValueError("phase must be screening or validation")
    selected_signals = [
        item
        for item in config["signals"]
        if signal_ids is None or str(item["id"]) in set(signal_ids)
    ]
    configured_models = config["models"][phase]
    selected_models = [
        item
        for item in configured_models
        if model_ids is None or str(item["id"]) in set(model_ids)
    ]
    selected_quarters = list(
        quarters
        if quarters is not None
        else config["periods"][f"{phase}_quarters"]
    )
    windows = (
        [int(config["periods"]["screening_window_days"])]
        if phase == "screening"
        else [int(value) for value in config["periods"]["validation_windows_days"]]
    )
    jobs = [
        SignalJob(
            phase=phase,
            signal_id=str(signal["id"]),
            model_id=str(model["id"]),
            quarter=str(quarter),
            window_days=window,
        )
        for signal in selected_signals
        for quarter in selected_quarters
        for window in windows
        for model in selected_models
    ]
    if any(job.quarter.startswith("2026") for job in jobs):
        raise ValueError("Goal 1 job construction attempted to access 2026")
    return jobs


def run_job_loop(
    jobs: Sequence[SignalJob],
    *,
    progress_path: Path,
    context_hash: str,
    resume: bool,
    run_job: Callable[[SignalJob], Mapping[str, Any]],
    pause_requested: Callable[[], bool],
) -> dict[str, Any]:
    existing = _read_json(progress_path) if resume else {}
    if resume and existing and existing.get("context_hash") != context_hash:
        raise ValueError("resume_context_mismatch")
    if not resume and progress_path.exists():
        raise ValueError("run_progress_already_exists")
    progress: dict[str, Any] = existing or {"context_hash": context_hash, "jobs": {}}
    progress["context_hash"] = context_hash
    progress.setdefault("jobs", {})
    progress["pause_state"] = "running"
    for job in jobs:
        previous = progress["jobs"].get(job.key) or {}
        prediction = Path(str(previous.get("prediction_path") or ""))
        model = Path(str(previous.get("model_path") or ""))
        if (
            previous.get("status") == "completed"
            and previous.get("context_hash") == context_hash
            and prediction.is_file()
            and model.is_file()
        ):
            continue
        progress["current_job"] = job.key
        progress["jobs"][job.key] = {
            "status": "running",
            "context_hash": context_hash,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(progress_path, progress)
        try:
            result = dict(run_job(job))
        except Exception as exc:
            progress["jobs"][job.key] = {
                "status": "failed",
                "context_hash": context_hash,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            progress["pause_state"] = "failed"
            progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _write_json(progress_path, progress)
            raise
        progress["jobs"][job.key] = {
            **result,
            "status": "completed",
            "context_hash": context_hash,
            "phase": job.phase,
            "signal_id": job.signal_id,
            "model_id": job.model_id,
            "quarter": job.quarter,
            "window_days": job.window_days,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        progress["completed_count"] = sum(
            value.get("status") == "completed" for value in progress["jobs"].values()
        )
        progress["expected_count"] = len(jobs)
        progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(progress_path, progress)
        if pause_requested():
            progress["pause_state"] = "paused"
            progress["pause_reason"] = "user_requested_after_current_job"
            progress["current_job"] = None
            progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            _write_json(progress_path, progress)
            return progress
    progress["pause_state"] = "completed"
    progress["current_job"] = None
    progress["completed_count"] = len(jobs)
    progress["expected_count"] = len(jobs)
    progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(progress_path, progress)
    return progress


def request_pause(report_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = report_dir / "runs" / run_id
    status = _read_json(run_dir / "status.json")
    if not status:
        raise ValueError("signal research run does not exist")
    payload = {
        "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": "after_current_job",
    }
    _write_json(run_dir / "pause.request.json", payload)
    status["pause_state"] = "pause_requested"
    status["updated_at"] = payload["requested_at"]
    _write_json(run_dir / "status.json", status)
    return payload


def _prediction_frame(
    model: Any,
    dataset: Any,
    benchmark: pd.Series,
    market_states: pd.Series,
) -> pd.DataFrame:
    prediction = model.predict(dataset, segment="test")
    if isinstance(prediction, pd.DataFrame):
        if prediction.shape[1] != 1:
            raise ValueError("prediction must contain one score column")
        prediction = prediction.iloc[:, 0]
    labels = dataset.prepare("test", col_set="label", data_key="learn")
    if isinstance(labels, pd.DataFrame) and labels.shape[1] == 1:
        labels = labels.iloc[:, 0]
    if not isinstance(prediction, pd.Series) or not isinstance(labels, pd.Series):
        raise ValueError("prediction and label must be Series")
    frame = pd.to_numeric(prediction, errors="coerce").rename("score").to_frame()
    frame = frame.join(pd.to_numeric(labels, errors="coerce").rename("label"), how="left")
    dates = pd.to_datetime(frame.index.get_level_values("datetime"), errors="coerce").normalize()
    frame["benchmark_return"] = pd.Series(dates, index=frame.index).map(benchmark)
    frame["market_state"] = pd.Series(dates, index=frame.index).map(market_states)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    if frame["benchmark_return"].isna().any():
        raise ValueError("benchmark return is missing for one or more prediction dates")
    if frame["market_state"].isna().any():
        raise ValueError("market state is missing for one or more prediction dates")
    if frame.empty:
        raise ValueError("prediction contains no finite scores")
    return frame


def _load_market_states(
    data_api: Any,
    *,
    market: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    instruments = data_api.instruments(market=market)
    close_frame = data_api.features(
        instruments,
        ["$close"],
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    if close_frame.empty:
        raise ValueError(f"Qlib market universe {market!r} returned no close data")
    close_panel = close_frame.iloc[:, 0].rename("close").reset_index()
    close_panel = close_panel.rename(
        columns={"datetime": "date", "instrument": "symbol"}
    )
    market_states = build_market_regimes(close_panel)
    if market_states.empty:
        raise ValueError(f"Qlib market universe {market!r} produced no market states")
    return market_states


class QlibJobRunner:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        provider: Path,
        quant_data_dir: Path,
        run_dir: Path,
        context_hash: str,
        mlflow_enabled: bool,
    ) -> None:
        import qlib

        qlib.init(provider_uri=str(provider), region="cn", kernels=8)
        self.config = config
        self.provider = provider
        self.quant_data_dir = quant_data_dir
        self.run_dir = run_dir
        self.context_hash = context_hash
        self.mlflow_enabled = mlflow_enabled
        self.calendar = _calendar(provider)
        self._cached_key: tuple[str, str, int] | None = None
        self._cached_dataset: Any = None
        self._cached_benchmark: pd.Series | None = None
        self._cached_market_states: pd.Series | None = None

    def _signal(self, signal_id: str) -> Mapping[str, Any]:
        return next(item for item in self.config["signals"] if item["id"] == signal_id)

    def _model(self, model_id: str) -> Any:
        entries = self.config["models"]["screening"] + self.config["models"]["validation"]
        entry = next(item for item in entries if item["id"] == model_id)
        if entry["class"] == "EqualWeightRankModel":
            return EqualWeightRankModel(**dict(entry.get("kwargs") or {}))
        from qlib.utils import init_instance_by_config

        return init_instance_by_config(
            {
                "class": entry["class"],
                "module_path": entry["module_path"],
                "kwargs": dict(entry.get("kwargs") or {}),
            }
        )

    def _dataset(
        self, job: SignalJob
    ) -> tuple[Any, pd.Series, pd.Series, RollingSplit]:
        key = (job.signal_id, job.quarter, job.window_days)
        signal = self._signal(job.signal_id)
        split = make_rolling_split(
            self.calendar,
            job.quarter,
            train_days=job.window_days,
            validation_days=int(self.config["periods"]["validation_days"]),
            label_horizon_days=int(signal["horizon_days"]),
        )
        if split.test_end >= LOCKED_CONFIRMATION_START:
            raise ValueError("Goal 1 dataset attempted to read 2026")
        if key == self._cached_key:
            return (
                self._cached_dataset,
                self._cached_benchmark.copy(),
                self._cached_market_states.copy(),
                split,
            )
        from qlib.data import D
        from qlib.utils import init_instance_by_config

        expression = label_expression(int(signal["horizon_days"]))
        benchmark_frame = D.features(
            [str(self.config["evaluation"]["benchmark"])],
            [expression],
            start_time=str(split.train_start.date()),
            end_time=str(split.test_end.date()),
            freq="day",
        )
        benchmark = benchmark_frame.iloc[:, 0].copy()
        benchmark.index = pd.to_datetime(
            benchmark.index.get_level_values("datetime"), errors="coerce"
        ).normalize()
        market_states = _load_market_states(
            D,
            market=str(self.config["evaluation"]["market"]),
            start_date=str(
                _lookback_start(
                    self.calendar,
                    split.train_start,
                    int(self.config["evaluation"]["market_state_lookback_days"]),
                ).date()
            ),
            end_date=str(split.test_end.date()),
        )
        market_states = market_states.loc[
            (market_states.index >= split.train_start)
            & (market_states.index <= split.test_end)
        ]
        if market_states.empty:
            raise ValueError("market states are missing for signal dataset dates")
        family = self.config["feature_families"][str(signal["feature_family"])]
        dataset_config = build_signal_dataset_config(
            split=split,
            signal=signal,
            feature_prefixes=family.get("prefixes") or [],
            industry_intervals_path=self.quant_data_dir
            / str(self.config["paths"]["industry_intervals"]),
            benchmark_returns={str(index.date()): float(value) for index, value in benchmark.items()},
            market_states={
                str(index.date()): str(value)
                for index, value in market_states.items()
            },
        )
        dataset = init_instance_by_config(dataset_config)
        self._cached_key = key
        self._cached_dataset = dataset
        self._cached_benchmark = benchmark
        self._cached_market_states = market_states
        return dataset, benchmark.copy(), market_states.copy(), split

    @contextmanager
    def _mlflow_context(self, job: SignalJob):
        if not self.mlflow_enabled:
            yield None
            return
        from qlib.workflow import R

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        with R.start(
            experiment_name=str(self.config["artifacts"]["experiment_name"]),
            recorder_name=f"{job.key}-{uuid.uuid4().hex[:8]}",
            uri=f"file://{mlruns_root()}",
        ):
            yield R.get_recorder()

    def __call__(self, job: SignalJob) -> Mapping[str, Any]:
        dataset, benchmark, market_states, split = self._dataset(job)
        model = self._model(job.model_id)
        prediction_path = self.run_dir / job.phase / "predictions" / f"{job.key}.parquet"
        model_path = self.run_dir / job.phase / "models" / f"{job.key}.pkl"
        diagnostics_path = (
            self.run_dir
            / job.phase
            / "diagnostics"
            / f"{job.signal_id}__{job.quarter}__w{job.window_days}.json"
        )
        if not diagnostics_path.is_file():
            raw_features = dataset.prepare_raw_features("test")
            features = dataset.prepare("test", col_set="feature", data_key="infer")
            diagnostic_kwargs = {
                "correlation_threshold": float(
                    self.config["evaluation"]["high_correlation_threshold"]
                ),
                "low_variance_threshold": float(
                    self.config["evaluation"]["low_variance_threshold"]
                ),
            }
            diagnostics = {
                "schema_version": 2,
                "raw": feature_diagnostics(raw_features, **diagnostic_kwargs),
                "processed": feature_diagnostics(features, **diagnostic_kwargs),
            }
            _write_json(diagnostics_path, diagnostics)
        with self._mlflow_context(job) as active_run:
            model.fit(dataset)
            frame = _prediction_frame(model, dataset, benchmark, market_states)
            saved = frame.reset_index().loc[
                :,
                [
                    "datetime",
                    "instrument",
                    "score",
                    "label",
                    "benchmark_return",
                    "market_state",
                ],
            ]
            _atomic_parquet(saved, prediction_path)
            _atomic_pickle(model, model_path)
            rank_ics = quarterly_rank_ic(frame)
            valid_rank_ics = [value for value in rank_ics.values() if value is not None]
            mlflow_run_id = None
            if active_run is not None:
                active_run.log_params(
                    **{
                        "phase": job.phase,
                        "signal_id": job.signal_id,
                        "model_id": job.model_id,
                        "quarter": job.quarter,
                        "window_days": job.window_days,
                        "context_hash": self.context_hash,
                    }
                )
                if valid_rank_ics:
                    active_run.log_metrics(rank_ic_mean=float(np.mean(valid_rank_ics)))
                active_run.log_artifact(str(prediction_path), artifact_path="predictions")
                active_run.log_artifact(str(model_path), artifact_path="models")
                mlflow_run_id = str(active_run.id)
        return {
            "prediction_path": str(prediction_path),
            "model_path": str(model_path),
            "diagnostics_path": str(diagnostics_path),
            "rows": int(len(saved)),
            "rank_ic_mean": float(np.mean(valid_rank_ics)) if valid_rank_ics else None,
            "mlflow_run_id": mlflow_run_id,
            "split": {
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "validation_start": str(split.validation_start.date()),
                "validation_end": str(split.validation_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
            },
        }


def _load_prediction(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "datetime",
        "instrument",
        "score",
        "label",
        "benchmark_return",
        "market_state",
    }
    if not required.issubset(frame.columns):
        raise ValueError("prediction artifact is missing required columns")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.normalize()
    if frame["datetime"].isna().any():
        raise ValueError("prediction artifact contains invalid dates")
    if (frame["datetime"] >= LOCKED_CONFIRMATION_START).any():
        raise ValueError("Goal 1 prediction artifact attempted to expose 2026 data")
    frame["instrument"] = frame["instrument"].astype(str)
    return frame.set_index(["datetime", "instrument"]).sort_index()


def _aggregate_signal_prediction_frames(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one signal prediction frame is required")
    aggregated = aggregate_prediction_frames(
        {name: frame.loc[:, ["score", "label"]] for name, frame in frames.items()}
    )
    first_name = next(iter(frames))
    first = frames[first_name].sort_index()
    for column in ("benchmark_return", "market_state"):
        if column not in first.columns:
            raise ValueError(f"signal prediction frame requires {column}")
        for name, raw in frames.items():
            current = raw.sort_index()
            if not current.index.equals(first.index):
                raise ValueError("prediction indexes differ across signal windows")
            if column not in current.columns:
                raise ValueError(f"signal prediction frame {name} requires {column}")
            if column == "benchmark_return":
                matches = np.allclose(
                    pd.to_numeric(current[column], errors="coerce").to_numpy(),
                    pd.to_numeric(first[column], errors="coerce").to_numpy(),
                    equal_nan=True,
                )
            else:
                matches = current[column].astype("string").equals(
                    first[column].astype("string")
                )
            if not matches:
                raise ValueError(f"{column} values differ across signal windows")
        aggregated[column] = first[column]
    return aggregated


def _board_attribution(frame: pd.DataFrame, *, topk: int) -> dict[str, Any]:
    data = frame.dropna(subset=["score", "label"]).copy()
    instruments = data.index.get_level_values("instrument")
    data["board"] = [board_from_symbol(value) for value in instruments]
    selected = data.sort_values("score", ascending=False).groupby(
        level="datetime", sort=True
    ).head(topk)
    return {
        str(board): {
            "selected_rows": int(len(group)),
            "mean_label": float(group["label"].mean()),
        }
        for board, group in selected.groupby("board", sort=True)
    }


def _summarize_factor_quality(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def summarize_block(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        diagnostics: list[dict[str, Any]] = []
        required = {
            "feature_count",
            "missing_rate_by_feature",
            "constant_features",
            "low_variance_features",
            "nonfinite_value_count",
            "high_correlation_pairs",
        }
        for payload in payloads:
            if not required.issubset(payload):
                raise ValueError("factor diagnostics are incomplete or invalid")
            if not isinstance(payload["missing_rate_by_feature"], dict) or not all(
                isinstance(payload[key], list)
                for key in (
                    "constant_features",
                    "low_variance_features",
                    "high_correlation_pairs",
                )
            ):
                raise ValueError("factor diagnostics are incomplete or invalid")
            try:
                feature_count = int(payload["feature_count"])
                nonfinite_count = int(payload["nonfinite_value_count"])
                missing_rates = [
                    float(value)
                    for value in payload["missing_rate_by_feature"].values()
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError("factor diagnostics are incomplete or invalid") from exc
            if (
                feature_count < 0
                or nonfinite_count < 0
                or any(
                    not math.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in missing_rates
                )
            ):
                raise ValueError("factor diagnostics are incomplete or invalid")
            diagnostics.append(
                {
                    "feature_count": feature_count,
                    "max_missing_rate": max(missing_rates, default=0.0),
                    "constant_features": [
                        str(value) for value in payload["constant_features"]
                    ],
                    "low_variance_features": [
                        str(value) for value in payload["low_variance_features"]
                    ],
                    "nonfinite_value_count": nonfinite_count,
                    "high_correlation_pair_count": len(
                        payload["high_correlation_pairs"]
                    ),
                }
            )
        if not diagnostics:
            raise ValueError("factor diagnostics are required for signal evaluation")
        return {
            "diagnostics_count": len(diagnostics),
            "feature_count_min": min(item["feature_count"] for item in diagnostics),
            "feature_count_max": max(item["feature_count"] for item in diagnostics),
            "max_missing_rate": max(item["max_missing_rate"] for item in diagnostics),
            "constant_features": sorted(
                {
                    feature
                    for item in diagnostics
                    for feature in item["constant_features"]
                }
            ),
            "constant_feature_count_max": max(
                len(item["constant_features"]) for item in diagnostics
            ),
            "low_variance_features": sorted(
                {
                    feature
                    for item in diagnostics
                    for feature in item["low_variance_features"]
                }
            ),
            "low_variance_feature_count_max": max(
                len(item["low_variance_features"]) for item in diagnostics
            ),
            "nonfinite_value_count_total": sum(
                item["nonfinite_value_count"] for item in diagnostics
            ),
            "high_correlation_pair_count_max": max(
                item["high_correlation_pair_count"] for item in diagnostics
            ),
        }

    raw_blocks: list[Mapping[str, Any]] = []
    processed_blocks: list[Mapping[str, Any]] = []
    for record in records:
        path = Path(str(record.get("diagnostics_path") or ""))
        if not path.is_file():
            raise ValueError("completed signal record is missing factor diagnostics")
        payload = _read_json(path)
        if payload.get("schema_version") != 2:
            raise ValueError("factor diagnostics are incomplete or invalid")
        raw = payload.get("raw")
        processed = payload.get("processed")
        if not isinstance(raw, dict) or not isinstance(processed, dict):
            raise ValueError("factor diagnostics are incomplete or invalid")
        raw_blocks.append(raw)
        processed_blocks.append(processed)
    raw_summary = summarize_block(raw_blocks)
    return {**raw_summary, "processed": summarize_block(processed_blocks)}


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    phase: str,
) -> list[dict[str, Any]]:
    signal_map = {str(item["id"]): item for item in config["signals"]}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["signal_id"]), str(record["model_id"])), []).append(
            record
        )
    reports: list[dict[str, Any]] = []
    topk = int(config["evaluation"]["primary_topk"])
    for (signal_id, model_id), items in sorted(grouped.items()):
        signal = signal_map[signal_id]
        factor_quality = _summarize_factor_quality(items)
        quarter_windows: dict[str, dict[int, pd.DataFrame]] = {}
        window_frames: dict[int, list[pd.DataFrame]] = {}
        for item in items:
            frame = _load_prediction(str(item["prediction_path"]))
            quarter = str(item["quarter"])
            window = int(item["window_days"])
            quarter_windows.setdefault(quarter, {})[window] = frame
            window_frames.setdefault(window, []).append(frame)
        windows = {
            window: pd.concat(frames).sort_index()
            for window, frames in window_frames.items()
        }
        quarter_aggregates = {
            quarter: _aggregate_signal_prediction_frames(
                {str(window): frame for window, frame in frames.items()}
            )
            for quarter, frames in quarter_windows.items()
        }
        combined = pd.concat(quarter_aggregates.values()).sort_index()
        rank_ic = quarterly_rank_ic(combined)
        rank_values = [value for value in rank_ic.values() if value is not None]
        monotonic = decile_monotonicity(
            combined,
            deciles=int(config["evaluation"]["deciles"]),
            spearman_min=float(config["evaluation"]["monotonic_spearman_min"]),
        )
        returns = horizon_aligned_excess(
            combined,
            topk=topk,
            horizon_days=int(signal["horizon_days"]),
            total_cost_bps=float(config["evaluation"]["total_cost_bps"]),
            labels_are_excess=str(signal["target"]) == "csi800_excess",
        )
        stability = (
            window_stability(
                {window: frame.loc[:, ["score", "label"]] for window, frame in windows.items()},
                topk=topk,
            )
            if len(windows) > 1
            else {
                "rank_correlation_median": None,
                "topk_overlap_mean": None,
                "pair_count": 0,
            }
        )
        values = {
            "positive_rank_ic_quarter_ratio": (
                sum(value > 0 for value in rank_values) / len(rank_ic) if rank_ic else 0.0
            ),
            "median_quarter_rank_ic": (
                float(np.median(rank_values)) if rank_values else None
            ),
            "window_rank_correlation_median": stability.get(
                "rank_correlation_median"
            ),
            "top50_overlap_mean": stability.get("topk_overlap_mean"),
            "positive_excess_quarter_ratio": returns["positive_quarter_ratio"],
            "annual_excess_return": returns["annual_excess_return"],
            "monotonic_quarter_ratio": monotonic["quarter_ratio"],
        }
        if phase == "screening":
            thresholds = config["screening"]
            screening_gates = {
                "positive_rank_ic_quarters": values["positive_rank_ic_quarter_ratio"]
                >= float(thresholds["positive_rank_ic_quarter_ratio_min"]),
                "median_quarter_rank_ic": values["median_quarter_rank_ic"] is not None
                and values["median_quarter_rank_ic"]
                >= float(thresholds["median_quarter_rank_ic_min"]),
                "monotonic_quarters": values["monotonic_quarter_ratio"]
                >= float(thresholds["monotonic_quarter_ratio_min"]),
            }
            verdict = {
                "passed": all(screening_gates.values()),
                "gates": screening_gates,
                "failed_gates": [
                    key for key, passed in screening_gates.items() if not passed
                ],
            }
        else:
            verdict = acceptance_verdict(values, config["acceptance"])
        reports.append(
            {
                "phase": phase,
                "signal_id": signal_id,
                "model_id": model_id,
                "passed": verdict["passed"],
                "failed_gates": verdict["failed_gates"],
                "values": values,
                "quarter_rank_ic": rank_ic,
                "window_stability": stability,
                "topk_sensitivity": topk_sensitivity(
                    combined,
                    topks=config["evaluation"]["sensitivity_topk"],
                    primary=topk,
                ),
                "prediction_dispersion": prediction_dispersion(combined, topk=topk),
                "factor_quality": factor_quality,
                "monotonicity": monotonic,
                "returns_after_20bps": returns,
                "board_attribution": _board_attribution(combined, topk=topk),
                "market_state_attribution": market_state_attribution(
                    combined,
                    topk=topk,
                    horizon_days=int(signal["horizon_days"]),
                    total_cost_bps=float(config["evaluation"]["total_cost_bps"]),
                    labels_are_excess=str(signal["target"]) == "csi800_excess",
                ),
            }
        )
    return reports


def select_screening_candidates(
    reports: Sequence[Mapping[str, Any]], *, maximum: int
) -> list[str]:
    by_signal: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        by_signal.setdefault(str(report["signal_id"]), []).append(report)
    ranked: list[tuple[float, str]] = []
    for signal_id, candidates in by_signal.items():
        passed = [item for item in candidates if item.get("passed") is True]
        if not passed:
            continue
        best = max(
            passed,
            key=lambda item: float(item["values"].get("median_quarter_rank_ic") or -math.inf),
        )
        score = float(best["values"].get("median_quarter_rank_ic") or -math.inf)
        ranked.append((score, signal_id))
    ranked.sort(reverse=True)
    return [signal_id for _, signal_id in ranked[: int(maximum)]]


def _factor_contract(
    *,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    signal = next(
        item for item in config["signals"] if item["id"] == report["signal_id"]
    )
    family = config["feature_families"][signal["feature_family"]]
    contract = {
        "schema_version": 1,
        "status": "signal_candidate_ready",
        "signal_id": signal["id"],
        "model_id": report["model_id"],
        "feature_family": signal["feature_family"],
        "feature_prefixes": list(family.get("prefixes") or []),
        "base_handler": "Alpha158",
        "raw_fields": ["open", "high", "low", "close", "volume", "factor"],
        "horizon_days": int(signal["horizon_days"]),
        "label_expression": label_expression(int(signal["horizon_days"])),
        "signal_cutoff": "T_close",
        "execution_start": "T_plus_1_open",
        "normalization_scope": signal["normalization_scope"],
        "target": signal["target"],
        "data_source": "verified_local_qlib_provider",
        "provider_tree_sha256": context.get("provider_tree_sha256"),
        "membership_snapshot_hash": context.get("membership_snapshot_hash"),
        "config_hash": context["config_hash"],
        "context_hash": context["context_hash"],
        "git_commit": context["git_commit"],
        "offline_daily_parity_required": True,
        "broker_adapter_contract": "target_positions_and_order_intents_only",
        "acceptance": dict(report["values"]),
    }
    contract["contract_hash"] = _sha256_bytes(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    return contract


def write_immutable_factor_contract(path: Path, contract: Mapping[str, Any]) -> None:
    existing = _read_json(path) if path.exists() else None
    if existing is not None and existing != dict(contract):
        raise ValueError("immutable_factor_contract_mismatch")
    if existing is None:
        _write_json(path, contract)


def _records_for_phase(progress: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in (progress.get("jobs") or {}).values()
        if value.get("status") == "completed" and value.get("phase") == phase
    ]


def _status_payload(
    *,
    run_id: str,
    context: Mapping[str, Any],
    lifecycle_status: str,
    status: str | None,
    current_stage: str,
    pause_state: str,
    completed_count: int,
    expected_count: int,
    reason_code: str | None = None,
    next_gate: str | None = None,
) -> dict[str, Any]:
    if status is not None and status not in FINAL_STATUSES:
        raise ValueError("invalid signal research status")
    return {
        "run_id": run_id,
        "status": status,
        "lifecycle_status": lifecycle_status,
        "current_stage": current_stage,
        "pause_state": pause_state,
        "reason_code": reason_code,
        "next_gate": next_gate,
        "completed_count": int(completed_count),
        "expected_count": int(expected_count),
        "config_hash": context.get("config_hash"),
        "git_commit": context.get("git_commit"),
        "data_manifest_hash": context.get("data_manifest_hash"),
        "context_hash": context.get("context_hash"),
        "provider_tree_sha256": context.get("provider_tree_sha256"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _write_failed_status(
    *,
    run_id: str,
    context: Mapping[str, Any],
    report_dir: Path,
    current_stage: str,
    completed_count: int,
    expected_count: int,
    error: BaseException,
) -> dict[str, Any]:
    status = _status_payload(
        run_id=run_id,
        context=context,
        lifecycle_status="failed",
        status=None,
        current_stage=current_stage,
        pause_state="failed",
        completed_count=completed_count,
        expected_count=expected_count,
        reason_code="job_failed",
        next_gate="inspect_failed_job",
    )
    status["error_type"] = type(error).__name__
    status["error"] = str(error)[:500]
    _write_json(report_dir / "runs" / run_id / "status.json", status)
    _write_json(report_dir / "robust_signal_status_latest.json", status)
    return status


def run_research(
    *,
    config_path: Path,
    quant_data_dir: Path,
    report_dir: Path,
    run_id: str,
    resume: bool,
    smoke: bool,
    smoke_quarter: str,
    smoke_signal_id: str,
    mlflow_enabled: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    report_root = reports_root()
    context = build_context(
        config_path=config_path,
        config=config,
        quant_data_dir=quant_data_dir,
        report_root=report_root,
    )
    if not smoke and context["git_clean"] is not True:
        raise ValueError("formal signal research requires a clean Git worktree")
    run_dir = report_dir / "runs" / run_id
    context_path = run_dir / "context.json"
    if resume:
        stored = _read_json(context_path)
        if stored.get("context_hash") != context["context_hash"]:
            raise ValueError("resume_context_mismatch")
        (run_dir / "pause.request.json").unlink(missing_ok=True)
    else:
        if context_path.exists():
            raise ValueError("signal research run already exists; use resume")
        _write_json(context_path, context)
    provider = Path(context["provider_uri"])
    job_runner = QlibJobRunner(
        config=config,
        provider=provider,
        quant_data_dir=quant_data_dir,
        run_dir=run_dir,
        context_hash=str(context["context_hash"]),
        mlflow_enabled=mlflow_enabled,
    )
    pause_path = run_dir / "pause.request.json"

    def paused() -> bool:
        return pause_path.is_file()

    screening_jobs = build_jobs(
        config,
        phase="screening",
        signal_ids=[smoke_signal_id] if smoke else None,
        quarters=[smoke_quarter] if smoke else None,
        model_ids=["equal_weight_rank"] if smoke else None,
    )
    status = _status_payload(
        run_id=run_id,
        context=context,
        lifecycle_status="running",
        status=None,
        current_stage="screening",
        pause_state="running",
        completed_count=0,
        expected_count=len(screening_jobs),
        next_gate="screening_diagnostics",
    )
    _write_json(run_dir / "status.json", status)
    _write_json(report_dir / "robust_signal_status_latest.json", status)
    try:
        screening_progress = run_job_loop(
            screening_jobs,
            progress_path=run_dir / "screening" / "progress.json",
            context_hash=str(context["context_hash"]),
            resume=resume,
            run_job=job_runner,
            pause_requested=paused,
        )
    except Exception as exc:
        progress = _read_json(run_dir / "screening" / "progress.json")
        _write_failed_status(
            run_id=run_id,
            context=context,
            report_dir=report_dir,
            current_stage="screening",
            completed_count=sum(
                value.get("status") == "completed"
                for value in (progress.get("jobs") or {}).values()
            ),
            expected_count=len(screening_jobs),
            error=exc,
        )
        raise
    if screening_progress.get("pause_state") == "paused":
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="stopped",
            status=None,
            current_stage="screening",
            pause_state="paused",
            completed_count=int(screening_progress.get("completed_count") or 0),
            expected_count=len(screening_jobs),
            reason_code="user_requested_after_current_job",
            next_gate="resume_screening",
        )
        _write_json(run_dir / "status.json", status)
        _write_json(report_dir / "robust_signal_status_latest.json", status)
        return status
    screening_reports = evaluate_records(
        _records_for_phase(screening_progress, "screening"),
        config=config,
        phase="screening",
    )
    _write_json(run_dir / "screening" / "screening_report.json", {"reports": screening_reports})
    if smoke:
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="completed",
            status="signal_diagnosis_complete",
            current_stage="smoke_complete",
            pause_state="completed",
            completed_count=len(screening_jobs),
            expected_count=len(screening_jobs),
            reason_code="single_quarter_smoke_complete",
            next_gate="formal_screening",
        )
        _write_json(run_dir / "status.json", status)
        _write_json(report_dir / "robust_signal_status_latest.json", status)
        return status
    candidates = select_screening_candidates(
        screening_reports, maximum=int(config["screening"]["maximum_candidates"])
    )
    if not candidates:
        result = {
            "status": "signal_diagnosis_complete",
            "reason_code": "price_volume_screening_failed",
            "screening_candidates": [],
            "next_stage": "historical_fundamental_data_required",
            "reports": screening_reports,
        }
        _write_json(run_dir / "result.json", result)
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="completed",
            status="signal_diagnosis_complete",
            current_stage="price_volume_complete",
            pause_state="completed",
            completed_count=len(screening_jobs),
            expected_count=len(screening_jobs),
            reason_code="price_volume_screening_failed",
            next_gate="build_historical_fundamentals",
        )
        _write_json(run_dir / "status.json", status)
        _write_json(report_dir / "robust_signal_status_latest.json", status)
        return status
    validation_jobs = build_jobs(config, phase="validation", signal_ids=candidates)
    validation_status = _status_payload(
        run_id=run_id,
        context=context,
        lifecycle_status="running",
        status="signal_diagnosis_complete",
        current_stage="validation",
        pause_state="running",
        completed_count=len(screening_jobs),
        expected_count=len(screening_jobs) + len(validation_jobs),
        next_gate="strict_signal_acceptance",
    )
    _write_json(run_dir / "status.json", validation_status)
    _write_json(report_dir / "robust_signal_status_latest.json", validation_status)
    try:
        validation_progress = run_job_loop(
            validation_jobs,
            progress_path=run_dir / "validation" / "progress.json",
            context_hash=str(context["context_hash"]),
            resume=resume,
            run_job=job_runner,
            pause_requested=paused,
        )
    except Exception as exc:
        progress = _read_json(run_dir / "validation" / "progress.json")
        _write_failed_status(
            run_id=run_id,
            context=context,
            report_dir=report_dir,
            current_stage="validation",
            completed_count=len(screening_jobs)
            + sum(
                value.get("status") == "completed"
                for value in (progress.get("jobs") or {}).values()
            ),
            expected_count=len(screening_jobs) + len(validation_jobs),
            error=exc,
        )
        raise
    if validation_progress.get("pause_state") == "paused":
        completed = len(screening_jobs) + int(validation_progress.get("completed_count") or 0)
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="stopped",
            status="signal_diagnosis_complete",
            current_stage="validation",
            pause_state="paused",
            completed_count=completed,
            expected_count=len(screening_jobs) + len(validation_jobs),
            reason_code="user_requested_after_current_job",
            next_gate="resume_validation",
        )
        _write_json(run_dir / "status.json", status)
        _write_json(report_dir / "robust_signal_status_latest.json", status)
        return status
    validation_reports = evaluate_records(
        _records_for_phase(validation_progress, "validation"),
        config=config,
        phase="validation",
    )
    _write_json(
        run_dir / "validation" / "validation_report.json",
        {"reports": validation_reports},
    )
    passed = [report for report in validation_reports if report.get("passed") is True]
    total_jobs = len(screening_jobs) + len(validation_jobs)
    if not passed:
        result = {
            "status": "signal_diagnosis_complete",
            "reason_code": "price_volume_validation_failed",
            "screening_candidates": candidates,
            "next_stage": "historical_fundamental_data_required",
            "reports": validation_reports,
        }
        _write_json(run_dir / "result.json", result)
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="completed",
            status="signal_diagnosis_complete",
            current_stage="price_volume_complete",
            pause_state="completed",
            completed_count=total_jobs,
            expected_count=total_jobs,
            reason_code="price_volume_validation_failed",
            next_gate="build_historical_fundamentals",
        )
    else:
        winner = max(
            passed,
            key=lambda report: (
                float(report["values"].get("annual_excess_return") or -math.inf),
                float(report["values"].get("median_quarter_rank_ic") or -math.inf),
            ),
        )
        contract = _factor_contract(config=config, context=context, report=winner)
        contract_path = run_dir / "factor_contract.json"
        write_immutable_factor_contract(contract_path, contract)
        result = {
            "status": "signal_candidate_ready",
            "screening_candidates": candidates,
            "selected_signal_id": winner["signal_id"],
            "selected_model_id": winner["model_id"],
            "factor_contract": str(contract_path),
            "challengers": "eligible_but_not_required_for_deterministic_signal_proof",
            "reports": validation_reports,
        }
        _write_json(run_dir / "result.json", result)
        _write_json(report_dir / "factor_contract_latest.json", contract)
        status = _status_payload(
            run_id=run_id,
            context=context,
            lifecycle_status="completed",
            status="signal_candidate_ready",
            current_stage="completed",
            pause_state="completed",
            completed_count=total_jobs,
            expected_count=total_jobs,
            next_gate="create_goal_2_trade_strategy",
        )
    _write_json(run_dir / "status.json", status)
    _write_json(report_dir / "robust_signal_status_latest.json", status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-quarter", default="2023Q1")
    parser.add_argument("--smoke-signal-id", default="trend_momentum_excess_5d")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if args.request_pause:
        payload = request_pause(report_dir, args.run_id)
    else:
        payload = run_research(
            config_path=args.config.expanduser().resolve(),
            quant_data_dir=args.quant_data_dir.expanduser().resolve(),
            report_dir=report_dir,
            run_id=args.run_id,
            resume=args.resume,
            smoke=args.smoke,
            smoke_quarter=args.smoke_quarter,
            smoke_signal_id=args.smoke_signal_id,
            mlflow_enabled=not args.no_mlflow,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
