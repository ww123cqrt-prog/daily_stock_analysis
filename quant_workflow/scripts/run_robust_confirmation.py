#!/usr/bin/env python3
"""Run the single-use 2026 confirmation for a frozen robust candidate."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from robust_backtest import simulate_strategy  # noqa: E402
from robust_evaluation import (  # noqa: E402
    aggregate_prediction_frames,
    aggregate_training_artifacts,
    confirmation_verdict,
)
from robust_market_data import (  # noqa: E402
    build_market_regimes,
    load_benchmark_frame,
    load_market_panel,
)
from robust_strategy_core import make_confirmation_split  # noqa: E402
from prepare_robust_provider import runtime_provider_identity  # noqa: E402
from run_robust_development import (  # noqa: E402
    NEURAL_MODELS,
    TREE_MODELS,
    TrainingJob,
    _atomic_write_parquet,
    _artifact_records,
    _split_record,
    build_training_jobs,
    load_config,
    make_job_split,
    completed_job_artifact_is_valid,
    collect_runtime_environment,
    run_training_jobs,
    frozen_context_hash,
    launch_runtime_evidence,
    payload_sha256,
    selection_contract_hash,
)
from run_robust_strategy import DEFAULT_CONFIG, file_sha256, git_worktree_state  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@contextmanager
def _claim_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".confirmation.lock"
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_confirmation_jobs(
    config: Mapping[str, Any], frozen: Mapping[str, Any]
) -> list[TrainingJob]:
    selected = [str(value) for value in frozen.get("selected_models") or []]
    if len(selected) < int(config["training"]["ensemble_models_min"]):
        raise ValueError("frozen candidate does not contain enough selected models")
    known = {str(item["id"]) for item in config["models"]}
    if any(model_id not in known for model_id in selected):
        raise ValueError("frozen candidate contains an unknown model")
    jobs: list[TrainingJob] = []
    for window in config["training"]["windows_trading_days"]:
        for model_id in selected:
            if model_id in TREE_MODELS:
                seeds: Sequence[int | None] = config["training"]["tree_seeds"]
            elif model_id in NEURAL_MODELS:
                seeds = config["training"].get(
                    "neural_seeds", config["training"]["tree_seeds"]
                )
            else:
                seeds = [None]
            for seed in seeds:
                jobs.append(
                    TrainingJob(
                        "confirmation_2026",
                        int(window),
                        model_id,
                        None if seed is None else int(seed),
                    )
                )
    return jobs


def _frozen_config(
    config: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    frozen_models = frozen.get("selected_model_configs") or {}
    selected = set(str(value) for value in frozen.get("selected_models") or [])
    result["models"] = [
        item for item in result["models"] if str(item["id"]) in selected
    ]
    for model in result["models"]:
        stored = frozen_models.get(str(model["id"]))
        if not isinstance(stored, dict):
            continue
        model["class"] = stored["class"]
        model["module_path"] = stored["module_path"]
        model["kwargs"] = copy.deepcopy(stored.get("kwargs") or {})
    return result


def claim_confirmation(
    frozen_path: Path, state_path: Path, *, run_id: str, resume: bool
) -> dict[str, Any]:
    frozen_path = Path(frozen_path)
    state_path = Path(state_path)
    with _claim_lock(frozen_path.parent):
        frozen = _read_json(frozen_path)
        if frozen.get("status") != "research_candidate_ready":
            raise RuntimeError("frozen research candidate is missing")
        state = _read_json(state_path)
        if state:
            if (
                resume
                and state.get("run_id") == run_id
                and state.get("lifecycle_status") == "stopped"
            ):
                state["lifecycle_status"] = "running"
                state["resumed_at"] = (
                    datetime.now().astimezone().isoformat(timespec="seconds")
                )
                _write_json(state_path, state)
                return state
            raise RuntimeError("2026 confirmation was already claimed")
        if frozen.get("confirmation_run_started") is True:
            if resume and frozen.get("confirmation_run_id") == run_id:
                now = datetime.now().astimezone().isoformat(timespec="seconds")
                state = {
                    "status": "research_candidate_ready",
                    "lifecycle_status": "running",
                    "run_id": run_id,
                    "started_at": frozen.get("confirmation_started_at") or now,
                    "resumed_at": now,
                }
                _write_json(state_path, state)
                return state
            raise RuntimeError("2026 confirmation was already started")
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        state = {
            "status": "research_candidate_ready",
            "lifecycle_status": "running",
            "run_id": run_id,
            "started_at": now,
        }
        frozen["confirmation_run_started"] = True
        frozen["confirmation_run_id"] = run_id
        frozen["confirmation_started_at"] = now
        _write_json(frozen_path, frozen)
        _write_json(state_path, state)
        return state


def _failure(reason_code: str, *, run_id: str) -> dict[str, Any]:
    return {
        "status": "development_failed",
        "reason_code": reason_code,
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _current_membership_snapshot_hash(
    config: Mapping[str, Any], quant_data_dir: Path
) -> str:
    pit_dir = quant_data_dir / str(config["paths"]["pit_membership"])
    with membership_read_lock(pit_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        snapshot.validate_source(
            start_date=str(config["universe"]["start_date"]),
            end_date="2025-12-31",
        )
        return snapshot.fingerprint()


def _current_reference_snapshot_hash(
    config: Mapping[str, Any],
    quant_data_dir: Path,
    *,
    membership_snapshot_hash: str,
) -> str:
    reference_dir = (
        quant_data_dir / Path(str(config["paths"]["security_master"])).parent
    )
    with reference_read_lock(reference_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        snapshot.validate_source(
            start_date=str(config["universe"]["start_date"]),
            end_date="2025-12-31",
            expected_membership_snapshot_hash=membership_snapshot_hash,
        )
        return snapshot.fingerprint()


def _confirmation_model_artifacts(
    jobs: Sequence[TrainingJob], progress: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "job_key": job.key,
            "model_id": job.model_id,
            "window_days": job.window_days,
            "seed": job.seed,
            "model_path": str(Path(str(progress[job.key]["model_path"])).resolve()),
            "model_sha256": str(progress[job.key]["model_sha256"]),
            "split": copy.deepcopy(dict(progress[job.key]["split"])),
        }
        for job in sorted(jobs, key=lambda item: item.key)
    ]


def _invalid_confirmation_artifacts(
    jobs: Sequence[TrainingJob],
    progress: Mapping[str, Mapping[str, Any]],
    split_factory: Any,
) -> list[str]:
    invalid: list[str] = []
    for job in jobs:
        state = progress.get(job.key) or {}
        expected_split = _split_record(split_factory(job.quarter, job.window_days))
        if (
            not completed_job_artifact_is_valid(job, state)
            or state.get("split") != expected_split
        ):
            invalid.append(job.key)
    return invalid


def run_confirmation(
    *,
    config_path: Path,
    quant_data_dir: Path,
    report_dir: Path,
    frozen_path: Path,
    provider_state_path: Path,
    run_id: str,
    resume: bool = False,
    training_runner: Any = run_training_jobs,
    aggregator: Any = aggregate_training_artifacts,
    market_loader: Any = load_market_panel,
    benchmark_loader: Any = load_benchmark_frame,
    regime_builder: Any = build_market_regimes,
    strategy_runner: Any = simulate_strategy,
    git_state_provider: Any = git_worktree_state,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    quant_data_dir = Path(quant_data_dir).resolve()
    report_dir = Path(report_dir).resolve()
    frozen_path = Path(frozen_path).resolve()
    state_path = report_dir / "confirmation_state_latest.json"
    if not frozen_path.is_file():
        payload = _failure("frozen_candidate_missing", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    config = load_config(config_path)
    frozen = _read_json(frozen_path)
    if frozen.get("status") != "research_candidate_ready" or frozen.get(
        "config_hash"
    ) != file_sha256(config_path):
        payload = _failure("frozen_candidate_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    if frozen.get("frozen_context_hash") != frozen_context_hash(frozen):
        payload = _failure("frozen_context_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    launch_evidence = frozen.get("launch_runtime_evidence")
    if (
        not isinstance(launch_evidence, Mapping)
        or frozen.get("launch_runtime_evidence_hash")
        != payload_sha256(launch_evidence)
    ):
        payload = _failure("frozen_launch_runtime_evidence_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    runtime_environment = collect_runtime_environment(config)
    frozen_runtime = frozen.get("runtime_environment")
    if (
        not isinstance(frozen_runtime, Mapping)
        or frozen_runtime.get("environment_hash")
        != frozen.get("runtime_environment_hash")
        or frozen.get("runtime_environment_hash")
        != runtime_environment["environment_hash"]
    ):
        payload = _failure("frozen_runtime_environment_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    confirmation_config = _frozen_config(config, frozen)
    try:
        confirmation_launch_evidence = launch_runtime_evidence(
            confirmation_config,
            report_dir=report_dir,
            run_id=run_id,
            runtime_environment=runtime_environment,
            runner_namespace="confirmation_runner",
        )
    except ValueError:
        payload = _failure(
            "confirmation_launch_mps_preflight_missing_or_invalid", run_id=run_id
        )
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    confirmation_launch_evidence_hash = payload_sha256(
        confirmation_launch_evidence
    )
    development_artifacts = frozen.get("development_artifacts")
    if not isinstance(development_artifacts, list) or frozen.get(
        "development_artifacts_hash"
    ) != payload_sha256(development_artifacts):
        payload = _failure("frozen_development_artifacts_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    current_git = dict(git_state_provider())
    if current_git.get("clean") is not True or current_git.get("commit") != frozen.get(
        "git_commit"
    ):
        payload = _failure("frozen_git_state_mismatch", run_id=run_id)
        payload["git_commit"] = current_git.get("commit")
        payload["git_changes"] = list(current_git.get("changes") or [])[:100]
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    expected_selection_hash = selection_contract_hash(
        [str(value) for value in frozen.get("selected_models") or []],
        frozen.get("selected_model_configs") or {},
        str(frozen.get("selected_strategy") or ""),
    )
    if frozen.get("selection_hash") != expected_selection_hash:
        payload = _failure("frozen_selection_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    try:
        membership_snapshot_hash = _current_membership_snapshot_hash(
            config, quant_data_dir
        )
    except MembershipIntegrityError:
        payload = _failure("frozen_membership_snapshot_unavailable", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    if membership_snapshot_hash != frozen.get("membership_snapshot_hash"):
        payload = _failure("frozen_membership_snapshot_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    try:
        reference_snapshot_hash = _current_reference_snapshot_hash(
            config,
            quant_data_dir,
            membership_snapshot_hash=membership_snapshot_hash,
        )
    except ReferenceIntegrityError:
        payload = _failure("frozen_reference_snapshot_unavailable", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    if reference_snapshot_hash != frozen.get("reference_snapshot_hash"):
        payload = _failure("frozen_reference_snapshot_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    manifest_path = quant_data_dir / "DATA_MANIFEST.json"
    if not manifest_path.is_file() or file_sha256(manifest_path) != frozen.get(
        "data_manifest_hash"
    ):
        payload = _failure("frozen_data_manifest_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    provider_state = _read_json(Path(provider_state_path))
    provider_uri = Path(str(provider_state.get("provider_uri") or ""))
    if provider_state.get("status") != "completed" or not provider_uri.is_dir():
        payload = _failure("provider_state_missing", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    provider_identity_matches = (
        bool(frozen.get("provider_uri"))
        and provider_uri.resolve() == Path(str(frozen["provider_uri"])).resolve()
        and bool(frozen.get("provider_key"))
        and provider_state.get("provider_key") == frozen.get("provider_key")
    )
    if not provider_identity_matches:
        payload = _failure("frozen_provider_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    provider_identity = runtime_provider_identity(
        provider_uri, str(frozen["provider_key"])
    )
    if provider_identity is None or provider_identity.get(
        "provider_tree_sha256"
    ) != frozen.get("provider_tree_sha256"):
        payload = _failure("frozen_provider_content_mismatch", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    calendar_path = provider_uri / "calendars" / "day.txt"
    if not calendar_path.is_file():
        payload = _failure("provider_calendar_missing", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            calendar_path.read_text(encoding="utf-8").splitlines(), errors="coerce"
        )
    ).dropna()
    confirmation_days = calendar[calendar >= pd.Timestamp("2026-01-01")]
    if len(confirmation_days) < 7:
        payload = _failure("confirmation_labels_unavailable", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    selected_models = set(str(value) for value in frozen.get("selected_models") or [])
    development_jobs = [
        job for job in build_training_jobs(config) if job.model_id in selected_models
    ]
    artifacts_by_key = {
        str(item.get("job_key")): item
        for item in development_artifacts
        if isinstance(item, Mapping)
    }
    expected_keys = {job.key for job in development_jobs}
    development_run_dir = Path(str(frozen.get("development_run_dir") or ""))
    invalid_development_artifacts = []
    if set(artifacts_by_key) != expected_keys:
        invalid_development_artifacts.append("development_artifact_job_set")
    else:
        for job in development_jobs:
            split = make_job_split(
                calendar,
                job.quarter,
                window_days=job.window_days,
                validation_days=int(config["rolling_validation"]["validation_days"]),
                label_horizon_days=5,
            )
            if not completed_job_artifact_is_valid(
                job,
                artifacts_by_key[job.key],
                run_dir=development_run_dir,
                expected_split=_split_record(split),
                context_hash=str(
                    artifacts_by_key[job.key].get("context_hash") or ""
                ),
            ):
                invalid_development_artifacts.append(job.key)
    if invalid_development_artifacts:
        payload = _failure("frozen_development_artifacts_mismatch", run_id=run_id)
        payload["invalid_artifact_count"] = len(invalid_development_artifacts)
        payload["invalid_artifact_sample"] = invalid_development_artifacts[:20]
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    try:
        claim_confirmation(frozen_path, state_path, run_id=run_id, resume=resume)
    except RuntimeError:
        payload = _failure("confirmation_already_consumed", run_id=run_id)
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    latest_signal_day = confirmation_days[-7]
    latest_market_day = confirmation_days[-1]
    jobs = build_confirmation_jobs(confirmation_config, frozen)

    def split_factory(_: str, window_days: int):
        return make_confirmation_split(
            calendar,
            train_days=window_days,
            validation_days=int(config["rolling_validation"]["validation_days"]),
            label_horizon_days=5,
            test_start="2026-01-01",
            test_end=str(latest_signal_day.date()),
        )

    run_dir = report_dir / "runs" / run_id / "confirmation"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = training_runner(
        confirmation_config,
        jobs,
        provider_uri=provider_uri,
        calendar=calendar,
        run_dir=run_dir,
        mlflow_enabled=True,
        resume=resume,
        split_factory=split_factory,
        phase="confirmation",
        experiment_name="robust_csi800_confirmation",
        runtime_environment=runtime_environment,
    )
    if any(progress.get(job.key, {}).get("status") != "completed" for job in jobs):
        payload = _failure("confirmation_training_incomplete", run_id=run_id)
        _write_json(state_path, {**payload, "lifecycle_status": "stopped"})
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    invalid_artifacts = _invalid_confirmation_artifacts(jobs, progress, split_factory)
    if invalid_artifacts:
        payload = _failure("confirmation_artifact_invalid", run_id=run_id)
        payload["invalid_artifact_count"] = len(invalid_artifacts)
        payload["invalid_artifact_sample"] = invalid_artifacts[:20]
        _write_json(state_path, {**payload, "lifecycle_status": "stopped"})
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    model_artifacts = _confirmation_model_artifacts(jobs, progress)
    aggregate = aggregator(
        _artifact_records(jobs, progress),
        topk=int(config["portfolio"]["topk"]),
        thresholds=config["acceptance"],
    )
    selected_ids = [str(value) for value in frozen["selected_models"]]
    if set(aggregate["model_predictions"]) != set(selected_ids):
        payload = _failure("confirmation_model_set_mismatch", run_id=run_id)
        _write_json(state_path, {**payload, "lifecycle_status": "stopped"})
        _write_json(report_dir / "confirmation_latest.json", payload)
        return payload
    ensemble = aggregate_prediction_frames(
        {
            model_id: aggregate["model_predictions"][model_id]
            for model_id in selected_ids
        }
    )
    ensemble_path = (run_dir / "confirmation_ensemble_predictions.parquet").resolve()
    _atomic_write_parquet(
        ensemble.reset_index().loc[:, ["datetime", "instrument", "score", "label"]],
        ensemble_path,
    )
    ensemble_hash = file_sha256(ensemble_path)
    paths = config["paths"]
    market = market_loader(
        pit_dir=quant_data_dir / paths["pit_membership"],
        price_dir=quant_data_dir / paths["price_parquet"],
        reference_dir=quant_data_dir / Path(paths["security_master"]).parent,
        start_date="2026-01-01",
        end_date=str(latest_market_day.date()),
    )
    benchmark = benchmark_loader(
        provider_uri,
        start_date="2026-01-01",
        end_date=str(latest_market_day.date()),
    )
    regimes = regime_builder(market)
    strategy_id = str(frozen["selected_strategy"])
    portfolio = config["portfolio"]
    costs = config["costs"]
    result = strategy_runner(
        predictions=ensemble.reset_index().loc[:, ["datetime", "instrument", "score"]],
        market=market,
        benchmark=benchmark,
        strategy_id=strategy_id,
        regimes=regimes,
        regime_positions=config["market_regime"]["positions"],
        initial_cash=10_000_000.0,
        topk=int(portfolio["topk"]),
        fixed_position=float(config["market_regime"]["positions"]["risk_on"]),
        single_stock_max=float(portfolio["single_stock_weight_max"]),
        industry_max=float(portfolio["industry_weight_max"]),
        growth_star_max=float(portfolio["growth_and_star_weight_max"]),
        commission_rate=float(costs["commission_rate_each_side"]),
        minimum_commission=float(costs["minimum_commission_cny"]),
        slippage_bps=0.0,
        all_in_round_trip_bps=float(costs["scenarios"]["baseline_total_bps"]),
        defer_unfilled_open_orders=bool(
            config["execution_rules"]["defer_unfilled_open_orders"]
        ),
    )
    verdict = confirmation_verdict(
        annual_excess_return=float(result["metrics"]["annual_excess_return"]),
        information_ratio=float(result["metrics"]["information_ratio"]),
        max_drawdown=float(result["metrics"]["max_drawdown"]),
        thresholds=config["acceptance"],
    )
    final_invalid_artifacts = _invalid_confirmation_artifacts(
        jobs, progress, split_factory
    )
    if not ensemble_path.is_file() or file_sha256(ensemble_path) != ensemble_hash:
        final_invalid_artifacts.append("confirmation_ensemble_predictions")
    if final_invalid_artifacts:
        payload = _failure("confirmation_artifact_invalid", run_id=run_id)
        payload["invalid_artifact_count"] = len(final_invalid_artifacts)
        payload["invalid_artifact_sample"] = final_invalid_artifacts[:20]
        _write_json(state_path, {**payload, "lifecycle_status": "stopped"})
        _write_json(report_dir / "confirmation_latest.json", payload)
        with _claim_lock(frozen_path.parent):
            final_frozen = _read_json(frozen_path)
            final_frozen["confirmation_status"] = "development_failed"
            final_frozen["shadow_tracking_started"] = False
            _write_json(frozen_path, final_frozen)
        return payload
    pit_dir = quant_data_dir / str(paths["pit_membership"])
    reference_dir = quant_data_dir / Path(str(paths["security_master"])).parent
    try:
        with membership_read_lock(pit_dir) as final_membership:
            membership_integrity_failure: str | None = None
            final_membership_snapshot_hash: str | None = None
            try:
                final_membership.ensure_transaction_clean()
                final_membership.validate_source(
                    start_date=str(config["universe"]["start_date"]),
                    end_date="2025-12-31",
                )
                final_membership_snapshot_hash = final_membership.fingerprint()
            except MembershipIntegrityError as exc:
                membership_integrity_failure = exc.reason_code

            with reference_read_lock(reference_dir) as final_reference:
                reference_integrity_failure: str | None = None
                final_reference_snapshot_hash: str | None = None
                try:
                    final_reference.ensure_transaction_clean()
                    if final_membership_snapshot_hash is None:
                        raise ReferenceIntegrityError(
                            "reference_membership_snapshot_unavailable",
                            "membership snapshot is unavailable for reference validation",
                        )
                    final_reference.validate_source(
                        start_date=str(config["universe"]["start_date"]),
                        end_date="2025-12-31",
                        expected_membership_snapshot_hash=final_membership_snapshot_hash,
                    )
                    final_reference_snapshot_hash = final_reference.fingerprint()
                except ReferenceIntegrityError as exc:
                    reference_integrity_failure = exc.reason_code

                with _claim_lock(frozen_path.parent):
                    final_git = dict(git_state_provider())
                    final_frozen = _read_json(frozen_path)
                    final_provider_state = _read_json(Path(provider_state_path))
                    reproducibility_failures: list[str] = []
                    if final_git.get("clean") is not True or final_git.get(
                        "commit"
                    ) != frozen.get("git_commit"):
                        reproducibility_failures.append("git_state")
                    if file_sha256(config_path) != frozen.get("config_hash"):
                        reproducibility_failures.append("config_hash")
                    if not manifest_path.is_file() or file_sha256(
                        manifest_path
                    ) != frozen.get("data_manifest_hash"):
                        reproducibility_failures.append("data_manifest_hash")
                    final_selection_hash = selection_contract_hash(
                        [
                            str(value)
                            for value in final_frozen.get("selected_models") or []
                        ],
                        final_frozen.get("selected_model_configs") or {},
                        str(final_frozen.get("selected_strategy") or ""),
                    )
                    if final_frozen.get("selection_hash") != frozen.get(
                        "selection_hash"
                    ) or final_selection_hash != frozen.get("selection_hash"):
                        reproducibility_failures.append("selection_hash")
                    if final_frozen.get("frozen_context_hash") != frozen.get(
                        "frozen_context_hash"
                    ) or frozen_context_hash(final_frozen) != frozen.get(
                        "frozen_context_hash"
                    ):
                        reproducibility_failures.append("frozen_context_hash")
                    final_development_artifacts = final_frozen.get(
                        "development_artifacts"
                    )
                    if (
                        not isinstance(final_development_artifacts, list)
                        or final_frozen.get("development_artifacts_hash")
                        != frozen.get("development_artifacts_hash")
                        or payload_sha256(final_development_artifacts)
                        != frozen.get("development_artifacts_hash")
                    ):
                        reproducibility_failures.append("development_artifacts_hash")
                    if final_provider_state.get("status") != "completed":
                        reproducibility_failures.append("provider_status")
                    final_provider_uri = str(
                        final_provider_state.get("provider_uri") or ""
                    )
                    if (
                        not final_provider_uri
                        or Path(final_provider_uri).resolve() != provider_uri.resolve()
                    ):
                        reproducibility_failures.append("provider_uri")
                    if final_provider_state.get("provider_key") != frozen.get(
                        "provider_key"
                    ):
                        reproducibility_failures.append("provider_key")
                    final_provider_identity = runtime_provider_identity(
                        provider_uri, str(frozen.get("provider_key") or "")
                    )
                    if final_provider_identity is None:
                        reproducibility_failures.append("provider_snapshot")
                    elif final_provider_identity.get(
                        "provider_tree_sha256"
                    ) != frozen.get("provider_tree_sha256"):
                        reproducibility_failures.append("provider_tree_sha256")
                    if (
                        collect_runtime_environment(config).get("environment_hash")
                        != runtime_environment.get("environment_hash")
                    ):
                        reproducibility_failures.append("runtime_environment_hash")
                    if (
                        membership_integrity_failure is not None
                        or final_membership_snapshot_hash
                        != frozen.get("membership_snapshot_hash")
                    ):
                        reproducibility_failures.append("membership_snapshot_hash")
                    if (
                        reference_integrity_failure is not None
                        or final_reference_snapshot_hash
                        != frozen.get("reference_snapshot_hash")
                    ):
                        reproducibility_failures.append("reference_snapshot_hash")
                    if reproducibility_failures:
                        payload = _failure(
                            "reproducibility_state_changed_during_confirmation",
                            run_id=run_id,
                        )
                        payload["reproducibility_failures"] = reproducibility_failures
                        _write_json(
                            state_path, {**payload, "lifecycle_status": "completed"}
                        )
                        _write_json(report_dir / "confirmation_latest.json", payload)
                        final_frozen["confirmation_status"] = "development_failed"
                        final_frozen["shadow_tracking_started"] = False
                        _write_json(frozen_path, final_frozen)
                        return payload

                    status = (
                        "shadow_observation"
                        if verdict["passed"]
                        else "development_failed"
                    )
                    daily_path = run_dir / "confirmation_backtest.parquet"
                    _atomic_write_parquet(result["daily"], daily_path)
                    payload = {
                        "status": status,
                        "reason_code": None
                        if verdict["passed"]
                        else "confirmation_failed",
                        "run_id": run_id,
                        "generated_at": datetime.now()
                        .astimezone()
                        .isoformat(timespec="seconds"),
                        "config_hash": frozen["config_hash"],
                        "runtime_environment_hash": runtime_environment[
                            "environment_hash"
                        ],
                        "confirmation_launch_runtime_evidence": (
                            confirmation_launch_evidence
                        ),
                        "confirmation_launch_runtime_evidence_hash": (
                            confirmation_launch_evidence_hash
                        ),
                        "development_git_commit": frozen.get("git_commit"),
                        "selected_models": selected_ids,
                        "selected_strategy": strategy_id,
                        "confirmation_start": str(confirmation_days[0].date()),
                        "latest_signal_day": str(latest_signal_day.date()),
                        "latest_market_day": str(latest_market_day.date()),
                        "confirmation_model_artifact_count": len(model_artifacts),
                        "confirmation_ensemble_prediction_path": str(ensemble_path),
                        "confirmation_ensemble_prediction_hash": ensemble_hash,
                        "metrics": result["metrics"],
                        "verdict": verdict,
                        "backtest_artifact": str(daily_path.resolve()),
                    }
                    _write_json(run_dir / "confirmation.json", payload)
                    _write_json(report_dir / "confirmation_latest.json", payload)
                    _write_json(
                        state_path, {**payload, "lifecycle_status": "completed"}
                    )
                    final_frozen["confirmation_status"] = status
                    final_frozen["confirmation_run_id"] = run_id
                    final_frozen["confirmation_launch_runtime_evidence"] = (
                        confirmation_launch_evidence
                    )
                    final_frozen["confirmation_launch_runtime_evidence_hash"] = (
                        confirmation_launch_evidence_hash
                    )
                    final_frozen["shadow_tracking_started"] = verdict["passed"]
                    if verdict["passed"]:
                        final_frozen["confirmation_model_artifacts"] = model_artifacts
                        final_frozen["confirmation_ensemble_prediction_path"] = str(
                            ensemble_path
                        )
                        final_frozen["confirmation_ensemble_prediction_hash"] = (
                            ensemble_hash
                        )
                        final_frozen["shadow_start_after"] = str(
                            latest_market_day.date()
                        )
                    _write_json(frozen_path, final_frozen)
                    return payload
    except (MembershipIntegrityError, ReferenceIntegrityError) as exc:
        payload = _failure(
            "reproducibility_state_changed_during_confirmation", run_id=run_id
        )
        payload["reproducibility_failures"] = [exc.reason_code]
        with _claim_lock(frozen_path.parent):
            final_frozen = _read_json(frozen_path)
            _write_json(state_path, {**payload, "lifecycle_status": "completed"})
            _write_json(report_dir / "confirmation_latest.json", payload)
            final_frozen["confirmation_status"] = "development_failed"
            final_frozen["shadow_tracking_started"] = False
            _write_json(frozen_path, final_frozen)
        return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the single-use frozen 2026 confirmation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument(
        "--report-dir", type=Path, default=reports_root() / "robust_strategy"
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=reports_root() / "robust_strategy" / "frozen_strategy_latest.json",
    )
    parser.add_argument(
        "--provider-state",
        type=Path,
        default=reports_root() / "robust_strategy" / "provider_latest.json",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("confirmation_%Y%m%d_%H%M%S")
    result = run_confirmation(
        config_path=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        frozen_path=args.frozen,
        provider_state_path=args.provider_state,
        run_id=run_id,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "shadow_observation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
