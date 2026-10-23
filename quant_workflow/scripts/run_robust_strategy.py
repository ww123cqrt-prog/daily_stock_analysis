#!/usr/bin/env python3
"""Independent, resumable orchestrator for the robust CSI800 strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple
from uuid import uuid4

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file  # noqa: E402
from _paths import (  # noqa: E402
    qlib_workflow_provider_uri,
    quant_data_root,
    reports_root,
    workflow_root,
)
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from _robust_reference_integrity import (  # noqa: E402
    ReferenceIntegrityError,
    reference_read_lock,
)
from prepare_robust_provider import (  # noqa: E402
    runtime_provider_identity,
)


WORKFLOW_ROOT = workflow_root()
DEFAULT_CONFIG = WORKFLOW_ROOT / "configs" / "robust_strategy.yaml"
DEFAULT_QUANT_DATA = quant_data_root()
DEFAULT_REPORT_DIR = reports_root() / "robust_strategy"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FINAL_STATUSES = {
    "blocked_data",
    "development_failed",
    "research_candidate_ready",
    "shadow_observation",
    "robust_strategy_qualified",
}
STAGE_REPORT_FILENAMES = {
    "build_pit_csi800": "build_status_latest.json",
    "download_robust_reference": "download_robust_reference_latest.json",
    "preflight_robust_data": "preflight_latest.json",
    "prepare_robust_provider": "provider_latest.json",
    "run_robust_development": "development.json",
}
STAGE_SUCCESS_STATUSES = {
    "build_pit_csi800": "completed",
    "download_robust_reference": "completed",
    "preflight_robust_data": "ready",
    "prepare_robust_provider": "completed",
    "run_robust_development": "research_candidate_ready",
}
STAGE_ACCEPTED_STATUSES = {
    **{name: {status} for name, status in STAGE_SUCCESS_STATUSES.items()},
    "run_robust_development": {
        "research_candidate_ready",
        "development_failed",
    },
}
PIPELINE_CONTEXT_KEYS = (
    "run_id",
    "config_hash",
    "git_commit",
    "data_manifest_hash",
    "as_of_date",
    "membership_snapshot_hash",
    "reference_snapshot_hash",
    "provider_uri",
    "provider_key",
    "provider_tree_sha256",
)
CANONICAL_EVIDENCE_SOURCES = {"run_scoped", "run_status"}
BASE_EVIDENCE_CONTEXT_KEYS = (
    "run_id",
    "config_hash",
    "git_commit",
    "data_manifest_hash",
    "as_of_date",
)
STAGE_REQUIRED_CONTEXT_KEYS = {
    "build_pit_csi800": BASE_EVIDENCE_CONTEXT_KEYS,
    "download_robust_reference": BASE_EVIDENCE_CONTEXT_KEYS
    + ("membership_snapshot_hash",),
    "preflight_robust_data": BASE_EVIDENCE_CONTEXT_KEYS
    + ("membership_snapshot_hash", "reference_snapshot_hash"),
    "prepare_robust_provider": BASE_EVIDENCE_CONTEXT_KEYS
    + ("membership_snapshot_hash", "reference_snapshot_hash"),
    "run_robust_development": PIPELINE_CONTEXT_KEYS,
}


class Stage(NamedTuple):
    name: str
    argv: list[str]


def new_run_id() -> str:
    """Return a readable run id with enough entropy for concurrent launches."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{uuid4().hex[:8]}"


@contextmanager
def _claim_run(report_dir: Path, run_id: str, *, resume: bool) -> Iterator[None]:
    runs_dir = Path(report_dir) / "runs"
    lock_path = runs_dir / ".locks" / f"{run_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        try:
            lock_file(stream, exclusive=True, blocking=False)
        except BlockingIOError as exc:
            raise ValueError(
                f"robust strategy run {run_id!r} is already running"
            ) from exc
        try:
            status_path = runs_dir / run_id / "status.json"
            if not resume and status_path.exists():
                raise ValueError(
                    f"robust strategy run {run_id!r} already exists; use resume"
                )
            yield
        finally:
            unlock_file(stream)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_at_least(
    payload: dict[str, Any], path: tuple[str, ...], minimum: float
) -> None:
    value: Any = payload
    for key in path:
        value = value[key]
    if float(value) < minimum:
        raise ValueError(f"{'.'.join(path)} must not be lower than {minimum}")


def _require_at_most(
    payload: dict[str, Any], path: tuple[str, ...], maximum: float
) -> None:
    value: Any = payload
    for key in path:
        value = value[key]
    if float(value) > maximum:
        raise ValueError(f"{'.'.join(path)} must not be higher than {maximum}")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("robust strategy config must be a mapping")
    if payload.get("version") != 1:
        raise ValueError("robust strategy config version must be 1")
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    for forbidden in (
        "margin",
        "short_selling",
        "broker_connection",
        "automatic_orders",
    ):
        if safety.get(forbidden) is not False:
            raise ValueError(f"safety.{forbidden} must remain false")
    if safety.get("long_only") is not True:
        raise ValueError("safety.long_only must remain true")
    if payload.get("universe", {}).get("kind") != "pit_csi800":
        raise ValueError("universe.kind must remain pit_csi800")
    if payload.get("universe", {}).get("membership_source") not in {
        "auto",
        "jqdata",
        "baostock",
    }:
        raise ValueError("universe.membership_source must be a verified PIT source")
    if payload.get("universe", {}).get("reference_source") not in {
        "auto",
        "jqdata",
        "baostock",
    }:
        raise ValueError("universe.reference_source must be a verified PIT source")
    if payload.get("universe", {}).get("reference_mode", "download") not in {
        "download",
        "reuse_existing",
    }:
        raise ValueError("universe.reference_mode must be download or reuse_existing")
    if payload.get("universe", {}).get("benchmark_mode", "download") not in {
        "download",
        "reuse_existing",
    }:
        raise ValueError("universe.benchmark_mode must be download or reuse_existing")
    if payload.get("statuses") != [
        "blocked_data",
        "development_failed",
        "research_candidate_ready",
        "shadow_observation",
        "robust_strategy_qualified",
    ]:
        raise ValueError("statuses must keep the approved robust-strategy lifecycle")

    _require_at_least(payload, ("data_gates", "membership_coverage_min"), 0.99)
    _require_at_least(payload, ("data_gates", "ohlcv_coverage_min"), 0.98)
    _require_at_least(
        payload, ("acceptance", "positive_excess_quarter_ratio_min"), 0.70
    )
    _require_at_least(payload, ("acceptance", "annual_excess_return_min"), 0.05)
    _require_at_least(
        payload, ("acceptance", "median_quarter_information_ratio_min"), 0.5
    )
    _require_at_least(
        payload, ("acceptance", "positive_rank_ic_interval_ratio_min"), 0.70
    )
    _require_at_least(
        payload, ("acceptance", "window_rank_correlation_median_min"), 0.70
    )
    _require_at_least(payload, ("acceptance", "top50_overlap_mean_min"), 0.60)
    _require_at_least(
        payload, ("acceptance", "seed_rank_correlation_median_min"), 0.70
    )
    _require_at_least(
        payload, ("acceptance", "seed_top50_overlap_mean_min"), 0.60
    )
    _require_at_least(
        payload, ("acceptance", "stress_cost_excess_return_min_exclusive"), 0.0
    )
    _require_at_least(
        payload,
        ("acceptance", "delayed_execution_excess_return_min_exclusive"),
        0.0,
    )
    _require_at_least(payload, ("acceptance", "positive_regime_count_min"), 3)
    _require_at_least(payload, ("acceptance", "regime_days_for_floor"), 60)
    _require_at_least(payload, ("acceptance", "regime_annual_excess_floor"), -0.05)
    _require_at_least(
        payload, ("acceptance", "confirmation_excess_return_min_exclusive"), 0.0
    )
    _require_at_least(
        payload,
        ("acceptance", "confirmation_information_ratio_min_exclusive"),
        0.0,
    )
    _require_at_least(payload, ("acceptance", "shadow_days"), 60)
    _require_at_least(
        payload, ("acceptance", "shadow_net_excess_return_min_exclusive"), 0.0
    )
    _require_at_least(payload, ("acceptance", "shadow_task_success_ratio_min"), 0.95)
    _require_at_most(payload, ("acceptance", "development_max_drawdown_abs_max"), 0.20)
    _require_at_most(payload, ("acceptance", "confirmation_max_drawdown_abs_max"), 0.20)
    _require_at_most(payload, ("acceptance", "shadow_max_drawdown_abs_max"), 0.12)
    if payload.get("training", {}).get("windows_trading_days") != [504, 756, 1008]:
        raise ValueError("training.windows_trading_days must remain [504, 756, 1008]")
    tuning_screen_reference_window = payload.get("training", {}).get(
        "tuning_screen_reference_window"
    )
    if (
        isinstance(tuning_screen_reference_window, bool)
        or not isinstance(tuning_screen_reference_window, int)
        or tuning_screen_reference_window not in [504, 756, 1008]
    ):
        raise ValueError(
            "training.tuning_screen_reference_window must be one of [504, 756, 1008]"
        )
    if payload.get("training", {}).get("tree_seeds") != [42, 2026, 3407]:
        raise ValueError("training.tree_seeds must remain [42, 2026, 3407]")
    full_matrix_finalists = payload.get("training", {}).get(
        "bounded_tuning_full_matrix_finalists", 1
    )
    if (
        isinstance(full_matrix_finalists, bool)
        or not isinstance(full_matrix_finalists, int)
        or not 1 <= full_matrix_finalists <= 3
    ):
        raise ValueError(
            "training.bounded_tuning_full_matrix_finalists must be between 1 and 3"
        )
    _require_at_most(
        payload,
        ("training", "ensemble_model_rank_correlation_abs_max"),
        0.95,
    )
    _require_at_most(
        payload,
        ("training", "ensemble_model_top50_overlap_max"),
        0.90,
    )
    model_ids = {str(item.get("id")) for item in payload.get("models") or []}
    if "mps_dnn" in model_ids:
        if payload.get("training", {}).get("neural_seeds") != [42, 2026, 3407]:
            raise ValueError("training.neural_seeds must remain [42, 2026, 3407]")
        mps_model = next(
            item for item in payload["models"] if str(item.get("id")) == "mps_dnn"
        )
        mps_kwargs = mps_model.get("kwargs") or {}
        if str(mps_kwargs.get("device")) != "mps":
            raise ValueError("mps_dnn must use device=mps without CPU fallback")
        try:
            max_steps = int(mps_kwargs["max_steps"])
            eval_steps = int(mps_kwargs["eval_steps"])
            early_stop_rounds = int(mps_kwargs["early_stop_rounds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "mps_dnn early stopping parameters must be positive integers"
            ) from exc
        if max_steps <= 0 or eval_steps <= 0 or early_stop_rounds <= 0:
            raise ValueError(
                "mps_dnn early stopping parameters must be positive integers"
            )
        evaluation_count = (max_steps + eval_steps - 1) // eval_steps
        if early_stop_rounds >= evaluation_count:
            raise ValueError("mps_dnn early stopping must be reachable before max_steps")
        repeatability = payload.get("training", {}).get("neural_repeatability")
        if (
            not isinstance(repeatability, dict)
            or repeatability.get("enabled") is not True
        ):
            raise ValueError("neural repeatability check must remain enabled")
        try:
            repeatability_tolerance = float(repeatability["score_max_abs_diff"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "training.neural_repeatability.score_max_abs_diff must be finite and at most 1e-7"
            ) from exc
        if (
            not math.isfinite(repeatability_tolerance)
            or repeatability_tolerance < 0.0
            or repeatability_tolerance > 1e-7
        ):
            raise ValueError(
                "training.neural_repeatability.score_max_abs_diff must be finite and at most 1e-7"
            )
    if len(payload.get("rolling_validation", {}).get("test_quarters") or []) != 12:
        raise ValueError("rolling_validation must contain 12 development quarters")
    strategies = payload.get("strategies") or []
    expected_strategy_ids = [
        "weekly_top50",
        "low_turnover",
        "regime_scaled",
        "sector_constrained",
        "full_risk_control",
    ]
    if [str(item.get("id")) for item in strategies] != expected_strategy_ids:
        raise ValueError("strategies must keep the five approved candidate rules")
    eligible = [
        str(item["id"])
        for item in strategies
        if item.get("eligible_for_final_selection") is True
    ]
    if eligible != ["full_risk_control"]:
        raise ValueError("only full_risk_control may be eligible for final selection")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _resolved_data_path(quant_data_dir: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"robust data path must be relative to quant_data: {value}")
    return quant_data_dir / relative


def build_stages(
    config: dict[str, Any],
    *,
    config_path: Path,
    quant_data_dir: Path,
    report_dir: Path,
    as_of_date: str,
    skip_membership_build: bool = False,
    run_id: str | None = None,
    resume: bool = False,
    allow_legacy_compatibility: bool = False,
) -> list[Stage]:
    paths = config["paths"]
    universe = config["universe"]
    gates = config["data_gates"]
    pit_dir = _resolved_data_path(quant_data_dir, paths["pit_membership"])
    reference_dir = _resolved_data_path(
        quant_data_dir, str(Path(paths["security_master"]).parent)
    )
    price_dir = _resolved_data_path(quant_data_dir, paths["price_parquet"])
    configured_provider_dir = _resolved_data_path(quant_data_dir, paths["qlib_provider"])
    legacy_compatibility = bool(allow_legacy_compatibility)
    provider_dir = (
        configured_provider_dir
        if legacy_compatibility
        else qlib_workflow_provider_uri("cleaned")
    )
    runtime_root = quant_data_dir / "runtime" / "robust_strategy"
    stage_report_root = (
        report_dir / "runs" / run_id / "stage_reports" if run_id else report_dir
    )

    def stage_report_dir(stage_name: str) -> Path:
        return stage_report_root / stage_name if run_id else stage_report_root

    python = sys.executable
    stages: list[Stage] = []
    if not skip_membership_build:
        stages.append(
            Stage(
                "build_pit_csi800",
                [
                    python,
                    str(SCRIPT_DIR / "build_pit_csi800.py"),
                    "--start-date",
                    str(universe["start_date"]),
                    "--end-date",
                    as_of_date,
                    "--output-dir",
                    str(pit_dir),
                    "--status-out-dir",
                    str(stage_report_dir("build_pit_csi800")),
                    "--membership-source",
                    str(universe["membership_source"]),
                ],
            )
        )
    reference_argv = [
        python,
        str(SCRIPT_DIR / "download_robust_reference.py"),
        "--pit-dir",
        str(pit_dir),
        "--reference-dir",
        str(reference_dir),
        "--start-date",
        str(universe["start_date"]),
        "--end-date",
        as_of_date,
        "--status-out-dir",
        str(stage_report_dir("download_robust_reference")),
        "--reference-source",
        str(universe["reference_source"]),
    ]
    if run_id:
        reference_argv += ["--run-id", str(run_id)]
    if universe.get("reference_mode", "download") == "reuse_existing":
        reference_argv.append("--reuse-existing")
    stages.append(Stage("download_robust_reference", reference_argv))
    benchmark_argv = [
        python,
        str(SCRIPT_DIR / "add_index_benchmark.py"),
        "--provider-dir",
        str(provider_dir),
        "--required-start-date",
        str(universe["start_date"]),
    ]
    if universe.get("benchmark_mode", "download") == "reuse_existing":
        benchmark_argv.append("--reuse-existing")
    if legacy_compatibility:
        stages.append(Stage("add_robust_benchmarks", benchmark_argv))
    stages.append(
        Stage(
            "preflight_robust_data",
            [
                python,
                str(SCRIPT_DIR / "preflight_robust_data.py"),
                "--pit-dir",
                str(pit_dir),
                "--price-dir",
                str(price_dir),
                "--provider-dir",
                str(provider_dir),
                "--reference-dir",
                str(reference_dir),
                "--start-date",
                str(universe["start_date"]),
                "--end-date",
                as_of_date,
                "--as-of-date",
                as_of_date,
                "--expected-members-per-day",
                str(gates["expected_members_per_day"]),
                "--membership-coverage-min",
                str(gates["membership_coverage_min"]),
                "--ohlcv-coverage-min",
                str(gates["ohlcv_coverage_min"]),
                "--out-dir",
                str(stage_report_dir("preflight_robust_data")),
            ],
        )
    )
    stages.append(
        Stage(
            "prepare_robust_provider",
            [
                python,
                str(SCRIPT_DIR / "prepare_robust_provider.py"),
                "--canonical-provider",
                str(provider_dir),
                "--intervals",
                str(pit_dir / "membership_intervals.parquet"),
                "--runtime-root",
                str(runtime_root),
                "--data-manifest",
                str((quant_data_dir / "DATA_MANIFEST.json").resolve()),
                "--status-out-dir",
                str(stage_report_dir("prepare_robust_provider")),
            ],
        )
    )
    development_argv = [
        python,
        str(SCRIPT_DIR / "run_robust_development.py"),
        "--config",
        str(Path(config_path).resolve()),
        "--quant-data-dir",
        str(quant_data_dir),
        "--report-dir",
        str(report_dir),
        "--provider-state",
        str(
            report_dir / "runs" / run_id / "evidence" / "prepare_robust_provider.json"
            if run_id
            else report_dir / "provider_latest.json"
        ),
        "--as-of-date",
        as_of_date,
    ]
    if run_id:
        development_argv += ["--run-id", str(run_id)]
    if resume:
        development_argv.append("--resume")
    stages.append(
        Stage(
            "run_robust_development",
            development_argv,
        )
    )
    return stages


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(WORKFLOW_ROOT.parent), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return (
        value
        if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value)
        else None
    )


def git_worktree_state() -> dict[str, Any]:
    """Return the exact repository revision and whether tracked/untracked code is clean."""
    commit = _git_commit()
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(WORKFLOW_ROOT.parent),
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": commit, "clean": False, "changes": ["git_status_unavailable"]}
    if completed.returncode != 0:
        return {"commit": commit, "clean": False, "changes": ["git_status_failed"]}
    changes = [line for line in completed.stdout.splitlines() if line.strip()]
    return {"commit": commit, "clean": not changes, "changes": changes}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage_option(stage: Stage, option: str) -> str | None:
    try:
        return stage.argv[stage.argv.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _run_scoped_report_path(stage: Stage, report_dir: Path, run_id: str) -> Path:
    if stage.name == "run_robust_development":
        return report_dir / "runs" / run_id / "development" / "development.json"
    return (
        report_dir
        / "runs"
        / run_id
        / "stage_reports"
        / stage.name
        / STAGE_REPORT_FILENAMES[stage.name]
    )


def _canonical_evidence_path(stage: Stage, report_dir: Path, run_id: str) -> Path:
    return report_dir / "runs" / run_id / "evidence" / f"{stage.name}.json"


def _invalid_stage_evidence(reason_code: str) -> dict[str, Any]:
    return {"status": "blocked_data", "reason_code": reason_code}


def _pipeline_context(state: dict[str, Any] | None, run_id: str) -> dict[str, Any]:
    context = {"run_id": run_id}
    if state is not None:
        context.update(
            {key: state[key] for key in PIPELINE_CONTEXT_KEYS if key in state}
        )
    return context


def _provider_report_identity_matches(payload: dict[str, Any]) -> bool:
    provider_uri = payload.get("provider_uri")
    provider_key = payload.get("provider_key")
    provider_tree_sha256 = payload.get("provider_tree_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (provider_uri, provider_key, provider_tree_sha256)
    ):
        return False
    identity = runtime_provider_identity(Path(provider_uri), provider_key)
    return bool(
        identity and identity.get("provider_tree_sha256") == provider_tree_sha256
    )


def _evidence_context_matches(
    payload: dict[str, Any],
    *,
    stage: Stage,
    report_dir: Path,
    quant_data_dir: Path,
    config: dict[str, Any],
    run_id: str,
    state: dict[str, Any] | None,
    require_persisted_identity: bool = False,
) -> bool:
    if payload.get("run_id") != run_id:
        return False
    if require_persisted_identity:
        if payload.get("evidence_stage") != stage.name:
            return False
        if payload.get("evidence_source") not in CANONICAL_EVIDENCE_SOURCES:
            return False
        persisted_context = payload.get("pipeline_context")
        required_keys = STAGE_REQUIRED_CONTEXT_KEYS.get(stage.name, ())
        if not isinstance(persisted_context, dict) or any(
            key not in persisted_context for key in required_keys
        ):
            return False
    elif payload.get("evidence_stage", stage.name) != stage.name:
        return False

    expected = state or {}
    aliases = {
        "config_hash": "config_hash",
        "git_commit": "git_commit",
        "data_manifest_hash": "data_manifest_hash",
        "data_manifest_sha256": "data_manifest_hash",
        "as_of_date": "as_of_date",
        "membership_snapshot_hash": "membership_snapshot_hash",
        "reference_snapshot_hash": "reference_snapshot_hash",
        "provider_uri": "provider_uri",
        "provider_key": "provider_key",
        "provider_tree_sha256": "provider_tree_sha256",
        "context_hash": "context_hash",
    }
    introduced_provider_identity = False
    for report_key, state_key in aliases.items():
        if report_key not in payload:
            continue
        if state_key in expected:
            if payload[report_key] != expected[state_key]:
                return False
            continue
        if stage.name == "prepare_robust_provider" and state_key in {
            "provider_uri",
            "provider_key",
            "provider_tree_sha256",
        }:
            if not introduced_provider_identity:
                introduced_provider_identity = _provider_report_identity_matches(
                    payload
                )
            if not introduced_provider_identity:
                return False
            continue
        if stage.name == "build_pit_csi800" and state_key == "membership_snapshot_hash":
            try:
                current_hash = _membership_snapshot_hash(
                    _resolved_data_path(
                        quant_data_dir, config["paths"]["pit_membership"]
                    ),
                    start_date=str(config["universe"]["start_date"]),
                    end_date=str(expected.get("as_of_date") or ""),
                )
            except MembershipIntegrityError:
                return False
            if payload[report_key] != current_hash:
                return False
            continue
        if (
            stage.name == "download_robust_reference"
            and state_key == "reference_snapshot_hash"
        ):
            membership_hash = expected.get("membership_snapshot_hash")
            if not membership_hash:
                return False
            try:
                current_hash = _reference_snapshot_hash(
                    _resolved_data_path(
                        quant_data_dir,
                        str(Path(config["paths"]["security_master"]).parent),
                    ),
                    start_date=str(config["universe"]["start_date"]),
                    end_date=str(expected.get("as_of_date") or ""),
                    membership_snapshot_hash=str(membership_hash),
                )
            except ReferenceIntegrityError:
                return False
            if payload[report_key] != current_hash:
                return False
            continue
        return False

    development_context_hash = payload.get("development_context_hash")
    if development_context_hash is not None:
        development_context = _read_json(
            report_dir / "runs" / run_id / "development" / "development_context.json"
        )
        if development_context.get("context_hash") != development_context_hash:
            return False

    persisted_context = payload.get("pipeline_context")
    if persisted_context is not None:
        if not isinstance(persisted_context, dict):
            return False
        if payload.get("pipeline_context_hash") != _payload_sha256(persisted_context):
            return False
        current_context = _pipeline_context(state, run_id)
        if any(
            current_context.get(key) != value
            for key, value in persisted_context.items()
        ):
            return False
    return True


def _persist_stage_evidence(
    payload: dict[str, Any],
    *,
    stage: Stage,
    report_dir: Path,
    run_id: str,
    state: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    persisted = dict(payload)
    persisted["run_id"] = run_id
    persisted["evidence_stage"] = stage.name
    persisted["evidence_source"] = source
    persisted_context = _pipeline_context(state, run_id)
    persisted["pipeline_context"] = persisted_context
    persisted["pipeline_context_hash"] = _payload_sha256(persisted_context)
    _atomic_write(
        _canonical_evidence_path(stage, report_dir, run_id),
        json.dumps(persisted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return persisted


def _clear_run_scoped_stage_evidence(
    stage: Stage, report_dir: Path, run_id: str
) -> None:
    if stage.name not in STAGE_REPORT_FILENAMES:
        return
    for path in (
        _canonical_evidence_path(stage, report_dir, run_id),
        _run_scoped_report_path(stage, report_dir, run_id),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)


def _stage_evidence(
    stage: Stage,
    report_dir: Path,
    quant_data_dir: Path,
    config: dict[str, Any],
    *,
    run_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage.name not in STAGE_REPORT_FILENAMES:
        return {}
    run_id = run_id or _stage_option(stage, "--run-id")
    if not run_id:
        return _invalid_stage_evidence("stage_evidence_unverified")

    canonical_path = _canonical_evidence_path(stage, report_dir, run_id)
    run_scoped_path = _run_scoped_report_path(stage, report_dir, run_id)
    if canonical_path.is_file():
        payload = _read_json(canonical_path)
        source = "canonical"
    elif run_scoped_path.is_file():
        payload = _read_json(run_scoped_path)
        if payload.get("run_id", run_id) != run_id:
            return _invalid_stage_evidence("stage_evidence_unverified")
        payload["run_id"] = run_id
        source = "run_scoped"
    else:
        return _invalid_stage_evidence("stage_evidence_unverified")

    if not payload:
        return _invalid_stage_evidence("stage_evidence_unverified")
    if payload.get("evidence_source") == "shared_latest":
        return _invalid_stage_evidence("stage_evidence_context_mismatch")
    if not _evidence_context_matches(
        payload,
        stage=stage,
        report_dir=report_dir,
        quant_data_dir=quant_data_dir,
        config=config,
        run_id=run_id,
        state=state,
        require_persisted_identity=source == "canonical",
    ):
        return _invalid_stage_evidence("stage_evidence_context_mismatch")
    evidence_status = payload.get("status")
    if evidence_status == "blocked_data" and not re.fullmatch(
        r"[a-z][a-z0-9_]*", str(payload.get("reason_code") or "")
    ):
        return _invalid_stage_evidence("stage_evidence_status_invalid")
    if evidence_status != "blocked_data" and (
        evidence_status not in STAGE_ACCEPTED_STATUSES[stage.name]
    ):
        return _invalid_stage_evidence("stage_evidence_status_invalid")
    if source == "canonical":
        return payload
    return _persist_stage_evidence(
        payload,
        stage=stage,
        report_dir=report_dir,
        run_id=run_id,
        state=state,
        source=source,
    )


def _default_stage_runner(stage: Stage) -> int:
    completed = subprocess.run(stage.argv, cwd=WORKFLOW_ROOT, check=False)
    return int(completed.returncode)


def _save_state(state: dict[str, Any], run_dir: Path, report_dir: Path) -> None:
    serialized = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(run_dir / "status.json", serialized)
    _atomic_write(report_dir / "robust_strategy_status_latest.json", serialized)


def _membership_snapshot_hash(pit_dir: Path, *, start_date: str, end_date: str) -> str:
    with membership_read_lock(pit_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        snapshot.validate_source(start_date=start_date, end_date=end_date)
        return snapshot.fingerprint()


def _reference_snapshot_hash(
    reference_dir: Path,
    *,
    start_date: str,
    end_date: str,
    membership_snapshot_hash: str,
) -> str:
    with reference_read_lock(reference_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        snapshot.validate_source(
            start_date=start_date,
            end_date=end_date,
            expected_membership_snapshot_hash=membership_snapshot_hash,
        )
        return snapshot.fingerprint()


@contextmanager
def _final_identity_guard(
    *,
    state: dict[str, Any],
    config_path: Path,
    quant_data_dir: Path,
    pit_dir: Path,
    reference_dir: Path,
    membership_start: str,
    as_of_date: str,
    git_state_provider: Callable[[], dict[str, Any]],
) -> Iterator[str | None]:
    reason: str | None = None
    stack = ExitStack()
    try:
        membership_snapshot = stack.enter_context(membership_read_lock(pit_dir))
        membership_snapshot.ensure_transaction_clean()
        membership_snapshot.validate_source(
            start_date=membership_start,
            end_date=as_of_date,
        )
        membership_hash = membership_snapshot.fingerprint()
        if membership_hash != state.get("membership_snapshot_hash"):
            reason = "membership_snapshot_changed"
        reference_snapshot = stack.enter_context(reference_read_lock(reference_dir))
        reference_snapshot.ensure_transaction_clean()
        reference_snapshot.validate_source(
            start_date=membership_start,
            end_date=as_of_date,
            expected_membership_snapshot_hash=str(
                state.get("membership_snapshot_hash") or ""
            ),
        )
        reference_hash = reference_snapshot.fingerprint()
        if reason is None and reference_hash != state.get("reference_snapshot_hash"):
            reason = "reference_snapshot_changed"

        try:
            latest_git = dict(git_state_provider())
        except Exception:
            latest_git = {}
        if reason is None and (
            latest_git.get("clean") is not True
            or latest_git.get("changes") != []
            or latest_git.get("commit") != state.get("git_commit")
        ):
            reason = "git_worktree_changed"
        try:
            current_config_hash = file_sha256(config_path)
        except OSError:
            current_config_hash = None
        if reason is None and current_config_hash != state.get("config_hash"):
            reason = "config_changed"
        manifest = quant_data_dir / "DATA_MANIFEST.json"
        current_manifest_hash = file_sha256(manifest) if manifest.is_file() else None
        if reason is None and current_manifest_hash != state.get("data_manifest_hash"):
            reason = "data_manifest_changed"
        provider_uri = str(state.get("provider_uri") or "")
        provider_key = str(state.get("provider_key") or "")
        provider_identity = (
            runtime_provider_identity(Path(provider_uri), provider_key)
            if provider_uri and provider_key
            else None
        )
        if reason is None and (
            not provider_identity
            or provider_identity.get("provider_tree_sha256")
            != state.get("provider_tree_sha256")
        ):
            reason = "provider_snapshot_changed"
    except MembershipIntegrityError:
        reason = "membership_snapshot_changed"
    except ReferenceIntegrityError:
        if reason is None:
            reason = "reference_snapshot_changed"
    try:
        yield reason
    finally:
        stack.close()


def _run_pipeline_claimed(
    config_path: Path = DEFAULT_CONFIG,
    *,
    quant_data_dir: Path = DEFAULT_QUANT_DATA,
    report_dir: Path = DEFAULT_REPORT_DIR,
    as_of_date: str | None = None,
    run_id: str | None = None,
    skip_membership_build: bool = False,
    resume: bool = False,
    allow_legacy_compatibility: bool = False,
    stage_runner: Callable[[Stage], int] | None = None,
    git_state_provider: Callable[[], dict[str, Any]] = git_worktree_state,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    as_of_date = str(as_of_date or date.today().isoformat())
    if as_of_date > date.today().isoformat():
        raise ValueError("as_of_date must not be in the future")
    run_id = run_id or new_run_id()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    quant_data_dir = Path(quant_data_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    run_dir = report_dir / "runs" / run_id
    status_path = run_dir / "status.json"
    config_hash = file_sha256(config_path)
    current_git = dict(git_state_provider())
    current_commit = current_git.get("commit")
    if (
        current_git.get("clean") is not True
        or not isinstance(current_commit, str)
        or not current_commit
        or current_git.get("changes") != []
    ):
        raise ValueError(
            "robust strategy pipeline requires a clean git worktree, "
            "a non-empty commit, and no changes"
        )
    manifest = quant_data_dir / "DATA_MANIFEST.json"
    current_manifest_hash = file_sha256(manifest) if manifest.is_file() else None
    pit_dir = _resolved_data_path(quant_data_dir, config["paths"]["pit_membership"])
    reference_dir = _resolved_data_path(
        quant_data_dir, str(Path(config["paths"]["security_master"]).parent)
    )
    membership_start = str(config["universe"]["start_date"])
    resume_verified_stages: set[str] = set()
    completed_stages: dict[str, Any] = {}

    if resume and status_path.is_file():
        state = _read_json(status_path)
        if state.get("git_clean") is not True:
            raise ValueError("cannot resume state without git_clean=True")
        if state.get("run_id") != run_id:
            raise ValueError("cannot resume state from a different run id")
        if state.get("config_hash") != config_hash:
            raise ValueError("cannot resume a run with a different config hash")
        if state.get("git_commit") != current_git.get("commit"):
            raise ValueError("cannot resume a run from a different git commit")
        if state.get("data_manifest_hash") != current_manifest_hash:
            raise ValueError("cannot resume a run with a different data manifest")
        if state.get("as_of_date") != as_of_date:
            raise ValueError("cannot resume a run with a different as-of date")
        completed_stages = (
            state.get("stages") if isinstance(state.get("stages"), dict) else {}
        )
        membership_completed = skip_membership_build or (
            isinstance(completed_stages.get("build_pit_csi800"), dict)
            and completed_stages["build_pit_csi800"].get("status") == "completed"
        )
        reference_completed = (
            isinstance(completed_stages.get("download_robust_reference"), dict)
            and completed_stages["download_robust_reference"].get("status")
            == "completed"
        )
        provider_completed = (
            isinstance(completed_stages.get("prepare_robust_provider"), dict)
            and completed_stages["prepare_robust_provider"].get("status") == "completed"
        )
        if membership_completed:
            recorded_membership_hash = state.get("membership_snapshot_hash")
            if not recorded_membership_hash:
                raise ValueError("cannot resume without membership snapshot hash")
            try:
                current_membership_hash = _membership_snapshot_hash(
                    pit_dir, start_date=membership_start, end_date=as_of_date
                )
            except MembershipIntegrityError as exc:
                raise ValueError(
                    "cannot resume with unavailable membership snapshot"
                ) from exc
            if recorded_membership_hash != current_membership_hash:
                raise ValueError("cannot resume with a different membership snapshot")
            if (
                isinstance(completed_stages.get("build_pit_csi800"), dict)
                and completed_stages["build_pit_csi800"].get("status") == "completed"
            ):
                resume_verified_stages.add("build_pit_csi800")
        if reference_completed:
            recorded_membership_hash = state.get("membership_snapshot_hash")
            if not recorded_membership_hash:
                raise ValueError("cannot resume without membership snapshot hash")
            recorded_reference_hash = state.get("reference_snapshot_hash")
            if not recorded_reference_hash:
                raise ValueError("cannot resume without reference snapshot hash")
            try:
                current_reference_hash = _reference_snapshot_hash(
                    reference_dir,
                    start_date=membership_start,
                    end_date=as_of_date,
                    membership_snapshot_hash=str(recorded_membership_hash),
                )
            except ReferenceIntegrityError as exc:
                raise ValueError(
                    "cannot resume with unavailable reference snapshot"
                ) from exc
            if recorded_reference_hash != current_reference_hash:
                raise ValueError("cannot resume with a different reference snapshot")
            resume_verified_stages.add("download_robust_reference")
        if provider_completed:
            provider_uri = str(state.get("provider_uri") or "")
            provider_key = str(state.get("provider_key") or "")
            provider_identity = (
                runtime_provider_identity(Path(provider_uri), provider_key)
                if provider_uri and provider_key
                else None
            )
            recorded_provider_tree = state.get("provider_tree_sha256")
            if not provider_identity or (
                recorded_provider_tree
                and recorded_provider_tree
                != provider_identity.get("provider_tree_sha256")
            ):
                raise ValueError(
                    "cannot resume with an unavailable or changed provider"
                )
            state["provider_tree_sha256"] = provider_identity["provider_tree_sha256"]
    else:
        state = {
            "run_id": run_id,
            "status": "running",
            "lifecycle_status": "running",
            "reason_code": None,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "git_commit": current_commit,
            "git_clean": True,
            "data_manifest_hash": current_manifest_hash,
            "as_of_date": as_of_date,
            "stages": {},
        }
        if skip_membership_build:
            try:
                state["membership_snapshot_hash"] = _membership_snapshot_hash(
                    pit_dir, start_date=membership_start, end_date=as_of_date
                )
            except MembershipIntegrityError as exc:
                state["reason_code"] = exc.reason_code
                state["lifecycle_status"] = "stopped"
                state["updated_at"] = (
                    datetime.now().astimezone().isoformat(timespec="seconds")
                )
                _save_state(state, run_dir, report_dir)
                return state
    state["status"] = "running"
    state["lifecycle_status"] = "running"
    state["reason_code"] = None
    state.pop("failed_stage", None)
    state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _save_state(state, run_dir, report_dir)

    stages = build_stages(
        config,
        config_path=config_path,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        as_of_date=as_of_date,
        skip_membership_build=skip_membership_build,
        run_id=run_id,
        resume=resume,
        allow_legacy_compatibility=allow_legacy_compatibility,
    )
    if resume:
        provider_stage = next(
            stage for stage in stages if stage.name == "prepare_robust_provider"
        )
        previous_provider = completed_stages.get(provider_stage.name)
        if (
            isinstance(previous_provider, dict)
            and previous_provider.get("status") == "completed"
        ):
            canonical_provider_evidence = _canonical_evidence_path(
                provider_stage, report_dir, run_id
            )
            if canonical_provider_evidence.is_file():
                provider_evidence = _stage_evidence(
                    provider_stage,
                    report_dir,
                    quant_data_dir,
                    config,
                    run_id=run_id,
                    state=state,
                )
                if provider_evidence.get("status") != "blocked_data":
                    resume_verified_stages.add(provider_stage.name)
            else:
                previous_evidence = previous_provider.get("evidence")
                provider_evidence = (
                    dict(previous_evidence)
                    if isinstance(previous_evidence, dict)
                    else {}
                )
                if provider_evidence.get("run_id") not in {None, run_id}:
                    raise ValueError("cannot resume provider evidence from another run")
                provider_evidence.update(
                    {
                        "status": "completed",
                        "run_id": run_id,
                        "provider_uri": state["provider_uri"],
                        "provider_key": state["provider_key"],
                        "provider_tree_sha256": state["provider_tree_sha256"],
                    }
                )
                if not _evidence_context_matches(
                    provider_evidence,
                    stage=provider_stage,
                    report_dir=report_dir,
                    quant_data_dir=quant_data_dir,
                    config=config,
                    run_id=run_id,
                    state=state,
                ):
                    raise ValueError("cannot resume with mismatched provider evidence")
                _persist_stage_evidence(
                    provider_evidence,
                    stage=provider_stage,
                    report_dir=report_dir,
                    run_id=run_id,
                    state=state,
                    source="run_status",
                )
                _save_state(state, run_dir, report_dir)
                resume_verified_stages.add(provider_stage.name)
    runner = stage_runner or _default_stage_runner
    for stage in stages:
        previous = state.get("stages", {}).get(stage.name)
        previously_completed = (
            resume
            and isinstance(previous, dict)
            and previous.get("status") == "completed"
            and previous.get("return_code") == 0
        )
        if previously_completed:
            runtime_verified = (
                stage.name
                not in {
                    "build_pit_csi800",
                    "download_robust_reference",
                    "prepare_robust_provider",
                }
                or stage.name in resume_verified_stages
            )
            if runtime_verified and stage.name == "add_robust_benchmarks":
                continue
            if runtime_verified and stage.name in STAGE_REPORT_FILENAMES:
                resume_evidence = _stage_evidence(
                    stage,
                    report_dir,
                    quant_data_dir,
                    config,
                    run_id=run_id,
                    state=state,
                )
                if resume_evidence.get("status") != "blocked_data":
                    previous["evidence"] = resume_evidence
                    _save_state(state, run_dir, report_dir)
                    continue
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        state.setdefault("stages", {})[stage.name] = {
            "status": "running",
            "started_at": started_at,
            "argv": stage.argv,
        }
        state["current_stage"] = stage.name
        state["updated_at"] = started_at
        _save_state(state, run_dir, report_dir)
        if stage.name in STAGE_REPORT_FILENAMES:
            _clear_run_scoped_stage_evidence(stage, report_dir, run_id)
        try:
            return_code = int(runner(stage))
        except (OSError, subprocess.SubprocessError) as exc:
            return_code = 127
            evidence: dict[str, Any] = {
                "reason_code": "stage_launch_failed",
                "error_type": type(exc).__name__,
            }
        else:
            evidence = _stage_evidence(
                stage,
                report_dir,
                quant_data_dir,
                config,
                run_id=run_id,
                state=state,
            )

        stage_status = "completed" if return_code == 0 else "failed"
        if evidence.get("status") == "blocked_data":
            stage_status = "blocked_data"
        elif evidence.get("status") == "development_failed":
            stage_status = "development_failed"
        state["stages"][stage.name].update(
            {
                "status": stage_status,
                "return_code": return_code,
                "finished_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "evidence": evidence,
            }
        )
        if return_code != 0 or evidence.get("status") in {
            "blocked_data",
            "development_failed",
        }:
            state["status"] = (
                "blocked_data"
                if stage.name != "run_robust_development"
                else "development_failed"
            )
            state["lifecycle_status"] = "stopped"
            state["failed_stage"] = stage.name
            state["reason_code"] = evidence.get("reason_code") or (
                "data_preflight_failed"
                if stage.name != "run_robust_development"
                else "development_stage_failed"
            )
            state["updated_at"] = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            _save_state(state, run_dir, report_dir)
            return state

        try:
            if stage.name == "build_pit_csi800":
                state["membership_snapshot_hash"] = _membership_snapshot_hash(
                    pit_dir, start_date=membership_start, end_date=as_of_date
                )
            elif stage.name == "download_robust_reference":
                state["reference_snapshot_hash"] = _reference_snapshot_hash(
                    reference_dir,
                    start_date=membership_start,
                    end_date=as_of_date,
                    membership_snapshot_hash=str(state["membership_snapshot_hash"]),
                )
        except (MembershipIntegrityError, ReferenceIntegrityError) as exc:
            state["stages"][stage.name]["status"] = "blocked_data"
            state["status"] = "blocked_data"
            state["lifecycle_status"] = "stopped"
            state["failed_stage"] = stage.name
            state["reason_code"] = exc.reason_code
            state["updated_at"] = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            _save_state(state, run_dir, report_dir)
            return state
        if stage.name in {"build_pit_csi800", "download_robust_reference"}:
            _save_state(state, run_dir, report_dir)
        if stage.name == "prepare_robust_provider":
            state["provider_uri"] = evidence.get("provider_uri")
            state["provider_key"] = evidence.get("provider_key")
            state["provider_tree_sha256"] = evidence.get("provider_tree_sha256")
            _save_state(state, run_dir, report_dir)

        if stage.name == "preflight_robust_data" and evidence.get("status") not in {
            None,
            "ready",
        }:
            state["status"] = "blocked_data"
            state["lifecycle_status"] = "stopped"
            state["failed_stage"] = stage.name
            state["reason_code"] = "data_preflight_report_invalid"
            _save_state(state, run_dir, report_dir)
            return state

    development = _stage_evidence(
        stages[-1],
        report_dir,
        quant_data_dir,
        config,
        run_id=run_id,
        state=state,
    )
    final_status = development.get("status")
    if final_status not in FINAL_STATUSES or final_status in {
        "blocked_data",
        "development_failed",
    }:
        final_status = "development_failed"
    with _final_identity_guard(
        state=state,
        config_path=config_path,
        quant_data_dir=quant_data_dir,
        pit_dir=pit_dir,
        reference_dir=reference_dir,
        membership_start=membership_start,
        as_of_date=as_of_date,
        git_state_provider=git_state_provider,
    ) as identity_failure:
        if identity_failure is not None:
            state["status"] = "blocked_data"
            state["lifecycle_status"] = "stopped"
            state["reason_code"] = identity_failure
            state["failed_stage"] = "final_identity_check"
        else:
            state["status"] = final_status
            state["lifecycle_status"] = (
                "completed" if final_status != "development_failed" else "stopped"
            )
            state["reason_code"] = development.get("reason_code")
            state.pop("failed_stage", None)
        state.pop("current_stage", None)
        state["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _save_state(state, run_dir, report_dir)
    return state


def run_pipeline(
    config_path: Path = DEFAULT_CONFIG,
    *,
    quant_data_dir: Path = DEFAULT_QUANT_DATA,
    report_dir: Path = DEFAULT_REPORT_DIR,
    as_of_date: str | None = None,
    run_id: str | None = None,
    skip_membership_build: bool = False,
    resume: bool = False,
    allow_legacy_compatibility: bool = False,
    stage_runner: Callable[[Stage], int] | None = None,
    git_state_provider: Callable[[], dict[str, Any]] = git_worktree_state,
) -> dict[str, Any]:
    claimed_run_id = run_id or new_run_id()
    if not RUN_ID_RE.fullmatch(claimed_run_id):
        raise ValueError("run_id contains unsupported characters")
    resolved_report_dir = Path(report_dir).expanduser().resolve()
    with _claim_run(resolved_report_dir, claimed_run_id, resume=resume):
        return _run_pipeline_claimed(
            config_path,
            quant_data_dir=quant_data_dir,
            report_dir=resolved_report_dir,
            as_of_date=as_of_date,
            run_id=claimed_run_id,
            skip_membership_build=skip_membership_build,
            resume=resume,
            allow_legacy_compatibility=allow_legacy_compatibility,
            stage_runner=stage_runner,
            git_state_provider=git_state_provider,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the independent robust PIT CSI800 research workflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=DEFAULT_QUANT_DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-membership-build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-legacy-compatibility",
        action="store_true",
        help=(
            "explicitly replay a historical Qlib Provider; never use for a normal live run"
        ),
    )
    args = parser.parse_args(argv)
    result = run_pipeline(
        args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        as_of_date=args.as_of_date,
        run_id=args.run_id,
        skip_membership_build=args.skip_membership_build,
        resume=args.resume,
        allow_legacy_compatibility=args.allow_legacy_compatibility,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0
        if result["status"]
        in {
            "research_candidate_ready",
            "shadow_observation",
            "robust_strategy_qualified",
        }
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
