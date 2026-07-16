#!/usr/bin/env python3
"""Independent, resumable orchestrator for the robust CSI800 strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402


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


class Stage(NamedTuple):
    name: str
    argv: list[str]


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
    _require_at_least(payload, ("acceptance", "shadow_task_success_ratio_min"), 0.95)
    _require_at_most(payload, ("acceptance", "development_max_drawdown_abs_max"), 0.20)
    _require_at_most(payload, ("acceptance", "confirmation_max_drawdown_abs_max"), 0.20)
    _require_at_most(payload, ("acceptance", "shadow_max_drawdown_abs_max"), 0.12)
    if payload.get("training", {}).get("windows_trading_days") != [504, 756, 1008]:
        raise ValueError("training.windows_trading_days must remain [504, 756, 1008]")
    if payload.get("training", {}).get("tree_seeds") != [42, 2026, 3407]:
        raise ValueError("training.tree_seeds must remain [42, 2026, 3407]")
    if len(payload.get("rolling_validation", {}).get("test_quarters") or []) != 12:
        raise ValueError("rolling_validation must contain 12 development quarters")
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
) -> list[Stage]:
    paths = config["paths"]
    universe = config["universe"]
    gates = config["data_gates"]
    pit_dir = _resolved_data_path(quant_data_dir, paths["pit_membership"])
    reference_dir = _resolved_data_path(
        quant_data_dir, str(Path(paths["security_master"]).parent)
    )
    price_dir = _resolved_data_path(quant_data_dir, paths["price_parquet"])
    provider_dir = _resolved_data_path(quant_data_dir, paths["qlib_provider"])
    runtime_root = quant_data_dir / "runtime" / "robust_strategy"
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
                    str(report_dir),
                ],
            )
        )
    stages.append(
        Stage(
            "download_robust_reference",
            [
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
                str(report_dir),
            ],
        )
    )
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
                str(report_dir),
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
                str(report_dir),
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
        str(report_dir / "provider_latest.json"),
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


def _stage_evidence(
    stage: Stage, report_dir: Path, quant_data_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    if stage.name == "build_pit_csi800":
        report_payload = _read_json(report_dir / "build_status_latest.json")
        if report_payload:
            return report_payload
        pit_dir = _resolved_data_path(quant_data_dir, config["paths"]["pit_membership"])
        return _read_json(pit_dir / "build_status_latest.json")
    if stage.name == "download_robust_reference":
        return _read_json(report_dir / "download_robust_reference_latest.json")
    if stage.name == "preflight_robust_data":
        return _read_json(report_dir / "preflight_latest.json")
    if stage.name == "prepare_robust_provider":
        return _read_json(report_dir / "provider_latest.json")
    if stage.name == "run_robust_development":
        return _read_json(report_dir / "development_latest.json")
    return {}


def _default_stage_runner(stage: Stage) -> int:
    completed = subprocess.run(stage.argv, cwd=WORKFLOW_ROOT, check=False)
    return int(completed.returncode)


def _save_state(state: dict[str, Any], run_dir: Path, report_dir: Path) -> None:
    serialized = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(run_dir / "status.json", serialized)
    _atomic_write(report_dir / "robust_strategy_status_latest.json", serialized)


def run_pipeline(
    config_path: Path = DEFAULT_CONFIG,
    *,
    quant_data_dir: Path = DEFAULT_QUANT_DATA,
    report_dir: Path = DEFAULT_REPORT_DIR,
    as_of_date: str | None = None,
    run_id: str | None = None,
    skip_membership_build: bool = False,
    resume: bool = False,
    stage_runner: Callable[[Stage], int] | None = None,
    git_state_provider: Callable[[], dict[str, Any]] = git_worktree_state,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    as_of_date = str(as_of_date or date.today().isoformat())
    if as_of_date > date.today().isoformat():
        raise ValueError("as_of_date must not be in the future")
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    quant_data_dir = Path(quant_data_dir).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    run_dir = report_dir / "runs" / run_id
    status_path = run_dir / "status.json"
    config_hash = file_sha256(config_path)
    current_git = dict(git_state_provider())
    manifest = quant_data_dir / "DATA_MANIFEST.json"
    current_manifest_hash = file_sha256(manifest) if manifest.is_file() else None

    if resume and status_path.is_file():
        state = _read_json(status_path)
        if state.get("config_hash") != config_hash:
            raise ValueError("cannot resume a run with a different config hash")
        if state.get("git_commit") != current_git.get("commit"):
            raise ValueError("cannot resume a run from a different git commit")
        if state.get("data_manifest_hash") != current_manifest_hash:
            raise ValueError("cannot resume a run with a different data manifest")
        if state.get("as_of_date") != as_of_date:
            raise ValueError("cannot resume a run with a different as-of date")
    else:
        state = {
            "run_id": run_id,
            "status": "blocked_data",
            "lifecycle_status": "running",
            "reason_code": "preflight_not_completed",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "config_hash": config_hash,
            "git_commit": current_git.get("commit"),
            "data_manifest_hash": current_manifest_hash,
            "as_of_date": as_of_date,
            "stages": {},
        }
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
    )
    runner = stage_runner or _default_stage_runner
    for stage in stages:
        previous = state.get("stages", {}).get(stage.name)
        if (
            resume
            and isinstance(previous, dict)
            and previous.get("status") == "completed"
        ):
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
        try:
            return_code = int(runner(stage))
        except (OSError, subprocess.SubprocessError) as exc:
            return_code = 127
            evidence: dict[str, Any] = {
                "reason_code": "stage_launch_failed",
                "error_type": type(exc).__name__,
            }
        else:
            evidence = _stage_evidence(stage, report_dir, quant_data_dir, config)

        stage_status = "completed" if return_code == 0 else "failed"
        if evidence.get("status") == "blocked_data":
            stage_status = "blocked_data"
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
        if return_code != 0 or evidence.get("status") == "blocked_data":
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

    development = _stage_evidence(stages[-1], report_dir, quant_data_dir, config)
    final_status = development.get("status")
    if final_status not in FINAL_STATUSES or final_status in {
        "blocked_data",
        "development_failed",
    }:
        final_status = "development_failed"
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
    args = parser.parse_args(argv)
    result = run_pipeline(
        args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        as_of_date=args.as_of_date,
        run_id=args.run_id,
        skip_membership_build=args.skip_membership_build,
        resume=args.resume,
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
