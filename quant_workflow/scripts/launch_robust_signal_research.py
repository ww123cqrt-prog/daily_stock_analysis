#!/usr/bin/env python3
"""Launch, pause, or resume robust signal research through launchd."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, quant_data_root, reports_root  # noqa: E402
from launch_robust_strategy import (  # noqa: E402
    cancel_launchd_runner,
    launch,
    launch_result_exit_code,
)
from robust_signal_data_contract import (  # noqa: E402
    RobustSignalDataBlocked,
    resolve_robust_signal_data_readiness,
)
from run_robust_signal_research import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REPORT_DIR,
    _read_json,
    _write_json,
    build_context,
    build_jobs,
    load_config,
    request_pause,
)


def new_run_id() -> str:
    return f"robust_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_plan_payload(
    *, config: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    screening_jobs = build_jobs(config, phase="screening")
    maximum_candidates = max(0, int(config.get("screening", {}).get("maximum_candidates") or 0))
    candidate_ids = [
        str(item["id"])
        for item in list(config.get("signals") or [])[:maximum_candidates]
    ]
    validation_jobs = build_jobs(
        config,
        phase="validation",
        signal_ids=candidate_ids,
    )
    return {
        "status": "plan_ready",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "training_started": False,
        "screening": {
            "quarters": list(config["periods"]["screening_quarters"]),
            "signal_count": len(config.get("signals") or []),
            "model_count": len(config.get("models", {}).get("screening") or []),
            "job_count": len(screening_jobs),
        },
        "validation": {
            "quarters": list(config["periods"]["validation_quarters"]),
            "maximum_candidates": maximum_candidates,
            "model_count": len(config.get("models", {}).get("validation") or []),
            "window_days": list(config["periods"]["validation_windows_days"]),
            "maximum_job_count": len(validation_jobs),
        },
        "locked_confirmation_start": str(
            config.get("safety", {}).get("locked_confirmation_start")
            or context.get("locked_confirmation_start")
            or "2026-01-01"
        ),
        "config_hash": context.get("config_hash"),
        "git_commit": context.get("git_commit"),
        "git_clean": context.get("git_clean"),
        "data_manifest_hash": context.get("data_manifest_hash"),
        "provider_tree_sha256": context.get("provider_tree_sha256"),
        "context_hash": context.get("context_hash"),
    }


def preview_signal_research_plan(
    *, config_path: Path, quant_data_dir: Path, report_dir: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    try:
        readiness_artifact, readiness_sha256, _readiness = (
            resolve_robust_signal_data_readiness(
                report_dir / "robust_signal_data_readiness_latest.json"
            )
        )
        context = build_context(
            config_path=config_path,
            config=config,
            quant_data_dir=quant_data_dir,
            report_root=reports_root(),
            readiness_artifact_path=readiness_artifact,
            readiness_artifact_sha256=readiness_sha256,
        )
        payload = build_plan_payload(config=config, context=context)
    except RobustSignalDataBlocked as exc:
        reason_code = exc.reason_code
        payload = {
            "status": "blocked_data",
            "reason_code": reason_code,
            "summary": str(getattr(exc, "summary", "Robust CSI800 data is blocked.")),
            "summary_zh": str(getattr(exc, "summary_zh", str(exc))),
            "training_started": False,
        }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _write_json(report_dir / "robust_signal_plan_latest.json", payload)
    _write_json(report_dir / "plans" / f"robust_signal_plan_{timestamp}.json", payload)
    return payload


def build_command(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    run_id: str,
    resume: bool,
    smoke: bool,
    no_mlflow: bool,
    readiness_artifact: Path,
    readiness_artifact_sha256: str,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_robust_signal_research.py"),
        "--config",
        str(config.resolve()),
        "--quant-data-dir",
        str(quant_data_dir.resolve()),
        "--report-dir",
        str(report_dir.resolve()),
        "--run-id",
        run_id,
        "--readiness-artifact",
        str(readiness_artifact.resolve()),
        "--readiness-artifact-sha256",
        readiness_artifact_sha256,
    ]
    if resume:
        command.append("--resume")
    if smoke:
        command.append("--smoke")
    if no_mlflow:
        command.append("--no-mlflow")
    return command


def launch_signal_research(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    mlruns_dir: Path,
    run_id: str,
    resume: bool,
    smoke: bool,
    no_mlflow: bool,
    warmup_seconds: float,
    readiness_artifact: Path,
    readiness_artifact_sha256: str,
) -> dict[str, Any]:
    command = build_command(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        run_id=run_id,
        resume=resume,
        smoke=smoke,
        no_mlflow=no_mlflow,
        readiness_artifact=readiness_artifact,
        readiness_artifact_sha256=readiness_artifact_sha256,
    )
    return launch(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        mlruns_dir=mlruns_dir,
        as_of_date=date.today().isoformat(),
        run_id=run_id,
        resume=resume,
        skip_membership_build=True,
        warmup_seconds=warmup_seconds,
        runner_namespace="runner",
        command_override=command,
        mps_required_override=False,
        update_pipeline_state_on_reconcile=False,
    )


def cancel_signal_research(*, report_dir: Path, run_id: str) -> dict[str, Any]:
    report_dir = Path(report_dir).expanduser().resolve()
    status_path = report_dir / "runs" / run_id / "status.json"
    runner = cancel_launchd_runner(
        report_dir=report_dir,
        run_id=run_id,
        runner_namespace="runner",
        terminal_state_path=status_path,
    )
    if runner.get("cancel_outcome") == "already_terminal":
        terminal_status = _read_json(status_path)
        if (
            terminal_status.get("run_id") != run_id
            or terminal_status.get("lifecycle_status") not in {"cancelled", "completed", "failed", "stopped"}
        ):
            raise RuntimeError("terminal signal research status could not be verified")
        return {**terminal_status, "cancel_outcome": "already_terminal"}
    if runner.get("status") != "cancelled":
        raise RuntimeError("durable signal runner cancellation was not confirmed")

    status = _read_json(status_path)
    if status.get("run_id") != run_id:
        raise RuntimeError("signal research run status does not match the cancelled runner")
    cancelled_at = datetime.now().astimezone().isoformat(timespec="seconds")
    status.update(
        {
            "status": "cancelled",
            "lifecycle_status": "cancelled",
            "pause_state": "cancelled",
            "reason_code": "user_cancelled",
            "next_gate": "start_new_run",
            "updated_at": cancelled_at,
            "cancelled_at": cancelled_at,
        }
    )
    _write_json(status_path, status)
    latest_path = report_dir / "robust_signal_status_latest.json"
    if _read_json(latest_path).get("run_id") == run_id:
        _write_json(latest_path, status)
    return {**status, "cancel_outcome": "cancelled"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--mlruns-dir", type=Path, default=mlruns_root())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--readiness-artifact", type=Path)
    parser.add_argument("--readiness-artifact-sha256")
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if args.cancel and not args.run_id:
        parser.error("--cancel requires --run-id")
    run_id = args.run_id or new_run_id()
    if args.plan and (args.request_pause or args.cancel or args.resume or args.smoke):
        parser.error("--plan cannot be combined with pause, resume, or smoke")
    if args.cancel and (args.request_pause or args.resume or args.smoke):
        parser.error("--cancel cannot be combined with pause, resume, or smoke")
    starts_runner = not (args.plan or args.request_pause or args.cancel)
    if starts_runner and (
        args.readiness_artifact is None or not args.readiness_artifact_sha256
    ):
        parser.error(
            "starting or resuming research requires --readiness-artifact and "
            "--readiness-artifact-sha256"
        )
    if args.plan:
        result = preview_signal_research_plan(
            config_path=args.config.expanduser().resolve(),
            quant_data_dir=args.quant_data_dir.expanduser().resolve(),
            report_dir=report_dir,
        )
    elif args.request_pause:
        result = request_pause(report_dir, run_id)
    elif args.cancel:
        result = cancel_signal_research(report_dir=report_dir, run_id=run_id)
    else:
        result = launch_signal_research(
            config=args.config.expanduser().resolve(),
            quant_data_dir=args.quant_data_dir.expanduser().resolve(),
            report_dir=report_dir,
            mlruns_dir=args.mlruns_dir.expanduser().resolve(),
            run_id=run_id,
            resume=args.resume,
            smoke=args.smoke,
            no_mlflow=args.no_mlflow,
            warmup_seconds=args.warmup_seconds,
            readiness_artifact=args.readiness_artifact.expanduser().resolve(),
            readiness_artifact_sha256=args.readiness_artifact_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.plan or args.request_pause or args.cancel:
        return 0
    return launch_result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
