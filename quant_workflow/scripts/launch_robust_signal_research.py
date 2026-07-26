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
from launch_robust_strategy import launch  # noqa: E402
from run_robust_signal_research import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_REPORT_DIR,
    request_pause,
)


def new_run_id() -> str:
    return f"robust_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_command(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    run_id: str,
    resume: bool,
    smoke: bool,
    no_mlflow: bool,
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
) -> dict[str, Any]:
    command = build_command(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        run_id=run_id,
        resume=resume,
        smoke=smoke,
        no_mlflow=no_mlflow,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--mlruns-dir", type=Path, default=mlruns_root())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    run_id = args.run_id or new_run_id()
    if args.request_pause:
        result = request_pause(report_dir, run_id)
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
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
