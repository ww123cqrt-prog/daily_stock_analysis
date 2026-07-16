#!/usr/bin/env python3
"""Launch the robust strategy orchestrator as a durable local process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402
from run_robust_strategy import DEFAULT_CONFIG, RUN_ID_RE  # noqa: E402


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build_command(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    as_of_date: str,
    run_id: str,
    resume: bool,
    skip_membership_build: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_robust_strategy.py"),
        "--config",
        str(Path(config).expanduser().resolve()),
        "--quant-data-dir",
        str(Path(quant_data_dir).expanduser().resolve()),
        "--report-dir",
        str(Path(report_dir).expanduser().resolve()),
        "--as-of-date",
        as_of_date,
        "--run-id",
        run_id,
    ]
    if resume:
        command.append("--resume")
    if skip_membership_build:
        command.append("--skip-membership-build")
    return command


def _cpu_percent(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        value = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return value if completed.returncode == 0 else None


def launch(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    as_of_date: str,
    run_id: str,
    resume: bool,
    skip_membership_build: bool,
    warmup_seconds: float = 10.0,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
    cpu_probe: Callable[[int], float | None] = _cpu_percent,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be nonnegative")
    report_dir = Path(report_dir).expanduser().resolve()
    runner_dir = report_dir / "runner" / run_id
    runner_dir.mkdir(parents=True, exist_ok=True)
    log_path = runner_dir / "runner.log"
    pid_path = runner_dir / "pid"
    launch_path = runner_dir / "launch.json"
    command = build_command(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        as_of_date=as_of_date,
        run_id=run_id,
        resume=resume,
        skip_membership_build=skip_membership_build,
    )
    initial = {
        "status": "starting",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "log_path": str(log_path),
        "pid_path": str(pid_path),
    }
    _atomic_write(
        launch_path,
        json.dumps(initial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = popen_factory(
            command,
            cwd=str(workflow_root()),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            shell=False,
        )
    finally:
        log_handle.close()
    _atomic_write(pid_path, f"{int(process.pid)}\n")
    sleep_fn(float(warmup_seconds))
    return_code = process.poll()
    if return_code is not None:
        failed = {
            **initial,
            "status": "failed_to_start",
            "pid": int(process.pid),
            "return_code": int(return_code),
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _atomic_write(
            launch_path,
            json.dumps(failed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        raise RuntimeError(
            f"robust strategy process exited during warmup with code {return_code}"
        )
    result = {
        **initial,
        "status": "running",
        "pid": int(process.pid),
        "cpu_percent": cpu_probe(int(process.pid)),
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_write(
        launch_path,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch robust strategy training outside the terminal session."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument(
        "--report-dir", type=Path, default=reports_root() / "robust_strategy"
    )
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-membership-build", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result = launch(
        config=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        as_of_date=args.as_of_date,
        run_id=run_id,
        resume=args.resume,
        skip_membership_build=args.skip_membership_build,
        warmup_seconds=args.warmup_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
