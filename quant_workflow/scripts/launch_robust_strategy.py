#!/usr/bin/env python3
"""Launch the robust strategy orchestrator as a durable local process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file  # noqa: E402
from _paths import mlruns_root, quant_data_root, reports_root, workflow_root  # noqa: E402
from run_robust_strategy import (  # noqa: E402
    DEFAULT_CONFIG,
    RUN_ID_RE,
    load_config,
    new_run_id,
)


LAUNCHCTL = "/bin/launchctl"
ACTIVE_LAUNCH_STATUSES = {"starting", "running"}
TERMINAL_LIFECYCLE_STATUSES = {"cancelled", "completed", "failed", "stopped"}
FAILED_LAUNCH_STATUSES = {"failed", "failed_to_start", "stale", "stopped"}
WORKER_ENV_NAMES = (
    "HOME",
    "LANG",
    "LC_ALL",
    "NO_COLOR",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _config_requests_mps(config: Path) -> bool:
    path = Path(config).expanduser().resolve()
    if not path.is_file():
        return False
    payload = load_config(path)
    return "mps_dnn" in {str(item.get("id")) for item in payload.get("models") or []}


def _mps_probe() -> dict[str, Any]:
    import torch

    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    if not built or not available:
        raise RuntimeError("MPS is required but unavailable in the launchd worker")
    tensor = torch.ones(4, device="mps")
    result = float((tensor * tensor).sum().cpu())
    if result != 4.0:
        raise RuntimeError("MPS launchd worker tensor probe returned an invalid result")
    return {
        "python_executable": str(Path(sys.executable).absolute()),
        "python_resolved": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "mps_built": built,
        "mps_available": available,
        "device": str(tensor.device),
        "tensor_sum": result,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


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


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            import psutil
        except ImportError:
            return True
        try:
            return psutil.Process(pid).is_running()
        except psutil.NoSuchProcess:
            return False
        except (psutil.Error, OSError, ValueError):
            return True  # Do not replace a runner whose status is unreadable.
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def launch_result_exit_code(result: Mapping[str, Any]) -> int:
    try:
        return_code = int(result.get("return_code") or 0)
    except (TypeError, ValueError):
        return_code = 0
    if return_code != 0:
        return return_code if 0 < return_code <= 255 else 1
    if str(result.get("status") or "") in FAILED_LAUNCH_STATUSES:
        return 1
    return 0


@contextmanager
def _claim_launch_record(
    report_dir: Path, run_id: str, *, runner_namespace: str = "runner"
) -> Iterator[None]:
    lock_path = (
        Path(report_dir) / runner_namespace / ".locks" / f"{run_id}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        try:
            lock_file(stream, exclusive=True, blocking=False)
        except BlockingIOError as exc:
            raise ValueError(f"robust strategy launch {run_id!r} is already running") from exc
        try:
            yield
        finally:
            unlock_file(stream)


def _launchd_label(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:24]
    return f"com.dsa.robust-strategy.{digest}"


def _launchd_job_info(
    domain: str,
    label: str,
    *,
    launchctl_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any] | None:
    completed = launchctl_runner(
        [LAUNCHCTL, "print", f"{domain}/{label}"],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if int(completed.returncode) != 0:
        return None
    output = str(completed.stdout or "")
    state_match = re.search(r"(?m)^\s*state\s*=\s*([^\s]+)\s*$", output)
    pid_match = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", output)
    return {
        "state": state_match.group(1) if state_match else None,
        "pid": int(pid_match.group(1)) if pid_match else None,
    }


def _write_pipeline_state(path: Path, state: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def cancel_launchd_runner(
    *,
    report_dir: Path,
    run_id: str,
    runner_namespace: str = "runner",
    terminal_state_path: Path | None = None,
    launchctl_runner: Callable[..., Any] = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    pid_alive: Callable[[int], bool] = _pid_alive,
    uid: int | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Stop an owned launchd job and verify that its recorded child is gone."""

    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", runner_namespace) is None:
        raise ValueError("runner_namespace contains unsupported characters")
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be nonnegative")

    report_dir = Path(report_dir).expanduser().resolve()
    runner_dir = report_dir / runner_namespace / run_id
    launch_path = runner_dir / "launch.json"
    pid_path = runner_dir / "pid"
    expected_domain = f"gui/{os.getuid() if uid is None else int(uid)}"
    expected_label = _launchd_label(run_id)

    def preserve_terminal_state(
        launch_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        if terminal_state_path is None:
            return None
        run_state = _read_json(Path(terminal_state_path))
        lifecycle = str(run_state.get("lifecycle_status") or "")
        if run_state.get("run_id") != run_id or lifecycle not in TERMINAL_LIFECYCLE_STATUSES:
            return None
        reconciled = {
            **launch_record,
            "status": lifecycle,
            "cancel_outcome": "already_terminal",
            "observed_run_status": run_state.get("status"),
            "reconciled_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        _write_pipeline_state(launch_path, reconciled)
        return reconciled

    with _claim_launch_record(
        report_dir, run_id, runner_namespace=runner_namespace
    ):
        if launch_path.is_symlink() or not launch_path.is_file():
            raise RuntimeError("launchd runner ownership record is unavailable")
        launch_record = _read_json(launch_path)
        if (
            launch_record.get("run_id") != run_id
            or launch_record.get("launch_backend") != "launchd"
            or launch_record.get("launchd_domain") != expected_domain
            or launch_record.get("job_label") != expected_label
        ):
            raise RuntimeError("launchd runner ownership could not be verified")

        already_terminal = preserve_terminal_state(launch_record)
        if already_terminal is not None:
            return already_terminal

        try:
            record_pid = int(launch_record.get("pid") or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("launchd runner ownership has an invalid child PID") from exc
        child_pid = record_pid
        if pid_path.exists() or pid_path.is_symlink():
            if pid_path.is_symlink() or not pid_path.is_file():
                raise RuntimeError("launchd runner ownership PID record is invalid")
            try:
                child_pid = int(pid_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "launchd runner ownership PID record is invalid"
                ) from exc
            if record_pid > 0 and child_pid != record_pid:
                raise RuntimeError("launchd runner ownership PID no longer matches")
        if child_pid <= 0:
            raise RuntimeError("launchd runner ownership has no recorded child PID")

        job = _launchd_job_info(
            expected_domain,
            expected_label,
            launchctl_runner=launchctl_runner,
        )
        recorded_launchd_pid = int(launch_record.get("launchd_pid") or 0)
        current_launchd_pid = int((job or {}).get("pid") or 0)
        if (
            job is not None
            and recorded_launchd_pid > 0
            and current_launchd_pid > 0
            and current_launchd_pid != recorded_launchd_pid
        ):
            raise RuntimeError("launchd runner ownership PID no longer matches")
        if job is None and pid_alive(child_pid):
            raise RuntimeError(
                "launchd runner ownership is missing; refusing to signal an orphan PID"
            )

        bootout_return_code: int | None = None
        if job is not None:
            completed = launchctl_runner(
                [LAUNCHCTL, "bootout", f"{expected_domain}/{expected_label}"],
                text=True,
                capture_output=True,
                check=False,
                timeout=15,
            )
            bootout_return_code = int(completed.returncode)

        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            remaining_job = _launchd_job_info(
                expected_domain,
                expected_label,
                launchctl_runner=launchctl_runner,
            )
            child_still_alive = pid_alive(child_pid)
            if remaining_job is None and not child_still_alive:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "launchd runner cancellation could not be confirmed"
                )
            sleep_fn(0.1)

        if bootout_return_code is None or bootout_return_code != 0:
            already_terminal = preserve_terminal_state(_read_json(launch_path))
            if already_terminal is not None:
                return already_terminal

        cancelled_at = datetime.now().astimezone().isoformat(timespec="seconds")
        cancelled = {
            **launch_record,
            "status": "cancelled",
            "reason_code": "user_cancelled",
            "bootout_return_code": bootout_return_code,
            "cancelled_at": cancelled_at,
            "verified_stopped_at": cancelled_at,
        }
        _write_pipeline_state(launch_path, cancelled)
        return cancelled


def _reconcile_existing_runner(
    runner_dir: Path,
    *,
    report_dir: Path,
    run_id: str,
    pid_alive: Callable[[int], bool],
    launchd_probe: Callable[[str, str], dict[str, Any] | None],
    update_pipeline_state: bool = True,
) -> bool:
    launch_path = runner_dir / "launch.json"
    launch_record = _read_json(launch_path)
    if launch_record.get("status") not in ACTIVE_LAUNCH_STATUSES:
        return False
    pid_path = runner_dir / "pid"
    try:
        recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        recorded_pid = int(launch_record.get("pid") or 0)

    job_label = launch_record.get("job_label")
    domain = launch_record.get("launchd_domain")
    if isinstance(job_label, str) and isinstance(domain, str):
        job = launchd_probe(domain, job_label)
        launchd_pid = int((job or {}).get("pid") or 0)
        active = bool(
            job
            and job.get("state") == "running"
            and (
                (recorded_pid > 0 and pid_alive(recorded_pid))
                or (launchd_pid > 0 and pid_alive(launchd_pid))
            )
        )
    else:
        active = recorded_pid > 0 and pid_alive(recorded_pid)
    if active:
        return True

    if not update_pipeline_state:
        return False
    run_state_path = report_dir / "runs" / run_id / "status.json"
    run_state = _read_json(run_state_path)
    lifecycle = run_state.get("lifecycle_status")
    reconciled_status = lifecycle if lifecycle in TERMINAL_LIFECYCLE_STATUSES else "stale"
    reconciled = {
        **launch_record,
        "status": reconciled_status,
        "reconciled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_pipeline_state(launch_path, reconciled)

    if run_state and lifecycle not in TERMINAL_LIFECYCLE_STATUSES:
        run_state.update(
            {
                "lifecycle_status": "stopped",
                "status": "blocked_data",
                "reason_code": "runner_process_missing",
                "updated_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        )
        _write_pipeline_state(run_state_path, run_state)
        latest_path = report_dir / "robust_strategy_status_latest.json"
        if _read_json(latest_path).get("run_id") == run_id:
            _write_pipeline_state(latest_path, run_state)
    return False


def _worker_program_arguments(
    *,
    launch_path: Path,
    pid_path: Path,
    report_dir: Path,
    run_id: str,
    command: list[str],
    environment: dict[str, str],
) -> list[str]:
    return [
        "/usr/bin/env",
        "-i",
        *(f"{key}={value}" for key, value in sorted(environment.items())),
        sys.executable,
        str(Path(__file__).resolve()),
        "--launchd-worker",
        "--launch-path",
        str(launch_path),
        "--pid-path",
        str(pid_path),
        "--report-dir",
        str(report_dir),
        "--run-id",
        run_id,
        "--",
        *command,
    ]


def _sanitized_worker_environment(
    source: Mapping[str, str] | None = None,
    *,
    mlruns_dir: Path | None = None,
) -> dict[str, str]:
    current = os.environ if source is None else source
    environment = {
        name: str(current[name])
        for name in WORKER_ENV_NAMES
        if current.get(name)
    }
    environment.setdefault("HOME", str(Path.home()))
    environment.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    configured_mlruns = mlruns_dir or Path(
        current.get("QUANT_WORKFLOW_MLRUNS_DIR") or mlruns_root()
    )
    environment["QUANT_WORKFLOW_MLRUNS_DIR"] = str(
        Path(configured_mlruns).expanduser().resolve()
    )
    return environment


def _write_launchd_plist(
    path: Path,
    *,
    label: str,
    program_arguments: list[str],
    log_path: Path,
    error_log_path: Path,
) -> None:
    payload = {
        "Label": label,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(workflow_root()),
        "RunAtLoad": True,
        "KeepAlive": False,
        "LaunchOnlyOnce": True,
        "ProcessType": "Standard",
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(error_log_path),
    }
    _atomic_write_bytes(path, plistlib.dumps(payload, sort_keys=True))


def run_launchd_worker(
    *,
    launch_path: Path,
    pid_path: Path,
    report_dir: Path,
    run_id: str,
    command: list[str],
    popen_factory: Callable[..., Any] = subprocess.Popen,
    mps_probe: Callable[[], dict[str, Any]] = _mps_probe,
) -> int:
    launch_record = _read_json(launch_path)
    if launch_record.get("mps_required") is True:
        try:
            mps_evidence = mps_probe()
        except Exception as exc:
            _write_pipeline_state(
                launch_path,
                {
                    **launch_record,
                    "status": "failed_to_start",
                    "reason_code": "mps_preflight_failed",
                    "error_type": type(exc).__name__,
                    "finished_at": datetime.now()
                    .astimezone()
                    .isoformat(timespec="seconds"),
                },
            )
            return 78
        launch_record = {**launch_record, "mps_preflight": mps_evidence}
        _write_pipeline_state(launch_path, launch_record)
    try:
        process = popen_factory(
            command,
            cwd=str(workflow_root()),
            env=_sanitized_worker_environment(),
            stdin=subprocess.DEVNULL,
            start_new_session=False,
            close_fds=True,
            shell=False,
        )
    except OSError as exc:
        _write_pipeline_state(
            launch_path,
            {
                **launch_record,
                "status": "failed_to_start",
                "error_type": type(exc).__name__,
                "finished_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
            },
        )
        return 127

    child_pid = int(process.pid)
    _atomic_write(pid_path, f"{child_pid}\n")
    _write_pipeline_state(
        launch_path,
        {
            **launch_record,
            "status": "running",
            "pid": child_pid,
            "worker_pid": os.getpid(),
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    return_code = int(process.wait())
    run_state = _read_json(report_dir / "runs" / run_id / "status.json")
    lifecycle = run_state.get("lifecycle_status")
    if lifecycle in TERMINAL_LIFECYCLE_STATUSES:
        status = str(lifecycle)
    else:
        status = "completed" if return_code == 0 else "failed"
    _write_pipeline_state(
        launch_path,
        {
            **_read_json(launch_path),
            "status": status,
            "return_code": return_code,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
    )
    return return_code


def _launch_claimed(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    mlruns_dir: Path,
    as_of_date: str,
    run_id: str,
    resume: bool,
    skip_membership_build: bool,
    warmup_seconds: float = 10.0,
    launchctl_runner: Callable[..., Any] = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    cpu_probe: Callable[[int], float | None] = _cpu_percent,
    pid_alive: Callable[[int], bool] = _pid_alive,
    uid: int | None = None,
    runner_namespace: str = "runner",
    command_override: list[str] | None = None,
    mps_required_override: bool | None = None,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds must be nonnegative")
    report_dir = Path(report_dir).expanduser().resolve()
    runner_dir = report_dir / runner_namespace / run_id
    runner_dir.mkdir(parents=True, exist_ok=True)
    log_path = runner_dir / "runner.log"
    error_log_path = runner_dir / "runner.err.log"
    pid_path = runner_dir / "pid"
    launch_path = runner_dir / "launch.json"
    plist_path = runner_dir / "launchd.plist"
    domain = f"gui/{os.getuid() if uid is None else int(uid)}"
    job_label = _launchd_label(run_id)
    command = list(command_override) if command_override is not None else build_command(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        as_of_date=as_of_date,
        run_id=run_id,
        resume=resume,
        skip_membership_build=skip_membership_build,
    )
    mps_required = (
        bool(mps_required_override)
        if mps_required_override is not None
        else _config_requests_mps(config)
    )
    initial = {
        "status": "starting",
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "log_path": str(log_path),
        "error_log_path": str(error_log_path),
        "pid_path": str(pid_path),
        "plist_path": str(plist_path),
        "launch_backend": "launchd",
        "launchd_domain": domain,
        "job_label": job_label,
        "mps_required": mps_required,
    }
    _atomic_write(
        launch_path,
        json.dumps(initial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    log_path.touch(exist_ok=True)
    error_log_path.touch(exist_ok=True)
    _write_launchd_plist(
        plist_path,
        label=job_label,
        program_arguments=_worker_program_arguments(
            launch_path=launch_path,
            pid_path=pid_path,
            report_dir=report_dir,
            run_id=run_id,
            command=command,
            environment=_sanitized_worker_environment(mlruns_dir=mlruns_dir),
        ),
        log_path=log_path,
        error_log_path=error_log_path,
    )
    completed = launchctl_runner(
        [LAUNCHCTL, "bootstrap", domain, str(plist_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if int(completed.returncode) != 0:
        failed = {
            **initial,
            "status": "failed_to_start",
            "launchctl_return_code": int(completed.returncode),
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_pipeline_state(launch_path, failed)
        raise RuntimeError("launchd rejected the robust strategy job")

    sleep_fn(float(warmup_seconds))
    current = _read_json(launch_path)
    if current.get("status") not in ACTIVE_LAUNCH_STATUSES:
        return current
    job = _launchd_job_info(domain, job_label, launchctl_runner=launchctl_runner)
    launchd_pid = int((job or {}).get("pid") or 0)
    try:
        child_pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        child_pid = launchd_pid
    if (
        not job
        or job.get("state") != "running"
        or child_pid <= 0
        or not pid_alive(child_pid)
    ):
        failed = {
            **initial,
            "status": "failed_to_start",
            "pid": child_pid or None,
            "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_pipeline_state(launch_path, failed)
        raise RuntimeError("launchd job was not running after warmup")
    result = {
        **current,
        "status": "running",
        "pid": child_pid,
        "launchd_pid": launchd_pid,
        "cpu_percent": cpu_probe(child_pid),
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _atomic_write(pid_path, f"{child_pid}\n")
    _write_pipeline_state(launch_path, result)
    return result


def launch(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    mlruns_dir: Path | None = None,
    as_of_date: str,
    run_id: str,
    resume: bool,
    skip_membership_build: bool,
    warmup_seconds: float = 10.0,
    launchctl_runner: Callable[..., Any] = subprocess.run,
    sleep_fn: Callable[[float], None] = time.sleep,
    cpu_probe: Callable[[int], float | None] = _cpu_percent,
    pid_alive: Callable[[int], bool] = _pid_alive,
    uid: int | None = None,
    runner_namespace: str = "runner",
    command_override: list[str] | None = None,
    mps_required_override: bool | None = None,
    update_pipeline_state_on_reconcile: bool = True,
) -> dict[str, Any]:
    report_dir = Path(report_dir).expanduser().resolve()
    resolved_mlruns_dir = Path(mlruns_dir or mlruns_root()).expanduser().resolve()
    runner_dir = report_dir / runner_namespace / run_id

    def launchd_probe(domain: str, label: str) -> dict[str, Any] | None:
        return _launchd_job_info(
            domain, label, launchctl_runner=launchctl_runner
        )

    with _claim_launch_record(
        report_dir, run_id, runner_namespace=runner_namespace
    ):
        if _reconcile_existing_runner(
            runner_dir,
            report_dir=report_dir,
            run_id=run_id,
            pid_alive=pid_alive,
            launchd_probe=launchd_probe,
            update_pipeline_state=update_pipeline_state_on_reconcile,
        ):
            raise ValueError(f"robust strategy run {run_id!r} is already running")
        if not resume and (report_dir / "runs" / run_id / "status.json").exists():
            raise ValueError(f"robust strategy run {run_id!r} already exists; use resume")
        previous = _read_json(runner_dir / "launch.json")
        previous_label = previous.get("job_label")
        previous_domain = previous.get("launchd_domain")
        if isinstance(previous_label, str) and isinstance(previous_domain, str):
            launchctl_runner(
                [LAUNCHCTL, "bootout", f"{previous_domain}/{previous_label}"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
        return _launch_claimed(
            config=config,
            quant_data_dir=quant_data_dir,
            report_dir=report_dir,
            mlruns_dir=resolved_mlruns_dir,
            as_of_date=as_of_date,
            run_id=run_id,
            resume=resume,
            skip_membership_build=skip_membership_build,
            warmup_seconds=warmup_seconds,
            launchctl_runner=launchctl_runner,
            sleep_fn=sleep_fn,
            cpu_probe=cpu_probe,
            pid_alive=pid_alive,
            uid=uid,
            runner_namespace=runner_namespace,
            command_override=command_override,
            mps_required_override=mps_required_override,
        )


def _launchd_worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--launch-path", type=Path, required=True)
    parser.add_argument("--pid-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("launchd worker command is required")
    return run_launchd_worker(
        launch_path=args.launch_path,
        pid_path=args.pid_path,
        report_dir=args.report_dir,
        run_id=args.run_id,
        command=command,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--launchd-worker":
        return _launchd_worker_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Launch robust strategy training outside the terminal session."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument(
        "--report-dir", type=Path, default=reports_root() / "robust_strategy"
    )
    parser.add_argument("--mlruns-dir", type=Path, default=mlruns_root())
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-membership-build", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    run_id = args.run_id or new_run_id()
    result = launch(
        config=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        mlruns_dir=args.mlruns_dir,
        as_of_date=args.as_of_date,
        run_id=run_id,
        resume=args.resume,
        skip_membership_build=args.skip_membership_build,
        warmup_seconds=args.warmup_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return launch_result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
