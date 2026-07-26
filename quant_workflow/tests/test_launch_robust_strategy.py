from __future__ import annotations

import importlib.util
import json
import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "launch_robust_strategy.py"
)
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_module():
    spec = importlib.util.spec_from_file_location("launch_robust_strategy", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_command_targets_foreground_orchestrator_and_preserves_resume(
    tmp_path: Path,
) -> None:
    module = load_module()

    command = module.build_command(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        as_of_date="2026-07-15",
        run_id="robust-run",
        resume=True,
        skip_membership_build=True,
    )

    assert command[1].endswith("run_robust_strategy.py")
    assert "launch_robust_strategy.py" not in " ".join(command)
    assert "--resume" in command
    assert "--skip-membership-build" in command


def test_config_requests_mps_only_for_the_unified_six_model_config() -> None:
    module = load_module()

    assert module._config_requests_mps(CONFIG_DIR / "robust_strategy.yaml") is False
    assert (
        module._config_requests_mps(CONFIG_DIR / "robust_strategy_mps_dnn.yaml")
        is True
    )


def test_launch_registers_launchd_job_and_writes_verified_record(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    monkeypatch.setenv("GITHUB_PAT_TOKEN", "must-not-reach-launchd-worker")
    commands: list[list[str]] = []

    def launchctl_runner(command, **kwargs):
        commands.append(command)
        if command[1] == "bootstrap":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[1] == "print":
            return SimpleNamespace(
                returncode=0,
                stdout="state = running\n\tpid = 43210\n",
                stderr="",
            )
        raise AssertionError(command)

    result = module.launch(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        mlruns_dir=tmp_path / "mlruns",
        as_of_date="2026-07-15",
        run_id="robust-run",
        resume=False,
        skip_membership_build=False,
        warmup_seconds=0,
        launchctl_runner=launchctl_runner,
        sleep_fn=lambda _: None,
        cpu_probe=lambda pid: 612.5,
        pid_alive=lambda pid: pid == 43210,
        uid=501,
    )

    runner_dir = tmp_path / "reports" / "runner" / "robust-run"
    record = json.loads((runner_dir / "launch.json").read_text(encoding="utf-8"))
    assert result["status"] == "running"
    assert record["pid"] == 43210
    assert record["cpu_percent"] == 612.5
    assert (runner_dir / "pid").read_text(encoding="utf-8").strip() == "43210"
    assert Path(record["log_path"]).is_file()
    assert record["launch_backend"] == "launchd"
    assert record["launchd_domain"] == "gui/501"
    assert record["job_label"].startswith("com.dsa.robust-strategy.")
    plist_path = Path(record["plist_path"])
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload["Label"] == record["job_label"]
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is False
    assert payload["LaunchOnlyOnce"] is True
    program_arguments = payload["ProgramArguments"]
    assert program_arguments[:2] == ["/usr/bin/env", "-i"]
    assert f"QUANT_WORKFLOW_MLRUNS_DIR={tmp_path / 'mlruns'}" in program_arguments
    assert "must-not-reach-launchd-worker" not in " ".join(program_arguments)
    launcher_index = next(
        index
        for index, value in enumerate(program_arguments)
        if value.endswith("launch_robust_strategy.py")
    )
    assert program_arguments[launcher_index + 1] == "--launchd-worker"
    assert commands[0][:3] == ["/bin/launchctl", "bootstrap", "gui/501"]
    assert commands[1] == [
        "/bin/launchctl",
        "print",
        f"gui/501/{record['job_label']}",
    ]


def test_launch_fails_when_launchd_rejects_job(tmp_path: Path) -> None:
    module = load_module()

    try:
        module.launch(
            config=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=tmp_path / "reports",
            as_of_date="2026-07-15",
            run_id="failed-run",
            resume=False,
            skip_membership_build=False,
            warmup_seconds=0,
            launchctl_runner=lambda *args, **kwargs: SimpleNamespace(
                returncode=5,
                stdout="",
                stderr="bootstrap failed",
            ),
            sleep_fn=lambda _: None,
            cpu_probe=lambda pid: 0.0,
            uid=501,
        )
    except RuntimeError as exc:
        assert "launchd" in str(exc)
    else:
        raise AssertionError("launchd rejection was reported as running")
    record = json.loads(
        (tmp_path / "reports" / "runner" / "failed-run" / "launch.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "failed_to_start"


def test_duplicate_active_launch_preserves_original_pid_and_record(tmp_path: Path) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"
    runner_dir = report_dir / "runner" / "same-run"
    runner_dir.mkdir(parents=True)
    (runner_dir / "pid").write_text("111\n", encoding="utf-8")
    original = {"status": "running", "run_id": "same-run", "pid": 111}
    (runner_dir / "launch.json").write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="already running"):
        module.launch(
            config=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            as_of_date="2026-07-15",
            run_id="same-run",
            resume=False,
            skip_membership_build=False,
            warmup_seconds=0,
            launchctl_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("must not register launchd job")
            ),
            sleep_fn=lambda _: None,
            cpu_probe=lambda pid: 0.0,
            pid_alive=lambda pid: pid == 111,
        )

    assert (runner_dir / "pid").read_text(encoding="utf-8") == "111\n"
    assert json.loads((runner_dir / "launch.json").read_text(encoding="utf-8")) == original


def test_reconcile_marks_dead_runner_and_running_pipeline_as_stopped(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"
    run_id = "stale-run"
    runner_dir = report_dir / "runner" / run_id
    run_dir = report_dir / "runs" / run_id
    runner_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (runner_dir / "pid").write_text("222\n", encoding="utf-8")
    (runner_dir / "launch.json").write_text(
        json.dumps(
            {
                "status": "running",
                "run_id": run_id,
                "pid": 222,
                "job_label": "com.dsa.robust-strategy.stale",
                "launchd_domain": "gui/501",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "status": "blocked_data",
        "lifecycle_status": "running",
        "reason_code": "preflight_not_completed",
        "run_id": run_id,
    }
    (run_dir / "status.json").write_text(json.dumps(state), encoding="utf-8")
    (report_dir / "robust_strategy_status_latest.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    active = module._reconcile_existing_runner(
        runner_dir,
        report_dir=report_dir,
        run_id=run_id,
        pid_alive=lambda _: False,
        launchd_probe=lambda domain, label: None,
    )

    assert active is False
    launch = json.loads((runner_dir / "launch.json").read_text(encoding="utf-8"))
    stopped = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    latest = json.loads(
        (report_dir / "robust_strategy_status_latest.json").read_text(encoding="utf-8")
    )
    assert launch["status"] == "stale"
    assert stopped["lifecycle_status"] == "stopped"
    assert stopped["reason_code"] == "runner_process_missing"
    assert latest == stopped


def test_launchd_worker_records_child_and_terminal_pipeline_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    monkeypatch.setenv("GITHUB_PAT_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("QUANT_WORKFLOW_MLRUNS_DIR", str(tmp_path / "mlruns"))
    report_dir = tmp_path / "reports"
    run_id = "worker-run"
    launch_path = report_dir / "runner" / run_id / "launch.json"
    pid_path = launch_path.parent / "pid"
    launch_path.parent.mkdir(parents=True)
    launch_path.write_text(
        json.dumps({"status": "starting", "run_id": run_id}),
        encoding="utf-8",
    )
    run_state_path = report_dir / "runs" / run_id / "status.json"
    run_state_path.parent.mkdir(parents=True)
    run_state_path.write_text(
        json.dumps(
            {
                "status": "blocked_data",
                "lifecycle_status": "stopped",
                "reason_code": "membership_validation_failed",
                "run_id": run_id,
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeProcess:
        pid = 9876

        def wait(self):
            return 2

    def popen_factory(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    return_code = module.run_launchd_worker(
        launch_path=launch_path,
        pid_path=pid_path,
        report_dir=report_dir,
        run_id=run_id,
        command=["python", "run.py"],
        popen_factory=popen_factory,
    )

    record = json.loads(launch_path.read_text(encoding="utf-8"))
    assert return_code == 2
    assert pid_path.read_text(encoding="utf-8").strip() == "9876"
    assert record["status"] == "stopped"
    assert record["return_code"] == 2
    assert captured["command"] == ["python", "run.py"]
    assert captured["kwargs"]["start_new_session"] is False
    assert captured["kwargs"]["shell"] is False
    assert "GITHUB_PAT_TOKEN" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["env"]["QUANT_WORKFLOW_MLRUNS_DIR"] == str(
        tmp_path / "mlruns"
    )


def test_launchd_worker_records_required_mps_preflight_before_starting_child(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"
    run_id = "mps-worker-run"
    launch_path = report_dir / "runner" / run_id / "launch.json"
    pid_path = launch_path.parent / "pid"
    launch_path.parent.mkdir(parents=True)
    launch_path.write_text(
        json.dumps(
            {"status": "starting", "run_id": run_id, "mps_required": True}
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeProcess:
        pid = 2468

        def wait(self):
            return 0

    def popen_factory(command, **kwargs):
        captured["record_before_child"] = json.loads(
            launch_path.read_text(encoding="utf-8")
        )
        return FakeProcess()

    evidence = {
        "python_executable": "/venv/bin/python",
        "python_resolved": "/runtime/bin/python3.11",
        "torch_version": "2.5.1",
        "mps_built": True,
        "mps_available": True,
        "device": "mps:0",
        "tensor_sum": 4.0,
        "checked_at": "2026-07-20T06:00:00+08:00",
    }

    return_code = module.run_launchd_worker(
        launch_path=launch_path,
        pid_path=pid_path,
        report_dir=report_dir,
        run_id=run_id,
        command=["python", "run.py"],
        popen_factory=popen_factory,
        mps_probe=lambda: evidence,
    )

    record = json.loads(launch_path.read_text(encoding="utf-8"))
    assert return_code == 0
    assert captured["record_before_child"]["mps_preflight"] == evidence
    assert record["mps_preflight"] == evidence


def test_launchd_worker_refuses_child_when_required_mps_preflight_fails(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"
    run_id = "mps-worker-failed"
    launch_path = report_dir / "runner" / run_id / "launch.json"
    pid_path = launch_path.parent / "pid"
    launch_path.parent.mkdir(parents=True)
    launch_path.write_text(
        json.dumps(
            {"status": "starting", "run_id": run_id, "mps_required": True}
        ),
        encoding="utf-8",
    )

    return_code = module.run_launchd_worker(
        launch_path=launch_path,
        pid_path=pid_path,
        report_dir=report_dir,
        run_id=run_id,
        command=["python", "run.py"],
        popen_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training child must not start without MPS")
        ),
        mps_probe=lambda: (_ for _ in ()).throw(RuntimeError("MPS unavailable")),
    )

    record = json.loads(launch_path.read_text(encoding="utf-8"))
    assert return_code == 78
    assert not pid_path.exists()
    assert record["status"] == "failed_to_start"
    assert record["reason_code"] == "mps_preflight_failed"
    assert record["error_type"] == "RuntimeError"
