from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "launch_robust_strategy.py"
)


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


def test_launch_writes_pid_log_and_cpu_verified_record(tmp_path: Path) -> None:
    module = load_module()

    class FakeProcess:
        pid = 43210

        def poll(self):
            return None

    captured = {}

    def popen_factory(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    result = module.launch(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        as_of_date="2026-07-15",
        run_id="robust-run",
        resume=False,
        skip_membership_build=False,
        warmup_seconds=0,
        popen_factory=popen_factory,
        sleep_fn=lambda _: None,
        cpu_probe=lambda pid: 612.5,
    )

    runner_dir = tmp_path / "reports" / "runner" / "robust-run"
    record = json.loads((runner_dir / "launch.json").read_text(encoding="utf-8"))
    assert result["status"] == "running"
    assert record["pid"] == 43210
    assert record["cpu_percent"] == 612.5
    assert (runner_dir / "pid").read_text(encoding="utf-8").strip() == "43210"
    assert Path(record["log_path"]).is_file()
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False


def test_launch_fails_when_process_exits_during_warmup(tmp_path: Path) -> None:
    module = load_module()

    class FakeProcess:
        pid = 54321

        def poll(self):
            return 2

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
            popen_factory=lambda *args, **kwargs: FakeProcess(),
            sleep_fn=lambda _: None,
            cpu_probe=lambda pid: 0.0,
        )
    except RuntimeError as exc:
        assert "warmup" in str(exc)
    else:
        raise AssertionError("early process exit was reported as running")
    record = json.loads(
        (tmp_path / "reports" / "runner" / "failed-run" / "launch.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "failed_to_start"
