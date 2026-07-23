from pathlib import Path
from unittest.mock import patch

from launch_robust_signal_research import build_command, launch_signal_research


def test_build_command_uses_isolated_signal_entrypoint_and_resume(tmp_path: Path) -> None:
    command = build_command(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "data",
        report_dir=tmp_path / "reports",
        run_id="signal-run",
        resume=True,
        smoke=False,
        no_mlflow=False,
    )

    assert command[1].endswith("run_robust_signal_research.py")
    assert "--resume" in command
    assert "run_robust_strategy.py" not in " ".join(command)


def test_launcher_reuses_sanitized_launchd_runner_without_mps(tmp_path: Path) -> None:
    with patch("launch_robust_signal_research.launch", return_value={"status": "running"}) as mocked:
        result = launch_signal_research(
            config=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "data",
            report_dir=tmp_path / "reports",
            mlruns_dir=tmp_path / "mlruns",
            run_id="signal-run",
            resume=False,
            smoke=False,
            no_mlflow=False,
            warmup_seconds=0,
        )

    assert result["status"] == "running"
    assert mocked.call_args.kwargs["mps_required_override"] is False
    assert mocked.call_args.kwargs["runner_namespace"] == "runner"
