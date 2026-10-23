from pathlib import Path
from unittest.mock import patch

from launch_robust_fundamentals import build_command, launch_fundamental_download


def test_build_command_uses_goal1_fundamental_entrypoint(tmp_path: Path) -> None:
    command = build_command(
        quant_data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        report_dir=tmp_path / "reports",
        run_id="fundamental-run",
    )

    joined = " ".join(command)
    assert command[1].endswith("download_robust_fundamentals.py")
    assert "--data-end 2025-12-31" in joined
    assert "--report-end 2025-12-31" in joined
    for flag in ("--data-start", "--data-end", "--report-start", "--report-end"):
        assert command[command.index(flag) + 1] < "2026-01-01"
    assert str(tmp_path / "reports" / "runs" / "fundamental-run" / "status.json") in command


def test_launcher_uses_launchd_without_mps(tmp_path: Path) -> None:
    with patch(
        "launch_robust_fundamentals.launch", return_value={"status": "running"}
    ) as mocked:
        result = launch_fundamental_download(
            config=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "data",
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            report_dir=tmp_path / "reports",
            mlruns_dir=tmp_path / "mlruns",
            run_id="fundamental-run",
            warmup_seconds=0,
        )

    assert result["status"] == "running"
    assert mocked.call_args.kwargs["runner_namespace"] == "fundamental_runner"
    assert mocked.call_args.kwargs["mps_required_override"] is False
