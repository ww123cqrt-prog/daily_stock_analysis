import json
from pathlib import Path
from unittest.mock import patch

from launch_robust_signal_research import (
    build_command,
    build_plan_payload,
    cancel_signal_research,
    launch_signal_research,
    main,
    preview_signal_research_plan,
)
from robust_signal_data_contract import RobustSignalDataBlocked


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


def test_main_returns_nonzero_when_signal_child_fails_during_warmup(
    tmp_path: Path,
) -> None:
    with patch(
        "launch_robust_signal_research.launch_signal_research",
        return_value={"status": "failed_to_start", "return_code": 78},
    ):
        return_code = main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--quant-data-dir",
                str(tmp_path / "quant_data"),
                "--report-dir",
                str(tmp_path / "reports"),
                "--mlruns-dir",
                str(tmp_path / "mlruns"),
                "--run-id",
                "failed-signal-warmup",
                "--warmup-seconds",
                "0",
            ]
        )

    assert return_code == 78


def test_cancel_signal_research_stops_runner_and_marks_run_non_resumable(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    run_id = "signal-run"
    run_dir = report_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    running = {
        "run_id": run_id,
        "lifecycle_status": "running",
        "pause_state": "running",
        "current_stage": "screening",
        "completed_count": 3,
        "expected_count": 96,
    }
    (run_dir / "status.json").write_text(json.dumps(running), encoding="utf-8")
    latest = report_dir / "robust_signal_status_latest.json"
    latest.write_text(json.dumps(running), encoding="utf-8")

    with patch(
        "launch_robust_signal_research.cancel_launchd_runner",
        return_value={"status": "cancelled", "run_id": run_id},
    ) as cancel_runner:
        result = cancel_signal_research(report_dir=report_dir, run_id=run_id)

    cancel_runner.assert_called_once_with(
        report_dir=report_dir,
        run_id=run_id,
        runner_namespace="runner",
        terminal_state_path=run_dir / "status.json",
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert result["cancel_outcome"] == "cancelled"
    assert status["lifecycle_status"] == "cancelled"
    assert status["pause_state"] == "cancelled"
    assert status["reason_code"] == "user_cancelled"
    assert status["next_gate"] == "start_new_run"
    assert json.loads(latest.read_text(encoding="utf-8")) == status


def test_cancel_signal_research_does_not_overwrite_natural_completion(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    run_id = "signal-run-completed"
    run_dir = report_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    completed = {
        "run_id": run_id,
        "status": "signal_candidate_ready",
        "lifecycle_status": "completed",
        "pause_state": "completed",
        "current_stage": "complete",
        "completed_count": 96,
        "expected_count": 96,
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(completed), encoding="utf-8")
    latest = report_dir / "robust_signal_status_latest.json"
    latest.write_text(json.dumps(completed), encoding="utf-8")

    with patch(
        "launch_robust_signal_research.cancel_launchd_runner",
        return_value={
            "status": "completed",
            "run_id": run_id,
            "cancel_outcome": "already_terminal",
        },
    ) as cancel_runner:
        result = cancel_signal_research(report_dir=report_dir, run_id=run_id)

    cancel_runner.assert_called_once_with(
        report_dir=report_dir,
        run_id=run_id,
        runner_namespace="runner",
        terminal_state_path=status_path,
    )
    assert result == {**completed, "cancel_outcome": "already_terminal"}
    assert json.loads(status_path.read_text(encoding="utf-8")) == completed
    assert json.loads(latest.read_text(encoding="utf-8")) == completed


def test_plan_payload_counts_registered_jobs_without_2026() -> None:
    config = {
        "periods": {
            "screening_quarters": ["2022Q4"],
            "validation_quarters": ["2025Q4"],
            "screening_window_days": 756,
            "validation_windows_days": [504, 756],
        },
        "signals": [
            {"id": "signal_a", "horizon_days": 5, "feature_family": "full"},
            {"id": "signal_b", "horizon_days": 10, "feature_family": "full"},
        ],
        "feature_families": {"full": {"prefixes": []}},
        "models": {
            "screening": [{"id": "ridge"}],
            "validation": [{"id": "ols"}, {"id": "ridge"}],
        },
        "screening": {"maximum_candidates": 1},
        "safety": {"locked_confirmation_start": "2026-01-01"},
    }

    payload = build_plan_payload(
        config=config,
        context={
            "config_hash": "config-hash",
            "git_commit": "commit",
            "git_clean": True,
            "data_manifest_hash": "data-hash",
            "context_hash": "context-hash",
            "provider_tree_sha256": "provider-hash",
        },
    )

    assert payload["status"] == "plan_ready"
    assert payload["screening"]["job_count"] == 2
    assert payload["validation"]["maximum_job_count"] == 4
    assert payload["locked_confirmation_start"] == "2026-01-01"
    assert all("2026" not in quarter for quarter in payload["screening"]["quarters"])
    assert all("2026" not in quarter for quarter in payload["validation"]["quarters"])


def test_plan_preview_publishes_structured_blocked_data_instead_of_traceback(
    tmp_path: Path,
) -> None:
    blocked = RobustSignalDataBlocked(
        "robust_readiness_missing",
        "Robust CSI800 data has not been prepared.",
        "鲁棒 CSI800 数据尚未验收。",
    )
    report_dir = tmp_path / "reports"

    with (
        patch("launch_robust_signal_research.load_config", return_value={}),
        patch("launch_robust_signal_research.build_context", side_effect=blocked),
    ):
        payload = preview_signal_research_plan(
            config_path=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
        )

    assert payload == {
        "status": "blocked_data",
        "reason_code": "robust_readiness_missing",
        "summary": "Robust CSI800 data has not been prepared.",
        "summary_zh": "鲁棒 CSI800 数据尚未验收。",
        "training_started": False,
    }
    assert json.loads(
        (report_dir / "robust_signal_plan_latest.json").read_text(encoding="utf-8")
    ) == payload
