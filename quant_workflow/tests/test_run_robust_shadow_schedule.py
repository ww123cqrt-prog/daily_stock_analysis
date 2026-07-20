from __future__ import annotations

import importlib.util
import json
import plistlib
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_robust_shadow_schedule.py"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
PLIST_TEMPLATE = (
    REPO_ROOT / "scripts" / "launchd" / "com.dsa.robust-shadow.plist.template"
)
DAILY_PLIST_TEMPLATE = (
    REPO_ROOT / "scripts" / "launchd" / "com.dsa.quant-daily-update.plist.template"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_robust_shadow_schedule", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def active_inputs(tmp_path: Path, *, selected_models: list[str]) -> dict[str, Path]:
    reports = tmp_path / "reports"
    robust = reports / "robust_strategy"
    frozen = robust / "frozen_strategy_latest.json"
    confirmation = robust / "confirmation_latest.json"
    provider = robust / "provider_latest.json"
    summary = reports / "daily_update_summary_latest.json"
    write_json(
        frozen,
        {
            "confirmation_status": "shadow_observation",
            "shadow_tracking_started": True,
            "selected_models": selected_models,
        },
    )
    write_json(confirmation, {"status": "shadow_observation"})
    write_json(provider, {"status": "completed"})
    write_json(
        summary,
        {
            "generated_at": "2026-07-20T18:55:00+08:00",
            "status": "ok_with_warnings",
        },
    )
    return {
        "reports": reports,
        "robust": robust,
        "frozen": frozen,
        "confirmation": confirmation,
        "provider": provider,
        "summary": summary,
    }


def test_inactive_shadow_schedule_is_a_clean_skip(tmp_path: Path) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge"])
    write_json(paths["confirmation"], {"status": "development_failed"})
    calls = []

    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=lambda **kwargs: calls.append(kwargs),
    )

    assert result["status"] == "inactive"
    assert result["outcome"] == "skipped"
    assert calls == []
    assert module.exit_code_for_schedule(result) == 0


def test_finalized_shadow_schedule_is_a_clean_skip(tmp_path: Path) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge"])
    write_json(
        paths["robust"] / "shadow_state_latest.json",
        {"status": "robust_strategy_qualified"},
    )
    calls = []

    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=lambda **kwargs: calls.append(kwargs),
    )

    assert result["status"] == "inactive"
    assert result["reason_code"] == "shadow_tracking_finalized"
    assert calls == []
    assert module.exit_code_for_schedule(result) == 0


def test_active_mps_shadow_waits_for_daily_update_and_records_probe(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge", "mps_dnn"])
    probe = {
        "python_executable": "/tmp/.venv-mps-dnn/bin/python",
        "mps_built": True,
        "mps_available": True,
        "device": "mps:0",
        "tensor_sum": 4.0,
    }
    monkeypatch.setattr(module, "_mps_probe", lambda: probe)
    calls = []

    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "status": "shadow_observation",
                "outcome": "completed",
                "reason_code": None,
            }
        ),
    )

    assert result["status"] == "shadow_observation"
    assert result["daily_update_status"] == "ok_with_warnings"
    assert result["mps_preflight"] == probe
    assert len(calls) == 1
    saved = json.loads(
        (paths["robust"] / "shadow_runner" / "schedule_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["mps_preflight"]["device"] == "mps:0"
    assert module.exit_code_for_schedule(result) == 0


def test_failed_daily_update_or_shadow_run_returns_nonzero(tmp_path: Path) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge"])
    write_json(
        paths["summary"],
        {
            "generated_at": "2026-07-20T18:55:00+08:00",
            "status": "failed",
        },
    )
    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
    )
    assert result["status"] == "failed"
    assert result["reason_code"] == "daily_update_failed"
    assert module.exit_code_for_schedule(result) != 0


def test_morning_daily_update_summary_does_not_satisfy_evening_dependency(
    tmp_path: Path,
) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge"])
    write_json(
        paths["summary"],
        {
            "generated_at": "2026-07-20T10:00:00+08:00",
            "status": "ok",
        },
    )
    calls = []

    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=lambda **kwargs: calls.append(kwargs),
    )

    assert result["status"] == "failed"
    assert result["reason_code"] == "daily_update_timeout"
    assert calls == []

    write_json(
        paths["summary"],
        {
            "generated_at": "2026-07-20T18:55:00+08:00",
            "status": "ok",
        },
    )
    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=lambda **kwargs: {
            "status": "development_failed",
            "reason_code": "inference_failed",
            "outcome": "completed",
        },
    )
    assert result["status"] == "development_failed"
    assert module.exit_code_for_schedule(result) != 0


def test_dry_run_update_and_unhandled_shadow_error_are_persisted_as_failures(
    tmp_path: Path,
) -> None:
    module = load_module()
    paths = active_inputs(tmp_path, selected_models=["linear_ridge"])
    write_json(
        paths["summary"],
        {
            "generated_at": "2026-07-20T18:55:00+08:00",
            "status": "ok",
            "dry_run": True,
        },
    )
    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
    )
    assert result["reason_code"] == "daily_update_not_executed"
    assert module.exit_code_for_schedule(result) != 0

    write_json(
        paths["summary"],
        {
            "generated_at": "2026-07-20T18:55:00+08:00",
            "status": "ok",
            "dry_run": False,
        },
    )

    def crash(**kwargs):
        raise RuntimeError("unexpected runner failure")

    result = module.run_scheduled_shadow(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        reports_root=paths["reports"],
        as_of_date="2026-07-20",
        wait_timeout_seconds=0,
        shadow_runner=crash,
    )
    assert result["status"] == "failed"
    assert result["reason_code"] == "shadow_runner_failed"
    assert result["error_type"] == "RuntimeError"
    assert module.exit_code_for_schedule(result) != 0
    saved = json.loads(
        (paths["robust"] / "shadow_runner" / "schedule_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["reason_code"] == "shadow_runner_failed"


def render_plist(path: Path) -> dict:
    rendered = (
        path.read_text(encoding="utf-8")
        .replace("__PYTHON__", "/tmp/python")
        .replace("__SCRIPT__", "/tmp/run.py")
        .replace("__CONFIG__", "/tmp/config.yaml")
        .replace("__HOME__", "/tmp/home")
        .replace("__WORKDIR__", "/tmp")
        .replace("__STDOUT__", "/tmp/out.log")
        .replace("__STDERR__", "/tmp/err.log")
    )
    return plistlib.loads(rendered.encode("utf-8"))


def test_shadow_launchd_template_runs_monday_to_friday_at_1910() -> None:
    payload = render_plist(PLIST_TEMPLATE)

    assert payload["Label"] == "com.dsa.robust-shadow"
    assert payload["ProgramArguments"][:2] == ["/usr/bin/env", "-i"]
    assert "HOME=/tmp/home" in payload["ProgramArguments"]
    assert "/tmp/python" in payload["ProgramArguments"]
    assert payload["ProgramArguments"][-2:] == ["--config", "/tmp/config.yaml"]
    schedule = payload["StartCalendarInterval"]
    # launchd uses 0 and 7 for Sunday, so 1..5 is Monday through Friday.
    assert [item["Weekday"] for item in schedule] == [1, 2, 3, 4, 5]
    assert {(item["Hour"], item["Minute"]) for item in schedule} == {(19, 10)}
    assert load_module().DEFAULT_WAIT_TIMEOUT_SECONDS == 4 * 60 * 60


def test_daily_update_launchd_template_runs_monday_to_friday_at_1840() -> None:
    payload = render_plist(DAILY_PLIST_TEMPLATE)
    schedule = payload["StartCalendarInterval"]

    assert payload["ProgramArguments"][:2] == ["/usr/bin/env", "-i"]
    assert "HOME=/tmp/home" in payload["ProgramArguments"]
    assert "/tmp/python" in payload["ProgramArguments"]
    assert [item["Weekday"] for item in schedule] == [1, 2, 3, 4, 5]
    assert {(item["Hour"], item["Minute"]) for item in schedule} == {(18, 40)}
