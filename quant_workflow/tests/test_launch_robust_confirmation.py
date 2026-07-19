from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "launch_robust_confirmation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("launch_robust_confirmation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_confirmation_command_targets_single_use_confirmation(tmp_path: Path) -> None:
    module = load_module()

    command = module.build_confirmation_command(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen=tmp_path / "frozen.json",
        provider_state=tmp_path / "provider.json",
        run_id="confirmation-one",
        resume=True,
    )

    assert command[0] == module.sys.executable
    assert command[1].endswith("run_robust_confirmation.py")
    assert "--frozen" in command
    assert "--provider-state" in command
    assert "--resume" in command


def test_launch_confirmation_uses_isolated_runner_and_frozen_model_device(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "research_candidate_ready",
                "as_of_date": "2026-07-15",
                "selected_models": ["linear_ridge", "mps_dnn"],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return {"status": "running", "run_id": kwargs["run_id"]}

    monkeypatch.setattr(module, "launch", fake_launch)

    result = module.launch_confirmation(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        mlruns_dir=tmp_path / "mlruns",
        frozen=frozen,
        provider_state=tmp_path / "provider.json",
        run_id="confirmation-one",
        resume=False,
        warmup_seconds=0,
    )

    assert result["status"] == "running"
    assert captured["runner_namespace"] == "confirmation_runner"
    assert captured["mps_required_override"] is True
    assert captured["update_pipeline_state_on_reconcile"] is False
    assert captured["command_override"][1].endswith("run_robust_confirmation.py")


def test_launch_confirmation_does_not_require_mps_for_classical_selection(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "research_candidate_ready",
                "selected_models": ["linear_ridge", "lightgbm"],
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        module,
        "launch",
        lambda **kwargs: captured.update(kwargs) or {"status": "running"},
    )

    module.launch_confirmation(
        config=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        mlruns_dir=tmp_path / "mlruns",
        frozen=frozen,
        provider_state=tmp_path / "provider.json",
        run_id="confirmation-classical",
        resume=False,
        warmup_seconds=0,
    )

    assert captured["mps_required_override"] is False
