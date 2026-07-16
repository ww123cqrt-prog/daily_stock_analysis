from __future__ import annotations

import json
from pathlib import Path

from src.services import mlflow_service


class _FakeProcess:
    pid = 4321


def test_start_local_mlflow_service_uses_fixed_detached_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-mlflow")
    monkeypatch.setenv("JQDATA_PASSWORD", "must-not-reach-mlflow")
    monkeypatch.setattr(mlflow_service, "_port_is_open", lambda: False)

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(mlflow_service.subprocess, "Popen", fake_popen)

    result = mlflow_service.start_local_mlflow_service(
        repo_root=tmp_path,
        python_executable=Path("/opt/dsa/.venv/bin/python"),
        wait_seconds=0,
    )

    store_uri = (tmp_path / "quant_workflow" / "mlruns").resolve().as_uri()
    assert observed["command"] == [
        "/opt/dsa/.venv/bin/python",
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        store_uri,
        "--host",
        "127.0.0.1",
        "--port",
        "6006",
    ]
    kwargs = observed["kwargs"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "JQDATA_PASSWORD" not in kwargs["env"]
    assert result["status"] == "starting"
    assert result["pid"] == 4321

    launch_record = json.loads(
        (tmp_path / ".local" / "services" / "mlflow-6006.json").read_text(encoding="utf-8")
    )
    assert launch_record["pid"] == 4321
    assert launch_record["port"] == 6006
    assert launch_record["status"] == "starting"


def test_start_local_mlflow_service_does_not_launch_duplicate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mlflow_service, "_port_is_open", lambda: True)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not run when MLflow is already available")

    monkeypatch.setattr(mlflow_service.subprocess, "Popen", unexpected_popen)

    result = mlflow_service.start_local_mlflow_service(repo_root=tmp_path, wait_seconds=0)

    assert result == {"status": "available", "port": 6006}
