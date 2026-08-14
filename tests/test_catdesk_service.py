from __future__ import annotations

import json
from pathlib import Path

from src.services import catdesk_service


class _FakeProcess:
    pid = 87650

    @staticmethod
    def poll() -> None:
        return None


def test_start_local_catdesk_service_uses_fixed_detached_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    dsa_root = tmp_path / "daily_stock_analysis"
    catdesk_root = tmp_path / "a-share-us-catalyst"
    python = catdesk_root / ".venv" / "bin" / "python"
    (catdesk_root / "src" / "ashare_us_catalyst").mkdir(parents=True)
    (catdesk_root / "src" / "ashare_us_catalyst" / "web.py").touch()
    python.parent.mkdir(parents=True)
    python.touch()

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-catdesk")
    monkeypatch.setenv("JQDATA_PASSWORD", "must-not-reach-catdesk")
    monkeypatch.setattr(catdesk_service, "_service_is_ready", lambda: False)

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(catdesk_service.subprocess, "Popen", fake_popen)

    result = catdesk_service.start_local_catdesk_service(
        dsa_root=dsa_root,
        catdesk_root=catdesk_root,
        python_executable=python,
        wait_seconds=0,
    )

    assert observed["command"] == [
        str(python),
        "-m",
        "ashare_us_catalyst.web",
        "8765",
    ]
    kwargs = observed["kwargs"]
    assert kwargs["cwd"] == catdesk_root
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["PYTHONPATH"] == str(catdesk_root / "src")
    assert kwargs["env"]["CATDESK_ALLOW_A_HISTORY_NETWORK_FALLBACK"] == "0"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "JQDATA_PASSWORD" not in kwargs["env"]
    assert result["status"] == "starting"
    assert result["pid"] == 87650

    launch_record = json.loads(
        (dsa_root / ".local" / "services" / "catdesk-8765.json").read_text(encoding="utf-8")
    )
    assert launch_record["pid"] == 87650
    assert launch_record["port"] == 8765
    assert launch_record["status"] == "starting"


def test_start_local_catdesk_service_does_not_launch_duplicate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(catdesk_service, "_service_is_ready", lambda: True)

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("Popen must not run when CatDesk is already available")

    monkeypatch.setattr(catdesk_service.subprocess, "Popen", unexpected_popen)

    result = catdesk_service.start_local_catdesk_service(dsa_root=tmp_path, wait_seconds=0)

    assert result == {"status": "available", "port": 8765}


def test_start_local_catdesk_service_rejects_missing_fixed_runtime(tmp_path: Path) -> None:
    result = catdesk_service.start_local_catdesk_service(
        dsa_root=tmp_path,
        catdesk_root=tmp_path / "missing-catdesk",
        wait_seconds=0,
    )

    assert result == {
        "status": "failed",
        "message": "九点猫研运行环境不可用，请检查本机安装。",
        "port": 8765,
    }
