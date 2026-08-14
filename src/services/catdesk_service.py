"""Start the fixed local CatDesk service used by the DSA landing page."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


CATDESK_HOST = "127.0.0.1"
CATDESK_PORT = 8765
CATDESK_HEALTH_TIMEOUT_SECONDS = 0.35
CATDESK_START_WAIT_SECONDS = 2.0
CATDESK_START_FAILURE_MESSAGE = "九点猫研运行环境不可用，请检查本机安装。"
_START_LOCK = threading.Lock()
_CHILD_ENV_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "NO_COLOR",
        "PATH",
        "PYTHONIOENCODING",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
        "USER",
    }
)


def _default_dsa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_catdesk_root() -> Path:
    return _default_dsa_root().parent / "a-share-us-catalyst"


def _service_is_ready() -> bool:
    request = Request(
        f"http://{CATDESK_HOST}:{CATDESK_PORT}/api/health",
        headers={"User-Agent": "DSA-CatDesk-health-check/1.0"},
    )
    try:
        with urlopen(request, timeout=CATDESK_HEALTH_TIMEOUT_SECONDS) as response:
            return 200 <= int(response.getcode()) < 400
    except (OSError, TimeoutError, ValueError):
        return False


def _write_launch_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def start_local_catdesk_service(
    *,
    dsa_root: Path | None = None,
    catdesk_root: Path | None = None,
    python_executable: Path | None = None,
    wait_seconds: float = CATDESK_START_WAIT_SECONDS,
) -> dict[str, Any]:
    """Launch CatDesk using the one approved local command, if needed."""

    with _START_LOCK:
        if _service_is_ready():
            return {"status": "available", "port": CATDESK_PORT}

        dsa = (dsa_root or _default_dsa_root()).resolve()
        root = (catdesk_root or _default_catdesk_root()).resolve()
        python = Path(
            python_executable or root / ".venv" / "bin" / "python"
        ).expanduser().absolute()
        web_module = root / "src" / "ashare_us_catalyst" / "web.py"
        if not root.is_dir() or not python.is_file() or not web_module.is_file():
            return {
                "status": "failed",
                "message": CATDESK_START_FAILURE_MESSAGE,
                "port": CATDESK_PORT,
            }

        log_path = dsa / "logs" / "catdesk_8765.log"
        record_path = dsa / ".local" / "services" / "catdesk-8765.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            "-m",
            "ashare_us_catalyst.web",
            str(CATDESK_PORT),
        ]
        child_env = {
            name: value
            for name, value in os.environ.items()
            if name in _CHILD_ENV_NAMES
        }
        child_env["PYTHONPATH"] = str(root / "src")
        child_env["VIRTUAL_ENV"] = str(root / ".venv")
        child_env["CATDESK_ALLOW_A_HISTORY_NETWORK_FALLBACK"] = "0"

        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError:
            return {
                "status": "failed",
                "message": CATDESK_START_FAILURE_MESSAGE,
                "port": CATDESK_PORT,
            }

        deadline = time.monotonic() + max(0.0, wait_seconds)
        return_code: int | None = None
        while time.monotonic() < deadline:
            if _service_is_ready():
                status = "available"
                break
            return_code = process.poll()
            if return_code is not None:
                status = "failed"
                break
            time.sleep(0.1)
        else:
            return_code = process.poll()
            status = "failed" if return_code is not None else "starting"

        launch_record: dict[str, Any] = {
            "command": command,
            "log": str(log_path.relative_to(dsa)),
            "pid": process.pid,
            "port": CATDESK_PORT,
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        result: dict[str, Any] = {
            "status": status,
            "pid": process.pid,
            "port": CATDESK_PORT,
        }
        if status == "failed":
            launch_record["returnCode"] = return_code
            result["message"] = "九点猫研进程启动后立即退出，请检查本机日志。"
        _write_launch_record(record_path, launch_record)
        return result
