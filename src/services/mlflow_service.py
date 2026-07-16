"""Start the fixed local MLflow service used by DSA's experiment proxy."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = 6006
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
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "TMPDIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
    }
)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _port_is_open() -> bool:
    try:
        with socket.create_connection((MLFLOW_HOST, MLFLOW_PORT), timeout=0.25):
            return True
    except OSError:
        return False


def _write_launch_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def start_local_mlflow_service(
    *,
    repo_root: Path | None = None,
    python_executable: Path | None = None,
    wait_seconds: float = 3.0,
) -> dict[str, Any]:
    """Launch MLflow on the fixed loopback endpoint without accepting user commands."""

    with _START_LOCK:
        if _port_is_open():
            return {"status": "available", "port": MLFLOW_PORT}

        root = (repo_root or _default_repo_root()).resolve()
        python = (python_executable or Path(sys.executable)).resolve()
        configured_store = os.getenv("QUANT_WORKFLOW_MLRUNS_DIR", "").strip()
        store = Path(configured_store).expanduser().resolve() if configured_store else root / "quant_workflow" / "mlruns"
        log_path = root / "logs" / "mlflow_6006.log"
        record_path = root / ".local" / "services" / "mlflow-6006.json"
        store.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            str(python),
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            store.resolve().as_uri(),
            "--host",
            MLFLOW_HOST,
            "--port",
            str(MLFLOW_PORT),
        ]
        child_env = {
            name: value
            for name, value in os.environ.items()
            if name in _CHILD_ENV_NAMES
        }
        child_env["MLFLOW_ALLOW_FILE_STORE"] = "true"
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

        deadline = time.monotonic() + max(0.0, wait_seconds)
        while time.monotonic() < deadline:
            if _port_is_open():
                status = "available"
                break
            time.sleep(0.1)
        else:
            status = "starting"

        _write_launch_record(
            record_path,
            {
                "command": command,
                "log": str(log_path.relative_to(root)),
                "pid": process.pid,
                "port": MLFLOW_PORT,
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "status": status,
            },
        )
        return {"status": status, "pid": process.pid, "port": MLFLOW_PORT}
