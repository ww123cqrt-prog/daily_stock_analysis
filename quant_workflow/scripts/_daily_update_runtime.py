"""Runtime locking and atomic status writes for the daily data update."""

from __future__ import annotations

import errno
import json
import os
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


class LockHeldError(RuntimeError):
    def __init__(self, owner: dict[str, Any]):
        self.owner = owner
        run_id = owner.get("run_id", "unknown")
        pid = owner.get("pid", "unknown")
        super().__init__(f"daily update already active: run_id={run_id} pid={pid}")


class DailyUpdateLock:
    def __init__(self, path: Path, *, malformed_stale_after_seconds: int = 24 * 60 * 60):
        self.path = Path(path)
        self.malformed_stale_after_seconds = int(malformed_stale_after_seconds)
        self.owner: dict[str, Any] | None = None

    def _read_existing(self) -> tuple[dict[str, Any], bool]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "malformed_lock", "path": str(self.path)}, False
        if not isinstance(payload, dict):
            return {"status": "malformed_lock", "path": str(self.path)}, False
        return payload, isinstance(payload.get("pid"), int)

    def _can_reclaim(self, owner: dict[str, Any], has_pid: bool) -> bool:
        if has_pid:
            return not pid_is_alive(int(owner["pid"]))
        try:
            age = max(0.0, time.time() - self.path.stat().st_mtime)
        except OSError:
            return True
        owner["age_seconds"] = round(age, 1)
        return age >= self.malformed_stale_after_seconds

    def acquire(self, run_id: str) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            lock_id = uuid.uuid4().hex
            owner = {
                "pid": os.getpid(),
                "run_id": str(run_id),
                "lock_id": lock_id,
                "hostname": socket.gethostname(),
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                existing, has_pid = self._read_existing()
                if not self._can_reclaim(existing, has_pid):
                    raise LockHeldError(existing)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue

            try:
                os.write(fd, json.dumps(owner, ensure_ascii=False, indent=2).encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            self.owner = owner
            return owner
        raise LockHeldError({"status": "lock_race", "path": str(self.path)})

    def release(self) -> None:
        if self.owner is None:
            return
        existing, _has_pid = self._read_existing()
        if existing.get("lock_id") == self.owner.get("lock_id"):
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.owner = None

    def __enter__(self) -> "DailyUpdateLock":
        if self.owner is None:
            raise RuntimeError("call acquire() before entering DailyUpdateLock context")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
