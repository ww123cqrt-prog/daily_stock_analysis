import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_file_lock.py"
    spec = importlib.util.spec_from_file_location("robust_file_lock", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_windows_lock_creates_lock_byte_and_uses_nonblocking_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    calls: list[tuple[int, int]] = []
    fake = SimpleNamespace(
        LK_LOCK=1,
        LK_NBLCK=2,
        LK_UNLCK=3,
        locking=lambda _fd, mode, size: calls.append((mode, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    lock_path = tmp_path / "snapshot.lock"
    with lock_path.open("a+b") as handle:
        module._lock_windows(handle, blocking=False)
        module._unlock_windows(handle)

    assert lock_path.stat().st_size == 1
    assert calls == [(fake.LK_NBLCK, 1), (fake.LK_UNLCK, 1)]


def test_windows_nonblocking_contention_becomes_blocking_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()

    def fail_lock(_fd, _mode, _size):
        raise OSError("already locked")

    fake = SimpleNamespace(
        LK_LOCK=1,
        LK_NBLCK=2,
        LK_UNLCK=3,
        locking=fail_lock,
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    with (tmp_path / "snapshot.lock").open("a+b") as handle:
        with pytest.raises(BlockingIOError, match="already held"):
            module._lock_windows(handle, blocking=False)
