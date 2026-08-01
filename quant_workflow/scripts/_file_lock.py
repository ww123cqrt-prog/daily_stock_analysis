"""Small cross-platform advisory file-lock adapter.

Windows' standard library only exposes exclusive byte-range locks.  Readers are
therefore serialized on Windows, while POSIX keeps the existing shared-reader
semantics.  Both implementations preserve the same safety guarantee: a reader
never overlaps a writer.
"""

from __future__ import annotations

import errno
import os
import time
from typing import BinaryIO


IS_WINDOWS = os.name == "nt"
WINDOWS_LOCK_RETRY_INTERVAL_SECONDS = 0.1
WINDOWS_LOCK_MAX_WAIT_SECONDS = 3600.0


def _prepare_windows_lock_file(handle: BinaryIO) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    handle.seek(0)


def _lock_windows(handle: BinaryIO, *, blocking: bool) -> None:
    import msvcrt

    _prepare_windows_lock_file(handle)
    started_at = time.monotonic()
    while True:
        handle.seek(0)
        try:
            # LK_LOCK has a short, implementation-defined retry limit.  Use
            # non-blocking attempts so a long training run can hold the lock.
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if not blocking:
                raise BlockingIOError("file lock is already held") from exc
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            if time.monotonic() - started_at >= WINDOWS_LOCK_MAX_WAIT_SECONDS:
                raise TimeoutError("timed out waiting for the Windows file lock") from exc
            time.sleep(WINDOWS_LOCK_RETRY_INTERVAL_SECONDS)


def _unlock_windows(handle: BinaryIO) -> None:
    import msvcrt

    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _lock_posix(handle: BinaryIO, *, exclusive: bool, blocking: bool) -> None:
    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(handle.fileno(), operation)


def _unlock_posix(handle: BinaryIO) -> None:
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def lock_file(
    handle: BinaryIO, *, exclusive: bool, blocking: bool = True
) -> None:
    if IS_WINDOWS:
        _lock_windows(handle, blocking=blocking)
        return
    _lock_posix(handle, exclusive=exclusive, blocking=blocking)


def unlock_file(handle: BinaryIO) -> None:
    if IS_WINDOWS:
        _unlock_windows(handle)
        return
    _unlock_posix(handle)
