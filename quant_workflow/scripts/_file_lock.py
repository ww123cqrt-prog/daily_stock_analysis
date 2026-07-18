"""Small cross-platform advisory file-lock adapter.

Windows' standard library only exposes exclusive byte-range locks.  Readers are
therefore serialized on Windows, while POSIX keeps the existing shared-reader
semantics.  Both implementations preserve the same safety guarantee: a reader
never overlaps a writer.
"""

from __future__ import annotations

import os
from typing import BinaryIO


IS_WINDOWS = os.name == "nt"


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
    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
    try:
        msvcrt.locking(handle.fileno(), mode, 1)
    except OSError as exc:
        if not blocking:
            raise BlockingIOError("file lock is already held") from exc
        raise


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
