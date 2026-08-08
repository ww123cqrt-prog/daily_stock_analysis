#!/usr/bin/env python3
"""Versioned, auditable contracts for datasets stored under ``quant_data``."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, NamedTuple, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file  # noqa: E402


SCHEMA_VERSION = 2
DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,95}$")
PRICE_BASES = {"raw", "qfq", "hfq", "total_return", "derived", "not_applicable"}
ROLES = {"canonical", "audit", "legacy"}
STATUSES = {"ready", "stale", "blocked", "legacy_unknown"}
ADJUSTMENT_METHODS = {
    "none",
    "local_factor",
    "provider_native_paired",
    "unknown",
}
REQUIRED_BAR_COLUMNS = ("date", "open", "high", "low", "close", "volume")
BAR_OHLC_RELATIVE_TOLERANCE = 2e-6
BAR_OHLC_ABSOLUTE_TOLERANCE = 1e-4
ADJUSTED_PRICE_BASES = {"qfq", "hfq"}
QLIB_LEGACY_BENCHMARK_SYMBOLS = ("SH000300", "SH000905", "SH000906")
QLIB_BENCHMARK_SYMBOLS = (
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
)
QLIB_BENCHMARK_FIELDS = ("open", "high", "low", "close", "volume", "factor")
AUTHORITATIVE_QUALITY_MIN_CALENDAR_DAYS = 252
AUTHORITATIVE_QUALITY_MIN_INSTRUMENTS = 100
QLIB_PROVIDER_PRODUCER_LEGACY_OMISSION_VERSIONS = {
    ("qlib_ashare_v2", "build_qlib_v2.py"): frozenset({"2", "3", "4", "5"}),
    (
        "qlib_ashare_cleaned_v2",
        "build_cleaned_qlib_v2.py",
    ): frozenset({"1", "2"}),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataContractError(RuntimeError):
    """Raised when a dataset cannot prove its storage or price contract."""


class ResolvedDataset(NamedTuple):
    dataset_id: str
    snapshot_path: Path
    manifest: dict[str, Any]
    catalog_entry: dict[str, Any]


ValidatedResolutionCache = MutableMapping[str, ResolvedDataset]


def _legacy_provider_manifest_may_omit_producer(
    dataset_id: Any,
    producer: Mapping[str, Any],
) -> bool:
    script = str(producer.get("script") or "")
    version = str(producer.get("version") or "")
    return version in QLIB_PROVIDER_PRODUCER_LEGACY_OMISSION_VERSIONS.get(
        (str(dataset_id or ""), script), frozenset()
    )


def _legacy_cleaned_provider_may_omit_dependency(
    producer: Mapping[str, Any],
) -> bool:
    return (
        producer.get("script") == "build_cleaned_qlib_v2.py"
        and str(producer.get("version") or "") == "1"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filesystem_state_fingerprint(
    root: Path,
    *,
    max_entries: int | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a cheap mutation fingerprint after full snapshot validation.

    This intentionally hashes filesystem metadata rather than file contents. A
    producer or audit scan must still perform the authoritative content-hash
    validation first; consumers can then use this proof to detect deletion,
    replacement, or modification without rereading the full dataset. ``ctime``
    and inode/device identity prevent an equal-size rewrite from being hidden by
    restoring the original ``mtime``.
    """

    root = Path(root).expanduser().absolute()
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise DataContractError("fingerprint root is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise DataContractError("fingerprint root is unavailable")
    try:
        resolved_root = root.resolve(strict=True)
        resolved_metadata = root.lstat()
    except OSError as exc:
        raise DataContractError("fingerprint root changed during scan") from exc
    if (
        stat.S_ISLNK(resolved_metadata.st_mode)
        or not stat.S_ISDIR(resolved_metadata.st_mode)
        or (resolved_metadata.st_dev, resolved_metadata.st_ino)
        != (root_metadata.st_dev, root_metadata.st_ino)
    ):
        raise DataContractError("fingerprint root changed during scan")

    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.scandir not in os.supports_fd
    ):
        return _filesystem_state_fingerprint_fallback(
            root,
            max_entries=max_entries,
            max_seconds=max_seconds,
        )

    directory_flags = os.O_RDONLY
    directory_flags |= os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        file_flags |= os.O_NONBLOCK
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    scanned_entries = 0
    started_at = time.monotonic()
    directory_states: list[tuple[str, tuple[int, int, int, int]]] = []
    file_states: list[tuple[str, tuple[int, int, int, int, int]]] = []

    def check_limits() -> None:
        if max_entries is not None and scanned_entries > max_entries:
            raise DataContractError("filesystem state scan exceeded entry limit")
        if max_seconds is not None and time.monotonic() - started_at > max_seconds:
            raise DataContractError("filesystem state scan exceeded time limit")

    def visit(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        nonlocal file_count, scanned_entries, total_bytes
        directory_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise DataContractError("fingerprint directory changed during scan")
        directory_states.append(
            (
                "/".join(relative_parts),
                (
                    directory_before.st_dev,
                    directory_before.st_ino,
                    directory_before.st_mtime_ns,
                    directory_before.st_ctime_ns,
                ),
            )
        )
        names: list[str] = []
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                scanned_entries += 1
                check_limits()
                names.append(entry.name)
        names.sort()
        check_limits()

        # A depth-first walk over sorted path components preserves the v2
        # digest order; compact metadata records support the final stability check.
        for name in names:
            check_limits()
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DataContractError(
                    "fingerprint entry changed during scan"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise DataContractError("fingerprint root cannot contain symlinks")
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise DataContractError(
                        "fingerprint directory changed or became a symlink"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISDIR(opened.st_mode)
                        or (opened.st_dev, opened.st_ino)
                        != (metadata.st_dev, metadata.st_ino)
                    ):
                        raise DataContractError(
                            "fingerprint directory changed during scan"
                        )
                    visit(child_fd, (*relative_parts, name))
                finally:
                    os.close(child_fd)
                continue
            if name.endswith(".lock"):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            file_fd: int | None = None
            try:
                file_fd = os.open(name, file_flags, dir_fd=directory_fd)
                opened = os.fstat(file_fd)
                after_open = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                opened_identity = (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or not stat.S_ISREG(after_open.st_mode)
                    or opened_identity
                    != (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_size,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    )
                    or opened_identity
                    != (
                        after_open.st_dev,
                        after_open.st_ino,
                        after_open.st_size,
                        after_open.st_mtime_ns,
                        after_open.st_ctime_ns,
                    )
                ):
                    raise DataContractError(
                        "fingerprint file changed during scan"
                    )
                metadata = opened
            except OSError as exc:
                raise DataContractError(
                    "fingerprint file changed or became a symlink"
                ) from exc
            finally:
                if file_fd is not None:
                    os.close(file_fd)
            relative = "/".join((*relative_parts, name))
            file_states.append((relative, opened_identity))
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(metadata.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_ctime_ns).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_dev).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_ino).encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            total_bytes += metadata.st_size

        directory_after = os.fstat(directory_fd)
        directory_state_before = (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_mtime_ns,
            directory_before.st_ctime_ns,
        )
        directory_state_after = (
            directory_after.st_dev,
            directory_after.st_ino,
            directory_after.st_mtime_ns,
            directory_after.st_ctime_ns,
        )
        if (
            not stat.S_ISDIR(directory_after.st_mode)
            or directory_state_before != directory_state_after
        ):
            raise DataContractError("fingerprint directory changed during scan")

    def verify_final_states(root_fd: int) -> None:
        for relative, expected in directory_states:
            check_limits()
            try:
                current = (
                    os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
                    if relative
                    else os.fstat(root_fd)
                )
            except OSError as exc:
                raise DataContractError(
                    "fingerprint directory changed during final verification"
                ) from exc
            current_state = (
                current.st_dev,
                current.st_ino,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            if not stat.S_ISDIR(current.st_mode) or current_state != expected:
                raise DataContractError(
                    "fingerprint directory changed during final verification"
                )

        for relative, expected in file_states:
            check_limits()
            try:
                current = os.stat(
                    relative,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise DataContractError(
                    "fingerprint file changed during final verification"
                ) from exc
            current_state = (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            if not stat.S_ISREG(current.st_mode) or current_state != expected:
                raise DataContractError(
                    "fingerprint file changed during final verification"
                )

    try:
        root_fd = os.open(resolved_root, directory_flags)
    except OSError as exc:
        raise DataContractError("fingerprint root changed during scan") from exc
    try:
        opened_root = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or (opened_root.st_dev, opened_root.st_ino)
            != (root_metadata.st_dev, root_metadata.st_ino)
        ):
            raise DataContractError("fingerprint root changed during scan")
        visit(root_fd, ())
        verify_final_states(root_fd)
    finally:
        os.close(root_fd)
    return {
        "algorithm": "relative-path-size-mtime-ctime-device-inode-v2",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "scanned_entries": scanned_entries,
        "sha256": digest.hexdigest(),
    }


def _filesystem_state_fingerprint_fallback(
    root: Path,
    *,
    max_entries: int | None,
    max_seconds: float | None,
) -> dict[str, Any]:
    """Fingerprint on platforms without POSIX directory-fd primitives.

    The fallback checks every component with ``lstat`` and validates the opened
    file identity before and after reading. It cannot provide the same
    race-free path traversal guarantee as ``openat`` plus ``O_NOFOLLOW``, but it
    remains bounded and rejects symlinks instead of silently following them.
    """

    root = Path(root).expanduser().absolute()
    try:
        root_before = root.lstat()
    except OSError as exc:
        raise DataContractError("fingerprint root is unavailable") from exc
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise DataContractError("fingerprint root is unavailable")

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    scanned_entries = 0
    started_at = time.monotonic()
    directory_states: list[tuple[Path, tuple[int, int, int, int]]] = []
    file_states: list[tuple[Path, tuple[int, int, int, int, int]]] = []

    def state(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def directory_state(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def check_limits() -> None:
        if max_entries is not None and scanned_entries > max_entries:
            raise DataContractError("filesystem state scan exceeded entry limit")
        if max_seconds is not None and time.monotonic() - started_at > max_seconds:
            raise DataContractError("filesystem state scan exceeded time limit")

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        nonlocal file_count, scanned_entries, total_bytes
        try:
            before = directory.lstat()
        except OSError as exc:
            raise DataContractError("fingerprint directory changed during scan") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise DataContractError("fingerprint directory changed during scan")
        directory_states.append((directory, directory_state(before)))
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise DataContractError("fingerprint directory changed during scan") from exc
        for entry in entries:
            scanned_entries += 1
            check_limits()
            try:
                metadata = entry.lstat()
            except OSError as exc:
                raise DataContractError("fingerprint entry changed during scan") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise DataContractError("fingerprint root cannot contain symlinks")
            if stat.S_ISDIR(metadata.st_mode):
                visit(entry, (*relative_parts, entry.name))
                continue
            if entry.name.endswith(".lock"):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            try:
                with entry.open("rb") as handle:
                    opened = os.fstat(handle.fileno())
                after_open = entry.lstat()
            except OSError as exc:
                raise DataContractError(
                    "fingerprint file changed or became a symlink"
                ) from exc
            if (
                not stat.S_ISREG(opened.st_mode)
                or stat.S_ISLNK(after_open.st_mode)
                or state(opened) != state(metadata)
                or state(opened) != state(after_open)
            ):
                raise DataContractError("fingerprint file changed during scan")
            relative = "/".join((*relative_parts, entry.name))
            file_states.append((entry, state(opened)))
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(metadata.st_size).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_mtime_ns).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_ctime_ns).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_dev).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(metadata.st_ino).encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            total_bytes += metadata.st_size
        try:
            after = directory.lstat()
        except OSError as exc:
            raise DataContractError("fingerprint directory changed during scan") from exc
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISDIR(after.st_mode)
            or directory_state(after) != directory_state(before)
        ):
            raise DataContractError("fingerprint directory changed during scan")

    visit(root, ())
    for directory, expected in directory_states:
        try:
            current = directory.lstat()
        except OSError as exc:
            raise DataContractError(
                "fingerprint directory changed during final verification"
            ) from exc
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or directory_state(current) != expected
        ):
            raise DataContractError(
                "fingerprint directory changed during final verification"
            )
    for file_path, expected in file_states:
        try:
            current = file_path.lstat()
        except OSError as exc:
            raise DataContractError(
                "fingerprint file changed during final verification"
            ) from exc
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or state(current) != expected
        ):
            raise DataContractError(
                "fingerprint file changed during final verification"
            )
    try:
        root_after = root.lstat()
    except OSError as exc:
        raise DataContractError("fingerprint root changed during scan") from exc
    if stat.S_ISLNK(root_after.st_mode) or directory_state(root_after) != directory_state(root_before):
        raise DataContractError("fingerprint root changed during scan")
    return {
        "algorithm": "relative-path-size-mtime-ctime-device-inode-v2",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "scanned_entries": scanned_entries,
        "sha256": digest.hexdigest(),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = _json_bytes(payload)
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_snapshot_tree(snapshot: Path) -> None:
    """Make every staged file and directory durable before publication."""

    if os.name == "nt":
        return
    snapshot = Path(snapshot)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DataContractError("staging snapshot directory is unsafe")
    for current_root, directories, files in os.walk(
        snapshot,
        topdown=False,
        followlinks=False,
    ):
        current = Path(current_root)
        for directory in directories:
            if (current / directory).is_symlink():
                raise DataContractError("staging snapshot cannot contain symlinks")
        for filename in files:
            path = current / filename
            if path.is_symlink() or not path.is_file():
                raise DataContractError("staging snapshot contains an unsafe file")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(current)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"JSON object required: {path}")
    return payload


def _iso_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{field} is required")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DataContractError(f"{field} must include a timezone")
    return value


def _safe_relative(path: Path, root: Path, field: str) -> str:
    path = Path(path).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DataContractError(f"{field} must stay under quant_data root") from exc


def _safe_child(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DataContractError(f"{field} is required")
    candidate = (Path(root).resolve() / relative).resolve()
    try:
        candidate.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise DataContractError(f"{field} escapes quant_data root") from exc
    return candidate


def _snapshot_file(snapshot: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DataContractError(f"{field} is required")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise DataContractError(f"{field} must stay inside snapshot")
    unresolved = Path(snapshot) / relative_path
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(Path(snapshot).resolve())
    except ValueError as exc:
        raise DataContractError(f"{field} must stay inside snapshot") from exc
    current = Path(snapshot)
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise DataContractError(f"{field} cannot use symlinks")
    if not candidate.is_file():
        raise DataContractError(f"{field} is unavailable")
    return candidate


def _validate_bar_frame(frame: pd.DataFrame, path: Path) -> pd.Series:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"bar file {path.name} is missing columns: {missing}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DataContractError(f"bar dates are invalid: {path.name}")
    numeric_columns: dict[str, np.ndarray] = {}
    for column in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        if values.isna().any() or not np.isfinite(numeric).all():
            raise DataContractError(f"bar column {column} is invalid: {path.name}")
        numeric_columns[column] = numeric
    if any(
        bool((numeric_columns[column] <= 0).any())
        for column in ("open", "high", "low", "close")
    ):
        raise DataContractError(f"bar price is not positive: {path.name}")
    if bool((numeric_columns["volume"] < 0).any()):
        raise DataContractError(f"bar volume is negative: {path.name}")
    open_ = numeric_columns["open"]
    high = numeric_columns["high"]
    low = numeric_columns["low"]
    close = numeric_columns["close"]
    upper_reference = np.maximum.reduce([open_, low, close])
    lower_reference = np.minimum.reduce([open_, high, close])
    invalid = (
        (high < upper_reference)
        & ~np.isclose(
            high,
            upper_reference,
            rtol=BAR_OHLC_RELATIVE_TOLERANCE,
            atol=BAR_OHLC_ABSOLUTE_TOLERANCE,
        )
    ) | (
        (low > lower_reference)
        & ~np.isclose(
            low,
            lower_reference,
            rtol=BAR_OHLC_RELATIVE_TOLERANCE,
            atol=BAR_OHLC_ABSOLUTE_TOLERANCE,
        )
    )
    if bool(invalid.any()):
        raise DataContractError(f"bar OHLC relationships are invalid: {path.name}")
    return dates


def build_file_index(bars_dir: Path) -> pd.DataFrame:
    bars_dir = Path(bars_dir)
    files = sorted(bars_dir.glob("*.parquet")) if bars_dir.is_dir() else []
    if not files:
        raise DataContractError("snapshot contains no bar parquet files")
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataContractError(f"cannot read bar parquet: {path.name}") from exc
        dates = _validate_bar_frame(frame, path)
        sources = (
            sorted(str(value) for value in frame["source_id"].dropna().unique())
            if "source_id" in frame.columns
            else []
        )
        rows.append(
            {
                "symbol": path.stem,
                "relative_path": f"bars/{path.name}",
                "row_count": int(len(frame)),
                "first_date": dates.iloc[0].date().isoformat(),
                "last_date": dates.iloc[-1].date().isoformat(),
                "source_ids": json.dumps(sources, ensure_ascii=False),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def build_factor_index(factors_dir: Path, bars_index: pd.DataFrame) -> pd.DataFrame:
    factors_dir = Path(factors_dir)
    rows: list[dict[str, Any]] = []
    for bar_row in bars_index.itertuples(index=False):
        path = factors_dir / f"{bar_row.symbol}.parquet"
        if not path.is_file() or path.is_symlink():
            raise DataContractError(f"factor file is missing: {bar_row.symbol}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataContractError(
                f"cannot read factor file: {bar_row.symbol}"
            ) from exc
        if not {"date", "factor"}.issubset(frame.columns):
            raise DataContractError(f"factor columns are invalid: {bar_row.symbol}")
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        values = pd.to_numeric(frame["factor"], errors="coerce")
        if (
            frame.empty
            or dates.isna().any()
            or dates.duplicated().any()
            or not dates.is_monotonic_increasing
            or values.isna().any()
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or (values <= 0).any()
        ):
            raise DataContractError(f"factor values are invalid: {bar_row.symbol}")
        bars_path = factors_dir.parent / str(bar_row.relative_path)
        bars_dates = pd.to_datetime(
            pd.read_parquet(bars_path, columns=["date"])["date"], errors="coerce"
        ).dt.normalize()
        if set(dates) != set(bars_dates):
            raise DataContractError(f"factor dates do not cover bars: {bar_row.symbol}")
        rows.append(
            {
                "symbol": str(bar_row.symbol),
                "relative_path": f"factors/{path.name}",
                "row_count": int(len(frame)),
                "first_date": dates.iloc[0].date().isoformat(),
                "last_date": dates.iloc[-1].date().isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def _content_hash(snapshot: Path, manifest: Mapping[str, Any] | None = None) -> str:
    files = sorted(
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "DATASET_MANIFEST.json"
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(snapshot).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    if manifest is not None:
        stable = dict(manifest)
        stable.pop("content_sha256", None)
        digest.update(_json_bytes(stable))
    return digest.hexdigest()


def _validate_adjustment_contract(
    *,
    role: str,
    status: str,
    price_basis: str,
    adjustment: Mapping[str, Any],
    snapshot: Path,
) -> dict[str, Any]:
    method = adjustment.get("method")
    if method not in ADJUSTMENT_METHODS:
        raise DataContractError("adjustment method is invalid")
    normalized = dict(adjustment)
    if price_basis not in ADJUSTED_PRICE_BASES:
        if method != "none":
            raise DataContractError("unadjusted data must use adjustment method none")
        return normalized

    canonical_ready = role == "canonical" and status == "ready"
    if canonical_ready and method not in {"local_factor", "provider_native_paired"}:
        raise DataContractError("canonical adjusted data requires adjustment evidence")
    required = {
        "adjusted_at",
        "adjustment_as_of",
        "adjusted_fields",
        "normalization_rule",
        "algorithm_version",
    }
    if canonical_ready and any(not normalized.get(field) for field in required):
        raise DataContractError("canonical adjusted data requires adjustment evidence")
    if canonical_ready:
        _iso_timestamp(normalized["adjusted_at"], "adjusted_at")
        if method == "provider_native_paired" and not normalized.get(
            "paired_raw_snapshot_id"
        ):
            raise DataContractError("canonical adjusted data requires adjustment evidence")
        factors = snapshot / "factors"
        if not factors.is_dir() or not any(factors.glob("*.parquet")):
            raise DataContractError("canonical adjusted data requires adjustment evidence")
        actions = snapshot / "corporate_actions.parquet"
        if not actions.is_file():
            raise DataContractError("canonical adjusted data requires adjustment evidence")
        normalized["factor_path"] = "factors"
    return normalized


def finalize_price_snapshot(
    snapshot: Path,
    *,
    dataset_id: str,
    snapshot_id: str,
    role: str,
    status: str,
    source: Mapping[str, Any],
    price_basis: str,
    adjustment: Mapping[str, Any],
    producer: Mapping[str, Any],
    source_retrieved_at: str,
    upstream_snapshots: list[Mapping[str, str]] | None = None,
) -> Path:
    snapshot = Path(snapshot).resolve()
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise DataContractError("dataset_id is invalid")
    if not snapshot_id or "/" in snapshot_id or snapshot_id in {".", ".."}:
        raise DataContractError("snapshot_id is invalid")
    if role not in ROLES or status not in STATUSES or price_basis not in PRICE_BASES:
        raise DataContractError("dataset role, status, or price basis is invalid")
    if not isinstance(source, Mapping) or not source.get("provider"):
        raise DataContractError("source provider is required")
    if not isinstance(producer, Mapping) or not producer.get("script"):
        raise DataContractError("producer script is required")
    _iso_timestamp(source_retrieved_at, "source_retrieved_at")

    index = build_file_index(snapshot / "bars")
    normalized_adjustment = _validate_adjustment_contract(
        role=role,
        status=status,
        price_basis=price_basis,
        adjustment=adjustment,
        snapshot=snapshot,
    )
    index_path = snapshot / "FILE_INDEX.parquet"
    index.to_parquet(index_path, index=False)
    factor_index_path: str | None = None
    if price_basis in ADJUSTED_PRICE_BASES and role == "canonical" and status == "ready":
        factor_index = build_factor_index(snapshot / "factors", index)
        factor_index_path = "FACTOR_INDEX.parquet"
        factor_index.to_parquet(snapshot / factor_index_path, index=False)
        normalized_adjustment["factor_index_path"] = factor_index_path
    corporate_actions_path = (
        "corporate_actions.parquet"
        if (snapshot / "corporate_actions.parquet").is_file()
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "data_kind": "daily_bars",
        "role": role,
        "status": status,
        "source": dict(source),
        "price_basis": price_basis,
        "source_retrieved_at": source_retrieved_at,
        "adjustment": normalized_adjustment,
        "producer": dict(producer),
        "upstream_snapshots": list(upstream_snapshots or []),
        "date_range": {
            "start": str(index["first_date"].min()),
            "end": str(index["last_date"].max()),
        },
        "latest_trade_date": str(index["last_date"].max()),
        "symbol_count": int(len(index)),
        "row_count": int(index["row_count"].sum()),
        "file_index_path": "FILE_INDEX.parquet",
        "factor_index_path": factor_index_path,
        "corporate_actions_path": corporate_actions_path,
        "corporate_actions_sha256": (
            sha256_file(snapshot / corporate_actions_path)
            if corporate_actions_path
            else None
        ),
    }
    manifest["content_sha256"] = _content_hash(snapshot, manifest)
    atomic_write_json(snapshot / "DATASET_MANIFEST.json", manifest)
    validate_snapshot(snapshot, verify_hashes=True)
    return snapshot / "DATASET_MANIFEST.json"


def _validate_factor_coverage(
    snapshot: Path,
    index: pd.DataFrame,
    *,
    factor_index_path: object,
    verify_hashes: bool,
) -> None:
    if not isinstance(factor_index_path, str) or not factor_index_path:
        raise DataContractError("factor file index is missing")
    factor_index_file = snapshot / factor_index_path
    try:
        factor_index = pd.read_parquet(factor_index_file)
    except Exception as exc:
        raise DataContractError("factor file index is unavailable") from exc
    expected_columns = {
        "symbol",
        "relative_path",
        "row_count",
        "first_date",
        "last_date",
        "sha256",
    }
    if factor_index.empty or not expected_columns.issubset(factor_index.columns):
        raise DataContractError("factor file index is invalid")
    if set(factor_index["symbol"].astype(str)) != set(index["symbol"].astype(str)):
        raise DataContractError("factor file index does not match bar symbols")
    rebuilt = build_factor_index(snapshot / "factors", index)
    rebuilt_by_symbol = rebuilt.set_index("symbol")
    for row in factor_index.itertuples(index=False):
        expected = rebuilt_by_symbol.loc[str(row.symbol)]
        if (
            int(row.row_count) != int(expected["row_count"])
            or str(row.first_date) != str(expected["first_date"])
            or str(row.last_date) != str(expected["last_date"])
            or str(row.relative_path) != str(expected["relative_path"])
        ):
            raise DataContractError(f"factor file index is stale: {row.symbol}")
        if verify_hashes and str(row.sha256) != str(expected["sha256"]):
            raise DataContractError(f"factor hash mismatch: {row.symbol}")


def validate_snapshot(snapshot: Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DataContractError("snapshot directory is unavailable")
    manifest = _read_json(snapshot / "DATASET_MANIFEST.json")
    required = {
        "schema_version",
        "dataset_id",
        "snapshot_id",
        "data_kind",
        "role",
        "status",
        "source",
        "price_basis",
        "source_retrieved_at",
        "adjustment",
        "producer",
        "date_range",
        "latest_trade_date",
        "file_index_path",
        "content_sha256",
    }
    if not required.issubset(manifest) or manifest["schema_version"] != SCHEMA_VERSION:
        raise DataContractError("snapshot manifest is incomplete or has the wrong version")
    if manifest["role"] not in ROLES or manifest["status"] not in STATUSES:
        raise DataContractError("snapshot role or status is invalid")
    if manifest["price_basis"] not in PRICE_BASES:
        raise DataContractError("snapshot price basis is invalid")
    _iso_timestamp(manifest["source_retrieved_at"], "source_retrieved_at")
    index_path = snapshot / str(manifest["file_index_path"])
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        raise DataContractError("snapshot file index is unavailable") from exc
    expected_index_columns = {
        "symbol",
        "relative_path",
        "row_count",
        "first_date",
        "last_date",
        "source_ids",
        "sha256",
    }
    if index.empty or not expected_index_columns.issubset(index.columns):
        raise DataContractError("snapshot file index is invalid")
    for row in index.itertuples(index=False):
        path = (snapshot / str(row.relative_path)).resolve()
        try:
            path.relative_to(snapshot)
        except ValueError as exc:
            raise DataContractError("file index path escapes snapshot") from exc
        if not path.is_file() or path.is_symlink():
            raise DataContractError(f"indexed bar file is unavailable: {row.symbol}")
        frame = pd.read_parquet(path)
        dates = _validate_bar_frame(frame, path)
        if (
            int(row.row_count) != len(frame)
            or str(row.first_date) != dates.iloc[0].date().isoformat()
            or str(row.last_date) != dates.iloc[-1].date().isoformat()
        ):
            raise DataContractError(f"file index metadata is stale: {row.symbol}")
        if verify_hashes and str(row.sha256) != sha256_file(path):
            raise DataContractError(f"bar hash mismatch: {row.symbol}")
    adjustment = manifest["adjustment"]
    if not isinstance(adjustment, dict):
        raise DataContractError("adjustment contract is invalid")
    _validate_adjustment_contract(
        role=str(manifest["role"]),
        status=str(manifest["status"]),
        price_basis=str(manifest["price_basis"]),
        adjustment=adjustment,
        snapshot=snapshot,
    )
    if manifest["price_basis"] in ADJUSTED_PRICE_BASES and manifest["role"] == "canonical":
        _validate_factor_coverage(
            snapshot,
            index,
            factor_index_path=manifest.get("factor_index_path"),
            verify_hashes=verify_hashes,
        )
    actions_path = manifest.get("corporate_actions_path")
    actions_hash = manifest.get("corporate_actions_sha256")
    if actions_path:
        actions = snapshot / str(actions_path)
        if not actions.is_file() or actions.is_symlink():
            raise DataContractError("corporate actions file is unavailable")
        if verify_hashes and actions_hash != sha256_file(actions):
            raise DataContractError("corporate actions hash mismatch")
    if verify_hashes and str(manifest["content_sha256"]) != _content_hash(
        snapshot, manifest
    ):
        raise DataContractError("snapshot content hash mismatch")
    return manifest


def _build_qlib_file_index(
    snapshot: Path,
    *,
    hash_files: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(snapshot).rglob("*")):
        if path.is_symlink():
            raise DataContractError(
                f"Qlib snapshot cannot contain symlinks: {path.relative_to(snapshot)}"
            )
        if (
            not path.is_file()
            or path.name in {"DATASET_MANIFEST.json", "FILE_INDEX.parquet"}
        ):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path) if hash_files else None,
            }
        )
    if not rows:
        raise DataContractError("Qlib snapshot contains no indexed files")
    return pd.DataFrame(rows)


def finalize_qlib_snapshot(
    snapshot: Path,
    *,
    dataset_id: str,
    snapshot_id: str,
    source: Mapping[str, Any],
    source_retrieved_at: str,
    producer: Mapping[str, Any],
    upstream_snapshot: Mapping[str, str],
    benchmark_evidence: Mapping[str, Any] | None = None,
    instrument_pool_evidence: Mapping[str, Any] | None = None,
    transformation_evidence: Mapping[str, Any] | None = None,
    dependency_snapshots: Sequence[Mapping[str, str]] | None = None,
) -> Path:
    snapshot = Path(snapshot).resolve()
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise DataContractError("dataset_id is invalid")
    if not snapshot_id or "/" in snapshot_id or snapshot_id in {".", ".."}:
        raise DataContractError("snapshot_id is invalid")
    if (
        not source.get("provider")
        or not producer.get("script")
        or not producer.get("version")
    ):
        raise DataContractError("Qlib source and producer are required")
    normalized_producer = json.loads(json.dumps(dict(producer)))
    _iso_timestamp(source_retrieved_at, "source_retrieved_at")
    required_upstream = {
        "dataset_id",
        "snapshot_id",
        "content_sha256",
        "price_basis",
    }
    if not required_upstream.issubset(upstream_snapshot):
        raise DataContractError("Qlib upstream snapshot contract is incomplete")
    if upstream_snapshot["price_basis"] != "qfq":
        raise DataContractError("Qlib v2 requires a qfq upstream snapshot")
    normalized_dependencies = [
        json.loads(json.dumps(dict(item))) for item in (dependency_snapshots or [])
    ]
    dependency_ids: set[str] = set()
    for dependency in normalized_dependencies:
        if (
            not required_upstream.issubset(dependency)
            or dependency["price_basis"] != "qfq"
            or not DATASET_ID_RE.fullmatch(str(dependency["dataset_id"]))
            or not str(dependency["snapshot_id"]).strip()
            or not SHA256_RE.fullmatch(str(dependency["content_sha256"]))
            or dependency["dataset_id"] in dependency_ids
            or dependency == dict(upstream_snapshot)
        ):
            raise DataContractError("Qlib dependency snapshot contract is invalid")
        dependency_ids.add(str(dependency["dataset_id"]))
    calendar_path = snapshot / "calendars" / "day.txt"
    instruments_path = snapshot / "instruments" / "all.txt"
    provider_manifest_path = snapshot / "PROVIDER_MANIFEST.json"
    try:
        calendar = [
            value.strip()
            for value in calendar_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        instruments = [
            value.strip()
            for value in instruments_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    except OSError as exc:
        raise DataContractError("Qlib calendar or instruments are unavailable") from exc
    parsed_calendar = pd.to_datetime(calendar, errors="coerce")
    if (
        not calendar
        or parsed_calendar.isna().any()
        or calendar != sorted(set(calendar))
        or not instruments
    ):
        raise DataContractError("Qlib calendar or instruments are invalid")
    provider_manifest = _read_json(provider_manifest_path)
    if provider_manifest.get("source_snapshot") != dict(upstream_snapshot):
        raise DataContractError("Qlib provider points to the wrong upstream snapshot")
    provider_manifest["producer"] = normalized_producer
    if normalized_dependencies:
        provider_manifest["dependency_snapshots"] = normalized_dependencies
    if benchmark_evidence is not None:
        normalized_benchmarks = json.loads(json.dumps(dict(benchmark_evidence)))
        _validate_qlib_benchmark_evidence(snapshot, normalized_benchmarks)
        provider_manifest["benchmark_complete"] = True
        provider_manifest["benchmarks"] = normalized_benchmarks
    if instrument_pool_evidence is not None:
        normalized_pools = json.loads(json.dumps(dict(instrument_pool_evidence)))
        _validate_qlib_instrument_pool_evidence(snapshot, normalized_pools)
        provider_manifest["instrument_pools"] = normalized_pools
    if transformation_evidence is not None:
        normalized_transformation = json.loads(
            json.dumps(dict(transformation_evidence))
        )
        if normalized_transformation.get("source_snapshot") != dict(
            upstream_snapshot
        ):
            raise DataContractError(
                "Qlib transformation points to the wrong upstream snapshot"
            )
        _validate_qlib_transformation_evidence(snapshot, normalized_transformation)
        provider_manifest["transformation"] = normalized_transformation
    atomic_write_json(provider_manifest_path, provider_manifest)
    index = _build_qlib_file_index(snapshot)
    index.to_parquet(snapshot / "FILE_INDEX.parquet", index=False)
    feature_files = index["relative_path"].str.startswith("features/")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "data_kind": "qlib_provider",
        "role": "canonical",
        "status": "ready",
        "source": dict(source),
        "price_basis": "qfq",
        "source_retrieved_at": source_retrieved_at,
        "adjustment": {
            "method": "upstream_qfq_snapshot",
            "upstream_snapshot_id": upstream_snapshot["snapshot_id"],
        },
        "producer": normalized_producer,
        "upstream_snapshots": [dict(upstream_snapshot)],
        "dependency_snapshots": normalized_dependencies,
        "date_range": {"start": calendar[0], "end": calendar[-1]},
        "latest_trade_date": calendar[-1],
        "instrument_count": len(instruments),
        "feature_file_count": int(feature_files.sum()),
        "provider_manifest_path": "PROVIDER_MANIFEST.json",
        "file_index_path": "FILE_INDEX.parquet",
    }
    if benchmark_evidence is not None:
        manifest["benchmark_complete"] = True
        manifest["benchmarks"] = normalized_benchmarks
    if instrument_pool_evidence is not None:
        manifest["instrument_pools"] = normalized_pools
    if transformation_evidence is not None:
        manifest["transformation"] = normalized_transformation
    manifest["content_sha256"] = _content_hash(snapshot, manifest)
    atomic_write_json(snapshot / "DATASET_MANIFEST.json", manifest)
    validate_qlib_snapshot(snapshot, verify_hashes=True)
    return snapshot / "DATASET_MANIFEST.json"


def _read_qlib_feature(path: Path, calendar_size: int) -> np.ndarray:
    try:
        raw = np.fromfile(path, dtype="<f4")
    except OSError as exc:
        raise DataContractError(f"Qlib benchmark feature is unavailable: {path}") from exc
    if len(raw) < 2 or not np.isfinite(raw[0]) or raw[0] != int(raw[0]):
        raise DataContractError(f"Qlib benchmark feature is invalid: {path}")
    start = int(raw[0])
    values = raw[1:]
    if start < 0 or start + len(values) > calendar_size:
        raise DataContractError(f"Qlib benchmark feature range is invalid: {path}")
    aligned = np.full(calendar_size, np.nan, dtype=np.float32)
    aligned[start : start + len(values)] = values
    return aligned


def _validate_qlib_instrument_pool_evidence(
    snapshot: Path,
    evidence: Mapping[str, Any],
) -> None:
    if not isinstance(evidence, Mapping):
        raise DataContractError("Qlib instrument pool evidence is invalid")
    source = evidence.get("source")
    pools = evidence.get("pools")
    if (
        not isinstance(source, Mapping)
        or not {"dataset_id", "path", "sha256"}.issubset(source)
        or not isinstance(pools, Mapping)
        or set(pools) != {"csi300", "csi500", "csi800"}
    ):
        raise DataContractError("Qlib instrument pool evidence is incomplete")
    source_path = Path(str(source.get("path", "")))
    if source_path.is_absolute() or ".." in source_path.parts or not source_path.parts:
        raise DataContractError("Qlib instrument pool source path is invalid")
    if not isinstance(source.get("sha256"), str) or not SHA256_RE.fullmatch(
        str(source["sha256"])
    ):
        raise DataContractError("Qlib instrument pool source hash is invalid")
    all_path = snapshot / "instruments" / "all.txt"
    try:
        all_symbols = {
            line.split()[0].upper()
            for line in all_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError as exc:
        raise DataContractError("Qlib all-instrument pool is unavailable") from exc
    if not all_symbols:
        raise DataContractError("Qlib all-instrument pool is empty")
    for pool_name in ("csi300", "csi500", "csi800"):
        item = pools[pool_name]
        expected_relative = f"instruments/{pool_name}.txt"
        if (
            not isinstance(item, Mapping)
            or item.get("relative_path") != expected_relative
            or not isinstance(item.get("sha256"), str)
            or not SHA256_RE.fullmatch(str(item["sha256"]))
        ):
            raise DataContractError(f"Qlib instrument pool evidence is invalid: {pool_name}")
        path = snapshot / expected_relative
        if path.is_symlink() or not path.is_file():
            raise DataContractError(f"Qlib instrument pool is unavailable: {pool_name}")
        if item["sha256"] != sha256_file(path):
            raise DataContractError(f"Qlib instrument pool hash mismatch: {pool_name}")
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        symbols = [line.split()[0].upper() for line in lines if len(line.split()) == 3]
        if (
            len(symbols) != len(lines)
            or len(symbols) != int(item.get("instrument_count", -1))
            or len(symbols) != len(set(symbols))
            or not set(symbols).issubset(all_symbols)
            or set(symbols).intersection(QLIB_BENCHMARK_SYMBOLS)
        ):
            raise DataContractError(f"Qlib instrument pool contents are invalid: {pool_name}")


def _validate_qlib_transformation_evidence(
    snapshot: Path,
    evidence: Mapping[str, Any],
) -> None:
    required = {"name", "algorithm", "source_snapshot", "summary", "artifacts"}
    if not isinstance(evidence, Mapping) or not required.issubset(evidence):
        raise DataContractError("Qlib transformation evidence is incomplete")
    algorithm = evidence.get("algorithm")
    source_snapshot = evidence.get("source_snapshot")
    summary = evidence.get("summary")
    artifacts = evidence.get("artifacts")
    if (
        not str(evidence.get("name") or "").strip()
        or not isinstance(algorithm, Mapping)
        or not str(algorithm.get("script") or "").strip()
        or not str(algorithm.get("version") or "").strip()
        or not isinstance(source_snapshot, Mapping)
        or not {
            "dataset_id",
            "snapshot_id",
            "content_sha256",
            "price_basis",
        }.issubset(source_snapshot)
        or not isinstance(summary, Mapping)
        or not isinstance(artifacts, Mapping)
        or not artifacts
    ):
        raise DataContractError("Qlib transformation evidence is invalid")
    for field in (
        "source_symbols",
        "output_symbols",
        "source_rows",
        "output_rows",
        "excluded_symbols",
        "masked_volume_rows",
        "masked_event_rows",
    ):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataContractError("Qlib transformation summary is invalid")
    for name, artifact in artifacts.items():
        if not str(name).strip() or not isinstance(artifact, Mapping):
            raise DataContractError("Qlib transformation artifact is invalid")
        relative_path = artifact.get("relative_path")
        expected_hash = artifact.get("sha256")
        expected_size = artifact.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise DataContractError("Qlib transformation artifact is invalid")
        path = _snapshot_file(snapshot, relative_path, "Qlib transformation artifact")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise DataContractError("Qlib transformation artifact hash mismatch")


def _validate_qlib_benchmark_evidence(
    snapshot: Path,
    evidence: Mapping[str, Any],
    *,
    evidence_root: Path | None = None,
) -> None:
    if not isinstance(evidence, Mapping):
        raise DataContractError("Qlib benchmark evidence is invalid")
    calendar_path = snapshot / "calendars" / "day.txt"
    try:
        calendar = [
            value.strip()
            for value in calendar_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    except OSError as exc:
        raise DataContractError("Qlib benchmark calendar is unavailable") from exc
    if evidence.get("calendar_sha256") != sha256_file(calendar_path):
        raise DataContractError("Qlib benchmark calendar hash mismatch")
    source = evidence.get("source")
    required_source = {
        "provider",
        "interface",
        "source_dataset_id",
        "source_path",
        "source_run_id",
        "evidence_path",
        "evidence_sha256",
    }
    if not isinstance(source, Mapping) or not required_source.issubset(source):
        raise DataContractError("Qlib benchmark source evidence is incomplete")
    source_manifest_hash = source.get("source_manifest_sha256") or source.get(
        "provider_manifest_sha256"
    )
    if not isinstance(source_manifest_hash, str) or not SHA256_RE.fullmatch(
        source_manifest_hash
    ):
        raise DataContractError("Qlib benchmark source manifest hash is invalid")
    for hash_field in (
        "evidence_sha256",
        "source_manifest_sha256",
        "provider_manifest_sha256",
    ):
        if hash_field not in source:
            continue
        if not isinstance(source.get(hash_field), str) or not SHA256_RE.fullmatch(
            str(source[hash_field])
        ):
            raise DataContractError("Qlib benchmark source hash is invalid")
    for path_field in ("source_path", "evidence_path"):
        raw_path = Path(str(source.get(path_field, "")))
        if raw_path.is_absolute() or ".." in raw_path.parts or not raw_path.parts:
            raise DataContractError("Qlib benchmark source path is invalid")
    if evidence_root is not None:
        evidence_path = _safe_child(
            Path(evidence_root),
            source["evidence_path"],
            "Qlib benchmark evidence path",
        )
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise DataContractError("Qlib benchmark source evidence is unavailable")
        if sha256_file(evidence_path) != source["evidence_sha256"]:
            raise DataContractError("Qlib benchmark source evidence hash mismatch")
    declared_required = evidence.get("required_symbols")
    if declared_required is None:
        required_symbols = QLIB_LEGACY_BENCHMARK_SYMBOLS
    elif isinstance(declared_required, list) and tuple(declared_required) == QLIB_BENCHMARK_SYMBOLS:
        required_symbols = QLIB_BENCHMARK_SYMBOLS
    else:
        raise DataContractError("Qlib benchmark required symbols are invalid")
    symbols = evidence.get("symbols")
    if not isinstance(symbols, Mapping) or set(symbols) != set(required_symbols):
        raise DataContractError("Qlib benchmark symbol evidence is incomplete")
    for symbol in required_symbols:
        item = symbols[symbol]
        if not isinstance(item, Mapping):
            raise DataContractError(f"Qlib benchmark evidence is invalid: {symbol}")
        coverage = item.get("coverage")
        features = item.get("features")
        if (
            not isinstance(coverage, Mapping)
            or not isinstance(features, Mapping)
            or set(features) != set(QLIB_BENCHMARK_FIELDS)
        ):
            raise DataContractError(f"Qlib benchmark evidence is incomplete: {symbol}")
        try:
            start = calendar.index(str(coverage["start"]))
            end = calendar.index(str(coverage["end"]))
            observations = int(coverage["observations"])
        except (KeyError, ValueError, TypeError) as exc:
            raise DataContractError(
                f"Qlib benchmark coverage is invalid: {symbol}"
            ) from exc
        if start > end or end != len(calendar) - 1 or observations != end - start + 1:
            raise DataContractError(f"Qlib benchmark coverage is invalid: {symbol}")
        masks: list[np.ndarray] = []
        for field in QLIB_BENCHMARK_FIELDS:
            feature = features[field]
            expected_relative = f"features/{symbol.lower()}/{field}.day.bin"
            if not isinstance(feature, Mapping) or feature.get("relative_path") != expected_relative:
                raise DataContractError(
                    f"Qlib benchmark feature path is invalid: {symbol} {field}"
                )
            path = snapshot / expected_relative
            if path.is_symlink() or not path.is_file():
                raise DataContractError(
                    f"Qlib benchmark feature is unavailable: {symbol} {field}"
                )
            if int(feature.get("size_bytes", -1)) != path.stat().st_size:
                raise DataContractError(
                    f"Qlib benchmark file size mismatch: {symbol} {field}"
                )
            if feature.get("sha256") != sha256_file(path):
                raise DataContractError(
                    f"Qlib benchmark file hash mismatch: {symbol} {field}"
                )
            aligned = _read_qlib_feature(path, len(calendar))
            required = aligned if field == "factor" else aligned[start:]
            valid = np.isfinite(required)
            valid &= required >= 0 if field == "volume" else required > 0
            if not bool(valid.all()):
                raise DataContractError(
                    f"Qlib benchmark required dates are incomplete: {symbol} {field}"
                )
            if field != "factor":
                masks.append(np.isfinite(aligned))
        if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
            raise DataContractError(f"Qlib benchmark dates disagree: {symbol}")


def validate_qlib_snapshot(
    snapshot: Path,
    *,
    verify_hashes: bool = True,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DataContractError("Qlib snapshot directory is unavailable")
    manifest = _read_json(snapshot / "DATASET_MANIFEST.json")
    required = {
        "schema_version",
        "dataset_id",
        "snapshot_id",
        "data_kind",
        "role",
        "status",
        "source",
        "price_basis",
        "source_retrieved_at",
        "producer",
        "upstream_snapshots",
        "date_range",
        "latest_trade_date",
        "instrument_count",
        "feature_file_count",
        "provider_manifest_path",
        "file_index_path",
        "content_sha256",
    }
    if (
        not required.issubset(manifest)
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["data_kind"] != "qlib_provider"
        or manifest["status"] != "ready"
        or manifest["price_basis"] != "qfq"
    ):
        raise DataContractError("Qlib snapshot manifest is incomplete or invalid")
    _iso_timestamp(manifest["source_retrieved_at"], "source_retrieved_at")
    upstream = manifest["upstream_snapshots"]
    if not isinstance(upstream, list) or len(upstream) != 1:
        raise DataContractError("Qlib upstream snapshot contract is invalid")
    provider_manifest_path = _snapshot_file(
        snapshot,
        manifest["provider_manifest_path"],
        "Qlib provider manifest",
    )
    provider_manifest = _read_json(provider_manifest_path)
    if provider_manifest.get("source_snapshot") != upstream[0]:
        raise DataContractError("Qlib provider and dataset lineage disagree")
    producer = manifest.get("producer")
    if (
        not isinstance(producer, Mapping)
        or not str(producer.get("script") or "").strip()
        or not str(producer.get("version") or "").strip()
    ):
        raise DataContractError("Qlib producer contract is invalid")
    provider_producer = provider_manifest.get("producer")
    if not (
        provider_producer == producer
        or (
            provider_producer is None
            and _legacy_provider_manifest_may_omit_producer(
                manifest.get("dataset_id"), producer
            )
        )
    ):
        raise DataContractError("Qlib provider and dataset producer lineage disagree")
    dependencies = manifest.get("dependency_snapshots", [])
    if not isinstance(dependencies, list):
        raise DataContractError("Qlib dependency snapshot contract is invalid")
    if provider_manifest.get("dependency_snapshots", []) != dependencies:
        raise DataContractError("Qlib provider and dataset dependencies disagree")
    dependency_ids: set[str] = set()
    for dependency in dependencies:
        if (
            not isinstance(dependency, Mapping)
            or not {
                "dataset_id",
                "snapshot_id",
                "content_sha256",
                "price_basis",
            }.issubset(dependency)
            or dependency.get("price_basis") != "qfq"
            or not DATASET_ID_RE.fullmatch(str(dependency.get("dataset_id") or ""))
            or not str(dependency.get("snapshot_id") or "").strip()
            or not SHA256_RE.fullmatch(str(dependency.get("content_sha256") or ""))
            or str(dependency["dataset_id"]) in dependency_ids
            or dependency == upstream[0]
        ):
            raise DataContractError("Qlib dependency snapshot contract is invalid")
        dependency_ids.add(str(dependency["dataset_id"]))
    if manifest.get("dataset_id") == "qlib_ashare_cleaned_v2":
        dependency_is_exact = (
            len(dependencies) == 1
            and dependencies[0].get("dataset_id") == "qlib_ashare_v2"
        )
        if not dependency_is_exact and not (
            not dependencies
            and _legacy_cleaned_provider_may_omit_dependency(producer)
        ):
            raise DataContractError(
                "cleaned Qlib must declare its exact canonical Qlib dependency"
            )
    benchmark_complete = manifest.get("benchmark_complete")
    provider_benchmark_complete = provider_manifest.get("benchmark_complete")
    if benchmark_complete is not None or provider_benchmark_complete is not None:
        if benchmark_complete is not True or provider_benchmark_complete is not True:
            raise DataContractError("Qlib benchmark completion state is invalid")
        if manifest.get("benchmarks") != provider_manifest.get("benchmarks"):
            raise DataContractError("Qlib provider and dataset benchmark evidence disagree")
        _validate_qlib_benchmark_evidence(
            snapshot,
            manifest["benchmarks"],
            evidence_root=evidence_root,
        )
    elif "benchmarks" in manifest or "benchmarks" in provider_manifest:
        raise DataContractError("Qlib benchmark evidence lacks completion state")
    if "instrument_pools" in manifest or "instrument_pools" in provider_manifest:
        if manifest.get("instrument_pools") != provider_manifest.get(
            "instrument_pools"
        ):
            raise DataContractError(
                "Qlib provider and dataset instrument pool evidence disagree"
            )
        _validate_qlib_instrument_pool_evidence(
            snapshot, manifest["instrument_pools"]
        )
    if "transformation" in manifest or "transformation" in provider_manifest:
        if manifest.get("transformation") != provider_manifest.get("transformation"):
            raise DataContractError(
                "Qlib provider and dataset transformation evidence disagree"
            )
        transformation = manifest["transformation"]
        if transformation.get("source_snapshot") != upstream[0]:
            raise DataContractError(
                "Qlib transformation and dataset lineage disagree"
            )
        _validate_qlib_transformation_evidence(snapshot, transformation)
    try:
        file_index_path = _snapshot_file(
            snapshot,
            manifest["file_index_path"],
            "Qlib file index",
        )
        recorded = pd.read_parquet(file_index_path)
    except Exception as exc:
        raise DataContractError("Qlib file index is unavailable") from exc
    required_index = {"relative_path", "size_bytes", "sha256"}
    if recorded.empty or not required_index.issubset(recorded.columns):
        raise DataContractError("Qlib file index is invalid")
    rebuilt = _build_qlib_file_index(snapshot, hash_files=verify_hashes)
    recorded = recorded.sort_values("relative_path").reset_index(drop=True)
    rebuilt = rebuilt.sort_values("relative_path").reset_index(drop=True)
    if recorded["relative_path"].tolist() != rebuilt["relative_path"].tolist():
        raise DataContractError("Qlib file index is stale")
    for actual, expected in zip(
        recorded.itertuples(index=False), rebuilt.itertuples(index=False)
    ):
        if int(actual.size_bytes) != int(expected.size_bytes):
            raise DataContractError(f"Qlib file size mismatch: {actual.relative_path}")
        if verify_hashes and str(actual.sha256) != str(expected.sha256):
            raise DataContractError(f"Qlib file hash mismatch: {actual.relative_path}")
    if int(manifest["feature_file_count"]) != int(
        rebuilt["relative_path"].str.startswith("features/").sum()
    ):
        raise DataContractError("Qlib feature file count is stale")
    if verify_hashes and str(manifest["content_sha256"]) != _content_hash(
        snapshot, manifest
    ):
        raise DataContractError("Qlib snapshot content hash mismatch")
    return manifest


def validate_dataset_snapshot(
    snapshot: Path,
    *,
    verify_hashes: bool = True,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(Path(snapshot) / "DATASET_MANIFEST.json")
    if manifest.get("data_kind") == "qlib_provider":
        return validate_qlib_snapshot(
            snapshot,
            verify_hashes=verify_hashes,
            evidence_root=evidence_root,
        )
    return validate_snapshot(snapshot, verify_hashes=verify_hashes)


def validate_latest_pointer(
    pointer_path: Path,
    *,
    quant_data_root: Path,
    expected_dataset_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Validate a LATEST pointer before following it to an immutable snapshot."""

    quant_data_root = Path(quant_data_root).expanduser().resolve()
    pointer_path = Path(pointer_path).expanduser().resolve()
    _safe_relative(pointer_path, quant_data_root, "latest_pointer")
    if pointer_path.name != "LATEST.json":
        raise DataContractError("latest pointer filename is invalid")
    pointer = _read_json(pointer_path)
    required = {
        "schema_version",
        "dataset_id",
        "snapshot_id",
        "snapshot_path",
        "manifest_sha256",
    }
    if not required.issubset(pointer) or pointer.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("latest pointer contract is incomplete")
    dataset_id = str(pointer.get("dataset_id") or "")
    snapshot_id = str(pointer.get("snapshot_id") or "")
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise DataContractError("latest pointer dataset_id is invalid")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise DataContractError("latest pointer belongs to another dataset")
    if not snapshot_id or Path(snapshot_id).name != snapshot_id:
        raise DataContractError("latest pointer snapshot_id is invalid")
    manifest_sha256 = str(pointer.get("manifest_sha256") or "")
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise DataContractError("latest pointer manifest hash is invalid")
    snapshot = _safe_child(
        quant_data_root, pointer.get("snapshot_path"), "snapshot_path"
    )
    snapshots_root = (pointer_path.parent / "snapshots").resolve()
    try:
        snapshot.relative_to(snapshots_root)
    except ValueError as exc:
        raise DataContractError(
            "latest pointer snapshot is outside its dataset root"
        ) from exc
    if snapshot.parent != snapshots_root or snapshot.name != snapshot_id:
        raise DataContractError("latest pointer snapshot path is invalid")
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    try:
        actual_manifest_sha256 = sha256_file(manifest_path)
    except OSError as exc:
        raise DataContractError("latest pointer snapshot manifest is unavailable") from exc
    if actual_manifest_sha256 != manifest_sha256:
        raise DataContractError("latest pointer manifest hash mismatch")
    return pointer, snapshot


@contextmanager
def dataset_publication_lock(quant_data_root: Path):
    """Serialize every formal V2 snapshot and root-catalog publication."""

    root = Path(quant_data_root).expanduser().resolve()
    lock_path = root / "runtime" / "locks" / "dataset_publication.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        try:
            lock_file(handle, exclusive=True, blocking=False)
            acquired = True
        except BlockingIOError as exc:
            raise DataContractError(
                "dataset publication is already in progress"
            ) from exc
        yield
    finally:
        try:
            if acquired:
                unlock_file(handle)
        finally:
            handle.close()


def publish_snapshot(
    staging_snapshot: Path,
    *,
    dataset_root: Path,
    quant_data_root: Path,
    publication_lock_held: bool = False,
) -> Path:
    return publish_snapshot_group(
        [(staging_snapshot, dataset_root)],
        quant_data_root=quant_data_root,
        publication_lock_held=publication_lock_held,
    )[0]


def _publish_snapshot_group_unlocked(
    snapshots: list[tuple[Path, Path]],
    *,
    quant_data_root: Path,
    update_root_catalog: bool = False,
    validated_cache: ValidatedResolutionCache | None = None,
) -> list[Path]:
    """Publish related snapshots and roll back every formal pointer on failure."""

    if not snapshots:
        raise DataContractError("snapshot group is empty")
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    staging_root = quant_data_root / "staging"
    prepared: list[dict[str, Any]] = []
    dataset_roots: set[Path] = set()
    for staging_snapshot, dataset_root in snapshots:
        staging_snapshot = Path(staging_snapshot).expanduser().resolve()
        dataset_root = Path(dataset_root).expanduser().resolve()
        _safe_relative(dataset_root, quant_data_root, "dataset_root")
        try:
            staging_snapshot.relative_to(staging_root)
        except ValueError as exc:
            raise DataContractError(
                "staging snapshot must be under quant_data/staging"
            ) from exc
        if dataset_root in dataset_roots:
            raise DataContractError("snapshot group contains a duplicate dataset root")
        dataset_roots.add(dataset_root)
        manifest = validate_dataset_snapshot(
            staging_snapshot,
            verify_hashes=True,
            evidence_root=quant_data_root,
        )
        if manifest["status"] != "ready":
            raise DataContractError("only ready snapshots can be published")
        snapshot_id = str(manifest["snapshot_id"])
        destination = dataset_root / "snapshots" / snapshot_id
        if destination.exists():
            raise DataContractError("snapshot_id is already published")
        prepared.append(
            {
                "staging": staging_snapshot,
                "dataset_root": dataset_root,
                "manifest": manifest,
                "destination": destination,
            }
        )

    catalog_backup = (
        _load_root_manifest(quant_data_root) if update_root_catalog else None
    )
    pointer_backups: dict[Path, dict[str, Any] | None] = {}
    moved: list[dict[str, Any]] = []
    try:
        for item in prepared:
            destination = item["destination"]
            _fsync_snapshot_tree(item["staging"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(item["staging"], destination)
            moved.append(item)
            _fsync_directory(item["staging"].parent)
            _fsync_directory(destination.parent)
        published_at = datetime.now().astimezone().isoformat(timespec="seconds")
        for item in prepared:
            pointer_path = item["dataset_root"] / "LATEST.json"
            pointer_backups[pointer_path] = (
                _read_json(pointer_path) if pointer_path.is_file() else None
            )
            manifest = item["manifest"]
            destination = item["destination"]
            atomic_write_json(
                pointer_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": manifest["dataset_id"],
                    "snapshot_id": manifest["snapshot_id"],
                    "snapshot_path": _safe_relative(
                        destination, quant_data_root, "snapshot_path"
                    ),
                    "manifest_sha256": sha256_file(
                        destination / "DATASET_MANIFEST.json"
                    ),
                    "published_at": published_at,
                },
            )
        if catalog_backup is not None:
            datasets = dict(catalog_backup["datasets"])
            for item in prepared:
                manifest = item["manifest"]
                catalog_entry = _catalog_entry_from_validated_manifest(
                    item["destination"],
                    quant_data_root=quant_data_root,
                    dataset_root=item["dataset_root"],
                    manifest=manifest,
                )
                datasets[str(manifest["dataset_id"])] = catalog_entry
                item["catalog_entry"] = catalog_entry
            write_root_manifest(
                quant_data_root,
                datasets,
                existing=catalog_backup,
            )
    except Exception as publish_error:
        rollback_errors: list[tuple[str, Exception]] = []
        for pointer_path, backup in pointer_backups.items():
            try:
                if backup is None:
                    pointer_path.unlink(missing_ok=True)
                else:
                    atomic_write_json(pointer_path, backup)
            except Exception as exc:
                rollback_errors.append((f"restore pointer {pointer_path}", exc))
        if catalog_backup is not None:
            try:
                atomic_write_json(
                    quant_data_root / "DATA_MANIFEST.json", catalog_backup
                )
            except Exception as exc:
                rollback_errors.append(("restore DATA_MANIFEST.json", exc))
        for item in reversed(moved):
            try:
                item["staging"].parent.mkdir(parents=True, exist_ok=True)
                os.replace(item["destination"], item["staging"])
                _fsync_directory(item["destination"].parent)
                _fsync_directory(item["staging"].parent)
            except Exception as exc:
                rollback_errors.append(
                    (f"restore staged snapshot {item['manifest']['snapshot_id']}", exc)
                )
        if rollback_errors:
            failures = "; ".join(
                f"{label}: {type(exc).__name__}: {exc}"
                for label, exc in rollback_errors
            )
            raise OSError(
                "snapshot publication failed: "
                f"{type(publish_error).__name__}: {publish_error}; "
                f"rollback failed: {failures}"
            ) from publish_error
        raise
    if validated_cache is not None:
        for item in prepared:
            manifest = item["manifest"]
            catalog_entry = item.get("catalog_entry")
            if not isinstance(catalog_entry, dict):
                catalog_entry = _catalog_entry_from_validated_manifest(
                    item["destination"],
                    quant_data_root=quant_data_root,
                    dataset_root=item["dataset_root"],
                    manifest=manifest,
                )
            dataset_id = str(manifest["dataset_id"])
            validated_cache[dataset_id] = ResolvedDataset(
                dataset_id,
                item["destination"],
                dict(manifest),
                dict(catalog_entry),
            )
    return [item["destination"] for item in prepared]


def publish_snapshot_group(
    snapshots: list[tuple[Path, Path]],
    *,
    quant_data_root: Path,
    update_root_catalog: bool = False,
    publication_lock_held: bool = False,
    validated_cache: ValidatedResolutionCache | None = None,
) -> list[Path]:
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    if publication_lock_held:
        return _publish_snapshot_group_unlocked(
            snapshots,
            quant_data_root=quant_data_root,
            update_root_catalog=update_root_catalog,
            validated_cache=validated_cache,
        )
    with dataset_publication_lock(quant_data_root):
        return _publish_snapshot_group_unlocked(
            snapshots,
            quant_data_root=quant_data_root,
            update_root_catalog=update_root_catalog,
            validated_cache=validated_cache,
        )


def publish_qlib_snapshot(
    staging_snapshot: Path,
    *,
    dataset_root: Path,
    quant_data_root: Path,
    update_root_catalog: bool = False,
    publication_lock_held: bool = False,
    validated_cache: ValidatedResolutionCache | None = None,
) -> Path:
    return publish_snapshot_group(
        [(staging_snapshot, dataset_root)],
        quant_data_root=quant_data_root,
        update_root_catalog=update_root_catalog,
        publication_lock_held=publication_lock_held,
        validated_cache=validated_cache,
    )[0]


def _catalog_entry_from_validated_manifest(
    snapshot: Path,
    *,
    quant_data_root: Path,
    dataset_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    quant_data_root = Path(quant_data_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    snapshot = Path(snapshot).resolve()
    if manifest["data_kind"] == "qlib_provider":
        entry = {
            "dataset_id": manifest["dataset_id"],
            "data_kind": manifest["data_kind"],
            "role": manifest["role"],
            "status": manifest["status"],
            "source": manifest["source"],
            "price_basis": manifest["price_basis"],
            "latest_trade_date": manifest["latest_trade_date"],
            "date_range": manifest["date_range"],
            "instrument_count": manifest["instrument_count"],
            "feature_file_count": manifest["feature_file_count"],
            "source_retrieved_at": manifest["source_retrieved_at"],
            "adjustment": manifest.get("adjustment", {}),
            "content_sha256": manifest["content_sha256"],
            "producer": manifest["producer"],
            "upstream_snapshots": manifest["upstream_snapshots"],
            "dependency_snapshots": manifest.get("dependency_snapshots", []),
            "provider_manifest_path": manifest["provider_manifest_path"],
            "file_index_path": manifest["file_index_path"],
            "dataset_root": _safe_relative(
                dataset_root, quant_data_root, "dataset_root"
            ),
            "latest_pointer": _safe_relative(
                dataset_root / "LATEST.json", quant_data_root, "latest_pointer"
            ),
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_manifest_sha256": sha256_file(
                snapshot / "DATASET_MANIFEST.json"
            ),
        }
        if "benchmark_complete" in manifest:
            entry["benchmark_complete"] = manifest["benchmark_complete"]
            entry["benchmarks"] = manifest["benchmarks"]
        if "instrument_pools" in manifest:
            entry["instrument_pools"] = manifest["instrument_pools"]
        if "transformation" in manifest:
            entry["transformation"] = manifest["transformation"]
        return entry
    return {
        "dataset_id": manifest["dataset_id"],
        "data_kind": manifest["data_kind"],
        "role": manifest["role"],
        "status": manifest["status"],
        "source": manifest["source"],
        "price_basis": manifest["price_basis"],
        "latest_trade_date": manifest["latest_trade_date"],
        "date_range": manifest["date_range"],
        "row_count": manifest.get("row_count"),
        "symbol_count": manifest.get("symbol_count"),
        "source_retrieved_at": manifest["source_retrieved_at"],
        "adjustment": manifest["adjustment"],
        "file_index_path": manifest.get("file_index_path"),
        "factor_index_path": manifest.get("factor_index_path"),
        "corporate_actions_path": manifest.get("corporate_actions_path"),
        "corporate_actions_sha256": manifest.get("corporate_actions_sha256"),
        "content_sha256": manifest.get("content_sha256"),
        "producer": manifest.get("producer"),
        "upstream_snapshots": manifest.get("upstream_snapshots", []),
        "dataset_root": _safe_relative(dataset_root, quant_data_root, "dataset_root"),
        "latest_pointer": _safe_relative(
            dataset_root / "LATEST.json", quant_data_root, "latest_pointer"
        ),
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_manifest_sha256": sha256_file(
            snapshot / "DATASET_MANIFEST.json"
        ),
    }


def catalog_entry_from_snapshot(
    snapshot: Path,
    *,
    quant_data_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    quant_data_root = Path(quant_data_root).resolve()
    snapshot = Path(snapshot).resolve()
    manifest = validate_dataset_snapshot(
        snapshot,
        verify_hashes=True,
        evidence_root=quant_data_root,
    )
    return _catalog_entry_from_validated_manifest(
        snapshot,
        quant_data_root=quant_data_root,
        dataset_root=dataset_root,
        manifest=manifest,
    )


def write_root_manifest(
    quant_data_root: Path,
    datasets: Mapping[str, Mapping[str, Any]],
    *,
    existing: Mapping[str, Any] | None = None,
) -> Path:
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    payload = dict(existing or {})
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "path_format": "relative_to_quant_data_root",
            "datasets": {key: dict(value) for key, value in sorted(datasets.items())},
        }
    )
    path = quant_data_root / "DATA_MANIFEST.json"
    atomic_write_json(path, payload)
    return path


def _load_root_manifest(quant_data_root: Path) -> dict[str, Any]:
    manifest = _read_json(Path(quant_data_root) / "DATA_MANIFEST.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("DATA_MANIFEST.json schema version 2 is required")
    if manifest.get("path_format") != "relative_to_quant_data_root":
        raise DataContractError("DATA_MANIFEST.json relative path format is required")
    if not isinstance(manifest.get("datasets"), dict):
        raise DataContractError("DATA_MANIFEST.json datasets catalog is missing")
    return manifest


def _validate_current_upstream(
    catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Reject a derived provider whose frozen input is no longer current."""

    if manifest.get("data_kind") != "qlib_provider":
        return
    upstream_snapshots = manifest.get("upstream_snapshots")
    if not isinstance(upstream_snapshots, list) or len(upstream_snapshots) != 1:
        raise DataContractError("dataset upstream snapshot contract is invalid")
    dependencies = manifest.get("dependency_snapshots", [])
    if not isinstance(dependencies, list):
        raise DataContractError("dataset dependency snapshot contract is invalid")
    identities = [*upstream_snapshots, *dependencies]
    dataset_ids: set[str] = set()
    for upstream in identities:
        if not isinstance(upstream, Mapping):
            raise DataContractError("dataset upstream snapshot contract is invalid")
        upstream_dataset_id = str(upstream.get("dataset_id") or "")
        if not upstream_dataset_id or upstream_dataset_id in dataset_ids:
            raise DataContractError("dataset upstream snapshot contract is invalid")
        dataset_ids.add(upstream_dataset_id)
        current = catalog.get("datasets", {}).get(upstream_dataset_id)
        if not isinstance(current, Mapping) or current.get("status") != "ready":
            raise DataContractError("dataset upstream snapshot is stale")
        for field in ("snapshot_id", "content_sha256", "price_basis"):
            if not upstream.get(field) or upstream.get(field) != current.get(field):
                raise DataContractError("dataset upstream snapshot is stale")


def resolve_dataset(
    dataset_id: str,
    *,
    quant_data_root: Path,
    require_basis: str | None = None,
    require_status: str = "ready",
    validated_cache: ValidatedResolutionCache | None = None,
) -> ResolvedDataset:
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    catalog = _load_root_manifest(quant_data_root)
    entry = catalog["datasets"].get(dataset_id)
    if not isinstance(entry, dict):
        raise DataContractError(f"dataset is not registered: {dataset_id}")
    if entry.get("status") != require_status:
        raise DataContractError(
            f"dataset status is {entry.get('status')}, required {require_status}"
        )
    if require_basis is not None and entry.get("price_basis") != require_basis:
        raise DataContractError(
            f"dataset price basis is {entry.get('price_basis')}, required {require_basis}"
        )
    pointer_path = _safe_child(
        quant_data_root, entry.get("latest_pointer"), "latest_pointer"
    )
    pointer, snapshot = validate_latest_pointer(
        pointer_path,
        quant_data_root=quant_data_root,
        expected_dataset_id=dataset_id,
    )
    if entry.get("snapshot_id") != pointer.get("snapshot_id"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    if entry.get("snapshot_manifest_sha256") != pointer.get("manifest_sha256"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    cached = validated_cache.get(dataset_id) if validated_cache is not None else None
    if (
        cached is not None
        and cached.dataset_id == dataset_id
        and cached.snapshot_path.resolve() == snapshot.resolve()
        and cached.catalog_entry == entry
    ):
        current_manifest = _read_json(snapshot / "DATASET_MANIFEST.json")
        if current_manifest == cached.manifest:
            _validate_current_upstream(catalog, current_manifest)
            return cached
        validated_cache.pop(dataset_id, None)
    manifest = validate_dataset_snapshot(
        snapshot,
        verify_hashes=True,
        evidence_root=quant_data_root,
    )
    if manifest.get("dataset_id") != dataset_id:
        raise DataContractError("latest pointer belongs to another dataset")
    if manifest.get("snapshot_id") != pointer.get("snapshot_id"):
        raise DataContractError("latest pointer snapshot_id mismatch")
    if manifest.get("status") != require_status:
        raise DataContractError(
            f"snapshot status is {manifest.get('status')}, required {require_status}"
        )
    if require_basis is not None and manifest.get("price_basis") != require_basis:
        raise DataContractError(
            f"snapshot price basis is {manifest.get('price_basis')}, required {require_basis}"
        )
    for field in ("dataset_id", "role", "status", "price_basis", "content_sha256"):
        if entry.get(field) != manifest.get(field):
            raise DataContractError(
                f"dataset catalog and snapshot disagree on {field}"
            )
    if manifest.get("data_kind") == "qlib_provider":
        for field in ("upstream_snapshots", "dependency_snapshots"):
            if entry.get(field, []) != manifest.get(field, []):
                raise DataContractError(
                    f"dataset catalog and snapshot disagree on {field}"
                )
    _validate_current_upstream(catalog, manifest)
    resolved = ResolvedDataset(dataset_id, snapshot, manifest, dict(entry))
    if validated_cache is not None:
        validated_cache[dataset_id] = resolved
    return resolved


def describe_dataset(dataset_id: str, *, quant_data_root: Path) -> dict[str, Any]:
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    catalog = _load_root_manifest(quant_data_root)
    entry = catalog["datasets"].get(dataset_id)
    if not isinstance(entry, dict):
        raise DataContractError(f"dataset is not registered: {dataset_id}")
    if not entry.get("latest_pointer"):
        return {**entry, "snapshot_available": False}
    resolved = resolve_dataset(dataset_id, quant_data_root=quant_data_root)
    return {
        **resolved.catalog_entry,
        "snapshot_available": True,
        "snapshot_path": _safe_relative(
            resolved.snapshot_path, quant_data_root, "snapshot_path"
        ),
        "snapshot_manifest": resolved.manifest,
    }


def copy_snapshot(source: Path, destination: Path) -> None:
    """Copy helper for producers that need to seed staging from an old snapshot."""
    if Path(destination).exists():
        raise DataContractError("staging destination already exists")
    shutil.copytree(source, destination)
