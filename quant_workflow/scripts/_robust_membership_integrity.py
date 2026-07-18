"""Fail-closed integrity checks for PIT CSI800 membership snapshots."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file


MEMBERSHIP_SOURCE_FILENAME = "MEMBERSHIP_SOURCE.json"
MEMBERSHIP_INTERVALS_FILENAME = "membership_intervals.parquet"
MEMBERSHIP_LOCK_FILENAME = ".build_pit_csi800.lock"
MEMBERSHIP_TRANSACTION_FILENAME = ".build_pit_csi800.transaction.json"
MEMBERSHIP_BUILD_IN_PROGRESS_FILENAME = ".build_pit_csi800.in-progress.json"
EXPECTED_INDICES = {
    "csi300": "000300.XSHG",
    "csi500": "000905.XSHG",
}
KNOWN_JQDATA_FALLBACK_REASONS = {
    "jqdata_auth_failed",
    "jqdata_sdk_unavailable",
    "jqdata_permission_denied",
    "jqdata_quota_exceeded",
}
_LOCK_TOKEN = object()


class MembershipIntegrityError(ValueError):
    """A stable, sanitized membership integrity failure."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _iso_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError
    return date.fromisoformat(value)


class LockedMembershipSnapshot:
    """Membership operations bound to one active shared-lock lifetime."""

    def __init__(self, pit_dir: Path, token: object) -> None:
        if token is not _LOCK_TOKEN:
            raise RuntimeError("membership snapshots require membership_read_lock")
        self.pit_dir = Path(pit_dir)
        self._active = True

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("membership read lock is no longer active")

    def ensure_transaction_clean(self) -> None:
        self._require_active()
        if (self.pit_dir / MEMBERSHIP_BUILD_IN_PROGRESS_FILENAME).exists():
            raise MembershipIntegrityError(
                "membership_build_in_progress",
                "membership build is still in progress",
            )
        if (self.pit_dir / MEMBERSHIP_TRANSACTION_FILENAME).exists():
            raise MembershipIntegrityError(
                "membership_transaction_incomplete",
                "membership transaction journal is present",
            )

    def validate_source(self, *, start_date: str, end_date: str) -> dict[str, Any]:
        self._require_active()
        path = self.pit_dir / MEMBERSHIP_SOURCE_FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MembershipIntegrityError(
                "membership_source_missing", "membership source metadata is missing"
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise MembershipIntegrityError(
                "membership_source_invalid", "membership source metadata is invalid"
            ) from exc
        if not isinstance(raw, dict):
            raise MembershipIntegrityError(
                "membership_source_invalid",
                "membership source metadata must be an object",
            )

        requested = raw.get("requested")
        selected = raw.get("selected")
        fallback_reason = raw.get("fallback_reason")
        combination_valid = (
            (requested == "jqdata" and selected == "jqdata" and fallback_reason is None)
            or (
                requested == "baostock"
                and selected == "baostock"
                and fallback_reason is None
            )
            or (
                requested == "auto" and selected == "jqdata" and fallback_reason is None
            )
            or (
                requested == "auto"
                and selected == "baostock"
                and fallback_reason in KNOWN_JQDATA_FALLBACK_REASONS
            )
        )
        if not combination_valid or raw.get("indices") != EXPECTED_INDICES:
            raise MembershipIntegrityError(
                "membership_source_invalid",
                "membership source metadata contains invalid fields",
            )

        date_range = raw.get("date_range")
        try:
            if not isinstance(date_range, dict):
                raise ValueError
            source_start = _iso_date(date_range.get("start_date"))
            source_end = _iso_date(date_range.get("end_date"))
            requested_start = _iso_date(start_date)
            requested_end = _iso_date(end_date)
        except ValueError as exc:
            raise MembershipIntegrityError(
                "membership_source_invalid", "membership source date range is invalid"
            ) from exc
        if source_start > source_end:
            raise MembershipIntegrityError(
                "membership_source_invalid", "membership source date range is invalid"
            )
        if source_start > requested_start or source_end < requested_end:
            raise MembershipIntegrityError(
                "membership_source_range_incomplete",
                "membership source range does not cover the requested period",
            )

        sanitized: dict[str, Any] = {
            "requested": requested,
            "selected": selected,
            "date_range": {
                "start_date": source_start.isoformat(),
                "end_date": source_end.isoformat(),
            },
            "indices": dict(EXPECTED_INDICES),
        }
        if fallback_reason is not None:
            sanitized["fallback_reason"] = fallback_reason
        return sanitized

    def daily_files(self) -> list[Path]:
        self._require_active()
        daily_dir = self.pit_dir / "daily"
        files = sorted(daily_dir.glob("*.parquet"), key=lambda path: path.name)
        if not files:
            raise MembershipIntegrityError(
                "membership_snapshot_invalid", "membership daily partitions are missing"
            )
        for path in files:
            try:
                mode = path.stat().st_mode
            except OSError as exc:
                raise MembershipIntegrityError(
                    "membership_snapshot_invalid",
                    "membership snapshot files are incomplete",
                ) from exc
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise MembershipIntegrityError(
                    "membership_snapshot_invalid",
                    "membership snapshot contains an invalid file",
                )
        return files

    def fingerprint(self) -> str:
        self._require_active()
        self.ensure_transaction_clean()
        paths = [
            self.pit_dir / MEMBERSHIP_SOURCE_FILENAME,
            self.pit_dir / MEMBERSHIP_INTERVALS_FILENAME,
            *self.daily_files(),
        ]
        digest = hashlib.sha256()
        try:
            for path in paths:
                relative_name = (
                    path.relative_to(self.pit_dir).as_posix().encode("utf-8")
                )
                mode = path.stat().st_mode
                if path.is_symlink() or not stat.S_ISREG(mode):
                    raise OSError
                digest.update(len(relative_name).to_bytes(4, "big"))
                digest.update(relative_name)
                digest.update(path.stat().st_size.to_bytes(8, "big"))
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
        except OSError as exc:
            raise MembershipIntegrityError(
                "membership_snapshot_invalid",
                "membership snapshot files are incomplete",
            ) from exc
        return digest.hexdigest()


@contextmanager
def membership_read_lock(pit_dir: Path) -> Iterator[LockedMembershipSnapshot]:
    pit_dir = Path(pit_dir)
    if not pit_dir.is_dir():
        raise MembershipIntegrityError(
            "membership_snapshot_invalid", "membership snapshot directory is missing"
        )
    lock_path = pit_dir / MEMBERSHIP_LOCK_FILENAME
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        raise MembershipIntegrityError(
            "membership_snapshot_invalid", "membership lock is unavailable"
        ) from exc
    snapshot = LockedMembershipSnapshot(pit_dir, _LOCK_TOKEN)
    try:
        lock_file(handle, exclusive=False, blocking=False)
    except BlockingIOError as exc:
        snapshot._active = False
        handle.close()
        raise MembershipIntegrityError(
            "membership_writer_active", "membership writer is active"
        ) from exc
    except OSError as exc:
        snapshot._active = False
        handle.close()
        raise MembershipIntegrityError(
            "membership_snapshot_invalid", "membership lock is unavailable"
        ) from exc
    try:
        yield snapshot
    finally:
        snapshot._active = False
        try:
            unlock_file(handle)
        finally:
            handle.close()


def read_membership_snapshot(
    pit_dir: Path, *, start_date: str, end_date: str
) -> tuple[dict[str, Any], str]:
    with membership_read_lock(pit_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        metadata = snapshot.validate_source(start_date=start_date, end_date=end_date)
        return metadata, snapshot.fingerprint()
