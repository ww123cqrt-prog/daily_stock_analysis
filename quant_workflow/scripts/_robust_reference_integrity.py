"""Fail-closed integrity checks for locally published robust reference data."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator


REFERENCE_SOURCE_FILENAME = "REFERENCE_SOURCE.json"
REFERENCE_TRANSACTION_FILENAME = ".download_robust_reference.transaction.json"
ALLOWED_REQUESTED_SOURCES = {"auto", "jqdata", "baostock"}
ALLOWED_SELECTED_SOURCES = {"jqdata", "baostock"}
SAFE_REASON = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
KNOWN_AUTO_FALLBACK_REASONS = {
    "jqdata_auth_failed",
    "jqdata_sdk_unavailable",
    "jqdata_permission_denied",
    "jqdata_quota_exceeded",
}
REFERENCE_SNAPSHOT_FILES = (
    REFERENCE_SOURCE_FILENAME,
    "security_master.parquet",
    "industry_intervals.parquet",
    "st_status.parquet",
    "suspension_status.parquet",
)
_LOCK_TOKEN = object()


class ReferenceIntegrityError(ValueError):
    """A stable, sanitized reference integrity failure."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _iso_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError
    return date.fromisoformat(value)


class LockedReferenceSnapshot:
    """Reference operations bound to one active shared-lock lifetime."""

    def __init__(self, reference_dir: Path, token: object) -> None:
        if token is not _LOCK_TOKEN:
            raise RuntimeError("reference snapshots require reference_read_lock")
        self.reference_dir = Path(reference_dir)
        self._active = True

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("reference read lock is no longer active")

    def ensure_transaction_clean(self) -> None:
        self._require_active()
        journal = self.reference_dir / REFERENCE_TRANSACTION_FILENAME
        if journal.exists():
            raise ReferenceIntegrityError(
                "reference_transaction_incomplete",
                "reference transaction journal is present; reference parquet files may be mixed",
            )

    def validate_source(
        self,
        *,
        start_date: str,
        end_date: str,
        expected_membership_snapshot_hash: str | None = None,
    ) -> dict[str, Any]:
        self._require_active()
        path = self.reference_dir / REFERENCE_SOURCE_FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReferenceIntegrityError(
                "reference_source_missing", "reference source metadata is missing"
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReferenceIntegrityError(
                "reference_source_invalid", "reference source metadata is invalid"
            ) from exc

        if not isinstance(raw, dict):
            raise ReferenceIntegrityError(
                "reference_source_invalid",
                "reference source metadata must be an object",
            )
        requested = raw.get("requested")
        selected = raw.get("selected")
        industry = raw.get("industry_classification")
        fallback_reason = raw.get("fallback_reason")
        membership_snapshot_hash = raw.get("membership_snapshot_hash")
        fields_valid = (
            requested in ALLOWED_REQUESTED_SOURCES
            and selected in ALLOWED_SELECTED_SOURCES
            and isinstance(industry, str)
            and bool(industry.strip())
            and isinstance(membership_snapshot_hash, str)
            and SHA256_HEX.fullmatch(membership_snapshot_hash) is not None
            and (
                fallback_reason is None
                or (
                    isinstance(fallback_reason, str)
                    and SAFE_REASON.fullmatch(fallback_reason) is not None
                )
            )
        )
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
                and fallback_reason in KNOWN_AUTO_FALLBACK_REASONS
            )
        )
        if not fields_valid or not combination_valid:
            raise ReferenceIntegrityError(
                "reference_source_invalid",
                "reference source metadata contains invalid fields",
            )
        if (
            expected_membership_snapshot_hash is not None
            and membership_snapshot_hash != expected_membership_snapshot_hash
        ):
            raise ReferenceIntegrityError(
                "reference_membership_snapshot_mismatch",
                "reference data was built from a different membership snapshot",
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
            raise ReferenceIntegrityError(
                "reference_source_invalid", "reference source date range is invalid"
            ) from exc
        if source_start > source_end:
            raise ReferenceIntegrityError(
                "reference_source_invalid", "reference source date range is invalid"
            )
        if source_start > requested_start or source_end < requested_end:
            raise ReferenceIntegrityError(
                "reference_source_range_incomplete",
                "reference source range does not cover the requested period",
            )

        sanitized: dict[str, Any] = {
            "requested": requested,
            "selected": selected,
            "date_range": {
                "start_date": source_start.isoformat(),
                "end_date": source_end.isoformat(),
            },
            "industry_classification": industry.strip(),
            "membership_snapshot_hash": membership_snapshot_hash,
        }
        if fallback_reason is not None:
            sanitized["fallback_reason"] = fallback_reason
        return sanitized

    def fingerprint(self) -> str:
        self._require_active()
        self.ensure_transaction_clean()
        digest = hashlib.sha256()
        try:
            for file_name in REFERENCE_SNAPSHOT_FILES:
                encoded_name = file_name.encode("utf-8")
                digest.update(len(encoded_name).to_bytes(4, "big"))
                digest.update(encoded_name)
                path = self.reference_dir / file_name
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags)
                file_stat = os.fstat(descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    os.close(descriptor)
                    raise OSError
                digest.update(file_stat.st_size.to_bytes(8, "big"))
                with os.fdopen(descriptor, "rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
        except OSError as exc:
            raise ReferenceIntegrityError(
                "reference_snapshot_invalid", "reference snapshot files are incomplete"
            ) from exc
        return digest.hexdigest()


@contextmanager
def reference_read_lock(reference_dir: Path) -> Iterator[LockedReferenceSnapshot]:
    reference_dir = Path(reference_dir)
    reference_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = (
        reference_dir.parent / f".{reference_dir.name}.download_robust_reference.lock"
    )
    handle = lock_path.open("a+b")
    snapshot = LockedReferenceSnapshot(reference_dir, _LOCK_TOKEN)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
    except Exception:
        snapshot._active = False
        handle.close()
        raise
    try:
        yield snapshot
    finally:
        snapshot._active = False
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def ensure_reference_transaction_clean(reference_dir: Path) -> None:
    with reference_read_lock(reference_dir) as snapshot:
        snapshot.ensure_transaction_clean()


def validate_reference_source(
    reference_dir: Path,
    *,
    start_date: str,
    end_date: str,
    expected_membership_snapshot_hash: str | None = None,
) -> dict[str, Any]:
    with reference_read_lock(reference_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        return snapshot.validate_source(
            start_date=start_date,
            end_date=end_date,
            expected_membership_snapshot_hash=expected_membership_snapshot_hash,
        )


def reference_snapshot_fingerprint(reference_dir: Path) -> str:
    with reference_read_lock(reference_dir) as snapshot:
        return snapshot.fingerprint()
