#!/usr/bin/env python3
"""Import a verified remote price bundle into an immutable local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_robust_price_bundle as validator  # noqa: E402
from _paths import quant_data_root, reports_root  # noqa: E402
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)


DEFAULT_PIT_DIR = quant_data_root() / "reference" / "pit_csi800"
DEFAULT_DESTINATION_ROOT = (
    quant_data_root() / "runtime" / "robust_strategy" / "price_bundles"
)
DEFAULT_STATUS_DIR = reports_root() / "robust_strategy"
IMPORT_VERSION = 1
MANIFEST_FILENAME = "PRICE_BUNDLE_MANIFEST.json"
IMPORT_FILENAME = "IMPORT.json"


class PriceBundleImportError(ValueError):
    def __init__(self, reason_code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_symbol(value: Any) -> str:
    raw = str(value).strip().upper()
    if len(raw) >= 6 and raw[-6:].isdigit():
        return raw[-6:]
    return raw


def _normalize_membership_symbols(values: Iterable[Any]) -> set[str]:
    symbols = {_normalize_symbol(value) for value in values}
    if not symbols or any(not value.isdigit() or len(value) != 6 for value in symbols):
        raise PriceBundleImportError(
            "pit_symbol_set_invalid", "PIT membership contains invalid symbols"
        )
    return symbols


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PriceBundleImportError(
            "source_manifest_invalid", "source price manifest is missing or unsafe"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriceBundleImportError(
            "source_manifest_invalid", "source price manifest is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise PriceBundleImportError(
            "source_manifest_invalid", "source price manifest must be an object"
        )
    return value


def _manifest_symbols(manifest: dict[str, Any]) -> set[str]:
    files = manifest.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        raise PriceBundleImportError(
            "source_manifest_invalid", "source manifest files are invalid"
        )
    symbols = {_normalize_symbol(item.get("symbol")) for item in files}
    if len(symbols) != len(files):
        raise PriceBundleImportError(
            "source_manifest_invalid", "source manifest contains duplicate symbols"
        )
    return _normalize_membership_symbols(symbols)


def _bundle_key(
    *, source_manifest_sha256: str, membership_snapshot_hash: str
) -> str:
    payload = {
        "import_version": IMPORT_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "membership_snapshot_hash": membership_snapshot_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise PriceBundleImportError(
                "symlink_forbidden", "imported price bundle contains a symlink"
            )
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(path, mode)
    os.chmod(
        root,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )


def _make_tree_writable(root: Path) -> None:
    if root.is_symlink() or not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        try:
            os.chmod(
                path,
                path.stat().st_mode
                | stat.S_IWUSR
                | (stat.S_IXUSR if path.is_dir() else 0),
            )
        except FileNotFoundError:
            continue


def _remove_tree(root: Path) -> None:
    if root.is_symlink():
        root.unlink()
        return
    if root.exists():
        _make_tree_writable(root)
        shutil.rmtree(root)


def _tree_is_read_only(root: Path) -> bool:
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        return not any(
            path.is_symlink() or path.stat().st_mode & write_mask
            for path in [root, *root.rglob("*")]
        )
    except OSError:
        return False


def _verify_imported_bundle(
    *,
    bundle_dir: Path,
    bundle_key: str,
    source_manifest_sha256: str,
    membership_snapshot_hash: str,
) -> dict[str, Any] | None:
    price_dir = bundle_dir / "prices"
    manifest_path = bundle_dir / MANIFEST_FILENAME
    import_path = bundle_dir / IMPORT_FILENAME
    if (
        bundle_dir.is_symlink()
        or not price_dir.is_dir()
        or not manifest_path.is_file()
        or not import_path.is_file()
    ):
        return None
    try:
        metadata = json.loads(import_path.read_text(encoding="utf-8"))
        if (
            metadata.get("import_version") != IMPORT_VERSION
            or metadata.get("bundle_key") != bundle_key
            or metadata.get("source_manifest_sha256") != source_manifest_sha256
            or metadata.get("membership_snapshot_hash") != membership_snapshot_hash
        ):
            return None
        verified = validator.verify_bundle_manifest(
            price_dir=price_dir,
            manifest_path=manifest_path,
            expected_source="baostock",
            expected_adjustment="qfq",
            expected_membership_snapshot_hash=membership_snapshot_hash,
        )
    except (OSError, json.JSONDecodeError, validator.PriceBundleError):
        return None
    if verified["manifest_sha256"] != source_manifest_sha256:
        return None
    if not _tree_is_read_only(bundle_dir):
        return None
    return verified


def import_price_bundle(
    *,
    source_dir: Path,
    source_manifest: Path,
    destination_root: Path,
    membership_symbols: Iterable[Any],
    membership_snapshot_hash: str,
) -> dict[str, Any]:
    source_dir = Path(source_dir).expanduser().resolve()
    source_manifest = Path(source_manifest).expanduser().resolve()
    destination_root = Path(destination_root).expanduser().resolve()
    if destination_root == source_dir or source_dir in destination_root.parents:
        raise PriceBundleImportError(
            "destination_inside_source",
            "destination root must not be inside the source price directory",
        )
    membership_snapshot_hash = str(membership_snapshot_hash).strip().lower()
    membership_set = _normalize_membership_symbols(membership_symbols)
    manifest = _read_manifest(source_manifest)
    try:
        source_verified = validator.verify_bundle_manifest(
            price_dir=source_dir,
            manifest_path=source_manifest,
            expected_source="baostock",
            expected_adjustment="qfq",
            expected_membership_snapshot_hash=membership_snapshot_hash,
        )
    except validator.PriceBundleError as exc:
        raise PriceBundleImportError(
            exc.reason_code, str(exc), **exc.details
        ) from exc
    manifest_set = _manifest_symbols(manifest)
    if manifest_set != membership_set:
        raise PriceBundleImportError(
            "pit_symbol_set_mismatch",
            "price bundle symbols do not match the local PIT snapshot",
            missing_symbols=sorted(membership_set - manifest_set),
            unexpected_symbols=sorted(manifest_set - membership_set),
        )

    source_manifest_sha256 = source_verified["manifest_sha256"]
    key = _bundle_key(
        source_manifest_sha256=source_manifest_sha256,
        membership_snapshot_hash=membership_snapshot_hash,
    )
    bundle_dir = destination_root / key[:16]
    existing = _verify_imported_bundle(
        bundle_dir=bundle_dir,
        bundle_key=key,
        source_manifest_sha256=source_manifest_sha256,
        membership_snapshot_hash=membership_snapshot_hash,
    )
    reused = existing is not None
    if not reused:
        if bundle_dir.exists():
            raise PriceBundleImportError(
                "destination_bundle_invalid",
                "existing destination bundle failed integrity validation",
            )
        destination_root.mkdir(parents=True, exist_ok=True)
        temporary = destination_root / f".{key[:16]}.{os.getpid()}.tmp"
        if temporary.exists():
            _remove_tree(temporary)
        temporary.mkdir()
        try:
            shutil.copytree(source_dir, temporary / "prices", copy_function=shutil.copy2)
            shutil.copy2(source_manifest, temporary / MANIFEST_FILENAME)
            copied_verified = validator.verify_bundle_manifest(
                price_dir=temporary / "prices",
                manifest_path=temporary / MANIFEST_FILENAME,
                expected_source="baostock",
                expected_adjustment="qfq",
                expected_membership_snapshot_hash=membership_snapshot_hash,
            )
            if copied_verified != source_verified:
                raise PriceBundleImportError(
                    "copied_bundle_mismatch",
                    "copied bundle does not match the verified source",
                )
            metadata = {
                "import_version": IMPORT_VERSION,
                "bundle_key": key,
                "source_manifest_sha256": source_manifest_sha256,
                "membership_snapshot_hash": membership_snapshot_hash,
                "file_count": source_verified["file_count"],
                "row_count": source_verified["row_count"],
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            (temporary / IMPORT_FILENAME).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _make_tree_read_only(temporary)
            if (
                _verify_imported_bundle(
                    bundle_dir=temporary,
                    bundle_key=key,
                    source_manifest_sha256=source_manifest_sha256,
                    membership_snapshot_hash=membership_snapshot_hash,
                )
                is None
            ):
                raise PriceBundleImportError(
                    "copied_bundle_mismatch",
                    "temporary imported bundle failed publication checks",
                )
            os.replace(temporary, bundle_dir)
        finally:
            if temporary.exists():
                _remove_tree(temporary)
        existing = _verify_imported_bundle(
            bundle_dir=bundle_dir,
            bundle_key=key,
            source_manifest_sha256=source_manifest_sha256,
            membership_snapshot_hash=membership_snapshot_hash,
        )
        if existing is None:
            raise PriceBundleImportError(
                "destination_bundle_invalid",
                "published destination bundle failed integrity validation",
            )

    result = {
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bundle_key": key,
        "bundle_dir": str(bundle_dir),
        "price_dir": str(bundle_dir / "prices"),
        "manifest": str(bundle_dir / MANIFEST_FILENAME),
        "source_manifest_sha256": source_manifest_sha256,
        "membership_snapshot_hash": membership_snapshot_hash,
        "file_count": source_verified["file_count"],
        "row_count": source_verified["row_count"],
        "reused": reused,
    }
    _atomic_write(
        destination_root / "latest_import.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def _pit_symbols(pit_dir: Path) -> set[str]:
    path = pit_dir / "membership_intervals.parquet"
    try:
        frame = pd.read_parquet(path, columns=["symbol"])
    except Exception as exc:
        raise PriceBundleImportError(
            "pit_symbol_set_invalid", "cannot read PIT membership symbols"
        ) from exc
    return _normalize_membership_symbols(frame["symbol"].tolist())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a verified price bundle into immutable robust-strategy storage."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pit-dir", type=Path, default=DEFAULT_PIT_DIR)
    parser.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    parser.add_argument("--status-out-dir", type=Path, default=DEFAULT_STATUS_DIR)
    args = parser.parse_args(argv)
    try:
        manifest = _read_manifest(args.source_manifest)
        requested = manifest.get("requested_range") or {}
        start_date = str(requested.get("start_date", ""))
        end_date = str(requested.get("end_date", ""))
        with membership_read_lock(args.pit_dir) as snapshot:
            snapshot.ensure_transaction_clean()
            snapshot.validate_source(start_date=start_date, end_date=end_date)
            membership_hash = snapshot.fingerprint()
            membership_symbols = _pit_symbols(args.pit_dir)
        result = import_price_bundle(
            source_dir=args.source_dir,
            source_manifest=args.source_manifest,
            destination_root=args.destination_root,
            membership_symbols=membership_symbols,
            membership_snapshot_hash=membership_hash,
        )
        with membership_read_lock(args.pit_dir) as snapshot:
            snapshot.ensure_transaction_clean()
            if snapshot.fingerprint() != membership_hash:
                raise PriceBundleImportError(
                    "membership_snapshot_changed",
                    "PIT membership snapshot changed during price import",
                )
        status = result
        return_code = 0
    except (
        MembershipIntegrityError,
        PriceBundleImportError,
        validator.PriceBundleError,
        OSError,
        ValueError,
    ) as exc:
        status = {
            "status": "failed",
            "reason_code": getattr(exc, "reason_code", "price_bundle_import_failed"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        }
        return_code = 2
    _atomic_write(
        args.status_out_dir / "price_bundle_import_latest.json",
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
