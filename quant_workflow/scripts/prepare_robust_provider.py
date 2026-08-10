#!/usr/bin/env python3
"""Build an immutable Qlib provider view with PIT CSI800 instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:  # noqa: E402
    from ._paths import provider_calendar_days, qlib_workflow_provider_uri, quant_data_root
except ImportError:  # Direct script execution.
    from _paths import provider_calendar_days, qlib_workflow_provider_uri, quant_data_root


DEFAULT_CANONICAL_PROVIDER: Path | None = None
DEFAULT_INTERVALS = (
    quant_data_root() / "reference" / "pit_csi800" / "membership_intervals.parquet"
)
DEFAULT_RUNTIME_ROOT = quant_data_root() / "runtime" / "robust_strategy"
DEFAULT_DATA_MANIFEST = quant_data_root() / "DATA_MANIFEST.json"
REQUIRED_COLUMNS = {"symbol", "start_date", "end_date"}
PROVIDER_METADATA_FILENAME = "ROBUST_PROVIDER.json"
PROVIDER_CONTENT_FILENAME = "ROBUST_PROVIDER_CONTENT.json"
PROVIDER_FORMAT_VERSION = 3
PROVIDER_CONTENT_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_entries(
    root: Path, *, prefixes: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    root = Path(root)
    allowed = (
        None if prefixes is None else {str(value).strip("/") for value in prefixes}
    )
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if allowed is not None and relative.split("/", 1)[0] not in allowed:
            continue
        if path.is_symlink():
            raise ValueError("provider content must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("provider content contains a non-regular file")
        entries[relative] = {
            "size": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    return entries


def _content_tree_sha256(entries: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _calendar(provider: Path) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(provider_calendar_days(provider), errors="coerce"))
    values = values.dropna().normalize().unique().sort_values()
    if values.empty:
        raise ValueError("canonical provider calendar is empty")
    return values


def normalize_union_intervals(
    intervals: pd.DataFrame, calendar: Sequence[Any]
) -> pd.DataFrame:
    if not REQUIRED_COLUMNS.issubset(intervals.columns):
        missing = sorted(REQUIRED_COLUMNS - set(intervals.columns))
        raise ValueError(
            f"membership intervals are missing columns: {', '.join(missing)}"
        )
    days = (
        pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )
    if days.empty:
        raise ValueError("calendar is empty")
    positions = {day: index for index, day in enumerate(days)}
    frame = intervals.loc[:, ["symbol", "start_date", "end_date"]].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["start_date"] = pd.to_datetime(
        frame["start_date"], errors="coerce"
    ).dt.normalize()
    frame["end_date"] = pd.to_datetime(
        frame["end_date"], errors="coerce"
    ).dt.normalize()
    if frame[["start_date", "end_date"]].isna().any(axis=None):
        raise ValueError("membership intervals contain invalid dates")
    if (frame["start_date"] > frame["end_date"]).any():
        raise ValueError("membership interval start_date is after end_date")
    outside = frame.loc[
        ~frame["start_date"].isin(days) | ~frame["end_date"].isin(days),
        ["symbol", "start_date", "end_date"],
    ]
    if not outside.empty:
        raise ValueError(
            "membership intervals extend outside the canonical trading calendar"
        )

    merged_rows: list[dict[str, Any]] = []
    for symbol, group in frame.sort_values(
        ["symbol", "start_date", "end_date"]
    ).groupby("symbol", sort=True):
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            start = pd.Timestamp(row.start_date)
            end = pd.Timestamp(row.end_date)
            if current_start is None:
                current_start, current_end = start, end
                continue
            assert current_end is not None
            if positions[start] <= positions[current_end]:
                raise ValueError(f"membership intervals overlap for {symbol}")
            if positions[start] == positions[current_end] + 1:
                current_end = end
                continue
            merged_rows.append(
                {"symbol": symbol, "start_date": current_start, "end_date": current_end}
            )
            current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            merged_rows.append(
                {"symbol": symbol, "start_date": current_start, "end_date": current_end}
            )
    return pd.DataFrame(merged_rows, columns=["symbol", "start_date", "end_date"])


def _provider_key(
    canonical_provider: Path, intervals_path: Path, data_manifest_path: Path
) -> str:
    calendar_hash = hashlib.sha256(
        "\n".join(provider_calendar_days(canonical_provider)).encode("utf-8")
    ).hexdigest()
    payload = {
        "canonical_provider": str(canonical_provider.resolve()),
        "calendar_sha256": calendar_hash,
        "intervals_sha256": file_sha256(intervals_path),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "canonical_tree_sha256": _content_tree_sha256(
            _content_entries(canonical_provider, prefixes=("calendars", "features"))
        ),
        "format_version": PROVIDER_FORMAT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _write_instruments(path: Path, intervals: pd.DataFrame) -> None:
    lines = [
        f"{row.symbol}\t{pd.Timestamp(row.start_date).date()}\t{pd.Timestamp(row.end_date).date()}"
        for row in intervals.sort_values(["symbol", "start_date"]).itertuples(
            index=False
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _snapshot_tree(source: Path, destination: Path) -> str:
    """Create an independent tree, using APFS copy-on-write clones when available."""
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["cp", "-cR", str(source), str(destination)],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            return "apfs_clone"
        if destination.exists():
            shutil.rmtree(destination)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return "full_copy"


def _assert_no_symlinks(root: Path, label: str) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"{label} must not contain symlinks")


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ValueError("provider snapshot must not contain symlinks")
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
            mode = path.stat().st_mode
            os.chmod(path, mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
        except FileNotFoundError:
            continue


def _remove_tree(root: Path) -> None:
    if root.is_symlink():
        root.unlink()
        return
    if not root.exists():
        return
    _make_tree_writable(root)
    shutil.rmtree(root)


def _snapshot_is_reusable(
    provider: Path,
    key: str,
    *,
    file_hasher: Any = None,
) -> bool:
    def hash_file(path: Path, info: os.stat_result) -> str:
        if file_hasher is None:
            return file_sha256(path)
        return file_hasher(path, info)

    metadata_path = provider / PROVIDER_METADATA_FILENAME
    content_path = provider / PROVIDER_CONTENT_FILENAME
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        provider_info = provider.lstat()
        if (
            stat.S_ISLNK(provider_info.st_mode)
            or not stat.S_ISDIR(provider_info.st_mode)
            or provider_info.st_mode & write_mask
        ):
            return False
        file_inventory: dict[str, os.stat_result] = {}
        for path in provider.rglob("*"):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or info.st_mode & write_mask:
                return False
            if stat.S_ISREG(info.st_mode):
                file_inventory[path.relative_to(provider).as_posix()] = info
            elif not stat.S_ISDIR(info.st_mode):
                return False
    except OSError:
        return False
    if not {
        PROVIDER_METADATA_FILENAME,
        PROVIDER_CONTENT_FILENAME,
        "instruments/csi800.txt",
    }.issubset(file_inventory):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("format_version") != PROVIDER_FORMAT_VERSION
        or metadata.get("provider_key") != key
        or metadata.get("content_manifest_sha256")
        != hash_file(
            content_path,
            file_inventory[PROVIDER_CONTENT_FILENAME],
        )
        or not isinstance(content, dict)
        or content.get("version") != PROVIDER_CONTENT_VERSION
        or content.get("provider_key") != key
        or not isinstance(content.get("files"), dict)
    ):
        return False
    files = content["files"]
    if content.get("tree_sha256") != _content_tree_sha256(files):
        return False
    expected_paths = set(files) | {
        PROVIDER_METADATA_FILENAME,
        PROVIDER_CONTENT_FILENAME,
    }
    if set(file_inventory) != expected_paths:
        return False
    try:
        for relative, expected in files.items():
            path = provider / relative
            info = file_inventory.get(relative)
            if info is None:
                return False
            if int(expected.get("size", -1)) != info.st_size:
                return False
            if expected.get("sha256") != hash_file(path, info):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


class RuntimeProviderIdentityCache:
    """Reuse content hashes only while each immutable file identity is unchanged."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._provider_identity: tuple[str, str] | None = None
        self._digests: dict[str, tuple[tuple[int, ...], str]] = {}

    @staticmethod
    def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
        return (
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
            int(stat.S_IMODE(info.st_mode)),
        )

    def _cached_sha256(self, path: Path, info: os.stat_result) -> str:
        cache_key = os.path.abspath(os.fspath(path))
        before = self._fingerprint(info)
        cached = self._digests.get(cache_key)
        if cached is not None and cached[0] == before:
            return cached[1]
        digest = file_sha256(path)
        after = self._fingerprint(Path(path).lstat())
        if after != before:
            raise OSError("provider file changed during identity verification")
        self._digests[cache_key] = (after, digest)
        return digest

    def __call__(self, provider: Path, key: str) -> dict[str, str] | None:
        provider = Path(provider)
        descriptor = _provider_descriptor_path(provider)
        if descriptor is not None:
            digest = file_sha256(descriptor)
            if str(key) != digest or _read_provider_context(provider) is None:
                return None
            return {"provider_key": digest, "provider_tree_sha256": digest}
        identity = (os.path.abspath(os.fspath(provider)), str(key))
        with self._lock:
            if self._provider_identity != identity:
                self._provider_identity = identity
                self._digests.clear()
            if not _snapshot_is_reusable(
                provider,
                key,
                file_hasher=self._cached_sha256,
            ):
                return None
            try:
                metadata = json.loads(
                    (provider / PROVIDER_METADATA_FILENAME).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return None
            tree_hash = metadata.get("provider_tree_sha256")
            if not isinstance(tree_hash, str) or len(tree_hash) != 64:
                return None
            return {"provider_key": key, "provider_tree_sha256": tree_hash}


def _provider_descriptor_path(provider: Path) -> Path | None:
    try:
        for child in Path(provider).iterdir():
            if child.name == "PROVIDER_DESCRIPTOR.json":
                return child
    except OSError:
        return None
    return None


def _read_provider_context(provider: Path) -> Any | None:
    descriptor = _provider_descriptor_path(provider)
    if descriptor is None or descriptor.is_symlink():
        return None
    try:
        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        root = Path(provider)
        for parent in [root, *root.parents]:
            if parent.name == "quant_data":
                root = parent
                break
        return _load_descriptor(descriptor, root)
    except Exception:
        return None


def runtime_provider_identity(provider: Path, key: str) -> dict[str, str] | None:
    provider = Path(provider)
    descriptor = _provider_descriptor_path(provider)
    if descriptor is not None:
        digest = file_sha256(descriptor)
        if str(key) != digest:
            return None
        if _read_provider_context(provider) is None:
            return None
        return {"provider_key": digest, "provider_tree_sha256": digest}
    if not _snapshot_is_reusable(provider, key):
        return None
    try:
        metadata = json.loads(
            (provider / PROVIDER_METADATA_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    tree_hash = metadata.get("provider_tree_sha256")
    if not isinstance(tree_hash, str) or len(tree_hash) != 64:
        return None
    return {"provider_key": key, "provider_tree_sha256": tree_hash}


def prepare_runtime_provider(
    *,
    canonical_provider: Path,
    intervals_path: Path,
    runtime_root: Path,
    data_manifest_path: Path,
) -> dict[str, Any]:
    canonical_provider = Path(canonical_provider).expanduser().resolve()
    intervals_path = Path(intervals_path).expanduser().resolve()
    runtime_root = Path(runtime_root).expanduser().resolve()
    data_manifest_path = Path(data_manifest_path).expanduser().resolve()
    descriptor = _provider_descriptor_path(canonical_provider)
    if descriptor is not None:
        if not intervals_path.is_file():
            raise FileNotFoundError(f"membership intervals are missing: {intervals_path}")
        if not data_manifest_path.is_file():
            raise FileNotFoundError(f"data manifest is missing: {data_manifest_path}")
        context = _read_provider_context(canonical_provider)
        if context is None:
            raise ValueError("Qlib Provider descriptor is unavailable or invalid")
        days = pd.DatetimeIndex(context.calendar).normalize()
        raw_intervals = pd.read_parquet(intervals_path)
        intervals = normalize_union_intervals(raw_intervals, days)
        if intervals.empty:
            raise ValueError("membership intervals are empty")
        descriptor_hash = file_sha256(descriptor)
        result = {
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "provider_uri": str(canonical_provider),
            "canonical_provider": str(canonical_provider),
            "membership_intervals": str(intervals_path),
            "provider_key": descriptor_hash,
            "provider_tree_sha256": descriptor_hash,
            "data_manifest": str(data_manifest_path),
            "data_manifest_sha256": file_sha256(data_manifest_path),
            "reused": True,
            "provider_backend": "quant_data",
            "snapshot_id": str(
                (context.descriptor.get("dataset") or {}).get("snapshot_id") or ""
            ),
            "cleaning_mode": str(
                (context.descriptor.get("cleaning") or {}).get("mode") or "canonical"
            ),
            "symbols": int(intervals["symbol"].nunique()),
            "intervals": int(len(intervals)),
        }
        _atomic_write(
            runtime_root / "latest_provider.json",
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return result
    if not (canonical_provider / "features").is_dir():
        raise FileNotFoundError(
            f"canonical provider features are missing: {canonical_provider / 'features'}"
        )
    if not intervals_path.is_file():
        raise FileNotFoundError(f"membership intervals are missing: {intervals_path}")
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"data manifest is missing: {data_manifest_path}")
    _assert_no_symlinks(
        canonical_provider / "calendars", "canonical provider calendars"
    )
    _assert_no_symlinks(canonical_provider / "features", "canonical provider features")
    days = _calendar(canonical_provider)
    raw_intervals = pd.read_parquet(intervals_path)
    intervals = normalize_union_intervals(raw_intervals, days)
    if intervals.empty:
        raise ValueError("membership intervals are empty")

    key = _provider_key(canonical_provider, intervals_path, data_manifest_path)
    providers_root = runtime_root / "providers"
    provider = providers_root / key[:16]
    reused = _snapshot_is_reusable(provider, key)
    if not reused:
        if provider.exists():
            _remove_tree(provider)
        providers_root.mkdir(parents=True, exist_ok=True)
        temporary = providers_root / f".{key[:16]}.{os.getpid()}.tmp"
        if temporary.exists():
            _remove_tree(temporary)
        temporary.mkdir()
        try:
            calendar_snapshot_method = _snapshot_tree(
                canonical_provider / "calendars", temporary / "calendars"
            )
            feature_snapshot_method = _snapshot_tree(
                canonical_provider / "features", temporary / "features"
            )
            _write_instruments(temporary / "instruments" / "csi800.txt", intervals)
            content_entries = _content_entries(temporary)
            provider_tree_sha256 = _content_tree_sha256(content_entries)
            content_payload = {
                "version": PROVIDER_CONTENT_VERSION,
                "provider_key": key,
                "tree_sha256": provider_tree_sha256,
                "files": content_entries,
            }
            (temporary / PROVIDER_CONTENT_FILENAME).write_text(
                json.dumps(
                    content_payload, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = {
                "format_version": PROVIDER_FORMAT_VERSION,
                "provider_key": key,
                "provider_tree_sha256": provider_tree_sha256,
                "content_manifest_sha256": file_sha256(
                    temporary / PROVIDER_CONTENT_FILENAME
                ),
                "canonical_provider": str(canonical_provider),
                "canonical_calendar_sha256": file_sha256(
                    canonical_provider / "calendars" / "day.txt"
                ),
                "membership_intervals": str(intervals_path),
                "membership_intervals_sha256": file_sha256(intervals_path),
                "data_manifest": str(data_manifest_path),
                "data_manifest_sha256": file_sha256(data_manifest_path),
                "snapshot_method": {
                    "calendars": calendar_snapshot_method,
                    "features": feature_snapshot_method,
                },
                "symbols": int(intervals["symbol"].nunique()),
                "intervals": int(len(intervals)),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            (temporary / PROVIDER_METADATA_FILENAME).write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            _make_tree_read_only(temporary)
            try:
                os.replace(temporary, provider)
            except OSError:
                if not _snapshot_is_reusable(provider, key):
                    raise
                reused = True
        finally:
            if temporary.exists():
                _remove_tree(temporary)

    result = {
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider_uri": str(provider.resolve()),
        "canonical_provider": str(canonical_provider),
        "membership_intervals": str(intervals_path),
        "provider_key": key,
        "provider_tree_sha256": json.loads(
            (provider / PROVIDER_METADATA_FILENAME).read_text(encoding="utf-8")
        )["provider_tree_sha256"],
        "data_manifest": str(data_manifest_path),
        "data_manifest_sha256": file_sha256(data_manifest_path),
        "reused": reused,
        "symbols": int(intervals["symbol"].nunique()),
        "intervals": int(len(intervals)),
    }
    _atomic_write(
        runtime_root / "latest_provider.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an immutable Qlib provider view with PIT CSI800 instruments."
    )
    parser.add_argument(
        "--canonical-provider", type=Path, default=DEFAULT_CANONICAL_PROVIDER
    )
    parser.add_argument("--intervals", type=Path, default=DEFAULT_INTERVALS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--status-out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = prepare_runtime_provider(
        canonical_provider=args.canonical_provider or qlib_workflow_provider_uri("cleaned"),
        intervals_path=args.intervals,
        runtime_root=args.runtime_root,
        data_manifest_path=args.data_manifest,
    )
    if args.status_out_dir is not None:
        _atomic_write(
            args.status_out_dir / "provider_latest.json",
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
