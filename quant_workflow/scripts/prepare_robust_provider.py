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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root  # noqa: E402


DEFAULT_CANONICAL_PROVIDER = qlib_provider_uri("cleaned")
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
    path = provider / "calendars" / "day.txt"
    if not path.is_file():
        raise FileNotFoundError(f"canonical provider calendar is missing: {path}")
    values = pd.DatetimeIndex(
        pd.to_datetime(path.read_text(encoding="utf-8").splitlines(), errors="coerce")
    )
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
    calendar_hash = file_sha256(canonical_provider / "calendars" / "day.txt")
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


def _snapshot_is_reusable(provider: Path, key: str) -> bool:
    metadata_path = provider / PROVIDER_METADATA_FILENAME
    content_path = provider / PROVIDER_CONTENT_FILENAME
    instrument_path = provider / "instruments" / "csi800.txt"
    if (
        provider.is_symlink()
        or not provider.is_dir()
        or not metadata_path.is_file()
        or not content_path.is_file()
        or not instrument_path.is_file()
    ):
        return False
    if any(path.is_symlink() for path in provider.rglob("*")):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        content = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("format_version") != PROVIDER_FORMAT_VERSION
        or metadata.get("provider_key") != key
        or metadata.get("content_manifest_sha256") != file_sha256(content_path)
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
    actual_paths = {
        path.relative_to(provider).as_posix()
        for path in provider.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        return False
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if any(
        path.stat().st_mode & write_mask for path in [provider, *provider.rglob("*")]
    ):
        return False
    try:
        for relative, expected in files.items():
            path = provider / relative
            if path.is_symlink() or not path.is_file():
                return False
            if int(expected.get("size", -1)) != path.stat().st_size:
                return False
            if expected.get("sha256") != file_sha256(path):
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def runtime_provider_identity(provider: Path, key: str) -> dict[str, str] | None:
    provider = Path(provider)
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
        canonical_provider=args.canonical_provider,
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
