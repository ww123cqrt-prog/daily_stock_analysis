#!/usr/bin/env python3
"""Atomically promote a validated daily-update candidate with rollback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import write_json_atomic  # noqa: E402
from _paths import quant_data_root as default_quant_data_root, reports_root as default_reports_root  # noqa: E402
from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402
from sync_data_manifest import sync_data_manifest  # noqa: E402


REQUIRED_ARTIFACTS = (
    "merged_parquet",
    "merged_qlib",
    "cleaned_parquet",
    "cleaned_qlib",
)


def _has_v2_publication_evidence(
    quant_data_root: Path, targets: dict[str, Path]
) -> bool:
    root = Path(quant_data_root)
    marker_paths = [
        root
        / "parquet"
        / "ashare_1d"
        / "ashare_daily_qfq"
        / "LATEST.json",
        root / "qlib" / "cn_data_ashare_v2" / "LATEST.json",
    ]
    if any(path.is_file() for path in marker_paths):
        return True
    for name in ("merged_parquet", "cleaned_parquet"):
        parquet_target = targets.get(name)
        if parquet_target is None:
            continue
        compatibility_marker = (
            Path(parquet_target)
            / ".metadata"
            / "COMPATIBILITY_MANIFEST.json"
        )
        if compatibility_marker.is_file():
            return True
    for name in ("merged_qlib", "cleaned_qlib"):
        target = targets.get(name)
        if target is None:
            continue
        provider_marker = Path(target) / "PROVIDER_MANIFEST.json"
        if provider_marker.is_file():
            return True
    return False


def _restore_manifest(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
            _fsync_directory(path.parent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(previous)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_with_rollback_retries(source: Path, target: Path) -> None:
    last_error: OSError | None = None
    for _attempt in range(3):
        try:
            source.replace(target)
            return
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _rollback_publication(
    *,
    promoted: list[str],
    backed_up: list[str],
    candidates: dict[str, Path],
    targets: dict[str, Path],
    backup_root: Path,
) -> list[str]:
    errors: list[str] = []
    for name in reversed(promoted):
        target = targets[name]
        candidate = candidates[name]
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            continue
        try:
            _replace_with_rollback_retries(target, candidate)
        except OSError as exc:
            errors.append(f"restore candidate {name}: {type(exc).__name__}")
    for name in reversed(backed_up):
        backup = backup_root / name
        target = targets[name]
        if not backup.exists():
            continue
        if target.exists():
            errors.append(f"restore official {name}: target is still occupied")
            continue
        try:
            _replace_with_rollback_retries(backup, target)
        except OSError as exc:
            errors.append(f"restore official {name}: {type(exc).__name__}")
    return errors


def _validate_artifacts(candidates: dict[str, Path], targets: dict[str, Path], backup_root: Path) -> None:
    missing_names = [name for name in REQUIRED_ARTIFACTS if name not in candidates or name not in targets]
    if missing_names:
        raise ValueError(f"missing artifact mappings: {', '.join(missing_names)}")
    missing_paths = [str(candidates[name]) for name in REQUIRED_ARTIFACTS if not candidates[name].is_dir()]
    if missing_paths:
        raise FileNotFoundError(f"candidate artifacts are missing: {', '.join(missing_paths)}")
    if backup_root.exists() and any(backup_root.iterdir()):
        raise FileExistsError(f"backup root is not empty: {backup_root}")

    for name in REQUIRED_ARTIFACTS:
        target_parent = targets[name].parent
        target_parent.mkdir(parents=True, exist_ok=True)
        if candidates[name].stat().st_dev != target_parent.stat().st_dev:
            raise OSError(f"candidate and target must share a filesystem for atomic rename: {name}")


def publish_candidates(
    *,
    candidates: dict[str, Path],
    targets: dict[str, Path],
    backup_root: Path,
    quant_data_root: Path,
    reports_root: Path,
    sync_manifest_fn: Callable[..., Path] = sync_data_manifest,
) -> dict[str, object]:
    candidates = {name: Path(path) for name, path in candidates.items()}
    targets = {name: Path(path) for name, path in targets.items()}
    backup_root = Path(backup_root)
    quant_data_root = Path(quant_data_root)
    reports_root = Path(reports_root)
    try:
        resolve_dataset(
            "ashare_daily_qfq",
            quant_data_root=quant_data_root,
            require_basis="qfq",
        )
    except (DataContractError, OSError, ValueError):
        if _has_v2_publication_evidence(quant_data_root, targets):
            raise RuntimeError(
                "V2 compatibility evidence exists; legacy publisher cannot "
                "replace the official compatibility paths even though the V2 "
                "catalog cannot currently be resolved."
            )
    else:
        raise RuntimeError(
            "V2 canonical data is ready; legacy publisher cannot replace V1 "
            "compatibility paths. Use publish_v2_compatibility.py."
        )
    _validate_artifacts(candidates, targets, backup_root)
    backup_root.mkdir(parents=True, exist_ok=True)

    manifest_path = quant_data_root / "DATA_MANIFEST.json"
    previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    backed_up: list[str] = []
    promoted: list[str] = []
    try:
        for name in REQUIRED_ARTIFACTS:
            target = targets[name]
            backup = backup_root / name
            if target.exists():
                target.replace(backup)
                backed_up.append(name)
        for name in REQUIRED_ARTIFACTS:
            candidates[name].replace(targets[name])
            promoted.append(name)

        synced_manifest = sync_manifest_fn(
            quant_data_root=quant_data_root,
            reports_root=reports_root,
        )
    except Exception as publish_error:
        rollback_errors = _rollback_publication(
            promoted=promoted,
            backed_up=backed_up,
            candidates=candidates,
            targets=targets,
            backup_root=backup_root,
        )
        try:
            _restore_manifest(manifest_path, previous_manifest)
        except Exception as exc:
            rollback_errors.append(f"restore manifest: {type(exc).__name__}")
        if rollback_errors:
            raise RuntimeError(
                "Legacy publication failed "
                f"({type(publish_error).__name__}) and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        raise

    return {
        "status": "published",
        "published_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "artifacts": list(REQUIRED_ARTIFACTS),
        "backup_root": str(backup_root),
        "manifest_path": str(synced_manifest),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a validated daily-update candidate.")
    for name in REQUIRED_ARTIFACTS:
        parser.add_argument(f"--candidate-{name.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--target-{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--quant-data-root", type=Path, default=default_quant_data_root())
    parser.add_argument("--reports-root", type=Path, default=default_reports_root())
    parser.add_argument("--status-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = {name: getattr(args, f"candidate_{name}") for name in REQUIRED_ARTIFACTS}
    targets = {name: getattr(args, f"target_{name}") for name in REQUIRED_ARTIFACTS}
    status_dir = args.status_dir or args.reports_root / "runtime" / "status"
    try:
        result = publish_candidates(
            candidates=candidates,
            targets=targets,
            backup_root=args.backup_root,
            quant_data_root=args.quant_data_root,
            reports_root=args.reports_root,
        )
    except Exception as exc:
        write_json_atomic(
            status_dir / "official_publish_latest.json",
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": "failed",
                "error_type": type(exc).__name__,
            },
        )
        print(f"ERROR: official publish failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    write_json_atomic(status_dir / "official_publish_latest.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
