#!/usr/bin/env python3
"""Publish existing PIT/reference artifacts through the quant_data v2 contract.

This command is deliberately offline: it copies already verified local files into
immutable snapshots and never calls a market-data provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from _robust_membership_integrity import membership_read_lock  # noqa: E402
from _robust_reference_integrity import (  # noqa: E402
    REFERENCE_SNAPSHOT_FILES,
    reference_read_lock,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_artifact_snapshot,
    publish_snapshot,
    sha256_file,
    validate_artifact_snapshot,
    validate_latest_pointer,
)
from sync_data_manifest import sync_data_manifest  # noqa: E402


PRODUCER = {"script": "publish_reference_snapshots.py", "version": "1"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"source manifest is invalid: {path.name}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"source manifest is invalid: {path.name}")
    return payload


def _source_timestamp(paths: Sequence[Path]) -> str:
    try:
        value = max(path.stat().st_mtime for path in paths)
    except (OSError, ValueError) as exc:
        raise DataContractError("source artifact timestamp is unavailable") from exc
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


def _tree_hash(root: Path, relative_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda item: item.as_posix()):
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _copy_files(source_root: Path, destination: Path, relative_paths: Sequence[Path]) -> None:
    source_root = source_root.resolve()
    for relative in relative_paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise DataContractError("source artifact path is invalid")
        source = source_root / relative
        if source.is_symlink() or not source.is_file():
            raise DataContractError(f"source artifact is unavailable: {relative}")
        resolved = source.resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError as exc:
            raise DataContractError("source artifact escapes its dataset") from exc
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _current_for_source_hash(
    *,
    quant_data_root: Path,
    dataset_root: Path,
    dataset_id: str,
    source_hash: str,
) -> Path | None:
    pointer_path = dataset_root / "LATEST.json"
    if not pointer_path.is_file():
        return None
    try:
        _, snapshot = validate_latest_pointer(
            pointer_path,
            quant_data_root=quant_data_root,
            expected_dataset_id=dataset_id,
        )
        manifest = validate_artifact_snapshot(snapshot, verify_hashes=True)
    except (DataContractError, OSError, TypeError, ValueError):
        return None
    metadata = manifest.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("source_snapshot_hash") == source_hash:
        return snapshot
    return None


def _publish_artifacts(
    *,
    quant_data_root: Path,
    dataset_root: Path,
    source_root: Path,
    dataset_id: str,
    data_kind: str,
    source: Mapping[str, Any],
    source_retrieved_at: str,
    date_range: Mapping[str, str],
    relative_paths: Sequence[Path],
    components: Mapping[str, Mapping[str, Any]],
    source_hash: str,
    upstream_snapshots: Sequence[Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = _current_for_source_hash(
        quant_data_root=quant_data_root,
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        source_hash=source_hash,
    )
    if current is not None:
        return {
            "status": "reused",
            "dataset_id": dataset_id,
            "snapshot_id": current.name,
        }
    snapshot_id = f"{dataset_id.replace('_', '-')}-{source_hash[:16]}"
    destination = dataset_root / "snapshots" / snapshot_id
    if destination.exists():
        raise DataContractError(
            f"matching {dataset_id} snapshot exists but is not the current pointer"
        )
    staging = (
        quant_data_root
        / "staging"
        / "reference-snapshots"
        / uuid.uuid4().hex
        / snapshot_id
    )
    try:
        _copy_files(source_root, staging, relative_paths)
        final_metadata = dict(metadata or {})
        final_metadata.update(
            {
                "source_snapshot_hash": source_hash,
                "source_retrieved_at_evidence": "latest_source_artifact_mtime",
            }
        )
        finalize_artifact_snapshot(
            staging,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            data_kind=data_kind,
            role="canonical",
            status="ready",
            source=source,
            price_basis="not_applicable",
            producer=PRODUCER,
            source_retrieved_at=source_retrieved_at,
            date_range=date_range,
            components=components,
            upstream_snapshots=upstream_snapshots,
            metadata=final_metadata,
        )
        published = publish_snapshot(
            staging,
            dataset_root=dataset_root,
            quant_data_root=quant_data_root,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "status": "published",
        "dataset_id": dataset_id,
        "snapshot_id": published.name,
    }


def _snapshot_identity(dataset_root: Path, quant_data_root: Path) -> dict[str, str]:
    pointer, snapshot = validate_latest_pointer(
        dataset_root / "LATEST.json",
        quant_data_root=quant_data_root,
    )
    manifest = validate_artifact_snapshot(snapshot, verify_hashes=True)
    return {
        "dataset_id": str(pointer["dataset_id"]),
        "snapshot_id": str(pointer["snapshot_id"]),
        "content_sha256": str(manifest["content_sha256"]),
        "price_basis": str(manifest["price_basis"]),
    }


def publish_pit_snapshot(*, quant_data_root: Path) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    pit = root / "reference" / "pit_csi800"
    with membership_read_lock(pit) as locked:
        locked.ensure_transaction_clean()
        source_payload = _load_json(pit / "MEMBERSHIP_SOURCE.json")
        date_range = source_payload.get("date_range")
        if not isinstance(date_range, Mapping):
            raise DataContractError("membership source date range is invalid")
        source = locked.validate_source(
            start_date=str(date_range.get("start_date") or ""),
            end_date=str(date_range.get("end_date") or ""),
        )
        daily = [path.relative_to(pit) for path in locked.daily_files()]
        source_hash = locked.fingerprint()
        relative_paths = [
            Path("MEMBERSHIP_SOURCE.json"),
            Path("membership_intervals.parquet"),
            *daily,
        ]
        retrieved_at = _source_timestamp([pit / path for path in relative_paths])
        result = _publish_artifacts(
            quant_data_root=root,
            dataset_root=pit,
            source_root=pit,
            dataset_id="pit_csi800",
            data_kind="point_in_time_membership",
            source={
                "provider": source["selected"],
                "interface": "pit_membership_builder",
            },
            source_retrieved_at=retrieved_at,
            date_range={
                "start": source["date_range"]["start_date"],
                "end": source["date_range"]["end_date"],
            },
            relative_paths=relative_paths,
            components={
                "daily": {"path": "daily", "kind": "partitioned_parquet"},
                "membership_intervals": {
                    "path": "membership_intervals.parquet",
                    "kind": "parquet",
                },
                "source_manifest": {
                    "path": "MEMBERSHIP_SOURCE.json",
                    "kind": "json",
                },
            },
            source_hash=source_hash,
            metadata={"membership_snapshot_hash": source_hash},
        )
    return result


def publish_robust_reference_snapshot(*, quant_data_root: Path) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    reference = root / "reference"
    pit_identity = _snapshot_identity(reference / "pit_csi800", root)
    pit_manifest = validate_artifact_snapshot(
        reference / "pit_csi800" / "snapshots" / pit_identity["snapshot_id"]
    )
    pit_metadata = pit_manifest.get("metadata")
    membership_hash = (
        pit_metadata.get("membership_snapshot_hash")
        if isinstance(pit_metadata, Mapping)
        else None
    )
    if not isinstance(membership_hash, str):
        raise DataContractError("PIT membership snapshot hash is unavailable")
    with reference_read_lock(reference) as locked:
        locked.ensure_transaction_clean()
        source_payload = _load_json(reference / "REFERENCE_SOURCE.json")
        date_range = source_payload.get("date_range")
        if not isinstance(date_range, Mapping):
            raise DataContractError("reference source date range is invalid")
        source = locked.validate_source(
            start_date=str(date_range.get("start_date") or ""),
            end_date=str(date_range.get("end_date") or ""),
            expected_membership_snapshot_hash=membership_hash,
        )
        source_hash = locked.fingerprint()
        relative_paths = [Path(name) for name in REFERENCE_SNAPSHOT_FILES]
        retrieved_at = _source_timestamp([reference / path for path in relative_paths])
        result = _publish_artifacts(
            quant_data_root=root,
            dataset_root=reference,
            source_root=reference,
            dataset_id="robust_reference",
            data_kind="point_in_time_reference",
            source={
                "provider": source["selected"],
                "interface": "robust_reference_builder",
            },
            source_retrieved_at=retrieved_at,
            date_range={
                "start": source["date_range"]["start_date"],
                "end": source["date_range"]["end_date"],
            },
            relative_paths=relative_paths,
            components={
                "security_master": {
                    "path": "security_master.parquet",
                    "kind": "parquet",
                },
                "industry_intervals": {
                    "path": "industry_intervals.parquet",
                    "kind": "parquet",
                },
                "st_status": {"path": "st_status.parquet", "kind": "parquet"},
                "suspension_status": {
                    "path": "suspension_status.parquet",
                    "kind": "parquet",
                },
                "source_manifest": {
                    "path": "REFERENCE_SOURCE.json",
                    "kind": "json",
                },
            },
            source_hash=source_hash,
            upstream_snapshots=[pit_identity],
            metadata={
                "reference_snapshot_hash": source_hash,
                "membership_snapshot_hash": membership_hash,
            },
        )
    return result


def _validate_fundamentals_source(source_root: Path) -> tuple[
    dict[str, Any], list[Path], str
]:
    source_path = source_root / "FUNDAMENTAL_SOURCE.json"
    source = _load_json(source_path)
    coverage = source.get("coverage")
    if (
        source.get("status") != "pass"
        or source.get("mixed_sources") is not False
        or source.get("effective_date_rule")
        != "max_notice_or_update_plus_one_trading_day"
        or not isinstance(coverage, Mapping)
        or coverage.get("passed") is not True
        or float(coverage.get("minimum", 0.0)) < 0.98
    ):
        raise DataContractError("fundamental source contract is not ready")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise DataContractError("fundamental artifact manifest is missing")
    relative_paths = [Path("FUNDAMENTAL_SOURCE.json")]
    for relative_text, expected_hash in sorted(artifacts.items()):
        relative = Path(str(relative_text))
        if relative.is_absolute() or ".." in relative.parts:
            raise DataContractError("fundamental artifact path is invalid")
        path = source_root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            raise DataContractError(f"fundamental artifact hash mismatch: {relative}")
        relative_paths.append(relative)
    actual_parquet = {
        path.relative_to(source_root)
        for path in source_root.rglob("*.parquet")
        if "snapshots" not in path.relative_to(source_root).parts
    }
    expected_parquet = {path for path in relative_paths if path.suffix == ".parquet"}
    if actual_parquet != expected_parquet:
        raise DataContractError("fundamental artifact manifest is stale")
    events = pd.read_parquet(source_root / "fundamental_events.parquet")
    required_events = {"source_available_date", "effective_date", "report_date"}
    if not required_events.issubset(events.columns):
        raise DataContractError("fundamental event dates are incomplete")
    available = pd.to_datetime(events["source_available_date"], errors="coerce")
    effective = pd.to_datetime(events["effective_date"], errors="coerce")
    if available.isna().any() or effective.isna().any() or (effective <= available).any():
        raise DataContractError("fundamental event effective dates are invalid")
    for relative in expected_parquet:
        if relative.parent != Path("daily"):
            continue
        frame = pd.read_parquet(source_root / relative)
        required = {"datetime", "instrument", "effective_date", "source_available_date"}
        if not required.issubset(frame.columns):
            raise DataContractError(f"fundamental daily columns are invalid: {relative}")
        dates = pd.to_datetime(frame["datetime"], errors="coerce")
        effective = pd.to_datetime(frame["effective_date"], errors="coerce")
        invalid = effective.notna() & (effective > dates)
        if dates.isna().any() or invalid.any():
            raise DataContractError(f"fundamental daily PIT dates are invalid: {relative}")
    return source, relative_paths, _tree_hash(source_root, relative_paths)


def publish_fundamentals_snapshot(*, quant_data_root: Path) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    source_root = root / "reference" / "fundamentals"
    source, relative_paths, source_hash = _validate_fundamentals_source(source_root)
    data_range = source.get("data_range")
    if not isinstance(data_range, Mapping):
        raise DataContractError("fundamental data range is invalid")
    retrieved_at = str(source.get("source_retrieved_at") or "")
    try:
        parsed_retrieved_at = datetime.fromisoformat(retrieved_at)
        if parsed_retrieved_at.tzinfo is None:
            raise ValueError
    except ValueError:
        retrieved_at = _source_timestamp([source_root / path for path in relative_paths])
    result = _publish_artifacts(
        quant_data_root=root,
        dataset_root=source_root,
        source_root=source_root,
        dataset_id="ashare_fundamentals_pit",
        data_kind="point_in_time_fundamentals",
        source={
            "provider": str(source.get("selected_source") or ""),
            "interface": "historical_statements_with_announcement_dates",
        },
        source_retrieved_at=retrieved_at,
        date_range={
            "start": str(data_range.get("start") or ""),
            "end": str(data_range.get("end") or ""),
        },
        relative_paths=relative_paths,
        components={
            "daily": {"path": "daily", "kind": "partitioned_parquet"},
            "events": {
                "path": "fundamental_events.parquet",
                "kind": "parquet",
            },
            "source_manifest": {
                "path": "FUNDAMENTAL_SOURCE.json",
                "kind": "json",
            },
        },
        source_hash=source_hash,
        metadata={
            "effective_date_rule": source["effective_date_rule"],
            "source_contract_version": source.get("source_contract_version"),
            "coverage": source["coverage"],
        },
    )
    return result


def publish_existing_reference_snapshots(
    *,
    quant_data_root: Path,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    results = {
        "pit_csi800": publish_pit_snapshot(quant_data_root=root),
        "robust_reference": publish_robust_reference_snapshot(quant_data_root=root),
    }
    fundamentals = root / "reference" / "fundamentals" / "FUNDAMENTAL_SOURCE.json"
    if fundamentals.is_file():
        results["ashare_fundamentals_pit"] = publish_fundamentals_snapshot(
            quant_data_root=root
        )
    else:
        results["ashare_fundamentals_pit"] = {
            "status": "missing",
            "dataset_id": "ashare_fundamentals_pit",
        }
    sync_data_manifest(
        quant_data_root=root,
        reports_root=reports_root,
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-data-root",
        type=Path,
        default=default_quant_data_root(),
    )
    parser.add_argument("--reports-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = publish_existing_reference_snapshots(
            quant_data_root=args.quant_data_root,
            reports_root=args.reports_root,
        )
    except (DataContractError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
