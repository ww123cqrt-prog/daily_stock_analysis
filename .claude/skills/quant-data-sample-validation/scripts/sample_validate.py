#!/usr/bin/env python3
"""Read-only deterministic spot validation for governed quant_data snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


class SampleValidationError(RuntimeError):
    """Raised when sampled snapshot evidence is inconsistent."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleValidationError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise SampleValidationError(f"{label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_child(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SampleValidationError(f"{label} is missing")
    relative = Path(value)
    if relative.is_absolute():
        raise SampleValidationError(f"{label} must be relative")
    root = root.resolve()
    unresolved = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SampleValidationError(f"{label} cannot use symlinks")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SampleValidationError(f"{label} escapes its root") from exc
    return resolved


def _select_rows(index: pd.DataFrame, count: int, seed: str) -> pd.DataFrame:
    if count < 1:
        raise SampleValidationError("sample size must be positive")
    if "relative_path" not in index.columns or index.empty:
        raise SampleValidationError("file index is empty or missing relative_path")
    ranked = index.copy()
    ranked["_sample_rank"] = ranked["relative_path"].astype(str).map(
        lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()
    )
    return ranked.sort_values("_sample_rank", kind="stable").head(count).drop(
        columns="_sample_rank"
    )


def _validate_indexed_file(snapshot: Path, row: Any) -> dict[str, Any]:
    relative_path = str(row.relative_path)
    path = _safe_child(snapshot, relative_path, "indexed file path")
    if not path.is_file() or path.is_symlink():
        raise SampleValidationError(f"indexed file is unavailable: {relative_path}")
    expected_hash = str(row.sha256)
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise SampleValidationError(f"file hash mismatch: {relative_path}")
    if hasattr(row, "size_bytes") and int(row.size_bytes) != path.stat().st_size:
        raise SampleValidationError(f"file size mismatch: {relative_path}")

    result: dict[str, Any] = {
        "relative_path": relative_path,
        "sha256": actual_hash,
    }
    if (
        path.suffix == ".parquet"
        and hasattr(row, "row_count")
        and hasattr(row, "first_date")
        and hasattr(row, "last_date")
    ):
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise SampleValidationError(f"cannot read parquet file: {relative_path}") from exc
        if "date" not in frame.columns:
            raise SampleValidationError(f"date column is missing: {relative_path}")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise SampleValidationError(f"date sequence is invalid: {relative_path}")
        if len(frame) != int(row.row_count):
            raise SampleValidationError(f"row count mismatch: {relative_path}")
        if dates.iloc[0].date().isoformat() != str(row.first_date):
            raise SampleValidationError(f"first date mismatch: {relative_path}")
        if dates.iloc[-1].date().isoformat() != str(row.last_date):
            raise SampleValidationError(f"last date mismatch: {relative_path}")
        price_fields = {"open", "high", "low", "close"}
        if price_fields.intersection(frame.columns) and not price_fields.issubset(frame.columns):
            raise SampleValidationError(f"OHLC columns are incomplete: {relative_path}")
        result.update(
            {
                "row_count": len(frame),
                "first_date": dates.iloc[0].date().isoformat(),
                "last_date": dates.iloc[-1].date().isoformat(),
            }
        )
    return result


def _validate_dataset(
    root: Path,
    catalog: dict[str, Any],
    dataset_id: str,
    *,
    sample_size: int,
    seed: str,
) -> dict[str, Any]:
    datasets = catalog.get("datasets")
    entry = datasets.get(dataset_id) if isinstance(datasets, dict) else None
    if not isinstance(entry, dict):
        raise SampleValidationError(f"dataset is not registered: {dataset_id}")
    if entry.get("status") != "ready":
        raise SampleValidationError(
            f"dataset status is {entry.get('status')}, not ready: {dataset_id}"
        )
    pointer_path = _safe_child(root, entry.get("latest_pointer"), "latest pointer")
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise SampleValidationError(f"latest pointer is unavailable: {dataset_id}")
    pointer = _read_json(pointer_path, "latest pointer")
    snapshot = _safe_child(root, pointer.get("snapshot_path"), "snapshot path")
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise SampleValidationError(f"snapshot directory is unavailable: {dataset_id}")
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest = _read_json(manifest_path, "snapshot manifest")

    for field in ("dataset_id", "snapshot_id"):
        expected = dataset_id if field == "dataset_id" else pointer.get("snapshot_id")
        if pointer.get(field) != expected or manifest.get(field) != expected:
            raise SampleValidationError(f"{field} mismatch: {dataset_id}")
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != pointer.get("manifest_sha256"):
        raise SampleValidationError(f"manifest hash mismatch: {dataset_id}")
    if entry.get("snapshot_manifest_sha256") != manifest_hash:
        raise SampleValidationError(f"catalog and pointer disagree: {dataset_id}")
    for field in ("role", "status", "price_basis", "content_sha256"):
        if entry.get(field) != manifest.get(field):
            raise SampleValidationError(f"catalog and manifest disagree on {field}: {dataset_id}")

    index_path = _safe_child(snapshot, manifest.get("file_index_path"), "file index")
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        raise SampleValidationError(f"file index is unavailable: {dataset_id}") from exc
    if "sha256" not in index.columns:
        raise SampleValidationError(f"file index is missing sha256: {dataset_id}")
    selected = _select_rows(
        index,
        min(sample_size, len(index)),
        f"{seed}\0{dataset_id}\0{pointer.get('snapshot_id')}",
    )
    checked_files = [
        _validate_indexed_file(snapshot, row)
        for row in selected.itertuples(index=False)
    ]

    checked_factors: list[dict[str, Any]] = []
    factor_index_value = manifest.get("factor_index_path")
    if factor_index_value and "symbol" in selected.columns:
        factor_index_path = _safe_child(snapshot, factor_index_value, "factor index")
        try:
            factor_index = pd.read_parquet(factor_index_path)
        except Exception as exc:
            raise SampleValidationError(f"factor index is unavailable: {dataset_id}") from exc
        selected_symbols = set(selected["symbol"].astype(str))
        factor_sample = factor_index[
            factor_index["symbol"].astype(str).isin(selected_symbols)
        ]
        if len(factor_sample) != len(selected_symbols):
            raise SampleValidationError(f"sampled factor coverage is incomplete: {dataset_id}")
        checked_factors = [
            _validate_indexed_file(snapshot, row)
            for row in factor_sample.itertuples(index=False)
        ]

    actions_path_value = manifest.get("corporate_actions_path")
    if actions_path_value:
        actions_path = _safe_child(snapshot, actions_path_value, "corporate actions")
        if not actions_path.is_file() or actions_path.is_symlink():
            raise SampleValidationError(f"corporate actions are unavailable: {dataset_id}")
        if _sha256(actions_path) != manifest.get("corporate_actions_sha256"):
            raise SampleValidationError(f"corporate actions hash mismatch: {dataset_id}")

    return {
        "dataset_id": dataset_id,
        "snapshot_id": pointer.get("snapshot_id"),
        "status": "sample_validation_passed",
        "indexed_files_checked": len(checked_files),
        "indexed_files_total": len(index),
        "indexed_files_unchecked": len(index) - len(checked_files),
        "factor_files_checked": len(checked_factors),
        "selected_files": checked_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", default="quant-data-sample-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    try:
        catalog = _read_json(root / "DATA_MANIFEST.json", "root data manifest")
        registered = catalog.get("datasets")
        if not isinstance(registered, dict):
            raise SampleValidationError("root data manifest has no datasets map")
        dataset_ids = args.datasets or sorted(
            dataset_id
            for dataset_id, entry in registered.items()
            if isinstance(entry, dict)
            and entry.get("status") == "ready"
            and entry.get("latest_pointer")
        )
        if not dataset_ids:
            raise SampleValidationError("no ready snapshot datasets were selected")
        results = [
            _validate_dataset(
                root,
                catalog,
                dataset_id,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            for dataset_id in dataset_ids
        ]
    except SampleValidationError as exc:
        print(
            json.dumps(
                {"status": "sample_validation_failed", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "sample_validation_passed",
                "mode": "deterministic_spot_check",
                "seed": args.seed,
                "datasets": results,
                "warning": "Unchecked files remain; this is not full validation.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
