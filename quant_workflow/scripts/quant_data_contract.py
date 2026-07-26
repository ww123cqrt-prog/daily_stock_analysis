#!/usr/bin/env python3
"""Versioned, auditable contracts for datasets stored under ``quant_data``."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import pandas as pd


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
ADJUSTED_PRICE_BASES = {"qfq", "hfq"}


class DataContractError(RuntimeError):
    """Raised when a dataset cannot prove its storage or price contract."""


class ResolvedDataset(NamedTuple):
    dataset_id: str
    snapshot_path: Path
    manifest: dict[str, Any]
    catalog_entry: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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


def _validate_bar_frame(frame: pd.DataFrame, path: Path) -> pd.Series:
    missing = [column for column in REQUIRED_BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"bar file {path.name} is missing columns: {missing}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise DataContractError(f"bar dates are invalid: {path.name}")
    for column in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise DataContractError(f"bar column {column} is invalid: {path.name}")
    if (pd.to_numeric(frame["volume"], errors="coerce") < 0).any():
        raise DataContractError(f"bar volume is negative: {path.name}")
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


def _build_qlib_file_index(snapshot: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(snapshot).rglob("*")):
        if (
            not path.is_file()
            or path.name in {"DATASET_MANIFEST.json", "FILE_INDEX.parquet"}
        ):
            continue
        rows.append(
            {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
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
) -> Path:
    snapshot = Path(snapshot).resolve()
    if not DATASET_ID_RE.fullmatch(dataset_id):
        raise DataContractError("dataset_id is invalid")
    if not snapshot_id or "/" in snapshot_id or snapshot_id in {".", ".."}:
        raise DataContractError("snapshot_id is invalid")
    if not source.get("provider") or not producer.get("script"):
        raise DataContractError("Qlib source and producer are required")
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
        "producer": dict(producer),
        "upstream_snapshots": [dict(upstream_snapshot)],
        "date_range": {"start": calendar[0], "end": calendar[-1]},
        "latest_trade_date": calendar[-1],
        "instrument_count": len(instruments),
        "feature_file_count": int(feature_files.sum()),
        "provider_manifest_path": "PROVIDER_MANIFEST.json",
        "file_index_path": "FILE_INDEX.parquet",
    }
    manifest["content_sha256"] = _content_hash(snapshot, manifest)
    atomic_write_json(snapshot / "DATASET_MANIFEST.json", manifest)
    validate_qlib_snapshot(snapshot, verify_hashes=True)
    return snapshot / "DATASET_MANIFEST.json"


def validate_qlib_snapshot(
    snapshot: Path,
    *,
    verify_hashes: bool = True,
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
    provider_manifest = _read_json(snapshot / str(manifest["provider_manifest_path"]))
    if provider_manifest.get("source_snapshot") != upstream[0]:
        raise DataContractError("Qlib provider and dataset lineage disagree")
    try:
        recorded = pd.read_parquet(snapshot / str(manifest["file_index_path"]))
    except Exception as exc:
        raise DataContractError("Qlib file index is unavailable") from exc
    required_index = {"relative_path", "size_bytes", "sha256"}
    if recorded.empty or not required_index.issubset(recorded.columns):
        raise DataContractError("Qlib file index is invalid")
    rebuilt = _build_qlib_file_index(snapshot)
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
) -> dict[str, Any]:
    manifest = _read_json(Path(snapshot) / "DATASET_MANIFEST.json")
    if manifest.get("data_kind") == "qlib_provider":
        return validate_qlib_snapshot(snapshot, verify_hashes=verify_hashes)
    return validate_snapshot(snapshot, verify_hashes=verify_hashes)


def publish_snapshot(
    staging_snapshot: Path,
    *,
    dataset_root: Path,
    quant_data_root: Path,
) -> Path:
    quant_data_root = Path(quant_data_root).expanduser().resolve()
    staging_snapshot = Path(staging_snapshot).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    _safe_relative(dataset_root, quant_data_root, "dataset_root")
    staging_root = quant_data_root / "staging"
    try:
        staging_snapshot.relative_to(staging_root)
    except ValueError as exc:
        raise DataContractError("staging snapshot must be under quant_data/staging") from exc
    manifest = validate_snapshot(staging_snapshot, verify_hashes=True)
    if manifest["status"] != "ready":
        raise DataContractError("only ready snapshots can be published")
    snapshot_id = str(manifest["snapshot_id"])
    destination = dataset_root / "snapshots" / snapshot_id
    if destination.exists():
        raise DataContractError("snapshot_id is already published")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_snapshot, destination)
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": manifest["dataset_id"],
        "snapshot_id": snapshot_id,
        "snapshot_path": _safe_relative(destination, quant_data_root, "snapshot_path"),
        "manifest_sha256": sha256_file(destination / "DATASET_MANIFEST.json"),
        "published_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    atomic_write_json(dataset_root / "LATEST.json", pointer)
    return destination


def publish_snapshot_group(
    snapshots: list[tuple[Path, Path]],
    *,
    quant_data_root: Path,
    update_root_catalog: bool = False,
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
        manifest = validate_dataset_snapshot(staging_snapshot, verify_hashes=True)
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
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(item["staging"], destination)
            moved.append(item)
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
                datasets[str(manifest["dataset_id"])] = catalog_entry_from_snapshot(
                    item["destination"],
                    quant_data_root=quant_data_root,
                    dataset_root=item["dataset_root"],
                )
            write_root_manifest(
                quant_data_root,
                datasets,
                existing=catalog_backup,
            )
    except Exception:
        for pointer_path, backup in pointer_backups.items():
            try:
                if backup is None:
                    pointer_path.unlink(missing_ok=True)
                else:
                    atomic_write_json(pointer_path, backup)
            except Exception:
                pass
        if catalog_backup is not None:
            try:
                atomic_write_json(
                    quant_data_root / "DATA_MANIFEST.json", catalog_backup
                )
            except Exception:
                pass
        for item in reversed(moved):
            try:
                item["staging"].parent.mkdir(parents=True, exist_ok=True)
                os.replace(item["destination"], item["staging"])
            except Exception:
                pass
        raise
    return [item["destination"] for item in prepared]


def publish_qlib_snapshot(
    staging_snapshot: Path,
    *,
    dataset_root: Path,
    quant_data_root: Path,
    update_root_catalog: bool = False,
) -> Path:
    return publish_snapshot_group(
        [(staging_snapshot, dataset_root)],
        quant_data_root=quant_data_root,
        update_root_catalog=update_root_catalog,
    )[0]


def catalog_entry_from_snapshot(
    snapshot: Path,
    *,
    quant_data_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    quant_data_root = Path(quant_data_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    snapshot = Path(snapshot).resolve()
    manifest = validate_dataset_snapshot(snapshot, verify_hashes=True)
    if manifest["data_kind"] == "qlib_provider":
        return {
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
    if not isinstance(manifest.get("datasets"), dict):
        raise DataContractError("DATA_MANIFEST.json datasets catalog is missing")
    return manifest


def resolve_dataset(
    dataset_id: str,
    *,
    quant_data_root: Path,
    require_basis: str | None = None,
    require_status: str = "ready",
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
    pointer = _read_json(pointer_path)
    if pointer.get("dataset_id") != dataset_id:
        raise DataContractError("latest pointer belongs to another dataset")
    if entry.get("snapshot_id") != pointer.get("snapshot_id"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    if entry.get("snapshot_manifest_sha256") != pointer.get("manifest_sha256"):
        raise DataContractError("dataset catalog and latest pointer are out of sync")
    snapshot = _safe_child(
        quant_data_root, pointer.get("snapshot_path"), "snapshot_path"
    )
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    if sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise DataContractError("latest pointer manifest hash mismatch")
    manifest = validate_dataset_snapshot(snapshot, verify_hashes=True)
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
    return ResolvedDataset(dataset_id, snapshot, manifest, dict(entry))


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
