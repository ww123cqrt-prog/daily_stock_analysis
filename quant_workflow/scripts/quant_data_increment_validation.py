#!/usr/bin/env python3
"""Read-only semantic validation for governed quant_data increments."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from download_baostock_v2 import (
    build_local_qfq,
    expand_daily_factors,
    sha256_file,
)
from incremental_update_ashare_v2 import _validate_bundle
from quant_data_contract import DataContractError, validate_dataset_snapshot


_SNAPSHOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PRICE_FIELDS = ("open", "high", "low", "close")
_EVENT_VALUE_FIELDS = ("foreAdjustFactor", "backAdjustFactor", "adjustFactor")


class IncrementValidationError(DataContractError):
    """Raised when a candidate snapshot is not a valid increment."""


@dataclass(frozen=True)
class SnapshotView:
    dataset_id: str
    snapshot_id: str
    snapshot_path: Path
    manifest: dict[str, Any]
    index: pd.DataFrame


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncrementValidationError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise IncrementValidationError(f"{label} must be a JSON object")
    return payload


def _safe_child(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IncrementValidationError(f"{label} is missing")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise IncrementValidationError(f"{label} is unsafe")
    current = root
    for part in value.parts:
        current = current / part
        if current.is_symlink():
            raise IncrementValidationError(f"{label} cannot contain a symlink")
    try:
        current.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IncrementValidationError(f"{label} escapes its root") from exc
    return current


def _snapshot_id(value: str | None, *, label: str) -> str:
    if not isinstance(value, str) or not _SNAPSHOT_ID.fullmatch(value):
        raise IncrementValidationError(f"{label} is invalid")
    return value


def _load_index(snapshot: Path, manifest: Mapping[str, Any]) -> pd.DataFrame:
    index_path = _safe_child(
        snapshot, manifest.get("file_index_path"), label="snapshot file index"
    )
    try:
        index = pd.read_parquet(index_path)
    except (OSError, ValueError) as exc:
        raise IncrementValidationError("snapshot file index is unreadable") from exc
    required = {"symbol", "relative_path", "row_count", "first_date", "last_date", "sha256"}
    if required.difference(index.columns):
        raise IncrementValidationError("snapshot file index is incomplete")
    if index.empty or index["symbol"].duplicated().any():
        raise IncrementValidationError("snapshot symbol index is invalid")
    return index.sort_values("symbol", kind="stable").reset_index(drop=True)


def _load_snapshot(
    root: Path,
    dataset_id: str,
    snapshot_id: str | None,
    *,
    latest: bool = False,
) -> SnapshotView:
    root = Path(root).expanduser().resolve()
    catalog = _read_json(root / "DATA_MANIFEST.json", label="root data manifest")
    datasets = catalog.get("datasets")
    entry = datasets.get(dataset_id) if isinstance(datasets, dict) else None
    if not isinstance(entry, dict):
        raise IncrementValidationError(f"dataset is not registered: {dataset_id}")
    if entry.get("role") != "canonical" or entry.get("status") != "ready":
        raise IncrementValidationError("dataset is not canonical and ready")
    dataset_root = _safe_child(
        root, entry.get("dataset_root"), label="dataset root"
    )
    if latest:
        pointer = _read_json(dataset_root / "LATEST.json", label="latest pointer")
        selected_id = _snapshot_id(pointer.get("snapshot_id"), label="latest snapshot_id")
        if snapshot_id is not None and snapshot_id != selected_id:
            raise IncrementValidationError("candidate snapshot is not the current formal snapshot")
    else:
        selected_id = _snapshot_id(snapshot_id, label="snapshot_id")
    snapshot = dataset_root / "snapshots" / selected_id
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise IncrementValidationError(f"snapshot is unavailable: {selected_id}")
    try:
        manifest = validate_dataset_snapshot(
            snapshot,
            verify_hashes=False,
            evidence_root=root,
        )
    except DataContractError as exc:
        raise IncrementValidationError(str(exc)) from exc
    if manifest.get("dataset_id") != dataset_id:
        raise IncrementValidationError("snapshot dataset_id does not match")
    return SnapshotView(
        dataset_id=dataset_id,
        snapshot_id=selected_id,
        snapshot_path=snapshot,
        manifest=dict(manifest),
        index=_load_index(snapshot, manifest),
    )


def _indexed_path(snapshot: SnapshotView, row: Mapping[str, Any], *, label: str) -> Path:
    path = _safe_child(snapshot.snapshot_path, row.get("relative_path"), label=label)
    if path.is_symlink() or not path.is_file():
        raise IncrementValidationError(f"{label} is unavailable")
    expected = str(row.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
        raise IncrementValidationError(f"{label} hash mismatch")
    return path


def _normalized_frame(frame: pd.DataFrame, *, symbol: str, label: str) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise IncrementValidationError(f"{label} has no date column")
    result = frame.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    except (TypeError, ValueError) as exc:
        raise IncrementValidationError(f"{label} contains invalid dates") from exc
    if result["date"].duplicated().any():
        raise IncrementValidationError(f"{label} contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise IncrementValidationError(f"{label} dates are not sorted")
    if "symbol" in result.columns and set(result["symbol"].dropna().astype(str)) != {symbol}:
        raise IncrementValidationError(f"{label} contains the wrong symbol")
    return result.reset_index(drop=True)


def _read_indexed_frame(snapshot: SnapshotView, row: Mapping[str, Any], *, label: str) -> pd.DataFrame:
    path = _indexed_path(snapshot, row, label=label)
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise IncrementValidationError(f"{label} is unreadable") from exc
    symbol = str(row["symbol"])
    normalized = _normalized_frame(frame, symbol=symbol, label=label)
    if len(normalized) != int(row["row_count"]):
        raise IncrementValidationError(f"{label} row count mismatch")
    return normalized


def _rows_by_symbol(index: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["symbol"]): dict(row)
        for row in index.to_dict(orient="records")
    }


def _frame_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if set(left.columns) != set(right.columns):
        return False
    columns = sorted(left.columns)
    try:
        pd.testing.assert_frame_equal(
            left.loc[:, columns].reset_index(drop=True),
            right.loc[:, columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def _validate_new_rows(frame: pd.DataFrame, *, label: str) -> None:
    if frame.empty:
        return
    missing = set(_PRICE_FIELDS).difference(frame.columns)
    if missing:
        raise IncrementValidationError(f"{label} is missing OHLC fields")
    values = frame.loc[:, _PRICE_FIELDS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise IncrementValidationError(f"{label} contains invalid OHLC values")
    tolerance = 1e-10
    if (
        (values["high"] + tolerance < values[["open", "close", "low"]].max(axis=1)).any()
        or (values["low"] - tolerance > values[["open", "close", "high"]].min(axis=1)).any()
        or (values <= 0).any().any()
    ):
        raise IncrementValidationError(f"{label} contains invalid OHLC relationships")
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        if volume.isna().any() or (volume < 0).any():
            raise IncrementValidationError(f"{label} contains invalid volume")


def _factor_rows(snapshot: SnapshotView) -> dict[str, dict[str, Any]]:
    relative = snapshot.manifest.get("factor_index_path")
    if not isinstance(relative, str) or not relative:
        raise IncrementValidationError("qfq snapshot factor index is missing")
    path = _safe_child(snapshot.snapshot_path, relative, label="factor index")
    try:
        index = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise IncrementValidationError("factor index is unreadable") from exc
    required = {"symbol", "relative_path", "row_count", "sha256"}
    if required.difference(index.columns) or index["symbol"].duplicated().any():
        raise IncrementValidationError("factor index is invalid")
    return _rows_by_symbol(index)


def _corporate_action_groups(
    snapshot: SnapshotView,
    *,
    symbols: set[str],
) -> dict[str, pd.DataFrame]:
    path = _safe_child(
        snapshot.snapshot_path,
        snapshot.manifest.get("corporate_actions_path"),
        label="corporate actions",
    )
    if path.is_symlink() or not path.is_file():
        raise IncrementValidationError("corporate actions are unavailable")
    expected = snapshot.manifest.get("corporate_actions_sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise IncrementValidationError("corporate actions hash mismatch")
    try:
        actions = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise IncrementValidationError("corporate actions are unreadable") from exc
    if "action_type" in actions.columns:
        actions = actions.loc[actions["action_type"] == "adjustment_factor_event"]
    if actions.empty:
        return {}
    required = {"symbol", "event_date", *_EVENT_VALUE_FIELDS}
    if required.difference(actions.columns):
        raise IncrementValidationError("corporate actions are incomplete")
    normalized = actions.loc[:, ["symbol", "event_date", *_EVENT_VALUE_FIELDS]].copy()
    normalized["symbol"] = normalized["symbol"].astype(str)
    if not set(normalized["symbol"]).issubset(symbols):
        raise IncrementValidationError("corporate actions contain an unknown symbol")
    normalized["event_date"] = pd.to_datetime(
        normalized["event_date"], errors="coerce"
    ).dt.normalize().astype("datetime64[ns]")
    for field in _EVENT_VALUE_FIELDS:
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
    if normalized.loc[:, ["event_date", *_EVENT_VALUE_FIELDS]].isna().any().any():
        raise IncrementValidationError("corporate actions contain invalid values")
    if (normalized["foreAdjustFactor"] <= 0).any():
        raise IncrementValidationError("corporate actions contain invalid factors")
    normalized = (
        normalized.sort_values(["symbol", "event_date"], kind="stable")
        .drop_duplicates(["symbol", "event_date"], keep="last")
        .reset_index(drop=True)
    )
    return {
        symbol: group.drop(columns="symbol").reset_index(drop=True)
        for symbol, group in normalized.groupby("symbol", sort=False)
    }


def _corporate_action_change_symbols(
    base: SnapshotView,
    candidate: SnapshotView,
    *,
    symbols: set[str],
) -> set[str]:
    base_groups = _corporate_action_groups(base, symbols=symbols)
    candidate_groups = _corporate_action_groups(candidate, symbols=symbols)
    changed: set[str] = set()
    for symbol in symbols:
        left = base_groups.get(symbol)
        right = candidate_groups.get(symbol)
        if left is None or right is None:
            if left is not None or right is not None:
                changed.add(symbol)
            continue
        if len(left) != len(right) or not left["event_date"].equals(right["event_date"]):
            changed.add(symbol)
            continue
        if not np.isclose(
            left.loc[:, _EVENT_VALUE_FIELDS].to_numpy(dtype=float),
            right.loc[:, _EVENT_VALUE_FIELDS].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ).all():
            changed.add(symbol)
    return changed


def _factor_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if "factor" not in left.columns or "factor" not in right.columns:
        raise IncrementValidationError("factor file has no factor column")
    try:
        pd.testing.assert_frame_equal(
            left.loc[:, ["date", "factor"]].reset_index(drop=True),
            right.loc[:, ["date", "factor"]].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def _old_factor_history(
    snapshot: SnapshotView,
    row: Mapping[str, Any],
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    frame = _read_indexed_frame(snapshot, row, label=f"factor file {row['symbol']}")
    return frame.loc[frame["date"] <= cutoff].reset_index(drop=True)


def _validate_increment_lineage(
    base: SnapshotView, candidate: SnapshotView
) -> dict[str, Any]:
    source = candidate.manifest.get("source")
    source = source if isinstance(source, Mapping) else {}
    if source.get("interface") != "incremental_raw_plus_complete_factor_history":
        raise IncrementValidationError("candidate source is not an incremental update")
    try:
        requested_start = pd.Timestamp(source["requested_start"]).normalize()
        requested_through = pd.Timestamp(source["requested_through"]).normalize()
        base_latest = pd.Timestamp(base.manifest["latest_trade_date"]).normalize()
        candidate_latest = pd.Timestamp(candidate.manifest["latest_trade_date"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise IncrementValidationError("candidate incremental date lineage is invalid") from exc
    if requested_start > base_latest or requested_through != candidate_latest:
        raise IncrementValidationError("candidate incremental date lineage does not match snapshots")
    declared_base = source.get("base_snapshot_id")
    price_basis = str(candidate.manifest.get("price_basis"))
    expected_base = base.snapshot_id
    lineage_dataset_id = base.dataset_id
    if price_basis == "qfq":
        upstream = base.manifest.get("upstream_snapshots")
        upstream = upstream if isinstance(upstream, list) else []
        raw_upstream = [
            item
            for item in upstream
            if isinstance(item, Mapping)
            and item.get("dataset_id") == "ashare_daily_raw"
            and item.get("price_basis", "raw") == "raw"
        ]
        if len(raw_upstream) != 1 or not isinstance(
            raw_upstream[0].get("snapshot_id"), str
        ):
            raise IncrementValidationError(
                "selected qfq base does not declare exactly one raw upstream snapshot"
            )
        expected_base = str(raw_upstream[0]["snapshot_id"])
        lineage_dataset_id = "ashare_daily_raw"
    if declared_base != expected_base:
        raise IncrementValidationError("candidate does not declare the selected base snapshot")
    return {
        "source_interface": source.get("interface"),
        "requested_start": requested_start.date().isoformat(),
        "requested_through": requested_through.date().isoformat(),
        "declared_base_snapshot_id": declared_base,
        "selected_base_snapshot_id": base.snapshot_id,
        "lineage_base_dataset_id": lineage_dataset_id,
        "lineage_base_snapshot_id": expected_base,
    }


def compare_snapshot_increment(
    dataset_id: str,
    *,
    quant_data_root: Path,
    base_snapshot_id: str,
    candidate_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Prove that a formal candidate is a semantic increment of a base snapshot."""

    root = Path(quant_data_root).expanduser().resolve()
    base = _load_snapshot(root, dataset_id, base_snapshot_id)
    candidate = _load_snapshot(
        root,
        dataset_id,
        candidate_snapshot_id,
        latest=candidate_snapshot_id is None,
    )
    if base.snapshot_id == candidate.snapshot_id:
        raise IncrementValidationError("base and candidate snapshots must differ")
    if base.manifest.get("price_basis") != candidate.manifest.get("price_basis"):
        raise IncrementValidationError("snapshot price bases differ")
    price_basis = str(candidate.manifest.get("price_basis"))
    if price_basis not in {"raw", "qfq"}:
        raise IncrementValidationError("increment comparison supports raw or qfq daily bars")
    base_latest = pd.Timestamp(base.manifest["latest_trade_date"]).normalize()
    candidate_latest = pd.Timestamp(candidate.manifest["latest_trade_date"]).normalize()
    if candidate_latest <= base_latest:
        raise IncrementValidationError("candidate does not extend the base snapshot")
    lineage = _validate_increment_lineage(base, candidate)

    base_rows = _rows_by_symbol(base.index)
    candidate_rows = _rows_by_symbol(candidate.index)
    if set(base_rows) != set(candidate_rows):
        raise IncrementValidationError("snapshot symbol sets differ")
    base_factor_rows: dict[str, dict[str, Any]] = {}
    candidate_factor_rows: dict[str, dict[str, Any]] = {}
    event_changed_symbols: set[str] = set()
    if price_basis == "qfq":
        base_factor_rows = _factor_rows(base)
        candidate_factor_rows = _factor_rows(candidate)
        if set(base_factor_rows) != set(base_rows) or set(candidate_factor_rows) != set(base_rows):
            raise IncrementValidationError("qfq factor symbol sets differ")
        event_changed_symbols = _corporate_action_change_symbols(
            base,
            candidate,
            symbols=set(base_rows),
        )

    unchanged_history = 0
    increment_rows = 0
    symbols_with_increment = 0
    factor_value_changed: list[str] = []
    factor_metadata_changed: list[str] = []
    qfq_history_changed: list[str] = []
    no_increment: list[str] = []
    for symbol in sorted(base_rows):
        base_frame = _read_indexed_frame(
            base, base_rows[symbol], label=f"base bars {symbol}"
        )
        candidate_frame = _read_indexed_frame(
            candidate, candidate_rows[symbol], label=f"candidate bars {symbol}"
        )
        base_history = base_frame.loc[base_frame["date"] <= base_latest].reset_index(drop=True)
        candidate_history = candidate_frame.loc[
            candidate_frame["date"] <= base_latest
        ].reset_index(drop=True)
        if base_history["date"].tolist() != candidate_history["date"].tolist():
            raise IncrementValidationError(f"historical dates changed for {symbol}")
        history_equal = _frame_equal(base_history, candidate_history)
        factor_value_history_changed = False
        if price_basis == "qfq":
            old_base_factors = _old_factor_history(
                base, base_factor_rows[symbol], cutoff=base_latest
            )
            old_candidate_factors = _old_factor_history(
                candidate, candidate_factor_rows[symbol], cutoff=base_latest
            )
            if old_base_factors["date"].tolist() != old_candidate_factors["date"].tolist():
                raise IncrementValidationError(f"historical factor dates changed for {symbol}")
            factor_metadata_history_changed = not _frame_equal(
                old_base_factors, old_candidate_factors
            )
            factor_value_history_changed = not _factor_values_equal(
                old_base_factors, old_candidate_factors
            )
            if factor_metadata_history_changed:
                factor_metadata_changed.append(symbol)
            if factor_value_history_changed:
                factor_value_changed.append(symbol)
                if symbol not in event_changed_symbols:
                    raise IncrementValidationError(
                        f"factor values changed without corporate action evidence for {symbol}"
                    )
        if not history_equal and not factor_value_history_changed:
            raise IncrementValidationError(f"historical rows changed for {symbol}")
        if not history_equal:
            qfq_history_changed.append(symbol)
        if history_equal:
            unchanged_history += 1

        new_rows = candidate_frame.loc[candidate_frame["date"] > base_latest].reset_index(drop=True)
        _validate_new_rows(new_rows, label=f"candidate increment {symbol}")
        increment_rows += len(new_rows)
        if new_rows.empty:
            no_increment.append(symbol)
        else:
            symbols_with_increment += 1

    if increment_rows == 0:
        raise IncrementValidationError("candidate contains no incremental rows")
    if price_basis == "qfq":
        expected = candidate.manifest.get("adjustment", {}).get(
            "factor_changed_symbols"
        )
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise IncrementValidationError("candidate factor change count is missing")
        if expected != len(event_changed_symbols):
            raise IncrementValidationError("candidate factor change count does not match evidence")

    return {
        "schema_version": 1,
        "status": "increment_comparison_passed",
        "dataset_id": dataset_id,
        "price_basis": price_basis,
        "base_snapshot": {
            "snapshot_id": base.snapshot_id,
            "content_sha256": base.manifest.get("content_sha256"),
            "latest_trade_date": str(base.manifest.get("latest_trade_date")),
        },
        "candidate_snapshot": {
            "snapshot_id": candidate.snapshot_id,
            "content_sha256": candidate.manifest.get("content_sha256"),
            "latest_trade_date": str(candidate.manifest.get("latest_trade_date")),
        },
        "lineage": lineage,
        "checked_symbols": len(base_rows),
        "unchanged_history_symbols": unchanged_history,
        "factor_changed_symbol_count": len(event_changed_symbols),
        "factor_changed_symbols": sorted(event_changed_symbols),
        "factor_value_changed_symbol_count": len(factor_value_changed),
        "factor_value_changed_symbols": factor_value_changed,
        "factor_metadata_changed_symbol_count": len(factor_metadata_changed),
        "qfq_history_changed_symbol_count": len(qfq_history_changed),
        "qfq_history_changed_symbols": qfq_history_changed,
        "symbols_with_increment": symbols_with_increment,
        "symbols_without_increment": len(no_increment),
        "increment_rows": increment_rows,
        "errors": [],
    }


def _sample_order(symbols: Sequence[str], *, key: str) -> list[str]:
    return sorted(
        symbols,
        key=lambda symbol: hashlib.sha256(
            f"{key}\0{symbol}".encode("utf-8")
        ).hexdigest(),
    )


def _select_samples(
    symbols: Sequence[str],
    *,
    increment_symbols: Sequence[str],
    no_increment_symbols: Sequence[str],
    factor_changed_symbols: Sequence[str],
    sample_size: int,
    key: str,
) -> tuple[list[str], dict[str, list[str]]]:
    if isinstance(sample_size, bool) or sample_size < 1:
        raise IncrementValidationError("sample_size must be positive")
    limit = min(sample_size, len(symbols))
    selected: list[str] = []
    coverage: dict[str, list[str]] = {
        "factor_changed": [],
        "increment": [],
        "no_increment": [],
    }
    categories = (
        ("factor_changed", factor_changed_symbols),
        ("increment", increment_symbols),
        ("no_increment", no_increment_symbols),
    )
    for name, candidates in categories:
        ordered = _sample_order(tuple(set(candidates)), key=f"{key}\0{name}")
        if ordered and len(selected) < limit:
            choice = next((value for value in ordered if value not in selected), None)
            if choice is not None:
                selected.append(choice)
                coverage[name].append(choice)
    for symbol in _sample_order(symbols, key=key):
        if len(selected) >= limit:
            break
        if symbol not in selected:
            selected.append(symbol)
    for name, candidates in categories:
        coverage[name] = [value for value in selected if value in set(candidates)]
    return selected, coverage


def _bundle_frame(
    bundle: Path,
    row: Mapping[str, Any],
    field: str,
    hash_field: str,
    *,
    label: str,
) -> pd.DataFrame:
    path = _safe_child(bundle, row.get(field), label=label)
    if path.is_symlink() or not path.is_file():
        raise IncrementValidationError(f"{label} is unavailable")
    expected = row.get(hash_field)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise IncrementValidationError(f"{label} hash mismatch")
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise IncrementValidationError(f"{label} is unreadable") from exc


def _validate_comparison_result(
    comparison_result: Mapping[str, Any] | None,
    *,
    base: SnapshotView,
    candidate: SnapshotView,
    price_basis: str,
) -> Mapping[str, Any]:
    if not isinstance(comparison_result, Mapping):
        raise IncrementValidationError("full increment comparison report is required")
    if comparison_result.get("status") != "increment_comparison_passed":
        raise IncrementValidationError("full increment comparison did not pass")
    if comparison_result.get("dataset_id") != candidate.dataset_id:
        raise IncrementValidationError("increment comparison dataset identity does not match")
    if comparison_result.get("price_basis") != price_basis:
        raise IncrementValidationError("increment comparison price basis does not match")
    if comparison_result.get("errors") != []:
        raise IncrementValidationError("increment comparison report contains errors")

    expected = (
        ("base_snapshot", base),
        ("candidate_snapshot", candidate),
    )
    for field, snapshot in expected:
        identity = comparison_result.get(field)
        if not isinstance(identity, Mapping):
            raise IncrementValidationError(f"increment comparison {field} identity is missing")
        if identity.get("snapshot_id") != snapshot.snapshot_id:
            raise IncrementValidationError(f"increment comparison {field} identity does not match")
        if identity.get("content_sha256") != snapshot.manifest.get("content_sha256"):
            raise IncrementValidationError(f"increment comparison {field} content does not match")
        if str(identity.get("latest_trade_date")) != str(
            snapshot.manifest.get("latest_trade_date")
        ):
            raise IncrementValidationError(f"increment comparison {field} date does not match")
    if comparison_result.get("checked_symbols") != len(base.index):
        raise IncrementValidationError("increment comparison symbol coverage is incomplete")
    return comparison_result


def sample_increment_against_bundle(
    dataset_id: str,
    *,
    quant_data_root: Path,
    base_snapshot_id: str,
    bundle_dir: Path,
    candidate_snapshot_id: str | None = None,
    sample_size: int = 100,
    seed: str = "increment-check",
    comparison_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministically sample candidate increment rows against a local bundle."""

    root = Path(quant_data_root).expanduser().resolve()
    bundle = Path(bundle_dir).expanduser().resolve()
    base = _load_snapshot(root, dataset_id, base_snapshot_id)
    candidate = _load_snapshot(
        root,
        dataset_id,
        candidate_snapshot_id,
        latest=candidate_snapshot_id is None,
    )
    price_basis = str(candidate.manifest.get("price_basis"))
    if price_basis not in {"raw", "qfq"}:
        raise IncrementValidationError("increment sampling supports raw or qfq daily bars")
    comparison = _validate_comparison_result(
        comparison_result,
        base=base,
        candidate=candidate,
        price_basis=price_basis,
    )
    actual_comparison = compare_snapshot_increment(
        dataset_id,
        quant_data_root=root,
        base_snapshot_id=base.snapshot_id,
        candidate_snapshot_id=candidate.snapshot_id,
    )
    if dict(comparison) != actual_comparison:
        raise IncrementValidationError(
            "full increment comparison report does not match current snapshots"
        )
    comparison = actual_comparison
    base_latest = pd.Timestamp(base.manifest["latest_trade_date"]).normalize()
    bundle_manifest, bundle_index = _validate_bundle(bundle)
    if str(bundle_manifest.get("base_latest_trade_date")) != base_latest.date().isoformat():
        raise IncrementValidationError("bundle base date does not match the selected snapshot")
    if str(bundle_manifest.get("requested_through")) != str(candidate.manifest.get("latest_trade_date")):
        raise IncrementValidationError("bundle target date does not match the candidate snapshot")
    candidate_source = candidate.manifest.get("source")
    candidate_source = candidate_source if isinstance(candidate_source, Mapping) else {}
    declared_bundle_hash = candidate_source.get("source_bundle_sha256")
    actual_bundle_hash = sha256_file(bundle / "BUNDLE_MANIFEST.json")
    if not isinstance(declared_bundle_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_bundle_hash
    ):
        raise IncrementValidationError("candidate source bundle identity is missing")
    if declared_bundle_hash != actual_bundle_hash:
        raise IncrementValidationError("candidate source bundle identity does not match")

    candidate_rows = _rows_by_symbol(candidate.index)
    bundle_rows = _rows_by_symbol(bundle_index)
    if set(candidate_rows) != set(bundle_rows):
        raise IncrementValidationError("bundle and candidate symbol sets differ")
    increment_symbols = [
        symbol
        for symbol, row in candidate_rows.items()
        if pd.Timestamp(row["last_date"]).normalize() > base_latest
    ]
    no_increment_symbols = sorted(set(candidate_rows) - set(increment_symbols))
    factor_changed_symbols = [
        str(value) for value in comparison.get("factor_changed_symbols", [])
    ]
    selection_key = (
        f"{dataset_id}\0{base.snapshot_id}\0{candidate.snapshot_id}\0{seed}"
    )
    selected, coverage = _select_samples(
        tuple(candidate_rows),
        increment_symbols=increment_symbols,
        no_increment_symbols=no_increment_symbols,
        factor_changed_symbols=factor_changed_symbols,
        sample_size=sample_size,
        key=selection_key,
    )

    sampled_rows = 0
    for symbol in selected:
        candidate_frame = _read_indexed_frame(
            candidate,
            candidate_rows[symbol],
            label=f"candidate bars {symbol}",
        )
        candidate_new = candidate_frame.loc[
            candidate_frame["date"] > base_latest
        ].reset_index(drop=True)
        bundle_row = bundle_rows[symbol]
        raw = _normalized_frame(
            _bundle_frame(
                bundle,
                bundle_row,
                "raw_path",
                "raw_sha256",
                label=f"bundle raw {symbol}",
            ),
            symbol=symbol,
            label=f"bundle raw {symbol}",
        )
        if len(raw) != int(bundle_row["raw_rows"]):
            raise IncrementValidationError(f"bundle raw row count mismatch for {symbol}")
        if price_basis == "raw":
            expected = raw.loc[raw["date"] > base_latest].reset_index(drop=True)
        else:
            events = _bundle_frame(
                bundle,
                bundle_row,
                "factor_events_path",
                "factor_events_sha256",
                label=f"bundle factor events {symbol}",
            )
            provider_qfq = None
            provider_relative = bundle_row.get("qfq_provider_path")
            if isinstance(provider_relative, str) and provider_relative:
                provider_qfq = _bundle_frame(
                    bundle,
                    bundle_row,
                    "qfq_provider_path",
                    "qfq_provider_sha256",
                    label=f"bundle qfq audit {symbol}",
                )
            if raw.empty:
                expected = raw.copy()
            else:
                try:
                    factors = expand_daily_factors(
                        raw,
                        events,
                        provider_qfq=provider_qfq,
                    )
                    expected = build_local_qfq(raw, factors)
                except Exception as exc:
                    raise IncrementValidationError(
                        f"bundle qfq evidence cannot be rebuilt for {symbol}"
                    ) from exc
                expected = _normalized_frame(
                    expected, symbol=symbol, label=f"rebuilt qfq {symbol}"
                )
                expected = expected.loc[expected["date"] > base_latest].reset_index(drop=True)
        if not (candidate_new.empty and expected.empty) and not _frame_equal(
            candidate_new, expected
        ):
            raise IncrementValidationError(f"bundle values differ for {symbol}")
        _validate_new_rows(candidate_new, label=f"sampled increment {symbol}")
        sampled_rows += len(candidate_new)

    return {
        "schema_version": 1,
        "status": "increment_sample_passed",
        "dataset_id": dataset_id,
        "price_basis": price_basis,
        "base_snapshot_id": base.snapshot_id,
        "candidate_snapshot_id": candidate.snapshot_id,
        "bundle_sha256": actual_bundle_hash,
        "seed": seed,
        "sample_size_requested": sample_size,
        "sampled_symbol_count": len(selected),
        "unchecked_symbol_count": len(candidate_rows) - len(selected),
        "sampled_increment_rows": sampled_rows,
        "sampled_symbols": selected,
        "category_coverage": coverage,
        "network_requests": 0,
        "errors": [],
    }
