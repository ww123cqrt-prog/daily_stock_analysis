"""Qlib providers that read governed ``quant_data`` Parquet snapshots directly.

The adapter keeps Qlib's expression, dataset, model, and backtest layers.  It
only replaces the storage-facing calendar, instrument, and feature providers.
No Qlib ``features/*.bin`` market-data copy is produced.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from qlib.data.data import (
    CalendarProvider,
    FeatureProvider,
    InstrumentProvider,
    LocalProvider,
)

from quant_workflow.scripts.quant_data_contract import (
    DataContractError,
    ResolvedDataset,
    resolve_dataset,
    sha256_file,
)
from quant_workflow.scripts._robust_membership_integrity import (
    MembershipIntegrityError,
    membership_read_lock,
)


DESCRIPTOR_NAME = "PROVIDER_DESCRIPTOR.json"
CALENDAR_NAME = "CALENDAR.parquet"
MEMBERSHIP_NAME = "MEMBERSHIP_INTERVALS.parquet"
DESCRIPTOR_SCHEMA_VERSION = 1
ADAPTER_VERSION = "1"
DEFAULT_DATASET_ID = "ashare_daily_qfq"
DEFAULT_MEMBERSHIP_PATH = Path(
    "reference/pit_csi800/membership_intervals.parquet"
)
PIT_MEMBERSHIP_ROOT = DEFAULT_MEMBERSHIP_PATH.parent
SUPPORTED_FIELDS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "turn": "turn",
    "pctchg": "pctChg",
    "preclose": "preclose",
    "tradestatus": "tradestatus",
    "isst": "isST",
}
POOL_INDEX_CODES = {
    "csi300": frozenset({"000300.XSHG", "SH000300", "000300"}),
    "csi500": frozenset({"000905.XSHG", "SH000905", "000905"}),
}


class DirectProviderError(RuntimeError):
    """Raised when a direct provider cannot prove or read its frozen input."""


@dataclass(frozen=True)
class _DescriptorContext:
    quant_data_root: Path
    descriptor_path: Path
    descriptor: dict[str, Any]
    snapshot_path: Path
    manifest: dict[str, Any]
    file_index: pd.DataFrame
    calendar: pd.DatetimeIndex
    membership: pd.DataFrame


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectProviderError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise DirectProviderError(f"{label} is invalid")
    return payload


def _relative_path(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DirectProviderError(f"{label} must stay under quant_data") from exc


def _absolute_without_resolving(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _safe_child(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DirectProviderError(f"{label} must be a relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise DirectProviderError(f"{label} must be a relative path")
    root = Path(root).expanduser().resolve()
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise DirectProviderError(f"{label} contains a symlink")
    candidate = current.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DirectProviderError(f"{label} escapes quant_data") from exc
    return candidate


def _safe_snapshot_child(snapshot: Path, relative: object, label: str) -> Path:
    return _safe_child(snapshot, relative, label)


def _path_under_root(root: Path, path: str | Path, label: str) -> Path:
    root = Path(root).expanduser().resolve()
    unresolved = Path(path).expanduser()
    if unresolved.is_absolute():
        absolute = _absolute_without_resolving(unresolved)
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise DirectProviderError(f"{label} must stay under quant_data") from exc
    else:
        relative = unresolved
    return _safe_child(root, relative.as_posix(), label)


def _descriptor_file(path: str | Path, root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        unresolved = root / unresolved
    unresolved = _absolute_without_resolving(unresolved)
    try:
        relative = unresolved.relative_to(root)
    except ValueError as exc:
        raise DirectProviderError(
            "provider descriptor must stay under quant_data"
        ) from exc
    candidate = _safe_child(root, relative.as_posix(), "provider descriptor")
    if candidate.is_dir():
        candidate = _safe_child(
            root,
            (relative / DESCRIPTOR_NAME).as_posix(),
            "provider descriptor",
        )
    return candidate


def _validate_sha256(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DirectProviderError(f"{label} is invalid")
    return text


def _read_membership_bytes(payload: bytes) -> pd.DataFrame:
    try:
        membership = pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise DirectProviderError("historical membership data is unavailable") from exc
    return _normalize_membership(membership)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_file_hash(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise DirectProviderError(f"{label} hash mismatch")


def _read_file_index(path: Path) -> pd.DataFrame:
    try:
        index = pd.read_parquet(path)
    except Exception as exc:
        raise DirectProviderError("file index is unavailable") from exc
    required = {
        "symbol",
        "relative_path",
        "row_count",
        "first_date",
        "last_date",
        "sha256",
    }
    if index.empty or not required.issubset(index.columns):
        raise DirectProviderError("file index is invalid")
    normalized = index.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    if normalized["symbol"].duplicated().any():
        raise DirectProviderError("file index contains duplicate symbols")
    return normalized.sort_values("symbol").reset_index(drop=True)


def _read_calendar(path: Path) -> pd.DatetimeIndex:
    try:
        frame = pd.read_parquet(path, columns=["date"])
    except Exception as exc:
        raise DirectProviderError("direct provider calendar is unavailable") from exc
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce")).normalize()
    if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise DirectProviderError("direct provider calendar is invalid")
    return dates


def _normalize_membership(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "index_code", "start_date", "end_date"}
    if membership.empty or not required.issubset(membership.columns):
        raise DirectProviderError("historical membership data is invalid")
    result = membership.loc[:, sorted(required)].copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["index_code"] = result["index_code"].astype(str).str.upper()
    result["start_date"] = pd.to_datetime(result["start_date"], errors="coerce").dt.normalize()
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce").dt.normalize()
    if (
        result[["start_date", "end_date"]].isna().any().any()
        or (result["start_date"] > result["end_date"]).any()
    ):
        raise DirectProviderError("historical membership intervals are invalid")
    return result.sort_values(
        ["symbol", "index_code", "start_date", "end_date"]
    ).reset_index(drop=True)


def _read_membership(path: Path) -> pd.DataFrame:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise DirectProviderError("historical membership data is unavailable") from exc
    return _read_membership_bytes(payload)


def _validated_membership_input(
    root: Path,
    membership_path: str | Path | None,
) -> tuple[bytes, pd.DataFrame, str, str]:
    root = Path(root).expanduser().resolve()
    pit_dir = _safe_child(root, PIT_MEMBERSHIP_ROOT.as_posix(), "membership root")
    expected_file = _safe_snapshot_child(
        pit_dir,
        DEFAULT_MEMBERSHIP_PATH.name,
        "membership path",
    )
    requested_file = _path_under_root(
        root,
        membership_path or DEFAULT_MEMBERSHIP_PATH,
        "membership path",
    )
    if requested_file != expected_file:
        raise DirectProviderError("membership path is not the official pit_csi800 input")

    catalog_path = _safe_child(root, "DATA_MANIFEST.json", "quant_data catalog")
    catalog = _read_json(catalog_path, "quant_data catalog")
    official_entries = catalog.get("official_entries")
    entry = (
        official_entries.get("pit_csi800")
        if isinstance(official_entries, dict)
        else None
    )
    if not isinstance(entry, dict):
        raise DirectProviderError("official pit_csi800 membership is not registered")
    if (
        entry.get("path") != PIT_MEMBERSHIP_ROOT.as_posix()
        or entry.get("status") != "do_not_move"
    ):
        raise DirectProviderError("official pit_csi800 membership entry is invalid")
    expected_snapshot_hash = _validate_sha256(
        entry.get("membership_snapshot_hash"),
        "membership snapshot hash",
    )
    date_range = entry.get("date_range")
    if not isinstance(date_range, dict):
        raise DirectProviderError("official pit_csi800 membership date range is invalid")

    try:
        with membership_read_lock(pit_dir) as locked:
            locked.ensure_transaction_clean()
            source = locked.validate_source(
                start_date=str(date_range.get("start_date") or ""),
                end_date=str(date_range.get("end_date") or ""),
            )
            actual_snapshot_hash = locked.fingerprint()
            if actual_snapshot_hash != expected_snapshot_hash:
                raise DirectProviderError(
                    "official pit_csi800 membership fingerprint is stale"
                )
            if entry.get("selected_source") != source.get("selected"):
                raise DirectProviderError(
                    "official pit_csi800 membership source is stale"
                )
            payload = expected_file.read_bytes()
    except MembershipIntegrityError as exc:
        raise DirectProviderError(
            f"official pit_csi800 membership is unavailable: {exc.reason_code}"
        ) from exc
    except OSError as exc:
        raise DirectProviderError(
            "official pit_csi800 membership is unavailable"
        ) from exc

    membership = _read_membership_bytes(payload)
    file_hash = hashlib.sha256(payload).hexdigest()
    return payload, membership, file_hash, actual_snapshot_hash


def _load_descriptor(
    descriptor_path: str | Path,
    quant_data_root: str | Path,
) -> _DescriptorContext:
    root = Path(quant_data_root).expanduser().resolve()
    descriptor_file = _descriptor_file(descriptor_path, root)
    if not descriptor_file.is_file() or descriptor_file.is_symlink():
        raise DirectProviderError("provider descriptor is unavailable")
    payload = _read_json(descriptor_file, "provider descriptor")
    if (
        payload.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION
        or payload.get("adapter") != "quant_data_qlib_provider"
        or payload.get("adapter_version") != ADAPTER_VERSION
        or payload.get("storage_mode") != "direct_parquet"
    ):
        raise DirectProviderError("provider descriptor contract is invalid")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise DirectProviderError("provider dataset identity is missing")
    snapshot = _safe_child(root, dataset.get("snapshot_path"), "snapshot_path")
    if not snapshot.is_dir():
        raise DirectProviderError("frozen snapshot is unavailable")
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    _validate_file_hash(
        manifest_path,
        dataset.get("manifest_sha256"),
        "snapshot manifest",
    )
    manifest = _read_json(manifest_path, "snapshot manifest")
    expected_identity = {
        "dataset_id": dataset.get("dataset_id"),
        "snapshot_id": dataset.get("snapshot_id"),
        "content_sha256": dataset.get("content_sha256"),
        "price_basis": "qfq",
        "status": "ready",
    }
    for field, expected in expected_identity.items():
        if not expected or manifest.get(field) != expected:
            raise DirectProviderError(f"frozen snapshot {field} mismatch")
    if manifest.get("data_kind") not in {"daily_bars", "paired_daily_bars"}:
        raise DirectProviderError("frozen snapshot is not a daily-bar dataset")

    index_path = _safe_snapshot_child(
        snapshot,
        dataset.get("file_index_path"),
        "file_index_path",
    )
    _validate_file_hash(index_path, dataset.get("file_index_sha256"), "file index")
    file_index = _read_file_index(index_path)

    calendar_spec = payload.get("calendar")
    if not isinstance(calendar_spec, dict):
        raise DirectProviderError("calendar descriptor is missing")
    calendar_path = _safe_snapshot_child(
        descriptor_file.parent,
        calendar_spec.get("path"),
        "calendar path",
    )
    _validate_file_hash(calendar_path, calendar_spec.get("sha256"), "calendar")
    calendar = _read_calendar(calendar_path)
    if (
        int(calendar_spec.get("row_count") or -1) != len(calendar)
        or calendar_spec.get("start") != calendar[0].date().isoformat()
        or calendar_spec.get("end") != calendar[-1].date().isoformat()
    ):
        raise DirectProviderError("calendar descriptor is stale")

    membership_spec = payload.get("membership")
    if not isinstance(membership_spec, dict):
        raise DirectProviderError("historical membership descriptor is missing")
    membership_path = _safe_snapshot_child(
        descriptor_file.parent,
        membership_spec.get("path"),
        "membership path",
    )
    _validate_file_hash(
        membership_path,
        membership_spec.get("sha256"),
        "historical membership",
    )
    membership = _read_membership(membership_path)
    if int(membership_spec.get("row_count") or -1) != len(membership):
        raise DirectProviderError("historical membership descriptor is stale")
    _validate_sha256(
        membership_spec.get("snapshot_hash"),
        "membership snapshot hash",
    )

    return _DescriptorContext(
        quant_data_root=root,
        descriptor_path=descriptor_file,
        descriptor=payload,
        snapshot_path=snapshot,
        manifest=manifest,
        file_index=file_index,
        calendar=calendar,
        membership=membership,
    )


def _calendar_from_snapshot(snapshot: Path, file_index: pd.DataFrame) -> pd.DatetimeIndex:
    day_numbers: set[int] = set()
    for row in file_index.itertuples(index=False):
        path = _safe_snapshot_child(snapshot, row.relative_path, "bar path")
        try:
            dates = pd.read_parquet(path, columns=["date"])["date"]
        except Exception as exc:
            raise DirectProviderError(
                f"cannot derive calendar from {row.symbol}"
            ) from exc
        parsed = pd.to_datetime(dates, errors="coerce")
        if parsed.isna().any():
            raise DirectProviderError(f"invalid dates in {row.symbol}")
        days = parsed.to_numpy(dtype="datetime64[D]").astype(np.int64)
        day_numbers.update(int(value) for value in days)
    if not day_numbers:
        raise DirectProviderError("cannot derive an empty trading calendar")
    dates = np.asarray(sorted(day_numbers), dtype="datetime64[D]")
    return pd.DatetimeIndex(dates).normalize()


def _membership_summary(membership: pd.DataFrame) -> dict[str, Any]:
    pools: dict[str, dict[str, Any]] = {}
    for pool, codes in POOL_INDEX_CODES.items():
        selected = membership[membership["index_code"].isin(codes)]
        if selected.empty:
            raise DirectProviderError(f"historical membership has no {pool} rows")
        pools[pool] = {
            "row_count": int(len(selected)),
            "symbol_count": int(selected["symbol"].nunique()),
            "coverage_start": selected["start_date"].min().date().isoformat(),
            "coverage_end": selected["end_date"].max().date().isoformat(),
        }
    pools["csi800"] = {
        "row_count": int(len(membership)),
        "symbol_count": int(membership["symbol"].nunique()),
        "coverage_start": membership["start_date"].min().date().isoformat(),
        "coverage_end": membership["end_date"].max().date().isoformat(),
    }
    return pools


def _current_dataset_identity(
    root: Path,
    dataset_id: str,
) -> tuple[dict[str, Any], Path]:
    catalog = _read_json(root / "DATA_MANIFEST.json", "quant_data catalog")
    if catalog.get("schema_version") != 2:
        raise DirectProviderError("quant_data catalog schema is unsupported")
    datasets = catalog.get("datasets")
    entry = datasets.get(dataset_id) if isinstance(datasets, dict) else None
    if not isinstance(entry, dict):
        raise DirectProviderError(f"dataset is not registered: {dataset_id}")
    if entry.get("status") != "ready" or entry.get("price_basis") != "qfq":
        raise DirectProviderError("formal qfq dataset is not ready")
    pointer_path = _safe_child(root, entry.get("latest_pointer"), "latest pointer")
    pointer = _read_json(pointer_path, "latest pointer")
    for field in ("dataset_id", "snapshot_id"):
        if pointer.get(field) != entry.get(field):
            raise DirectProviderError("catalog and latest pointer are out of sync")
    if pointer.get("manifest_sha256") != entry.get("snapshot_manifest_sha256"):
        raise DirectProviderError("catalog and latest pointer are out of sync")
    snapshot = _safe_child(root, pointer.get("snapshot_path"), "snapshot path")
    if snapshot.name != str(entry.get("snapshot_id") or ""):
        raise DirectProviderError("latest pointer snapshot path is invalid")
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    _validate_file_hash(
        manifest_path,
        pointer.get("manifest_sha256"),
        "latest snapshot manifest",
    )
    manifest = _read_json(manifest_path, "latest snapshot manifest")
    for field in (
        "dataset_id",
        "snapshot_id",
        "content_sha256",
        "price_basis",
        "status",
    ):
        if manifest.get(field) != entry.get(field):
            raise DirectProviderError(f"catalog and snapshot disagree on {field}")
    return dict(entry), snapshot


def _descriptor_payload(
    *,
    root: Path,
    resolved: ResolvedDataset,
    file_index_path: Path,
    calendar_path: Path,
    calendar: pd.DatetimeIndex,
    membership_path: Path,
    membership: pd.DataFrame,
    membership_snapshot_hash: str,
    membership_source_path: str,
) -> dict[str, Any]:
    manifest_path = resolved.snapshot_path / "DATASET_MANIFEST.json"
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "adapter": "quant_data_qlib_provider",
        "adapter_version": ADAPTER_VERSION,
        "storage_mode": "direct_parquet",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "dataset_id": resolved.dataset_id,
            "snapshot_id": resolved.manifest["snapshot_id"],
            "snapshot_path": _relative_path(
                resolved.snapshot_path,
                root,
                "snapshot_path",
            ),
            "content_sha256": resolved.manifest["content_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "price_basis": resolved.manifest["price_basis"],
            "latest_trade_date": resolved.manifest["latest_trade_date"],
            "file_index_path": file_index_path.relative_to(
                resolved.snapshot_path
            ).as_posix(),
            "file_index_sha256": sha256_file(file_index_path),
        },
        "calendar": {
            "path": calendar_path.name,
            "sha256": sha256_file(calendar_path),
            "row_count": int(len(calendar)),
            "start": calendar[0].date().isoformat(),
            "end": calendar[-1].date().isoformat(),
            "derivation": "union_of_frozen_bar_dates",
        },
        "membership": {
            "path": membership_path.name,
            "source_path": membership_source_path,
            "sha256": sha256_file(membership_path),
            "snapshot_hash": membership_snapshot_hash,
            "row_count": int(len(membership)),
            "pools": _membership_summary(membership),
        },
        "field_contract": {
            "price_input": "already_qfq",
            "factor_semantics": "unit_on_available_bar_no_second_adjustment",
            "missing_date_semantics": "nan_no_forward_fill",
            "supported_fields": [
                "$open",
                "$high",
                "$low",
                "$close",
                "$volume",
                "$amount",
                "$turn",
                "$pctChg",
                "$preclose",
                "$tradestatus",
                "$isST",
                "$factor",
                "$vwap",
            ],
        },
    }


def build_direct_provider_descriptor(
    *,
    quant_data_root: str | Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    membership_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Freeze a metadata-only Qlib view of the current formal Parquet snapshot."""

    root = Path(quant_data_root).expanduser().resolve()
    (
        membership_bytes,
        membership,
        membership_file_hash,
        membership_snapshot_hash,
    ) = _validated_membership_input(root, membership_path)

    base = _path_under_root(
        root,
        output_root
        or Path("runtime") / "qlib_direct" / "providers",
        "direct provider output",
    )
    current_entry, current_snapshot = _current_dataset_identity(root, dataset_id)
    provider_dir = (
        base
        / dataset_id
        / str(current_entry["snapshot_id"])
        / membership_snapshot_hash
    )
    descriptor_path = provider_dir / DESCRIPTOR_NAME
    if provider_dir.exists():
        context = _load_descriptor(descriptor_path, root)
        if (
            context.snapshot_path != current_snapshot
            or context.manifest["content_sha256"]
            != current_entry["content_sha256"]
            or context.descriptor["membership"]["snapshot_hash"]
            != membership_snapshot_hash
            or context.descriptor["membership"]["sha256"]
            != membership_file_hash
        ):
            raise DirectProviderError(
                "existing direct provider descriptor has a conflicting identity"
            )
        return descriptor_path

    try:
        resolved = resolve_dataset(
            dataset_id,
            quant_data_root=root,
            require_basis="qfq",
        )
    except DataContractError as exc:
        raise DirectProviderError(f"formal qfq dataset is unavailable: {exc}") from exc
    if resolved.manifest.get("data_kind") != "daily_bars":
        raise DirectProviderError("direct A-share provider requires daily_bars")
    if (
        resolved.snapshot_path != current_snapshot
        or resolved.manifest.get("content_sha256") != current_entry.get("content_sha256")
    ):
        raise DirectProviderError("formal qfq dataset changed during descriptor creation")
    index_path = _safe_snapshot_child(
        resolved.snapshot_path,
        resolved.manifest.get("file_index_path"),
        "file_index_path",
    )
    file_index = _read_file_index(index_path)

    provider_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = provider_dir.parent / f".{provider_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        calendar = _calendar_from_snapshot(resolved.snapshot_path, file_index)
        calendar_path = staging / CALENDAR_NAME
        pd.DataFrame({"date": calendar}).to_parquet(calendar_path, index=False)
        frozen_membership_path = staging / MEMBERSHIP_NAME
        frozen_membership_path.write_bytes(membership_bytes)
        payload = _descriptor_payload(
            root=root,
            resolved=resolved,
            file_index_path=index_path,
            calendar_path=calendar_path,
            calendar=calendar,
            membership_path=frozen_membership_path,
            membership=membership,
            membership_snapshot_hash=membership_snapshot_hash,
            membership_source_path=DEFAULT_MEMBERSHIP_PATH.as_posix(),
        )
        _atomic_json(staging / DESCRIPTOR_NAME, payload)
        try:
            os.replace(staging, provider_dir)
        except OSError:
            if not provider_dir.is_dir():
                raise
        context = _load_descriptor(descriptor_path, root)
        if (
            context.snapshot_path != current_snapshot
            or context.manifest["content_sha256"]
            != current_entry["content_sha256"]
            or context.descriptor["membership"]["snapshot_hash"]
            != membership_snapshot_hash
            or context.descriptor["membership"]["sha256"]
            != membership_file_hash
        ):
            raise DirectProviderError(
                "direct provider race produced a conflicting identity"
            )
        return descriptor_path
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if symbol.startswith(("SH", "SZ", "BJ")) and symbol[2:].isdigit():
        return symbol
    normalized = symbol.replace(".", "")
    for suffix, prefix in (("XSHG", "SH"), ("XSHE", "SZ"), ("XBSE", "BJ")):
        if normalized.endswith(suffix) and normalized[: -len(suffix)].isdigit():
            return prefix + normalized[: -len(suffix)]
    for suffix, prefix in (("SH", "SH"), ("SZ", "SZ"), ("BJ", "BJ")):
        if normalized.endswith(suffix) and normalized[: -len(suffix)].isdigit():
            return prefix + normalized[: -len(suffix)]
    raise DirectProviderError(f"unsupported instrument code: {value}")


def _merge_spans(spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    one_day = pd.Timedelta(days=1)
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + one_day:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


class QuantDataCalendarProvider(CalendarProvider):
    """Expose the frozen snapshot's derived day calendar to Qlib."""

    def __init__(self, *, descriptor_path: str | Path, quant_data_root: str | Path):
        super().__init__()
        self._context = _load_descriptor(descriptor_path, quant_data_root)

    @property
    def snapshot_path(self) -> Path:
        return self._context.snapshot_path

    def load_calendar(self, freq, future):
        if str(freq).lower() not in {"day", "1d"}:
            raise DirectProviderError("direct provider currently supports day frequency only")
        if future:
            raise DirectProviderError("direct provider has no unverified future calendar")
        return list(self._context.calendar)


class QuantDataInstrumentProvider(InstrumentProvider):
    """Map Qlib stock-pool requests to quant_data point-in-time intervals."""

    def __init__(self, *, descriptor_path: str | Path, quant_data_root: str | Path):
        super().__init__()
        self._context = _load_descriptor(descriptor_path, quant_data_root)
        self._all = {
            str(row.symbol): [
                (
                    pd.Timestamp(row.first_date).normalize(),
                    pd.Timestamp(row.last_date).normalize(),
                )
            ]
            for row in self._context.file_index.itertuples(index=False)
        }

    @property
    def snapshot_path(self) -> Path:
        return self._context.snapshot_path

    def _pool(self, market: str) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        market = market.lower()
        if market == "all":
            return self._all
        if market not in {"csi300", "csi500", "csi800"}:
            raise DirectProviderError(f"unsupported stock pool: {market}")
        membership = self._context.membership
        if market == "csi800":
            codes = POOL_INDEX_CODES["csi300"] | POOL_INDEX_CODES["csi500"]
        else:
            codes = POOL_INDEX_CODES[market]
        selected = membership[membership["index_code"].isin(codes)]
        result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for symbol, rows in selected.groupby("symbol", sort=True):
            result[str(symbol)] = _merge_spans(
                list(zip(rows["start_date"], rows["end_date"]))
            )
        return result

    def _check_membership_coverage(
        self,
        market: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> None:
        if market == "all":
            return
        pools = self._context.descriptor["membership"]["pools"]
        coverage = pools[market]
        coverage_start = pd.Timestamp(coverage["coverage_start"])
        coverage_end = pd.Timestamp(coverage["coverage_end"])
        if start_time < coverage_start:
            raise DirectProviderError(
                f"{market} membership coverage starts at {coverage_start.date()}"
            )
        if end_time > coverage_end:
            raise DirectProviderError(
                f"{market} membership coverage ends at {coverage_end.date()}"
            )

    def list_instruments(
        self,
        instruments,
        start_time=None,
        end_time=None,
        freq="day",
        as_list=False,
    ):
        if str(freq).lower() not in {"day", "1d"}:
            raise DirectProviderError("direct provider currently supports day frequency only")
        calendar = self._context.calendar
        start = pd.Timestamp(start_time or calendar[0]).normalize()
        end = pd.Timestamp(end_time or calendar[-1]).normalize()
        if start > end:
            raise DirectProviderError("instrument request start is after end")

        filter_pipe: list[Mapping[str, Any]] = []
        market = "all"
        if isinstance(instruments, Mapping) and "market" in instruments:
            market = str(instruments["market"]).lower()
            filter_pipe = list(instruments.get("filter_pipe") or [])
            base = self._pool(market)
            self._check_membership_coverage(market, start, end)
        elif isinstance(instruments, Mapping):
            base = {
                _normalize_symbol(symbol): [
                    (pd.Timestamp(span[0]).normalize(), pd.Timestamp(span[1]).normalize())
                    for span in spans
                ]
                for symbol, spans in instruments.items()
            }
        elif isinstance(instruments, (list, tuple, pd.Index, np.ndarray)):
            base = {
                symbol: self._all[symbol]
                for symbol in (_normalize_symbol(item) for item in instruments)
                if symbol in self._all
            }
        else:
            raise DirectProviderError("unsupported instruments request")

        clipped = {
            symbol: [
                (max(start, span_start), min(end, span_end))
                for span_start, span_end in spans
                if max(start, span_start) <= min(end, span_end)
            ]
            for symbol, spans in base.items()
        }
        clipped = {symbol: spans for symbol, spans in clipped.items() if spans}
        for config in filter_pipe:
            if not isinstance(config, Mapping) or not config.get("filter_type"):
                raise DirectProviderError("instrument filter config is invalid")
            from qlib.data import filter as qlib_filter

            filter_type = getattr(qlib_filter, str(config["filter_type"]), None)
            if filter_type is None:
                raise DirectProviderError("instrument filter type is unsupported")
            clipped = filter_type.from_config(dict(config))(
                clipped,
                start,
                end,
                freq,
            )
        if as_list:
            return sorted(clipped)
        return dict(sorted(clipped.items()))


class QuantDataFeatureProvider(FeatureProvider):
    """Read one frozen Parquet symbol file and align it to Qlib's calendar."""

    def __init__(
        self,
        *,
        descriptor_path: str | Path,
        quant_data_root: str | Path,
        max_cached_instruments: int = 8,
    ):
        super().__init__()
        self._context = _load_descriptor(descriptor_path, quant_data_root)
        self._rows = {
            str(row.symbol): row
            for row in self._context.file_index.itertuples(index=False)
        }
        self._cache_limit = max(1, int(max_cached_instruments))
        self._frames: OrderedDict[str, pd.DataFrame] = OrderedDict()

    @property
    def snapshot_path(self) -> Path:
        return self._context.snapshot_path

    def _load_frame(self, symbol: str) -> pd.DataFrame:
        cached = self._frames.get(symbol)
        if cached is not None:
            self._frames.move_to_end(symbol)
            return cached
        row = self._rows.get(symbol)
        if row is None:
            raise DirectProviderError(f"instrument is not in frozen snapshot: {symbol}")
        path = _safe_snapshot_child(
            self._context.snapshot_path,
            row.relative_path,
            "bar path",
        )
        _validate_file_hash(path, row.sha256, f"bar file {symbol}")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DirectProviderError(f"cannot read bar file {symbol}") from exc
        if "date" not in frame.columns:
            raise DirectProviderError(f"bar file has no date column: {symbol}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce")).normalize()
        if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise DirectProviderError(f"bar dates are invalid: {symbol}")
        loaded = frame.drop(columns=["date"]).copy()
        loaded.index = dates
        self._frames[symbol] = loaded
        self._frames.move_to_end(symbol)
        while len(self._frames) > self._cache_limit:
            self._frames.popitem(last=False)
        return loaded

    def feature(self, instrument, field, start_index, end_index, freq):
        if str(freq).lower() not in {"day", "1d"}:
            raise DirectProviderError("direct provider currently supports day frequency only")
        symbol = _normalize_symbol(instrument)
        calendar = self._context.calendar
        start = 0 if start_index is None else int(start_index)
        end = len(calendar) - 1 if end_index is None else int(end_index)
        start = max(0, start)
        end = min(len(calendar) - 1, end)
        if start > end:
            return pd.Series(dtype=np.float32)

        frame = self._load_frame(symbol)
        dates = calendar[start : end + 1]
        field_name = str(field).lstrip("$").lower()
        if field_name == "factor":
            values = pd.Series(
                np.where(dates.isin(frame.index), 1.0, np.nan),
                index=dates,
                dtype=float,
            )
        elif field_name == "vwap":
            if not {"amount", "volume"}.issubset(frame.columns):
                raise DirectProviderError(f"$vwap inputs are missing for {symbol}")
            aligned = frame.reindex(dates)
            amount = pd.to_numeric(aligned["amount"], errors="coerce")
            volume = pd.to_numeric(aligned["volume"], errors="coerce")
            values = amount.div(volume.where(volume != 0))
        else:
            source_field = SUPPORTED_FIELDS.get(field_name)
            if source_field is None:
                raise DirectProviderError(f"unsupported Qlib field: {field}")
            if source_field not in frame.columns:
                raise DirectProviderError(
                    f"field {source_field} is missing for {symbol}"
                )
            values = pd.to_numeric(
                frame[source_field].reindex(dates),
                errors="coerce",
            )
        return pd.Series(
            values.to_numpy(dtype=np.float32),
            index=np.arange(start, end + 1),
            dtype=np.float32,
        )


class QuantDataProvider(LocalProvider):
    """Qlib's public ``D`` facade backed by the direct storage providers."""


def qlib_direct_init_config(
    *,
    descriptor_path: str | Path,
    quant_data_root: str | Path,
    kernels: int | None = None,
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    descriptor = _descriptor_file(descriptor_path, root)
    _load_descriptor(descriptor, root)
    provider_kwargs = {
        "descriptor_path": str(descriptor),
        "quant_data_root": str(root),
    }
    config: dict[str, Any] = {
        "provider_uri": str(descriptor.parent),
        "region": "cn",
        "calendar_provider": {
            "class": "QuantDataCalendarProvider",
            "module_path": "quant_workflow.quant_data_qlib_provider",
            "kwargs": provider_kwargs,
        },
        "instrument_provider": {
            "class": "QuantDataInstrumentProvider",
            "module_path": "quant_workflow.quant_data_qlib_provider",
            "kwargs": provider_kwargs,
        },
        "feature_provider": {
            "class": "QuantDataFeatureProvider",
            "module_path": "quant_workflow.quant_data_qlib_provider",
            "kwargs": provider_kwargs,
        },
        "pit_provider": None,
        "expression_provider": {
            "class": "LocalExpressionProvider",
            "module_path": "qlib.data.data",
            "kwargs": {"time2idx": True},
        },
        "dataset_provider": {
            "class": "LocalDatasetProvider",
            "module_path": "qlib.data.data",
            "kwargs": {"align_time": True},
        },
        "provider": QuantDataProvider,
        "calendar_cache": None,
        "expression_cache": None,
        "dataset_cache": None,
        "default_disk_cache": 0,
    }
    if kernels is not None:
        config["kernels"] = max(1, int(kernels))
    return config


def init_qlib_from_quant_data(
    *,
    descriptor_path: str | Path,
    quant_data_root: str | Path,
    kernels: int | None = None,
    **overrides: Any,
) -> None:
    """Initialize Qlib without converting Parquet market data to ``.bin``."""

    import qlib

    config = qlib_direct_init_config(
        descriptor_path=descriptor_path,
        quant_data_root=quant_data_root,
        kernels=kernels,
    )
    config.update(overrides)
    qlib.init(default_conf="client", **config)
