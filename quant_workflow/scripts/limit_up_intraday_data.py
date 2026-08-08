"""Immutable data contract for historical limit-up minute-bar snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from limit_up_intraday_contract import normalize_minute_bars  # noqa: E402


DATASET_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
MANIFEST_NAME = "DATASET_MANIFEST.json"
MANIFEST_PATHS = {
    "events_path": "events.parquet",
    "recall_audit_path": "recall_audit.parquet",
    "market_state_path": "market_state.parquet",
    "file_index_path": "FILE_INDEX.parquet",
}
FORBIDDEN_MODEL_FEATURES = [
    "retrieval_ever_touched_limit",
    "final_status",
    "next_open",
    "next_close",
]
EVENT_COLUMNS = {
    "symbol",
    "date",
    "relative_path",
    "retrieval_ever_touched_limit",
    "final_status",
    "listing_date",
    "is_st",
    "board",
    "listing_age_sessions",
    "next_date",
    "next_open",
    "next_close",
}
RECALL_COLUMNS = {
    "date",
    "decision_time",
    "status",
    "expected_count",
    "recovered_count",
    "missing_symbols",
}
MARKET_STATE_COLUMNS = {
    "date",
    "decision_time",
    "sealed_count",
    "broken_board_rate",
    "max_consecutive_limit_days",
    "index_trend_20d",
    "advance_decline_ratio",
    "turnover_activity",
    "regime",
}


class IntradayDataBlocked(ValueError):
    """The immutable input does not satisfy formal research requirements."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("intraday snapshot manifest is unavailable") from exc
    if not isinstance(payload, dict):
        raise ValueError("intraday snapshot manifest is invalid")
    return payload


def _contained_path(snapshot: Path, relative_value: object) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("snapshot path escapes its root")
    candidate = snapshot / relative
    if candidate.is_symlink():
        raise ValueError("snapshot files must not be symbolic links")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(snapshot.resolve())
    except ValueError as exc:
        raise ValueError("snapshot path escapes its root") from exc
    return resolved


def resolve_snapshot_member(snapshot: Path, relative_value: object) -> Path:
    """Resolve one declared snapshot member without allowing root escape."""

    return _contained_path(Path(snapshot).resolve(), relative_value)


def _read_events(snapshot: Path) -> pd.DataFrame:
    path = _contained_path(snapshot, "events.parquet")
    try:
        events = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError("intraday events index is unavailable") from exc
    missing = EVENT_COLUMNS - set(events.columns)
    if events.empty or missing:
        raise ValueError("intraday events index is invalid")
    result = events.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["listing_date"] = pd.to_datetime(
        result["listing_date"], errors="coerce"
    ).dt.normalize()
    if result[["date", "listing_date"]].isna().any().any():
        raise ValueError("intraday event dates are invalid")
    if result[["symbol", "date"]].duplicated().any():
        raise ValueError("intraday events contain duplicates")
    if not result["retrieval_ever_touched_limit"].astype(bool).all():
        raise ValueError("intraday events must be exact touched-limit days")
    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _read_recall(
    snapshot: Path,
    event_dates: pd.Series | None = None,
    *,
    allowed_decision_times: tuple[str, ...] = ("14:00", "14:30"),
) -> pd.DataFrame:
    path = _contained_path(snapshot, "recall_audit.parquet")
    try:
        audit = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError("candidate recall audit is unavailable") from exc
    if audit.empty or not RECALL_COLUMNS.issubset(audit.columns):
        raise ValueError("candidate recall audit is invalid")
    result = audit.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if (
        result["date"].isna().any()
        or result[["date", "decision_time"]].duplicated().any()
        or not result["decision_time"].astype(str).isin(allowed_decision_times).all()
    ):
        raise ValueError("candidate recall audit keys are invalid")
    if event_dates is not None:
        expected_dates = pd.to_datetime(event_dates, errors="coerce").dt.normalize()
        if expected_dates.isna().any() or not set(result["date"]).issubset(set(expected_dates)):
            raise ValueError("recall audit date is outside event coverage")
    if not result["status"].astype(str).eq("passed").all():
        raise IntradayDataBlocked("candidate recall audit did not pass")
    expected = pd.to_numeric(result["expected_count"], errors="coerce")
    recovered = pd.to_numeric(result["recovered_count"], errors="coerce")
    if (
        expected.isna().any()
        or recovered.isna().any()
        or (expected < 0).any()
        or (recovered < 0).any()
        or not np.array_equal(expected.to_numpy(), np.floor(expected.to_numpy()))
        or not np.array_equal(recovered.to_numpy(), np.floor(recovered.to_numpy()))
    ):
        raise ValueError("candidate recall audit counts are invalid")

    def missing_symbols_are_empty(value: object) -> bool:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return False
        return isinstance(value, list) and not value

    if (
        not np.array_equal(expected.to_numpy(), recovered.to_numpy())
        or not result["missing_symbols"].map(missing_symbols_are_empty).all()
        or float(expected.sum()) <= 0
    ):
        raise IntradayDataBlocked("candidate recall audit is incomplete")
    result["expected_count"] = expected.astype(int)
    result["recovered_count"] = recovered.astype(int)
    return result


def _read_market_state(snapshot: Path, event_dates: pd.Series) -> pd.DataFrame:
    path = _contained_path(snapshot, "market_state.parquet")
    try:
        state = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError("point-in-time market state is unavailable") from exc
    if state.empty or not MARKET_STATE_COLUMNS.issubset(state.columns):
        raise ValueError("point-in-time market state is invalid")
    result = state.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any() or result[["date", "decision_time"]].duplicated().any():
        raise ValueError("point-in-time market state keys are invalid")
    if not result["decision_time"].isin(["14:00", "14:30"]).all():
        raise ValueError("point-in-time market state has an invalid decision time")
    expected = {
        (pd.Timestamp(date).normalize(), decision_time)
        for date in event_dates
        for decision_time in ("14:00", "14:30")
    }
    actual = set(zip(result["date"], result["decision_time"], strict=True))
    if not expected.issubset(actual):
        raise ValueError("point-in-time market state does not cover every event date")
    numeric = [
        "sealed_count",
        "broken_board_rate",
        "max_consecutive_limit_days",
        "index_trend_20d",
        "advance_decline_ratio",
        "turnover_activity",
    ]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("point-in-time market state contains invalid values")
    if not result["regime"].astype(str).isin(["hot", "normal", "retreat"]).all():
        raise ValueError("point-in-time market state has an invalid regime")
    return result.sort_values(["date", "decision_time"], kind="stable")


def _build_file_index(snapshot: Path, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    declared_paths: set[str] = set()
    for event in events.itertuples(index=False):
        relative = Path(str(event.relative_path)).as_posix()
        if relative in declared_paths:
            raise ValueError("multiple events reference the same minute-bar file")
        declared_paths.add(relative)
        path = _contained_path(snapshot, relative)
        if not path.is_file():
            raise ValueError(f"minute-bar file is missing: {relative}")
        bars = normalize_minute_bars(pd.read_parquet(path))
        event_date = pd.Timestamp(event.date).normalize()
        if pd.Timestamp(bars.iloc[0]["datetime"]).normalize() != event_date:
            raise ValueError(f"minute-bar date mismatch: {relative}")
        if not (bars["high"] >= bars["high_limit"] - 0.005).any():
            raise ValueError(f"event never touched its limit price: {relative}")
        rows.append(
            {
                "symbol": str(event.symbol),
                "date": str(event_date.date()),
                "relative_path": relative,
                "row_count": int(len(bars)),
                "first_timestamp": pd.Timestamp(bars.iloc[0]["datetime"]).isoformat(),
                "last_timestamp": pd.Timestamp(bars.iloc[-1]["datetime"]).isoformat(),
                "sha256": sha256_file(path),
            }
        )

    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in (snapshot / "bars").rglob("*.parquet")
        if path.is_file()
    }
    if actual_paths != declared_paths:
        raise ValueError("minute-bar files do not match the events index")
    return pd.DataFrame(rows).sort_values(["date", "symbol"], kind="stable")


def _content_hash(snapshot: Path, manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in snapshot.rglob("*")
        if item.is_file() and item.name != MANIFEST_NAME
    ):
        digest.update(path.relative_to(snapshot).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    stable = dict(manifest)
    stable.pop("content_sha256", None)
    digest.update(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def validate_intraday_snapshot(
    snapshot: Path,
    *,
    required_start: str | None = None,
    required_end: str | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Validate identity, recall evidence, coverage, and content hashes."""

    snapshot = Path(snapshot).resolve()
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("intraday snapshot is unavailable")
    manifest = _read_json(snapshot / MANIFEST_NAME)
    required = {
        "schema_version",
        "dataset_id",
        "snapshot_id",
        "data_kind",
        "status",
        "source",
        "date_range",
        "events_path",
        "recall_audit_path",
        "market_state_path",
        "file_index_path",
        "content_sha256",
        "decision_times",
    }
    if not required.issubset(manifest):
        raise ValueError("intraday snapshot manifest is incomplete")
    if manifest["status"] != "ready" or manifest["data_kind"] != "limit_up_intraday_events":
        raise IntradayDataBlocked("intraday snapshot is not ready research input")
    for field, expected_path in MANIFEST_PATHS.items():
        if manifest.get(field) != expected_path:
            raise ValueError(f"{field} must be {expected_path}")
    decision_times = manifest.get("decision_times")
    if (
        not isinstance(decision_times, list)
        or len(decision_times) != 2
        or set(decision_times) != {"14:00", "14:30"}
    ):
        raise ValueError("intraday snapshot decision times are not locked")

    events = _read_events(snapshot)
    _read_recall(
        snapshot,
        events["date"],
        allowed_decision_times=tuple(decision_times),
    )
    _read_market_state(snapshot, events["date"])
    index_path = _contained_path(snapshot, manifest["file_index_path"])
    try:
        index = pd.read_parquet(index_path)
    except Exception as exc:
        raise ValueError("intraday file index is unavailable") from exc
    expected_index = _build_file_index(snapshot, events).reset_index(drop=True)
    if len(index) != len(expected_index):
        raise ValueError("intraday file index is stale")
    actual = index.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    for expected_row, actual_row in zip(
        expected_index.itertuples(index=False), actual.itertuples(index=False), strict=True
    ):
        if (
            str(expected_row.symbol) != str(actual_row.symbol)
            or str(expected_row.date) != str(actual_row.date)
            or str(expected_row.relative_path) != str(actual_row.relative_path)
            or int(expected_row.row_count) != int(actual_row.row_count)
        ):
            raise ValueError("intraday file index is stale")
        if verify_hashes and str(expected_row.sha256) != str(actual_row.sha256):
            raise ValueError(f"minute-bar hash mismatch: {actual_row.symbol}")

    date_range = manifest.get("date_range")
    if not isinstance(date_range, Mapping):
        raise ValueError("intraday snapshot date range is invalid")
    try:
        start = pd.Timestamp(date_range["start"]).normalize()
        end = pd.Timestamp(date_range["end"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("intraday snapshot date range is invalid") from exc
    event_start = pd.Timestamp(events["date"].min()).normalize()
    event_end = pd.Timestamp(events["date"].max()).normalize()
    if start != event_start or end != event_end or start > end:
        raise ValueError("intraday snapshot date range does not match event coverage")

    if required_start is not None:
        required = pd.Timestamp(required_start).normalize()
        if start > required:
            raise IntradayDataBlocked(
                f"intraday snapshot does not cover required start {required.date()}"
            )
    if required_end is not None:
        required = pd.Timestamp(required_end).normalize()
        if end < required:
            raise IntradayDataBlocked(
                f"intraday snapshot does not cover required end {required.date()}"
            )
    if verify_hashes:
        expected_content = _content_hash(snapshot, manifest)
        if expected_content != str(manifest["content_sha256"]):
            raise ValueError("intraday snapshot content hash mismatch")
    return manifest
