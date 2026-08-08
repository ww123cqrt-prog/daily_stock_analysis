#!/usr/bin/env python3
"""Build staged A-share raw/qfq snapshots from a bounded incremental bundle."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_baostock_v2 import (  # noqa: E402
    BaostockV2Error,
    FACTOR_COLUMNS,
    PRICE_COLUMNS,
    atomic_parquet,
    build_local_qfq,
    expand_daily_factors,
    resolve_bundle_file,
    sha256_file,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_price_snapshot,
    resolve_dataset,
)


RAW_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_raw")
QFQ_DATASET_ROOT = Path("parquet/ashare_1d/ashare_daily_qfq")
_EVENT_VALUE_COLUMNS = (
    "foreAdjustFactor",
    "backAdjustFactor",
    "adjustFactor",
)


@dataclass(frozen=True)
class IncrementalPriceResult:
    raw_stage: Path
    qfq_stage: Path
    base_latest_trade_date: str
    latest_trade_date: str
    downloaded_rows: int
    reused_symbols: tuple[str, ...]
    factor_changed_symbols: tuple[str, ...]
    qfq_recalculated_symbols: tuple[str, ...]


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return value


def _bundle_file(bundle_dir: Path, relative: object, *, label: str) -> Path:
    try:
        return resolve_bundle_file(bundle_dir, relative, label=label)
    except BaostockV2Error as exc:
        raise DataContractError(str(exc)) from exc


def _validate_bundle(bundle_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = _bundle_file(
        bundle_dir,
        "BUNDLE_MANIFEST.json",
        label="incremental bundle manifest",
    )
    manifest = _read_json(manifest_path, label="incremental bundle manifest")
    if manifest.get("status") != "ready":
        raise DataContractError("incremental bundle is not ready")
    if manifest.get("bundle_mode") != "incremental":
        raise DataContractError("incremental update requires an incremental bundle")
    if manifest.get("factor_requested_start") != "1990-01-01":
        raise DataContractError("incremental bundle does not contain full factor history")
    try:
        requested_start = pd.Timestamp(manifest["requested_start"]).normalize()
        requested_through = pd.Timestamp(manifest["requested_through"]).normalize()
        base_latest = pd.Timestamp(manifest["base_latest_trade_date"]).normalize()
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("incremental bundle date contract is invalid") from exc
    if not requested_start <= base_latest < requested_through:
        raise DataContractError("incremental bundle does not overlap and extend the base")
    if (base_latest - requested_start).days > 62:
        raise DataContractError("incremental bundle overlap exceeds 62 calendar days")
    index_path = _bundle_file(
        bundle_dir,
        manifest.get("index_path"),
        label="incremental bundle index",
    )
    try:
        index = pd.read_parquet(index_path)
    except (OSError, ValueError) as exc:
        raise DataContractError("incremental bundle index is unavailable") from exc
    required = {
        "symbol",
        "raw_path",
        "factor_events_path",
        "raw_sha256",
        "factor_events_sha256",
        "raw_rows",
    }
    if required.difference(index.columns):
        raise DataContractError("incremental bundle index is incomplete")
    if index.empty or index["symbol"].duplicated().any():
        raise DataContractError("incremental bundle symbol index is invalid")
    if len(index) != int(manifest.get("completed_symbols", -1)):
        raise DataContractError("incremental bundle index row count is invalid")
    for row in index.itertuples(index=False):
        symbol = str(row.symbol)
        _bundle_file(
            bundle_dir,
            row.raw_path,
            label=f"incremental raw file for {symbol}",
        )
        _bundle_file(
            bundle_dir,
            row.factor_events_path,
            label=f"incremental factor file for {symbol}",
        )
    return manifest, index.sort_values("symbol", kind="stable").reset_index(drop=True)


def _normalize_bars(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = {"date", *PRICE_COLUMNS, "volume", "symbol", "source_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataContractError(f"bar schema is incomplete for {symbol}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    if normalized["date"].isna().any():
        raise DataContractError(f"bar dates are invalid for {symbol}")
    if normalized["date"].duplicated().any():
        raise DataContractError(f"bar dates are duplicated for {symbol}")
    if set(normalized["symbol"].astype(str)) != {symbol}:
        raise DataContractError(f"bar symbol does not match index for {symbol}")
    return normalized.sort_values("date", kind="stable").reset_index(drop=True)


def _merge_by_date(base: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    if update.empty:
        return base.copy().reset_index(drop=True)
    columns = list(dict.fromkeys([*base.columns, *update.columns]))
    merged = pd.concat(
        [base.reindex(columns=columns), update.reindex(columns=columns)],
        ignore_index=True,
    )
    return (
        merged.sort_values("date", kind="stable")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _canonical_events(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    events = frame.copy()
    if "action_type" in events.columns:
        events = events.loc[events["action_type"] == "adjustment_factor_event"]
    if "symbol" in events.columns:
        events = events.loc[events["symbol"].astype(str) == symbol]
    date_column = "event_date" if "event_date" in events.columns else "dividOperateDate"
    if date_column not in events.columns:
        if events.empty:
            return pd.DataFrame(columns=["event_date", *_EVENT_VALUE_COLUMNS])
        raise DataContractError(f"factor event dates are missing for {symbol}")
    missing = set(_EVENT_VALUE_COLUMNS).difference(events.columns)
    if missing:
        raise DataContractError(f"factor event values are missing for {symbol}")
    normalized = pd.DataFrame(
        {"event_date": pd.to_datetime(events[date_column], errors="coerce").dt.normalize()}
    )
    for column in _EVENT_VALUE_COLUMNS:
        normalized[column] = pd.to_numeric(events[column], errors="coerce")
    if normalized.isna().any().any() or (normalized["foreAdjustFactor"] <= 0).any():
        raise DataContractError(f"factor event history is invalid for {symbol}")
    return (
        normalized.sort_values("event_date", kind="stable")
        .drop_duplicates("event_date", keep="last")
        .reset_index(drop=True)
    )


def _events_changed(
    old_actions: pd.DataFrame,
    new_events: pd.DataFrame,
    *,
    symbol: str,
) -> bool:
    old = _canonical_events(old_actions, symbol=symbol)
    new = _canonical_events(new_events, symbol=symbol)
    if len(old) != len(new):
        return True
    if old.empty:
        return False
    if not old["event_date"].equals(new["event_date"]):
        return True
    return not bool(
        np.isclose(
            old[list(_EVENT_VALUE_COLUMNS)].to_numpy(dtype=float),
            new[list(_EVENT_VALUE_COLUMNS)].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ).all()
    )


def _event_output(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    output = frame.copy()
    if "dividOperateDate" in output.columns:
        output = output.rename(columns={"dividOperateDate": "event_date"})
    output["event_date"] = pd.to_datetime(output["event_date"], errors="raise").dt.normalize()
    output["symbol"] = symbol
    output["action_type"] = "adjustment_factor_event"
    return output


def _identity_extension(
    base_factors: pd.DataFrame,
    update_bars: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    if update_bars.empty:
        return base_factors.copy()
    latest = float(pd.to_numeric(base_factors["factor"], errors="coerce").iloc[-1])
    if not np.isclose(latest, 1.0, atol=1e-12, rtol=1e-12):
        raise DataContractError(f"empty factor history cannot extend {symbol}")
    extension = pd.DataFrame({"date": update_bars["date"].copy()})
    extension["source_event_date"] = pd.NaT
    extension["factor"] = 1.0
    extension["factor_source"] = "baostock_no_adjustment_events"
    extension["factor_backfilled_before_first_event"] = False
    extension["factor_paired_fallback"] = False
    extension["factor_terminal_normalized"] = False
    extension["provider_qfq_terminal_factor"] = np.nan
    return _merge_by_date(base_factors, extension)


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_incremental_price_snapshots(
    *,
    bundle_dir: Path,
    quant_data_root: Path,
) -> IncrementalPriceResult:
    """Create complete staged raw/qfq snapshots without moving formal pointers."""

    root = Path(quant_data_root).expanduser().resolve()
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest, index = _validate_bundle(bundle)
    raw_base = resolve_dataset("ashare_daily_raw", quant_data_root=root, require_basis="raw")
    qfq_base = resolve_dataset("ashare_daily_qfq", quant_data_root=root, require_basis="qfq")
    base_latest = str(manifest["base_latest_trade_date"])
    if raw_base.manifest.get("latest_trade_date") != base_latest:
        raise DataContractError("incremental bundle does not match current raw snapshot")
    if qfq_base.manifest.get("latest_trade_date") != base_latest:
        raise DataContractError("incremental bundle does not match current qfq snapshot")

    raw_symbols = {path.stem for path in (raw_base.snapshot_path / "bars").glob("*.parquet")}
    qfq_symbols = {path.stem for path in (qfq_base.snapshot_path / "bars").glob("*.parquet")}
    factor_symbols = {path.stem for path in (qfq_base.snapshot_path / "factors").glob("*.parquet")}
    bundle_symbols = set(index["symbol"].astype(str))
    if raw_symbols != qfq_symbols or raw_symbols != factor_symbols:
        raise DataContractError("current raw/qfq/factor symbol sets disagree")
    if bundle_symbols != raw_symbols:
        raise DataContractError("incremental bundle does not cover the current symbol set")

    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_root = root / "staging" / f"ashare-incremental-v2-{run_id}"
    raw_stage = run_root / f"ashare-baostock-raw-{run_id}"
    qfq_stage = run_root / f"ashare-baostock-qfq-{run_id}"
    actions_path = qfq_base.snapshot_path / "corporate_actions.parquet"
    try:
        old_actions = pd.read_parquet(actions_path)
    except (OSError, ValueError) as exc:
        raise DataContractError("current factor event evidence is unavailable") from exc

    reused: list[str] = []
    changed: list[str] = []
    actions: list[pd.DataFrame] = []
    downloaded_rows = 0
    requested_start = pd.Timestamp(manifest["requested_start"]).normalize()
    requested_through = pd.Timestamp(manifest["requested_through"]).normalize()

    try:
        for row in index.itertuples(index=False):
            symbol = str(row.symbol)
            raw_update_path = _bundle_file(
                bundle,
                row.raw_path,
                label=f"incremental raw file for {symbol}",
            )
            events_path = _bundle_file(
                bundle,
                row.factor_events_path,
                label=f"incremental factor file for {symbol}",
            )
            if sha256_file(raw_update_path) != str(row.raw_sha256):
                raise DataContractError(f"incremental raw hash mismatch for {symbol}")
            if sha256_file(events_path) != str(row.factor_events_sha256):
                raise DataContractError(f"incremental factor hash mismatch for {symbol}")
            try:
                raw_update = _normalize_bars(pd.read_parquet(raw_update_path), symbol=symbol)
                new_events = pd.read_parquet(events_path)
                base_raw = _normalize_bars(
                    pd.read_parquet(raw_base.snapshot_path / "bars" / f"{symbol}.parquet"),
                    symbol=symbol,
                )
                base_qfq = _normalize_bars(
                    pd.read_parquet(qfq_base.snapshot_path / "bars" / f"{symbol}.parquet"),
                    symbol=symbol,
                )
                base_factors = pd.read_parquet(
                    qfq_base.snapshot_path / "factors" / f"{symbol}.parquet"
                )
            except (OSError, ValueError) as exc:
                raise DataContractError(f"incremental inputs are unreadable for {symbol}") from exc
            downloaded_rows += len(raw_update)
            if len(raw_update) != int(row.raw_rows):
                raise DataContractError(f"incremental raw row count mismatch for {symbol}")
            if not raw_update.empty and (
                raw_update["date"].min() < requested_start
                or raw_update["date"].max() > requested_through
            ):
                raise DataContractError(f"incremental raw dates are out of range for {symbol}")
            merged_raw = _merge_by_date(base_raw, raw_update)
            factor_changed = _events_changed(old_actions, new_events, symbol=symbol)
            canonical_new_events = _canonical_events(new_events, symbol=symbol)
            if factor_changed:
                if canonical_new_events.empty:
                    raise DataContractError(f"factor history disappeared for {symbol}")
                daily_factors = expand_daily_factors(merged_raw, new_events)
                merged_qfq = build_local_qfq(merged_raw, daily_factors)
                changed.append(symbol)
            elif raw_update.empty:
                _copy_or_link(
                    qfq_base.snapshot_path / "bars" / f"{symbol}.parquet",
                    qfq_stage / "bars" / f"{symbol}.parquet",
                )
                _copy_or_link(
                    qfq_base.snapshot_path / "factors" / f"{symbol}.parquet",
                    qfq_stage / "factors" / f"{symbol}.parquet",
                )
                merged_qfq = None
                daily_factors = None
                reused.append(symbol)
            else:
                if canonical_new_events.empty:
                    daily_factors = _identity_extension(
                        base_factors, raw_update, symbol=symbol
                    )
                    update_factors = daily_factors.loc[
                        daily_factors["date"].isin(raw_update["date"])
                    ]
                else:
                    update_factors = expand_daily_factors(raw_update, new_events)
                    daily_factors = _merge_by_date(base_factors, update_factors)
                update_qfq = build_local_qfq(raw_update, update_factors)
                merged_qfq = _merge_by_date(base_qfq, update_qfq)
                reused.append(symbol)

            atomic_parquet(merged_raw, raw_stage / "bars" / f"{symbol}.parquet")
            if merged_qfq is not None and daily_factors is not None:
                atomic_parquet(merged_qfq, qfq_stage / "bars" / f"{symbol}.parquet")
                atomic_parquet(daily_factors, qfq_stage / "factors" / f"{symbol}.parquet")
            actions.append(_event_output(new_events, symbol=symbol))

        corporate_actions = (
            pd.concat(actions, ignore_index=True)
            if actions
            else pd.DataFrame(columns=["event_date", *FACTOR_COLUMNS, "symbol", "action_type"])
        )
        atomic_parquet(corporate_actions, raw_stage / "corporate_actions.parquet")
        atomic_parquet(corporate_actions, qfq_stage / "corporate_actions.parquet")
        source_retrieved_at = str(manifest.get("source_retrieved_at") or "")
        try:
            datetime.fromisoformat(source_retrieved_at)
        except ValueError as exc:
            raise DataContractError("incremental bundle retrieval timestamp is invalid") from exc
        source = {
            "provider": "baostock",
            "interface": "incremental_raw_plus_complete_factor_history",
            "source_bundle_sha256": sha256_file(
                _bundle_file(
                    bundle,
                    "BUNDLE_MANIFEST.json",
                    label="incremental bundle manifest",
                )
            ),
            "execution_environment": manifest.get("source", {}).get(
                "execution_environment", "private_remote_windows"
            ),
            "base_snapshot_id": raw_base.manifest["snapshot_id"],
            "requested_start": str(manifest["requested_start"]),
            "requested_through": str(manifest["requested_through"]),
        }
        finalize_price_snapshot(
            raw_stage,
            dataset_id="ashare_daily_raw",
            snapshot_id=raw_stage.name,
            role="canonical",
            status="ready",
            source=source,
            price_basis="raw",
            adjustment={"method": "none"},
            producer={"script": "incremental_update_ashare_v2.py", "version": "1"},
            source_retrieved_at=source_retrieved_at,
            upstream_snapshots=[
                {
                    "dataset_id": "ashare_daily_raw",
                    "snapshot_id": str(raw_base.manifest["snapshot_id"]),
                    "content_sha256": str(raw_base.manifest["content_sha256"]),
                    "price_basis": "raw",
                }
            ],
        )
        finalize_price_snapshot(
            qfq_stage,
            dataset_id="ashare_daily_qfq",
            snapshot_id=qfq_stage.name,
            role="canonical",
            status="ready",
            source=source,
            price_basis="qfq",
            adjustment={
                "method": "local_factor",
                "adjusted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "adjustment_as_of": str(manifest["requested_through"]),
                "adjusted_fields": list(PRICE_COLUMNS),
                "normalization_rule": "latest_valid_trade_date_factor_equals_1",
                "algorithm_version": "baostock-factor-incremental-v1",
                "factor_changed_symbols": len(changed),
                "volume_adjusted": False,
            },
            producer={"script": "incremental_update_ashare_v2.py", "version": "1"},
            source_retrieved_at=source_retrieved_at,
            upstream_snapshots=[
                {
                    "dataset_id": "ashare_daily_raw",
                    "snapshot_id": raw_stage.name,
                }
            ],
        )
    except Exception:
        shutil.rmtree(run_root, ignore_errors=True)
        raise

    latest = str(pd.Timestamp(manifest["requested_through"]).date())
    return IncrementalPriceResult(
        raw_stage=raw_stage,
        qfq_stage=qfq_stage,
        base_latest_trade_date=base_latest,
        latest_trade_date=latest,
        downloaded_rows=downloaded_rows,
        reused_symbols=tuple(sorted(reused)),
        factor_changed_symbols=tuple(sorted(changed)),
        qfq_recalculated_symbols=tuple(sorted(changed)),
    )
