#!/usr/bin/env python3
"""Incrementally update the formal ``cn_index_daily`` Parquet snapshot.

This updater handles only the five required index series.  It never touches
stock data or the retained native Qlib ``.bin`` compatibility snapshots.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _benchmark_sources import (  # noqa: E402
    EASTMONEY_BENCHMARKS,
    TENCENT_BENCHMARKS,
    download_eastmoney_benchmarks,
    download_tencent_benchmarks,
)
from _file_lock import lock_file, unlock_file  # noqa: E402
from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    dataset_publication_lock,
    finalize_price_snapshot,
    publish_snapshot_group,
    resolve_dataset,
    sha256_file,
)


DATASET_ID = "cn_index_daily"
DATASET_ROOT = Path("parquet/index_1d/cn_index_daily")
REQUIRED_SYMBOLS = (
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
)
FIELDS = ("open", "high", "low", "close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close")
OVERLAP_DAYS = 7
PRICE_RTOL = 5e-5
PRICE_ATOL = 5e-4
SCRIPT_VERSION = "1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


@contextmanager
def _single_instance_lock(root: Path):
    lock_path = root / "runtime" / "locks" / "cn_index_daily_update.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        try:
            lock_file(stream, exclusive=True, blocking=False)
        except (BlockingIOError, OSError) as exc:
            raise DataContractError(
                "another cn_index_daily update is already running"
            ) from exc
        yield
    finally:
        try:
            unlock_file(stream)
        finally:
            stream.close()


def _normalize_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataContractError(f"index {symbol} returned no rows")
    result = frame.copy()
    if "date" not in result.columns:
        if result.index.name in {"date", "datetime", "trade_date"}:
            result = result.reset_index()
        else:
            # Some provider clients return dates in an unnamed index.  Only
            # promote it when every index value parses as a date; resetting an
            # ordinary RangeIndex would hide a malformed response.
            parsed_index = pd.to_datetime(result.index, errors="coerce")
            if (
                len(parsed_index)
                and not isinstance(result.index, pd.RangeIndex)
                and not pd.api.types.is_numeric_dtype(result.index.dtype)
                and not pd.isna(parsed_index).any()
            ):
                result = result.reset_index().rename(columns={"index": "date"})
    if "date" not in result.columns:
        for candidate in ("datetime", "trade_date"):
            if candidate in result.columns:
                result = result.rename(columns={candidate: "date"})
                break
    missing = [field for field in ("date", *FIELDS) if field not in result.columns]
    if missing:
        raise DataContractError(f"index {symbol} is missing fields: {', '.join(missing)}")
    result = result.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise DataContractError(f"index {symbol} contains invalid dates")
    if result["date"].duplicated().any():
        raise DataContractError(f"index {symbol} contains duplicate dates")
    result = result.sort_values("date").reset_index(drop=True)
    for field in FIELDS:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    if result[list(FIELDS)].isna().any().any():
        raise DataContractError(f"index {symbol} contains null numeric fields")
    values = result[list(FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError(f"index {symbol} contains non-finite values")
    if (result[list(PRICE_FIELDS)] <= 0).any().any() or (result["volume"] < 0).any():
        raise DataContractError(f"index {symbol} contains invalid prices or volume")
    upper = result[["open", "low", "close"]].max(axis=1)
    lower = result[["open", "high", "close"]].min(axis=1)
    high = result["high"].to_numpy(dtype=float)
    low = result["low"].to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    high_tolerance = PRICE_ATOL + PRICE_RTOL * np.abs(upper_values)
    low_tolerance = PRICE_ATOL + PRICE_RTOL * np.abs(lower_values)
    if (high + high_tolerance < upper_values).any() or (
        low - low_tolerance > lower_values
    ).any():
        raise DataContractError(f"index {symbol} has invalid OHLC relationships")
    if "source_id" not in result.columns:
        result["source_id"] = "unknown"
    return result[["date", *FIELDS, "source_id"]]


def _read_snapshot_frames(snapshot: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in REQUIRED_SYMBOLS:
        path = snapshot / "bars" / f"{symbol}.parquet"
        if not path.is_file() or path.is_symlink():
            raise DataContractError(f"formal index snapshot is missing {symbol}")
        try:
            frames[symbol] = _normalize_frame(pd.read_parquet(path), symbol)
        except DataContractError:
            raise
        except Exception as exc:
            raise DataContractError(f"cannot read formal index {symbol}") from exc
    return frames


def _download_pair(
    start_date: str,
    end_date: str,
    *,
    use_system_proxy: bool,
    target_session: pd.Timestamp | None = None,
    tencent_download: Callable[..., dict[str, pd.DataFrame]],
    eastmoney_download: Callable[..., dict[str, pd.DataFrame]],
    source_order: Sequence[str] = ("eastmoney", "tencent"),
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if tuple(source_order) not in {
        ("eastmoney", "tencent"),
        ("tencent", "eastmoney"),
    }:
        raise ValueError("source_order must contain eastmoney and tencent exactly once")
    specs = {
        "eastmoney": (
            EASTMONEY_BENCHMARKS,
            download_eastmoney_benchmarks,
            eastmoney_download,
            "push2his.eastmoney.com index kline daily",
        ),
        "tencent": (
            TENCENT_BENCHMARKS,
            download_tencent_benchmarks,
            tencent_download,
            "web.ifzq.gtimg.cn appstock fqkline daily",
        ),
    }
    attempts: list[dict[str, str]] = []
    for source_name in source_order:
        codes, _default, downloader, interface = specs[source_name]
        try:
            raw = downloader(
                start_date,
                end_date,
                use_system_proxy=use_system_proxy,
                symbols=REQUIRED_SYMBOLS,
            )
            if set(raw) != set(REQUIRED_SYMBOLS):
                raise DataContractError("source returned an unexpected symbol set")
            frames = {
                symbol: _normalize_frame(raw[symbol], symbol)
                for symbol in REQUIRED_SYMBOLS
            }
            if target_session is not None:
                future_symbols = [
                    symbol
                    for symbol, frame in frames.items()
                    if (frame["date"] > target_session).any()
                ]
                if future_symbols:
                    raise DataContractError(
                        "source returned rows after target session "
                        f"{target_session.date()}: {', '.join(future_symbols)}"
                    )
                stale_symbols = [
                    symbol
                    for symbol, frame in frames.items()
                    if target_session
                    not in set(pd.DatetimeIndex(frame["date"]).normalize())
                ]
                if stale_symbols:
                    raise DataContractError(
                        "source is stale for target session "
                        f"{target_session.date()}: {', '.join(stale_symbols)}"
                    )
            for symbol, frame in frames.items():
                frame["source_id"] = f"{source_name}:{codes[symbol]}"
            return frames, {
                "provider": source_name,
                "interface": interface,
                "provider_codes": {symbol: codes[symbol] for symbol in REQUIRED_SYMBOLS},
                "attempts": attempts + [{"provider": source_name, "status": "selected"}],
            }
        except Exception as exc:
            attempts.append(
                {
                    "provider": source_name,
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
    raise DataContractError(
        "both formal index sources failed: " + json.dumps(attempts, ensure_ascii=False)
    )


def _assert_overlap_matches(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    symbol: str,
) -> None:
    left = current.set_index("date")
    right = incoming.set_index("date")
    overlap = left.join(right, how="inner", lsuffix="_old", rsuffix="_new")
    if overlap.empty:
        raise DataContractError(f"index {symbol} has no overlap with the existing snapshot")
    for field in FIELDS:
        if not np.isclose(
            overlap[f"{field}_old"].to_numpy(dtype=float),
            overlap[f"{field}_new"].to_numpy(dtype=float),
            rtol=PRICE_RTOL,
            atol=PRICE_ATOL,
        ).all():
            raise DataContractError(
                f"index {symbol} differs from the existing snapshot on overlap"
            )


def _merge_frames(
    current: Mapping[str, pd.DataFrame],
    incoming: Mapping[str, pd.DataFrame],
    through: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    if set(current) != set(REQUIRED_SYMBOLS) or set(incoming) != set(REQUIRED_SYMBOLS):
        raise DataContractError("formal index update must contain exactly five symbols")
    merged: dict[str, pd.DataFrame] = {}
    for symbol in REQUIRED_SYMBOLS:
        old = _normalize_frame(current[symbol], symbol)
        new = _normalize_frame(incoming[symbol], symbol)
        if (new["date"] > through).any():
            raise DataContractError(
                f"index {symbol} returned rows after {through.date()}"
            )
        _assert_overlap_matches(old, new, symbol)
        combined = pd.concat([old, new], ignore_index=True)
        combined = (
            combined.sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
        if through not in set(pd.DatetimeIndex(combined["date"]).normalize()):
            raise DataContractError(f"index {symbol} did not reach {through.date()}")
        merged[symbol] = combined[["date", *FIELDS, "source_id"]]
    return merged


def _last_exchange_session(value: pd.Timestamp) -> pd.Timestamp:
    """Resolve a requested calendar date to the latest A-share session."""

    try:
        import exchange_calendars

        session = exchange_calendars.get_calendar("XSHG").date_to_session(
            value, direction="previous"
        )
        return pd.Timestamp(session).normalize()
    except Exception:
        # Keep the updater usable in a minimal environment, but never treat a
        # weekday as a non-trading day merely because the calendar is absent.
        current = value.date()
        while current.weekday() >= 5:
            current -= timedelta(days=1)
        return pd.Timestamp(current)


def _write_source_evidence(
    root: Path,
    run_id: str,
    frames: Mapping[str, pd.DataFrame],
    source: Mapping[str, Any],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    evidence_dir = root / "runtime" / "index_updates" / "runs" / run_id
    evidence_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}
    for symbol in REQUIRED_SYMBOLS:
        path = evidence_dir / f"{symbol}.parquet"
        frames[symbol].to_parquet(path, index=False)
        files[symbol] = {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "row_count": int(len(frames[symbol])),
        }
    manifest_path = evidence_dir / "SOURCE_MANIFEST.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "operation": "incremental_cn_index_daily_update",
            "run_id": run_id,
            "requested_start": start_date,
            "requested_end": end_date,
            "requested_symbols": list(REQUIRED_SYMBOLS),
            "source": dict(source),
            "files": files,
            "retrieved_at": _now(),
        },
    )
    return {
        "path": manifest_path.relative_to(root).as_posix(),
        "sha256": sha256_file(manifest_path),
    }


def _identity(resolved: Any) -> tuple[str, str]:
    return (
        str(resolved.manifest["snapshot_id"]),
        str(resolved.manifest["content_sha256"]),
    )


def update_cn_index_snapshot(
    *,
    quant_data_root: Path,
    end_date: str | None = None,
    use_system_proxy: bool = False,
    tencent_download: Callable[..., dict[str, pd.DataFrame]] = download_tencent_benchmarks,
    eastmoney_download: Callable[..., dict[str, pd.DataFrame]] = download_eastmoney_benchmarks,
    source_order: Sequence[str] = ("eastmoney", "tencent"),
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    with _single_instance_lock(root):
        current = resolve_dataset(DATASET_ID, quant_data_root=root, require_basis="derived")
        current_frames = _read_snapshot_frames(current.snapshot_path)
        current_latest_by_symbol = {
            symbol: pd.Timestamp(frame["date"].max()).normalize()
            for symbol, frame in current_frames.items()
        }
        requested_end = pd.Timestamp(end_date or datetime.now().astimezone().date()).normalize()
        if pd.isna(requested_end):
            raise DataContractError("index update end date is invalid")
        target_session = _last_exchange_session(requested_end)
        common_latest = min(current_latest_by_symbol.values())
        if common_latest >= target_session:
            return {
                "status": "already_current",
                "dataset_id": DATASET_ID,
                "snapshot_id": current.manifest["snapshot_id"],
                "latest_trade_date": current.manifest["latest_trade_date"],
                "network_requests_for_symbols": [],
                "stock_symbols_downloaded": 0,
                "index_symbols_downloaded": 0,
            }
        fetch_start = common_latest - pd.Timedelta(days=OVERLAP_DAYS)
        start_text = fetch_start.date().isoformat()
        end_text = requested_end.date().isoformat()
        incoming, source = _download_pair(
            start_text,
            end_text,
            use_system_proxy=use_system_proxy,
            target_session=target_session,
            tencent_download=tencent_download,
            eastmoney_download=eastmoney_download,
            source_order=source_order,
        )
        merged = _merge_frames(current_frames, incoming, target_session)
        run_id = _run_id()
        evidence = _write_source_evidence(
            root,
            run_id,
            incoming,
            source,
            start_text,
            end_text,
        )
        source_metadata = {
            "provider": source["provider"],
            "interface": source["interface"],
            "provider_codes": source["provider_codes"],
            "evidence_path": evidence["path"],
            "evidence_sha256": evidence["sha256"],
            "base_snapshot_id": current.manifest["snapshot_id"],
        }
        snapshot_id = f"cn-index-daily-{run_id}"
        staging_run = root / "staging" / f"cn-index-daily-update-{run_id}"
        staging_snapshot = staging_run / snapshot_id
        dataset_root = root / DATASET_ROOT
        try:
            shutil.copytree(current.snapshot_path, staging_snapshot)
            for symbol, frame in merged.items():
                frame.to_parquet(staging_snapshot / "bars" / f"{symbol}.parquet", index=False)
            finalize_price_snapshot(
                staging_snapshot,
                dataset_id=DATASET_ID,
                snapshot_id=snapshot_id,
                role="canonical",
                status="ready",
                source=source_metadata,
                price_basis="derived",
                adjustment={"method": "none"},
                producer={
                    "script": "update_cn_index_dataset.py",
                    "version": SCRIPT_VERSION,
                },
                source_retrieved_at=_now(),
            )
            with dataset_publication_lock(root):
                current_again = resolve_dataset(
                    DATASET_ID,
                    quant_data_root=root,
                    require_basis="derived",
                )
                if _identity(current_again) != _identity(current):
                    raise DataContractError(
                        "cn_index_daily changed while the update was running"
                    )
                published = publish_snapshot_group(
                    [(staging_snapshot, dataset_root)],
                    quant_data_root=root,
                    update_root_catalog=True,
                    publication_lock_held=True,
                )[0]
            return {
                "status": "published",
                "dataset_id": DATASET_ID,
                "snapshot_id": published.name,
                "latest_trade_date": max(
                    str(frame["date"].max().date()) for frame in merged.values()
                ),
                "base_snapshot_id": current.manifest["snapshot_id"],
                "source": source_metadata,
                "network_requests_for_symbols": list(REQUIRED_SYMBOLS),
                "stock_symbols_downloaded": 0,
                "index_symbols_downloaded": len(REQUIRED_SYMBOLS),
            }
        except Exception:
            shutil.rmtree(staging_run, ignore_errors=True)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-root", type=Path, default=default_quant_data_root())
    parser.add_argument("--through", dest="end_date")
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument(
        "--source-order",
        nargs=2,
        choices=("eastmoney", "tencent"),
        default=("eastmoney", "tencent"),
        metavar=("FIRST", "SECOND"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = update_cn_index_snapshot(
            quant_data_root=args.quant_data_root,
            end_date=args.end_date,
            use_system_proxy=args.use_system_proxy,
            source_order=tuple(args.source_order),
        )
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
