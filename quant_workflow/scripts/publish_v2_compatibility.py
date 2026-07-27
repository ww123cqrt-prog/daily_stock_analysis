#!/usr/bin/env python3
"""Build and publish V1-compatible outputs from governed V2 snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from _file_lock import lock_file, unlock_file  # noqa: E402
from add_index_benchmark import BENCHMARKS, add_benchmark  # noqa: E402
from apply_data_cleaning import (  # noqa: E402
    build_cleaned_parquet,
    copy_benchmark_features,
    load_cleaning_plan,
)
from generate_csi300 import write_csi300  # noqa: E402
from merge_to_qlib import convert_to_bin  # noqa: E402
from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402
from report_data_anomalies import (  # noqa: E402
    SEVERITY_RANK,
    build_report as build_anomaly_report,
    scan_merged_files,
)


V1_PARQUET_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "factor",
    "amount",
    "turn",
    "pctChg",
]
REQUIRED_PRICE_COLUMNS = ["open", "close", "high", "low", "volume"]
ARTIFACT_NAMES = (
    "merged_parquet",
    "merged_qlib",
    "cleaned_parquet",
    "cleaned_qlib",
)
PREFIXED_SYMBOL_RE = re.compile(r"^(?:SH|SZ|BJ)(\d{6})$")
PLAIN_SYMBOL_RE = re.compile(r"^\d{6}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
BENCHMARK_SYMBOLS = frozenset(BENCHMARKS)


class CompatibilityLayout(NamedTuple):
    run_id: str
    run_root: Path
    reports_dir: Path
    candidates: dict[str, Path]


def _run_id() -> str:
    return (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def compatibility_layout(root: Path, run_id: str | None = None) -> CompatibilityLayout:
    identifier = str(run_id or _run_id())
    if not RUN_ID_RE.fullmatch(identifier):
        raise ValueError("compatibility run_id is invalid")
    run_root = Path(root) / "runtime" / "v2_compatibility" / "runs" / identifier
    staging = run_root / "staging"
    return CompatibilityLayout(
        run_id=identifier,
        run_root=run_root,
        reports_dir=run_root / "reports",
        candidates={
            "merged_parquet": staging / "merged_baostock_jqdata",
            "merged_qlib": staging / "cn_data_baostock_jqdata",
            "cleaned_parquet": staging / "merged_baostock_jqdata_cleaned",
            "cleaned_qlib": staging / "cn_data_baostock_jqdata_cleaned",
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
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


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _compatibility_index(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(output_dir).glob("*.parquet")):
        frame = pd.read_parquet(path)
        if list(frame.columns) != V1_PARQUET_COLUMNS:
            raise ValueError(f"V1 parquet schema mismatch: {path.name}")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if (
            frame.empty
            or dates.isna().any()
            or dates.duplicated().any()
            or not dates.is_monotonic_increasing
        ):
            raise ValueError(f"V1 parquet dates are invalid: {path.name}")
        factors = pd.to_numeric(frame["factor"], errors="coerce")
        if factors.isna().any() or not np.allclose(factors, 1.0):
            raise ValueError(f"V1 parquet factor must be unit factor: {path.name}")
        rows.append(
            {
                "symbol": path.stem,
                "relative_path": path.name,
                "row_count": int(len(frame)),
                "first_date": dates.iloc[0].date().isoformat(),
                "last_date": dates.iloc[-1].date().isoformat(),
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise ValueError(f"V1 compatibility output has no parquet files: {output_dir}")
    return pd.DataFrame(rows)


def _write_compatibility_metadata(
    output_dir: Path,
    *,
    upstream: Mapping[str, Any],
    transformation: Mapping[str, Any] | None = None,
    index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    index = _compatibility_index(output_dir) if index is None else index.copy()
    if index.empty:
        raise ValueError("V1 compatibility file index is empty")
    metadata_dir = Path(output_dir) / ".metadata"
    _atomic_parquet(index, metadata_dir / "FILE_INDEX.parquet")
    manifest = {
        "schema_version": 1,
        "contract": "v1_ashare_qfq_compatibility",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "upstream": dict(upstream),
        "transformation": dict(transformation or {"name": "v2_to_v1_contract"}),
        "price_basis": "qfq",
        "factor_semantics": "unit_factor_prices_already_qfq",
        "date_dtype": "datetime64[ns]",
        "columns": V1_PARQUET_COLUMNS,
        "symbol_count": int(len(index)),
        "row_count": int(index["row_count"].sum()),
        "date_range": {
            "start": str(index["first_date"].min()),
            "end": str(index["last_date"].max()),
        },
        "file_index_path": ".metadata/FILE_INDEX.parquet",
    }
    _atomic_json(metadata_dir / "COMPATIBILITY_MANIFEST.json", manifest)
    return manifest


def v1_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if PLAIN_SYMBOL_RE.fullmatch(symbol):
        return symbol
    match = PREFIXED_SYMBOL_RE.fullmatch(symbol)
    if match:
        return match.group(1)
    raise ValueError(f"unsupported A-share symbol for V1 output: {value}")


def to_v1_frame(source: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ["date", *REQUIRED_PRICE_COLUMNS] if column not in source]
    if missing:
        raise ValueError(f"V2 bar frame is missing required columns: {missing}")
    result = pd.DataFrame(index=source.index)
    result["date"] = pd.to_datetime(source["date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    for column in REQUIRED_PRICE_COLUMNS:
        result[column] = pd.to_numeric(source[column], errors="coerce")
    result["factor"] = 1.0
    for column in ("amount", "turn", "pctChg"):
        result[column] = (
            pd.to_numeric(source[column], errors="coerce")
            if column in source
            else float("nan")
        )
    if result[["date", *REQUIRED_PRICE_COLUMNS]].isna().any(axis=None):
        raise ValueError("V2 bar frame contains invalid values required by V1")
    result = result[V1_PARQUET_COLUMNS]
    return result.sort_values("date").drop_duplicates("date", keep="last")


def build_v1_parquet_output(
    source_bars: Path,
    output_dir: Path,
    *,
    upstream: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    source_bars = Path(source_bars)
    output_dir = Path(output_dir)
    source_files = sorted(source_bars.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"V2 snapshot has no bar files: {source_bars}")
    mapped: dict[str, Path] = {}
    for path in source_files:
        symbol = v1_symbol(path.stem)
        if symbol in mapped:
            raise ValueError(
                f"duplicate V1 symbol {symbol}: {mapped[symbol].name}, {path.name}"
            )
        mapped[symbol] = path

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"V1 compatibility output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    index_rows: list[dict[str, Any]] = []
    total_rows = 0
    for position, (symbol, source_path) in enumerate(sorted(mapped.items()), 1):
        frame = to_v1_frame(pd.read_parquet(source_path))
        output_path = output_dir / f"{symbol}.parquet"
        _atomic_parquet(frame, output_path)
        total_rows += len(frame)
        index_rows.append(
            {
                "symbol": symbol,
                "relative_path": output_path.name,
                "row_count": int(len(frame)),
                "first_date": frame["date"].iloc[0].date().isoformat(),
                "last_date": frame["date"].iloc[-1].date().isoformat(),
                "sha256": _sha256_file(output_path),
            }
        )
        if position % 1000 == 0 or position == len(mapped):
            print(f"  V1 parquet {position}/{len(mapped)}")

    manifest = _write_compatibility_metadata(
        output_dir,
        upstream=upstream,
        index=pd.DataFrame(index_rows),
    )
    if manifest["row_count"] != total_rows or manifest["symbol_count"] != len(index_rows):
        raise RuntimeError("V1 compatibility metadata does not match generated files")
    return manifest


def _sort_anomaly_frames(
    summary: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary.empty:
        summary = summary.sort_values(
            by=["severity", "symbol"],
            key=lambda column: (
                column.map(SEVERITY_RANK) if column.name == "severity" else column
            ),
            ascending=[False, True],
        )
    if not events.empty:
        events = events.sort_values(
            by=["severity", "symbol", "issue_type"],
            key=lambda column: (
                column.map(SEVERITY_RANK) if column.name == "severity" else column
            ),
            ascending=[False, True, True],
        )
    return summary, events


def _write_fresh_anomaly_reports(
    merged_dir: Path, reports_dir: Path
) -> tuple[Path, Path, dict[str, Any]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, events, coverage = scan_merged_files(merged_dir)
    summary, events = _sort_anomaly_frames(summary, events)
    stocks_csv = reports_dir / "data_anomaly_stocks_latest.csv"
    events_csv = reports_dir / "data_anomaly_events_latest.csv"
    report_path = reports_dir / "data_cleaning_strategy_latest.md"
    summary.to_csv(stocks_csv, index=False)
    events.to_csv(events_csv, index=False)
    report_path.write_text(
        build_anomaly_report(
            summary_df=summary,
            events_df=events,
            coverage=coverage,
            stock_csv=stocks_csv,
            events_csv=events_csv,
            merged_dir=merged_dir,
        ),
        encoding="utf-8",
    )
    blocker_count = (
        int((summary["severity"] == "blocker").sum())
        if not summary.empty
        else 0
    )
    if blocker_count:
        raise ValueError(
            f"V2 compatibility candidate has {blocker_count} structural blockers"
        )
    return stocks_csv, events_csv, {**coverage, "blocker_symbols": blocker_count}


def _provider_instruments(provider: Path, name: str) -> set[str]:
    path = Path(provider) / "instruments" / name
    if not path.is_file():
        raise ValueError(f"Qlib instrument file is missing: {name}")
    rows = {
        line.split("\t", 1)[0].strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not rows:
        raise ValueError(f"Qlib instrument file is empty: {name}")
    return rows


def _read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2 or not np.isfinite(values[0]):
        raise ValueError(f"Qlib feature bin is invalid: {path}")
    return values[1:]


def validate_qlib_output(
    provider: Path, *, require_benchmarks: bool = True
) -> dict[str, Any]:
    provider = Path(provider)
    calendar_path = provider / "calendars" / "day.txt"
    calendar = [
        line.strip()
        for line in calendar_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("Qlib compatibility calendar is invalid")
    all_symbols = _provider_instruments(provider, "all.txt")
    csi300 = _provider_instruments(provider, "csi300.txt")
    if not csi300.issubset(all_symbols):
        raise ValueError("Qlib csi300 contains symbols absent from all.txt")
    if (provider / "instruments" / "csi800.txt").exists():
        raise ValueError("general V1 compatibility provider must not contain csi800.txt")
    for symbol in all_symbols:
        values = _read_bin(
            provider / "features" / symbol.lower() / "factor.day.bin"
        )
        finite = values[np.isfinite(values)]
        if not len(finite) or not np.allclose(finite, 1.0):
            raise ValueError(f"Qlib compatibility requires unit factor: {symbol}")
    if require_benchmarks:
        for symbol in BENCHMARK_SYMBOLS:
            if symbol in all_symbols or symbol in csi300:
                raise ValueError(f"benchmark leaked into stock pool: {symbol}")
            feature_dir = provider / "features" / symbol.lower()
            for field in ("open", "high", "low", "close", "volume", "factor"):
                values = _read_bin(feature_dir / f"{field}.day.bin")
                if not np.isfinite(values).any():
                    raise ValueError(f"benchmark feature is empty: {symbol}/{field}")
    manifest_path = provider / "PROVIDER_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("conversion", {}).get("factor_semantics") != (
        "unit_factor_prices_already_qfq"
    ):
        raise ValueError("Qlib provider manifest has the wrong factor semantics")
    return {
        "calendar_days": len(calendar),
        "latest_trade_date": calendar[-1],
        "instruments": len(all_symbols),
        "csi300_symbols": len(csi300),
    }


def validate_parquet_output(
    directory: Path,
    *,
    expected_snapshot: Mapping[str, Any],
    cleaned: bool,
) -> dict[str, Any]:
    directory = Path(directory)
    manifest = json.loads(
        (directory / ".metadata" / "COMPATIBILITY_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    index = pd.read_parquet(directory / str(manifest["file_index_path"]))
    actual_files = {path.name for path in directory.glob("*.parquet")}
    indexed_files = set(index["relative_path"].astype(str))
    if actual_files != indexed_files:
        raise ValueError("V1 compatibility file index does not match files")
    if int(manifest["symbol_count"]) != len(index):
        raise ValueError("V1 compatibility symbol count is stale")
    expected_symbols = int(expected_snapshot["symbol_count"])
    if len(index) != expected_symbols:
        raise ValueError(
            f"V1 compatibility symbol count mismatch: {len(index)} != {expected_symbols}"
        )
    if str(index["last_date"].max()) != str(expected_snapshot["latest_trade_date"]):
        raise ValueError("V1 compatibility latest date does not match V2")
    if not cleaned and int(index["row_count"].sum()) != int(
        expected_snapshot["row_count"]
    ):
        raise ValueError("V1 compatibility row count does not match V2")
    return {
        "symbols": len(index),
        "rows": int(index["row_count"].sum()),
        "latest_trade_date": str(index["last_date"].max()),
    }


def validate_compatibility_candidates(
    candidates: Mapping[str, Path], *, source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    details = {
        "merged_parquet": validate_parquet_output(
            candidates["merged_parquet"],
            expected_snapshot=source_manifest,
            cleaned=False,
        ),
        "cleaned_parquet": validate_parquet_output(
            candidates["cleaned_parquet"],
            expected_snapshot=source_manifest,
            cleaned=True,
        ),
        "merged_qlib": validate_qlib_output(candidates["merged_qlib"]),
        "cleaned_qlib": validate_qlib_output(candidates["cleaned_qlib"]),
    }
    expected_latest = str(source_manifest["latest_trade_date"])
    for name in ("merged_qlib", "cleaned_qlib"):
        if details[name]["latest_trade_date"] != expected_latest:
            raise ValueError(f"{name} latest date does not match V2")
    return {"status": "pass", "artifacts": details}


def build_compatibility_candidates(
    *,
    quant_data_root: Path,
    run_id: str | None = None,
    benchmark_source: str = "tencent",
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    layout = compatibility_layout(root, run_id)
    if layout.run_root.exists():
        raise FileExistsError(f"compatibility run already exists: {layout.run_root}")
    qfq = resolve_dataset(
        "ashare_daily_qfq", quant_data_root=root, require_basis="qfq"
    )
    source_manifest = qfq.manifest
    stock_lists_path = root / "stock_lists.json"
    stock_lists_sha256 = _sha256_file(stock_lists_path)
    upstream = {
        key: str(source_manifest[key])
        for key in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    candidates = layout.candidates
    layout.run_root.mkdir(parents=True)
    try:
        merged_manifest = build_v1_parquet_output(
            qfq.snapshot_path / "bars",
            candidates["merged_parquet"],
            upstream=upstream,
        )
        if convert_to_bin(
            candidates["merged_parquet"],
            candidates["merged_qlib"],
            overwrite=True,
            source_contract=upstream,
            force_unit_factor=True,
        ):
            raise RuntimeError("merged V1 Qlib conversion failed")
        write_csi300(
            candidates["merged_qlib"] / "instruments" / "csi300.txt",
            candidates["merged_qlib"] / "calendars" / "day.txt",
            root / "stock_lists.json",
        )
        benchmark = add_benchmark(
            candidates["merged_qlib"],
            benchmark_source=benchmark_source,
            status_dir=layout.run_root / "status",
        )
        stocks_csv, events_csv, anomaly = _write_fresh_anomaly_reports(
            candidates["merged_parquet"], layout.reports_dir
        )
        plan = load_cleaning_plan(stocks_csv, events_csv)
        cleaning = build_cleaned_parquet(
            candidates["merged_parquet"],
            candidates["cleaned_parquet"],
            plan,
            overwrite=True,
        )
        _write_compatibility_metadata(
            candidates["cleaned_parquet"],
            upstream=upstream,
            transformation={
                "name": "fresh_anomaly_scan_and_cleaning",
                "anomaly_report": "reports/data_anomaly_stocks_latest.csv",
                "excluded_symbols": int(cleaning["excluded_symbols"]),
                "masked_volume_rows": int(cleaning["masked_volume_rows"]),
                "masked_event_rows": int(cleaning["masked_event_rows"]),
            },
        )
        if convert_to_bin(
            candidates["cleaned_parquet"],
            candidates["cleaned_qlib"],
            overwrite=True,
            calendar_path=candidates["merged_qlib"] / "calendars" / "day.txt",
            source_contract=upstream,
            force_unit_factor=True,
        ):
            raise RuntimeError("cleaned V1 Qlib conversion failed")
        write_csi300(
            candidates["cleaned_qlib"] / "instruments" / "csi300.txt",
            candidates["cleaned_qlib"] / "calendars" / "day.txt",
            root / "stock_lists.json",
        )
        copy_benchmark_features(
            candidates["merged_qlib"],
            candidates["cleaned_qlib"],
            set(BENCHMARK_SYMBOLS),
        )
        validation = validate_compatibility_candidates(
            candidates, source_manifest=source_manifest
        )
        result = {
            "status": "candidate_ready",
            "run_id": layout.run_id,
            "run_root": str(layout.run_root),
            "reports_dir": str(layout.reports_dir),
            "source_snapshot": upstream,
            "stock_lists_sha256": stock_lists_sha256,
            "candidates": {key: str(value) for key, value in candidates.items()},
            "merged_manifest": merged_manifest,
            "benchmark": benchmark,
            "anomaly": anomaly,
            "cleaning": {
                key: value for key, value in cleaning.items() if key != "stats"
            },
            "validation": validation,
        }
        _atomic_json(layout.run_root / "candidate_summary.json", result)
        return result
    except Exception:
        _atomic_json(
            layout.run_root / "candidate_failure.json",
            {
                "status": "failed",
                "run_id": layout.run_id,
                "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        raise


def _validate_publish_paths(
    candidates: Mapping[str, Path],
    targets: Mapping[str, Path],
    old_targets: Mapping[str, Path],
    backup_root: Path,
) -> None:
    for mapping_name, mapping in (
        ("candidates", candidates),
        ("targets", targets),
        ("old_targets", old_targets),
    ):
        missing = sorted(set(ARTIFACT_NAMES) - set(mapping))
        if missing:
            raise ValueError(f"{mapping_name} is missing artifacts: {missing}")
    missing_candidates = [
        str(candidates[name])
        for name in ARTIFACT_NAMES
        if not Path(candidates[name]).is_dir()
    ]
    if missing_candidates:
        raise FileNotFoundError(
            f"V2 compatibility candidates are missing: {missing_candidates}"
        )
    if backup_root.exists() and any(backup_root.iterdir()):
        raise FileExistsError(f"backup root is not empty: {backup_root}")
    for name in ARTIFACT_NAMES:
        target = Path(targets[name])
        old = Path(old_targets[name])
        if target == old:
            raise ValueError(f"target and old target are identical: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(candidates[name]).stat().st_dev != target.parent.stat().st_dev:
            raise OSError(f"candidate and target must share a filesystem: {name}")


def publish_v1_contract(
    *,
    candidates: Mapping[str, Path],
    targets: Mapping[str, Path],
    old_targets: Mapping[str, Path],
    backup_root: Path,
    after_publish: Callable[[], None],
    finalize_publish: Callable[[Mapping[str, Any]], None] | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    candidates = {name: Path(path) for name, path in candidates.items()}
    targets = {name: Path(path) for name, path in targets.items()}
    old_targets = {name: Path(path) for name, path in old_targets.items()}
    backup_root = Path(backup_root)
    publication_lock = Path(lock_path) if lock_path is not None else (
        backup_root.parent / ".v1_compatibility_publish.lock"
    )
    publication_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = publication_lock.open("a+b")
    try:
        lock_file(lock_handle, exclusive=True, blocking=False)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(
            "V1 compatibility publication is already in progress"
        ) from exc
    except Exception:
        lock_handle.close()
        raise

    previous_locations: dict[str, Path] = {}
    promoted: list[str] = []
    archived_as_old: list[str] = []
    result: dict[str, Any] | None = None
    rollback_failure_marker = (
        publication_lock.parent / "v1_compatibility_rollback_failed.json"
    )
    try:
        if rollback_failure_marker.is_file():
            raise RuntimeError(
                "V1 compatibility publication is blocked by an unresolved rollback failure"
            )
        _validate_publish_paths(candidates, targets, old_targets, backup_root)
        backup_root.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_NAMES:
            target = targets[name]
            if not target.exists():
                continue
            old_target = old_targets[name]
            previous = old_target if not old_target.exists() else backup_root / name
            previous.parent.mkdir(parents=True, exist_ok=True)
            target.replace(previous)
            previous_locations[name] = previous
            if previous == old_target:
                archived_as_old.append(name)
        for name in ARTIFACT_NAMES:
            candidates[name].replace(targets[name])
            promoted.append(name)
        after_publish()
        result = {
            "status": "published",
            "published_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "artifacts": list(ARTIFACT_NAMES),
            "archived_as_old": archived_as_old,
            "backup_root": str(backup_root),
        }
        if finalize_publish is not None:
            finalize_publish(result)
    except Exception as publish_error:
        rollback_errors = _rollback_v1_contract(
            promoted=promoted,
            candidates=candidates,
            targets=targets,
            previous_locations=previous_locations,
        )
        if rollback_errors:
            failure = {
                "status": "rollback_failed",
                "failed_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "publish_error_type": type(publish_error).__name__,
                "rollback_errors": rollback_errors,
            }
            marker_error_type: str | None = None
            try:
                _atomic_json(rollback_failure_marker, failure)
            except Exception as marker_error:
                marker_error_type = type(marker_error).__name__
                print(
                    "WARNING: V1 compatibility rollback was incomplete and its "
                    f"failure marker could not be written ({marker_error_type})",
                    file=sys.stderr,
                )
            message = (
                "V1 compatibility publication failed "
                f"({type(publish_error).__name__}) and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
            if marker_error_type:
                message += f"; rollback marker write failed ({marker_error_type})"
            raise RuntimeError(message) from publish_error
        raise
    finally:
        unlock_file(lock_handle)
        lock_handle.close()

    assert result is not None
    return result


def _replace_with_rollback_retries(source: Path, target: Path) -> None:
    last_error: OSError | None = None
    for _ in range(3):
        try:
            source.replace(target)
            return
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _rollback_v1_contract(
    *,
    promoted: Sequence[str],
    candidates: Mapping[str, Path],
    targets: Mapping[str, Path],
    previous_locations: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    for name in reversed(promoted):
        target = Path(targets[name])
        candidate = Path(candidates[name])
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            continue
        try:
            _replace_with_rollback_retries(target, candidate)
        except OSError as exc:
            errors.append(f"restore candidate {name}: {type(exc).__name__}")
    for name in reversed(ARTIFACT_NAMES):
        previous = previous_locations.get(name)
        if previous is None or not previous.exists():
            continue
        target = Path(targets[name])
        if target.exists():
            errors.append(f"restore official {name}: target is still occupied")
            continue
        try:
            _replace_with_rollback_retries(previous, target)
        except OSError as exc:
            errors.append(f"restore official {name}: {type(exc).__name__}")
    return errors


def _official_paths(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    parquet = Path(root) / "parquet" / "ashare_1d"
    qlib = Path(root) / "qlib"
    targets = {
        "merged_parquet": parquet / "merged_baostock_jqdata",
        "merged_qlib": qlib / "cn_data_baostock_jqdata",
        "cleaned_parquet": parquet / "merged_baostock_jqdata_cleaned",
        "cleaned_qlib": qlib / "cn_data_baostock_jqdata_cleaned",
    }
    old_targets = {name: path.with_name(path.name + "_old") for name, path in targets.items()}
    return targets, old_targets


def _restore_file_state(state: Mapping[Path, bytes | None]) -> None:
    errors: list[str] = []
    first_error: OSError | None = None
    for path, previous in state.items():
        temporary: Path | None = None
        try:
            if previous is None:
                path.unlink(missing_ok=True)
                if path.parent.exists():
                    _fsync_directory(path.parent)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(
                    f".{path.name}.{uuid.uuid4().hex}.restore"
                )
                with temporary.open("wb") as handle:
                    handle.write(previous)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
        except OSError as exc:
            if first_error is None:
                first_error = exc
            errors.append(f"{path} (restore): {type(exc).__name__}: {exc}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    if first_error is None:
                        first_error = exc
                    errors.append(
                        f"{path} (temporary cleanup): {type(exc).__name__}: {exc}"
                    )
    if errors:
        raise OSError("failed to restore tracked files: " + "; ".join(errors)) from first_error


def _publish_candidate_result(
    candidate: Mapping[str, Any],
    *,
    quant_data_root: Path,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    current = resolve_dataset(
        "ashare_daily_qfq", quant_data_root=root, require_basis="qfq"
    )
    source = candidate.get("source_snapshot")
    if not isinstance(source, Mapping):
        raise ValueError("compatibility candidate source snapshot is missing")
    for field in ("snapshot_id", "content_sha256"):
        if str(source.get(field) or "") != str(current.manifest.get(field) or ""):
            raise ValueError(
                "compatibility candidate is stale because the V2 source snapshot changed"
            )
    stock_lists_path = root / "stock_lists.json"
    expected_stock_lists_sha256 = str(candidate.get("stock_lists_sha256") or "")
    try:
        current_stock_lists_sha256 = _sha256_file(stock_lists_path)
    except OSError as exc:
        raise ValueError("stock list changed or is unavailable") from exc
    if (
        not expected_stock_lists_sha256
        or expected_stock_lists_sha256 != current_stock_lists_sha256
    ):
        raise ValueError("stock list changed after the compatibility candidate was built")
    raw_candidates = candidate.get("candidates")
    if not isinstance(raw_candidates, Mapping):
        raise ValueError("compatibility candidate paths are missing")
    candidates = {name: Path(str(path)) for name, path in raw_candidates.items()}
    validation = validate_compatibility_candidates(
        candidates, source_manifest=current.manifest
    )
    targets, old_targets = _official_paths(root)
    run_root = Path(str(candidate["run_root"]))
    try:
        run_root.resolve().relative_to(
            (root / "runtime" / "v2_compatibility" / "runs").resolve()
        )
    except ValueError as exc:
        raise ValueError("compatibility run root is outside quant_data runtime") from exc
    tracked_files = [
        root / "DATA_MANIFEST.json",
        root / "README.md",
        root / "parquet" / "ashare_1d" / "README.md",
        root / "qlib" / "README.md",
    ]
    previous_files = {
        path: path.read_bytes() if path.is_file() else None for path in tracked_files
    }

    def sync_manifest() -> None:
        from sync_data_manifest import sync_data_manifest

        sync_data_manifest(
            quant_data_root=root,
            reports_root=Path(reports_root) if reports_root is not None else None,
        )

    def full_result(published: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **published,
            "run_id": candidate["run_id"],
            "run_root": candidate["run_root"],
            "source_snapshot": candidate["source_snapshot"],
            "validation": validation,
            "targets": {name: str(path) for name, path in targets.items()},
            "old_targets": {name: str(path) for name, path in old_targets.items()},
        }

    def write_publish_summary(published: Mapping[str, Any]) -> None:
        _atomic_json(run_root / "publish_summary.json", full_result(published))

    try:
        published = publish_v1_contract(
            candidates=candidates,
            targets=targets,
            old_targets=old_targets,
            backup_root=run_root / "previous_official",
            after_publish=sync_manifest,
            finalize_publish=write_publish_summary,
            lock_path=root
            / "runtime"
            / "locks"
            / "v1_compatibility_publish.lock",
        )
    except Exception as publish_error:
        try:
            _restore_file_state(previous_files)
        except Exception as restore_error:
            raise RuntimeError(
                "V1 compatibility publication failed "
                f"({type(publish_error).__name__}); tracked file restore also "
                f"failed ({type(restore_error).__name__})"
            ) from restore_error
        raise
    return full_result(published)


def publish_candidate_run(
    *,
    quant_data_root: Path,
    run_id: str,
    reports_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    layout = compatibility_layout(root, run_id)
    summary_path = layout.run_root / "candidate_summary.json"
    try:
        candidate = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"validated compatibility candidate is unavailable: {run_id}") from exc
    if not isinstance(candidate, dict) or candidate.get("validation", {}).get("status") != "pass":
        raise ValueError("compatibility candidate has no passing validation")
    return _publish_candidate_result(
        candidate,
        quant_data_root=root,
        reports_root=reports_root,
    )


def rebuild_and_publish_v1_compatibility(
    *,
    quant_data_root: Path,
    reports_root: Path | None = None,
    run_id: str | None = None,
    benchmark_source: str = "tencent",
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    candidate = build_compatibility_candidates(
        quant_data_root=root,
        run_id=run_id,
        benchmark_source=benchmark_source,
    )
    return _publish_candidate_result(
        candidate,
        quant_data_root=root,
        reports_root=reports_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-root", type=Path, default=default_quant_data_root())
    parser.add_argument("--source-bars", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-id", default="ashare_daily_qfq")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--publish-candidate-run")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--benchmark-source", choices=("tencent", "baostock"), default="tencent"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_modes = sum(
            (
                bool(args.publish),
                bool(args.candidate_only),
                bool(args.publish_candidate_run),
            )
        )
        if selected_modes > 1:
            raise ValueError(
                "--publish, --candidate-only and --publish-candidate-run are mutually exclusive"
            )
        if args.publish_candidate_run:
            summary = publish_candidate_run(
                quant_data_root=args.quant_data_root,
                run_id=args.publish_candidate_run,
            )
        elif args.publish:
            summary = rebuild_and_publish_v1_compatibility(
                quant_data_root=args.quant_data_root,
                run_id=args.run_id,
                benchmark_source=args.benchmark_source,
            )
        elif args.candidate_only:
            summary = build_compatibility_candidates(
                quant_data_root=args.quant_data_root,
                run_id=args.run_id,
                benchmark_source=args.benchmark_source,
            )
        else:
            if args.source_bars is None or args.output_dir is None or not args.snapshot_id:
                raise ValueError(
                    "manual conversion requires --source-bars, --output-dir and --snapshot-id"
                )
            summary = build_v1_parquet_output(
                args.source_bars,
                args.output_dir,
                upstream={
                    "dataset_id": args.dataset_id,
                    "snapshot_id": args.snapshot_id,
                },
                overwrite=args.overwrite,
            )
    except (
        DataContractError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"V2 compatibility build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
