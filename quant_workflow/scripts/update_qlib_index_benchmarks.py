#!/usr/bin/env python3
"""Add ChiNext and STAR 50 indexes to native Qlib snapshots without rebuilding stocks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
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
from build_qlib_v2 import (  # noqa: E402
    BENCHMARK_FIELDS,
    BENCHMARK_START_DATES,
    _normalize_downloaded_frame,
    _require_valid_ohlc,
    _write_feature,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    dataset_publication_lock,
    finalize_qlib_snapshot,
    publish_snapshot_group,
    resolve_dataset,
    sha256_file,
)


TARGET_INDEXES = {
    "SZ399006": {"name": "ChiNext Index", "inception": "2010-06-01"},
    "SH000688": {"name": "STAR 50", "inception": "2020-01-02"},
}
TARGET_SYMBOLS = tuple(TARGET_INDEXES)
EXPECTED_BENCHMARK_SYMBOLS = tuple(BENCHMARK_START_DATES)
DATASET_ROOTS = {
    "qlib_ashare_v2": Path("qlib/cn_data_ashare_v2"),
    "qlib_ashare_cleaned_v2": Path("qlib/cn_data_ashare_cleaned_v2"),
}
SCRIPT_VERSION = "1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise DataContractError(f"JSON file must contain an object: {path.name}")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DataContractError("index evidence must stay inside quant_data") from exc


@contextmanager
def _single_instance_lock(root: Path):
    lock_path = root / "runtime" / "locks" / "qlib_index_benchmark_update.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        try:
            lock_file(stream, exclusive=True, blocking=False)
        except (BlockingIOError, OSError) as exc:
            raise DataContractError(
                "another Qlib index update is already running"
            ) from exc
        yield
    finally:
        try:
            unlock_file(stream)
        finally:
            stream.close()


def _clone_snapshot(source: Path, destination: Path) -> None:
    if destination.exists():
        raise DataContractError("index update staging directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["/bin/cp", "-cR", str(source), str(destination)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return
        except (OSError, subprocess.SubprocessError):
            shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _calendar(snapshot: Path) -> tuple[list[str], pd.DatetimeIndex]:
    try:
        days = [
            value.strip()
            for value in (snapshot / "calendars/day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if value.strip()
        ]
    except OSError as exc:
        raise DataContractError("Qlib calendar is unavailable") from exc
    parsed = pd.to_datetime(days, errors="coerce")
    if not days or parsed.isna().any() or days != sorted(set(days)):
        raise DataContractError("Qlib calendar is invalid")
    return days, pd.DatetimeIndex(parsed)


def _validate_target_frames(
    frames: Mapping[str, pd.DataFrame], calendar_days: list[str]
) -> dict[str, pd.DataFrame]:
    if set(frames) != set(TARGET_SYMBOLS):
        raise DataContractError(
            "index source returned symbols outside the requested pair"
        )
    calendar_index = pd.Index(calendar_days)
    parsed_calendar = pd.to_datetime(calendar_index, errors="raise")
    normalized: dict[str, pd.DataFrame] = {}
    for symbol in TARGET_SYMBOLS:
        frame = _normalize_downloaded_frame(
            frames[symbol], symbol=symbol, provider="targeted_index_source"
        )
        inception = pd.Timestamp(str(TARGET_INDEXES[symbol]["inception"]))
        required_positions = np.flatnonzero(parsed_calendar >= inception)
        if not len(required_positions):
            raise DataContractError(f"Qlib calendar has no dates for {symbol}")
        required_calendar = calendar_index[int(required_positions[0]) :]
        aligned = frame.reindex(required_calendar)
        if aligned.isna().any(axis=None):
            missing = aligned.index[aligned.isna().any(axis=1)].tolist()
            raise DataContractError(
                f"index {symbol} is missing required dates: "
                + ", ".join(str(value) for value in missing[:10])
            )
        values = {
            field: aligned[field].to_numpy(dtype=float)
            for field in BENCHMARK_FIELDS[:-1]
        }
        prices = aligned[["open", "high", "low", "close"]].to_numpy(dtype=float)
        volumes = aligned["volume"].to_numpy(dtype=float)
        if (
            not np.isfinite(prices).all()
            or not np.isfinite(volumes).all()
            or (prices <= 0).any()
            or (volumes < 0).any()
        ):
            raise DataContractError(f"index {symbol} contains invalid values")
        _require_valid_ohlc(values, required_calendar.tolist(), symbol=symbol)
        normalized[symbol] = frame
    return normalized


def _download_pair(
    start_date: str,
    end_date: str,
    *,
    use_system_proxy: bool,
    tencent_download: Callable[..., dict[str, pd.DataFrame]],
    eastmoney_download: Callable[..., dict[str, pd.DataFrame]],
    source_order: Sequence[str] = ("eastmoney", "tencent"),
    validator: Callable[[Mapping[str, pd.DataFrame]], dict[str, pd.DataFrame]]
    | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    attempts: list[dict[str, str]] = []
    source_specs = {
        "tencent": (
            "tencent",
            "web.ifzq.gtimg.cn appstock fqkline daily",
            TENCENT_BENCHMARKS,
            tencent_download,
        ),
        "eastmoney": (
            "eastmoney",
            "push2his.eastmoney.com index kline daily",
            EASTMONEY_BENCHMARKS,
            eastmoney_download,
        ),
    }
    if set(source_order) != set(source_specs) or len(source_order) != len(source_specs):
        raise ValueError("source_order must contain eastmoney and tencent exactly once")
    for source_name in source_order:
        provider, interface, provider_codes, downloader = source_specs[source_name]
        try:
            frames = downloader(
                start_date,
                end_date,
                use_system_proxy=use_system_proxy,
                symbols=TARGET_SYMBOLS,
            )
            if set(frames) != set(TARGET_SYMBOLS):
                raise DataContractError("source returned an unexpected symbol set")
            if validator is not None:
                frames = validator(frames)
            return frames, {
                "provider": provider,
                "interface": interface,
                "provider_codes": {
                    symbol: provider_codes[symbol] for symbol in TARGET_SYMBOLS
                },
                "attempts": attempts + [{"provider": provider, "status": "selected"}],
            }
        except Exception as exc:
            attempts.append(
                {
                    "provider": provider,
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
    raise DataContractError(
        "both targeted index sources failed: " + json.dumps(attempts)
    )


def _write_raw_evidence(
    frames: Mapping[str, pd.DataFrame],
    *,
    evidence_dir: Path,
    root: Path,
    run_id: str,
    source: Mapping[str, Any],
    canonical: Any,
    cleaned: Any,
) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}
    for symbol in TARGET_SYMBOLS:
        path = evidence_dir / f"{symbol}.parquet"
        output = frames[symbol].reset_index(names="date")
        output.to_parquet(path, index=False)
        files[symbol] = {
            "relative_path": _relative(path, root),
            "sha256": sha256_file(path),
            "row_count": int(len(output)),
            "start": str(output.iloc[0]["date"]),
            "end": str(output.iloc[-1]["date"]),
        }
    manifest_path = evidence_dir / "SOURCE_MANIFEST.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "operation": "targeted_qlib_index_addition",
            "run_id": run_id,
            "retrieved_at": _now(),
            "requested_symbols": list(TARGET_SYMBOLS),
            "requested_symbol_count": len(TARGET_SYMBOLS),
            "stock_symbols_requested": 0,
            "etf_symbols_requested": 0,
            "source": dict(source),
            "base_snapshots": {
                canonical.dataset_id: {
                    "snapshot_id": canonical.manifest["snapshot_id"],
                    "content_sha256": canonical.manifest["content_sha256"],
                },
                cleaned.dataset_id: {
                    "snapshot_id": cleaned.manifest["snapshot_id"],
                    "content_sha256": cleaned.manifest["content_sha256"],
                },
            },
            "files": files,
        },
    )
    digest = sha256_file(manifest_path)
    return {
        "provider": str(source["provider"]),
        "interface": "targeted Qlib index merge; " + str(source["interface"]),
        "source_dataset_id": "targeted_cn_index_daily",
        "source_path": _relative(evidence_dir, root),
        "source_run_id": run_id,
        "evidence_path": _relative(manifest_path, root),
        "evidence_sha256": digest,
        "source_manifest_sha256": digest,
        "retrieved_at": _now(),
    }


def _write_target_features(
    snapshot: Path,
    frames: Mapping[str, pd.DataFrame],
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    calendar_days, parsed_calendar = _calendar(snapshot)
    calendar_index = pd.Index(calendar_days)
    existing = _read_json(snapshot / "PROVIDER_MANIFEST.json").get("benchmarks", {})
    existing_symbols = (
        existing.get("symbols") if isinstance(existing, Mapping) else None
    )
    if not isinstance(existing_symbols, Mapping):
        raise DataContractError("base Qlib benchmark evidence is unavailable")
    symbols = json.loads(json.dumps(dict(existing_symbols)))
    for symbol in TARGET_SYMBOLS:
        target_dir = snapshot / "features" / symbol.lower()
        if target_dir.exists():
            raise DataContractError(
                f"target index already exists in partial snapshot: {symbol}"
            )
        target_dir.mkdir(parents=True)
        inception = pd.Timestamp(str(TARGET_INDEXES[symbol]["inception"]))
        positions = np.flatnonzero(parsed_calendar >= inception)
        start_index = int(positions[0])
        required_calendar = calendar_index[start_index:]
        aligned = frames[symbol].reindex(required_calendar)
        features: dict[str, Any] = {}
        for field in BENCHMARK_FIELDS:
            if field == "factor":
                write_start = 0
                values = np.ones(len(calendar_days), dtype=float)
            else:
                write_start = start_index
                values = aligned[field].to_numpy(dtype=float)
            path = target_dir / f"{field}.day.bin"
            _write_feature(path, write_start, values)
            features[field] = {
                "relative_path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        symbols[symbol] = {
            "coverage": {
                "start": calendar_days[start_index],
                "end": calendar_days[-1],
                "observations": len(calendar_days) - start_index,
            },
            "features": features,
        }
    if set(symbols) != set(EXPECTED_BENCHMARK_SYMBOLS):
        raise DataContractError(
            "merged benchmark set is not exactly the required five indexes"
        )
    return {
        "calendar_sha256": sha256_file(snapshot / "calendars/day.txt"),
        "required_symbols": list(EXPECTED_BENCHMARK_SYMBOLS),
        "source": dict(source_metadata),
        "symbols": symbols,
    }


def _assert_existing_features_reused(base: Path, staged: Path) -> None:
    base_index = pd.read_parquet(base / "FILE_INDEX.parquet")
    staged_index = pd.read_parquet(staged / "FILE_INDEX.parquet")
    new_prefixes = tuple(f"features/{symbol.lower()}/" for symbol in TARGET_SYMBOLS)
    base_features = base_index[
        base_index["relative_path"].astype(str).str.startswith("features/")
    ].set_index("relative_path")
    staged_features = staged_index[
        staged_index["relative_path"].astype(str).str.startswith("features/")
        & ~staged_index["relative_path"].astype(str).str.startswith(new_prefixes)
    ].set_index("relative_path")
    if base_features.index.tolist() != staged_features.index.tolist():
        raise DataContractError(
            "existing Qlib feature file set changed during index merge"
        )
    if not base_features["sha256"].equals(staged_features["sha256"]):
        raise DataContractError(
            "existing Qlib feature content changed during index merge"
        )


def _identity(resolved: Any) -> dict[str, str]:
    return {
        "dataset_id": str(resolved.dataset_id),
        "snapshot_id": str(resolved.manifest["snapshot_id"]),
        "content_sha256": str(resolved.manifest["content_sha256"]),
        "price_basis": str(resolved.manifest["price_basis"]),
    }


def _targets_already_present(canonical: Any, cleaned: Any) -> bool:
    for resolved in (canonical, cleaned):
        symbols = resolved.manifest.get("benchmarks", {}).get("symbols", {})
        if set(TARGET_SYMBOLS).difference(symbols):
            return False
        if resolved.manifest.get("benchmarks", {}).get("required_symbols") != list(
            EXPECTED_BENCHMARK_SYMBOLS
        ):
            return False
    return True


def update_qlib_index_benchmarks(
    *,
    quant_data_root: Path,
    publish: bool,
    use_system_proxy: bool = False,
    tencent_download: Callable[
        ..., dict[str, pd.DataFrame]
    ] = download_tencent_benchmarks,
    eastmoney_download: Callable[
        ..., dict[str, pd.DataFrame]
    ] = download_eastmoney_benchmarks,
) -> dict[str, Any]:
    root = Path(quant_data_root).expanduser().resolve()
    with _single_instance_lock(root):
        canonical = resolve_dataset(
            "qlib_ashare_v2", quant_data_root=root, require_basis="qfq"
        )
        cleaned = resolve_dataset(
            "qlib_ashare_cleaned_v2", quant_data_root=root, require_basis="qfq"
        )
        if _targets_already_present(canonical, cleaned):
            return {
                "status": "already_present",
                "network_requests_for_symbols": [],
                "canonical_snapshot_id": canonical.manifest["snapshot_id"],
                "cleaned_snapshot_id": cleaned.manifest["snapshot_id"],
            }
        if cleaned.manifest.get("dependency_snapshots") != [_identity(canonical)]:
            raise DataContractError(
                "cleaned Qlib does not depend on the current canonical Qlib"
            )
        canonical_calendar, _ = _calendar(canonical.snapshot_path)
        cleaned_calendar, _ = _calendar(cleaned.snapshot_path)
        if canonical_calendar != cleaned_calendar:
            raise DataContractError("canonical and cleaned Qlib calendars differ")
        frames, source = _download_pair(
            canonical_calendar[0],
            canonical_calendar[-1],
            use_system_proxy=use_system_proxy,
            tencent_download=tencent_download,
            eastmoney_download=eastmoney_download,
            validator=lambda values: _validate_target_frames(
                values, canonical_calendar
            ),
        )
        run_id = (
            datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        staging_run = root / "staging" / f"qlib-index-update-{run_id}"
        canonical_staging = staging_run / f"qlib-ashare-v2-index-{run_id}"
        cleaned_staging = staging_run / f"qlib-ashare-cleaned-v2-index-{run_id}"
        evidence_dir = root / "runtime" / "qlib_index_updates" / "runs" / run_id
        source_metadata = _write_raw_evidence(
            frames,
            evidence_dir=evidence_dir,
            root=root,
            run_id=run_id,
            source=source,
            canonical=canonical,
            cleaned=cleaned,
        )
        try:
            _clone_snapshot(canonical.snapshot_path, canonical_staging)
            _clone_snapshot(cleaned.snapshot_path, cleaned_staging)
            canonical_evidence = _write_target_features(
                canonical_staging, frames, source_metadata
            )
            cleaned_evidence = _write_target_features(
                cleaned_staging, frames, source_metadata
            )
            canonical_provider = _read_json(
                canonical_staging / "PROVIDER_MANIFEST.json"
            )
            cleaned_provider = _read_json(cleaned_staging / "PROVIDER_MANIFEST.json")
            upstream = canonical_provider.get("source_snapshot")
            if (
                not isinstance(upstream, Mapping)
                or cleaned_provider.get("source_snapshot") != upstream
            ):
                raise DataContractError("Qlib upstream lineage differs")
            retrieved_at = _now()
            finalize_qlib_snapshot(
                canonical_staging,
                dataset_id="qlib_ashare_v2",
                snapshot_id=canonical_staging.name,
                source={
                    "provider": "quant_workflow",
                    "interface": "targeted_index_merge",
                },
                source_retrieved_at=retrieved_at,
                producer={
                    "script": "update_qlib_index_benchmarks.py",
                    "version": SCRIPT_VERSION,
                },
                upstream_snapshot=dict(upstream),
                benchmark_evidence=canonical_evidence,
                instrument_pool_evidence=canonical_provider.get("instrument_pools"),
            )
            canonical_manifest = _read_json(canonical_staging / "DATASET_MANIFEST.json")
            new_canonical_identity = {
                "dataset_id": "qlib_ashare_v2",
                "snapshot_id": canonical_manifest["snapshot_id"],
                "content_sha256": canonical_manifest["content_sha256"],
                "price_basis": "qfq",
            }
            finalize_qlib_snapshot(
                cleaned_staging,
                dataset_id="qlib_ashare_cleaned_v2",
                snapshot_id=cleaned_staging.name,
                source={
                    "provider": "quant_workflow",
                    "interface": "targeted_index_merge",
                },
                source_retrieved_at=retrieved_at,
                producer={
                    "script": "update_qlib_index_benchmarks.py",
                    "version": SCRIPT_VERSION,
                },
                upstream_snapshot=dict(upstream),
                benchmark_evidence=cleaned_evidence,
                instrument_pool_evidence=cleaned_provider.get("instrument_pools"),
                transformation_evidence=cleaned_provider.get("transformation"),
                dependency_snapshots=[new_canonical_identity],
            )
            _assert_existing_features_reused(canonical.snapshot_path, canonical_staging)
            _assert_existing_features_reused(cleaned.snapshot_path, cleaned_staging)
            result: dict[str, Any] = {
                "status": "staged",
                "run_id": run_id,
                "requested_symbols": list(TARGET_SYMBOLS),
                "source": source["provider"],
                "stock_symbols_downloaded": 0,
                "etf_symbols_downloaded": 0,
                "index_symbols_downloaded": len(TARGET_SYMBOLS),
                "canonical_base_snapshot_id": canonical.manifest["snapshot_id"],
                "cleaned_base_snapshot_id": cleaned.manifest["snapshot_id"],
                "canonical_snapshot_id": canonical_staging.name,
                "cleaned_snapshot_id": cleaned_staging.name,
                "evidence_path": _relative(evidence_dir / "SOURCE_MANIFEST.json", root),
            }
            if publish:
                with dataset_publication_lock(root):
                    current_canonical = resolve_dataset(
                        "qlib_ashare_v2", quant_data_root=root, require_basis="qfq"
                    )
                    current_cleaned = resolve_dataset(
                        "qlib_ashare_cleaned_v2",
                        quant_data_root=root,
                        require_basis="qfq",
                    )
                    if _identity(current_canonical) != _identity(
                        canonical
                    ) or _identity(current_cleaned) != _identity(cleaned):
                        raise DataContractError(
                            "Qlib snapshots changed during index update"
                        )
                    published = publish_snapshot_group(
                        [
                            (canonical_staging, root / DATASET_ROOTS["qlib_ashare_v2"]),
                            (
                                cleaned_staging,
                                root / DATASET_ROOTS["qlib_ashare_cleaned_v2"],
                            ),
                        ],
                        quant_data_root=root,
                        update_root_catalog=True,
                        publication_lock_held=True,
                    )
                result["status"] = "published"
                result["published_paths"] = [
                    _relative(path, root) for path in published
                ]
                shutil.rmtree(staging_run, ignore_errors=True)
            return result
        except Exception:
            shutil.rmtree(staging_run, ignore_errors=True)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-data-root", type=Path, default=default_quant_data_root()
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    status_path = reports / "qlib_index_benchmark_update_latest.json"
    try:
        result = update_qlib_index_benchmarks(
            quant_data_root=args.quant_data_root,
            publish=args.publish,
            use_system_proxy=args.use_system_proxy,
        )
    except Exception as exc:
        atomic_write_json(
            status_path,
            {
                "status": "failed",
                "failed_at": _now(),
                "requested_symbols": list(TARGET_SYMBOLS),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(f"Qlib index update failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    atomic_write_json(status_path, {**result, "completed_at": _now()})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
