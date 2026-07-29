#!/usr/bin/env python3
"""Build an immutable Qlib provider from the governed A-share qfq snapshot."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from _benchmark_sources import (  # noqa: E402
    TENCENT_BENCHMARKS,
    download_tencent_benchmarks,
)
from _symbols import to_qlib_name  # noqa: E402
from merge_to_qlib import convert_to_bin  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    ValidatedResolutionCache,
    atomic_write_json,
    finalize_qlib_snapshot,
    publish_qlib_snapshot,
    resolve_dataset,
    sha256_file,
)


BENCHMARK_FIELDS = ("open", "high", "low", "close", "volume", "factor")
BENCHMARK_START_DATES = {
    "SH000300": "2005-04-08",
    "SH000905": "2007-01-15",
    "SH000906": "2007-01-15",
}
INSTRUMENT_POOLS = ("csi300", "csi500", "csi800")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return payload


def _safe_relative(path: Path, root: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DataContractError(f"{label} must be inside quant_data") from exc


def _read_calendar(path: Path) -> tuple[list[str], bytes]:
    try:
        raw = path.read_bytes()
        calendar = [
            value.strip()
            for value in raw.decode("utf-8").splitlines()
            if value.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise DataContractError("benchmark calendar is unavailable") from exc
    parsed = pd.to_datetime(calendar, errors="coerce")
    if (
        not calendar
        or parsed.isna().any()
        or calendar != sorted(calendar)
        or len(calendar) != len(set(calendar))
    ):
        raise DataContractError("benchmark calendar is invalid")
    return calendar, raw


def _read_feature(path: Path, calendar_size: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"benchmark feature is unavailable: {path.name}")
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) < 2 or not np.isfinite(raw[0]) or raw[0] != int(raw[0]):
        raise DataContractError(f"benchmark feature header is invalid: {path.name}")
    start = int(raw[0])
    values = raw[1:]
    if start < 0 or start + len(values) > calendar_size:
        raise DataContractError(f"benchmark feature range is invalid: {path.name}")
    aligned = np.full(calendar_size, np.nan, dtype=np.float32)
    aligned[start : start + len(values)] = values
    return aligned


def _validate_source_metadata(
    source: Mapping[str, str],
    *,
    benchmark_provider: Path,
    root: Path,
) -> dict[str, str]:
    required = {
        "provider",
        "interface",
        "source_dataset_id",
        "source_path",
        "source_run_id",
        "evidence_path",
        "evidence_sha256",
    }
    normalized = {str(key): str(value) for key, value in source.items()}
    if not required.issubset(normalized) or any(
        not normalized[key].strip() for key in required
    ):
        raise DataContractError("benchmark source evidence is incomplete")
    expected_source_path = _safe_relative(
        benchmark_provider, root, label="benchmark provider"
    )
    if normalized["source_path"] != expected_source_path:
        raise DataContractError("benchmark source path does not match provider")
    evidence_path = Path(normalized["evidence_path"])
    if evidence_path.is_absolute() or ".." in evidence_path.parts:
        raise DataContractError("benchmark source evidence path is invalid")
    if not SHA256_RE.fullmatch(normalized["evidence_sha256"]):
        raise DataContractError("benchmark source evidence hash is invalid")
    resolved_evidence = (root / evidence_path).resolve()
    try:
        resolved_evidence.relative_to(root.resolve())
    except ValueError as exc:
        raise DataContractError("benchmark source evidence path is invalid") from exc
    if resolved_evidence.is_symlink() or not resolved_evidence.is_file():
        raise DataContractError("benchmark source evidence is unavailable")
    if sha256_file(resolved_evidence) != normalized["evidence_sha256"]:
        raise DataContractError("benchmark source evidence hash mismatch")
    return normalized


def native_benchmark_source(
    resolved_provider: Any,
    *,
    root: Path,
) -> dict[str, str]:
    manifest_path = resolved_provider.snapshot_path / "DATASET_MANIFEST.json"
    source = (
        resolved_provider.manifest.get("benchmarks", {}).get("source", {})
        if isinstance(resolved_provider.manifest.get("benchmarks"), Mapping)
        else {}
    )
    origin: dict[str, str] = {}
    if isinstance(source, Mapping):
        if source.get("origin_provider"):
            origin = {
                str(key): str(value)
                for key, value in source.items()
                if str(key).startswith("origin_") and str(value).strip()
            }
        else:
            for key in (
                "provider",
                "interface",
                "source_dataset_id",
                "source_path",
                "source_run_id",
                "evidence_path",
                "evidence_sha256",
            ):
                value = source.get(key)
                if value is not None and str(value).strip():
                    origin[f"origin_{key}"] = str(value)
    return {
        "provider": "quant_workflow",
        "interface": "native_v2_benchmark_copy",
        "source_dataset_id": str(resolved_provider.dataset_id),
        "source_path": _safe_relative(
            resolved_provider.snapshot_path,
            root,
            label="native benchmark provider",
        ),
        "source_run_id": str(resolved_provider.manifest["snapshot_id"]),
        "evidence_path": _safe_relative(
            manifest_path,
            root,
            label="native benchmark evidence",
        ),
        "evidence_sha256": sha256_file(manifest_path),
        **origin,
    }


def _discover_benchmark_source(
    root: Path,
    benchmark_provider: Path,
    upstream: Mapping[str, str],
) -> dict[str, str]:
    runs_root = root / "runtime" / "v2_compatibility" / "runs"
    candidates: list[tuple[datetime, Path, dict[str, Any], dict[str, Any]]] = []
    for publish_path in runs_root.glob("*/publish_summary.json"):
        try:
            published = _read_json(publish_path, label="compatibility publish evidence")
            candidate_path = publish_path.with_name("candidate_summary.json")
            candidate = _read_json(candidate_path, label="compatibility candidate evidence")
            target = Path(str(published.get("targets", {}).get("merged_qlib", "")))
            timestamp = datetime.fromisoformat(str(published.get("published_at", "")))
        except (DataContractError, ValueError, TypeError):
            continue
        if (
            published.get("status") != "published"
            or candidate.get("status") != "candidate_ready"
            or target.expanduser().resolve() != benchmark_provider.resolve()
            or published.get("source_snapshot") != dict(upstream)
            or candidate.get("source_snapshot") != dict(upstream)
            or candidate.get("benchmark", {}).get("status") != "completed"
        ):
            continue
        candidates.append((timestamp, candidate_path, published, candidate))
    if not candidates:
        raise DataContractError(
            "no published benchmark provenance matches the current qfq snapshot"
        )
    _, candidate_path, published, candidate = max(candidates, key=lambda item: item[0])
    source_name = str(candidate["benchmark"].get("source", "")).strip()
    if not source_name:
        raise DataContractError("benchmark provider source is missing")
    interfaces = {
        "tencent": "qt.gtimg.cn historical daily",
        "baostock": "query_history_k_data_plus",
    }
    return {
        "provider": source_name,
        "interface": interfaces.get(source_name, "verified compatibility benchmark"),
        "source_dataset_id": "qlib_ashare_merged_compatibility",
        "source_path": _safe_relative(
            benchmark_provider, root, label="benchmark provider"
        ),
        "source_run_id": str(published.get("run_id", candidate.get("run_id", ""))),
        "evidence_path": _safe_relative(
            candidate_path, root, label="benchmark evidence"
        ),
        "evidence_sha256": sha256_file(candidate_path),
    }


def _import_benchmarks(
    source_provider: Path,
    target_provider: Path,
    *,
    root: Path,
    upstream: Mapping[str, str],
    source_metadata: Mapping[str, str] | None,
) -> dict[str, Any]:
    source_provider = Path(source_provider).expanduser().resolve()
    target_provider = Path(target_provider).expanduser().resolve()
    if source_provider == target_provider:
        raise DataContractError("benchmark source and target providers must differ")
    source_manifest = _read_json(
        source_provider / "PROVIDER_MANIFEST.json",
        label="benchmark provider manifest",
    )
    if source_manifest.get("source_snapshot") != dict(upstream):
        raise DataContractError(
            "benchmark provider does not match the current qfq snapshot"
        )
    source_calendar, source_calendar_bytes = _read_calendar(
        source_provider / "calendars" / "day.txt"
    )
    target_calendar, target_calendar_bytes = _read_calendar(
        target_provider / "calendars" / "day.txt"
    )
    if source_calendar_bytes != target_calendar_bytes or source_calendar != target_calendar:
        raise DataContractError("benchmark calendar differs from native Qlib calendar")
    source = _validate_source_metadata(
        source_metadata
        or _discover_benchmark_source(root, source_provider, upstream),
        benchmark_provider=source_provider,
        root=root,
    )
    source["provider_manifest_sha256"] = sha256_file(
        source_provider / "PROVIDER_MANIFEST.json"
    )
    source["source_manifest_sha256"] = source["provider_manifest_sha256"]
    calendar_sha256 = sha256_file(source_provider / "calendars" / "day.txt")
    calendar_index = pd.Index(source_calendar)
    parsed_calendar = pd.to_datetime(calendar_index, errors="raise")
    symbols: dict[str, Any] = {}
    for symbol, inception in BENCHMARK_START_DATES.items():
        required_positions = np.flatnonzero(parsed_calendar >= pd.Timestamp(inception))
        if not len(required_positions):
            raise DataContractError(
                f"native Qlib calendar has no required dates for {symbol}"
            )
        required_start = int(required_positions[0])
        source_dir = source_provider / "features" / symbol.lower()
        target_dir = target_provider / "features" / symbol.lower()
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise DataContractError(f"benchmark source directory is invalid: {symbol}")
        target_dir.mkdir(parents=True, exist_ok=False)
        features: dict[str, Any] = {}
        validity_masks: dict[str, np.ndarray] = {}
        try:
            for field in BENCHMARK_FIELDS:
                source_path = source_dir / f"{field}.day.bin"
                aligned = _read_feature(source_path, len(source_calendar))
                required = aligned if field == "factor" else aligned[required_start:]
                valid = np.isfinite(required)
                if field == "volume":
                    valid &= required >= 0
                else:
                    valid &= required > 0
                if not bool(valid.all()):
                    raise DataContractError(
                        f"benchmark {symbol} {field} is missing required trading dates"
                    )
                if field != "factor" and required_start > 0:
                    pre_inception = aligned[:required_start]
                    if np.isfinite(pre_inception).any():
                        raise DataContractError(
                            f"benchmark {symbol} {field} fabricates pre-inception values"
                        )
                validity_masks[field] = np.isfinite(aligned)
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                if sha256_file(target_path) != sha256_file(source_path):
                    raise DataContractError(
                        f"benchmark {symbol} {field} copy hash mismatch"
                    )
                features[field] = {
                    "relative_path": target_path.relative_to(target_provider).as_posix(),
                    "size_bytes": target_path.stat().st_size,
                    "sha256": sha256_file(target_path),
                }
            price_masks = [validity_masks[field] for field in BENCHMARK_FIELDS[:-1]]
            if any(not np.array_equal(price_masks[0], mask) for mask in price_masks[1:]):
                raise DataContractError(
                    f"benchmark {symbol} OHLCV fields have inconsistent dates"
                )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        symbols[symbol] = {
            "coverage": {
                "start": source_calendar[required_start],
                "end": source_calendar[-1],
                "observations": len(source_calendar) - required_start,
            },
            "features": features,
        }
    return {
        "calendar_sha256": calendar_sha256,
        "source": source,
        "symbols": symbols,
    }


def _write_feature(path: Path, start_index: int, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.asarray([start_index], dtype="<f4").tofile(stream)
            np.asarray(values, dtype="<f4").tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_downloaded_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataContractError(f"Tencent returned no benchmark data for {symbol}")
    missing_fields = set(BENCHMARK_FIELDS[:-1]).difference(frame.columns)
    if missing_fields:
        raise DataContractError(
            f"Tencent benchmark schema is incomplete for {symbol}: "
            + ", ".join(sorted(missing_fields))
        )
    parsed = pd.to_datetime(frame.index, errors="coerce")
    if parsed.isna().any():
        raise DataContractError(f"Tencent benchmark dates are invalid for {symbol}")
    normalized = frame.loc[:, list(BENCHMARK_FIELDS[:-1])].copy()
    normalized.index = pd.Index(parsed.strftime("%Y-%m-%d"), name="date")
    if normalized.index.has_duplicates:
        raise DataContractError(f"Tencent benchmark dates are duplicated for {symbol}")
    normalized = normalized.sort_index()
    for field in BENCHMARK_FIELDS[:-1]:
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
    return normalized


def _download_benchmark_evidence(
    target_provider: Path,
    *,
    root: Path,
    run_id: str,
    use_system_proxy: bool,
    fallback_error: Exception,
) -> dict[str, Any]:
    calendar, _ = _read_calendar(target_provider / "calendars" / "day.txt")
    calendar_index = pd.Index(calendar)
    parsed_calendar = pd.to_datetime(calendar_index, errors="raise")
    downloaded = download_tencent_benchmarks(
        calendar[0],
        calendar[-1],
        use_system_proxy=use_system_proxy,
    )
    if set(downloaded) != set(BENCHMARK_START_DATES):
        raise DataContractError("Tencent benchmark response is missing required indexes")

    evidence_root = root / "runtime" / "qlib_v2_upgrade" / "runs" / run_id
    raw_root = evidence_root / "benchmarks"
    raw_root.mkdir(parents=True, exist_ok=False)
    calendar_sha256 = sha256_file(target_provider / "calendars" / "day.txt")
    symbols: dict[str, Any] = {}
    source_symbols: dict[str, Any] = {}
    created_targets: list[Path] = []
    try:
        for symbol, inception in BENCHMARK_START_DATES.items():
            normalized = _normalize_downloaded_frame(downloaded[symbol], symbol=symbol)
            required_positions = np.flatnonzero(
                parsed_calendar >= pd.Timestamp(inception)
            )
            if not len(required_positions):
                raise DataContractError(
                    f"native Qlib calendar has no required dates for {symbol}"
                )
            start_index = int(required_positions[0])
            required_calendar = calendar_index[start_index:]
            aligned = normalized.reindex(required_calendar)
            if aligned.isna().any(axis=None):
                missing = aligned.index[aligned.isna().any(axis=1)].tolist()
                raise DataContractError(
                    f"Tencent benchmark {symbol} is missing required trading dates: "
                    + ", ".join(missing[:10])
                )
            prices = aligned[["open", "high", "low", "close"]].to_numpy(
                dtype=float
            )
            volumes = aligned["volume"].to_numpy(dtype=float)
            if (
                not np.isfinite(prices).all()
                or not np.isfinite(volumes).all()
                or (prices <= 0).any()
                or (volumes < 0).any()
            ):
                raise DataContractError(f"Tencent benchmark values are invalid for {symbol}")

            target_dir = target_provider / "features" / symbol.lower()
            target_dir.mkdir(parents=True, exist_ok=False)
            created_targets.append(target_dir)
            features: dict[str, Any] = {}
            for field in BENCHMARK_FIELDS[:-1]:
                target_path = target_dir / f"{field}.day.bin"
                _write_feature(
                    target_path,
                    start_index,
                    aligned[field].to_numpy(dtype="<f4"),
                )
                features[field] = {
                    "relative_path": target_path.relative_to(target_provider).as_posix(),
                    "size_bytes": target_path.stat().st_size,
                    "sha256": sha256_file(target_path),
                }
            factor_path = target_dir / "factor.day.bin"
            _write_feature(
                factor_path,
                0,
                np.ones(len(calendar), dtype="<f4"),
            )
            features["factor"] = {
                "relative_path": factor_path.relative_to(target_provider).as_posix(),
                "size_bytes": factor_path.stat().st_size,
                "sha256": sha256_file(factor_path),
            }

            raw_path = raw_root / f"{symbol}.parquet"
            normalized.reset_index().to_parquet(raw_path, index=False)
            coverage = {
                "start": str(required_calendar[0]),
                "end": str(required_calendar[-1]),
                "observations": len(required_calendar),
            }
            symbols[symbol] = {"coverage": coverage, "features": features}
            source_symbols[symbol] = {
                "provider_code": TENCENT_BENCHMARKS[symbol],
                "coverage": coverage,
                "row_count": len(normalized),
                "raw_path": _safe_relative(raw_path, root, label="benchmark raw file"),
                "raw_sha256": sha256_file(raw_path),
            }
    except Exception:
        for target in created_targets:
            shutil.rmtree(target, ignore_errors=True)
        raise

    retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    evidence_path = evidence_root / "SOURCE_MANIFEST.json"
    atomic_write_json(
        evidence_path,
        {
            "schema_version": 1,
            "status": "complete",
            "run_id": run_id,
            "retrieved_at": retrieved_at,
            "source": {
                "provider": "tencent",
                "interface": "web.ifzq.gtimg.cn appstock fqkline daily",
                "price_basis": "provider_index_series",
            },
            "request_range": {"start": calendar[0], "end": calendar[-1]},
            "calendar_sha256": calendar_sha256,
            "fallback_reason": {
                "error_type": type(fallback_error).__name__,
                "message": str(fallback_error)[:300],
            },
            "symbols": source_symbols,
        },
    )
    evidence_sha256 = sha256_file(evidence_path)
    source = {
        "provider": "tencent",
        "interface": "web.ifzq.gtimg.cn appstock fqkline daily",
        "source_dataset_id": "tencent_index_daily",
        "source_path": _safe_relative(
            evidence_root, root, label="benchmark source evidence"
        ),
        "source_run_id": run_id,
        "evidence_path": _safe_relative(
            evidence_path, root, label="benchmark source evidence"
        ),
        "evidence_sha256": evidence_sha256,
        "source_manifest_sha256": evidence_sha256,
        "retrieved_at": retrieved_at,
    }
    return {
        "calendar_sha256": calendar_sha256,
        "source": source,
        "symbols": symbols,
    }


def _install_instrument_pools(target_provider: Path, *, root: Path) -> dict[str, Any]:
    source_path = root / "stock_lists.json"
    source = _read_json(source_path, label="stock pool source")
    all_path = target_provider / "instruments" / "all.txt"
    ranges: dict[str, tuple[str, str]] = {}
    try:
        for raw_line in all_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) != 3:
                raise DataContractError("native Qlib all-instrument pool is invalid")
            symbol = parts[0].upper()
            if symbol in ranges:
                raise DataContractError("native Qlib all-instrument pool has duplicates")
            ranges[symbol] = (parts[1], parts[2])
    except OSError as exc:
        raise DataContractError("native Qlib all-instrument pool is unavailable") from exc
    if not ranges:
        raise DataContractError("native Qlib all-instrument pool is empty")

    pool_evidence: dict[str, Any] = {}
    for pool_name in INSTRUMENT_POOLS:
        raw_symbols = source.get(pool_name)
        if not isinstance(raw_symbols, list) or not raw_symbols:
            raise DataContractError(f"stock_lists.json is missing {pool_name}")
        selected: list[str] = []
        seen: set[str] = set()
        missing: list[str] = []
        for raw_symbol in raw_symbols:
            symbol = to_qlib_name(raw_symbol)
            if symbol in seen:
                continue
            seen.add(symbol)
            if symbol not in ranges:
                missing.append(symbol)
                continue
            selected.append(symbol)
        if missing:
            raise DataContractError(
                f"{pool_name} contains symbols absent from native Qlib: "
                + ", ".join(missing[:10])
            )
        if not selected:
            raise DataContractError(f"{pool_name} has no native Qlib instruments")
        output_path = target_provider / "instruments" / f"{pool_name}.txt"
        output_path.write_text(
            "".join(
                f"{symbol}\t{ranges[symbol][0]}\t{ranges[symbol][1]}\n"
                for symbol in selected
            ),
            encoding="utf-8",
        )
        pool_evidence[pool_name] = {
            "relative_path": output_path.relative_to(target_provider).as_posix(),
            "instrument_count": len(selected),
            "sha256": sha256_file(output_path),
        }
    return {
        "source": {
            "dataset_id": "stock_lists",
            "path": _safe_relative(source_path, root, label="stock pool source"),
            "sha256": sha256_file(source_path),
        },
        "pools": pool_evidence,
    }


def build_qlib_v2(
    *,
    quant_data_root: Path,
    publish: bool = False,
    benchmark_provider: Path | None = None,
    benchmark_source: Mapping[str, str] | None = None,
    benchmark_use_system_proxy: bool = False,
    publication_lock_held: bool = False,
    validated_cache: ValidatedResolutionCache | None = None,
) -> Path:
    root = Path(quant_data_root).expanduser().resolve()
    qfq = resolve_dataset(
        "ashare_daily_qfq",
        quant_data_root=root,
        require_basis="qfq",
        validated_cache=validated_cache,
    )
    run_id = (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    snapshot_id = f"qlib-ashare-v2-{run_id}"
    staging = root / "staging" / f"qlib-v2-{run_id}" / snapshot_id
    upstream = {
        "dataset_id": qfq.dataset_id,
        "snapshot_id": str(qfq.manifest["snapshot_id"]),
        "content_sha256": str(qfq.manifest["content_sha256"]),
        "price_basis": str(qfq.manifest["price_basis"]),
    }
    result = convert_to_bin(
        qfq.snapshot_path / "bars",
        staging,
        overwrite=True,
        source_contract=upstream,
        force_unit_factor=True,
    )
    if result != 0:
        raise DataContractError("Qlib v2 conversion failed")
    benchmark_provider_is_explicit = benchmark_provider is not None
    if benchmark_provider_is_explicit:
        benchmark_evidence = _import_benchmarks(
            Path(benchmark_provider),
            staging,
            root=root,
            upstream=upstream,
            source_metadata=benchmark_source,
        )
    else:
        fallback_error: DataContractError | None = None
        try:
            current_native = resolve_dataset(
                "qlib_ashare_v2",
                quant_data_root=root,
                require_basis="qfq",
                validated_cache=validated_cache,
            )
            benchmark_evidence = _import_benchmarks(
                current_native.snapshot_path,
                staging,
                root=root,
                upstream=upstream,
                source_metadata=native_benchmark_source(current_native, root=root),
            )
        except DataContractError as exc:
            fallback_error = exc
        if fallback_error is not None:
            for symbol in BENCHMARK_START_DATES:
                shutil.rmtree(
                    staging / "features" / symbol.lower(), ignore_errors=True
                )
            benchmark_evidence = _download_benchmark_evidence(
                staging,
                root=root,
                run_id=run_id,
                use_system_proxy=benchmark_use_system_proxy,
                fallback_error=fallback_error,
            )
    instrument_pool_evidence = _install_instrument_pools(staging, root=root)
    finalize_qlib_snapshot(
        staging,
        dataset_id="qlib_ashare_v2",
        snapshot_id=snapshot_id,
        source={
            "provider": "quant_workflow",
            "interface": "v2_snapshot_to_qlib",
        },
        source_retrieved_at=str(qfq.manifest["source_retrieved_at"]),
        producer={"script": "build_qlib_v2.py", "version": "5"},
        upstream_snapshot=upstream,
        benchmark_evidence=benchmark_evidence,
        instrument_pool_evidence=instrument_pool_evidence,
    )
    if not publish:
        return staging
    dataset_root = root / "qlib" / "cn_data_ashare_v2"
    published = publish_qlib_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=root,
        update_root_catalog=True,
        publication_lock_held=publication_lock_held,
        validated_cache=validated_cache,
    )
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-data-root", type=Path, default=default_quant_data_root()
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--benchmark-provider",
        type=Path,
        default=None,
        help="Verified Qlib provider supplying the three index benchmarks.",
    )
    parser.add_argument(
        "--benchmark-use-system-proxy",
        action="store_true",
        help="Allow the direct Tencent benchmark fallback to use system proxies.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = build_qlib_v2(
            quant_data_root=args.quant_data_root,
            publish=args.publish,
            benchmark_provider=args.benchmark_provider,
            benchmark_use_system_proxy=args.benchmark_use_system_proxy,
        )
    except (DataContractError, OSError, ValueError, KeyError) as exc:
        print(f"Qlib v2 build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
