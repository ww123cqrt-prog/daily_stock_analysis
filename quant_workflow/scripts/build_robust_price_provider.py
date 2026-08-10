#!/usr/bin/env python3
"""Build the legacy robust-strategy compatibility product.

The formal robust workflow uses ``quant_data``'s direct Parquet Provider. This
script is retained only for an explicitly identified old frozen run and may
therefore create a physical ``.bin`` tree under its isolated compatibility
product directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import add_index_benchmark  # noqa: E402
import apply_data_cleaning  # noqa: E402
import import_robust_price_bundle as price_importer  # noqa: E402
import merge_to_qlib  # noqa: E402
import report_data_anomalies  # noqa: E402
import validate_robust_price_bundle as price_validator  # noqa: E402
from _paths import quant_data_root, reports_root  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    quant_data_root() / "runtime" / "robust_strategy" / "price_products"
)
DEFAULT_STATUS_DIR = reports_root() / "robust_strategy"
BUILD_VERSION = 1
BUILD_FILENAME = "BUILD.json"
CONTENT_FILENAME = "CONTENT.json"
BENCHMARK_SYMBOLS = tuple(add_index_benchmark.BENCHMARKS)


class RobustPriceProviderError(ValueError):
    def __init__(self, reason_code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = details


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_entries(
    root: Path, *, excluded_names: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    excluded = excluded_names or set()
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.name in excluded and "/" not in relative:
            continue
        if path.is_symlink():
            raise RobustPriceProviderError(
                "symlink_forbidden", "price product contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise RobustPriceProviderError(
                "non_regular_file", "price product contains a non-regular file"
            )
        entries[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256": price_validator.file_sha256(path),
        }
    return entries


def _entries_sha256(entries: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _benchmark_fingerprint(provider: Path, required_start_date: str) -> str:
    provider = Path(provider)
    add_index_benchmark.add_benchmark(
        provider,
        reuse_existing=True,
        required_start_date=required_start_date,
    )
    paths = [provider / "calendars" / "day.txt"]
    for symbol in BENCHMARK_SYMBOLS:
        paths.extend(sorted((provider / "features" / symbol.lower()).glob("*.day.bin")))
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RobustPriceProviderError(
                "benchmark_provider_invalid",
                "verified benchmark provider contains an unsafe file",
            )
        relative = path.relative_to(provider).as_posix()
        entries[relative] = {
            "bytes": int(path.stat().st_size),
            "sha256": price_validator.file_sha256(path),
        }
    return _entries_sha256(entries)


def _make_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise RobustPriceProviderError(
                "symlink_forbidden", "price product contains a symlink"
            )
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(path, mode)
    os.chmod(
        root,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )


def _tree_is_read_only(root: Path) -> bool:
    write_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        return not any(
            path.is_symlink() or path.stat().st_mode & write_mask
            for path in [root, *root.rglob("*")]
        )
    except OSError:
        return False


def _make_tree_writable(root: Path) -> None:
    if root.is_symlink() or not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        try:
            os.chmod(
                path,
                path.stat().st_mode
                | stat.S_IWUSR
                | (stat.S_IXUSR if path.is_dir() else 0),
            )
        except FileNotFoundError:
            continue


def _remove_tree(root: Path) -> None:
    if root.is_symlink():
        root.unlink()
    elif root.exists():
        _make_tree_writable(root)
        shutil.rmtree(root)


def _load_imported_bundle(bundle_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import_path = bundle_dir / price_importer.IMPORT_FILENAME
    try:
        metadata = json.loads(import_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustPriceProviderError(
            "price_bundle_invalid", "imported price bundle metadata is invalid"
        ) from exc
    required = {
        "bundle_key",
        "source_manifest_sha256",
        "membership_snapshot_hash",
    }
    if not required.issubset(metadata):
        raise RobustPriceProviderError(
            "price_bundle_invalid", "imported price bundle metadata is incomplete"
        )
    verified = price_importer.verify_imported_bundle(
        bundle_dir=bundle_dir,
        bundle_key=str(metadata["bundle_key"]),
        source_manifest_sha256=str(metadata["source_manifest_sha256"]),
        membership_snapshot_hash=str(metadata["membership_snapshot_hash"]),
    )
    if verified is None:
        raise RobustPriceProviderError(
            "price_bundle_invalid", "imported price bundle failed integrity validation"
        )
    return metadata, verified


def _build_key(
    *, price_bundle_key: str, benchmark_fingerprint: str, required_start_date: str
) -> str:
    payload = {
        "build_version": BUILD_VERSION,
        "price_bundle_key": price_bundle_key,
        "benchmark_fingerprint": benchmark_fingerprint,
        "required_start_date": required_start_date,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _product_is_reusable(product: Path, key: str) -> dict[str, Any] | None:
    build_path = product / BUILD_FILENAME
    content_path = product / CONTENT_FILENAME
    if (
        product.is_symlink()
        or not product.is_dir()
        or not build_path.is_file()
        or not content_path.is_file()
    ):
        return None
    try:
        build = json.loads(build_path.read_text(encoding="utf-8"))
        content = json.loads(content_path.read_text(encoding="utf-8"))
        if (
            build.get("build_version") != BUILD_VERSION
            or build.get("product_key") != key
            or content.get("build_version") != BUILD_VERSION
            or content.get("product_key") != key
            or build.get("content_manifest_sha256")
            != price_validator.file_sha256(content_path)
            or not isinstance(content.get("files"), dict)
        ):
            return None
        actual = _tree_entries(
            product, excluded_names={BUILD_FILENAME, CONTENT_FILENAME}
        )
        if actual != content["files"]:
            return None
        if content.get("tree_sha256") != _entries_sha256(actual):
            return None
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        RobustPriceProviderError,
    ):
        return None
    if not _tree_is_read_only(product):
        return None
    return build


def _write_anomaly_inputs(price_dir: Path, anomaly_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    summary, events, coverage = report_data_anomalies.scan_merged_files(price_dir)
    if summary.empty:
        raise RobustPriceProviderError(
            "anomaly_scan_empty", "anomaly scan found no price files"
        )
    if events.empty:
        events = pd.DataFrame(
            columns=["symbol", "issue_type", "severity", "count", "sample_dates", "detail"]
        )
    stocks_path = anomaly_dir / "data_anomaly_stocks.csv"
    events_path = anomaly_dir / "data_anomaly_events.csv"
    summary.to_csv(stocks_path, index=False)
    events.to_csv(events_path, index=False)
    return stocks_path, events_path, coverage


def build_robust_price_provider(
    *,
    price_bundle_dir: Path,
    benchmark_provider: Path,
    output_root: Path,
    required_start_date: str = "2014-01-02",
) -> dict[str, Any]:
    price_bundle_dir = Path(price_bundle_dir).expanduser().resolve()
    benchmark_provider = Path(benchmark_provider).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root == price_bundle_dir or price_bundle_dir in output_root.parents:
        raise RobustPriceProviderError(
            "output_inside_price_bundle", "output root must not be inside the price bundle"
        )
    import_metadata, price_verified = _load_imported_bundle(price_bundle_dir)
    benchmark_fingerprint = _benchmark_fingerprint(
        benchmark_provider, required_start_date
    )
    key = _build_key(
        price_bundle_key=str(import_metadata["bundle_key"]),
        benchmark_fingerprint=benchmark_fingerprint,
        required_start_date=required_start_date,
    )
    product = output_root / key[:16]
    existing = _product_is_reusable(product, key)
    reused = existing is not None
    if not reused:
        if product.exists():
            raise RobustPriceProviderError(
                "existing_product_invalid",
                "existing robust price product failed integrity validation",
            )
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = output_root / f".{key[:16]}.{os.getpid()}.tmp"
        if temporary.exists():
            _remove_tree(temporary)
        temporary.mkdir()
        try:
            price_dir = price_bundle_dir / "prices"
            anomaly_dir = temporary / "anomalies"
            cleaned_dir = temporary / "cleaned_parquet"
            qlib_dir = temporary / "qlib_provider"
            stocks_csv, events_csv, coverage = _write_anomaly_inputs(
                price_dir, anomaly_dir
            )
            cleaning_plan = apply_data_cleaning.load_cleaning_plan(
                stocks_csv, events_csv
            )
            cleaning_summary = apply_data_cleaning.build_cleaned_parquet(
                price_dir,
                cleaned_dir,
                cleaning_plan,
                overwrite=True,
            )
            if cleaning_summary["written_symbols"] <= 0:
                raise RobustPriceProviderError(
                    "cleaned_price_product_empty",
                    "cleaning produced no usable price symbols",
                )
            cleaning_summary["source_dir"] = "imported_price_bundle/prices"
            cleaning_summary["cleaned_dir"] = "cleaned_parquet"
            (anomaly_dir / "cleaning_summary.json").write_text(
                json.dumps(
                    cleaning_summary,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return_code = merge_to_qlib.convert_to_bin(
                cleaned_dir,
                qlib_dir,
                overwrite=True,
                dropna_fields=[],
                calendar_path=benchmark_provider / "calendars" / "day.txt",
            )
            if return_code != 0:
                raise RobustPriceProviderError(
                    "qlib_conversion_failed", "cleaned Qlib conversion failed"
                )
            benchmark_result = add_index_benchmark.add_benchmark(
                qlib_dir,
                import_provider_dir=benchmark_provider,
                required_start_date=required_start_date,
            )
            benchmark_result["provider_dir"] = "qlib_provider"
            if _benchmark_fingerprint(
                benchmark_provider, required_start_date
            ) != benchmark_fingerprint:
                raise RobustPriceProviderError(
                    "benchmark_source_changed",
                    "benchmark provider changed during product build",
                )
            if _load_imported_bundle(price_bundle_dir)[1] != price_verified:
                raise RobustPriceProviderError(
                    "price_bundle_changed",
                    "imported price bundle changed during product build",
                )

            data_entries = _tree_entries(
                temporary, excluded_names={BUILD_FILENAME, CONTENT_FILENAME}
            )
            content = {
                "build_version": BUILD_VERSION,
                "product_key": key,
                "tree_sha256": _entries_sha256(data_entries),
                "files": data_entries,
            }
            (temporary / CONTENT_FILENAME).write_text(
                json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            build = {
                "build_version": BUILD_VERSION,
                "product_key": key,
                "price_bundle_key": import_metadata["bundle_key"],
                "price_manifest_sha256": price_verified["manifest_sha256"],
                "membership_snapshot_hash": import_metadata[
                    "membership_snapshot_hash"
                ],
                "benchmark_fingerprint": benchmark_fingerprint,
                "required_start_date": required_start_date,
                "source_symbols": cleaning_summary["source_symbols"],
                "cleaned_symbols": cleaning_summary["written_symbols"],
                "source_rows": cleaning_summary["source_rows"],
                "output_rows": cleaning_summary["output_rows"],
                "masked_volume_rows": cleaning_summary["masked_volume_rows"],
                "masked_event_rows": cleaning_summary["masked_event_rows"],
                "coverage": coverage,
                "benchmark_result": benchmark_result,
                "content_manifest_sha256": price_validator.file_sha256(
                    temporary / CONTENT_FILENAME
                ),
                "created_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
            }
            (temporary / BUILD_FILENAME).write_text(
                json.dumps(build, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            _make_tree_read_only(temporary)
            if _product_is_reusable(temporary, key) is None:
                raise RobustPriceProviderError(
                    "temporary_product_invalid",
                    "temporary robust price product failed publication checks",
                )
            os.replace(temporary, product)
        finally:
            if temporary.exists():
                _remove_tree(temporary)
        existing = _product_is_reusable(product, key)
        if existing is None:
            raise RobustPriceProviderError(
                "published_product_invalid",
                "published robust price product failed integrity validation",
            )

    assert existing is not None
    result = {
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "product_key": key,
        "product_dir": str(product),
        "cleaned_parquet": str(product / "cleaned_parquet"),
        "qlib_provider": str(product / "qlib_provider"),
        "membership_snapshot_hash": existing["membership_snapshot_hash"],
        "benchmark_fingerprint": existing["benchmark_fingerprint"],
        "cleaned_symbols": existing["cleaned_symbols"],
        "output_rows": existing["output_rows"],
        "reused": reused,
    }
    _atomic_write(
        output_root / "latest_product.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an immutable cleaned parquet and Qlib price product."
    )
    parser.add_argument("--price-bundle-dir", type=Path, required=True)
    parser.add_argument("--benchmark-provider", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--required-start-date", default="2014-01-02")
    parser.add_argument("--status-out-dir", type=Path, default=DEFAULT_STATUS_DIR)
    args = parser.parse_args(argv)
    try:
        status = build_robust_price_provider(
            price_bundle_dir=args.price_bundle_dir,
            benchmark_provider=args.benchmark_provider,
            output_root=args.output_root,
            required_start_date=args.required_start_date,
        )
        return_code = 0
    except (
        RobustPriceProviderError,
        price_importer.PriceBundleImportError,
        price_validator.PriceBundleError,
        OSError,
        ValueError,
    ) as exc:
        status = {
            "status": "failed",
            "reason_code": getattr(exc, "reason_code", "price_provider_build_failed"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        }
        return_code = 2
    _atomic_write(
        args.status_out_dir / "robust_price_provider_latest.json",
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
