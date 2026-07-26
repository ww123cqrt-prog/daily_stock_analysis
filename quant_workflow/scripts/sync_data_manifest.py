#!/usr/bin/env python3
"""Synchronize quant_data/DATA_MANIFEST.json with actual local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root, reports_root as default_reports_root  # noqa: E402
from _robust_membership_integrity import (  # noqa: E402
    MembershipIntegrityError,
    membership_read_lock,
)
from _robust_reference_integrity import (  # noqa: E402
    REFERENCE_SNAPSHOT_FILES,
    ReferenceIntegrityError,
    reference_read_lock,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    catalog_entry_from_snapshot,
)


REFERENCE_PARQUET_FILES = tuple(
    name for name in REFERENCE_SNAPSHOT_FILES if name.endswith(".parquet")
)
DATA_READ_ERRORS = (
    KeyError,
    TypeError,
    ValueError,
    OSError,
    UnicodeError,
    json.JSONDecodeError,
    pa.ArrowException,
)
README_BLOCK_START = "<!-- DATASET_CATALOG_V2_START -->"
README_BLOCK_END = "<!-- DATASET_CATALOG_V2_END -->"
README_TARGETS = {
    "README.md": ("quant_data 数据治理说明", ""),
    "parquet/ashare_1d/README.md": ("A 股日线 Parquet", "parquet/ashare_1d"),
    "parquet/etf_1d/README.md": ("ETF 日线数据", "parquet/etf_1d"),
    "qlib/README.md": ("Qlib Provider", "qlib"),
    "raw/README.md": ("供应商原始数据", "raw"),
    "reference/README.md": ("时点参考数据", "reference"),
}


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def read_first_last(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    first = ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if not first:
                first = value
            last = value
    return first, last


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def human_size(path: Path) -> str:
    size = directory_size_bytes(path)
    if not path.exists():
        return "missing"
    if size < 1024 * 1024:
        return f"{round(size / 1024, 1)}K"
    if size < 1024 * 1024 * 1024:
        return f"{round(size / (1024 * 1024))}M"
    return f"{round(size / (1024 * 1024 * 1024), 2)}G"


def parquet_entry(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "size": human_size(path),
        "file_count": len(list(path.glob("*.parquet"))) if path.exists() else 0,
    }


def qlib_entry(path: Path, role: str) -> dict[str, Any]:
    first, last = read_first_last(path / "calendars" / "day.txt")
    return {
        "path": str(path),
        "role": role,
        "size": human_size(path),
        "calendar_first": first,
        "calendar_last": last,
        "instruments_all": count_lines(path / "instruments" / "all.txt"),
    }


def _relative_path(path: Path, root: Path) -> str:
    return Path(path).expanduser().resolve().relative_to(
        Path(root).expanduser().resolve()
    ).as_posix()


def _relative_storage_entry(value: Any, quant_data: Path) -> Any:
    if isinstance(value, list):
        return [_relative_storage_entry(item, quant_data) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"path", "cleaned_parquet", "qlib_provider"} and isinstance(
            item, str
        ):
            candidate = Path(item).expanduser()
            if candidate.is_absolute():
                try:
                    result[key] = _relative_path(candidate, quant_data)
                except ValueError:
                    result[key] = item
            else:
                result[key] = item
        else:
            result[key] = _relative_storage_entry(item, quant_data)
    return result


def _source_manifest_details(path: Path) -> tuple[str, str]:
    manifest_path = path / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        return "unknown", ""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "unknown", ""
    generated_at = str(payload.get("generated_at") or "unknown")
    latest = ""
    symbols = payload.get("symbols")
    if isinstance(symbols, dict):
        latest = max(
            (
                str(item.get("actual_end") or "")
                for item in symbols.values()
                if isinstance(item, dict)
            ),
            default="",
        )
    return generated_at, latest


def _legacy_dataset_entry(
    *,
    quant_data: Path,
    dataset_id: str,
    path: Path,
    data_kind: str,
    source_provider: str,
    source_interface: str,
    price_basis: str,
    role: str = "legacy",
) -> dict[str, Any]:
    source_retrieved_at, latest = _source_manifest_details(path)
    exists = path.exists()
    entry: dict[str, Any] = {
        "dataset_id": dataset_id,
        "data_kind": data_kind,
        "role": role,
        "status": "legacy_unknown" if exists else "blocked",
        "source": {
            "provider": source_provider,
            "interface": source_interface,
        },
        "price_basis": price_basis,
        "path": _relative_path(path, quant_data),
        "source_retrieved_at": source_retrieved_at,
        "latest_trade_date": latest,
        "adjustment": {
            "method": "none" if price_basis in {"raw", "not_applicable"} else "unknown",
            "evidence_status": (
                "not_applicable"
                if price_basis in {"raw", "not_applicable"}
                else "missing_factor_or_paired_raw_snapshot"
            ),
        },
        "snapshot_available": False,
        "migration_note": (
            "Registered for audit. It is not resolvable through the v2 canonical data interface."
        ),
    }
    if path.is_dir():
        entry["file_count"] = len(list(path.glob("*.parquet")))
        entry["size_bytes"] = directory_size_bytes(path)
    elif path.is_file():
        entry["file_count"] = 1
        entry["size_bytes"] = path.stat().st_size
    else:
        entry["file_count"] = 0
        entry["size_bytes"] = 0
    return entry


def _v2_dataset_catalog(
    *,
    quant_data: Path,
    pit_entry: dict[str, Any] | None,
    reference_entry: dict[str, Any] | None,
    robust_product_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    parquet_root = quant_data / "parquet" / "ashare_1d"
    etf_root = quant_data / "parquet" / "etf_1d"
    qlib_root = quant_data / "qlib"
    datasets = {
        "ashare_daily_raw": {
            "dataset_id": "ashare_daily_raw",
            "data_kind": "daily_bars",
            "role": "canonical",
            "status": "blocked",
            "source": {
                "provider": "baostock_or_eastmoney",
                "interface": "raw_daily_history",
            },
            "price_basis": "raw",
            "path": "parquet/ashare_1d/ashare_daily_raw",
            "source_retrieved_at": "unknown",
            "adjustment": {"method": "none"},
            "snapshot_available": False,
            "blocker_codes": [
                "baostock_blacklisted_10001011",
                "eastmoney_raw_qfq_connection_closed",
                "initial_full_raw_snapshot_missing",
            ],
            "migration_note": (
                "No complete raw snapshot has been published. Legacy qfq files "
                "cannot be converted back into raw prices."
            ),
        },
        "ashare_daily_qfq": {
            "dataset_id": "ashare_daily_qfq",
            "data_kind": "daily_bars",
            "role": "canonical",
            "status": "blocked",
            "source": {
                "provider": "baostock_factor_or_eastmoney_paired",
                "interface": "local_factor_or_provider_native_pair",
            },
            "price_basis": "qfq",
            "path": "parquet/ashare_1d/ashare_daily_qfq",
            "source_retrieved_at": "unknown",
            "adjustment": {
                "method": "local_factor",
                "evidence_status": "raw_factor_snapshot_missing",
            },
            "snapshot_available": False,
            "blocker_codes": [
                "baostock_adjust_factor_unavailable",
                "eastmoney_paired_history_unavailable",
                "corporate_action_snapshot_missing",
            ],
            "migration_note": (
                "Canonical qfq publication remains blocked until raw prices, "
                "complete factors and corporate actions can be validated."
            ),
        },
        "qlib_ashare_v2": {
            "dataset_id": "qlib_ashare_v2",
            "data_kind": "qlib_provider",
            "role": "canonical",
            "status": "blocked",
            "source": {
                "provider": "quant_workflow",
                "interface": "v2_snapshot_to_qlib",
            },
            "price_basis": "qfq",
            "path": "qlib/cn_data_ashare_v2",
            "source_retrieved_at": "unknown",
            "adjustment": {
                "method": "local_factor",
                "evidence_status": "upstream_ashare_daily_qfq_blocked",
            },
            "snapshot_available": False,
            "blocker_codes": ["ashare_daily_qfq_not_ready"],
            "migration_note": (
                "Build only after the canonical A-share qfq snapshot and "
                "new-versus-legacy consistency comparison pass."
            ),
        },
        "ashare_baostock_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_baostock_qfq_legacy",
            path=parquet_root / "baostock_qfq",
            data_kind="daily_bars",
            source_provider="baostock",
            source_interface="query_history_k_data_plus_adjustflag_2",
            price_basis="qfq",
            role="audit",
        ),
        "ashare_jqdata_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_jqdata_qfq_legacy",
            path=parquet_root / "jqdata",
            data_kind="daily_bars",
            source_provider="jqdata",
            source_interface="get_price_fq_pre",
            price_basis="qfq",
            role="audit",
        ),
        "ashare_eastmoney_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_eastmoney_qfq_legacy",
            path=parquet_root / "eastmoney_qfq",
            data_kind="daily_bars",
            source_provider="eastmoney",
            source_interface="push2his_kline_fqt_1",
            price_basis="qfq",
            role="audit",
        ),
        "ashare_akshare_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_akshare_qfq_legacy",
            path=parquet_root / "akshare_qfq",
            data_kind="daily_bars",
            source_provider="akshare_eastmoney",
            source_interface="stock_zh_a_hist_adjust_qfq",
            price_basis="qfq",
            role="audit",
        ),
        "ashare_merged_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_merged_qfq_legacy",
            path=parquet_root / "merged_baostock_jqdata",
            data_kind="daily_bars",
            source_provider="multi_source",
            source_interface="legacy_provider_qfq_date_fill",
            price_basis="qfq",
        ),
        "ashare_cleaned_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="ashare_cleaned_qfq_legacy",
            path=parquet_root / "merged_baostock_jqdata_cleaned",
            data_kind="daily_bars",
            source_provider="multi_source",
            source_interface="legacy_cleaned_provider_qfq",
            price_basis="qfq",
        ),
        "qlib_ashare_cleaned_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="qlib_ashare_cleaned_legacy",
            path=qlib_root / "cn_data_baostock_jqdata_cleaned",
            data_kind="qlib_provider",
            source_provider="quant_workflow",
            source_interface="legacy_merge_to_qlib",
            price_basis="qfq",
        ),
        "qlib_ashare_merged_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="qlib_ashare_merged_legacy",
            path=qlib_root / "cn_data_baostock_jqdata",
            data_kind="qlib_provider",
            source_provider="quant_workflow",
            source_interface="legacy_merge_to_qlib",
            price_basis="qfq",
            role="audit",
        ),
        "etf_momentum_sina_raw_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="etf_momentum_sina_raw_legacy",
            path=etf_root / "etf_momentum_rotation" / "v1",
            data_kind="daily_bars",
            source_provider="sina",
            source_interface="akshare_fund_etf_hist_sina",
            price_basis="raw",
            role="audit",
        ),
        "etf_momentum_eastmoney_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="etf_momentum_eastmoney_qfq_legacy",
            path=etf_root / "etf_momentum_rotation" / "eastmoney_qfq",
            data_kind="daily_bars",
            source_provider="eastmoney",
            source_interface="fund_etf_hist_em_adjust_qfq",
            price_basis="qfq",
            role="audit",
        ),
        "etf_momentum_tencent_qfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="etf_momentum_tencent_qfq_legacy",
            path=etf_root / "etf_momentum_rotation" / "tencent_qfq",
            data_kind="daily_bars",
            source_provider="tencent",
            source_interface="ifzq_qfqday",
            price_basis="qfq",
            role="audit",
        ),
        "global_daily_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="global_daily_legacy",
            path=quant_data / "parquet" / "global_1d",
            data_kind="daily_bars",
            source_provider="yahoo",
            source_interface="chart_api_normalized",
            price_basis="derived",
            role="audit",
        ),
        "industry_theme_etf_returns_hfq_legacy": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="industry_theme_etf_returns_hfq_legacy",
            path=etf_root / "industry_theme_etf_daily_returns_hfq.parquet",
            data_kind="derived_daily_returns",
            source_provider="unknown",
            source_interface="untraced_legacy_file",
            price_basis="hfq",
            role="legacy",
        ),
        "raw_global_daily": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="raw_global_daily",
            path=quant_data / "raw" / "global_1d" / "yahoo",
            data_kind="raw_provider_responses",
            source_provider="yahoo",
            source_interface="chart_api",
            price_basis="not_applicable",
            role="audit",
        ),
        "raw_ashare_snapshots": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="raw_ashare_snapshots",
            path=quant_data / "raw" / "ashare_snapshots",
            data_kind="quote_snapshots",
            source_provider="sina",
            source_interface="hq_sina",
            price_basis="not_applicable",
            role="audit",
        ),
        "raw_ashare_news": _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="raw_ashare_news",
            path=quant_data / "raw" / "ashare_news",
            data_kind="news",
            source_provider="eastmoney",
            source_interface="bounded_news_search",
            price_basis="not_applicable",
            role="audit",
        ),
    }
    if pit_entry is not None:
        datasets["pit_csi800"] = {
            "dataset_id": "pit_csi800",
            "data_kind": "point_in_time_membership",
            "role": "canonical",
            "status": "ready",
            "source": {
                "provider": pit_entry["selected_source"],
                "interface": "pit_membership_builder",
            },
            "price_basis": "not_applicable",
            "path": _relative_path(Path(pit_entry["path"]), quant_data),
            "adjustment": {"method": "none"},
            "snapshot_hash": pit_entry["membership_snapshot_hash"],
            "date_range": pit_entry["date_range"],
            "snapshot_available": False,
        }
    if reference_entry is not None:
        datasets["robust_reference"] = {
            "dataset_id": "robust_reference",
            "data_kind": "point_in_time_reference",
            "role": "canonical",
            "status": "ready",
            "source": {
                "provider": reference_entry["selected_source"],
                "interface": "robust_reference_builder",
            },
            "price_basis": "not_applicable",
            "path": _relative_path(Path(reference_entry["path"]), quant_data),
            "adjustment": {"method": "none"},
            "snapshot_hash": reference_entry["reference_snapshot_hash"],
            "date_range": reference_entry["date_range"],
            "snapshot_available": False,
        }
    if robust_product_entry is not None:
        datasets["robust_price_product_legacy"] = _legacy_dataset_entry(
            quant_data=quant_data,
            dataset_id="robust_price_product_legacy",
            path=Path(robust_product_entry["path"]),
            data_kind="immutable_research_product",
            source_provider="quant_workflow",
            source_interface="build_robust_price_provider",
            price_basis="qfq",
            role="audit",
        )

    for search_root in (
        quant_data / "parquet",
        quant_data / "qlib",
        quant_data / "runtime",
    ):
        if not search_root.is_dir():
            continue
        for pointer_path in search_root.rglob("LATEST.json"):
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                snapshot = quant_data / str(pointer["snapshot_path"])
                entry = catalog_entry_from_snapshot(
                    snapshot,
                    quant_data_root=quant_data,
                    dataset_root=pointer_path.parent,
                )
            except (
                DataContractError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            datasets[str(entry["dataset_id"])] = entry
    return dict(sorted(datasets.items()))


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _pit_manifest_entry_locked(pit_dir: Path, snapshot) -> dict[str, Any]:
    snapshot.ensure_transaction_clean()
    raw_source = json.loads(
        (pit_dir / "MEMBERSHIP_SOURCE.json").read_text(encoding="utf-8")
    )
    date_range = raw_source["date_range"]
    if not isinstance(date_range, dict):
        raise TypeError("membership date_range must be a mapping")
    source = snapshot.validate_source(
        start_date=str(date_range["start_date"]),
        end_date=str(date_range["end_date"]),
    )
    daily_files = snapshot.daily_files()
    membership_hash = snapshot.fingerprint()
    member_day_count = sum(parquet_row_count(path) for path in daily_files)
    symbols: set[str] = set()
    for path in daily_files:
        symbols.update(
            str(value)
            for value in pq.read_table(path, columns=["symbol"])
            .column("symbol")
            .to_pylist()
        )
    entry = {
        "path": str(pit_dir),
        "role": "Official point-in-time CSI800 membership snapshot.",
        "status": "do_not_move",
        "date_range": source["date_range"],
        "requested_source": source["requested"],
        "selected_source": source["selected"],
        "daily_file_count": len(daily_files),
        "member_day_count": member_day_count,
        "symbol_count": len(symbols),
        "interval_row_count": parquet_row_count(
            pit_dir / "membership_intervals.parquet"
        ),
        "membership_snapshot_hash": membership_hash,
    }
    if "fallback_reason" in source:
        entry["fallback_reason"] = source["fallback_reason"]
    return entry


def pit_manifest_entry(
    pit_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not pit_dir.is_dir():
        return None, {
            "path": str(pit_dir),
            "status": "not_ready",
            "reason_code": "membership_snapshot_missing",
        }
    try:
        with membership_read_lock(pit_dir) as snapshot:
            return _pit_manifest_entry_locked(pit_dir, snapshot), None
    except (*DATA_READ_ERRORS, MembershipIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "membership_snapshot_invalid")
        return None, {
            "path": str(pit_dir),
            "status": "not_ready",
            "reason_code": reason_code,
        }


def reference_manifest_entry(
    reference_dir: Path,
    pit_entry: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    missing = [
        name for name in REFERENCE_SNAPSHOT_FILES if not (reference_dir / name).is_file()
    ]
    if missing:
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": "reference_snapshot_incomplete",
            "missing_files": missing,
        }
    if pit_entry is None:
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": "membership_snapshot_not_ready",
        }
    try:
        with reference_read_lock(reference_dir) as snapshot:
            snapshot.ensure_transaction_clean()
            date_range = pit_entry["date_range"]
            source = snapshot.validate_source(
                start_date=str(date_range["start_date"]),
                end_date=str(date_range["end_date"]),
                expected_membership_snapshot_hash=str(
                    pit_entry["membership_snapshot_hash"]
                ),
            )
            reference_hash = snapshot.fingerprint()
            entry = {
                "path": str(reference_dir),
                "role": "Official PIT security, industry, ST, and suspension metadata.",
                "status": "do_not_move",
                "date_range": source["date_range"],
                "requested_source": source["requested"],
                "selected_source": source["selected"],
                "industry_classification": source["industry_classification"],
                "membership_snapshot_hash": source["membership_snapshot_hash"],
                "reference_snapshot_hash": reference_hash,
                "row_counts": {
                    name: parquet_row_count(reference_dir / name)
                    for name in REFERENCE_PARQUET_FILES
                },
            }
            if "fallback_reason" in source:
                entry["fallback_reason"] = source["fallback_reason"]
            if "build_run_id" in source:
                entry["build_run_id"] = source["build_run_id"]
            return entry, None
    except (*DATA_READ_ERRORS, ReferenceIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "reference_snapshot_invalid")
        return None, {
            "path": str(reference_dir),
            "status": "not_ready",
            "reason_code": reason_code,
        }


def robust_price_product_manifest_entry(
    runtime_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Register the immutable robust price product selected by its pointer."""
    products_root = (Path(runtime_root) / "price_products").resolve()
    pointer_path = products_root / "latest_product.json"
    pending = {
        "path": str(products_root),
        "status": "not_ready",
        "reason_code": "robust_price_product_missing",
    }
    if not pointer_path.is_file() or pointer_path.is_symlink():
        return None, pending
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        product = Path(pointer["product_dir"]).expanduser()
        if product.is_symlink() or product.resolve().parent != products_root:
            raise ValueError("robust price product is outside its content-addressed root")
        product = product.resolve()
        if not product.is_dir():
            raise ValueError("robust price product directory is missing")
        build_path = product / "BUILD.json"
        content_path = product / "CONTENT.json"
        if build_path.is_symlink() or content_path.is_symlink():
            raise ValueError("robust price product metadata contains a symlink")
        build = json.loads(build_path.read_text(encoding="utf-8"))
        content = json.loads(content_path.read_text(encoding="utf-8"))
        product_key = str(pointer["product_key"])
        if (
            pointer.get("status") != "completed"
            or build.get("product_key") != product_key
            or content.get("product_key") != product_key
            or not isinstance(content.get("files"), dict)
            or not isinstance(content.get("tree_sha256"), str)
            or build.get("content_manifest_sha256") is None
        ):
            raise ValueError("robust price product metadata is incomplete")
        expected_paths = set(content["files"]) | {"BUILD.json", "CONTENT.json"}
        actual_paths = {
            path.relative_to(product).as_posix()
            for path in product.rglob("*")
            if path.is_file()
        }
        if actual_paths != expected_paths:
            raise ValueError("robust price product file set does not match its manifest")
        if any(path.is_symlink() for path in [product, *product.rglob("*")]):
            raise ValueError("robust price product contains a symlink")
        entry = {
            "path": str(product),
            "role": "Immutable cleaned OHLCV and Qlib provider for robust CSI800 research.",
            "status": "do_not_move",
            "product_key": product_key,
            "price_manifest_sha256": build.get("price_manifest_sha256"),
            "membership_snapshot_hash": build.get("membership_snapshot_hash"),
            "benchmark_fingerprint": build.get("benchmark_fingerprint"),
            "product_tree_sha256": content["tree_sha256"],
            "content_manifest_sha256": build["content_manifest_sha256"],
            "cleaned_parquet": str(product / "cleaned_parquet"),
            "qlib_provider": str(product / "qlib_provider"),
            "cleaned_symbols": pointer.get("cleaned_symbols"),
            "output_rows": pointer.get("output_rows"),
            "size": human_size(product),
        }
        return entry, None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        pending["reason_code"] = "robust_price_product_invalid"
        pending["message"] = str(exc)
        return None, pending


def robust_manifest_entries(reference_dir: Path) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    pit_dir = reference_dir / "pit_csi800"
    if not pit_dir.is_dir():
        pit_entry, pit_pending = pit_manifest_entry(pit_dir)
        _, reference_pending = reference_manifest_entry(reference_dir, None)
        return pit_entry, pit_pending, None, reference_pending
    try:
        with membership_read_lock(pit_dir) as snapshot:
            pit_entry = _pit_manifest_entry_locked(pit_dir, snapshot)
            reference_entry, reference_pending = reference_manifest_entry(
                reference_dir, pit_entry
            )
            return pit_entry, None, reference_entry, reference_pending
    except (*DATA_READ_ERRORS, MembershipIntegrityError) as exc:
        reason_code = getattr(exc, "reason_code", "membership_snapshot_invalid")
        return (
            None,
            {
                "path": str(pit_dir),
                "status": "not_ready",
                "reason_code": reason_code,
            },
            None,
            {
                "path": str(reference_dir),
                "status": "not_ready",
                "reason_code": "membership_snapshot_not_ready",
            },
        )


def build_manifest(*, quant_data_root: Path, reports_root: Path) -> dict[str, Any]:
    quant_data = Path(quant_data_root).expanduser()
    parquet_root = quant_data / "parquet" / "ashare_1d"
    etf_root = quant_data / "parquet" / "etf_1d"
    global_parquet_root = quant_data / "parquet" / "global_1d"
    raw_root = quant_data / "raw"
    qlib_root = quant_data / "qlib"

    cleaned_parquet = parquet_root / "merged_baostock_jqdata_cleaned"
    cleaned_provider = qlib_root / "cn_data_baostock_jqdata_cleaned"
    canonical_provider = qlib_root / "cn_data_baostock_jqdata"
    reference_root = quant_data / "reference"
    robust_runtime_root = quant_data / "runtime" / "robust_strategy"
    pit_entry, pit_pending, reference_entry, reference_pending = (
        robust_manifest_entries(reference_root)
    )
    robust_product_entry, robust_product_pending = (
        robust_price_product_manifest_entry(robust_runtime_root)
    )

    official_entries: dict[str, Any] = {
        "cleaned_parquet": {
            **parquet_entry(
                cleaned_parquet,
                "Official cleaned A-share daily parquet dataset for DSA and Qlib conversion.",
            ),
            "status": "do_not_move",
        },
        "cleaned_qlib_provider": {
            **qlib_entry(
                cleaned_provider,
                "Official Qlib provider for current training and backtesting.",
            ),
            "status": "do_not_move",
        },
    }
    if pit_entry is not None:
        official_entries["pit_csi800"] = pit_entry
    if reference_entry is not None:
        official_entries["robust_reference"] = reference_entry
    if robust_product_entry is not None:
        official_entries["robust_price_product"] = robust_product_entry
    pending_entries: dict[str, Any] = {}
    if pit_pending is not None:
        pending_entries["pit_csi800"] = pit_pending
    if reference_pending is not None:
        pending_entries["robust_reference"] = reference_pending
    if robust_product_pending is not None:
        pending_entries["robust_price_product"] = robust_product_pending
    official_entries = _relative_storage_entry(official_entries, quant_data)
    pending_entries = _relative_storage_entry(pending_entries, quant_data)

    datasets = _v2_dataset_catalog(
        quant_data=quant_data,
        pit_entry=pit_entry,
        reference_entry=reference_entry,
        robust_product_entry=robust_product_entry,
    )
    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "path_format": "relative_to_quant_data_root",
        "datasets": datasets,
        "source_report": "data_hygiene_inventory_latest.json",
        "cleanup_plan": "data_hygiene_cleanup_plan_latest.json",
        "external_report_owner": "quant_workflow/reports",
        "policy": {
            "owner": "local DSA quant workbench",
            "dry_run_first": True,
            "no_delete_without_explicit_user_confirmation": True,
            "no_move_without_explicit_user_confirmation": True,
            "do_not_store_secrets": True,
            "data_root": ".",
        },
        "official_entries": official_entries,
        "pending_entries": pending_entries,
        "documented_inputs": _relative_storage_entry({
            "baostock_qfq": parquet_entry(
                parquet_root / "baostock_qfq",
                "Baostock source parquet used to reproduce merged and cleaned data.",
            ),
            "jqdata": parquet_entry(
                parquet_root / "jqdata",
                "JQData source parquet used to supplement and cross-check A-share data.",
            ),
            "eastmoney_qfq": parquet_entry(
                parquet_root / "eastmoney_qfq",
                "EastMoney public-source parquet used as redundant daily data input.",
            ),
            "akshare_qfq": parquet_entry(
                parquet_root / "akshare_qfq",
                "AkShare public-source parquet used as optional redundant daily data input.",
            ),
            "merged_baostock_jqdata": parquet_entry(
                parquet_root / "merged_baostock_jqdata",
                "Merged pre-cleaning parquet dataset retained for comparison and reproducibility.",
            ),
            "canonical_qlib_provider": qlib_entry(
                canonical_provider,
                "Merged pre-cleaning Qlib provider retained as a comparison baseline.",
            ),
            "etf_momentum_rotation": parquet_entry(
                etf_root / "etf_momentum_rotation" / "v1",
                "Isolated two-ETF momentum-rotation candidate data; not part of the Qlib stock provider.",
            ),
            "global_daily": parquet_entry(
                global_parquet_root,
                "Canonical US, Japan, and Korea daily parquet used by CatDesk research.",
            ),
            "raw_global_daily": _legacy_dataset_entry(
                quant_data=quant_data,
                dataset_id="raw_global_daily_input",
                path=raw_root / "global_1d" / "yahoo",
                data_kind="raw_provider_responses",
                source_provider="yahoo",
                source_interface="chart_api",
                price_basis="not_applicable",
                role="audit",
            ),
            "raw_ashare_snapshots": _legacy_dataset_entry(
                quant_data=quant_data,
                dataset_id="raw_ashare_snapshots_input",
                path=raw_root / "ashare_snapshots",
                data_kind="quote_snapshots",
                source_provider="sina",
                source_interface="hq_sina",
                price_basis="not_applicable",
                role="audit",
            ),
            "raw_ashare_news": _legacy_dataset_entry(
                quant_data=quant_data,
                dataset_id="raw_ashare_news_input",
                path=raw_root / "ashare_news",
                data_kind="news",
                source_provider="eastmoney",
                source_interface="bounded_news_search",
                price_basis="not_applicable",
                role="audit",
            ),
        }, quant_data),
        "archive_candidates": {
            "require_confirmation": True,
            "note": "These are documented by the dry-run cleanup plan only. They have not been moved or deleted.",
            "examples": [
                _relative_path(parquet_root / "qfq", quant_data),
                _relative_path(parquet_root / "qfq_full", quant_data),
                _relative_path(qlib_root / "cn_data", quant_data),
                _relative_path(qlib_root / "cn_data_jqdata", quant_data),
                _relative_path(qlib_root / "cn_data_jqdata_full", quant_data),
            ],
        },
        "root_file_policy": {
            "current_state": "Root-level logs, pid files, runtime JSON files, and temporary scripts may still exist for compatibility.",
            "desired_state": "Runtime state should live under quant_workflow/reports/runtime_state, logs under quant_workflow/logs/quant_data, and one-off scripts under quant_workflow/scripts/archive after explicit confirmation.",
            "no_action_taken": True,
        },
        "related_reports": {
            "quality_report": "data_quality_report_latest.md",
            "cleaning_strategy": "data_cleaning_strategy_latest.md",
            "latest_training_result": "latest_training_result.json",
            "daily_update_summary": "daily_update_summary_latest.json",
            "research_status": "research_status_latest.json",
        },
    }
    return manifest


def _dataset_storage_path(entry: dict[str, Any]) -> str:
    value = entry.get("dataset_root") or entry.get("path") or ""
    return str(value).strip("/")


def _catalog_rows_for_prefix(
    datasets: dict[str, Any], prefix: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id in sorted(datasets):
        entry = datasets[dataset_id]
        if not isinstance(entry, dict):
            continue
        storage = _dataset_storage_path(entry)
        if prefix and storage != prefix and not storage.startswith(prefix + "/"):
            continue
        rows.append(entry)
    return rows


def _markdown_cell(value: object) -> str:
    return str(value if value not in (None, "") else "-").replace("|", "\\|")


def _render_catalog_block(manifest: dict[str, Any], prefix: str) -> str:
    rows = _catalog_rows_for_prefix(manifest.get("datasets", {}), prefix)
    lines = [
        README_BLOCK_START,
        "## DATA_MANIFEST v2 自动清单",
        "",
        "本表由 `sync_data_manifest.py` 根据实际清单生成。不要手工修改本区块。",
        "正式读取必须使用 `quant_data_cli.py resolve`；`legacy_unknown` 只能审计。",
        "",
        "| dataset_id | 角色 | 状态 | 价格口径 | 最新交易日 | 来源 | 相对路径 |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in rows:
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        values = (
            entry.get("dataset_id"),
            entry.get("role"),
            entry.get("status"),
            entry.get("price_basis"),
            entry.get("latest_trade_date"),
            source.get("provider"),
            _dataset_storage_path(entry),
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    if not rows:
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend(["", README_BLOCK_END])
    return "\n".join(lines)


def _replace_readme_block(existing: str, block: str) -> str:
    if README_BLOCK_START in existing and README_BLOCK_END in existing:
        before, remainder = existing.split(README_BLOCK_START, 1)
        _, after = remainder.split(README_BLOCK_END, 1)
        return before.rstrip() + "\n\n" + block + after.rstrip() + "\n"
    return existing.rstrip() + "\n\n" + block + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def sync_catalog_readmes(quant_data: Path, manifest: dict[str, Any]) -> None:
    quant_data = Path(quant_data)
    datasets = manifest.get("datasets", {})
    for relative, (title, prefix) in README_TARGETS.items():
        path = quant_data / relative
        rows = _catalog_rows_for_prefix(datasets, prefix)
        if relative != "README.md" and not path.parent.exists() and not rows:
            continue
        existing = (
            path.read_text(encoding="utf-8")
            if path.is_file()
            else f"# {title}\n\n该目录的数据口径以根 `DATA_MANIFEST.json` 为准。\n"
        )
        _atomic_write_text(
            path,
            _replace_readme_block(existing, _render_catalog_block(manifest, prefix)),
        )


def merge_existing_manifest(existing: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Keep explicitly documented entries while generated current-state data wins."""
    merged = dict(existing)
    merged.update(generated)
    existing_official = existing.get("official_entries")
    generated_official = generated.get("official_entries")
    official = dict(existing_official) if isinstance(existing_official, dict) else {}
    if isinstance(generated_official, dict):
        official.update(generated_official)
    merged["official_entries"] = official
    existing_pending = existing.get("pending_entries")
    generated_pending = generated.get("pending_entries")
    pending = dict(existing_pending) if isinstance(existing_pending, dict) else {}
    if isinstance(generated_pending, dict):
        pending.update(generated_pending)
    merged["pending_entries"] = pending
    return merged


def sync_data_manifest(*, quant_data_root: Path | None = None, reports_root: Path | None = None) -> Path:
    quant_data = Path(quant_data_root or default_quant_data_root()).expanduser()
    reports = Path(reports_root or default_reports_root()).expanduser()
    quant_data.mkdir(parents=True, exist_ok=True)
    path = quant_data / "DATA_MANIFEST.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            existing = payload
    generated = build_manifest(quant_data_root=quant_data, reports_root=reports)
    manifest = merge_existing_manifest(existing, generated)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    sync_catalog_readmes(quant_data, manifest)
    return path


def _is_covered(candidate: Path, roots: list[Path]) -> bool:
    resolved = candidate.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        return True
    return False


def _governed_data_paths(quant_data: Path) -> list[Path]:
    candidates: set[Path] = set()
    ashare_root = quant_data / "parquet" / "ashare_1d"
    if ashare_root.is_dir():
        candidates.update(path for path in ashare_root.iterdir() if path.is_dir())
    etf_root = quant_data / "parquet" / "etf_1d"
    if etf_root.is_dir():
        candidates.update(path for path in etf_root.glob("*.parquet") if path.is_file())
        candidates.update(
            path
            for path in etf_root.rglob("*")
            if path.is_dir() and any(path.glob("*.parquet"))
        )
    qlib_root = quant_data / "qlib"
    if qlib_root.is_dir():
        candidates.update(path for path in qlib_root.iterdir() if path.is_dir())
    raw_root = quant_data / "raw"
    if raw_root.is_dir():
        candidates.update(
            path
            for path in raw_root.rglob("*")
            if path.is_dir() and any(child.is_file() for child in path.iterdir())
        )
    reference_root = quant_data / "reference"
    if reference_root.is_dir():
        candidates.update(path for path in reference_root.iterdir() if path.is_dir())
        candidates.update(path for path in reference_root.glob("*.parquet"))
    return sorted(candidates)


def _absolute_path_issues(value: Any, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            issues.extend(_absolute_path_issues(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_absolute_path_issues(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and Path(value).is_absolute():
        issues.append(f"absolute path in manifest: {prefix}")
    return issues


def _readme_drift_issues(quant_data: Path, manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    datasets = manifest.get("datasets", {})
    for relative, (_, prefix) in README_TARGETS.items():
        path = quant_data / relative
        rows = _catalog_rows_for_prefix(datasets, prefix)
        if relative != "README.md" and not path.parent.exists() and not rows:
            continue
        if not path.is_file():
            issues.append(f"dataset README is missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        if _render_catalog_block(manifest, prefix) not in content:
            issues.append(f"dataset README catalog is stale: {relative}")
    return issues


def check_manifest(
    *,
    quant_data_root: Path | None = None,
    reports_root: Path | None = None,
) -> list[str]:
    quant_data = Path(quant_data_root or default_quant_data_root()).expanduser().resolve()
    reports = Path(reports_root or default_reports_root()).expanduser()
    path = quant_data / "DATA_MANIFEST.json"
    issues: list[str] = []
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ["DATA_MANIFEST.json is missing or invalid"]
    expected = build_manifest(quant_data_root=quant_data, reports_root=reports)
    if actual.get("schema_version") != 2:
        issues.append("DATA_MANIFEST.json schema_version is not 2")
    if actual.get("path_format") != "relative_to_quant_data_root":
        issues.append("DATA_MANIFEST.json path_format is not relative")
    comparable_actual = dict(actual)
    comparable_expected = dict(expected)
    comparable_actual.pop("generated_at", None)
    comparable_expected.pop("generated_at", None)
    if comparable_actual != comparable_expected:
        issues.append("DATA_MANIFEST.json content is stale")
    issues.extend(_absolute_path_issues(actual))
    issues.extend(_readme_drift_issues(quant_data, actual))

    registered: list[Path] = []
    for entry in expected.get("datasets", {}).values():
        if not isinstance(entry, dict):
            continue
        relative = entry.get("dataset_root") or entry.get("path")
        if isinstance(relative, str) and relative:
            registered.append((quant_data / relative).resolve())
    archive_paths = [
        (quant_data / value).resolve()
        for value in expected.get("archive_candidates", {}).get("examples", [])
        if isinstance(value, str)
    ]
    for candidate in _governed_data_paths(quant_data):
        if not _is_covered(candidate, registered + archive_paths):
            issues.append(
                "unregistered governed data path: "
                + candidate.relative_to(quant_data).as_posix()
            )
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize quant_data/DATA_MANIFEST.json with actual artifacts.")
    parser.add_argument("--quant-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Reports root. Defaults to quant_workflow/reports.")
    parser.add_argument("--check", action="store_true", help="Validate the existing v2 manifest without rewriting it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        issues = check_manifest(
            quant_data_root=args.quant_data_root,
            reports_root=args.out_dir,
        )
        if issues:
            for issue in issues:
                print(f"ERROR: {issue}", file=sys.stderr)
            return 1
        print("quant_data manifest check: ok")
        return 0
    path = sync_data_manifest(quant_data_root=args.quant_data_root, reports_root=args.out_dir)
    print(f"Synced data manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
