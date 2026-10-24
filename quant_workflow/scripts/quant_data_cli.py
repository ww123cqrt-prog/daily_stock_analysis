#!/usr/bin/env python3
"""Inspect, resolve, validate, and update governed ``quant_data`` datasets."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from quant_data_contract import DataContractError as LegacyDataContractError  # noqa: E402
from quant_qmt_data.quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    describe_dataset,
    finalize_artifact_snapshot,
    publish_snapshot_group,
    resolve_dataset,
    resolve_dataset_artifact,
    resolve_dataset_component,
    validate_dataset_snapshot,
)
from quant_data_reader import build_update_universe, read_bars  # noqa: E402
from quant_data_increment_validation import (  # noqa: E402
    compare_snapshot_increment,
    sample_increment_against_bundle,
)
from quant_workflow.quant_data_qlib_provider import (  # noqa: E402
    build_direct_provider_descriptor,
)


ASHARE_UPDATE_POINTERS = (
    Path("DATA_MANIFEST.json"),
    Path("parquet/ashare_1d/ashare_daily_raw/LATEST.json"),
    Path("parquet/ashare_1d/ashare_daily_qfq/LATEST.json"),
    Path("reference/qlib_cleaning_plan/LATEST.json"),
)
ASHARE_UPDATE_TRANSACTION_PATH = Path(
    "runtime/transactions/ashare_v2_update.json"
)
ASHARE_UPDATE_TRANSACTION_SCHEMA_VERSION = 1
ASHARE_UPDATE_TRANSACTION_MAX_BYTES = 16 * 1024 * 1024
ASHARE_BUNDLE_MANIFEST_MAX_BYTES = 1024 * 1024
INCREMENT_COMPARISON_REPORT_MAX_BYTES = 16 * 1024 * 1024

ETF_RELEASE_DATASET_ID = "cn_etf_daily"
ETF_RELEASE_DATASET_ROOT = Path("parquet/etf_1d/cn_etf_daily")
# The formal ETF universe must cover every strategy input that the research
# workflow can consume.  Keeping this list here prevents a release from
# passing on the legacy two-ETF minimum while the three-ETF strategy is
# missing its defensive holding or the frozen required pool is incomplete.
ETF_RELEASE_REQUIRED_SYMBOLS = (
    "SH512890",
    "SH513010",
    "SH510720",
    "SZ159915",
    "SH513100",
)
ETF_RELEASE_LEGACY_DATASET_IDS = frozenset(
    {
        "cn_etf_daily_raw",
        "cn_etf_daily_qfq",
        "etf_momentum_daily_raw",
        "etf_momentum_daily_qfq",
        "etf_momentum_rotation",
    }
)
ETF_RELEASE_LEGACY_PATHS = (
    Path("parquet/etf_1d/cn_etf_daily_raw"),
    Path("parquet/etf_1d/cn_etf_daily_qfq"),
    Path("parquet/etf_1d/etf_momentum_daily_raw"),
    Path("parquet/etf_1d/etf_momentum_daily_qfq"),
    Path("parquet/etf_1d/etf_momentum_rotation"),
    Path("cache/etf_momentum_daily_raw"),
    Path("cache/etf_momentum_daily_qfq"),
)
ETF_RELEASE_CATALOG_FIELDS = (
    "dataset_id",
    "data_kind",
    "role",
    "status",
    "source",
    "price_basis",
    "available_price_bases",
    "source_retrieved_at",
    "adjustment",
    "producer",
    "date_range",
    "latest_trade_date",
    "row_count",
    "symbol_count",
    "components",
    "file_index_path",
    "raw_file_index_path",
    "factor_index_path",
    "corporate_actions_path",
    "corporate_actions_sha256",
    "content_sha256",
)


def _read_catalog(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads((root / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError("DATA_MANIFEST.json is unavailable") from exc
    if payload.get("schema_version") != 2 or not isinstance(
        payload.get("datasets"), dict
    ):
        raise DataContractError("DATA_MANIFEST.json schema version 2 is required")
    return payload


def _json_dump(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _read_json_object(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise DataContractError(f"{label} is unavailable")
    try:
        if resolved.stat().st_size > max_bytes:
            raise DataContractError(f"{label} is too large")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except DataContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{label} must be a JSON object")
    return payload


def _emit_json_report(payload: Mapping[str, Any], output: Path | None) -> None:
    if output is not None:
        atomic_write_json(output.expanduser().resolve(), dict(payload))
    _json_dump(payload)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _capture_file_state(root: Path) -> dict[Path, bytes | None]:
    return {
        root / relative: (root / relative).read_bytes()
        if (root / relative).is_file()
        else None
        for relative in ASHARE_UPDATE_POINTERS
    }


def _atomic_restore_bytes(path: Path, previous: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if previous is None:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.restore")
    try:
        with temporary.open("wb") as handle:
            handle.write(previous)
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


def _restore_file_state(state: Mapping[Path, bytes | None]) -> None:
    failures: list[str] = []
    for path, previous in state.items():
        try:
            _atomic_restore_bytes(path, previous)
        except OSError:
            failures.append(path.name)
    if failures:
        raise DataContractError(
            "failed to restore formal update pointers: " + ", ".join(failures)
        )


def _transaction_path(root: Path) -> Path:
    return Path(root) / ASHARE_UPDATE_TRANSACTION_PATH


def _transaction_file_records(
    root: Path,
    state: Mapping[Path, bytes | None],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in ASHARE_UPDATE_POINTERS:
        path = Path(root) / relative
        if path not in state:
            raise DataContractError(
                f"update transaction state is missing {relative.as_posix()}"
            )
        content = state[path]
        records.append(
            {
                "path": relative.as_posix(),
                "content_base64": (
                    base64.b64encode(content).decode("ascii")
                    if content is not None
                    else None
                ),
                "sha256": (
                    hashlib.sha256(content).hexdigest()
                    if content is not None
                    else None
                ),
            }
        )
    return records


def _write_update_transaction(
    root: Path,
    state: Mapping[Path, bytes | None],
    *,
    dataset_id: str,
    started_at: str,
) -> dict[str, Any]:
    path = _transaction_path(root)
    if path.exists() or path.is_symlink():
        raise DataContractError(
            "an interrupted A-share update transaction journal must be recovered first"
        )
    payload: dict[str, Any] = {
        "schema_version": ASHARE_UPDATE_TRANSACTION_SCHEMA_VERSION,
        "transaction_id": uuid.uuid4().hex,
        "dataset_id": dataset_id,
        "started_at": started_at,
        "status": "prepared",
        "active_step": "raw_qfq_v2",
        "completed_steps": [],
        "previous_files": _transaction_file_records(root, state),
    }
    atomic_write_json(path, payload)
    return payload


def _write_update_transaction_progress(
    root: Path,
    transaction: dict[str, Any],
    *,
    active_step: str,
    completed_steps: Sequence[str],
) -> None:
    transaction["status"] = "publishing"
    transaction["active_step"] = active_step
    transaction["completed_steps"] = list(completed_steps)
    atomic_write_json(_transaction_path(root), transaction)


def _remove_update_transaction(root: Path) -> None:
    path = _transaction_path(root)
    if path.is_symlink():
        raise DataContractError("A-share update transaction journal path is unsafe")
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _read_update_transaction(
    root: Path,
) -> tuple[dict[str, Any], dict[Path, bytes | None]]:
    path = _transaction_path(root)
    if path.is_symlink() or not path.is_file():
        raise DataContractError("A-share update transaction journal is unsafe")
    try:
        if path.stat().st_size > ASHARE_UPDATE_TRANSACTION_MAX_BYTES:
            raise DataContractError("A-share update transaction journal is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except DataContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(
            "A-share update transaction journal is invalid"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != ASHARE_UPDATE_TRANSACTION_SCHEMA_VERSION
        or not isinstance(payload.get("transaction_id"), str)
        or not payload.get("transaction_id")
        or payload.get("status") not in {"prepared", "publishing"}
        or not isinstance(payload.get("previous_files"), list)
    ):
        raise DataContractError("A-share update transaction journal is invalid")

    expected = {relative.as_posix() for relative in ASHARE_UPDATE_POINTERS}
    restored: dict[Path, bytes | None] = {}
    seen: set[str] = set()
    for record in payload["previous_files"]:
        if not isinstance(record, dict):
            raise DataContractError("A-share update transaction journal is invalid")
        relative_text = record.get("path")
        if relative_text not in expected or relative_text in seen:
            raise DataContractError("A-share update transaction journal is invalid")
        seen.add(relative_text)
        encoded = record.get("content_base64")
        recorded_sha256 = record.get("sha256")
        if encoded is None:
            if recorded_sha256 is not None:
                raise DataContractError(
                    "A-share update transaction journal is invalid"
                )
            content = None
        elif isinstance(encoded, str) and isinstance(recorded_sha256, str):
            try:
                content = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise DataContractError(
                    "A-share update transaction journal is invalid"
                ) from exc
            if hashlib.sha256(content).hexdigest() != recorded_sha256:
                raise DataContractError(
                    "A-share update transaction journal hash mismatch"
                )
        else:
            raise DataContractError("A-share update transaction journal is invalid")
        restored[Path(root) / Path(relative_text)] = content
    if seen != expected:
        raise DataContractError("A-share update transaction journal is incomplete")
    return payload, restored


def _recover_interrupted_update(root: Path) -> str | None:
    path = _transaction_path(root)
    if not path.exists() and not path.is_symlink():
        return None
    transaction, previous_files = _read_update_transaction(root)
    _restore_file_state(previous_files)
    _remove_update_transaction(root)
    _try_write_update_status(
        root,
        {
            "status": "recovered_interrupted_update",
            "dataset_id": transaction.get("dataset_id"),
            "started_at": transaction.get("started_at"),
            "transaction_id": transaction.get("transaction_id"),
            "interrupted_step": transaction.get("active_step"),
            "completed_steps": transaction.get("completed_steps", []),
        },
    )
    return str(transaction["transaction_id"])


def _write_update_status(root: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(
        root / "runtime" / "status" / "quant_data_update_latest.json",
        {"updated_at": _timestamp(), **dict(payload)},
    )


def _try_write_update_status(root: Path, payload: Mapping[str, Any]) -> str | None:
    try:
        _write_update_status(root, payload)
    except Exception as exc:
        error_type = type(exc).__name__
        print(
            f"WARNING: quant_data update status could not be written ({error_type})",
            file=sys.stderr,
        )
        return error_type
    return None


def _human_description(payload: dict[str, Any]) -> str:
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    adjustment = (
        payload.get("adjustment")
        if isinstance(payload.get("adjustment"), dict)
        else {}
    )
    producer = (
        payload.get("producer")
        if isinstance(payload.get("producer"), dict)
        else {}
    )
    upstream_snapshots = payload.get("upstream_snapshots")
    upstream = (
        upstream_snapshots[0]
        if isinstance(upstream_snapshots, list)
        and upstream_snapshots
        and isinstance(upstream_snapshots[0], dict)
        else {}
    )
    upstream_dataset_id = (
        upstream.get("dataset_id")
        or payload.get("upstream_dataset_id")
        or "not_applicable"
    )
    upstream_snapshot_id = (
        upstream.get("snapshot_id")
        or payload.get("upstream_snapshot_id")
        or "not_applicable"
    )
    upstream_content_sha256 = (
        upstream.get("content_sha256")
        or payload.get("upstream_content_sha256")
        or "not_applicable"
    )
    benchmarks = (
        payload.get("benchmarks")
        if isinstance(payload.get("benchmarks"), dict)
        else {}
    )
    benchmark_source = (
        benchmarks.get("source")
        if isinstance(benchmarks.get("source"), dict)
        else {}
    )
    benchmark_symbols = (
        benchmarks.get("symbols")
        if isinstance(benchmarks.get("symbols"), dict)
        else {}
    )
    raw_codes = source.get("raw_security_codes")
    raw_code_text = (
        ", ".join(f"{key}={value}" for key, value in sorted(raw_codes.items()))
        if isinstance(raw_codes, dict)
        else "unknown"
    )
    lines = [
        f"数据集: {payload.get('dataset_id', '')}",
        f"用途角色: {payload.get('role', '')}",
        f"可用状态: {payload.get('status', '')}",
        f"数据来源: {source.get('provider', 'unknown')} / {source.get('interface', 'unknown')}",
        f"价格口径: {payload.get('price_basis', 'not_applicable')}",
        "可读取价格口径: "
        + ", ".join(
            str(item)
            for item in payload.get(
                "available_price_bases",
                [payload.get("price_basis", "not_applicable")],
            )
        ),
        f"复权方法: {adjustment.get('method', 'unknown')}",
        f"原始数据取得时间: {payload.get('source_retrieved_at', 'unknown')}",
        f"复权执行时间: {adjustment.get('adjusted_at', 'not_applicable')}",
        f"复权覆盖日期: {adjustment.get('adjustment_as_of', 'not_applicable')}",
        f"复权字段: {', '.join(adjustment.get('adjusted_fields', [])) or 'not_applicable'}",
        f"复权归一规则: {adjustment.get('normalization_rule', 'not_applicable')}",
        f"复权算法版本: {adjustment.get('algorithm_version', 'not_applicable')}",
        f"供应商证券代码: {raw_code_text}",
        f"最新交易日: {payload.get('latest_trade_date', 'unknown')}",
        f"因子目录: {adjustment.get('factor_path', 'not_applicable')}",
        f"因子索引: {payload.get('factor_index_path', 'not_applicable')}",
        f"公司行为文件: {payload.get('corporate_actions_path', 'not_applicable')}",
        f"内容哈希: {payload.get('content_sha256') or payload.get('upstream_content_sha256', 'unknown')}",
        f"上游数据集: {upstream_dataset_id}",
        f"上游快照: {upstream_snapshot_id}",
        f"上游内容哈希: {upstream_content_sha256}",
        f"生成脚本: {producer.get('script', 'unknown')} / {producer.get('version', 'unknown')}",
        f"存储位置: {payload.get('snapshot_path') or payload.get('path', 'unknown')}",
    ]
    if benchmarks or "benchmark_complete" in payload:
        lines.extend(
            [
                f"指数完整: {'是' if payload.get('benchmark_complete') is True else '否'}",
                "指数来源: "
                f"{benchmark_source.get('provider', 'unknown')} / "
                f"{benchmark_source.get('interface', 'unknown')}",
                f"交易日历哈希: {benchmarks.get('calendar_sha256', 'unknown')}",
            ]
        )
        for symbol, details in sorted(benchmark_symbols.items()):
            coverage = (
                details.get("coverage")
                if isinstance(details, dict)
                and isinstance(details.get("coverage"), dict)
                else {}
            )
            lines.append(
                f"指数 {symbol}: {coverage.get('start', 'unknown')} 至 "
                f"{coverage.get('end', 'unknown')}"
            )
    if payload.get("lifecycle") == "compatibility_output":
        lines.append(
            "正式兼容输出: 是；保持旧程序的固定路径和列结构，真实复权因子请查看上游 V2 快照。"
        )
    elif not payload.get("snapshot_available", False):
        lines.append("正式快照: 无；该数据只能审计，不能通过统一解析接口用于训练或回测。")
    return "\n".join(lines)


def _list(root: Path, *, as_json: bool) -> int:
    catalog = _read_catalog(root)
    rows = [catalog["datasets"][key] for key in sorted(catalog["datasets"])]
    if as_json:
        _json_dump(rows)
        return 0
    print("dataset_id\trole\tstatus\tprice_basis\tlatest_trade_date")
    for row in rows:
        print(
            "\t".join(
                str(row.get(field, ""))
                for field in (
                    "dataset_id",
                    "role",
                    "status",
                    "price_basis",
                    "latest_trade_date",
                )
            )
        )
    return 0


def _describe(root: Path, dataset_id: str, *, as_json: bool) -> int:
    payload = describe_dataset(dataset_id, quant_data_root=root)
    if as_json:
        _json_dump(payload)
    else:
        print(_human_description(payload))
    return 0


def _resolve(
    root: Path,
    dataset_id: str,
    require_basis: str | None,
    component_name: str | None = None,
) -> int:
    if component_name is not None:
        component = resolve_dataset_artifact(
            dataset_id,
            component_name,
            quant_data_root=root,
        )
        print(component.artifact_path)
        return 0
    if require_basis is not None:
        component = resolve_dataset_component(
            dataset_id,
            quant_data_root=root,
            price_basis=require_basis,
        )
        print(component.component_path)
        return 0
    resolved = resolve_dataset(
        dataset_id,
        quant_data_root=root,
    )
    print(resolved.snapshot_path)
    return 0


def _bars(
    root: Path,
    dataset_id: str,
    symbol: str,
    *,
    basis: str | None,
    fields: list[str] | None,
    start: str | None,
    end: str | None,
) -> int:
    frame = read_bars(
        dataset_id,
        symbol,
        quant_data_root=root,
        price_basis=basis,
        fields=fields,
        start=start,
        end=end,
    )
    output = frame.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    _json_dump(json.loads(output.to_json(orient="records")))
    return 0


def _export_universe(
    root: Path,
    dataset_id: str,
    *,
    basis: str | None,
    output: Path,
) -> int:
    payload = build_update_universe(
        dataset_id,
        quant_data_root=root,
        price_basis=basis,
    )
    output = Path(output).expanduser().resolve()
    atomic_write_json(output, payload)
    _json_dump(
        {
            "dataset_id": payload["dataset_id"],
            "snapshot_id": payload["snapshot_id"],
            "symbol_count": payload["symbol_count"],
            "output": str(output),
        }
    )
    return 0


def _release_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataContractError(f"ETF release {label} is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataContractError(f"ETF release {label} is unsafe")
    root = Path(root).resolve()
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise DataContractError(f"ETF release {label} is unsafe")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DataContractError(f"ETF release {label} escapes its root") from exc
    return candidate


def _release_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"ETF release {label} is invalid")
    return value


def _release_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_release_index(
    snapshot: Path,
    manifest: Mapping[str, Any],
    field: str,
    label: str,
) -> pd.DataFrame:
    path = _release_path(snapshot, manifest.get(field), label)
    if path.is_symlink() or not path.is_file():
        raise DataContractError(f"ETF release {label} is unavailable")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DataContractError(f"ETF release {label} is unreadable") from exc
    required = {"symbol", "relative_path", "row_count", "first_date", "last_date"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataContractError(f"ETF release {label} is invalid")
    symbols = frame["symbol"].astype(str)
    if symbols.duplicated().any():
        raise DataContractError(f"ETF release {label} contains duplicate symbols")
    return frame


def _validate_release_universe(
    snapshot: Path,
    manifest: Mapping[str, Any],
    qfq_index: pd.DataFrame,
) -> dict[str, Any]:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise DataContractError("ETF release source metadata is invalid")
    universe_path = _release_path(
        snapshot,
        source.get("universe_path"),
        "universe path",
    )
    universe_manifest_path = _release_path(
        snapshot,
        source.get("universe_manifest_path"),
        "universe manifest path",
    )
    if universe_path.is_symlink() or not universe_path.is_file():
        raise DataContractError("ETF release universe file is unavailable")
    if universe_manifest_path.is_symlink() or not universe_manifest_path.is_file():
        raise DataContractError("ETF release universe manifest is unavailable")
    try:
        universe = pd.read_parquet(universe_path)
        universe_manifest = json.loads(
            universe_manifest_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise DataContractError("ETF release universe evidence is unreadable") from exc
    required = {"symbol", "code", "market", "name"}
    if universe.empty or not required.issubset(universe.columns):
        raise DataContractError("ETF release universe is invalid")
    universe_symbols = universe["symbol"].astype(str)
    if universe_symbols.duplicated().any():
        raise DataContractError("ETF release universe contains duplicate symbols")
    indexed_symbols = set(qfq_index["symbol"].astype(str))
    if set(universe_symbols) != indexed_symbols:
        raise DataContractError("ETF release universe and qfq index disagree")
    if not isinstance(universe_manifest, dict):
        raise DataContractError("ETF release universe manifest is invalid")
    recorded_file_hash = universe_manifest.get("universe_file_sha256")
    if recorded_file_hash != _release_sha256(universe_path):
        raise DataContractError("ETF release universe file hash mismatch")
    source_universe_hash = source.get("universe_sha256")
    if (
        source_universe_hash
        and universe_manifest.get("universe_sha256") != source_universe_hash
    ):
        raise DataContractError("ETF release universe lineage hash mismatch")
    recorded_count = universe_manifest.get("selected_count")
    if recorded_count is None:
        recorded_count = universe_manifest.get("symbol_count")
    if _release_int(recorded_count, "universe symbol count") != len(universe):
        raise DataContractError("ETF release universe symbol count is stale")
    return {
        "symbol_count": int(len(universe)),
        "file_sha256": str(recorded_file_hash),
    }


def _validate_release_snapshot(
    root: Path,
    snapshot: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    required_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise DataContractError("ETF release snapshot directory is unavailable")
    validated = (
        dict(manifest)
        if manifest is not None
        else validate_dataset_snapshot(
            snapshot,
            verify_hashes=True,
            evidence_root=root,
        )
    )
    if (
        validated.get("dataset_id") != ETF_RELEASE_DATASET_ID
        or validated.get("snapshot_id") != snapshot.name
        or validated.get("data_kind") != "paired_daily_bars"
        or validated.get("role") != "canonical"
        or validated.get("status") != "ready"
        or validated.get("price_basis") != "qfq"
    ):
        raise DataContractError(
            f"ETF release snapshot contract is invalid: {snapshot.name}"
        )

    qfq_index = _read_release_index(
        snapshot,
        validated,
        "file_index_path",
        "qfq file index",
    )
    raw_index = _read_release_index(
        snapshot,
        validated,
        "raw_file_index_path",
        "raw file index",
    )
    factor_index = _read_release_index(
        snapshot,
        validated,
        "factor_index_path",
        "factor file index",
    )
    qfq_symbols = set(qfq_index["symbol"].astype(str))
    if (
        set(raw_index["symbol"].astype(str)) != qfq_symbols
        or set(factor_index["symbol"].astype(str)) != qfq_symbols
    ):
        raise DataContractError("ETF release raw/qfq/factor symbols disagree")
    component_rows = {
        "raw": int(raw_index["row_count"].sum()),
        "qfq": int(qfq_index["row_count"].sum()),
        "factors": int(factor_index["row_count"].sum()),
    }
    if len(set(component_rows.values())) != 1:
        raise DataContractError("ETF release raw/qfq/factor row counts disagree")
    symbol_count = _release_int(
        validated.get("symbol_count"), "snapshot symbol count"
    )
    row_count = _release_int(validated.get("row_count"), "snapshot row count")
    if symbol_count != len(qfq_index):
        raise DataContractError("ETF release snapshot symbol count is stale")
    if row_count != component_rows["qfq"]:
        raise DataContractError("ETF release snapshot row count is stale")

    universe = _validate_release_universe(snapshot, validated, qfq_index)
    missing = sorted(set(required_symbols) - qfq_symbols)
    if missing:
        raise DataContractError(
            "ETF release strategy symbols are missing from the same qfq component: "
            + ", ".join(missing)
        )
    if required_symbols:
        by_symbol = {
            str(row.symbol): str(row.relative_path)
            for row in qfq_index.itertuples(index=False)
        }
        qfq_component = _release_path(
            snapshot,
            validated["components"]["qfq"]["path"],
            "qfq component path",
        )
        for symbol in required_symbols:
            path = _release_path(snapshot, by_symbol[symbol], f"{symbol} qfq bar")
            try:
                path.relative_to(qfq_component)
            except ValueError as exc:
                raise DataContractError(
                    f"ETF release strategy symbol is outside qfq component: {symbol}"
                ) from exc
            try:
                sample = pd.read_parquet(path, columns=["date", "close"])
            except Exception as exc:
                raise DataContractError(
                    f"ETF release strategy symbol is unreadable: {symbol}"
                ) from exc
            if sample.empty:
                raise DataContractError(
                    f"ETF release strategy symbol has no rows: {symbol}"
                )

    return {
        "snapshot_id": str(validated["snapshot_id"]),
        "latest_trade_date": str(validated["latest_trade_date"]),
        "content_sha256": str(validated["content_sha256"]),
        "symbol_count": symbol_count,
        "row_count": row_count,
        "component_rows": component_rows,
        "indexed_files_checked": int(
            len(raw_index) + len(qfq_index) + len(factor_index) + 1
        ),
        "universe": universe,
    }


def _snapshot_retention_contract(
    manifest: Mapping[str, Any], snapshot_count: int
) -> dict[str, Any]:
    source = manifest.get("source")
    retention = source.get("snapshot_retention") if isinstance(source, Mapping) else None
    if retention is None:
        return {
            "policy": "legacy_inferred",
            "max_complete_snapshots": snapshot_count,
            "metadata_status": "inferred_from_retained_snapshots",
        }
    if not isinstance(retention, Mapping):
        raise DataContractError("ETF release snapshot retention contract is invalid")
    policy = str(retention.get("policy") or "")
    if policy not in {"immutable_history", "current_plus_one_previous"}:
        raise DataContractError("ETF release snapshot retention policy is invalid")
    maximum = _release_int(
        retention.get("max_complete_snapshots"),
        "snapshot retention count",
    )
    if maximum < 1:
        raise DataContractError("ETF release snapshot retention count is invalid")
    result = {
        "policy": policy,
        "max_complete_snapshots": maximum,
        "metadata_status": "declared",
    }
    if snapshot_count > maximum:
        result["metadata_status"] = "declared_with_retention_overage"
    return result


def _validate_etf_release(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    catalog = _read_catalog(root)
    datasets = catalog["datasets"]
    entry = datasets.get(ETF_RELEASE_DATASET_ID)
    if not isinstance(entry, dict):
        raise DataContractError("ETF release dataset is not registered")
    if set(datasets).intersection(ETF_RELEASE_LEGACY_DATASET_IDS):
        found = sorted(set(datasets).intersection(ETF_RELEASE_LEGACY_DATASET_IDS))
        raise DataContractError(
            "ETF release legacy dataset identities remain registered: "
            + ", ".join(found)
        )
    canonical_etf_ids = sorted(
        str(dataset_id)
        for dataset_id, candidate in datasets.items()
        if isinstance(candidate, Mapping)
        and candidate.get("role") == "canonical"
        and (
            candidate.get("data_kind") == "paired_daily_bars"
            or str(candidate.get("dataset_root") or "").startswith(
                "parquet/etf_1d/"
            )
        )
    )
    if canonical_etf_ids != [ETF_RELEASE_DATASET_ID]:
        raise DataContractError(
            "ETF release must have exactly one canonical ETF dataset: cn_etf_daily"
        )
    if entry.get("dataset_root") != ETF_RELEASE_DATASET_ROOT.as_posix():
        raise DataContractError("ETF release dataset root is not canonical")
    expected_pointer = (ETF_RELEASE_DATASET_ROOT / "LATEST.json").as_posix()
    if entry.get("latest_pointer") != expected_pointer:
        raise DataContractError("ETF release latest pointer is not canonical")

    for relative in ETF_RELEASE_LEGACY_PATHS:
        path = root / relative
        if path.exists() or path.is_symlink():
            raise DataContractError(
                f"ETF release legacy path remains: {relative.as_posix()}"
            )
    etf_parent = root / "parquet" / "etf_1d"
    if not etf_parent.is_dir() or etf_parent.is_symlink():
        raise DataContractError("ETF release parent directory is unavailable")
    unexpected_directories = sorted(
        child.name
        for child in etf_parent.iterdir()
        if (child.is_dir() or child.is_symlink())
        and child.name != ETF_RELEASE_DATASET_ROOT.name
    )
    if unexpected_directories:
        raise DataContractError(
            "ETF release legacy or unregistered directories remain: "
            + ", ".join(unexpected_directories)
        )

    resolved = resolve_dataset(
        ETF_RELEASE_DATASET_ID,
        quant_data_root=root,
        require_basis="qfq",
    )
    current_manifest = resolved.manifest
    for field in ETF_RELEASE_CATALOG_FIELDS:
        if entry.get(field) != current_manifest.get(field):
            raise DataContractError(
                f"ETF release root manifest disagrees with current snapshot: {field}"
            )

    dataset_root = root / ETF_RELEASE_DATASET_ROOT
    snapshots_root = dataset_root / "snapshots"
    if snapshots_root.is_symlink() or not snapshots_root.is_dir():
        raise DataContractError("ETF release snapshots directory is unavailable")
    snapshots = sorted(snapshots_root.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_dir() for path in snapshots):
        raise DataContractError("ETF release snapshots directory contains unsafe entries")
    if not snapshots:
        raise DataContractError("ETF release requires at least one complete snapshot")
    current = resolved.snapshot_path.resolve()
    if current not in snapshots:
        raise DataContractError("ETF release current snapshot is not retained")

    snapshot_summaries: list[dict[str, Any]] = []
    for snapshot in snapshots:
        snapshot_summaries.append(
            _validate_release_snapshot(
                root,
                snapshot,
                manifest=current_manifest if snapshot == current else None,
                required_symbols=(
                    ETF_RELEASE_REQUIRED_SYMBOLS if snapshot == current else ()
                ),
            )
        )
    current_summary = next(
        summary for summary in snapshot_summaries if summary["snapshot_id"] == current.name
    )
    backup = next((path for path in reversed(snapshots) if path != current), None)
    backup_summary = (
        next(
            summary for summary in snapshot_summaries if summary["snapshot_id"] == backup.name
        )
        if backup is not None
        else None
    )
    retention = _snapshot_retention_contract(current_manifest, len(snapshots))
    return {
        "status": "full_validation_passed",
        "dataset_id": ETF_RELEASE_DATASET_ID,
        "network_requests": 0,
        "scope": "ETF only; no unrelated A-share dataset was scanned",
        "root_manifest": {
            "status": "passed",
            "canonical_etf_datasets": canonical_etf_ids,
        },
        "snapshot_retention": {
            "count": len(snapshots),
            "policy": retention["policy"],
            "max_complete_snapshots": retention["max_complete_snapshots"],
        },
        "retention_metadata_status": retention["metadata_status"],
        "validated_snapshot_ids": [
            summary["snapshot_id"] for summary in snapshot_summaries
        ],
        "snapshots": snapshot_summaries,
        "current": current_summary,
        "backup": backup_summary,
        "strategy_symbols": {
            "status": "passed",
            "component": "qfq",
            "symbols": list(ETF_RELEASE_REQUIRED_SYMBOLS),
        },
        "legacy_entries": {
            "status": "passed",
            "registered_dataset_ids": [],
            "existing_paths": [],
        },
    }


def _validate_release(root: Path, target: str) -> int:
    started = time.monotonic()
    try:
        if target != ETF_RELEASE_DATASET_ID:
            raise DataContractError(
                "--release currently supports only cn_etf_daily"
            )
        payload = _validate_etf_release(root)
    except Exception as exc:
        message = str(exc).replace(str(Path(root).resolve()), "<quant_data_root>")
        _json_dump(
            {
                "status": "full_validation_failed",
                "dataset_id": target,
                "network_requests": 0,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "error": {
                    "type": type(exc).__name__,
                    "message": message,
                },
            }
        )
        return 2
    payload["elapsed_seconds"] = round(time.monotonic() - started, 3)
    _json_dump(payload)
    return 0


def _validate(root: Path, target: str, *, release: bool = False) -> int:
    if release:
        return _validate_release(root, target)
    candidate = Path(target).expanduser()
    if candidate.exists():
        snapshot = candidate.resolve()
    else:
        snapshot = resolve_dataset(target, quant_data_root=root).snapshot_path
    manifest = validate_dataset_snapshot(
        snapshot,
        verify_hashes=True,
        evidence_root=root,
    )
    _json_dump(
        {
            "status": "ready",
            "dataset_id": manifest["dataset_id"],
            "snapshot_id": manifest["snapshot_id"],
            "latest_trade_date": manifest["latest_trade_date"],
            "content_sha256": manifest["content_sha256"],
        }
    )
    return 0


def _ashare_bundle_mode(bundle_dir: Path) -> str:
    manifest_path = Path(bundle_dir).expanduser().resolve() / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        # The selected builder will return the authoritative missing-manifest error.
        return "full"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DataContractError("A-share bundle manifest path is unsafe")
    try:
        if manifest_path.stat().st_size > ASHARE_BUNDLE_MANIFEST_MAX_BYTES:
            raise DataContractError("A-share bundle manifest is too large")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except DataContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError("A-share bundle manifest is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise DataContractError("A-share bundle manifest must be a JSON object")
    mode = str(payload.get("bundle_mode") or "full")
    if mode not in {"full", "incremental"}:
        raise DataContractError("A-share bundle mode is invalid")
    return mode


def _build_ashare_price_snapshots(
    *,
    root: Path,
    bundle_dir: Path,
) -> tuple[Path, Path, str, dict[str, Any] | None]:
    mode = _ashare_bundle_mode(bundle_dir)
    if mode == "incremental":
        from incremental_update_ashare_v2 import build_incremental_price_snapshots

        result = build_incremental_price_snapshots(
            bundle_dir=bundle_dir,
            quant_data_root=root,
        )
        raw_root = root / "parquet" / "ashare_1d" / "ashare_daily_raw"
        qfq_root = root / "parquet" / "ashare_1d" / "ashare_daily_qfq"
        raw, qfq = publish_snapshot_group(
            [(result.raw_stage, raw_root), (result.qfq_stage, qfq_root)],
            quant_data_root=root,
            update_root_catalog=True,
            publication_lock_held=True,
        )
        metrics = {
            "base_latest_trade_date": result.base_latest_trade_date,
            "latest_trade_date": result.latest_trade_date,
            "downloaded_rows": int(result.downloaded_rows),
            "reused_symbols": len(result.reused_symbols),
            "factor_changed_symbols": len(result.factor_changed_symbols),
            "qfq_recalculated_symbols": len(result.qfq_recalculated_symbols),
        }
        return raw, qfq, mode, metrics

    from download_baostock_v2 import build_snapshots_from_bundle

    raw, qfq = build_snapshots_from_bundle(
        bundle_dir=bundle_dir,
        quant_data_root=root,
        publish=True,
        publication_lock_held=True,
    )
    return raw, qfq, mode, None


def _refresh_qlib_cleaning_plan(root: Path) -> dict[str, Any]:
    catalog_path = root / "DATA_MANIFEST.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        datasets = catalog["datasets"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataContractError("quant_data catalog is unavailable") from exc
    if not isinstance(datasets, dict) or not isinstance(
        datasets.get("qlib_cleaning_plan"), dict
    ):
        return {"status": "not_registered"}

    qfq_entry = datasets.get("ashare_daily_qfq")
    if not isinstance(qfq_entry, dict):
        raise DataContractError("formal A-share qfq dataset is unavailable")
    upstream = {
        field: str(qfq_entry.get(field) or "")
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    if (
        upstream["dataset_id"] != "ashare_daily_qfq"
        or upstream["price_basis"] != "qfq"
        or not upstream["snapshot_id"]
        or len(upstream["content_sha256"]) != 64
    ):
        raise DataContractError("formal A-share qfq snapshot identity is invalid")

    module_path = root / "src" / "quant_data_service" / "cleaning_plan.py"
    if module_path.is_symlink() or not module_path.is_file():
        raise DataContractError("quant_data cleaning-plan preparer is unavailable")
    module_name = f"_quant_data_cleaning_plan_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise DataContractError("quant_data cleaning-plan preparer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise DataContractError("quant_data cleaning-plan preparer cannot be loaded") from exc
    finally:
        sys.modules.pop(module_name, None)
    prepare = getattr(module, "prepare_cleaning_plan_staging", None)
    if not callable(prepare):
        raise DataContractError("quant_data cleaning-plan preparer is invalid")

    source = root / "reference" / "qlib_cleaning" / "CLEANING_PLAN.json"
    try:
        source_payload = json.loads(source.read_text(encoding="utf-8"))
        prepared = prepare(root, source, upstream)
    except Exception as exc:
        raise DataContractError("Qlib cleaning plan cannot be staged") from exc
    try:
        source_retrieved_at = str(
            source_payload.get("created_at")
            or qfq_entry.get("source_retrieved_at")
            or ""
        )
        finalize_artifact_snapshot(
            prepared.staging_path,
            dataset_id="qlib_cleaning_plan",
            snapshot_id=prepared.snapshot_id,
            data_kind="qlib_cleaning_plan",
            role="canonical",
            status="ready",
            source={
                "provider": "quant_workflow",
                "interface": "build_direct_cleaning_plan",
            },
            price_basis="not_applicable",
            producer={"script": "quant_data_cli.py", "version": "1"},
            source_retrieved_at=source_retrieved_at,
            date_range=qfq_entry["date_range"],
            components={
                "plan": {"path": "CLEANING_PLAN.json", "kind": "json"},
                "provenance": {"path": "PLAN_PROVENANCE.json", "kind": "json"},
            },
            upstream_snapshots=[upstream],
            metadata={
                "source_plan_path": prepared.source_plan_path,
                "source_plan_sha256": prepared.source_plan_sha256,
                "source_plan_schema_version": source_payload.get("schema_version"),
            },
        )
        published = publish_snapshot_group(
            [
                (
                    prepared.staging_path,
                    root / "reference" / "qlib_cleaning_plan",
                )
            ],
            quant_data_root=root,
            update_root_catalog=True,
            publication_lock_held=True,
        )[0]
        manifest = json.loads(
            (published / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
        )
    except BaseException:
        if prepared.staging_path.exists():
            shutil.rmtree(prepared.staging_path)
        raise
    return {
        "status": "published",
        "dataset_id": str(manifest["dataset_id"]),
        "snapshot_id": str(manifest["snapshot_id"]),
        "content_sha256": str(manifest["content_sha256"]),
    }


def _update_ashare_from_bundle(
    root: Path,
    dataset_id: str,
    bundle_dir: Path,
    *,
    refresh_legacy_compatibility: bool = False,
) -> int:
    previous_files: dict[Path, bytes | None] = {}
    completed_steps: list[str] = []
    failed_step = "capture_state"
    started_at = _timestamp()
    transaction: dict[str, Any] | None = None
    update_mode = "unknown"
    incremental_metrics: dict[str, Any] | None = None
    try:
        previous_files = _capture_file_state(root)
        failed_step = "prepare_transaction"
        transaction = _write_update_transaction(
            root,
            previous_files,
            dataset_id=dataset_id,
            started_at=started_at,
        )
        failed_step = "raw_qfq_v2"
        raw, qfq, update_mode, incremental_metrics = _build_ashare_price_snapshots(
            root=root,
            bundle_dir=bundle_dir,
        )
        completed_steps.append("raw_qfq_v2")
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="qlib_cleaning_plan",
            completed_steps=completed_steps,
        )
        failed_step = "qlib_cleaning_plan"
        _refresh_qlib_cleaning_plan(root)
        completed_steps.append("qlib_cleaning_plan")
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="direct_provider_canonical",
            completed_steps=completed_steps,
        )
        failed_step = "direct_provider_canonical"
        qlib_provider = build_direct_provider_descriptor(
            quant_data_root=root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
        )
        completed_steps.append("direct_provider_canonical")
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="direct_provider_cleaned",
            completed_steps=completed_steps,
        )
        failed_step = "direct_provider_cleaned"
        qlib_cleaned_provider = build_direct_provider_descriptor(
            quant_data_root=root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="cleaned",
        )
        completed_steps.append("direct_provider_cleaned")
        failed_step = "commit_transaction"
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="commit_transaction",
            completed_steps=completed_steps,
        )
        _remove_update_transaction(root)
        transaction = None
    except Exception as exc:
        try:
            _restore_file_state(previous_files)
            if transaction is not None:
                _remove_update_transaction(root)
        except Exception as rollback_exc:
            status_error = _try_write_update_status(
                root,
                {
                    "status": "rollback_failed",
                    "dataset_id": dataset_id,
                    "started_at": started_at,
                    "failed_step": failed_step,
                    "completed_steps": completed_steps,
                    "error_type": type(exc).__name__,
                    "rollback_error_type": type(rollback_exc).__name__,
                },
            )
            message = (
                f"A-share update failed during {failed_step} "
                f"({type(exc).__name__}); rollback also failed "
                f"({type(rollback_exc).__name__})"
            )
            if status_error:
                message += f"; status record failed ({status_error})"
            raise DataContractError(message) from rollback_exc
        status_error = _try_write_update_status(
            root,
            {
                "status": "rolled_back",
                "dataset_id": dataset_id,
                "started_at": started_at,
                "failed_step": failed_step,
                "completed_steps": completed_steps,
                "error_type": type(exc).__name__,
            },
        )
        message = f"A-share update failed during {failed_step} and was rolled back"
        if status_error:
            message += f"; status record failed ({status_error})"
        raise DataContractError(message) from exc

    legacy_compatibility: dict[str, Any] = {"status": "not_requested"}
    update_status = "completed"
    if refresh_legacy_compatibility:
        try:
            from publish_v2_compatibility import rebuild_and_publish_v1_compatibility

            legacy_compatibility = rebuild_and_publish_v1_compatibility(
                quant_data_root=root,
                publication_lock_held=True,
            )
        except Exception as exc:
            update_status = "completed_with_legacy_warning"
            legacy_compatibility = {
                "status": "legacy_refresh_failed",
                "error_type": type(exc).__name__,
            }

    status_error = _try_write_update_status(
        root,
        {
            "status": update_status,
            "dataset_id": dataset_id,
            "started_at": started_at,
            "completed_steps": completed_steps,
            "update_mode": update_mode,
            "incremental": incremental_metrics,
            "legacy_compatibility": legacy_compatibility,
        },
    )
    output = {
        "raw": str(raw),
        "qfq": str(qfq),
        "qlib_provider": str(Path(qlib_provider).parent),
        "qlib_cleaned_provider": str(Path(qlib_cleaned_provider).parent),
        "legacy_compatibility": legacy_compatibility,
    }
    if update_mode == "incremental":
        output["update_mode"] = update_mode
        output["incremental"] = incremental_metrics
    if status_error:
        output["status_record"] = {
            "status": "write_failed",
            "error_type": status_error,
        }
    _json_dump(output)
    return 0


def _update(
    root: Path,
    dataset_id: str,
    through: str | None,
    bundle_dir: Path | None,
    *,
    refresh_legacy_compatibility: bool = False,
    use_system_proxy: bool = False,
    etf_provider: str | None = None,
) -> int:
    if etf_provider is not None and dataset_id != "cn_etf_daily":
        raise DataContractError("--provider is supported only by cn_etf_daily")
    if dataset_id in {"ashare_daily_raw", "ashare_daily_qfq", "cn_etf_daily"}:
        import quant_qmt_data as provider
        from quant_qmt_data.download_baostock_v2 import BaostockV2Error
        from quant_qmt_data.download_cn_etf_data import CnEtfDownloadError

        if refresh_legacy_compatibility:
            raise DataContractError("the lightweight updater does not rebuild legacy compatibility datasets")

        try:
            if not bundle_dir and (dataset_id != "cn_etf_daily" or (etf_provider or "miniqmt") == "miniqmt"):
                result = provider.maintain_market("etf" if dataset_id == "cn_etf_daily" else "stock",
                                                  quant_data_dir=root, through=through)
                _json_dump({**result, "backend": "quant_qmt_data"})
                return 2 if result.get("status") in {"blocked", "failed", "partial"} else 0
            if dataset_id == "cn_etf_daily":
                result = {}
                path = provider.update_cn_etf_snapshot(
                    quant_data_dir=root, end_date=through,
                    provider=etf_provider or "miniqmt", _result=result)
                _json_dump({**result, "snapshot_path": str(path), "backend": "quant_qmt_data"})
                return 2 if result.get("status") in {"blocked", "failed", "partial"} else 0
            if bundle_dir:
                paths = provider.publish_ashare_bundle(quant_data_dir=root, bundle_dir=bundle_dir)
            else:
                paths = provider.update_ashare_snapshot(quant_data_dir=root, through=through)
            _json_dump({"backend": "quant_qmt_data", "snapshots": [str(path) for path in paths]})
            return 0
        except (provider.DataContractError, provider.MiniQMTError, BaostockV2Error, CnEtfDownloadError) as exc:
            raise DataContractError(str(exc)) from exc
    if dataset_id == "cn_index_daily":
        from update_cn_index_dataset import update_cn_index_snapshot

        kwargs: dict[str, Any] = {
            "quant_data_root": root,
            "end_date": through,
        }
        if use_system_proxy:
            kwargs["use_system_proxy"] = True
        result = update_cn_index_snapshot(**kwargs)
        _json_dump(result)
        return 0
    else:
        raise DataContractError(f"dataset has no registered updater: {dataset_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_quant_data_root())
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    describe_parser = commands.add_parser("describe")
    describe_parser.add_argument("dataset_id")
    describe_parser.add_argument("--json", action="store_true")

    resolve_parser = commands.add_parser("resolve")
    resolve_parser.add_argument("dataset_id")
    resolve_mode = resolve_parser.add_mutually_exclusive_group()
    resolve_mode.add_argument("--require-basis")
    resolve_mode.add_argument("--component")

    bars_parser = commands.add_parser("bars")
    bars_parser.add_argument("dataset_id")
    bars_parser.add_argument("symbol")
    bars_parser.add_argument("--basis")
    bars_parser.add_argument("--fields", nargs="+")
    bars_parser.add_argument("--start")
    bars_parser.add_argument("--end")

    universe_parser = commands.add_parser("export-universe")
    universe_parser.add_argument("dataset_id")
    universe_parser.add_argument("--basis")
    universe_parser.add_argument("--output", type=Path, required=True)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("target")
    validate_parser.add_argument(
        "--release",
        action="store_true",
        help="Run the read-only ETF release acceptance check without network access",
    )

    check_etf_parser = commands.add_parser(
        "check-etf", help="Read-only ETF data diagnostics; never a release/read gate"
    )
    check_etf_parser.add_argument("--output-dir", type=Path)
    check_etf_parser.add_argument("--calendar", type=Path)
    check_etf_parser.add_argument("--split-evidence", type=Path)

    strategy_check_parser = commands.add_parser(
        "check-strategy", help="Seeded paired-snapshot strategy diagnostics; never a read gate"
    )
    from check_strategy_data import add_arguments

    add_arguments(strategy_check_parser)

    compare_parser = commands.add_parser("compare-snapshots")
    compare_parser.add_argument("dataset_id")
    compare_parser.add_argument("--base-snapshot", required=True)
    compare_parser.add_argument("--candidate-snapshot")
    compare_parser.add_argument("--output", type=Path)

    sample_parser = commands.add_parser("sample-increment")
    sample_parser.add_argument("dataset_id")
    sample_parser.add_argument("--base-snapshot", required=True)
    sample_parser.add_argument("--candidate-snapshot")
    sample_parser.add_argument("--bundle-dir", type=Path, required=True)
    sample_parser.add_argument("--sample-size", type=int, default=100)
    sample_parser.add_argument("--seed", default="increment-check")
    sample_parser.add_argument("--comparison-report", type=Path, required=True)
    sample_parser.add_argument("--output", type=Path)

    expand_etf_parser = commands.add_parser("expand-cn-etf-universe")
    expand_etf_parser.add_argument(
        "--target-size",
        type=int,
        choices=(200,),
        default=200,
        help="Registered representative ETF universe size (currently 200 only)",
    )
    expand_etf_parser.add_argument("--through", required=True)

    update_parser = commands.add_parser("update")
    update_parser.add_argument("dataset_id")
    update_parser.add_argument("--through")
    update_parser.add_argument("--bundle-dir", type=Path)
    update_parser.add_argument("--provider", choices=("miniqmt", "tencent", "eastmoney", "baostock"),
                               help="Explicit ETF source override; the default is MiniQMT")
    update_parser.add_argument(
        "--refresh-legacy-compatibility",
        action="store_true",
        help="Explicitly rebuild retained mutable V1 compatibility outputs",
    )
    update_parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Use the configured system proxy for bounded public index requests",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "list":
            return _list(root, as_json=args.json)
        if args.command == "describe":
            return _describe(root, args.dataset_id, as_json=args.json)
        if args.command == "resolve":
            return _resolve(
                root,
                args.dataset_id,
                args.require_basis,
                args.component,
            )
        if args.command == "bars":
            return _bars(
                root,
                args.dataset_id,
                args.symbol,
                basis=args.basis,
                fields=args.fields,
                start=args.start,
                end=args.end,
            )
        if args.command == "export-universe":
            return _export_universe(
                root,
                args.dataset_id,
                basis=args.basis,
                output=args.output,
            )
        if args.command == "validate":
            return _validate(root, args.target, release=args.release)
        if args.command == "check-etf":
            from check_etf_data import main as check_etf

            options = ["--root", str(root)]
            if args.output_dir is not None:
                options.extend(["--output-dir", str(args.output_dir)])
            if args.calendar is not None:
                options.extend(["--calendar", str(args.calendar)])
            if args.split_evidence is not None:
                options.extend(["--split-evidence", str(args.split_evidence)])
            return check_etf(options)
        if args.command == "check-strategy":
            from check_strategy_data import main as check_strategy

            options = ["--root", str(root), "--reference-root", str(args.reference_root),
                       "--market", args.market, "--batches", str(args.batches),
                       "--sample-size", str(args.sample_size), "--sessions", str(args.sessions),
                       "--seed", str(args.seed)]
            for option, value in (("--start", args.start), ("--end", args.end),
                                  ("--plan", args.plan), ("--evidence", args.evidence),
                                  ("--calendar", args.calendar),
                                  ("--output-dir", args.output_dir)):
                if value is not None:
                    options.extend((option, str(value)))
            if args.codes:
                options.extend(("--codes", *args.codes))
            return check_strategy(options)
        if args.command == "compare-snapshots":
            payload = compare_snapshot_increment(
                args.dataset_id,
                quant_data_root=root,
                base_snapshot_id=args.base_snapshot,
                candidate_snapshot_id=args.candidate_snapshot,
            )
            _emit_json_report(payload, args.output)
            return 0
        if args.command == "sample-increment":
            comparison = (
                _read_json_object(
                    args.comparison_report,
                    label="increment comparison report",
                    max_bytes=INCREMENT_COMPARISON_REPORT_MAX_BYTES,
                )
                if args.comparison_report is not None
                else None
            )
            payload = sample_increment_against_bundle(
                args.dataset_id,
                quant_data_root=root,
                base_snapshot_id=args.base_snapshot,
                candidate_snapshot_id=args.candidate_snapshot,
                bundle_dir=args.bundle_dir.expanduser().resolve(),
                sample_size=args.sample_size,
                seed=args.seed,
                comparison_result=comparison,
            )
            _emit_json_report(payload, args.output)
            return 0
        if args.command == "expand-cn-etf-universe":
            from download_cn_etf_data import (
                CnEtfDownloadError,
                expand_cn_etf_universe,
            )

            try:
                published = expand_cn_etf_universe(
                    quant_data_dir=root,
                    target_size=args.target_size,
                    end_date=args.through,
                )
            except CnEtfDownloadError as exc:
                raise DataContractError(str(exc)) from exc
            print(published)
            return 0
        if args.command == "update":
            return _update(
                root,
                args.dataset_id,
                args.through,
                args.bundle_dir,
                refresh_legacy_compatibility=args.refresh_legacy_compatibility,
                use_system_proxy=args.use_system_proxy,
                etf_provider=args.provider,
            )
    except (DataContractError, LegacyDataContractError, ValueError, TypeError) as exc:
        print(f"不能使用 quant_data 数据：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
