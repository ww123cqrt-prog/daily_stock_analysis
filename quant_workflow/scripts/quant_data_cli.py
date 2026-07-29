#!/usr/bin/env python3
"""Inspect, resolve, validate, and update governed ``quant_data`` datasets."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from _file_lock import lock_file, unlock_file  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    dataset_publication_lock,
    describe_dataset,
    resolve_dataset,
    validate_dataset_snapshot,
)


ASHARE_UPDATE_POINTERS = (
    Path("DATA_MANIFEST.json"),
    Path("parquet/ashare_1d/ashare_daily_raw/LATEST.json"),
    Path("parquet/ashare_1d/ashare_daily_qfq/LATEST.json"),
    Path("qlib/cn_data_ashare_v2/LATEST.json"),
    Path("qlib/cn_data_ashare_cleaned_v2/LATEST.json"),
)
ASHARE_UPDATE_TRANSACTION_PATH = Path(
    "runtime/transactions/ashare_v2_update.json"
)
ASHARE_UPDATE_TRANSACTION_SCHEMA_VERSION = 1
ASHARE_UPDATE_TRANSACTION_MAX_BYTES = 16 * 1024 * 1024


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


def _resolve(root: Path, dataset_id: str, require_basis: str | None) -> int:
    resolved = resolve_dataset(
        dataset_id,
        quant_data_root=root,
        require_basis=require_basis,
    )
    print(resolved.snapshot_path)
    return 0


def _validate(root: Path, target: str) -> int:
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


def _update_ashare_from_bundle(
    root: Path,
    dataset_id: str,
    bundle_dir: Path,
    *,
    refresh_legacy_compatibility: bool = False,
) -> int:
    from build_cleaned_qlib_v2 import build_cleaned_qlib_v2
    from build_qlib_v2 import build_qlib_v2
    from download_baostock_v2 import build_snapshots_from_bundle

    previous_files: dict[Path, bytes | None] = {}
    validated_cache: dict[str, Any] = {}
    completed_steps: list[str] = []
    failed_step = "capture_state"
    started_at = _timestamp()
    transaction: dict[str, Any] | None = None
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
        raw, qfq = build_snapshots_from_bundle(
            bundle_dir=bundle_dir,
            quant_data_root=root,
            publish=True,
            publication_lock_held=True,
        )
        completed_steps.append("raw_qfq_v2")
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="qlib_v2",
            completed_steps=completed_steps,
        )
        failed_step = "qlib_v2"
        qlib_v2 = build_qlib_v2(
            quant_data_root=root,
            publish=True,
            publication_lock_held=True,
            validated_cache=validated_cache,
        )
        completed_steps.append("qlib_v2")
        _write_update_transaction_progress(
            root,
            transaction,
            active_step="qlib_cleaned_v2",
            completed_steps=completed_steps,
        )
        failed_step = "qlib_cleaned_v2"
        qlib_cleaned_v2 = build_cleaned_qlib_v2(
            quant_data_root=root,
            publish=True,
            publication_lock_held=True,
            validated_cache=validated_cache,
        )
        completed_steps.append("qlib_cleaned_v2")
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
            "legacy_compatibility": legacy_compatibility,
        },
    )
    output = {
        "raw": str(raw),
        "qfq": str(qfq),
        "qlib_v2": str(qlib_v2),
        "qlib_cleaned_v2": str(qlib_cleaned_v2),
        "legacy_compatibility": legacy_compatibility,
    }
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
) -> int:
    if dataset_id in {"ashare_daily_raw", "ashare_daily_qfq"} and bundle_dir:
        update_lock_path = root / "runtime" / "locks" / "ashare_v2_update.lock"
        update_lock_path.parent.mkdir(parents=True, exist_ok=True)
        update_lock_handle = update_lock_path.open("a+b")
        try:
            lock_file(update_lock_handle, exclusive=True, blocking=False)
        except BlockingIOError as exc:
            update_lock_handle.close()
            raise DataContractError(
                "A-share v2 update is already in progress"
            ) from exc
        except Exception:
            update_lock_handle.close()
            raise

        try:
            with dataset_publication_lock(root):
                _recover_interrupted_update(root)
                return _update_ashare_from_bundle(
                    root,
                    dataset_id,
                    bundle_dir,
                    refresh_legacy_compatibility=refresh_legacy_compatibility,
                )
        finally:
            unlock_file(update_lock_handle)
            update_lock_handle.close()
    if dataset_id in {"ashare_daily_raw", "ashare_daily_qfq"}:
        payload = describe_dataset(dataset_id, quant_data_root=root)
        blockers = payload.get("blocker_codes") or []
        raise DataContractError(
            "A-share v2 update is blocked: "
            + ", ".join(str(item) for item in blockers)
            + "; pass --bundle-dir after the remote raw/factor/qfq bundle is verified"
        )
    if dataset_id != "etf_momentum_daily_qfq":
        raise DataContractError(f"dataset has no registered updater: {dataset_id}")
    from download_etf_momentum_data import publish_canonical_snapshot

    path = publish_canonical_snapshot(
        quant_data_dir=root,
        end_date=through,
    )
    print(path)
    return 0


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
    resolve_parser.add_argument("--require-basis")

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("target")

    update_parser = commands.add_parser("update")
    update_parser.add_argument("dataset_id")
    update_parser.add_argument("--through")
    update_parser.add_argument("--bundle-dir", type=Path)
    update_parser.add_argument(
        "--refresh-legacy-compatibility",
        action="store_true",
        help="Explicitly rebuild retained mutable V1 compatibility outputs",
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
            return _resolve(root, args.dataset_id, args.require_basis)
        if args.command == "validate":
            return _validate(root, args.target)
        if args.command == "update":
            return _update(
                root,
                args.dataset_id,
                args.through,
                args.bundle_dir,
                refresh_legacy_compatibility=args.refresh_legacy_compatibility,
            )
    except DataContractError as exc:
        print(f"不能使用 quant_data 数据：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
