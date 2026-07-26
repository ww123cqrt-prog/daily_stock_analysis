#!/usr/bin/env python3
"""Inspect, resolve, validate, and update governed ``quant_data`` datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    describe_dataset,
    resolve_dataset,
    validate_dataset_snapshot,
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
        f"内容哈希: {payload.get('content_sha256', 'unknown')}",
        f"生成脚本: {producer.get('script', 'unknown')} / {producer.get('version', 'unknown')}",
        f"存储位置: {payload.get('snapshot_path') or payload.get('path', 'unknown')}",
    ]
    if not payload.get("snapshot_available", False):
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
    manifest = validate_dataset_snapshot(snapshot, verify_hashes=True)
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


def _update(
    root: Path,
    dataset_id: str,
    through: str | None,
    bundle_dir: Path | None,
) -> int:
    if dataset_id in {"ashare_daily_raw", "ashare_daily_qfq"} and bundle_dir:
        from download_baostock_v2 import build_snapshots_from_bundle

        raw, qfq = build_snapshots_from_bundle(
            bundle_dir=bundle_dir,
            quant_data_root=root,
            publish=True,
        )
        _json_dump({"raw": str(raw), "qfq": str(qfq)})
        return 0
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
            return _update(root, args.dataset_id, args.through, args.bundle_dir)
    except DataContractError as exc:
        print(f"不能使用 quant_data 数据：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
