#!/usr/bin/env python3
"""Prepare and attest the immutable CSI800 provider used by signal research."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from prepare_robust_provider import file_sha256, prepare_runtime_provider  # noqa: E402
from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402
from robust_signal_data_contract import (  # noqa: E402
    CONTRACT_VERSION,
    RobustSignalDataBlocked,
    load_verified_cleaned_quality_evidence,
)


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/[A-Za-z0-9_.~@%+,:=\\/-]+|[A-Za-z]:[\\/][^\s'\"<>]+)"
)


def _sanitized_error(exc: Exception) -> str:
    return _ABSOLUTE_PATH_RE.sub("[path]", str(exc))[:500]


DEFAULT_STATUS_PATH = (
    reports_root() / "robust_signal" / "robust_signal_data_readiness_latest.json"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _blocked_payload(reason_code: str) -> dict[str, Any]:
    messages = {
        "cleaned_provider_unavailable": (
            "The current cleaned Qlib V2 snapshot is unavailable.",
            "当前 cleaned Qlib V2 快照不可用，请先完成原生 V2 构建和数据质量验收。",
        ),
        "membership_intervals_missing": (
            "Historical CSI800 membership intervals are unavailable.",
            "历史时点 CSI800 成员区间不存在，无法准备鲁棒研究数据。",
        ),
        "data_manifest_missing": (
            "DATA_MANIFEST.json is unavailable.",
            "当前 DATA_MANIFEST.json 不存在，无法准备鲁棒研究数据。",
        ),
        "provider_build_failed": (
            "The robust CSI800 provider could not be prepared.",
            "鲁棒 CSI800 provider 准备失败，请查看本次任务日志。",
        ),
        "cleaned_provider_quality_mismatch": (
            "The cleaned provider does not match the approved quality evidence.",
            "当前 cleaned provider 与完整数据质量证据不一致，请重新验收数据。",
        ),
    }
    summary, summary_zh = messages[reason_code]
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "blocked_data",
        "trainable": False,
        "reason_code": reason_code,
        "summary": summary,
        "summary_zh": summary_zh,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def prepare_robust_signal_data(
    *,
    quant_data_dir: Path,
    status_path: Path,
    quality_pointer_path: Path | None = None,
) -> dict[str, Any]:
    quant_data_dir = Path(quant_data_dir).expanduser().resolve()
    status_path = Path(status_path).expanduser().resolve()
    manifest_path = quant_data_dir / "DATA_MANIFEST.json"
    intervals_path = (
        quant_data_dir / "reference" / "pit_csi800" / "membership_intervals.parquet"
    )
    runtime_root = quant_data_dir / "runtime" / "robust_strategy"
    quality_pointer_path = Path(
        quality_pointer_path
        or status_path.parent.parent / "data_quality_report_latest.json"
    ).expanduser()

    if not manifest_path.is_file():
        payload = _blocked_payload("data_manifest_missing")
        _write_json(status_path, payload)
        return payload
    if not intervals_path.is_file():
        payload = _blocked_payload("membership_intervals_missing")
        _write_json(status_path, payload)
        return payload

    try:
        quality = load_verified_cleaned_quality_evidence(quality_pointer_path)
    except RobustSignalDataBlocked as exc:
        payload = {
            "contract_version": CONTRACT_VERSION,
            **exc.to_payload(),
            "generated_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        _write_json(status_path, payload)
        return payload

    try:
        resolved = resolve_dataset(
            "qlib_ashare_cleaned_v2",
            quant_data_root=quant_data_dir,
            require_basis="qfq",
        )
        canonical_provider = Path(resolved.snapshot_path)
    except (DataContractError, OSError, RuntimeError, ValueError):
        payload = _blocked_payload("cleaned_provider_unavailable")
        _write_json(status_path, payload)
        return payload

    resolved_identity = {
        "dataset_id": str(resolved.dataset_id),
        "snapshot_id": str(resolved.manifest.get("snapshot_id") or ""),
        "manifest_sha256": str(
            resolved.catalog_entry.get("snapshot_manifest_sha256") or ""
        ),
        "content_sha256": str(resolved.manifest.get("content_sha256") or ""),
    }
    if resolved_identity != quality.get("cleaned_provider_identity"):
        payload = _blocked_payload("cleaned_provider_quality_mismatch")
        _write_json(status_path, payload)
        return payload

    try:
        result = prepare_runtime_provider(
            canonical_provider=canonical_provider,
            intervals_path=intervals_path,
            runtime_root=runtime_root,
            data_manifest_path=manifest_path,
        )
    except Exception as exc:
        payload = _blocked_payload("provider_build_failed")
        payload["error_type"] = type(exc).__name__
        payload["error"] = _sanitized_error(exc) or "provider build failed"
        _write_json(status_path, payload)
        return payload

    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "ready",
        "trainable": True,
        "reason_code": None,
        "summary": "Robust CSI800 data is ready and bound to the current data manifest.",
        "summary_zh": "鲁棒 CSI800 数据已验收，并绑定当前数据清单。",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider_uri": result["provider_uri"],
        "provider_key": result["provider_key"],
        "provider_tree_sha256": result["provider_tree_sha256"],
        "data_manifest_sha256": file_sha256(manifest_path),
        "membership_intervals_sha256": file_sha256(intervals_path),
        "quality_report_id": quality["quality_report_id"],
        "quality_evidence_sha256": quality["quality_evidence_sha256"],
        "cleaned_provider_identity": resolved_identity,
        "canonical_snapshot_id": canonical_provider.name,
        "symbols": result["symbols"],
        "intervals": result["intervals"],
        "reused": result["reused"],
    }
    _write_json(status_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument(
        "--quality-pointer-path",
        type=Path,
        default=reports_root() / "data_quality_report_latest.json",
    )
    args = parser.parse_args(argv)
    payload = prepare_robust_signal_data(
        quant_data_dir=args.quant_data_dir,
        status_path=args.status_path,
        quality_pointer_path=args.quality_pointer_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
