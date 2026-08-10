#!/usr/bin/env python3
"""Prepare and attest the immutable CSI800 provider used by signal research."""

from __future__ import annotations

import argparse
import json
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
from quant_workflow.quant_data_qlib_provider import (  # noqa: E402
    DirectProviderError,
    _load_descriptor,
    build_direct_provider_descriptor,
)
from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402
from robust_signal_data_contract import (  # noqa: E402
    ACCEPTANCE_CONTRACT,
    ACCEPTANCE_OWNER,
    CONTRACT_VERSION,
    RobustSignalDataBlocked,
    load_verified_cleaned_quality_evidence,
    publish_robust_signal_data_readiness,
)


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:/[A-Za-z0-9_.~@%+,:=\\/-]+|[A-Za-z]:[\\/][^\s'\"<>]+)"
)


def _sanitized_error(exc: Exception) -> str:
    return _ABSOLUTE_PATH_RE.sub("[path]", str(exc))[:500]


DEFAULT_STATUS_PATH = (
    reports_root() / "robust_signal" / "robust_signal_data_readiness_latest.json"
)


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
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "status": "blocked_data",
        "trainable": False,
        "reason_code": reason_code,
        "summary": summary,
        "summary_zh": summary_zh,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _direct_cleaned_identity(descriptor: Path, quant_data_dir: Path) -> dict[str, str]:
    context = _load_descriptor(descriptor, quant_data_dir)
    dataset = context.descriptor.get("dataset") or {}
    index = context.descriptor.get("index_dataset") or {}
    cleaning = context.descriptor.get("cleaning") or {}
    return {
        "provider_contract": "direct_parquet_v1",
        "dataset_id": str(dataset.get("dataset_id") or ""),
        "snapshot_id": str(dataset.get("snapshot_id") or ""),
        "manifest_sha256": str(dataset.get("manifest_sha256") or ""),
        "content_sha256": str(dataset.get("content_sha256") or ""),
        "descriptor_path": descriptor.relative_to(quant_data_dir).as_posix(),
        "descriptor_sha256": file_sha256(descriptor),
        "cleaning_mode": str(cleaning.get("mode") or ""),
        "index_dataset_id": str(index.get("dataset_id") or ""),
        "index_snapshot_id": str(index.get("snapshot_id") or ""),
        "index_content_sha256": str(index.get("content_sha256") or ""),
        "index_manifest_sha256": str(index.get("manifest_sha256") or ""),
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
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )
    if not intervals_path.is_file():
        payload = _blocked_payload("membership_intervals_missing")
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )

    try:
        quality = load_verified_cleaned_quality_evidence(quality_pointer_path)
    except RobustSignalDataBlocked as exc:
        payload = {
            "contract_version": CONTRACT_VERSION,
            "acceptance_owner": ACCEPTANCE_OWNER,
            "acceptance_contract": ACCEPTANCE_CONTRACT,
            **exc.to_payload(),
            "generated_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )

    quality_identity = quality.get("cleaned_provider_identity") or {}
    direct_mode = (
        quality_identity.get("provider_contract") == "direct_parquet_v1"
        and quality_identity.get("dataset_id") == "ashare_daily_qfq"
    )
    try:
        if direct_mode:
            descriptor = build_direct_provider_descriptor(
                quant_data_root=quant_data_dir,
                cleaning_mode="cleaned",
            )
            canonical_provider = descriptor.parent
            resolved_identity = _direct_cleaned_identity(
                descriptor,
                quant_data_dir,
            )
        else:
            resolved = resolve_dataset(
                "qlib_ashare_cleaned_v2",
                quant_data_root=quant_data_dir,
                require_basis="qfq",
            )
            canonical_provider = Path(resolved.snapshot_path)
            resolved_identity = {
                "dataset_id": str(resolved.dataset_id),
                "snapshot_id": str(resolved.manifest.get("snapshot_id") or ""),
                "manifest_sha256": str(
                    resolved.catalog_entry.get("snapshot_manifest_sha256") or ""
                ),
                "content_sha256": str(resolved.manifest.get("content_sha256") or ""),
            }
    except (DataContractError, DirectProviderError, OSError, RuntimeError, ValueError):
        payload = _blocked_payload("cleaned_provider_unavailable")
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )
    if resolved_identity != quality.get("cleaned_provider_identity"):
        payload = _blocked_payload("cleaned_provider_quality_mismatch")
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )

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
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )

    payload = {
        "contract_version": CONTRACT_VERSION,
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
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
        "quality_evidence_artifact_name": quality[
            "quality_evidence_artifact_name"
        ],
        "quality_evidence_sha256": quality["quality_evidence_sha256"],
        "cleaned_provider_identity": resolved_identity,
        "canonical_snapshot_id": resolved_identity.get("snapshot_id") or canonical_provider.name,
        "symbols": result["symbols"],
        "intervals": result["intervals"],
        "reused": result["reused"],
    }
    return publish_robust_signal_data_readiness(
        pointer_path=status_path, payload=payload
    )


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
