#!/usr/bin/env python3
"""Prepare and attest the immutable CSI800 provider used by signal research."""

from __future__ import annotations

import argparse
import hashlib
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

from _paths import qlib_workflow_provider_uri, quant_data_root, reports_root  # noqa: E402
from _mlflow_run_provenance import (  # noqa: E402
    RunProvenanceError,
    normalize_provider_lineage,
    provider_identity,
    provider_lineages_match,
)
from prepare_robust_provider import file_sha256, prepare_runtime_provider  # noqa: E402
from quant_data_service_client import (  # noqa: E402
    DEFAULT_QUANT_DATA_SERVICE_URL,
    QuantDataClientError,
    QuantDataServiceClient,
)
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
REQUIRED_DATASETS = (
    ("ashare_daily_qfq", "qfq"),
    ("cn_index_daily", "derived"),
    ("pit_csi800", "not_applicable"),
    ("robust_reference", "not_applicable"),
    ("ashare_fundamentals_pit", "not_applicable"),
    ("qlib_cleaning_plan", "not_applicable"),
)


def _blocked_payload(reason_code: str) -> dict[str, Any]:
    messages = {
        "cleaned_provider_unavailable": (
            "The current cleaned Qlib Provider is unavailable.",
            "当前 cleaned Qlib Provider 不可用，请先完成 quant_data 数据后端验收。",
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
        "quant_data_service_unavailable": (
            "The quant_data data backend is unavailable or rejected a required dataset.",
            "quant_data 数据后端不可用，或正式数据集未通过服务验收。",
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


def prepare_robust_signal_data(
    *,
    quant_data_dir: Path,
    status_path: Path,
    quality_pointer_path: Path | None = None,
    client: QuantDataServiceClient | None = None,
) -> dict[str, Any]:
    Path(quant_data_dir).expanduser().resolve()
    status_path = Path(status_path).expanduser().resolve()
    del quality_pointer_path

    owns_client = client is None
    service = client or QuantDataServiceClient(
        base_url=os.getenv(
            "QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL
        )
    )
    try:
        service.ready()
        data_identities = {
            dataset_id: service.resolve_dataset(dataset_id, basis=basis)
            .identity.as_dict()
            for dataset_id, basis in REQUIRED_DATASETS
        }
    except QuantDataClientError:
        payload = _blocked_payload("quant_data_service_unavailable")
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )
    finally:
        if owns_client:
            service.close()

    try:
        canonical_provider = qlib_workflow_provider_uri("cleaned")
        resolved_identity = normalize_provider_lineage(
            provider_identity(canonical_provider)
        )
    except (RunProvenanceError, OSError, RuntimeError, ValueError):
        payload = _blocked_payload("cleaned_provider_unavailable")
        return publish_robust_signal_data_readiness(
            pointer_path=status_path, payload=payload
        )
    identity_payload = json.dumps(
        data_identities, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    data_identity_sha256 = hashlib.sha256(identity_payload).hexdigest()
    provider_key = str(resolved_identity.get("descriptor_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", provider_key):
        payload = _blocked_payload("cleaned_provider_unavailable")
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
        "provider_uri": str(canonical_provider),
        "provider_key": provider_key,
        "provider_tree_sha256": provider_key,
        "data_identity_sha256": data_identity_sha256,
        "data_identities": data_identities,
        "cleaned_provider_identity": resolved_identity,
        "canonical_snapshot_id": (
            resolved_identity.get("snapshot_id") or canonical_provider.name
        ),
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
