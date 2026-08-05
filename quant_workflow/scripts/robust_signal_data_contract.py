#!/usr/bin/env python3
"""Validate the immutable data contract required by robust CSI800 research."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .prepare_robust_provider import (
        RuntimeProviderIdentityCache,
        runtime_provider_identity,
    )
except ImportError:  # Direct script execution and quant_workflow test mode.
    from prepare_robust_provider import (
        RuntimeProviderIdentityCache,
        runtime_provider_identity,
    )


CONTRACT_VERSION = "robust_signal_data_v1"
PROVIDER_FORMAT_VERSION = 3
PROVIDER_METADATA_FILENAME = "ROBUST_PROVIDER.json"
MAX_STATUS_BYTES = 1024 * 1024
MAX_QUALITY_POINTER_BYTES = 64 * 1024
MAX_QUALITY_EVIDENCE_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUALITY_REPORT_ID_RE = re.compile(r"^data-quality-[0-9a-f]{16}$")
QUALITY_EVIDENCE_NAME_RE = re.compile(
    r"^data_quality_report_\d{8}_\d{6}_[0-9a-f]{16}\.json$"
)
CLEANED_PROVIDER_DATASET_ID = "qlib_ashare_cleaned_v2"
CLEANED_PROVIDER_IDENTITY_FIELDS = (
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
)


class RobustSignalDataBlocked(ValueError):
    """A structured, user-actionable robust-data readiness failure."""

    def __init__(self, reason_code: str, summary: str, summary_zh: str) -> None:
        super().__init__(summary_zh)
        self.reason_code = reason_code
        self.summary = summary
        self.summary_zh = summary_zh

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": "blocked_data",
            "trainable": False,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "summary_zh": self.summary_zh,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocked(reason_code: str) -> RobustSignalDataBlocked:
    messages = {
        "robust_readiness_missing": (
            "Robust CSI800 data has not been prepared.",
            "鲁棒 CSI800 数据尚未验收，请先在数据页运行“验收并准备鲁棒 CSI800 数据”。",
        ),
        "robust_readiness_invalid": (
            "The robust CSI800 readiness record is invalid.",
            "鲁棒 CSI800 数据验收记录无效，请重新运行数据准备任务。",
        ),
        "data_manifest_missing": (
            "DATA_MANIFEST.json is unavailable.",
            "当前 DATA_MANIFEST.json 不存在，无法验收鲁棒 CSI800 数据。",
        ),
        "data_manifest_changed": (
            "The robust provider was built from an older data manifest.",
            "鲁棒 CSI800 provider 绑定的是旧数据清单，请重新验收并准备。",
        ),
        "provider_outside_runtime": (
            "The robust provider is outside the controlled runtime directory.",
            "鲁棒 CSI800 provider 不在受控运行目录中，已拒绝使用。",
        ),
        "provider_unavailable": (
            "The prepared robust provider is unavailable.",
            "已验收的鲁棒 CSI800 provider 不可用，请重新准备。",
        ),
        "provider_identity_mismatch": (
            "The robust provider identity no longer matches its readiness record.",
            "鲁棒 CSI800 provider 身份与验收记录不一致，请重新准备。",
        ),
        "csi800_instruments_missing": (
            "The prepared provider does not contain PIT CSI800 instruments.",
            "已准备的 provider 缺少历史时点 CSI800 股票池，请重新构建。",
        ),
        "cleaned_provider_unavailable": (
            "The current cleaned Qlib V2 snapshot is unavailable.",
            "当前 cleaned Qlib V2 快照不可用，请先完成原生 V2 构建和数据质量验收。",
        ),
        "membership_intervals_missing": (
            "Historical CSI800 membership intervals are unavailable.",
            "历史时点 CSI800 成员区间不存在，无法准备鲁棒研究数据。",
        ),
        "provider_build_failed": (
            "The robust CSI800 provider could not be prepared.",
            "鲁棒 CSI800 provider 准备失败，请查看本次任务日志。",
        ),
        "quality_evidence_missing": (
            "The full data quality evidence is unavailable.",
            "完整数据质量验收证据不存在，请先在数据页运行完整数据质量报告。",
        ),
        "quality_evidence_invalid": (
            "The full data quality evidence is invalid.",
            "完整数据质量验收证据无效，请重新运行完整数据质量报告。",
        ),
        "quality_evidence_not_trainable": (
            "The full data quality evidence does not permit training.",
            "完整数据质量报告未通过可训练门槛，已禁止准备鲁棒训练数据。",
        ),
        "quality_evidence_changed": (
            "The quality evidence changed after robust data preparation.",
            "鲁棒数据准备后，数据质量证据已经变化，请重新验收并准备。",
        ),
        "cleaned_provider_quality_mismatch": (
            "The cleaned provider does not match the approved quality evidence.",
            "当前 cleaned provider 与完整数据质量证据不一致，请重新验收数据。",
        ),
    }
    summary, summary_zh = messages[reason_code]
    return RobustSignalDataBlocked(reason_code, summary, summary_zh)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_STATUS_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if path.stat().st_size > max_bytes:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _valid_snapshot_fingerprint(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        value.get("algorithm")
        == "relative-path-size-mtime-ctime-device-inode-v2"
        and all(
            isinstance(value.get(field), int)
            and not isinstance(value.get(field), bool)
            and int(value[field]) >= 0
            for field in ("file_count", "total_bytes", "scanned_entries")
        )
        and isinstance(value.get("sha256"), str)
        and SHA256_RE.fullmatch(str(value["sha256"]))
    )


def load_verified_cleaned_quality_evidence(pointer_path: Path) -> dict[str, Any]:
    """Load one immutable full-quality publication for the cleaned V2 provider."""

    pointer_path = Path(pointer_path).expanduser()
    pointer_bytes = _read_regular_bytes(
        pointer_path, max_bytes=MAX_QUALITY_POINTER_BYTES
    )
    if pointer_bytes is None:
        raise _blocked("quality_evidence_missing")
    try:
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked("quality_evidence_invalid") from exc
    if not isinstance(pointer, Mapping):
        raise _blocked("quality_evidence_invalid")

    report_id = pointer.get("report_id")
    evidence_sha256 = pointer.get("evidence_sha256")
    evidence_artifact = pointer.get("evidence_artifact")
    if (
        pointer.get("schema_version") != 3
        or pointer.get("publication_contract")
        != "atomic_json_pointer_to_immutable_quality_evidence_v2"
        or not isinstance(report_id, str)
        or not QUALITY_REPORT_ID_RE.fullmatch(report_id)
        or not isinstance(evidence_sha256, str)
        or not SHA256_RE.fullmatch(evidence_sha256)
        or report_id != f"data-quality-{evidence_sha256[:16]}"
        or not isinstance(evidence_artifact, str)
        or not QUALITY_EVIDENCE_NAME_RE.fullmatch(evidence_artifact)
        or not evidence_artifact.endswith(f"_{evidence_sha256[:16]}.json")
    ):
        raise _blocked("quality_evidence_invalid")

    evidence_path = pointer_path.parent / evidence_artifact
    if evidence_path.parent != pointer_path.parent:
        raise _blocked("quality_evidence_invalid")
    evidence_bytes = _read_regular_bytes(
        evidence_path, max_bytes=MAX_QUALITY_EVIDENCE_BYTES
    )
    if (
        evidence_bytes is None
        or hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256
    ):
        raise _blocked("quality_evidence_invalid")
    try:
        evidence = json.loads(evidence_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked("quality_evidence_invalid") from exc
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("schema_version") != 3
        or evidence.get("evidence_contract")
        != "immutable_data_quality_evidence_v2"
    ):
        raise _blocked("quality_evidence_invalid")
    if evidence.get("trainable") is not True:
        raise _blocked("quality_evidence_not_trainable")

    thresholds = evidence.get("training_thresholds")
    scanned_instruments = evidence.get("scanned_instruments")
    if (
        evidence.get("scan_mode") != "full"
        or evidence.get("coverage_status") != "full"
        or evidence.get("verification_status") != "verified"
        or evidence.get("requested_sample") != 0
        or not isinstance(thresholds, Mapping)
        or thresholds.get("authoritative") is not True
        or any(
            not isinstance(thresholds.get(field), int)
            or isinstance(thresholds.get(field), bool)
            or int(thresholds[field]) < minimum
            for field, minimum in (
                ("min_calendar_days", 252),
                ("min_instruments", 100),
                ("min_merged_instruments", 100),
            )
        )
        or not isinstance(scanned_instruments, int)
        or isinstance(scanned_instruments, bool)
        or scanned_instruments < int(thresholds["min_instruments"])
    ):
        raise _blocked("quality_evidence_not_trainable")

    data_contract = evidence.get("data_contract")
    provider_identities = (
        data_contract.get("qlib_providers")
        if isinstance(data_contract, Mapping)
        else None
    )
    cleaned_identity = (
        provider_identities.get("cleaned")
        if isinstance(provider_identities, Mapping)
        else None
    )
    snapshot_state = (
        data_contract.get("snapshot_state")
        if isinstance(data_contract, Mapping)
        else None
    )
    if (
        not isinstance(data_contract, Mapping)
        or data_contract.get("status") != "verified"
        or not isinstance(cleaned_identity, Mapping)
        or cleaned_identity.get("dataset_id") != CLEANED_PROVIDER_DATASET_ID
        or not isinstance(cleaned_identity.get("snapshot_id"), str)
        or not cleaned_identity.get("snapshot_id")
        or Path(str(cleaned_identity.get("snapshot_id"))).name
        != cleaned_identity.get("snapshot_id")
        or any(
            not isinstance(cleaned_identity.get(field), str)
            or not SHA256_RE.fullmatch(str(cleaned_identity[field]))
            for field in ("manifest_sha256", "content_sha256")
        )
        or not isinstance(snapshot_state, Mapping)
        or snapshot_state.get("stable") is not True
        or snapshot_state.get("content_revalidated") is not True
    ):
        raise _blocked("quality_evidence_invalid")

    provider_quality = evidence.get("provider_quality")
    cleaned_quality = (
        provider_quality.get("cleaned")
        if isinstance(provider_quality, Mapping)
        else None
    )
    if not isinstance(cleaned_quality, Mapping):
        raise _blocked("quality_evidence_invalid")
    if any(
        cleaned_quality.get(field) != cleaned_identity.get(field)
        for field in CLEANED_PROVIDER_IDENTITY_FIELDS
    ):
        raise _blocked("cleaned_provider_quality_mismatch")

    cleaned_readiness = cleaned_quality.get("training_readiness")
    cleaned_failure_reasons = cleaned_quality.get("failure_reasons")
    readiness_failure_reasons = (
        cleaned_readiness.get("failure_reasons")
        if isinstance(cleaned_readiness, Mapping)
        else None
    )
    if (
        cleaned_quality.get("trainable") is not True
        or not isinstance(cleaned_readiness, Mapping)
        or cleaned_readiness.get("status") != "trainable"
        or not isinstance(cleaned_failure_reasons, list)
        or bool(cleaned_failure_reasons)
        or not isinstance(readiness_failure_reasons, list)
        or bool(readiness_failure_reasons)
    ):
        raise _blocked("quality_evidence_not_trainable")

    stage_names = ("before", "after_scan", "after_revalidation", "after")
    stage_states = [
        snapshot_state.get(stage, {}).get(CLEANED_PROVIDER_DATASET_ID)
        if isinstance(snapshot_state.get(stage), Mapping)
        else None
        for stage in stage_names
    ]
    if (
        not all(_valid_snapshot_fingerprint(state) for state in stage_states)
        or any(state != stage_states[0] for state in stage_states[1:])
    ):
        raise _blocked("quality_evidence_invalid")

    return {
        "quality_report_id": report_id,
        "quality_evidence_sha256": evidence_sha256,
        "cleaned_provider_identity": {
            field: str(cleaned_identity[field])
            for field in CLEANED_PROVIDER_IDENTITY_FIELDS
        },
    }


def evaluate_robust_signal_data(
    *,
    status_path: Path,
    data_manifest_path: Path,
    provider_root: Path,
    quality_pointer_path: Path | None = None,
    quality_evidence_loader: Callable[[Path], Mapping[str, Any]] = (
        load_verified_cleaned_quality_evidence
    ),
    provider_identity: Callable[[Path, str], Mapping[str, str] | None] = (
        runtime_provider_identity
    ),
) -> dict[str, Any]:
    """Return a verified readiness payload or raise a structured block."""

    status_path = Path(status_path)
    data_manifest_path = Path(data_manifest_path)
    provider_root = Path(provider_root).expanduser().resolve()
    if not data_manifest_path.is_file():
        raise _blocked("data_manifest_missing")
    current_manifest_sha256 = file_sha256(data_manifest_path)

    status = _read_json_object(status_path)
    if status is None:
        raise _blocked("robust_readiness_missing")
    if status.get("status") == "blocked_data":
        reason_code = str(status.get("reason_code") or "robust_readiness_invalid")
        if reason_code not in {
            "robust_readiness_missing",
            "robust_readiness_invalid",
            "data_manifest_missing",
            "data_manifest_changed",
            "provider_outside_runtime",
            "provider_unavailable",
            "provider_identity_mismatch",
            "csi800_instruments_missing",
            "cleaned_provider_unavailable",
            "membership_intervals_missing",
            "provider_build_failed",
            "quality_evidence_missing",
            "quality_evidence_invalid",
            "quality_evidence_not_trainable",
            "quality_evidence_changed",
            "cleaned_provider_quality_mismatch",
        }:
            reason_code = "robust_readiness_invalid"
        raise _blocked(reason_code)
    if (
        status.get("contract_version") != CONTRACT_VERSION
        or status.get("status") != "ready"
        or status.get("trainable") is not True
    ):
        raise _blocked("robust_readiness_invalid")

    recorded_manifest_sha256 = status.get("data_manifest_sha256")
    if not isinstance(recorded_manifest_sha256, str) or not SHA256_RE.fullmatch(
        recorded_manifest_sha256
    ):
        raise _blocked("robust_readiness_invalid")
    if recorded_manifest_sha256 != current_manifest_sha256:
        raise _blocked("data_manifest_changed")

    if quality_pointer_path is None:
        quality_pointer_path = status_path.parent.parent / "data_quality_report_latest.json"
    current_quality = dict(quality_evidence_loader(Path(quality_pointer_path)))
    if (
        status.get("quality_report_id") != current_quality.get("quality_report_id")
        or status.get("quality_evidence_sha256")
        != current_quality.get("quality_evidence_sha256")
    ):
        raise _blocked("quality_evidence_changed")
    if status.get("cleaned_provider_identity") != current_quality.get(
        "cleaned_provider_identity"
    ):
        raise _blocked("cleaned_provider_quality_mismatch")

    provider_text = status.get("provider_uri")
    if not isinstance(provider_text, str) or not provider_text.strip():
        raise _blocked("provider_unavailable")
    provider = Path(provider_text).expanduser()
    try:
        provider = provider.resolve(strict=False)
    except OSError as exc:
        raise _blocked("provider_unavailable") from exc
    if provider.parent != provider_root or provider.is_symlink():
        raise _blocked("provider_outside_runtime")
    if not provider.is_dir():
        raise _blocked("provider_unavailable")

    provider_key = status.get("provider_key")
    provider_tree_sha256 = status.get("provider_tree_sha256")
    if (
        not isinstance(provider_key, str)
        or not SHA256_RE.fullmatch(provider_key)
        or provider.name != provider_key[:16]
        or not isinstance(provider_tree_sha256, str)
        or not SHA256_RE.fullmatch(provider_tree_sha256)
    ):
        raise _blocked("provider_identity_mismatch")
    metadata = _read_json_object(provider / PROVIDER_METADATA_FILENAME)
    if (
        metadata is None
        or metadata.get("format_version") != PROVIDER_FORMAT_VERSION
        or metadata.get("provider_key") != provider_key
        or metadata.get("provider_tree_sha256") != provider_tree_sha256
        or metadata.get("data_manifest_sha256") != current_manifest_sha256
    ):
        raise _blocked("provider_identity_mismatch")
    runtime_identity = provider_identity(provider, provider_key)
    if (
        runtime_identity is None
        or runtime_identity.get("provider_key") != provider_key
        or runtime_identity.get("provider_tree_sha256") != provider_tree_sha256
    ):
        raise _blocked("provider_identity_mismatch")
    if not (provider / "instruments" / "csi800.txt").is_file():
        raise _blocked("csi800_instruments_missing")

    return {
        "contract_version": CONTRACT_VERSION,
        "status": "ready",
        "trainable": True,
        "reason_code": None,
        "summary": "Robust CSI800 data is ready and bound to the current data manifest.",
        "summary_zh": "鲁棒 CSI800 数据已验收，并绑定当前数据清单。",
        "generated_at": status.get("generated_at"),
        "provider_uri": str(provider),
        "provider_name": provider.name,
        "provider_key": provider_key,
        "provider_tree_sha256": provider_tree_sha256,
        "data_manifest_sha256": current_manifest_sha256,
        "membership_intervals_sha256": metadata.get(
            "membership_intervals_sha256"
        ),
        "quality_report_id": current_quality.get("quality_report_id"),
        "quality_evidence_sha256": current_quality.get("quality_evidence_sha256"),
        "cleaned_provider_identity": current_quality.get(
            "cleaned_provider_identity"
        ),
        "canonical_snapshot_id": status.get("canonical_snapshot_id"),
        "symbols": status.get("symbols"),
        "intervals": status.get("intervals"),
    }


def public_readiness_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove local paths before readiness data is returned by an API."""

    return {
        key: payload.get(key)
        for key in (
            "contract_version",
            "status",
            "trainable",
            "reason_code",
            "summary",
            "summary_zh",
            "generated_at",
            "provider_name",
            "provider_key",
            "provider_tree_sha256",
            "data_manifest_sha256",
            "membership_intervals_sha256",
            "quality_report_id",
            "quality_evidence_sha256",
            "cleaned_provider_identity",
            "canonical_snapshot_id",
            "symbols",
            "intervals",
        )
        if payload.get(key) is not None
    }
