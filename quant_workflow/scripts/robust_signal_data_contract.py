#!/usr/bin/env python3
"""Validate the immutable data contract required by robust CSI800 research."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from ._paths import reports_root
    from .prepare_robust_provider import (
        runtime_provider_identity,
    )
except ImportError:  # Direct script execution and quant_workflow test mode.
    from _paths import reports_root
    from prepare_robust_provider import (
        runtime_provider_identity,
    )


DIRECT_PROVIDER_CONTRACT = "direct_parquet_v1"
DIRECT_CLEANED_PROVIDER_DATASET_ID = "ashare_daily_qfq"


CONTRACT_VERSION = "robust_signal_data_v3"
ACCEPTANCE_OWNER = "data_module"
ACCEPTANCE_CONTRACT = "data_module_accepted_content_addressed_snapshot_v2"
READINESS_POINTER_CONTRACT = "robust_signal_data_readiness_pointer_v1"
READINESS_ARTIFACT_DIRNAME = "readiness_artifacts"
READINESS_ARTIFACT_PREFIX = "robust_signal_data_readiness_"
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
DIRECT_CLEANED_PROVIDER_IDENTITY_FIELDS = (
    "provider_contract",
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
    "descriptor_path",
    "descriptor_sha256",
    "cleaning_mode",
    "index_dataset_id",
    "index_snapshot_id",
    "index_content_sha256",
    "index_manifest_sha256",
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


def _cleaned_identity_kind(identity: Mapping[str, Any] | None) -> str | None:
    """Return the accepted cleaned-provider format without guessing from paths."""

    if not isinstance(identity, Mapping):
        return None
    dataset_id = identity.get("dataset_id")
    if dataset_id == CLEANED_PROVIDER_DATASET_ID:
        if all(
            isinstance(identity.get(field), str)
            and SHA256_RE.fullmatch(str(identity[field]))
            for field in ("manifest_sha256", "content_sha256")
        ):
            snapshot_id = str(identity.get("snapshot_id") or "")
            if snapshot_id and Path(snapshot_id).name == snapshot_id:
                return "native"
        return None
    if dataset_id != DIRECT_CLEANED_PROVIDER_DATASET_ID:
        return None
    if identity.get("provider_contract") != DIRECT_PROVIDER_CONTRACT:
        return None
    if identity.get("cleaning_mode") != "cleaned":
        return None
    if not all(
        isinstance(identity.get(field), str)
        and SHA256_RE.fullmatch(str(identity[field]))
        for field in (
            "manifest_sha256",
            "content_sha256",
            "descriptor_sha256",
            "index_content_sha256",
            "index_manifest_sha256",
        )
    ):
        return None
    snapshot_id = str(identity.get("snapshot_id") or "")
    index_snapshot_id = str(identity.get("index_snapshot_id") or "")
    descriptor_path = str(identity.get("descriptor_path") or "")
    if (
        not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or not index_snapshot_id
        or Path(index_snapshot_id).name != index_snapshot_id
        or not descriptor_path
        or Path(descriptor_path).is_absolute()
        or ".." in Path(descriptor_path).parts
        or identity.get("index_dataset_id") != "cn_index_daily"
    ):
        return None
    return "direct"


def _cleaned_identity_is_valid(identity: Mapping[str, Any] | None) -> bool:
    return _cleaned_identity_kind(identity) is not None


def _cleaned_identities_match(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> bool:
    left_kind = _cleaned_identity_kind(left)
    right_kind = _cleaned_identity_kind(right)
    if left_kind is None or left_kind != right_kind:
        return False
    fields = (
        CLEANED_PROVIDER_IDENTITY_FIELDS
        if left_kind == "native"
        else DIRECT_CLEANED_PROVIDER_IDENTITY_FIELDS
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _cleaned_identity_for_output(identity: Mapping[str, Any]) -> dict[str, str]:
    kind = _cleaned_identity_kind(identity)
    if kind is None:
        raise _blocked("quality_evidence_invalid")
    fields = (
        CLEANED_PROVIDER_IDENTITY_FIELDS
        if kind == "native"
        else DIRECT_CLEANED_PROVIDER_IDENTITY_FIELDS
    )
    return {field: str(identity[field]) for field in fields}


def _snapshot_state_dataset_id(kind: str) -> str:
    return (
        CLEANED_PROVIDER_DATASET_ID
        if kind == "native"
        else DIRECT_CLEANED_PROVIDER_DATASET_ID
    )


def _direct_descriptor_for_provider(provider: Path) -> Path | None:
    candidate = provider if provider.name == "PROVIDER_DESCRIPTOR.json" else provider / "PROVIDER_DESCRIPTOR.json"
    return candidate if candidate.is_file() and not candidate.is_symlink() else None


def _quant_data_root_for_provider(provider: Path) -> Path:
    for parent in (provider, *provider.parents):
        if parent.name == "quant_data":
            return parent
    raise _blocked("provider_outside_runtime")


def _direct_cleaned_identity(provider: Path) -> dict[str, str]:
    descriptor = _direct_descriptor_for_provider(provider)
    if descriptor is None:
        raise _blocked("provider_unavailable")
    try:
        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        root = _quant_data_root_for_provider(provider)
        context = _load_descriptor(descriptor, root)
    except RobustSignalDataBlocked:
        raise
    except Exception as exc:
        raise _blocked("provider_identity_mismatch") from exc
    payload = context.descriptor
    dataset = payload.get("dataset")
    index = payload.get("index_dataset")
    cleaning = payload.get("cleaning") or {}
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(index, Mapping)
        or cleaning.get("mode") != "cleaned"
        or dataset.get("dataset_id") != DIRECT_CLEANED_PROVIDER_DATASET_ID
        or index.get("dataset_id") != "cn_index_daily"
        or set(index.get("required_symbols") or ())
        != {
            "SH000300",
            "SH000905",
            "SH000906",
            "SZ399006",
            "SH000688",
        }
    ):
        raise _blocked("provider_identity_mismatch")
    try:
        descriptor_path = descriptor.relative_to(root).as_posix()
    except ValueError as exc:
        raise _blocked("provider_outside_runtime") from exc
    return {
        "provider_contract": DIRECT_PROVIDER_CONTRACT,
        "dataset_id": str(dataset.get("dataset_id") or ""),
        "snapshot_id": str(dataset.get("snapshot_id") or ""),
        "manifest_sha256": str(dataset.get("manifest_sha256") or ""),
        "content_sha256": str(dataset.get("content_sha256") or ""),
        "descriptor_path": descriptor_path,
        "descriptor_sha256": file_sha256(descriptor),
        "cleaning_mode": str(cleaning.get("mode") or ""),
        "index_dataset_id": str(index.get("dataset_id") or ""),
        "index_snapshot_id": str(index.get("snapshot_id") or ""),
        "index_content_sha256": str(index.get("content_sha256") or ""),
        "index_manifest_sha256": str(index.get("manifest_sha256") or ""),
    }


def _read_direct_membership(provider: Path) -> Any:
    descriptor = _direct_descriptor_for_provider(provider)
    if descriptor is None:
        return None
    try:
        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        return _load_descriptor(
            descriptor,
            _quant_data_root_for_provider(provider),
        ).membership
    except Exception as exc:
        raise _blocked("csi800_instruments_missing") from exc


def _has_csi800_membership(membership: Any) -> bool:
    try:
        codes = set(membership["index_code"].astype(str).str.upper())
    except (AttributeError, KeyError, TypeError):
        return False
    return bool(
        codes.intersection({"000300.XSHG", "SH000300", "000300"})
        and codes.intersection({"000905.XSHG", "SH000905", "000905"})
    )


def _provider_is_under(provider: Path, root: Path) -> bool:
    try:
        provider.relative_to(root)
        return True
    except ValueError:
        return False


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
        "readiness_artifact_missing": (
            "The frozen robust CSI800 readiness artifact is unavailable.",
            "冻结的鲁棒 CSI800 数据验收文件不存在，请重新运行数据准备任务。",
        ),
        "readiness_artifact_changed": (
            "The frozen robust CSI800 readiness artifact no longer matches its hash.",
            "冻结的鲁棒 CSI800 数据验收文件与提交时哈希不一致，已拒绝训练。",
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


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def readiness_artifact_path_for_sha(pointer_path: Path, artifact_sha256: str) -> Path:
    if not isinstance(artifact_sha256, str) or not SHA256_RE.fullmatch(artifact_sha256):
        raise _blocked("robust_readiness_invalid")
    pointer_path = Path(pointer_path).expanduser().resolve(strict=False)
    return (
        pointer_path.parent
        / READINESS_ARTIFACT_DIRNAME
        / f"{READINESS_ARTIFACT_PREFIX}{artifact_sha256}.json"
    )


def publish_robust_signal_data_readiness(
    *, pointer_path: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish one immutable readiness artifact and atomically advance latest."""

    pointer_path = Path(pointer_path).expanduser().resolve(strict=False)
    artifact_bytes = _json_bytes(payload)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_path = readiness_artifact_path_for_sha(pointer_path, artifact_sha256)
    existing = _read_regular_bytes(artifact_path, max_bytes=MAX_STATUS_BYTES)
    if existing is None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with artifact_path.open("xb") as stream:
                stream.write(artifact_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            artifact_path.chmod(0o444)
        except FileExistsError:
            existing = _read_regular_bytes(artifact_path, max_bytes=MAX_STATUS_BYTES)
    verified_artifact = _read_regular_bytes(artifact_path, max_bytes=MAX_STATUS_BYTES)
    if verified_artifact != artifact_bytes:
        raise RuntimeError("content-addressed readiness artifact collision")

    relative_artifact = artifact_path.relative_to(pointer_path.parent).as_posix()
    pointer = {
        "pointer_contract": READINESS_POINTER_CONTRACT,
        "artifact": relative_artifact,
        "artifact_sha256": artifact_sha256,
        "status": payload.get("status"),
        "trainable": payload.get("trainable"),
        "reason_code": payload.get("reason_code"),
        "generated_at": payload.get("generated_at"),
    }
    _atomic_write_bytes(pointer_path, _json_bytes(pointer))
    return {
        **dict(payload),
        "readiness_artifact_name": artifact_path.name,
        "readiness_artifact_sha256": artifact_sha256,
    }


def resolve_robust_signal_data_readiness(
    pointer_path: Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Resolve and hash-check the immutable artifact selected by latest."""

    pointer_path = Path(pointer_path).expanduser().resolve(strict=False)
    pointer = _read_json_object(pointer_path)
    if pointer is None:
        raise _blocked("robust_readiness_missing")
    artifact_sha256 = pointer.get("artifact_sha256")
    if (
        pointer.get("pointer_contract") != READINESS_POINTER_CONTRACT
        or not isinstance(artifact_sha256, str)
        or not SHA256_RE.fullmatch(artifact_sha256)
    ):
        raise _blocked("robust_readiness_invalid")
    artifact_path = readiness_artifact_path_for_sha(pointer_path, artifact_sha256)
    expected_relative = artifact_path.relative_to(pointer_path.parent).as_posix()
    if pointer.get("artifact") != expected_relative:
        raise _blocked("robust_readiness_invalid")
    artifact_bytes = _read_regular_bytes(artifact_path, max_bytes=MAX_STATUS_BYTES)
    if artifact_bytes is None:
        raise _blocked("readiness_artifact_missing")
    if hashlib.sha256(artifact_bytes).hexdigest() != artifact_sha256:
        raise _blocked("readiness_artifact_changed")
    try:
        payload = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked("robust_readiness_invalid") from exc
    if not isinstance(payload, dict):
        raise _blocked("robust_readiness_invalid")
    return artifact_path, artifact_sha256, payload


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
    cleaned_kind = _cleaned_identity_kind(cleaned_identity)
    if (
        not isinstance(data_contract, Mapping)
        or data_contract.get("status") != "verified"
        or cleaned_kind is None
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
    if not _cleaned_identities_match(cleaned_quality, cleaned_identity):
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
    stage_dataset_id = _snapshot_state_dataset_id(cleaned_kind)
    stage_states = [
        snapshot_state.get(stage, {}).get(stage_dataset_id)
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
        "quality_evidence_artifact_name": evidence_artifact,
        "quality_evidence_sha256": evidence_sha256,
        "cleaned_provider_identity": _cleaned_identity_for_output(cleaned_identity),
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
        or status.get("quality_evidence_artifact_name")
        != current_quality.get("quality_evidence_artifact_name")
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
    if provider.is_symlink() or not _provider_is_under(provider, provider_root):
        raise _blocked("provider_outside_runtime")
    if not provider.is_dir():
        raise _blocked("provider_unavailable")

    provider_key = status.get("provider_key")
    provider_tree_sha256 = status.get("provider_tree_sha256")
    direct_descriptor = _direct_descriptor_for_provider(provider)
    direct_provider = direct_descriptor is not None
    if (
        not isinstance(provider_key, str)
        or not SHA256_RE.fullmatch(provider_key)
        or not isinstance(provider_tree_sha256, str)
        or not SHA256_RE.fullmatch(provider_tree_sha256)
        or (not direct_provider and provider.name != provider_key[:16])
    ):
        raise _blocked("provider_identity_mismatch")
    if direct_provider:
        if provider_key != file_sha256(direct_descriptor) or provider_tree_sha256 != provider_key:
            raise _blocked("provider_identity_mismatch")
        actual_cleaned_identity = _direct_cleaned_identity(provider)
        if not _cleaned_identities_match(actual_cleaned_identity, status.get("cleaned_provider_identity")):
            raise _blocked("cleaned_provider_quality_mismatch")
        if not _has_csi800_membership(_read_direct_membership(provider)):
            raise _blocked("csi800_instruments_missing")
        metadata = None
    else:
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
    if not direct_provider and not (provider / "instruments" / "csi800.txt").is_file():
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
        "membership_intervals_sha256": (
            metadata.get("membership_intervals_sha256")
            if metadata is not None
            else status.get("membership_intervals_sha256")
        ),
        "quality_report_id": current_quality.get("quality_report_id"),
        "quality_evidence_artifact_name": current_quality.get(
            "quality_evidence_artifact_name"
        ),
        "quality_evidence_sha256": current_quality.get("quality_evidence_sha256"),
        "cleaned_provider_identity": current_quality.get(
            "cleaned_provider_identity"
        ),
        "canonical_snapshot_id": status.get("canonical_snapshot_id"),
        "symbols": status.get("symbols"),
        "intervals": status.get("intervals"),
    }


def consume_accepted_robust_signal_data(
    *,
    artifact_path: Path,
    artifact_sha256: str,
    provider_root: Path,
    reports_dir: Path | None = None,
    provider_identity: Callable[[Path, str], Mapping[str, str] | None] = (
        runtime_provider_identity
    ),
) -> dict[str, Any]:
    """Consume one submission-frozen data-module readiness artifact.

    The data module owns source resolution, cleaning, business-quality checks and
    publication. Training and status consumers only verify the frozen readiness
    record plus the provider's content identity so an updated data catalog cannot
    silently change an existing training input.
    """

    controlled_reports_dir = Path(reports_dir or reports_root()).expanduser().resolve()
    readiness_artifact_root = (
        controlled_reports_dir / "robust_signal" / READINESS_ARTIFACT_DIRNAME
    ).resolve()
    artifact_path = Path(os.path.abspath(Path(artifact_path).expanduser()))
    provider_root = Path(provider_root).expanduser().resolve()
    if (
        not isinstance(artifact_sha256, str)
        or not SHA256_RE.fullmatch(artifact_sha256)
        or artifact_path.parent.resolve(strict=False) != readiness_artifact_root
        or artifact_path.name != f"{READINESS_ARTIFACT_PREFIX}{artifact_sha256}.json"
    ):
        raise _blocked("robust_readiness_invalid")
    artifact_bytes = _read_regular_bytes(artifact_path, max_bytes=MAX_STATUS_BYTES)
    if artifact_bytes is None:
        raise _blocked("readiness_artifact_missing")
    if hashlib.sha256(artifact_bytes).hexdigest() != artifact_sha256:
        raise _blocked("readiness_artifact_changed")
    try:
        status = json.loads(artifact_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked("robust_readiness_invalid") from exc
    if not isinstance(status, dict):
        raise _blocked("robust_readiness_invalid")
    if status.get("status") == "blocked_data":
        reason_code = str(status.get("reason_code") or "robust_readiness_invalid")
        if reason_code not in {
            "robust_readiness_missing",
            "robust_readiness_invalid",
            "data_manifest_missing",
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
            "readiness_artifact_missing",
            "readiness_artifact_changed",
        }:
            reason_code = "robust_readiness_invalid"
        raise _blocked(reason_code)

    if (
        status.get("contract_version") != CONTRACT_VERSION
        or status.get("acceptance_owner") != ACCEPTANCE_OWNER
        or status.get("acceptance_contract") != ACCEPTANCE_CONTRACT
        or status.get("status") != "ready"
        or status.get("trainable") is not True
    ):
        raise _blocked("robust_readiness_invalid")

    recorded_manifest_sha256 = status.get("data_manifest_sha256")
    membership_intervals_sha256 = status.get("membership_intervals_sha256")
    if (
        not isinstance(recorded_manifest_sha256, str)
        or not SHA256_RE.fullmatch(recorded_manifest_sha256)
        or not isinstance(membership_intervals_sha256, str)
        or not SHA256_RE.fullmatch(membership_intervals_sha256)
    ):
        raise _blocked("robust_readiness_invalid")
    report_id = status.get("quality_report_id")
    evidence_artifact_name = status.get("quality_evidence_artifact_name")
    evidence_sha256 = status.get("quality_evidence_sha256")
    cleaned_identity = status.get("cleaned_provider_identity")
    if (
        not isinstance(report_id, str)
        or not QUALITY_REPORT_ID_RE.fullmatch(report_id)
        or not isinstance(evidence_artifact_name, str)
        or not QUALITY_EVIDENCE_NAME_RE.fullmatch(evidence_artifact_name)
        or not isinstance(evidence_sha256, str)
        or not SHA256_RE.fullmatch(evidence_sha256)
        or report_id != f"data-quality-{evidence_sha256[:16]}"
        or not evidence_artifact_name.endswith(
            f"_{evidence_sha256[:16]}.json"
        )
        or _cleaned_identity_kind(cleaned_identity) is None
    ):
        raise _blocked("robust_readiness_invalid")

    evidence_path = controlled_reports_dir / evidence_artifact_name
    if evidence_path.parent != controlled_reports_dir:
        raise _blocked("quality_evidence_invalid")
    evidence_bytes = _read_regular_bytes(
        evidence_path,
        max_bytes=MAX_QUALITY_EVIDENCE_BYTES,
    )
    if evidence_bytes is None:
        raise _blocked("quality_evidence_missing")
    if hashlib.sha256(evidence_bytes).hexdigest() != evidence_sha256:
        raise _blocked("quality_evidence_changed")
    try:
        evidence = json.loads(evidence_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked("quality_evidence_invalid") from exc
    data_contract = evidence.get("data_contract") if isinstance(evidence, Mapping) else None
    provider_identities = (
        data_contract.get("qlib_providers")
        if isinstance(data_contract, Mapping)
        else None
    )
    evidence_cleaned_identity = (
        provider_identities.get("cleaned")
        if isinstance(provider_identities, Mapping)
        else None
    )
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("schema_version") != 3
        or evidence.get("evidence_contract")
        != "immutable_data_quality_evidence_v2"
        or not isinstance(evidence_cleaned_identity, Mapping)
    ):
        raise _blocked("quality_evidence_invalid")
    if not _cleaned_identities_match(evidence_cleaned_identity, cleaned_identity):
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
    direct_descriptor = _direct_descriptor_for_provider(provider)
    direct_provider = direct_descriptor is not None
    if (
        not isinstance(provider_key, str)
        or not SHA256_RE.fullmatch(provider_key)
        or not isinstance(provider_tree_sha256, str)
        or not SHA256_RE.fullmatch(provider_tree_sha256)
        or (not direct_provider and provider.name != provider_key[:16])
    ):
        raise _blocked("provider_identity_mismatch")
    if direct_provider:
        if provider_key != file_sha256(direct_descriptor) or provider_tree_sha256 != provider_key:
            raise _blocked("provider_identity_mismatch")
        actual_cleaned_identity = _direct_cleaned_identity(provider)
        if not _cleaned_identities_match(actual_cleaned_identity, cleaned_identity):
            raise _blocked("cleaned_provider_quality_mismatch")
        if not _has_csi800_membership(_read_direct_membership(provider)):
            raise _blocked("csi800_instruments_missing")
        metadata = None
    else:
        metadata = _read_json_object(provider / PROVIDER_METADATA_FILENAME)
        if (
            metadata is None
            or metadata.get("format_version") != PROVIDER_FORMAT_VERSION
            or metadata.get("provider_key") != provider_key
            or metadata.get("provider_tree_sha256") != provider_tree_sha256
            or metadata.get("data_manifest_sha256") != recorded_manifest_sha256
            or metadata.get("membership_intervals_sha256") != membership_intervals_sha256
        ):
            raise _blocked("provider_identity_mismatch")
    runtime_identity = provider_identity(provider, provider_key)
    if (
        runtime_identity is None
        or runtime_identity.get("provider_key") != provider_key
        or runtime_identity.get("provider_tree_sha256") != provider_tree_sha256
    ):
        raise _blocked("provider_identity_mismatch")
    if not direct_provider and not (provider / "instruments" / "csi800.txt").is_file():
        raise _blocked("csi800_instruments_missing")

    return {
        "contract_version": CONTRACT_VERSION,
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "status": "ready",
        "trainable": True,
        "reason_code": None,
        "summary": "The data module published an accepted immutable Robust CSI800 snapshot.",
        "summary_zh": "数据模块已发布通过验收的不可变鲁棒 CSI800 数据快照。",
        "generated_at": status.get("generated_at"),
        "provider_uri": str(provider),
        "provider_name": provider.name,
        "provider_key": provider_key,
        "provider_tree_sha256": provider_tree_sha256,
        "data_manifest_sha256": recorded_manifest_sha256,
        "membership_intervals_sha256": membership_intervals_sha256,
        "quality_report_id": report_id,
        "quality_evidence_artifact_name": evidence_artifact_name,
        "quality_evidence_sha256": evidence_sha256,
        "cleaned_provider_identity": dict(cleaned_identity),
        "canonical_snapshot_id": status.get("canonical_snapshot_id"),
        "symbols": status.get("symbols"),
        "intervals": status.get("intervals"),
        "readiness_artifact_name": artifact_path.name,
        "readiness_artifact_sha256": artifact_sha256,
    }


def consume_latest_accepted_robust_signal_data(
    *,
    pointer_path: Path,
    provider_root: Path,
    reports_dir: Path | None = None,
    provider_identity: Callable[[Path, str], Mapping[str, str] | None] = (
        runtime_provider_identity
    ),
) -> dict[str, Any]:
    """Resolve latest once, then consume the selected immutable artifact."""

    artifact_path, artifact_sha256, _payload = resolve_robust_signal_data_readiness(
        pointer_path
    )
    return consume_accepted_robust_signal_data(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        provider_root=provider_root,
        reports_dir=reports_dir,
        provider_identity=provider_identity,
    )


def public_readiness_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove local paths before readiness data is returned by an API."""

    return {
        key: payload.get(key)
        for key in (
            "contract_version",
            "acceptance_owner",
            "acceptance_contract",
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
            "quality_evidence_artifact_name",
            "quality_evidence_sha256",
            "cleaned_provider_identity",
            "canonical_snapshot_id",
            "symbols",
            "intervals",
            "readiness_artifact_name",
            "readiness_artifact_sha256",
        )
        if payload.get(key) is not None
    }
