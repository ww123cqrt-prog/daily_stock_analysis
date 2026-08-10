import json
import hashlib
from pathlib import Path
from unittest.mock import patch

from prepare_robust_signal_data import prepare_robust_signal_data


def _cleaned_provider_lineage(
    *, snapshot_id: str = "snapshot-v2"
) -> dict[str, str]:
    return {
        "provider_contract": "direct_parquet_v1",
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": snapshot_id,
        "manifest_sha256": "c" * 64,
        "content_sha256": "d" * 64,
        "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
        "descriptor_sha256": "e" * 64,
        "cleaning_mode": "cleaned",
        "index_dataset_id": "cn_index_daily",
        "index_snapshot_id": "index-snapshot-v2",
        "index_content_sha256": "f" * 64,
        "index_manifest_sha256": "a" * 64,
    }


def _write_quality_evidence(
    reports_dir: Path,
    *,
    trainable: bool = True,
    snapshot_id: str = "snapshot-v2",
) -> tuple[Path, dict[str, str]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    identity = _cleaned_provider_lineage(snapshot_id=snapshot_id)
    state = {
        "algorithm": "relative-path-size-mtime-ctime-device-inode-v2",
        "file_count": 10,
        "total_bytes": 100,
        "scanned_entries": 12,
        "sha256": "e" * 64,
    }
    failure_reasons = [] if trainable else ["cleaned provider failed"]
    evidence = {
        "schema_version": 3,
        "evidence_contract": "immutable_data_quality_evidence_v2",
        "trainable": trainable,
        "scan_mode": "full",
        "coverage_status": "full",
        "verification_status": "verified",
        "requested_sample": 0,
        "scanned_instruments": 100,
        "training_thresholds": {
            "authoritative": True,
            "min_calendar_days": 252,
            "min_instruments": 100,
            "min_merged_instruments": 100,
        },
        "data_contract": {
            "status": "verified",
            "qlib_providers": {"cleaned": identity},
            "snapshot_state": {
                "stable": True,
                "content_revalidated": True,
                "before": {"qlib_ashare_cleaned_v2": state},
                "after_scan": {"qlib_ashare_cleaned_v2": state},
                "after_revalidation": {"qlib_ashare_cleaned_v2": state},
                "after": {"qlib_ashare_cleaned_v2": state},
            },
        },
        "provider_quality": {
            "cleaned": {
                **identity,
                "provider": "cleaned",
                "trainable": trainable,
                "failure_reasons": failure_reasons,
                "training_readiness": {
                    "status": "trainable" if trainable else "limited",
                    "failure_reasons": failure_reasons,
                },
            }
        },
    }
    evidence_bytes = (
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_name = f"data_quality_report_20260803_120000_{evidence_sha256[:16]}.json"
    (reports_dir / evidence_name).write_bytes(evidence_bytes)
    pointer = reports_dir / "data_quality_report_latest.json"
    pointer.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "publication_contract": (
                    "atomic_json_pointer_to_immutable_quality_evidence_v2"
                ),
                "report_id": f"data-quality-{evidence_sha256[:16]}",
                "evidence_artifact": evidence_name,
                "evidence_sha256": evidence_sha256,
            }
        ),
        encoding="utf-8",
    )
    return pointer, identity


def test_prepare_writes_dedicated_readiness_bound_to_current_manifest(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    canonical = quant_data / "qlib" / "cleaned" / "snapshots" / "snapshot-v2"
    canonical.mkdir(parents=True)
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 2}\n', encoding="utf-8")
    intervals = (
        quant_data / "reference" / "pit_csi800" / "membership_intervals.parquet"
    )
    intervals.parent.mkdir(parents=True)
    intervals.write_bytes(b"membership")
    status_path = tmp_path / "reports" / "readiness.json"
    quality_pointer, quality_identity = _write_quality_evidence(
        tmp_path / "reports" / "quality"
    )
    result = {
        "provider_uri": str(
            quant_data / "runtime" / "robust_strategy" / "providers" / ("a" * 16)
        ),
        "provider_key": "a" * 64,
        "provider_tree_sha256": "b" * 64,
        "symbols": 1633,
        "intervals": 2058,
        "reused": False,
    }

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            return_value=canonical,
        ),
        patch(
            "prepare_robust_signal_data.provider_identity",
            return_value={
                **quality_identity,
                "provider_uri": str(canonical),
            },
        ),
        patch(
            "prepare_robust_signal_data.prepare_runtime_provider",
            return_value=result,
        ),
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=quant_data,
            status_path=status_path,
            quality_pointer_path=quality_pointer,
        )

    assert payload["status"] == "ready"
    assert payload["trainable"] is True
    assert payload["canonical_snapshot_id"] == "snapshot-v2"
    assert payload["quality_report_id"].startswith("data-quality-")
    assert payload["quality_evidence_artifact_name"].startswith(
        "data_quality_report_"
    )
    assert len(payload["quality_evidence_sha256"]) == 64
    assert payload["cleaned_provider_identity"] == quality_identity
    assert "provider_uri" not in payload["cleaned_provider_identity"]
    assert payload["acceptance_owner"] == "data_module"
    assert (
        payload["acceptance_contract"]
        == "data_module_accepted_content_addressed_snapshot_v2"
    )
    assert payload["data_manifest_sha256"] != payload["membership_intervals_sha256"]
    assert status_path.is_file()
    pointer = json.loads(status_path.read_text(encoding="utf-8"))
    assert pointer["pointer_contract"] == "robust_signal_data_readiness_pointer_v1"
    assert pointer["artifact_sha256"] == payload["readiness_artifact_sha256"]
    artifact_path = status_path.parent / pointer["artifact"]
    assert artifact_path.name == (
        f"robust_signal_data_readiness_{pointer['artifact_sha256']}.json"
    )
    assert (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        == pointer["artifact_sha256"]
    )
    artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "provider_uri" not in artifact_payload["cleaned_provider_identity"]
    assert artifact_payload["provider_key"] == payload["provider_key"]
    assert (
        artifact_payload["quality_evidence_artifact_name"]
        == payload["quality_evidence_artifact_name"]
    )
    assert (
        artifact_payload["quality_evidence_sha256"]
        == payload["quality_evidence_sha256"]
    )


def test_prepare_records_blocked_data_when_pit_membership_is_missing(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    status_path = tmp_path / "reports" / "readiness.json"

    payload = prepare_robust_signal_data(
        quant_data_dir=quant_data,
        status_path=status_path,
    )

    assert payload["status"] == "blocked_data"
    assert payload["trainable"] is False
    assert payload["reason_code"] == "membership_intervals_missing"
    assert status_path.is_file()


def test_prepare_records_sanitized_bounded_provider_build_error(tmp_path: Path) -> None:
    quant_data = tmp_path / "quant_data"
    canonical = quant_data / "qlib" / "cleaned" / "snapshots" / "snapshot-v2"
    canonical.mkdir(parents=True)
    (quant_data / "DATA_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    intervals = (
        quant_data / "reference" / "pit_csi800" / "membership_intervals.parquet"
    )
    intervals.parent.mkdir(parents=True)
    intervals.write_bytes(b"membership")
    status_path = tmp_path / "reports" / "readiness.json"
    quality_pointer, quality_identity = _write_quality_evidence(
        tmp_path / "reports" / "quality"
    )
    private_path = tmp_path / "private" / "provider.bin"

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            return_value=canonical,
        ),
        patch(
            "prepare_robust_signal_data.provider_identity",
            return_value=quality_identity,
        ),
        patch(
            "prepare_robust_signal_data.prepare_runtime_provider",
            side_effect=RuntimeError(f"provider failed at {private_path}" + "x" * 1000),
        ),
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=quant_data,
            status_path=status_path,
            quality_pointer_path=quality_pointer,
        )

    assert payload.get("error_type") == "RuntimeError"
    assert isinstance(payload.get("error"), str)
    assert len(payload["error"]) <= 500
    assert str(tmp_path) not in payload["error"]
    pointer = json.loads(status_path.read_text(encoding="utf-8"))
    artifact_payload = json.loads(
        (status_path.parent / pointer["artifact"]).read_text(encoding="utf-8")
    )
    assert artifact_payload["error_type"] == payload["error_type"]
    assert artifact_payload["error"] == payload["error"]


def test_prepare_refuses_full_verified_quality_evidence_when_not_trainable(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    intervals = (
        quant_data / "reference" / "pit_csi800" / "membership_intervals.parquet"
    )
    intervals.parent.mkdir(parents=True)
    intervals.write_bytes(b"membership")
    quality_pointer, _identity = _write_quality_evidence(
        tmp_path / "reports" / "quality",
        trainable=False,
    )
    status_path = tmp_path / "reports" / "readiness.json"

    with patch("prepare_robust_signal_data.prepare_runtime_provider") as build_provider:
        payload = prepare_robust_signal_data(
            quant_data_dir=quant_data,
            status_path=status_path,
            quality_pointer_path=quality_pointer,
        )

    assert payload["status"] == "blocked_data"
    assert payload["trainable"] is False
    assert payload["reason_code"] == "quality_evidence_not_trainable"
    build_provider.assert_not_called()
