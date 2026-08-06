from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from prepare_robust_provider import prepare_runtime_provider
from robust_signal_data_contract import (
    ACCEPTANCE_CONTRACT,
    ACCEPTANCE_OWNER,
    CONTRACT_VERSION,
    RobustSignalDataBlocked,
    consume_accepted_robust_signal_data,
    consume_latest_accepted_robust_signal_data,
    evaluate_robust_signal_data,
    file_sha256,
    load_verified_cleaned_quality_evidence,
    publish_robust_signal_data_readiness,
)


def _write_quality_evidence(
    reports_dir: Path,
    *,
    cleaned_trainable: bool = True,
    include_cleaned_quality: bool = True,
    cleaned_quality_overrides: dict[str, object] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    cleaned_identity = {
        "dataset_id": "qlib_ashare_cleaned_v2",
        "snapshot_id": "cleaned-snapshot-v2",
        "manifest_sha256": "c" * 64,
        "content_sha256": "d" * 64,
    }
    snapshot_state = {
        "algorithm": "relative-path-size-mtime-ctime-device-inode-v2",
        "file_count": 10,
        "total_bytes": 100,
        "scanned_entries": 12,
        "sha256": "e" * 64,
    }
    cleaned_failure_reasons = [] if cleaned_trainable else ["cleaned failed"]
    cleaned_quality: dict[str, object] = {
        **cleaned_identity,
        "provider": "cleaned",
        "trainable": cleaned_trainable,
        "failure_reasons": cleaned_failure_reasons,
        "training_readiness": {
            "status": "trainable" if cleaned_trainable else "limited",
            "failure_reasons": cleaned_failure_reasons,
        },
    }
    if cleaned_quality_overrides:
        cleaned_quality.update(cleaned_quality_overrides)
    provider_quality = {
        "canonical": {
            "trainable": True,
            "failure_reasons": [],
            "training_readiness": {
                "status": "trainable",
                "failure_reasons": [],
            },
        },
    }
    if include_cleaned_quality:
        provider_quality["cleaned"] = cleaned_quality

    evidence = {
        "schema_version": 3,
        "evidence_contract": "immutable_data_quality_evidence_v2",
        "trainable": True,
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
            "qlib_providers": {"cleaned": cleaned_identity},
            "snapshot_state": {
                "stable": True,
                "content_revalidated": True,
                "before": {"qlib_ashare_cleaned_v2": snapshot_state},
                "after_scan": {"qlib_ashare_cleaned_v2": snapshot_state},
                "after_revalidation": {
                    "qlib_ashare_cleaned_v2": snapshot_state
                },
                "after": {"qlib_ashare_cleaned_v2": snapshot_state},
            },
        },
        "provider_quality": provider_quality,
    }
    evidence_bytes = (
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_name = (
        f"data_quality_report_20260805_120000_{evidence_sha256[:16]}.json"
    )
    (reports_dir / evidence_name).write_bytes(evidence_bytes)
    pointer_path = reports_dir / "data_quality_report_latest.json"
    pointer_path.write_text(
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
    return pointer_path


def _write_accepted_consumer_fixture(
    tmp_path: Path,
    *,
    reports_dir: Path,
    readiness_root: Path | None = None,
) -> tuple[Path, str, Path, dict[str, str], Path]:
    quant_data = tmp_path / "quant_data"
    canonical = quant_data / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    intervals = quant_data / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 2}\n', encoding="utf-8")
    provider_root = quant_data / "runtime" / "robust_strategy" / "providers"
    prepared = prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=provider_root.parent,
        data_manifest_path=manifest,
    )

    quality_pointer = _write_quality_evidence(reports_dir)
    quality_publication = json.loads(quality_pointer.read_text(encoding="utf-8"))
    cleaned_identity = {
        "dataset_id": "qlib_ashare_cleaned_v2",
        "snapshot_id": "cleaned-snapshot-v2",
        "manifest_sha256": "c" * 64,
        "content_sha256": "d" * 64,
    }
    readiness_payload = {
        "contract_version": CONTRACT_VERSION,
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "status": "ready",
        "trainable": True,
        "provider_uri": prepared["provider_uri"],
        "provider_key": prepared["provider_key"],
        "provider_tree_sha256": prepared["provider_tree_sha256"],
        "data_manifest_sha256": file_sha256(manifest),
        "membership_intervals_sha256": file_sha256(intervals),
        "quality_report_id": quality_publication["report_id"],
        "quality_evidence_artifact_name": quality_publication["evidence_artifact"],
        "quality_evidence_sha256": quality_publication["evidence_sha256"],
        "cleaned_provider_identity": cleaned_identity,
    }
    readiness_bytes = (
        json.dumps(
            readiness_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    artifact_root = readiness_root or (
        reports_dir / "robust_signal" / "readiness_artifacts"
    )
    artifact_path = artifact_root / (
        f"robust_signal_data_readiness_{readiness_sha256}.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(readiness_bytes)
    return (
        artifact_path,
        readiness_sha256,
        provider_root,
        cleaned_identity,
        reports_dir / quality_publication["evidence_artifact"],
    )


def test_quality_evidence_rejects_failed_cleaned_provider_when_canonical_passes(
    tmp_path: Path,
) -> None:
    pointer_path = _write_quality_evidence(
        tmp_path / "quality",
        cleaned_trainable=False,
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        load_verified_cleaned_quality_evidence(pointer_path)

    assert caught.value.reason_code == "quality_evidence_not_trainable"


def test_quality_evidence_accepts_trainable_cleaned_provider(
    tmp_path: Path,
) -> None:
    pointer_path = _write_quality_evidence(tmp_path / "quality")

    loaded = load_verified_cleaned_quality_evidence(pointer_path)

    assert loaded["cleaned_provider_identity"] == {
        "dataset_id": "qlib_ashare_cleaned_v2",
        "snapshot_id": "cleaned-snapshot-v2",
        "manifest_sha256": "c" * 64,
        "content_sha256": "d" * 64,
    }


def test_quality_evidence_requires_cleaned_provider_quality(
    tmp_path: Path,
) -> None:
    pointer_path = _write_quality_evidence(
        tmp_path / "quality",
        include_cleaned_quality=False,
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        load_verified_cleaned_quality_evidence(pointer_path)

    assert caught.value.reason_code == "quality_evidence_invalid"


@pytest.mark.parametrize(
    "cleaned_quality_overrides",
    [
        {"training_readiness": None},
        {
            "training_readiness": {
                "status": "limited",
                "failure_reasons": [],
            }
        },
        {"failure_reasons": ["cleaned blocker"]},
        {
            "training_readiness": {
                "status": "trainable",
                "failure_reasons": ["cleaned blocker"],
            }
        },
    ],
    ids=[
        "missing-readiness",
        "non-trainable-readiness",
        "provider-blocker",
        "readiness-blocker",
    ],
)
def test_quality_evidence_rejects_invalid_cleaned_readiness_or_blockers(
    tmp_path: Path,
    cleaned_quality_overrides: dict[str, object],
) -> None:
    pointer_path = _write_quality_evidence(
        tmp_path / "quality",
        cleaned_quality_overrides=cleaned_quality_overrides,
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        load_verified_cleaned_quality_evidence(pointer_path)

    assert caught.value.reason_code == "quality_evidence_not_trainable"


def test_quality_evidence_rejects_cleaned_quality_identity_mismatch(
    tmp_path: Path,
) -> None:
    pointer_path = _write_quality_evidence(
        tmp_path / "quality",
        cleaned_quality_overrides={"snapshot_id": "other-cleaned-snapshot"},
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        load_verified_cleaned_quality_evidence(pointer_path)

    assert caught.value.reason_code == "cleaned_provider_quality_mismatch"


def test_readiness_rejects_provider_content_changed_after_attestation(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    canonical = quant_data / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    intervals = quant_data / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 2}\n', encoding="utf-8")
    provider_root = quant_data / "runtime" / "robust_strategy" / "providers"
    prepared = prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=provider_root.parent,
        data_manifest_path=manifest,
    )
    provider = Path(prepared["provider_uri"])
    manifest_hash = file_sha256(manifest)
    quality = {
        "quality_report_id": "data-quality-" + "a" * 16,
        "quality_evidence_sha256": "a" * 64,
        "cleaned_provider_identity": {
            "dataset_id": "qlib_ashare_cleaned_v2",
            "snapshot_id": "cleaned-v2",
            "manifest_sha256": "b" * 64,
            "content_sha256": "c" * 64,
        },
    }
    status_path = tmp_path / "readiness.json"
    status_path.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "status": "ready",
                "trainable": True,
                "provider_uri": str(provider),
                "provider_key": prepared["provider_key"],
                "provider_tree_sha256": prepared["provider_tree_sha256"],
                "data_manifest_sha256": manifest_hash,
                **quality,
            }
        ),
        encoding="utf-8",
    )

    feature = provider / "features" / "sample.bin"
    feature.chmod(0o644)
    feature.write_bytes(b"tampered")
    feature.chmod(0o444)

    with pytest.raises(RobustSignalDataBlocked) as caught:
        evaluate_robust_signal_data(
            status_path=status_path,
            data_manifest_path=manifest,
            provider_root=provider_root,
            quality_pointer_path=tmp_path / "data_quality_report_latest.json",
            quality_evidence_loader=lambda _path: quality,
        )

    assert caught.value.reason_code == "provider_identity_mismatch"


def test_readiness_rejects_when_latest_quality_evidence_no_longer_matches(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "DATA_MANIFEST.json"
    manifest.write_text("{}\n", encoding="utf-8")
    provider_root = tmp_path / "providers"
    provider_key = "d" * 64
    provider = provider_root / provider_key[:16]
    (provider / "instruments").mkdir(parents=True)
    (provider / "instruments" / "csi800.txt").write_text(
        "SH600000\t2020-01-02\t2025-12-31\n",
        encoding="utf-8",
    )
    metadata = {
        "format_version": 3,
        "provider_key": provider_key,
        "provider_tree_sha256": "e" * 64,
        "data_manifest_sha256": file_sha256(manifest),
    }
    (provider / "ROBUST_PROVIDER.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    status = {
        "contract_version": CONTRACT_VERSION,
        "status": "ready",
        "trainable": True,
        "provider_uri": str(provider),
        "provider_key": provider_key,
        "provider_tree_sha256": "e" * 64,
        "data_manifest_sha256": file_sha256(manifest),
        "quality_report_id": "data-quality-" + "1" * 16,
        "quality_evidence_sha256": "1" * 64,
        "cleaned_provider_identity": {
            "dataset_id": "qlib_ashare_cleaned_v2",
            "snapshot_id": "cleaned-v1",
            "manifest_sha256": "2" * 64,
            "content_sha256": "3" * 64,
        },
    }
    status_path = tmp_path / "readiness.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    current_quality = {
        **status,
        "quality_report_id": "data-quality-" + "4" * 16,
        "quality_evidence_sha256": "4" * 64,
    }

    with pytest.raises(RobustSignalDataBlocked) as caught:
        evaluate_robust_signal_data(
            status_path=status_path,
            data_manifest_path=manifest,
            provider_root=provider_root,
            quality_pointer_path=tmp_path / "data_quality_report_latest.json",
            quality_evidence_loader=lambda _path: current_quality,
            provider_identity=lambda _provider, _key: {
                "provider_key": provider_key,
                "provider_tree_sha256": "e" * 64,
            },
        )

    assert caught.value.reason_code == "quality_evidence_changed"


def test_consumer_uses_frozen_publication_without_reading_latest_or_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("QUANT_WORKFLOW_REPORTS_ROOT", str(reports_dir))
    artifact, artifact_sha256, provider_root, _identity, _evidence = (
        _write_accepted_consumer_fixture(tmp_path, reports_dir=reports_dir)
    )
    manifest = tmp_path / "quant_data" / "DATA_MANIFEST.json"
    manifest_hash = file_sha256(manifest)
    manifest.unlink()
    (reports_dir / "data_quality_report_latest.json").unlink()

    consumed = consume_accepted_robust_signal_data(
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
        provider_root=provider_root,
    )

    assert consumed["provider_uri"] == str(Path(consumed["provider_uri"]).resolve())
    assert consumed["data_manifest_sha256"] == manifest_hash
    assert consumed["quality_evidence_artifact_name"].startswith(
        "data_quality_report_"
    )
    assert consumed["readiness_artifact_sha256"] == artifact_sha256


def test_consumer_rejects_readiness_artifact_outside_controlled_reports_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("QUANT_WORKFLOW_REPORTS_ROOT", str(reports_dir))
    artifact, artifact_sha256, provider_root, _identity, _evidence = (
        _write_accepted_consumer_fixture(
            tmp_path,
            reports_dir=reports_dir,
            readiness_root=tmp_path / "attacker" / "readiness_artifacts",
        )
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        consume_accepted_robust_signal_data(
            artifact_path=artifact,
            artifact_sha256=artifact_sha256,
            provider_root=provider_root,
        )

    assert caught.value.reason_code == "robust_readiness_invalid"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing", "quality_evidence_missing"),
        ("changed", "quality_evidence_changed"),
        ("identity", "cleaned_provider_quality_mismatch"),
    ],
)
def test_consumer_rejects_invalid_frozen_quality_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_reason: str,
) -> None:
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("QUANT_WORKFLOW_REPORTS_ROOT", str(reports_dir))
    artifact, artifact_sha256, provider_root, _identity, evidence = (
        _write_accepted_consumer_fixture(tmp_path, reports_dir=reports_dir)
    )
    if mutation == "missing":
        evidence.unlink()
    elif mutation == "changed":
        evidence.write_text("{}\n", encoding="utf-8")
    else:
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        payload["data_contract"]["qlib_providers"]["cleaned"]["snapshot_id"] = (
            "different-cleaned-snapshot"
        )
        mutated_bytes = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        evidence.write_bytes(mutated_bytes)
        readiness = json.loads(artifact.read_text(encoding="utf-8"))
        readiness["quality_evidence_sha256"] = hashlib.sha256(
            mutated_bytes
        ).hexdigest()
        readiness["quality_report_id"] = (
            "data-quality-" + readiness["quality_evidence_sha256"][:16]
        )
        renamed_evidence = evidence.with_name(
            "data_quality_report_20260805_120000_"
            + readiness["quality_evidence_sha256"][:16]
            + ".json"
        )
        evidence.rename(renamed_evidence)
        readiness["quality_evidence_artifact_name"] = renamed_evidence.name
        readiness_bytes = (
            json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        new_artifact_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
        new_artifact = artifact.with_name(
            f"robust_signal_data_readiness_{new_artifact_sha256}.json"
        )
        new_artifact.write_bytes(readiness_bytes)
        artifact = new_artifact
        artifact_sha256 = new_artifact_sha256

    with pytest.raises(RobustSignalDataBlocked) as caught:
        consume_accepted_robust_signal_data(
            artifact_path=artifact,
            artifact_sha256=artifact_sha256,
            provider_root=provider_root,
        )

    assert caught.value.reason_code == expected_reason


def test_latest_consumer_rejects_direct_manually_relabelled_readiness(
    tmp_path: Path,
) -> None:
    pointer_path = tmp_path / "robust_signal_data_readiness_latest.json"
    pointer_path.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "acceptance_owner": ACCEPTANCE_OWNER,
                "acceptance_contract": ACCEPTANCE_CONTRACT,
                "status": "ready",
                "trainable": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        consume_latest_accepted_robust_signal_data(
            pointer_path=pointer_path,
            provider_root=tmp_path / "providers",
        )

    assert caught.value.reason_code == "robust_readiness_invalid"


def test_publisher_does_not_advance_latest_when_artifact_path_is_a_symlink(
    tmp_path: Path,
) -> None:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "status": "ready",
        "trainable": True,
    }
    payload_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    pointer_path = tmp_path / "robust_signal_data_readiness_latest.json"
    artifact_path = (
        tmp_path
        / "readiness_artifacts"
        / f"robust_signal_data_readiness_{artifact_sha256}.json"
    )
    artifact_path.parent.mkdir()
    attacker_file = tmp_path / "attacker.json"
    attacker_file.write_text("{}\n", encoding="utf-8")
    artifact_path.symlink_to(attacker_file)

    with pytest.raises(RuntimeError, match="content-addressed readiness artifact"):
        publish_robust_signal_data_readiness(
            pointer_path=pointer_path,
            payload=payload,
        )

    assert not pointer_path.exists()
