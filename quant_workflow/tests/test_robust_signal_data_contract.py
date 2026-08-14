from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from prepare_robust_provider import prepare_runtime_provider
from quant_data_service_client import (
    QuantDataDatasetResult,
    QuantDataIdentity,
    QuantDataSnapshotChanged,
)
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
    public_readiness_payload,
    publish_robust_signal_data_readiness,
)


def _cleaned_provider_lineage(
    *, snapshot_id: str = "cleaned-snapshot-v2"
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


def _formal_data_identities() -> dict[str, dict[str, str]]:
    specs = {
        "ashare_daily_qfq": ("qfq", "daily_bars"),
        "cn_index_daily": ("derived", "daily_bars"),
        "pit_csi800": ("not_applicable", "point_in_time_membership"),
        "robust_reference": ("not_applicable", "point_in_time_reference"),
        "ashare_fundamentals_pit": (
            "not_applicable",
            "point_in_time_fundamentals",
        ),
        "qlib_cleaning_plan": ("not_applicable", "qlib_cleaning_plan"),
    }
    return {
        dataset_id: {
            "dataset_id": dataset_id,
            "snapshot_id": f"{dataset_id}-snapshot",
            "manifest_sha256": "a" * 64,
            "content_sha256": "b" * 64,
            "latest_trade_date": "2026-08-07",
            "price_basis": basis,
            "data_kind": data_kind,
        }
        for dataset_id, (basis, data_kind) in specs.items()
    }


def test_consumer_locks_service_snapshots_and_resolves_the_formal_provider(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports"
    identities = _formal_data_identities()
    identity_sha256 = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provider = tmp_path / "formal-provider"
    provider.mkdir()
    lineage = _cleaned_provider_lineage()
    payload = {
        "contract_version": CONTRACT_VERSION,
        "acceptance_owner": ACCEPTANCE_OWNER,
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "status": "ready",
        "trainable": True,
        "provider_uri": str(provider),
        "provider_key": lineage["descriptor_sha256"],
        "provider_tree_sha256": lineage["descriptor_sha256"],
        "data_identity_sha256": identity_sha256,
        "data_identities": identities,
        "cleaned_provider_identity": lineage,
    }
    readiness = publish_robust_signal_data_readiness(
        pointer_path=(
            reports_dir
            / "robust_signal"
            / "robust_signal_data_readiness_latest.json"
        ),
        payload=payload,
    )
    artifact = (
        reports_dir
        / "robust_signal"
        / "readiness_artifacts"
        / readiness["readiness_artifact_name"]
    )
    client = MagicMock()
    client.resolve_dataset.side_effect = lambda dataset_id, **_kwargs: (
        QuantDataDatasetResult(
            identity=QuantDataIdentity.from_mapping(identities[dataset_id]),
            metadata=identities[dataset_id],
            payload={"status": "ready"},
        )
    )
    provider_resolver = MagicMock(return_value=provider)

    consumed = consume_accepted_robust_signal_data(
        artifact_path=artifact,
        artifact_sha256=readiness["readiness_artifact_sha256"],
        reports_dir=reports_dir,
        client=client,
        provider_resolver=provider_resolver,
    )

    assert consumed["data_identities"] == identities
    assert client.resolve_dataset.call_args_list == [
        call(
            dataset_id,
            basis=identity["price_basis"],
            expected_identity=QuantDataIdentity.from_mapping(identity),
        )
        for dataset_id, identity in identities.items()
    ]
    provider_resolver.assert_called_once_with(
        "cleaned", expected_provider_identity=lineage
    )


def test_public_readiness_exposes_opaque_service_lineage_only() -> None:
    identities = _formal_data_identities()
    payload = public_readiness_payload(
        {
            "status": "ready",
            "trainable": True,
            "provider_name": "direct_parquet_v1",
            "provider_key": "a" * 64,
            "data_identity_sha256": "b" * 64,
            "data_identities": identities,
            "cleaned_provider_identity": _cleaned_provider_lineage(),
            "data_manifest_sha256": "c" * 64,
            "quality_report_id": "data-quality-" + "d" * 16,
        }
    )

    assert payload["provider_name"] == "Qlib Provider"
    assert payload["data_identity_sha256"] == "b" * 64
    assert payload["data_identities"] == identities
    serialized = json.dumps(payload, sort_keys=True)
    assert "direct_parquet_v1" not in serialized
    assert "cleaned_provider_identity" not in payload
    assert "data_manifest_sha256" not in payload
    assert "quality_report_id" not in payload


def _write_quality_evidence(
    reports_dir: Path,
    *,
    cleaned_trainable: bool = True,
    include_cleaned_quality: bool = True,
    cleaned_quality_overrides: dict[str, object] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    cleaned_identity = _cleaned_provider_lineage()
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
    cleaned_identity = _cleaned_provider_lineage()
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

    assert loaded["cleaned_provider_identity"] == _cleaned_provider_lineage()


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
        "cleaned_provider_identity": _cleaned_provider_lineage(
            snapshot_id="cleaned-v2"
        ),
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
        "cleaned_provider_identity": _cleaned_provider_lineage(
            snapshot_id="cleaned-v1"
        ),
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


def test_consumer_rejects_when_service_no_longer_serves_the_frozen_snapshot(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports"
    identities = _formal_data_identities()
    aggregate = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    provider = tmp_path / "formal-provider"
    provider.mkdir()
    lineage = _cleaned_provider_lineage()
    readiness = publish_robust_signal_data_readiness(
        pointer_path=(
            reports_dir
            / "robust_signal"
            / "robust_signal_data_readiness_latest.json"
        ),
        payload={
            "contract_version": CONTRACT_VERSION,
            "acceptance_owner": ACCEPTANCE_OWNER,
            "acceptance_contract": ACCEPTANCE_CONTRACT,
            "status": "ready",
            "trainable": True,
            "data_identity_sha256": aggregate,
            "data_identities": identities,
            "cleaned_provider_identity": lineage,
        },
    )
    artifact = (
        reports_dir
        / "robust_signal"
        / "readiness_artifacts"
        / readiness["readiness_artifact_name"]
    )
    client = MagicMock()
    client.resolve_dataset.side_effect = QuantDataSnapshotChanged(
        "snapshot changed", code="snapshot_changed", status_code=503
    )

    with pytest.raises(RobustSignalDataBlocked) as caught:
        consume_accepted_robust_signal_data(
            artifact_path=artifact,
            artifact_sha256=readiness["readiness_artifact_sha256"],
            reports_dir=reports_dir,
            client=client,
            provider_resolver=MagicMock(return_value=provider),
        )

    assert caught.value.reason_code == "quant_data_service_unavailable"


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
