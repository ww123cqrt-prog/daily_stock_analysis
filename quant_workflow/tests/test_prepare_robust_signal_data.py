import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from prepare_robust_signal_data import prepare_robust_signal_data
from quant_data_service_client import (
    QuantDataClientError,
    QuantDataDatasetResult,
    QuantDataIdentity,
)


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


def _service_result(dataset_id: str, *, basis: str, kind: str) -> QuantDataDatasetResult:
    identity = QuantDataIdentity(
        dataset_id=dataset_id,
        snapshot_id=f"{dataset_id}-snapshot",
        manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        latest_trade_date="2026-08-07",
        price_basis=basis,
        data_kind=kind,
    )
    return QuantDataDatasetResult(
        identity=identity,
        metadata={**identity.as_dict(), "validated": True},
        payload={
            "status": "ready",
            "metadata": {**identity.as_dict(), "validated": True},
        },
    )


def _ready_service_client() -> MagicMock:
    results = {
        "ashare_daily_qfq": _service_result(
            "ashare_daily_qfq", basis="qfq", kind="daily_bars"
        ),
        "cn_index_daily": _service_result(
            "cn_index_daily", basis="derived", kind="daily_bars"
        ),
        "pit_csi800": _service_result(
            "pit_csi800",
            basis="not_applicable",
            kind="point_in_time_membership",
        ),
        "robust_reference": _service_result(
            "robust_reference",
            basis="not_applicable",
            kind="point_in_time_reference",
        ),
        "ashare_fundamentals_pit": _service_result(
            "ashare_fundamentals_pit",
            basis="not_applicable",
            kind="point_in_time_fundamentals",
        ),
        "qlib_cleaning_plan": _service_result(
            "qlib_cleaning_plan",
            basis="not_applicable",
            kind="qlib_cleaning_plan",
        ),
    }
    client = MagicMock()
    client.resolve_dataset.side_effect = lambda dataset_id, **_kwargs: results[dataset_id]
    return client


def test_prepare_uses_service_identity_and_does_not_build_data(tmp_path: Path) -> None:
    provider = tmp_path / "formal-provider"
    provider.mkdir()
    status_path = tmp_path / "reports" / "readiness.json"
    results = {
        "ashare_daily_qfq": _service_result(
            "ashare_daily_qfq", basis="qfq", kind="daily_bars"
        ),
        "cn_index_daily": _service_result(
            "cn_index_daily", basis="derived", kind="daily_bars"
        ),
        "pit_csi800": _service_result(
            "pit_csi800",
            basis="not_applicable",
            kind="point_in_time_membership",
        ),
        "robust_reference": _service_result(
            "robust_reference",
            basis="not_applicable",
            kind="point_in_time_reference",
        ),
        "ashare_fundamentals_pit": _service_result(
            "ashare_fundamentals_pit",
            basis="not_applicable",
            kind="point_in_time_fundamentals",
        ),
        "qlib_cleaning_plan": _service_result(
            "qlib_cleaning_plan",
            basis="not_applicable",
            kind="qlib_cleaning_plan",
        ),
    }
    client = MagicMock()
    client.resolve_dataset.side_effect = lambda dataset_id, **_kwargs: results[dataset_id]

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            return_value=provider,
        ),
        patch(
            "prepare_robust_signal_data.provider_identity",
            return_value={
                **_cleaned_provider_lineage(),
                "provider_uri": str(provider),
            },
        ),
        patch("prepare_robust_signal_data.prepare_runtime_provider") as builder,
        patch("prepare_robust_signal_data.load_verified_cleaned_quality_evidence") as quality,
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=tmp_path / "empty-quant-data",
            status_path=status_path,
            client=client,
        )

    assert payload["status"] == "ready"
    assert set(payload["data_identities"]) == set(results)
    assert payload["data_identities"]["pit_csi800"]["snapshot_id"] == (
        "pit_csi800-snapshot"
    )
    assert client.resolve_dataset.call_args_list == [
        call("ashare_daily_qfq", basis="qfq"),
        call("cn_index_daily", basis="derived"),
        call("pit_csi800", basis="not_applicable"),
        call("robust_reference", basis="not_applicable"),
        call("ashare_fundamentals_pit", basis="not_applicable"),
        call("qlib_cleaning_plan", basis="not_applicable"),
    ]
    builder.assert_not_called()
    quality.assert_not_called()


def test_prepare_writes_content_addressed_readiness_for_service_snapshots(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "formal-provider"
    canonical.mkdir()
    status_path = tmp_path / "reports" / "readiness.json"
    lineage = _cleaned_provider_lineage()
    client = _ready_service_client()

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            return_value=canonical,
        ),
        patch(
            "prepare_robust_signal_data.provider_identity",
            return_value={**lineage, "provider_uri": str(canonical)},
        ),
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=tmp_path / "unused-quant-data",
            status_path=status_path,
            client=client,
        )

    assert payload["status"] == "ready"
    assert payload["trainable"] is True
    assert payload["canonical_snapshot_id"] == "snapshot-v2"
    assert len(payload["data_identity_sha256"]) == 64
    assert payload["cleaned_provider_identity"] == lineage
    assert "provider_uri" not in payload["cleaned_provider_identity"]
    assert payload["acceptance_owner"] == "data_module"
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
    assert artifact_payload["data_identities"] == payload["data_identities"]


def test_prepare_records_blocked_data_when_service_rejects_a_required_dataset(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "reports" / "readiness.json"
    client = _ready_service_client()
    client.resolve_dataset.side_effect = QuantDataClientError(
        "dataset unavailable", code="snapshot_not_ready", status_code=503
    )

    payload = prepare_robust_signal_data(
        quant_data_dir=tmp_path / "unused-quant-data",
        status_path=status_path,
        client=client,
    )

    assert payload["status"] == "blocked_data"
    assert payload["trainable"] is False
    assert payload["reason_code"] == "quant_data_service_unavailable"
    assert status_path.is_file()


def test_prepare_blocks_when_formal_qlib_provider_is_unavailable(tmp_path: Path) -> None:
    status_path = tmp_path / "reports" / "readiness.json"
    client = _ready_service_client()

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            side_effect=RuntimeError("formal Provider unavailable"),
        ),
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=tmp_path / "unused-quant-data",
            status_path=status_path,
            client=client,
        )

    assert payload["status"] == "blocked_data"
    assert payload["reason_code"] == "cleaned_provider_unavailable"


def test_prepare_does_not_consult_legacy_quality_pointer(
    tmp_path: Path,
) -> None:
    provider = tmp_path / "formal-provider"
    provider.mkdir()
    status_path = tmp_path / "reports" / "readiness.json"
    client = _ready_service_client()

    with (
        patch(
            "prepare_robust_signal_data.qlib_workflow_provider_uri",
            return_value=provider,
        ),
        patch(
            "prepare_robust_signal_data.provider_identity",
            return_value=_cleaned_provider_lineage(),
        ),
        patch("prepare_robust_signal_data.load_verified_cleaned_quality_evidence") as quality,
    ):
        payload = prepare_robust_signal_data(
            quant_data_dir=tmp_path / "unused-quant-data",
            status_path=status_path,
            quality_pointer_path=tmp_path / "obsolete-quality.json",
            client=client,
        )

    assert payload["status"] == "ready"
    quality.assert_not_called()
