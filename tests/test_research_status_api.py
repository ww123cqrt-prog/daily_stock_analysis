# -*- coding: utf-8 -*-
"""Tests for the research workspace status endpoint."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.app import create_app
from quant_workflow.scripts.quant_data_contract import (
    atomic_write_json,
    filesystem_state_fingerprint,
    finalize_qlib_snapshot,
    resolve_dataset,
)
from quant_workflow.scripts.quant_data_service_client import (
    QuantDataClientError,
    QuantDataIdentity,
    QuantDataReadyResult,
)
from quant_workflow.scripts.robust_signal_data_contract import (
    ACCEPTANCE_CONTRACT,
    ACCEPTANCE_OWNER,
    CONTRACT_VERSION,
    publish_robust_signal_data_readiness,
)


def write_quality_evidence(
    reports_dir: Path,
    evidence_fields: dict,
    *,
    markdown: str = "# Quality\n\n- Stock universe: 5509\n",
    timestamp: str = "20260731_110000",
) -> tuple[str, str]:
    markdown_bytes = markdown.encode("utf-8")
    markdown_sha256 = hashlib.sha256(markdown_bytes).hexdigest()
    markdown_artifact = reports_dir / (
        f"data_quality_report_{timestamp}_{markdown_sha256[:16]}.md"
    )
    markdown_artifact.write_bytes(markdown_bytes)
    evidence = {
        "schema_version": 3,
        "evidence_contract": "immutable_data_quality_evidence_v2",
        "trainable": True,
        "training_thresholds": {
            "min_calendar_days": 252,
            "min_instruments": 100,
            "min_merged_instruments": 100,
            "authoritative_min_calendar_days": 252,
            "authoritative_min_instruments": 100,
            "authoritative": True,
        },
        **evidence_fields,
        "markdown_artifact": markdown_artifact.name,
        "markdown_sha256": markdown_sha256,
    }
    evidence_bytes = (
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    evidence_artifact = reports_dir / (
        f"data_quality_report_{timestamp}_{evidence_sha256[:16]}.json"
    )
    evidence_artifact.write_bytes(evidence_bytes)
    report_id = f"data-quality-{evidence_sha256[:16]}"
    (reports_dir / "data_quality_report_latest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "publication_contract": (
                    "atomic_json_pointer_to_immutable_quality_evidence_v2"
                ),
                "report_id": report_id,
                "evidence_artifact": evidence_artifact.name,
                "evidence_sha256": evidence_sha256,
            }
        ),
        encoding="utf-8",
    )
    return report_id, evidence_sha256


def stable_state_fingerprint(marker: str = "0") -> dict:
    return {
        "algorithm": "relative-path-size-mtime-ctime-device-inode-v2",
        "file_count": 1,
        "total_bytes": 8,
        "scanned_entries": 1,
        "sha256": marker * 64,
    }


def valid_provider_semantic_metrics() -> dict:
    return {
        "calendar_valid": True,
        "invalid_calendar_entries": 0,
        "instrument_records_valid": True,
        "invalid_instrument_records": 0,
        "required_feature_min_coverage": 0.50,
        "minimum_valid_instrument_feature_coverage": 0.75,
        "misaligned_instrument_feature_dirs": 0,
    }


def stable_snapshot_state(dataset_id: str, state: dict) -> dict:
    return {
        "stable": True,
        "content_revalidated": True,
        "before": {dataset_id: state},
        "after_scan": {dataset_id: state},
        "after_revalidation": {dataset_id: state},
        "after": {dataset_id: state},
    }


def ready_training_data() -> QuantDataReadyResult:
    identities = tuple(
        QuantDataIdentity(
            dataset_id=dataset_id,
            snapshot_id=f"{dataset_id}-snapshot",
            manifest_sha256=marker * 64,
            content_sha256=marker * 64,
            latest_trade_date="2026-08-07",
            price_basis=price_basis,
            data_kind=data_kind,
        )
        for dataset_id, marker, price_basis, data_kind in (
            ("ashare_daily_qfq", "a", "qfq", "daily_bars"),
            ("cn_index_daily", "b", "raw", "daily_bars"),
            (
                "pit_csi800",
                "c",
                "not_applicable",
                "point_in_time_membership",
            ),
            (
                "qlib_cleaning_plan",
                "d",
                "not_applicable",
                "qlib_cleaning_plan",
            ),
        )
    )
    return QuantDataReadyResult(
        status="ready",
        validated_datasets=len(identities),
        identities=identities,
        datasets=tuple(
            {**identity.as_dict(), "role": "canonical", "status": "ready"}
            for identity in identities
        ),
    )



def local_training_catalog(readiness, lineage=None):
    """Metadata fixture for paired stock/index identities in one local root."""
    lineage = lineage or ready_provider_resolution().lineage
    items = []
    for identity in readiness.identities:
        item = {**identity.as_dict(), "role": "canonical", "status": "ready"}
        for prefix in ("", "index_"):
            if item["dataset_id"] == lineage.get(f"{prefix}dataset_id"):
                item.update({key: lineage[f"{prefix}{key}"]
                             for key in ("snapshot_id", "manifest_sha256", "content_sha256")})
        items.append(item)
    return {"datasets": items}


def ready_provider_resolution(provider: str = "canonical"):
    from api.v1.endpoints.research_status import _NativeQlibProviderResolution

    provider_name = "cleaned" if provider == "cleaned" else "canonical"
    marker = "e"
    lineage = {
        "provider_contract": "direct_parquet_v1",
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "ashare-daily-qfq-snapshot",
        "manifest_sha256": marker * 64,
        "content_sha256": marker * 64,
        "descriptor_path": (
            f"runtime/qlib/provider/{provider_name}/PROVIDER_DESCRIPTOR.json"
        ),
        "descriptor_sha256": ("1" if provider_name == "canonical" else "2")
        * 64,
        "cleaning_mode": provider_name,
        "index_dataset_id": "cn_index_daily",
        "index_snapshot_id": "cn-index-snapshot",
        "index_content_sha256": "3" * 64,
        "index_manifest_sha256": "4" * 64,
    }
    return _NativeQlibProviderResolution(
        authoritative=True,
        path=Path(f"/tmp/{provider_name}-provider"),
        lineage=lineage,
        provider_name=provider_name,
        dataset_id=lineage["dataset_id"],
        snapshot_id=lineage["snapshot_id"],
        manifest_sha256=lineage["manifest_sha256"],
        content_sha256=lineage["content_sha256"],
        calendar_end="2026-08-07",
        instrument_count=5516,
        benchmark_complete=True,
    )


def resolve_ready_provider(
    dataset_id: str = "ashare_daily_qfq",
    *,
    provider: str = "canonical",
):
    provider_name = (
        "cleaned"
        if provider == "cleaned" or str(dataset_id).endswith("_cleaned_v2")
        else "canonical"
    )
    return ready_provider_resolution(provider_name)


def write_native_ashare_fixture(
    data_root: Path,
    *,
    snapshot_id: str = "ashare-qfq-snapshot",
) -> tuple[Path, dict[str, str]]:
    dataset_id = "ashare_daily_qfq"
    dataset_root = Path("parquet/ashare_1d/ashare_daily_qfq")
    snapshot = data_root / dataset_root / "snapshots" / snapshot_id
    bars = snapshot / "bars"
    bars.mkdir(parents=True)
    (bars / "600000.parquet").write_bytes(b"parquet-fixture")
    manifest = {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "data_kind": "daily_bars",
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "content_sha256": "a" * 64,
    }
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    identity = {
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_sha256,
        "content_sha256": manifest["content_sha256"],
    }
    data_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        data_root / "DATA_MANIFEST.json",
        {
            "schema_version": 2,
            "path_format": "relative_to_quant_data_root",
            "datasets": {
                dataset_id: {
                    **identity,
                    "dataset_root": dataset_root.as_posix(),
                    "role": "canonical",
                    "status": "ready",
                    "price_basis": "qfq",
                    "snapshot_manifest_sha256": manifest_sha256,
                    "latest_pointer": (dataset_root / "LATEST.json").as_posix(),
                }
            },
        },
    )
    atomic_write_json(
        data_root / dataset_root / "LATEST.json",
        {
            "schema_version": 2,
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "manifest_sha256": manifest_sha256,
            "snapshot_path": (
                dataset_root / "snapshots" / snapshot_id
            ).as_posix(),
        },
    )
    return snapshot, identity


class ResearchStatusApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from api.v1.endpoints import research_status

        def resolve_test_provider(provider: str = "canonical") -> Path:
            dataset_id = (
                "qlib_ashare_cleaned_v2"
                if str(provider).strip().lower() == "cleaned"
                else "qlib_ashare_v2"
            )
            resolution = resolve_dataset(
                dataset_id,
                quant_data_root=research_status.QUANT_DATA_ROOT,
                require_basis="qfq",
            )
            return Path(resolution.snapshot_path)

        self._provider_facade_patch = patch.object(
            research_status,
            "qlib_workflow_provider_uri",
            side_effect=resolve_test_provider,
        )
        self._provider_facade_patch.start()
        self.addCleanup(self._provider_facade_patch.stop)

    def test_qlib_status_resolution_uses_provider_facade(self) -> None:
        from api.v1.endpoints import research_status

        provider = Path("/quant_data/runtime/qlib/provider")
        lineage = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "qfq-001",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "c" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-001",
            "index_content_sha256": "d" * 64,
            "index_manifest_sha256": "e" * 64,
        }
        ranges = {
            "SH600000": ("2020-01-02", "2026-08-07"),
            "SZ000001": ("2020-01-02", "2026-08-07"),
        }

        with (
            patch.object(
                research_status,
                "qlib_workflow_provider_uri",
                return_value=provider,
                create=True,
            ) as provider_uri,
            patch.object(
                research_status,
                "provider_calendar_days",
                return_value=["2026-08-06", "2026-08-07"],
                create=True,
            ) as calendar_days,
            patch.object(
                research_status,
                "provider_instrument_ranges",
                return_value=ranges,
                create=True,
            ) as instrument_ranges,
            patch.object(
                research_status,
                "provider_has_instrument_pool",
                return_value=True,
                create=True,
            ) as has_pool,
            patch.object(
                research_status,
                "provider_identity",
                return_value=lineage,
                create=True,
            ) as identity_reader,
        ):
            resolution = research_status._resolve_native_qlib_provider(
                provider="canonical"
            )

        self.assertEqual(resolution.path, provider)
        self.assertEqual(resolution.lineage, lineage)
        self.assertEqual(resolution.calendar_end, "2026-08-07")
        self.assertEqual(resolution.instrument_count, 2)
        self.assertTrue(resolution.benchmark_complete)
        provider_uri.assert_called_once_with("canonical")
        calendar_days.assert_called_once_with(provider)
        instrument_ranges.assert_called_once_with(provider)
        has_pool.assert_called_once_with(provider, "csi300")
        identity_reader.assert_called_once_with(
            provider,
            quant_data_root=research_status.QUANT_DATA_ROOT,
        )

    def test_saved_provider_lineage_is_compared_without_storage_branching(self) -> None:
        from api.v1.endpoints import research_status

        lineage = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "qfq-001",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "c" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-001",
            "index_content_sha256": "d" * 64,
            "index_manifest_sha256": "e" * 64,
        }
        resolution = research_status._NativeQlibProviderResolution(
            authoritative=True,
            path=Path("/quant_data/runtime/qlib/provider"),
            lineage=lineage,
            provider_name="canonical",
        )
        payload = {"provider": "canonical", "provider_identity": dict(lineage)}

        research_status._annotate_saved_provider(
            payload,
            resolution,
            artifact_label="训练结果",
        )

        self.assertTrue(payload["provider_current"])
        self.assertEqual(payload["provider_status"], "current")
        self.assertNotIn("provider_warning", payload)

    def test_public_qlib_provider_status_hides_storage_implementation_terms(self) -> None:
        from api.v1.endpoints import research_status

        lineage = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "qfq-001",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "c" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-001",
            "index_content_sha256": "d" * 64,
            "index_manifest_sha256": "e" * 64,
        }
        resolution = research_status._NativeQlibProviderResolution(
            authoritative=True,
            path=Path("/quant_data/runtime/qlib/provider"),
            lineage=lineage,
            provider_name="canonical",
            calendar_end="2026-08-07",
            instrument_count=5000,
            benchmark_complete=True,
        )

        with (
            patch.object(
                research_status,
                "_resolve_native_qlib_provider",
                return_value=resolution,
            ),
            patch.object(
                research_status,
                "_qlib_provider_from_path",
                return_value={
                    "path": "/quant_data/runtime/qlib/provider",
                    "calendar_days": 1000,
                    "instruments": 5000,
                    "benchmark_complete": True,
                },
            ),
        ):
            payload = research_status._missing_status_payload()

        public_text = json.dumps(
            {
                "qlib_provider": payload["qlib_provider"],
                "training_readiness": payload["training_readiness"],
            },
            ensure_ascii=False,
        ).lower()
        for forbidden in ("direct", "直读", "native v2", "原生 v2"):
            self.assertNotIn(forbidden, public_text)

    def test_blocked_qlib_provider_hides_storage_implementation_error(self) -> None:
        from api.v1.endpoints import research_status

        with patch.object(
            research_status,
            "qlib_workflow_provider_uri",
            side_effect=RuntimeError(
                "direct_parquet_v1 descriptor rejected a legacy .bin provider"
            ),
        ):
            resolution = research_status._resolve_native_qlib_provider()

        self.assertTrue(resolution.authoritative)
        self.assertIsNone(resolution.path)
        self.assertIn("Qlib Provider", str(resolution.error))
        public_error = str(resolution.error).lower()
        for forbidden in ("direct", "direct_parquet_v1", ".bin", "legacy"):
            self.assertNotIn(forbidden, public_error)

    def test_approved_ashare_snapshot_requires_matching_full_quality_evidence(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            approved_ashare_qfq_snapshot_identity,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            reports_dir = root / "reports"
            reports_dir.mkdir()
            snapshot, identity = write_native_ashare_fixture(data_root)
            snapshot_state = filesystem_state_fingerprint(snapshot)
            write_quality_evidence(
                reports_dir,
                {
                    "scan_mode": "full",
                    "coverage_status": "full",
                    "verification_status": "verified",
                    "requested_sample": 0,
                    "scanned_instruments": 100,
                    "data_contract": {
                        "status": "verified",
                        "ashare_daily_qfq": identity,
                        "snapshot_state": stable_snapshot_state(
                            "ashare_daily_qfq", snapshot_state
                        ),
                    },
                },
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
            ):
                approved = approved_ashare_qfq_snapshot_identity()

        self.assertEqual(
            {field: approved[field] for field in identity},
            identity,
        )
        self.assertEqual(approved["state_fingerprint"], snapshot_state)

    def test_approved_ashare_snapshot_rejects_non_trainable_full_quality_evidence(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            approved_ashare_qfq_snapshot_identity,
        )
        from quant_workflow.scripts.quant_data_contract import DataContractError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            reports_dir = root / "reports"
            reports_dir.mkdir()
            snapshot, identity = write_native_ashare_fixture(data_root)
            snapshot_state = filesystem_state_fingerprint(snapshot)
            write_quality_evidence(
                reports_dir,
                {
                    "trainable": False,
                    "scan_mode": "full",
                    "coverage_status": "full",
                    "verification_status": "verified",
                    "requested_sample": 0,
                    "scanned_instruments": 100,
                    "data_contract": {
                        "status": "verified",
                        "ashare_daily_qfq": identity,
                        "snapshot_state": stable_snapshot_state(
                            "ashare_daily_qfq", snapshot_state
                        ),
                    },
                },
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                self.assertRaises(DataContractError),
            ):
                approved_ashare_qfq_snapshot_identity()

    def test_approved_ashare_snapshot_rejects_data_changed_after_quality_scan(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            approved_ashare_qfq_snapshot_identity,
        )
        from quant_workflow.scripts.quant_data_contract import DataContractError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            reports_dir = root / "reports"
            reports_dir.mkdir()
            snapshot, identity = write_native_ashare_fixture(data_root)
            snapshot_state = filesystem_state_fingerprint(snapshot)
            write_quality_evidence(
                reports_dir,
                {
                    "scan_mode": "full",
                    "coverage_status": "full",
                    "verification_status": "verified",
                    "requested_sample": 0,
                    "scanned_instruments": 100,
                    "data_contract": {
                        "status": "verified",
                        "ashare_daily_qfq": identity,
                        "snapshot_state": stable_snapshot_state(
                            "ashare_daily_qfq", snapshot_state
                        ),
                    },
                },
            )
            (snapshot / "bars" / "600000.parquet").write_bytes(b"changed")

            with (
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                self.assertRaises(DataContractError),
            ):
                approved_ashare_qfq_snapshot_identity()

    def test_approved_ashare_snapshot_rejects_sampled_quality_evidence(self) -> None:
        from api.v1.endpoints.research_status import (
            approved_ashare_qfq_snapshot_identity,
        )
        from quant_workflow.scripts.quant_data_contract import DataContractError

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            reports_dir = root / "reports"
            reports_dir.mkdir()
            snapshot, identity = write_native_ashare_fixture(data_root)
            snapshot_state = filesystem_state_fingerprint(snapshot)
            write_quality_evidence(
                reports_dir,
                {
                    "scan_mode": "sampled",
                    "coverage_status": "sampled",
                    "verification_status": "sampled",
                    "requested_sample": 20,
                    "scanned_instruments": 20,
                    "data_contract": {
                        "status": "verified",
                        "ashare_daily_qfq": identity,
                        "snapshot_state": stable_snapshot_state(
                            "ashare_daily_qfq", snapshot_state
                        ),
                    },
                },
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                self.assertRaises(DataContractError),
            ):
                approved_ashare_qfq_snapshot_identity()

    def _write_native_qlib_fixture(
        self,
        data_root: Path,
        *,
        snapshot_id: str = "native-v2",
        dataset_id: str = "qlib_ashare_v2",
        dataset_root: Path = Path("qlib/cn_data_ashare_v2"),
        dependency_snapshots: list[dict[str, str]] | None = None,
    ) -> Path:
        provider = data_root / dataset_root / "snapshots" / snapshot_id
        (provider / "calendars").mkdir(parents=True)
        (provider / "instruments").mkdir(parents=True)
        (provider / "features" / "sh600000").mkdir(parents=True)
        (provider / "calendars" / "day.txt").write_text(
            "2026-07-23\n2026-07-24\n", encoding="utf-8"
        )
        (provider / "instruments" / "all.txt").write_text(
            "SH600000\t2026-07-23\t2026-07-24\n", encoding="utf-8"
        )
        (provider / "features" / "sh600000" / "close.day.bin").write_bytes(
            b"feature-bytes"
        )
        upstream_snapshot = {
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "qfq-source-001",
            "content_sha256": "a" * 64,
            "price_basis": "qfq",
        }
        atomic_write_json(
            provider / "PROVIDER_MANIFEST.json",
            {
                "schema_version": 2,
                "status": "ready",
                "source_snapshot": upstream_snapshot,
            },
        )
        finalize_qlib_snapshot(
            provider,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            source={
                "provider": "quant_workflow",
                "interface": "v2_snapshot_to_qlib",
            },
            source_retrieved_at="2026-07-28T18:42:05+08:00",
            producer={"script": "build_qlib_v2.py", "version": "2"},
            upstream_snapshot=upstream_snapshot,
            dependency_snapshots=dependency_snapshots,
        )
        dataset_manifest_path = provider / "DATASET_MANIFEST.json"
        dataset_manifest_bytes = dataset_manifest_path.read_bytes()
        dataset_manifest = json.loads(dataset_manifest_bytes.decode("utf-8"))
        manifest_sha256 = hashlib.sha256(dataset_manifest_bytes).hexdigest()
        data_root.mkdir(parents=True, exist_ok=True)
        catalog_path = data_root / "DATA_MANIFEST.json"
        catalog = (
            json.loads(catalog_path.read_text(encoding="utf-8"))
            if catalog_path.is_file()
            else {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {},
            }
        )
        catalog.setdefault("schema_version", 2)
        catalog.setdefault("path_format", "relative_to_quant_data_root")
        catalog.setdefault("datasets", {}).setdefault(
            "ashare_daily_qfq",
            {
                "dataset_id": "ashare_daily_qfq",
                "role": "audit",
                "status": "ready",
                "price_basis": upstream_snapshot["price_basis"],
                "snapshot_id": upstream_snapshot["snapshot_id"],
                "content_sha256": upstream_snapshot["content_sha256"],
            },
        )
        catalog.setdefault("datasets", {})[dataset_id] = {
            "dataset_id": dataset_id,
            "dataset_root": dataset_root.as_posix(),
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "content_sha256": dataset_manifest["content_sha256"],
            "snapshot_id": snapshot_id,
            "snapshot_manifest_sha256": manifest_sha256,
            "latest_pointer": (dataset_root / "LATEST.json").as_posix(),
            "upstream_snapshots": [upstream_snapshot],
            "dependency_snapshots": dependency_snapshots or [],
        }
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        latest_pointer = data_root / dataset_root / "LATEST.json"
        latest_pointer.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "dataset_id": dataset_id,
                    "snapshot_id": snapshot_id,
                    "manifest_sha256": manifest_sha256,
                    "snapshot_path": (dataset_root / "snapshots" / snapshot_id).as_posix(),
                }
            ),
            encoding="utf-8",
        )
        return provider

    def _write_native_hygiene_inventory(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        from quant_workflow.scripts.report_data_hygiene import collect_v2_catalog

        reports_dir = root / "quant_workflow" / "reports"
        reports_dir.mkdir(parents=True)
        quant_data_root = root / "quant_data"
        provider = self._write_native_qlib_fixture(quant_data_root)
        catalog_path = quant_data_root / "DATA_MANIFEST.json"
        root_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        root_catalog["datasets"]["qlib_ashare_v2"]["data_kind"] = "qlib_provider"
        catalog_path.write_text(json.dumps(root_catalog), encoding="utf-8")
        catalog = collect_v2_catalog(quant_data_root)
        self._write_hygiene_bundle(
            reports_dir,
            {
                "generated_at": "2026-07-30T12:00:00+08:00",
                "catalog": catalog,
                "items": [
                    {
                        "category": "official",
                        "dataset_id": item["dataset_id"],
                    }
                    for item in catalog["datasets"]
                    if item.get("canonical_valid") is True
                ],
            },
        )
        return reports_dir, quant_data_root, provider

    def test_v2_manifest_evidence_rejects_partial_catalog_with_malformed_entry(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            _manifest_official_entry_evidence,
        )

        manifest = {
            "schema_version": 2,
            "datasets": {
                "qlib_ashare_v2": {
                    "dataset_id": "qlib_ashare_v2",
                    "role": "canonical",
                    "status": "ready",
                },
                "malformed": "not-an-object",
            },
        }
        manifest_sha256 = "a" * 64
        inventory = {
            "catalog": {
                "status": "partial",
                "errors": ["malformed: dataset entry must be an object"],
                "manifest_sha256": manifest_sha256,
                "validation_contract_version": 2,
                "dataset_count": 1,
                "validated_dataset_ids": ["qlib_ashare_v2"],
                "datasets": [
                    {
                        "dataset_id": "qlib_ashare_v2",
                        "canonical_valid": True,
                        "snapshot_path": "/tmp/native-v2",
                        "state_fingerprint": {},
                    }
                ],
            }
        }

        with (
            patch(
                "api.v1.endpoints.research_status._inventory_latest_pointer_status",
                return_value="verified",
            ),
            patch(
                "api.v1.endpoints.research_status._inventory_snapshot_state_status",
                return_value="verified",
            ),
        ):
            evidence = _manifest_official_entry_evidence(
                manifest,
                inventory=inventory,
                current_manifest_sha256=manifest_sha256,
            )

        self.assertEqual(evidence["official_entries"], 0)
        self.assertEqual(evidence["validation_status"], "unverified")

    def _write_hygiene_bundle(
        self,
        reports_dir: Path,
        inventory: dict,
        cleanup_plan: dict | None = None,
    ) -> Path:
        cleanup_plan = cleanup_plan or {
            "generated_at": "2026-07-30T12:00:00+08:00",
            "dry_run_only": True,
            "actions": [],
        }
        inventory_bytes = (
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        cleanup_bytes = (
            json.dumps(cleanup_plan, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        bundle_digest = hashlib.sha256(
            inventory_bytes + b"\0" + cleanup_bytes
        ).hexdigest()
        suffix = f"20260730_120000_{bundle_digest[:16]}"
        inventory_artifact = reports_dir / f"data_hygiene_inventory_{suffix}.json"
        cleanup_artifact = reports_dir / f"data_hygiene_cleanup_plan_{suffix}.json"
        inventory_artifact.write_bytes(inventory_bytes)
        cleanup_artifact.write_bytes(cleanup_bytes)
        (reports_dir / "data_hygiene_inventory_latest.json").write_bytes(
            inventory_bytes
        )
        (reports_dir / "data_hygiene_cleanup_plan_latest.json").write_bytes(
            cleanup_bytes
        )
        pointer = reports_dir / "data_hygiene_reports_latest.json"
        pointer.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "publication_contract": (
                        "atomic_json_pointer_to_immutable_hygiene_bundle_v1"
                    ),
                    "bundle_id": f"data-hygiene-{bundle_digest[:16]}",
                    "inventory_artifact": inventory_artifact.name,
                    "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
                    "cleanup_plan_artifact": cleanup_artifact.name,
                    "cleanup_plan_sha256": hashlib.sha256(cleanup_bytes).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        return pointer

    def _read_hygiene_inventory(self, reports_dir: Path) -> dict:
        pointer = json.loads(
            (reports_dir / "data_hygiene_reports_latest.json").read_text(
                encoding="utf-8"
            )
        )
        return json.loads(
            (reports_dir / pointer["inventory_artifact"]).read_text(
                encoding="utf-8"
            )
        )

    def test_status_rejects_root_catalog_without_v2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            catalog_path = data_root / "DATA_MANIFEST.json"
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog.pop("schema_version")
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertIn("Qlib Provider", payload["qlib_provider"]["provider_error"])

    def test_status_rejects_latest_pointer_without_v2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            pointer_path = data_root / "qlib" / "cn_data_ashare_v2" / "LATEST.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer.pop("schema_version")
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertIn("Qlib Provider", payload["qlib_provider"]["provider_error"])

    def test_saved_cleaned_v2_result_is_current_against_cleaned_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            canonical = self._write_native_qlib_fixture(data_root)
            canonical_manifest = json.loads(
                (canonical / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
            )
            canonical_identity = {
                field: str(canonical_manifest[field])
                for field in (
                    "dataset_id",
                    "snapshot_id",
                    "content_sha256",
                    "price_basis",
                )
            }
            cleaned = self._write_native_qlib_fixture(
                data_root,
                snapshot_id="cleaned-v2",
                dataset_id="qlib_ashare_cleaned_v2",
                dataset_root=Path("qlib/cn_data_ashare_cleaned_v2"),
                dependency_snapshots=[canonical_identity],
            )
            manifest_bytes = (cleaned / "DATASET_MANIFEST.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "latest_training_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provider": "cleaned",
                        "provider_uri": str(cleaned),
                        "provider_identity": {
                            "provider_contract": "immutable_snapshot_v2",
                            "dataset_id": "qlib_ashare_cleaned_v2",
                            "snapshot_id": "cleaned-v2",
                            "content_sha256": manifest["content_sha256"],
                            "manifest_sha256": hashlib.sha256(
                                manifest_bytes
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = self._build_status_for_roots(reports_dir, data_root)

        self.assertTrue(payload["latest_training"]["provider_current"])
        self.assertEqual(
            payload["latest_training"]["provider_status"],
            "current",
        )
        self.assertTrue(payload["latest_training_provider"]["provider_current"])
        self.assertEqual(
            payload["qlib_provider"]["dataset_id"],
            "qlib_ashare_v2",
        )
        self.assertEqual(
            Path(payload["qlib_provider"]["path"]).resolve(),
            canonical.resolve(),
        )
        self.assertEqual(
            payload["latest_training_provider"]["dataset_id"],
            "qlib_ashare_cleaned_v2",
        )
        self.assertEqual(
            Path(payload["latest_training_provider"]["path"]).resolve(),
            cleaned.resolve(),
        )

    def test_incomplete_saved_identity_is_never_current_native_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            manifest_bytes = (provider / "DATASET_MANIFEST.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "latest_training_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provider_uri": str(provider),
                        "provider_identity": {
                            "dataset_id": manifest["dataset_id"],
                            "snapshot_id": manifest["snapshot_id"],
                            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = self._build_status_for_roots(reports_dir, data_root)

        self.assertFalse(payload["latest_training"]["provider_current"])
        self.assertEqual(payload["latest_training"]["provider_status"], "unknown")

    def _build_status_for_roots(
        self, reports_dir: Path, data_root: Path
    ) -> dict[str, object]:
        from api.v1.endpoints.research_status import build_research_status_payload

        reports_dir.mkdir(parents=True, exist_ok=True)
        status_path = reports_dir / "research_status_latest.json"
        if not status_path.exists():
            status_path.write_text("{}", encoding="utf-8")
        with (
            patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                status_path,
            ),
            patch(
                "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                reports_dir,
            ),
            patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                data_root,
            ),
            patch(
                "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                data_root / "qlib",
            ),
        ):
            # These fixtures exercise the retained legacy .bin compatibility
            # catalog.  The production default is the direct Parquet Provider;
            # opt into the old catalog explicitly in legacy-focused tests.
            return build_research_status_payload(
                provider_dataset_id="qlib_ashare_v2"
            )

    def test_robust_signal_status_exposes_bounded_sanitized_matrices(self) -> None:
        from api.v1.endpoints.research_status import _robust_signal_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            signal_dir = reports_dir / "robust_signal"
            run_id = "robust_signal_20260723"
            run_dir = signal_dir / "runs" / run_id
            (run_dir / "screening").mkdir(parents=True)
            (run_dir / "validation").mkdir(parents=True)
            (signal_dir / "robust_signal_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "signal_candidate_ready",
                        "lifecycle_status": "completed",
                        "run_id": run_id,
                        "current_stage": "completed",
                        "pause_state": "completed",
                        "completed_count": 48,
                        "expected_count": 48,
                        "next_gate": "create_goal_2_trade_strategy",
                        "updated_at": "2026-07-23T12:00:00+08:00",
                        "provider_uri": "/Users/cq/private/provider",
                    }
                ),
                encoding="utf-8",
            )
            report = {
                "phase": "validation",
                "signal_id": "alpha158_excess_5d",
                "model_id": "linear_ridge",
                "passed": True,
                "failed_gates": [],
                "values": {
                    "median_quarter_rank_ic": 0.03,
                    "top50_overlap_mean": 0.66,
                },
                "quarter_rank_ic": {"2023Q1": 0.03},
                "window_stability": {
                    "rank_correlation_median": 0.78,
                    "topk_overlap_mean": 0.66,
                },
                "topk_sensitivity": {
                    "primary_topk": 50,
                    "results": {"20": {"mean_label": 0.02}, "50": {"mean_label": 0.01}},
                },
                "factor_quality": {
                    "feature_count_min": 158,
                    "feature_count_max": 158,
                    "max_missing_rate": 0.02,
                    "constant_features": [],
                    "high_correlation_pair_count_max": 3,
                },
                "market_state_attribution": {
                    "risk_on": {"period_count": 10, "annual_excess_return": 0.08},
                    "neutral": {"period_count": 8, "annual_excess_return": 0.03},
                    "defensive": {"period_count": 4, "annual_excess_return": -0.01},
                    "risk_off": {"period_count": 2, "annual_excess_return": 0.01},
                },
                "private_path": "/Users/cq/private/prediction.parquet",
            }
            (run_dir / "validation" / "validation_report.json").write_text(
                json.dumps({"reports": [report]}), encoding="utf-8"
            )
            (run_dir / "screening" / "screening_report.json").write_text(
                json.dumps({"reports": []}), encoding="utf-8"
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "status": "signal_candidate_ready",
                        "selected_signal_id": "alpha158_excess_5d",
                        "selected_model_id": "linear_ridge",
                        "factor_contract": "/Users/cq/private/factor_contract.json",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "factor_contract.json").write_text(
                json.dumps(
                    {
                        "status": "signal_candidate_ready",
                        "signal_id": "alpha158_excess_5d",
                        "model_id": "linear_ridge",
                        "horizon_days": 5,
                        "contract_hash": "abc123",
                        "provider_tree_sha256": "provider-hash",
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_signal_payload(reports_dir)

        self.assertEqual(payload["status"], "signal_candidate_ready")
        self.assertEqual(payload["progress"]["completed_count"], 48)
        self.assertEqual(payload["result"]["selected_signal_id"], "alpha158_excess_5d")
        self.assertEqual(
            payload["validation_reports"][0]["factor_quality"]["feature_count_min"],
            158,
        )
        self.assertEqual(
            set(payload["validation_reports"][0]["market_state_attribution"]),
            {"risk_on", "neutral", "defensive", "risk_off"},
        )
        self.assertNotIn("/Users/", json.dumps(payload))

    def test_robust_signal_status_exposes_dedicated_current_manifest_readiness(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _robust_signal_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            signal_dir = reports_dir / "robust_signal"
            signal_dir.mkdir(parents=True)
            provider = root / "qlib-provider"
            provider.mkdir()
            identity_specs = {
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
            data_identities = {
                dataset_id: {
                    "dataset_id": dataset_id,
                    "snapshot_id": f"{dataset_id}-snapshot",
                    "manifest_sha256": "a" * 64,
                    "content_sha256": "b" * 64,
                    "latest_trade_date": "2026-08-07",
                    "price_basis": basis,
                    "data_kind": data_kind,
                }
                for dataset_id, (basis, data_kind) in identity_specs.items()
            }
            data_identity_sha256 = hashlib.sha256(
                json.dumps(
                    data_identities,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            cleaned_identity = {
                "provider_contract": "direct_parquet_v1",
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "cleaned-snapshot-v2",
                "manifest_sha256": "d" * 64,
                "content_sha256": "e" * 64,
                "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
                "descriptor_sha256": "f" * 64,
                "cleaning_mode": "cleaned",
                "index_dataset_id": "cn_index_daily",
                "index_snapshot_id": "index-snapshot-v2",
                "index_content_sha256": "1" * 64,
                "index_manifest_sha256": "2" * 64,
            }
            publish_robust_signal_data_readiness(
                pointer_path=(signal_dir / "robust_signal_data_readiness_latest.json"),
                payload={
                    "contract_version": CONTRACT_VERSION,
                    "acceptance_owner": ACCEPTANCE_OWNER,
                    "acceptance_contract": ACCEPTANCE_CONTRACT,
                    "status": "ready",
                    "trainable": True,
                    "generated_at": "2026-08-02T16:00:00+08:00",
                    "provider_uri": str(provider),
                    "provider_key": cleaned_identity["descriptor_sha256"],
                    "provider_tree_sha256": cleaned_identity["descriptor_sha256"],
                    "data_identity_sha256": data_identity_sha256,
                    "data_identities": data_identities,
                    "cleaned_provider_identity": cleaned_identity,
                    "canonical_snapshot_id": "cleaned-snapshot-v2",
                    "symbols": 1633,
                    "intervals": 2058,
                },
            )
            service = MagicMock()

            with (
                patch(
                    "quant_workflow.scripts.robust_signal_data_contract."
                    "QuantDataServiceClient",
                    return_value=service,
                ),
                patch(
                    "api.v1.endpoints.research_status.qlib_workflow_provider_uri",
                    return_value=provider,
                ) as provider_resolver,
            ):
                payload = _robust_signal_payload(reports_dir)

        readiness = payload["data_readiness"]
        self.assertEqual(readiness["status"], "ready")
        self.assertTrue(readiness["trainable"])
        self.assertEqual(readiness["provider_name"], "Qlib Provider")
        self.assertEqual(readiness["data_identity_sha256"], data_identity_sha256)
        self.assertEqual(readiness["data_identities"], data_identities)
        self.assertEqual(
            readiness["canonical_snapshot_id"], "ashare_daily_qfq-snapshot"
        )
        self.assertTrue(readiness["readiness_artifact_name"].startswith(
            "robust_signal_data_readiness_"
        ))
        self.assertEqual(len(readiness["readiness_artifact_sha256"]), 64)
        self.assertNotIn("cleaned_provider_identity", readiness)
        self.assertNotIn("data_manifest_sha256", readiness)
        service.ready.assert_called_once_with()
        self.assertEqual(
            service.resolve_dataset.call_args_list,
            [
                call(dataset_id, basis=identity["price_basis"], expected_identity=ANY)
                for dataset_id, identity in data_identities.items()
            ],
        )
        provider_resolver.assert_called_once()
        self.assertNotIn(str(root), json.dumps(payload))

    def test_robust_signal_status_keeps_smoke_run_separate_and_reads_live_progress(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _robust_signal_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            signal_dir = reports_dir / "robust_signal"
            formal_run_id = "robust_signal_20260802_120000"
            smoke_run_id = "robust_signal_20260802_130000"
            formal_run_dir = signal_dir / "runs" / formal_run_id
            smoke_run_dir = signal_dir / "runs" / smoke_run_id
            formal_run_dir.mkdir(parents=True)
            (smoke_run_dir / "screening").mkdir(parents=True)

            (signal_dir / "robust_signal_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "signal_diagnosis_complete",
                        "lifecycle_status": "completed",
                        "run_id": formal_run_id,
                        "current_stage": "completed",
                        "pause_state": "completed",
                        "completed_count": 320,
                        "expected_count": 320,
                    }
                ),
                encoding="utf-8",
            )
            smoke_alias = {
                "status": "signal_diagnosis_complete",
                "lifecycle_status": "running",
                "run_id": smoke_run_id,
                "current_stage": "screening",
                "pause_state": "running",
                "completed_count": 0,
                "expected_count": 1,
            }
            (signal_dir / "robust_signal_smoke_status_latest.json").write_text(
                json.dumps(smoke_alias), encoding="utf-8"
            )
            (smoke_run_dir / "status.json").write_text(
                json.dumps({**smoke_alias, "pause_state": "pause_requested"}),
                encoding="utf-8",
            )
            (smoke_run_dir / "screening" / "progress.json").write_text(
                json.dumps(
                    {
                        "current_job": (
                            "screening__trend_momentum_excess_5d__"
                            "equal_weight_rank__2023Q1__w756"
                        ),
                        "completed_count": 0,
                        "expected_count": 1,
                        "pause_state": "running",
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_signal_payload(reports_dir)

        self.assertEqual(payload["run_id"], formal_run_id)
        self.assertEqual(payload["completed_count"], 320)
        smoke = payload["smoke_run"]
        self.assertTrue(smoke["test_only"])
        self.assertEqual(smoke["run_id"], smoke_run_id)
        self.assertEqual(smoke["pause_state"], "pause_requested")
        self.assertEqual(smoke["progress"]["expected_count"], 1)
        self.assertIn("2023Q1", smoke["progress"]["current_job"])
        self.assertNotIn("/Users/", json.dumps(payload))

    def test_robust_signal_progress_ignores_corrupt_count_fields(self) -> None:
        from api.v1.endpoints.research_status import _robust_signal_progress_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            screening_dir = run_dir / "screening"
            screening_dir.mkdir()
            (screening_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "current_job": "screening__2023Q1",
                        "completed_count": {"invalid": True},
                    }
                ),
                encoding="utf-8",
            )

            progress = _robust_signal_progress_payload(
                {
                    "lifecycle_status": "running",
                    "current_stage": "screening",
                    "completed_count": ["invalid"],
                    "expected_count": "invalid",
                },
                run_dir,
            )

        self.assertEqual(progress["completed_count"], 0)
        self.assertEqual(progress["expected_count"], 0)
        self.assertEqual(progress["current_job"], "screening__2023Q1")

    def test_robust_status_exposes_run_scoped_membership_progress(self) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            run_id = "robust_20260717_progress"
            robust_dir.mkdir()
            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "lifecycle_status": "running",
                        "reason_code": "preflight_not_completed",
                        "run_id": run_id,
                        "current_stage": "build_pit_csi800",
                        "updated_at": "2026-07-17T10:00:00+08:00",
                        "stages": {
                            "build_pit_csi800": {
                                "status": "running",
                                "started_at": "2026-07-17T09:00:00+08:00",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            progress_dir = (
                robust_dir
                / "runs"
                / run_id
                / "stage_reports"
                / "build_pit_csi800"
            )
            progress_dir.mkdir(parents=True)
            (progress_dir / "build_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "completed_date_count": 1250,
                        "expected_date_count": 3047,
                        "generated_at": "2026-07-17T09:59:00+08:00",
                        "membership_source": {
                            "selected": "baostock",
                            "private_path": "/Users/cq/private/data",
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_strategy_payload(reports_dir)

        self.assertEqual(
            payload["progress"],
            {
                "stage": "build_pit_csi800",
                "status": "running",
                "completed_date_count": 1250,
                "expected_date_count": 3047,
                "generated_at": "2026-07-17T09:59:00+08:00",
            },
        )
        self.assertNotIn("/Users/", json.dumps(payload))

    def test_status_exposes_sanitized_robust_strategy_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-16T08:00:00+08:00",
                        "trainable": False,
                        "training_readiness": {
                            "status": "blocked",
                            "summary": "history unavailable",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            robust_dir = root / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "lifecycle_status": "stopped",
                        "reason_code": "jqdata_permission_denied",
                        "run_id": "robust_20260716",
                        "failed_stage": "build_pit_csi800",
                        "updated_at": "2026-07-16T08:10:00+08:00",
                        "stages": {
                            "build_pit_csi800": {
                                "status": "blocked_data",
                                "started_at": "2026-07-16T08:09:00+08:00",
                                "finished_at": "2026-07-16T08:10:00+08:00",
                                "argv": ["python", "/Users/cq/private/script.py"],
                                "evidence": {
                                    "reason_code": "jqdata_permission_denied",
                                    "secret_path": "/Users/cq/private/data.parquet",
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "development_latest.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "development_1",
                        "generated_at": "2026-07-16T08:00:00+08:00",
                        "selected_models": ["lightgbm", "linear_ridge"],
                        "selected_strategy": "full_risk_control",
                        "model_reports": [
                            {
                                "model_id": "lightgbm",
                                "eligible": True,
                                "stability": {
                                    "rank_correlation_median": 0.78,
                                    "topk_overlap_mean": 0.66,
                                },
                                "model_path": "/Users/cq/private/model.pkl",
                            }
                        ],
                        "evaluation": {
                            "argv": ["python", "/Users/cq/private/dev.py"],
                            "strategies": [
                                {
                                    "strategy_id": "full_risk_control",
                                    "eligible_for_final_selection": True,
                                    "acceptance": {
                                        "passed": True,
                                        "failed_gates": [],
                                        "values": {
                                            "aggregate_annual_excess_return": 0.08,
                                            "positive_excess_quarter_ratio": 0.75,
                                        },
                                        "thresholds": {
                                            "annual_excess_return_min": 0.05,
                                            "positive_excess_quarter_ratio_min": 0.70,
                                        },
                                    },
                                    "baseline": {
                                        "metrics": {"annual_excess_return": 0.08}
                                    },
                                    "stress": {
                                        "metrics": {"annual_excess_return": 0.02}
                                    },
                                    "delayed": {
                                        "metrics": {"annual_excess_return": 0.01}
                                    },
                                    "quarter_metrics": [
                                        {"quarter": "2025Q4", "rank_ic": 0.03}
                                    ],
                                    "regime_metrics": {
                                        "risk_on": {
                                            "annual_excess_return": 0.09,
                                            "days": 120,
                                        }
                                    },
                                    "board_attribution": {
                                        "main": {
                                            "average_exposure": 0.55,
                                            "max_exposure": 0.68,
                                            "return_contribution": 0.12,
                                            "excess_contribution": 0.03,
                                            "days": 756,
                                        }
                                    },
                                    "attribution_reconciliation": {
                                        "reconciled": True,
                                        "cash_excess_contribution": -0.01,
                                        "cumulative_excess_return": 0.02,
                                        "excess_residual": 0.0,
                                    },
                                }
                            ],
                        },
                        "provider_uri": "/Users/cq/private/provider",
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "confirmation_latest.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_observation",
                        "run_id": "confirmation_1",
                        "generated_at": "2026-07-16T08:01:00+08:00",
                        "latest_market_day": "2026-07-15",
                        "metrics": {"annual_excess_return": 0.02, "max_drawdown": -0.1},
                        "backtest_artifact": "/Users/cq/private/confirmation.parquet",
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "shadow_state_latest.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_observation",
                        "start_after": "2026-07-15",
                        "updated_at": "2026-07-16T08:02:00+08:00",
                        "metrics": {
                            "trading_days": 3,
                            "net_excess_return": 0.01,
                            "observations": [
                                {"date": "2026-07-16", "daily_return": 0.01}
                            ],
                        },
                        "verdict": {"remaining_days": 57, "passed": False},
                        "observations": [
                            {"date": "2026-07-16", "portfolio_return": 0.01}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        robust = resp.json()["robust_strategy"]
        self.assertEqual(robust["status"], "blocked_data")
        self.assertEqual(robust["lifecycle_status"], "stopped")
        self.assertEqual(robust["reason_code"], "jqdata_permission_denied")
        self.assertEqual(robust["stages"][0]["name"], "build_pit_csi800")
        self.assertEqual(robust["stages"][0]["reason_code"], "jqdata_permission_denied")
        self.assertNotIn("argv", robust["stages"][0])
        self.assertEqual(
            robust["development"]["selected_models"], ["lightgbm", "linear_ridge"]
        )
        self.assertEqual(
            robust["development"]["model_stability"][0]["model_id"], "lightgbm"
        )
        self.assertEqual(
            robust["development"]["strategy_matrix"][0]["strategy_id"],
            "full_risk_control",
        )
        strategy = robust["development"]["strategy_matrix"][0]
        self.assertTrue(strategy["eligible_for_final_selection"])
        self.assertEqual(
            strategy["acceptance"]["thresholds"]["annual_excess_return_min"],
            0.05,
        )
        self.assertEqual(
            strategy["board_attribution"]["main"]["excess_contribution"], 0.03
        )
        self.assertTrue(strategy["attribution_reconciliation"]["reconciled"])
        self.assertEqual(
            strategy["attribution_reconciliation"]["cash_excess_contribution"],
            -0.01,
        )
        self.assertEqual(
            robust["development"]["strategy_matrix"][0]["quarter_metrics"][0][
                "quarter"
            ],
            "2025Q4",
        )
        self.assertNotIn("evaluation", robust["development"])
        self.assertNotIn("model_reports", robust["development"])
        self.assertEqual(
            robust["confirmation"]["metrics"]["annual_excess_return"], 0.02
        )
        self.assertEqual(robust["shadow"]["metrics"]["trading_days"], 3)
        self.assertEqual(robust["shadow"]["verdict"]["remaining_days"], 57)
        self.assertNotIn("observations", robust["shadow"])
        self.assertNotIn("observations", robust["shadow"]["metrics"])
        self.assertEqual(
            robust["mlflow_experiments"],
            ["robust_csi800_development", "robust_csi800_confirmation"],
        )
        self.assertNotIn("/Users/", json.dumps(robust))

    def test_robust_status_marks_each_missing_subreport_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-16T08:00:00+08:00",
                        "trainable": False,
                        "training_readiness": {
                            "status": "blocked",
                            "summary": "history unavailable",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            robust_dir = root / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps({"status": "running", "lifecycle_status": "running"}),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        robust = resp.json()["robust_strategy"]
        self.assertEqual(robust["status"], "running")
        for key in (
            "preflight",
            "development",
            "confirmation",
            "shadow",
            "shadow_daily",
        ):
            self.assertEqual(robust[key], {"status": "missing"})

    def test_robust_lifecycle_uses_newest_report_timestamp(self) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "lifecycle_status": "stopped",
                        "run_id": "development-one",
                        "updated_at": "2026-07-16T08:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "confirmation_latest.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_observation",
                        "run_id": "confirmation-one",
                        "generated_at": "2026-07-16T09:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )

            newer_confirmation = _robust_strategy_payload(reports_dir)
            self.assertEqual(newer_confirmation["status"], "shadow_observation")
            self.assertEqual(
                newer_confirmation["lifecycle_status"], "shadow_observation"
            )
            self.assertEqual(newer_confirmation["run_id"], "confirmation-one")

            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "lifecycle_status": "stopped",
                        "reason_code": "membership_coverage_below_threshold",
                        "run_id": "development-two",
                        "updated_at": "2026-07-16T10:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )

            newer_block = _robust_strategy_payload(reports_dir)

        self.assertEqual(newer_block["status"], "blocked_data")
        self.assertEqual(
            newer_block["reason_code"], "membership_coverage_below_threshold"
        )
        self.assertEqual(newer_block["run_id"], "development-two")

    def test_robust_lifecycle_compares_explicit_timestamp_with_mtime_fallback(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            orchestration_path = robust_dir / "robust_strategy_status_latest.json"
            development_path = robust_dir / "development_latest.json"
            orchestration_path.write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "lifecycle_status": "stopped",
                        "run_id": "old-explicit",
                        "updated_at": "2000-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            development_path.write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "run_id": "new-mtime-fallback",
                    }
                ),
                encoding="utf-8",
            )
            newer_mtime = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()
            os.utime(development_path, (newer_mtime, newer_mtime))

            payload = _robust_strategy_payload(reports_dir)

        self.assertEqual(payload["status"], "research_candidate_ready")
        self.assertEqual(payload["run_id"], "new-mtime-fallback")

    def test_shadow_daily_is_exposed_without_a_lifecycle_report(self) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "shadow_daily_latest.json").write_text(
                json.dumps(
                    {
                        "status": "development_failed",
                        "reason_code": "inference_failed",
                        "as_of": "2026-07-16",
                        "inference_success": False,
                        "generated_at": "2026-07-16T18:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_strategy_payload(reports_dir)

        self.assertEqual(payload["status"], "missing")
        self.assertEqual(payload["lifecycle_status"], "not_started")
        self.assertEqual(payload["shadow_daily"]["status"], "development_failed")
        self.assertEqual(payload["shadow_daily"]["reason_code"], "inference_failed")
        self.assertEqual(payload["shadow_daily"]["as_of"], "2026-07-16")

    def test_shadow_daily_failure_is_separate_from_strategy_lifecycle(self) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "confirmation_latest.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_observation",
                        "run_id": "confirmation-one",
                        "generated_at": "2026-07-16T09:00:00+08:00",
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "shadow_daily_latest.json").write_text(
                json.dumps(
                    {
                        "status": "development_failed",
                        "reason_code": "inference_failed",
                        "as_of": "2026-07-16",
                        "outcome": "completed",
                        "inference_success": False,
                        "appended_days": 1,
                        "generated_at": "2026-07-16T18:00:00+08:00",
                        "error_type": "RuntimeError",
                        "command": ["python", "/Users/cq/private/run.py"],
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_strategy_payload(reports_dir)

        self.assertEqual(payload["status"], "shadow_observation")
        self.assertEqual(payload["shadow_daily"]["status"], "development_failed")
        self.assertEqual(payload["shadow_daily"]["reason_code"], "inference_failed")
        self.assertEqual(payload["shadow_daily"]["as_of"], "2026-07-16")
        self.assertNotIn("command", payload["shadow_daily"])
        self.assertNotIn("/Users/", json.dumps(payload))

    def test_malformed_stages_do_not_break_research_or_quant_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-16T08:00:00+08:00",
                        "trainable": False,
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            robust_dir = root / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "robust_strategy_status_latest.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "updated_at": "2026-07-16T09:00:00+08:00",
                        "stages": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                research = client.get("/api/v1/research/status")
                quant = client.get("/api/v1/quant/status")

        self.assertEqual(research.status_code, 200)
        self.assertEqual(quant.status_code, 200)
        self.assertEqual(research.json()["robust_strategy"]["stages"], [])
        self.assertEqual(research.json()["robust_strategy"]["stages_status"], "invalid")

    def test_robust_payload_redacts_cross_platform_paths_and_rejects_string_gates(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            (robust_dir / "preflight_latest.json").write_text(
                json.dumps(
                    {
                        "status": "blocked_data",
                        "generated_at": "2026-07-16T09:00:00+08:00",
                        "summary": {
                            "linux_mount": "/mnt/private/history.parquet",
                            "linux_service": "/srv/internal/provider",
                            "unc": "\\\\research-srv\\secret\\provider",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (robust_dir / "confirmation_latest.json").write_text(
                json.dumps(
                    {
                        "status": "shadow_observation",
                        "generated_at": "2026-07-16T10:00:00+08:00",
                        "verdict": {
                            "gates": {
                                "real_false": False,
                                "string_false": "false",
                                "string_zero": "0",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_strategy_payload(reports_dir)

        serialized = json.dumps(payload)
        self.assertNotIn("/mnt/private", serialized)
        self.assertNotIn("/srv/internal", serialized)
        self.assertNotIn("research-srv", serialized)
        self.assertEqual(
            payload["confirmation"]["verdict"]["gates"], {"real_false": False}
        )
        self.assertEqual(
            payload["confirmation"]["verdict"]["invalid_gates"],
            ["string_false", "string_zero"],
        )

    def test_robust_lists_are_bounded_and_report_truncation(self) -> None:
        from api.v1.endpoints.research_status import (
            ROBUST_LIST_PREVIEW_LIMIT,
            _robust_strategy_payload,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            total = ROBUST_LIST_PREVIEW_LIMIT + 3
            (robust_dir / "development_latest.json").write_text(
                json.dumps(
                    {
                        "status": "research_candidate_ready",
                        "generated_at": "2026-07-16T10:00:00+08:00",
                        "model_reports": [
                            {"model_id": f"model-{index}"} for index in range(total)
                        ],
                        "evaluation": {
                            "strategies": [
                                {
                                    "strategy_id": "bounded",
                                    "quarter_metrics": [
                                        {"quarter": f"quarter-{index}", "rank_ic": 0.01}
                                        for index in range(total)
                                    ],
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = _robust_strategy_payload(reports_dir)

        development = payload["development"]
        self.assertEqual(len(development["model_stability"]), ROBUST_LIST_PREVIEW_LIMIT)
        self.assertEqual(development["model_stability_total"], total)
        self.assertTrue(development["model_stability_truncated"])
        strategy = development["strategy_matrix"][0]
        self.assertEqual(len(strategy["quarter_metrics"]), ROBUST_LIST_PREVIEW_LIMIT)
        self.assertEqual(strategy["quarter_metrics_total"], total)
        self.assertTrue(strategy["quarter_metrics_truncated"])

    def test_oversized_robust_report_is_marked_invalid_without_parsing(self) -> None:
        from api.v1.endpoints.research_status import _robust_strategy_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            robust_dir = reports_dir / "robust_strategy"
            robust_dir.mkdir()
            report = robust_dir / "robust_strategy_status_latest.json"
            report.write_text(
                '{"status":"running","padding":"0123456789"}', encoding="utf-8"
            )

            with patch("api.v1.endpoints.research_status.ROBUST_REPORT_MAX_BYTES", 16):
                payload = _robust_strategy_payload(reports_dir)

        self.assertEqual(payload["status"], "invalid")
        self.assertEqual(payload["reason_code"], "report_too_large")

    def test_status_reads_latest_quant_workflow_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            app_path = Path(temp_dir) / "同花顺.app"
            app_path.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            log_path = Path(temp_dir) / "experiment_log.md"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            inventory_path = Path(temp_dir) / "data_hygiene_inventory_latest.json"
            cleanup_plan_path = Path(temp_dir) / "data_hygiene_cleanup_plan_latest.json"
            qlib_research_path = (
                Path(temp_dir) / "qlib_research_model_explainability_latest.json"
            )
            quality_report_path = Path(temp_dir) / "data_quality_report_latest.md"
            cleaning_strategy_path = Path(temp_dir) / "data_cleaning_strategy_latest.md"
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            projects_root = Path(temp_dir) / "WorkingProjects"
            manifest_path = quant_data_root / "DATA_MANIFEST.json"
            readme_path = quant_data_root / "README.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "Qlib provider has enough CSI300 coverage.",
                        },
                        "data_sources": [
                            {
                                "name": "akshare",
                                "files": 899,
                                "rows": 1000,
                                "date_range": "2005-01-04 ~ 2026-06-10",
                                "status": "partial",
                            }
                        ],
                        "qlib_provider": {
                            "path": "/tmp/qlib",
                            "calendar_days": 1316,
                            "calendar_range": "2020-01-02 ~ 2025-06-11",
                            "instruments": 5340,
                        },
                        "latest_training": {
                            "status": "completed",
                            "experiment_name": "baseline_lgb_alpha158",
                            "run_id": "abc123",
                            "metrics": {"IC": 0.01},
                        },
                    }
                ),
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：长历史 baseline 已跑通，但收益质量偏弱。\n"
                "- 下一步：补充风险控制和市场择时实验。\n",
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps(
                    {
                        "run_id": "abc123",
                        "experiment_name": "baseline_lgb_alpha158",
                        "status": "completed",
                        "model": "LGBModel",
                        "strategy": "TopkDropoutStrategy",
                        "dataset_range": "2020-01-02 ~ 2026-07-01",
                        "conclusion": "长历史 baseline 已跑通，但收益质量偏弱。",
                        "next_step": "补充风险控制和市场择时实验。",
                        "metrics": {
                            "1day.excess_return_with_cost.annualized_return": 0.033970674,
                            "1day.excess_return_with_cost.information_ratio": 0.261546,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            inventory_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T08:00:00+08:00",
                        "items": (
                            [
                                {"path": f"/tmp/official_{idx}", "category": "official"}
                                for idx in range(2)
                            ]
                            + [
                                {
                                    "path": f"/tmp/candidate_{idx}",
                                    "categories": ["candidate"],
                                }
                                for idx in range(4)
                            ]
                            + [
                                {"path": f"/tmp/legacy_{idx}", "category": "legacy"}
                                for idx in range(5)
                            ]
                            + [
                                {
                                    "path": f"/tmp/archive_{idx}",
                                    "categories": ["archive_candidate"],
                                }
                                for idx in range(8)
                            ]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            cleanup_plan_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T08:00:00+08:00",
                        "dry_run_only": True,
                        "actions": [
                            {
                                "action": "archive",
                                "source": str(
                                    projects_root / "quant_data" / "qlib" / "cn_data"
                                ),
                                "destination": str(
                                    projects_root
                                    / "quant_data"
                                    / "archive"
                                    / "qlib"
                                    / "cn_data"
                                ),
                                "reason": "旧测试 provider，不是当前正式入口。",
                                "risk": "可能影响旧实验复现。",
                                "requires_user_confirmation": True,
                            },
                            {
                                "action": "review",
                                "requires_user_confirmation": False,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            qlib_research_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T09:00:00+08:00",
                        "mode": "read_only_qlib_research",
                        "section": "model_explainability",
                        "title": "模型效果深度解释",
                        "status": "available",
                        "artifacts": {
                            "latest_json": str(qlib_research_path),
                            "latest_markdown": str(
                                Path(temp_dir)
                                / "qlib_research_model_explainability_latest.md"
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            quality_markdown = (
                "# A-Share Data Quality Report\n\n"
                "- Stock universe: 5510\n"
                "- Canonical qfq usable instruments: 5341\n"
                "- Canonical qfq rows: 7,580,088\n"
                "- Canonical qfq date range: 2020-01-03 ~ 2026-07-06\n"
            )
            quality_report_path.write_text(
                quality_markdown,
                encoding="utf-8",
            )
            quality_report_id, quality_evidence_sha256 = write_quality_evidence(
                Path(temp_dir),
                {
                    "scan_mode": "full",
                    "coverage_status": "full",
                    "verification_status": "verified",
                    "trainable": True,
                    "summary": {
                        "stock_universe": 5510,
                        "usable_instruments": 5341,
                        "total_rows": "7,580,088",
                        "date_range": "2020-01-03 ~ 2026-07-06",
                    },
                },
                markdown=quality_markdown,
                timestamp="20260706_090000",
            )
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            status_payload["quality_report_id"] = quality_report_id
            status_payload["quality_evidence_sha256"] = quality_evidence_sha256
            status_path.write_text(json.dumps(status_payload), encoding="utf-8")
            cleaning_strategy_path.write_text("# Cleaning Strategy\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-06T21:49:15+08:00",
                        "source_report": str(inventory_path),
                        "official_entries": {
                            "cleaned_parquet": {"path": "/tmp/cleaned_parquet"},
                            "cleaned_qlib_provider": {"path": "/tmp/cleaned_provider"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            readme_path.write_text("# quant_data\n", encoding="utf-8")

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root
                ),
                patch("api.v1.endpoints.research_status.PROJECTS_ROOT", projects_root),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_ready_provider,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(ready_training_data()),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
                patch.dict(
                    os.environ,
                    {
                        "MLFLOW_URL": "http://127.0.0.1:6106/",
                        "TONGHUASHUN_APP_PATH": str(app_path),
                    },
                    clear=False,
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "trainable")
        self.assertEqual(body["training_preflight"]["status"], "passed")
        self.assertEqual(body["data_sources"], [])
        self.assertEqual(body["provider_quality"], {})
        self.assertIsNone(body["quality_report_id"])
        self.assertIsNone(body["quality_evidence_sha256"])
        self.assertEqual(body["quality_scan"], {})
        self.assertEqual(body["latest_training"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["status"], "available")
        self.assertEqual(body["mlflow"]["url"], "http://127.0.0.1:6106/")
        self.assertEqual(body["mlflow"]["status"], "linked")
        self.assertFalse(body["mlflow"]["health_checked"])
        self.assertIn("workflow page", body["mlflow"]["error"])
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual(body["experiment_notebook"]["log_path"], "experiment_log.md")
        self.assertEqual(
            body["experiment_notebook"]["runs_path"], "experiment_runs.jsonl"
        )
        self.assertIn(
            "长历史 baseline 已跑通", body["experiment_notebook"]["latest_note"]
        )
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "abc123")
        self.assertEqual(
            body["experiment_notebook"]["runs"][0]["dataset_range"],
            "2020-01-02 ~ 2026-07-01",
        )
        self.assertEqual(body["data_hygiene"]["inventory"]["status"], "available")
        self.assertEqual(body["data_hygiene"]["inventory"]["official_entries"], 2)
        self.assertEqual(body["data_hygiene"]["inventory"]["archive_candidates"], 8)
        self.assertTrue(body["data_hygiene"]["cleanup_plan"]["dry_run_only"])
        self.assertEqual(body["data_hygiene"]["cleanup_plan"]["total_actions"], 2)
        self.assertFalse(body["data_hygiene"]["cleanup_plan"]["truncated"])
        self.assertEqual(body["data_hygiene"]["cleanup_plan"]["preview_limit"], 50)
        self.assertTrue(
            body["data_hygiene"]["cleanup_plan"]["actions"][0]["requires_confirmation"]
        )
        self.assertEqual(
            body["data_hygiene"]["cleanup_plan"]["actions"][0]["source"],
            str(Path("WorkingProjects") / "quant_data" / "qlib" / "cn_data"),
        )
        self.assertEqual(
            body["data_hygiene"]["cleanup_plan"]["actions"][0]["destination"],
            str(Path("WorkingProjects") / "quant_data" / "archive" / "qlib" / "cn_data"),
        )
        self.assertEqual(
            body["data_hygiene"]["cleanup_plan"]["actions"][1]["source"], ""
        )
        self.assertIsNone(
            body["data_hygiene"]["cleanup_plan"]["actions"][1]["destination"]
        )
        self.assertEqual(
            body["data_hygiene"]["quality_report"],
            {
                "status": "deprecated",
                "path": None,
                "summary": None,
            },
        )
        self.assertEqual(
            body["data_hygiene"]["cleaning_strategy"]["status"], "available"
        )
        self.assertEqual(body["data_hygiene"]["manifest"]["status"], "available")
        self.assertEqual(
            body["data_hygiene"]["manifest"]["source_report"],
            "data_hygiene_inventory_latest.json",
        )
        self.assertEqual(body["data_hygiene"]["manifest"]["official_entries"], 2)
        self.assertEqual(body["data_hygiene"]["readme"]["status"], "available")
        self.assertNotIn(
            str(temp_dir), json.dumps(body["data_hygiene"], ensure_ascii=False)
        )
        self.assertNotIn(
            str(projects_root), json.dumps(body["data_hygiene"], ensure_ascii=False)
        )
        self.assertEqual(body["qlib_research"]["status"], "available")
        self.assertEqual(body["qlib_research"]["available_sections"], 1)
        self.assertEqual(
            body["qlib_research"]["sections"]["model_explainability"]["artifacts"][
                "latest_json"
            ],
            "qlib_research_model_explainability_latest.json",
        )
        self.assertEqual(
            body["qlib_research"]["sections"]["risk_model"]["status"], "missing"
        )
        self.assertNotIn(
            str(temp_dir), json.dumps(body["qlib_research"], ensure_ascii=False)
        )
        matrix_by_id = {item["id"]: item for item in body["capability_matrix"]}
        self.assertEqual(matrix_by_id["dsa_web_core"]["status"], "available")
        self.assertEqual(matrix_by_id["quant_workflow_pipeline"]["route"], "/workflow")
        self.assertEqual(matrix_by_id["qlib_integration"]["route"], "/backtest")
        self.assertEqual(
            matrix_by_id["china_astock_quant_reuse"]["status"], "available"
        )
        self.assertEqual(matrix_by_id["china_astock_quant_reuse"]["route"], "/workflow")
        self.assertEqual(
            matrix_by_id["external_tools_tonghuashun"]["status"], "read_only"
        )
        self.assertEqual(body["external_tools"]["tonghuashun"]["path"], str(app_path))
        self.assertEqual(matrix_by_id["trading_execution"]["status"], "not_planned")
        self.assertIn("不自动下单", matrix_by_id["trading_execution"]["restrictions"])
        self.assertIn(
            "不读取券商密码", matrix_by_id["trading_execution"]["restrictions"]
        )
        inventory_by_id = {item["id"]: item for item in body["function_inventory"]}
        self.assertEqual(inventory_by_id["qlib_data_pipeline"]["route"], "/workflow")
        self.assertEqual(
            inventory_by_id["china_astock_reference"]["status"], "available"
        )
        self.assertIn(
            "report_china_astock_indicators",
            inventory_by_id["china_astock_reference"]["verification"],
        )
        self.assertEqual(
            inventory_by_id["quant_workflow_script_coverage"]["route"], "/research"
        )
        self.assertEqual(
            inventory_by_id["quant_workflow_script_coverage"]["status"], "available"
        )
        self.assertIn(
            "report_data_hygiene.py",
            inventory_by_id["quant_workflow_script_coverage"]["coverage"],
        )
        self.assertIn(
            "dry-run cleanup plan",
            inventory_by_id["quant_workflow_script_coverage"]["coverage"],
        )
        self.assertIn(
            "add_index_benchmark.py",
            inventory_by_id["quant_workflow_script_coverage"]["coverage"],
        )
        self.assertIn(
            "generate_csi300.py",
            inventory_by_id["quant_workflow_script_coverage"]["coverage"],
        )
        self.assertIn(
            "强制 dry-run",
            inventory_by_id["quant_workflow_script_coverage"]["coverage"],
        )
        self.assertIn(
            "supervise_baostock.py",
            inventory_by_id["quant_workflow_script_coverage"]["gaps"],
        )
        self.assertIn(
            "不允许网页写入 provider",
            inventory_by_id["quant_workflow_script_coverage"]["gaps"],
        )
        reuse_by_id = {item["id"]: item for item in body["china_astock_reuse"]}
        self.assertEqual(reuse_by_id["indicators"]["priority"], "P1")
        self.assertEqual(reuse_by_id["indicators"]["status"], "available")
        self.assertIn("工作流指标报告", reuse_by_id["indicators"]["target"])
        self.assertEqual(reuse_by_id["paper_trading"]["status"], "not_planned")
        self.assertIn("不自动下单", reuse_by_id["paper_trading"]["restrictions"])
        acceptance_by_id = {item["id"]: item for item in body["acceptance_checklist"]}
        self.assertEqual(acceptance_by_id["project_hygiene"]["status"], "partial")
        self.assertIn(
            "无散落未跟踪源码", acceptance_by_id["project_hygiene"]["evidence"]
        )
        self.assertIn(
            "未提交的正式改动", acceptance_by_id["project_hygiene"]["evidence"]
        )
        self.assertNotIn("已暂存", acceptance_by_id["project_hygiene"]["evidence"])
        self.assertEqual(acceptance_by_id["qlib_web_flow"]["route"], "/workflow")
        self.assertIn("MLflow", acceptance_by_id["qlib_web_flow"]["evidence"])
        self.assertIn(
            "强制 dry-run", acceptance_by_id["quant_workflow_wrapped"]["evidence"]
        )
        self.assertIn("二次确认", acceptance_by_id["quant_workflow_wrapped"]["next"])
        self.assertEqual(acceptance_by_id["opencode_review"]["status"], "verified")
        self.assertIn("opencode", acceptance_by_id["opencode_review"]["evidence"])

    def test_experiment_file_guard_uses_shared_cross_platform_lock_adapter(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _experiment_file_guard

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            with (
                patch(
                    "api.v1.endpoints.research_status.lock_file",
                    create=True,
                ) as lock_file,
                patch(
                    "api.v1.endpoints.research_status.unlock_file",
                    create=True,
                ) as unlock_file,
            ):
                with _experiment_file_guard(reports_dir):
                    self.assertTrue((reports_dir / ".experiment.lock").exists())

        lock_file.assert_called_once()
        lock_handle = lock_file.call_args.args[0]
        self.assertEqual(lock_file.call_args.kwargs, {"exclusive": True})
        self.assertIn("b", lock_handle.mode)
        unlock_file.assert_called_once_with(lock_handle)

    def test_bounded_file_read_supports_platforms_without_fd_capabilities(self) -> None:
        from api.v1.endpoints import research_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            target = nested / "status.json"
            target.write_bytes(b'{"status":"ok"}')
            with (
                patch.object(research_status.os, "supports_dir_fd", set()),
                patch.object(research_status.os, "supports_fd", set()),
            ):
                content = research_status._read_bounded_regular_file_at(
                    root,
                    Path("nested/status.json"),
                    max_bytes=1024,
                )

        self.assertEqual(content, b'{"status":"ok"}')

    def test_bounded_file_read_rejects_symlink_on_fallback_platform_path(self) -> None:
        from api.v1.endpoints import research_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outside = root / "outside.json"
            outside.write_bytes(b"secret")
            link = root / "status.json"
            try:
                link.symlink_to(outside)
            except OSError as exc:  # pragma: no cover - platform privilege fallback.
                self.skipTest(f"symlink unavailable: {exc}")
            with (
                patch.object(research_status.os, "supports_dir_fd", set()),
                patch.object(research_status.os, "supports_fd", set()),
            ):
                content = research_status._read_bounded_regular_file_at(
                    root,
                    Path("status.json"),
                    max_bytes=1024,
                )

        self.assertIsNone(content)

    def test_status_keeps_data_entry_separate_from_latest_training_provider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            cleaned_provider = Path(temp_dir) / "cn_data_baostock_jqdata_cleaned"
            (cleaned_provider / "calendars").mkdir(parents=True)
            (cleaned_provider / "instruments").mkdir(parents=True)
            (cleaned_provider / "features" / "SH600000").mkdir(parents=True)
            (cleaned_provider / "calendars" / "day.txt").write_text(
                "2008-01-02\n2026-07-03\n",
                encoding="utf-8",
            )
            (cleaned_provider / "instruments" / "all.txt").write_text(
                "SH600000\t2008-01-02\t2026-07-03\nSZ000001\t2008-01-02\t2026-07-03\n",
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
                        "latest_training": {
                            "status": "completed",
                            "experiment_name": "baseline_lgb_alpha158",
                            "run_id": "old-run",
                            "provider": "canonical",
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-04T13:51:55+08:00",
                        "status": "completed",
                        "experiment_name": "baseline_lgb_alpha158_cleaned",
                        "run_id": "new-cleaned-run",
                        "provider": "cleaned",
                        "provider_uri": str(cleaned_provider),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch(
                    "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                    cleaned_provider.parent,
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "new-cleaned-run")
        self.assertEqual(body["latest_training"]["provider"], "cleaned")
        self.assertEqual(body["qlib_provider"]["path"], "/tmp/stale-cn-data")
        self.assertEqual(body["latest_training_provider"]["status"], "blocked")
        self.assertNotIn("path", body["latest_training_provider"])
        self.assertFalse(body["latest_training_provider"]["provider_current"])
        self.assertFalse(body["latest_training_provider"]["provider_trusted"])
        self.assertIn(
            "完整的 Qlib Provider 血缘",
            body["qlib_provider"]["provider_uri_warning"],
        )

    def test_status_does_not_present_legacy_training_provider_as_current_native_v2(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            legacy_provider = data_root / "qlib" / "cn_data_baostock_jqdata"
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-26T19:10:50+08:00",
                        "qlib_provider": {"path": str(legacy_provider)},
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "latest_training_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "legacy-run",
                        "provider": "canonical",
                        "provider_uri": str(legacy_provider),
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "candidate_pipeline_latest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "provider_uri": str(legacy_provider),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                    data_root / "qlib",
                ),
            ):
                payload = build_research_status_payload(
                    provider_dataset_id="qlib_ashare_v2"
                )

        self.assertEqual(
            Path(payload["qlib_provider"]["path"]).resolve(), provider.resolve()
        )
        self.assertEqual(payload["qlib_provider"]["dataset_id"], "qlib_ashare_v2")
        self.assertEqual(payload["qlib_provider"]["provider_source"], "Qlib Provider")
        self.assertFalse(payload["qlib_provider"]["latest_training_provider_current"])
        self.assertEqual(payload["latest_training"]["provider_status"], "stale")
        self.assertFalse(payload["latest_training"]["provider_current"])
        self.assertEqual(payload["candidate_pipeline"]["provider_status"], "stale")
        self.assertFalse(payload["candidate_pipeline"]["provider_current"])
        self.assertIn(
            "完整的 Qlib Provider 血缘",
            payload["qlib_provider"]["provider_uri_warning"],
        )

    def test_status_fails_closed_when_native_v2_catalog_hash_is_invalid(self) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            manifest_path = data_root / "DATA_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["datasets"]["qlib_ashare_v2"][
                "snapshot_manifest_sha256"
            ] = "broken-hash"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            legacy_provider = data_root / "qlib" / "cn_data_baostock_jqdata"
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"qlib_provider": {"path": str(legacy_provider)}}),
                encoding="utf-8",
            )
            for filename in (
                "latest_training_result.json",
                "candidate_pipeline_latest.json",
            ):
                (reports_dir / filename).write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "provider_uri": str(legacy_provider),
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                    data_root / "qlib",
                ),
            ):
                payload = build_research_status_payload(
                    provider_dataset_id="qlib_ashare_v2"
                )

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])
        self.assertEqual(payload["training_readiness"]["status"], "blocked")
        self.assertEqual(payload["qlib_provider"]["provider_source"], "Qlib Provider")
        self.assertNotEqual(payload["qlib_provider"].get("path"), str(legacy_provider))
        self.assertEqual(
            payload["latest_training"]["provider_status"], "catalog_invalid"
        )
        self.assertFalse(payload["latest_training"]["provider_current"])
        self.assertEqual(
            payload["candidate_pipeline"]["provider_status"], "catalog_invalid"
        )
        self.assertFalse(payload["candidate_pipeline"]["provider_current"])

    def test_status_fails_closed_when_catalog_content_hash_disagrees_with_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            manifest_path = data_root / "DATA_MANIFEST.json"
            catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
            catalog["datasets"]["qlib_ashare_v2"]["content_sha256"] = "b" * 64
            manifest_path.write_text(json.dumps(catalog), encoding="utf-8")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])
        self.assertIn("Qlib Provider", payload["qlib_provider"]["provider_error"])

    def test_status_describes_the_historical_snapshot_used_by_latest_training(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            historical = self._write_native_qlib_fixture(
                data_root,
                snapshot_id="qlib-native-historical",
            )
            historical_manifest_bytes = (
                historical / "DATASET_MANIFEST.json"
            ).read_bytes()
            historical_manifest = json.loads(historical_manifest_bytes.decode("utf-8"))
            current = self._write_native_qlib_fixture(
                data_root,
                snapshot_id="qlib-native-current",
            )
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "latest_training_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "historical-run",
                        "provider_identity": {
                            "provider_contract": "immutable_snapshot_v2",
                            "dataset_id": "qlib_ashare_v2",
                            "snapshot_id": historical.name,
                            "content_sha256": historical_manifest["content_sha256"],
                            "manifest_sha256": hashlib.sha256(
                                historical_manifest_bytes
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.resolve_immutable_provider_identity",
                return_value=historical,
            ):
                payload = self._build_status_for_roots(reports_dir, data_root)

        self.assertEqual(Path(payload["qlib_provider"]["path"]).resolve(), current.resolve())
        self.assertFalse(payload["latest_training"]["provider_current"])
        self.assertEqual(payload["latest_training"]["provider_status"], "stale")
        self.assertEqual(
            Path(payload["latest_training_provider"]["path"]).resolve(),
            historical.resolve(),
        )
        self.assertEqual(
            payload["latest_training_provider"]["snapshot_id"], historical.name
        )
        self.assertTrue(payload["latest_training_provider"]["provider_trusted"])
        self.assertFalse(payload["latest_training_provider"]["provider_current"])

    def test_status_rejects_native_pointer_to_legacy_dataset_root(self) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            legacy_provider = data_root / "qlib" / "cn_data_baostock_jqdata"
            (legacy_provider / "calendars").mkdir(parents=True)
            (legacy_provider / "instruments").mkdir(parents=True)
            (legacy_provider / "features" / "SH600000").mkdir(parents=True)
            pointer_path = data_root / "qlib" / "cn_data_ashare_v2" / "LATEST.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["snapshot_path"] = "qlib/cn_data_baostock_jqdata"
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps({"qlib_provider": {"path": str(legacy_provider)}}),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_WORKFLOW_REPORTS_ROOT",
                    reports_dir,
                ),
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                    data_root / "qlib",
                ),
            ):
                payload = build_research_status_payload(
                    provider_dataset_id="qlib_ashare_v2"
                )

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertNotEqual(payload["qlib_provider"].get("path"), str(legacy_provider))
        self.assertIn("Qlib Provider", payload["qlib_provider"]["provider_error"])

    def test_status_blocks_native_snapshot_when_an_indexed_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            (provider / "features" / "sh600000" / "close.day.bin").unlink()

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])
        self.assertEqual(payload["training_readiness"]["status"], "blocked")

    def test_status_blocks_same_size_native_feature_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            feature = provider / "features" / "sh600000" / "close.day.bin"
            self.assertEqual(len(feature.read_bytes()), len(b"tampered-data"))
            feature.write_bytes(b"tampered-data")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])

    def test_status_blocks_native_snapshot_with_a_symlinked_child_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            outside_features = root / "outside-features"
            shutil.copytree(provider / "features", outside_features)
            shutil.rmtree(provider / "features")
            try:
                (provider / "features").symlink_to(
                    outside_features, target_is_directory=True
                )
            except OSError as exc:  # pragma: no cover - platform privilege fallback.
                self.skipTest(f"symlink unavailable: {exc}")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])

    def test_path_alias_without_frozen_identity_is_not_current_native_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            data_root = root / "quant_data"
            provider = self._write_native_qlib_fixture(data_root)
            alias = data_root / "qlib" / "legacy-provider-alias"
            try:
                alias.symlink_to(provider, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - platform privilege fallback.
                self.skipTest(f"symlink unavailable: {exc}")
            reports_dir.mkdir()
            for filename in (
                "latest_training_result.json",
                "candidate_pipeline_latest.json",
            ):
                (reports_dir / filename).write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "provider_uri": str(alias),
                        }
                    ),
                    encoding="utf-8",
                )

            payload = self._build_status_for_roots(reports_dir, data_root)

        self.assertFalse(payload["latest_training"]["provider_current"])
        self.assertEqual(payload["latest_training"]["provider_status"], "unknown")
        self.assertFalse(payload["candidate_pipeline"]["provider_current"])
        self.assertEqual(payload["candidate_pipeline"]["provider_status"], "unknown")

    def test_status_blocks_invalid_utf8_native_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)
            (data_root / "DATA_MANIFEST.json").write_bytes(b"\xff\xfe\x00")

            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])

    def test_status_blocks_when_provider_facade_rejects_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)

            with patch(
                "api.v1.endpoints.research_status.qlib_workflow_provider_uri",
                side_effect=RuntimeError("provider integrity changed"),
            ):
                payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "blocked")
        self.assertFalse(payload["trainable"])

    def test_status_leaves_physical_snapshot_validation_to_provider_facade(self) -> None:
        from api.v1.endpoints import research_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "quant_data"
            self._write_native_qlib_fixture(data_root)

            self.assertFalse(
                hasattr(research_status, "_validate_native_qlib_snapshot_cached")
            )
            payload = self._build_status_for_roots(root / "reports", data_root)

        self.assertEqual(payload["qlib_provider"]["status"], "ready")

    def test_directory_size_cache_avoids_repeated_provider_scans_until_ttl(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            _clear_directory_size_cache,
            _directory_size_mb,
        )

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                (provider / "first.bin").write_bytes(b"0" * 1024 * 1024)

                with patch(
                    "api.v1.endpoints.research_status.time.monotonic",
                    side_effect=[0.0, 0.0, 0.0, 10.0, 61.0, 61.0, 61.0],
                ):
                    first = _directory_size_mb(provider)
                    (provider / "second.bin").write_bytes(b"1" * 1024 * 1024)
                    cached = _directory_size_mb(provider)
                    refreshed = _directory_size_mb(provider)
        finally:
            _clear_directory_size_cache()

        self.assertEqual(first, 1.0)
        self.assertEqual(cached, 1.0)
        self.assertEqual(refreshed, 2.0)

    def test_directory_size_scan_returns_unknown_after_time_budget(self) -> None:
        from api.v1.endpoints.research_status import (
            _clear_directory_size_cache,
            _directory_size_mb,
        )

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                for index in range(100):
                    (provider / f"{index:03d}.bin").write_bytes(b"0")

                with (
                    patch(
                        "api.v1.endpoints.research_status.DIRECTORY_SIZE_SCAN_MAX_SECONDS",
                        0.1,
                    ),
                    patch(
                        "api.v1.endpoints.research_status.time.monotonic",
                        side_effect=[0.0, 0.0, 0.2],
                    ),
                ):
                    self.assertIsNone(_directory_size_mb(provider))
        finally:
            _clear_directory_size_cache()

    def test_directory_size_scan_returns_unknown_after_entry_limit(self) -> None:
        from api.v1.endpoints.research_status import (
            _clear_directory_size_cache,
            _directory_size_mb,
        )

        _clear_directory_size_cache()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "provider"
                provider.mkdir()
                (provider / "first.bin").write_bytes(b"0")
                (provider / "second.bin").write_bytes(b"1")

                with patch(
                    "api.v1.endpoints.research_status.DIRECTORY_SIZE_SCAN_MAX_ENTRIES",
                    1,
                ):
                    self.assertIsNone(_directory_size_mb(provider))
        finally:
            _clear_directory_size_cache()

    def test_data_hygiene_cleanup_plan_reports_truncated_action_preview(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            quant_data_root = root / "quant_data"
            quant_data_root.mkdir()
            (reports_dir / "data_hygiene_inventory_latest.json").write_text(
                json.dumps({"generated_at": "2026-07-07T00:00:00+08:00", "items": []}),
                encoding="utf-8",
            )
            (reports_dir / "data_hygiene_cleanup_plan_latest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-07T00:00:00+08:00",
                        "dry_run_only": True,
                        "actions": [
                            {"action": "archive", "source": f"/tmp/item_{idx}"}
                            for idx in range(55)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["cleanup_plan"]["total_actions"], 55)
        self.assertEqual(len(payload["cleanup_plan"]["actions"]), 50)
        self.assertTrue(payload["cleanup_plan"]["truncated"])
        self.assertEqual(payload["cleanup_plan"]["preview_limit"], 50)

    def test_data_hygiene_manifest_counts_list_official_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            (quant_data_root / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-07T00:00:00+08:00",
                        "source_report": str(
                            reports_dir / "data_hygiene_inventory_latest.json"
                        ),
                        "official_entries": [
                            {"path": "/tmp/cleaned_parquet"},
                            {"path": "/tmp/cleaned_provider"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root
            ):
                from api.v1.endpoints.research_status import _data_hygiene_payload

                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["manifest"]["status"], "available")
        self.assertEqual(payload["manifest"]["official_entries"], 2)

    def test_data_hygiene_manifest_rejects_unverified_canonical_ready_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            manifest_path = quant_data_root / "DATA_MANIFEST.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "path_format": "relative_to_quant_data_root",
                        "official_entries": {"legacy_pointer": {}},
                        "datasets": {
                            dataset_id: {
                                "dataset_id": dataset_id,
                                "role": role,
                                "status": status,
                            }
                            for dataset_id, role, status in (
                                ("ashare_daily_qfq", "canonical", "ready"),
                                ("qlib_ashare_v2", "canonical", "ready"),
                                ("unverified_canonical", "canonical", "ready"),
                                ("pending_canonical", "canonical", "building"),
                                ("legacy_compatibility", "legacy", "ready"),
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            ashare_snapshot = quant_data_root / "snapshots" / "ashare"
            qlib_snapshot = quant_data_root / "snapshots" / "qlib"
            for snapshot, content in (
                (ashare_snapshot, b"ashare"),
                (qlib_snapshot, b"qlib"),
            ):
                snapshot.mkdir(parents=True)
                (snapshot / "data.bin").write_bytes(content)
            inventory = {
                "catalog": {
                    "manifest_sha256": manifest_sha256,
                    "validation_contract_version": 2,
                    "validated_dataset_ids": [
                        "ashare_daily_qfq",
                        "qlib_ashare_v2",
                    ],
                    "datasets": [
                        {
                            "dataset_id": "ashare_daily_qfq",
                            "canonical_valid": True,
                            "snapshot_path": str(ashare_snapshot),
                            "state_fingerprint": filesystem_state_fingerprint(
                                ashare_snapshot
                            ),
                        },
                        {
                            "dataset_id": "qlib_ashare_v2",
                            "canonical_valid": True,
                            "snapshot_path": str(qlib_snapshot),
                            "state_fingerprint": filesystem_state_fingerprint(
                                qlib_snapshot
                            ),
                        },
                        {
                            "dataset_id": "unverified_canonical",
                            "canonical_valid": False,
                        },
                    ],
                },
                "items": [
                    {"category": "official", "dataset_id": "ashare_daily_qfq"},
                    {"category": "official", "dataset_id": "qlib_ashare_v2"},
                    {"category": "legacy", "dataset_id": "legacy_compatibility"},
                ],
            }
            self._write_hygiene_bundle(reports_dir, inventory)

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root
            ):
                from api.v1.endpoints.research_status import _data_hygiene_payload

                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["manifest"]["status"], "unverified")
        self.assertEqual(payload["manifest"]["official_entries"], 0)
        self.assertEqual(payload["manifest"]["validation_status"], "unverified")
        self.assertEqual(payload["inventory"]["status"], "unverified")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_manifest_rejects_stale_inventory_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            quant_data_root = Path(temp_dir) / "quant_data"
            quant_data_root.mkdir()
            manifest_path = quant_data_root / "DATA_MANIFEST.json"
            original_manifest = {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {
                    "ashare_daily_qfq": {
                        "dataset_id": "ashare_daily_qfq",
                        "role": "canonical",
                        "status": "ready",
                    }
                },
            }
            manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
            original_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (reports_dir / "data_hygiene_inventory_latest.json").write_text(
                json.dumps(
                    {
                        "catalog": {
                            "manifest_sha256": original_hash,
                            "validated_dataset_ids": ["ashare_daily_qfq"],
                            "datasets": [
                                {
                                    "dataset_id": "ashare_daily_qfq",
                                    "canonical_valid": True,
                                }
                            ],
                        },
                        "items": [
                            {"category": "official", "dataset_id": "ashare_daily_qfq"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        **original_manifest,
                        "datasets": {
                            "replacement": {
                                "dataset_id": "replacement",
                                "role": "canonical",
                                "status": "ready",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT", quant_data_root
            ):
                from api.v1.endpoints.research_status import _data_hygiene_payload

                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["manifest"]["official_entries"], 0)
        self.assertEqual(payload["manifest"]["validation_status"], "stale")
        self.assertEqual(payload["manifest"]["status"], "stale")
        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["validation_status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_inventory_after_snapshot_file_is_deleted(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            (provider / "features" / "sh600000" / "close.day.bin").unlink()

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["manifest"]["validation_status"], "stale")
        self.assertEqual(payload["manifest"]["status"], "stale")
        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_inventory_after_root_manifest_is_deleted(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            (quant_data_root / "DATA_MANIFEST.json").unlink()

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_inventory_after_latest_pointer_changes(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            latest = provider.parents[1] / "LATEST.json"
            pointer = json.loads(latest.read_text(encoding="utf-8"))
            pointer["snapshot_id"] = "missing-snapshot"
            pointer["snapshot_path"] = (
                "qlib/cn_data_ashare_v2/snapshots/missing-snapshot"
            )
            latest.write_text(json.dumps(pointer), encoding="utf-8")

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_detects_equal_size_content_change_with_restored_mtime(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            feature = provider / "features" / "sh600000" / "close.day.bin"
            before = feature.stat()
            content = feature.read_bytes()
            feature.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            os.utime(
                feature,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_windows_data_hygiene_rechecks_bytes_when_metadata_is_unchanged(
        self,
    ) -> None:
        from api.v1.endpoints import research_status

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir, data_root, provider = self._write_native_hygiene_inventory(
                Path(temp_dir)
            )
            frozen_state = filesystem_state_fingerprint(provider)
            with (
                patch.object(research_status, "QUANT_DATA_ROOT", data_root),
                patch.object(research_status.sys, "platform", "win32"),
                patch.object(
                    research_status,
                    "_filesystem_state_fingerprint_live",
                    return_value=frozen_state,
                ),
            ):
                initial = research_status._data_hygiene_payload(reports_dir)
                self.assertEqual(initial["inventory"]["status"], "available")
                self.assertEqual(initial["inventory"]["official_entries"], 1)

                feature = provider / "features" / "sh600000" / "close.day.bin"
                before = feature.stat()
                content = feature.read_bytes()
                feature.write_bytes(bytes([content[0] ^ 1]) + content[1:])
                os.utime(feature, ns=(before.st_atime_ns, before.st_mtime_ns))
                changed = research_status._data_hygiene_payload(reports_dir)

        self.assertEqual(changed["inventory"]["status"], "stale")
        self.assertEqual(changed["inventory"]["official_entries"], 0)

    def test_data_hygiene_rechecks_snapshot_on_every_status_read(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                initial = _data_hygiene_payload(reports_dir)
                feature = provider / "features" / "sh600000" / "close.day.bin"
                feature.write_bytes(feature.read_bytes() + b"changed")
                changed = _data_hygiene_payload(reports_dir)

        self.assertEqual(initial["inventory"]["status"], "available")
        self.assertEqual(changed["inventory"]["status"], "stale")
        self.assertEqual(changed["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_snapshot_path_with_symlink_component(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            inventory = self._read_hygiene_inventory(reports_dir)
            linked_snapshot = quant_data_root / "linked_snapshot"
            try:
                linked_snapshot.symlink_to(provider, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not supported on this filesystem: {exc}")
            record = next(
                item
                for item in inventory["catalog"]["datasets"]
                if item.get("dataset_id") == "qlib_ashare_v2"
            )
            record["snapshot_path"] = str(linked_snapshot)
            self._write_hygiene_bundle(reports_dir, inventory)

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_latest_pointer_with_symlink_component(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            inventory = self._read_hygiene_inventory(reports_dir)
            record = next(
                item
                for item in inventory["catalog"]["datasets"]
                if item.get("dataset_id") == "qlib_ashare_v2"
            )
            real_pointer = Path(record["latest_pointer_path"])
            linked_pointer = quant_data_root / "linked_latest.json"
            try:
                linked_pointer.symlink_to(real_pointer)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not supported on this filesystem: {exc}")
            record["latest_pointer_path"] = str(linked_pointer)
            self._write_hygiene_bundle(reports_dir, inventory)

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_rejects_symlinked_root_manifest(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            manifest_path = quant_data_root / "DATA_MANIFEST.json"
            real_manifest = quant_data_root / "DATA_MANIFEST.real.json"
            manifest_path.replace(real_manifest)
            try:
                manifest_path.symlink_to(real_manifest)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symlink not supported on this filesystem: {exc}")

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertNotEqual(payload["manifest"]["status"], "available")
        self.assertNotEqual(payload["manifest"]["validation_status"], "verified")
        self.assertEqual(payload["manifest"]["official_entries"], 0)

    def test_data_hygiene_hides_cleanup_actions_when_manifest_is_stale(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, provider = (
                self._write_native_hygiene_inventory(root)
            )
            inventory = self._read_hygiene_inventory(reports_dir)
            self._write_hygiene_bundle(
                reports_dir,
                inventory,
                cleanup_plan={
                    "generated_at": "2026-07-30T12:00:00+08:00",
                    "dry_run_only": True,
                    "actions": [
                        {
                            "action": "archive",
                            "source": str(provider),
                            "requires_user_confirmation": True,
                        }
                    ],
                },
            )
            feature = provider / "features" / "sh600000" / "close.day.bin"
            feature.write_bytes(feature.read_bytes() + b"changed")

            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["cleanup_plan"]["status"], "stale")
        self.assertEqual(payload["cleanup_plan"]["total_actions"], 0)
        self.assertEqual(payload["cleanup_plan"]["actions"], [])

    def test_data_hygiene_reads_immutable_bundle_not_compatibility_latest(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            (reports_dir / "data_hygiene_inventory_latest.json").write_text(
                '{"items":[]}',
                encoding="utf-8",
            )
            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "available")
        self.assertGreater(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_fails_closed_when_v2_bundle_pointer_is_deleted(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            (reports_dir / "data_hygiene_reports_latest.json").unlink()
            with patch(
                "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                quant_data_root,
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_state_scan_fails_closed_at_entry_limit(self) -> None:
        from api.v1.endpoints.research_status import _data_hygiene_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports_dir, quant_data_root, _provider = (
                self._write_native_hygiene_inventory(root)
            )
            with (
                patch(
                    "api.v1.endpoints.research_status.QUANT_DATA_ROOT",
                    quant_data_root,
                ),
                patch(
                    "api.v1.endpoints.research_status.DATA_HYGIENE_STATE_SCAN_MAX_ENTRIES",
                    0,
                ),
            ):
                payload = _data_hygiene_payload(reports_dir)

        self.assertEqual(payload["inventory"]["status"], "stale")
        self.assertEqual(payload["inventory"]["official_entries"], 0)

    def test_data_hygiene_state_scan_waits_for_concurrent_reader(self) -> None:
        import api.v1.endpoints.research_status as module

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "snapshot"
            snapshot.mkdir()
            (snapshot / "data.bin").write_bytes(b"stable")
            first_scan_started = threading.Event()
            release_first_scan = threading.Event()
            second_scan_started = threading.Event()
            second_scan_finished = threading.Event()
            results: list[dict] = []
            errors: list[BaseException] = []
            original_fingerprint = module.filesystem_state_fingerprint
            call_count = 0
            call_count_lock = threading.Lock()

            def controlled_fingerprint(path: Path, **kwargs):
                nonlocal call_count
                with call_count_lock:
                    call_count += 1
                    is_first = call_count == 1
                if is_first:
                    first_scan_started.set()
                    self.assertTrue(release_first_scan.wait(timeout=2))
                return original_fingerprint(path, **kwargs)

            def scan(*, second: bool = False) -> None:
                if second:
                    second_scan_started.set()
                try:
                    results.append(module._filesystem_state_fingerprint_live(snapshot))
                except BaseException as exc:  # noqa: BLE001 - captured for assertion.
                    errors.append(exc)
                finally:
                    if second:
                        second_scan_finished.set()

            with patch.object(
                module,
                "filesystem_state_fingerprint",
                side_effect=controlled_fingerprint,
            ):
                first = threading.Thread(target=scan)
                second = threading.Thread(target=scan, kwargs={"second": True})
                first.start()
                self.assertTrue(first_scan_started.wait(timeout=2))
                second.start()
                self.assertTrue(second_scan_started.wait(timeout=2))
                finished_while_locked = second_scan_finished.wait(timeout=0.1)
                release_first_scan.set()
                first.join(timeout=2)
                second.join(timeout=2)

        self.assertFalse(finished_while_locked)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])


    def test_research_status_route_is_sync_for_fastapi_threadpool(self) -> None:
        from api.v1.endpoints.research_status import get_research_status

        self.assertFalse(inspect.iscoroutinefunction(get_research_status))

    def test_status_uses_lightweight_training_preflight_without_quality_report(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            _NativeQlibProviderResolution,
            build_research_status_payload,
        )

        identities = tuple(
            QuantDataIdentity(
                dataset_id=dataset_id,
                snapshot_id=f"{dataset_id}-snapshot",
                manifest_sha256=marker * 64,
                content_sha256=marker.upper().lower() * 64,
                latest_trade_date="2026-08-07",
                price_basis=price_basis,
                data_kind=data_kind,
            )
            for dataset_id, marker, price_basis, data_kind in (
                ("ashare_daily_qfq", "a", "qfq", "daily_bars"),
                ("cn_index_daily", "b", "raw", "daily_bars"),
                (
                    "pit_csi800",
                    "c",
                    "not_applicable",
                    "point_in_time_membership",
                ),
            )
        )
        readiness = QuantDataReadyResult(
            status="ready",
            validated_datasets=len(identities),
            identities=identities,
            datasets=tuple(
                {**identity.as_dict(), "role": "canonical", "status": "ready"}
                for identity in identities
            ),
        )
        lineage = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "provider-snapshot",
            "manifest_sha256": "d" * 64,
            "content_sha256": "e" * 64,
            "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "f" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-snapshot",
            "index_content_sha256": "1" * 64,
            "index_manifest_sha256": "2" * 64,
        }
        native = _NativeQlibProviderResolution(
            authoritative=True,
            path=Path("/tmp/current-provider"),
            lineage=lineage,
            provider_name="canonical",
            dataset_id="ashare_daily_qfq",
            snapshot_id="provider-snapshot",
            manifest_sha256="d" * 64,
            content_sha256="e" * 64,
            calendar_end="2026-08-07",
            instrument_count=5516,
            benchmark_complete=True,
        )
        service = MagicMock()
        service.__enter__.return_value = service
        service.ready.return_value = readiness

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-08T10:00:00+08:00",
                        "trainable": False,
                        "training_readiness": {
                            "status": "blocked",
                            "summary": "Old DSA quality evidence is stale.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    return_value=native,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(readiness, lineage),
                ) as service_client,
                patch.dict(
                    os.environ,
                    {"QUANT_DATA_SERVICE_URL": "http://localhost:8766"},
                    clear=False,
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                payload = build_research_status_payload()

        self.assertTrue(payload["trainable"])
        self.assertEqual(payload["training_readiness"]["status"], "trainable")
        self.assertEqual(payload["training_preflight"]["status"], "passed")
        self.assertEqual(
            set(payload["training_preflight"]["required_datasets"]),
            {"ashare_daily_qfq", "cn_index_daily", "pit_csi800"},
        )
        self.assertEqual(payload["qlib_provider"]["snapshot_id"], "provider-snapshot")
        service_client.assert_any_call(
            quant_data_root=ANY,
            dataset_ids=("ashare_daily_qfq", "cn_index_daily", "pit_csi800"),
        )
        service.ready.assert_not_called()

    def test_status_blocks_training_when_required_quant_data_dataset_is_missing(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            _NativeQlibProviderResolution,
            _apply_training_preflight,
        )

        identities = tuple(
            QuantDataIdentity(
                dataset_id=dataset_id,
                snapshot_id=f"{dataset_id}-snapshot",
                manifest_sha256=marker * 64,
                content_sha256=marker * 64,
                latest_trade_date="2026-08-07",
                price_basis=price_basis,
                data_kind="daily_bars",
            )
            for dataset_id, marker, price_basis in (
                ("ashare_daily_qfq", "a", "qfq"),
                ("cn_index_daily", "b", "raw"),
            )
        )
        readiness = QuantDataReadyResult(
            status="ready",
            validated_datasets=len(identities),
            identities=identities,
            datasets=tuple(
                {**identity.as_dict(), "role": "canonical", "status": "ready"}
                for identity in identities
            ),
        )
        native = _NativeQlibProviderResolution(
            authoritative=True,
            path=Path("/tmp/current-provider"),
            provider_name="canonical",
        )
        status = {"trainable": True}

        with patch(
            "quant_qmt_data.inspect_datasets",
            return_value=local_training_catalog(readiness),
        ):
            _apply_training_preflight(status, native)

        self.assertFalse(status["trainable"])
        self.assertEqual(status["training_preflight"]["status"], "blocked")
        self.assertIn("pit_csi800", status["training_readiness"]["summary"])

    def test_status_blocks_training_when_quant_data_service_is_unavailable(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import (
            _NativeQlibProviderResolution,
            _apply_training_preflight,
        )

        status = {"trainable": True}
        native = _NativeQlibProviderResolution(
            authoritative=True,
            path=Path("/tmp/current-provider"),
            provider_name="canonical",
        )

        with patch(
            "quant_qmt_data.inspect_datasets",
            side_effect=QuantDataClientError("private service detail"),
        ):
            _apply_training_preflight(status, native)

        self.assertFalse(status["trainable"])
        self.assertEqual(status["training_preflight"]["status"], "blocked")
        self.assertIn("quant_data", status["training_readiness"]["summary"])
        self.assertNotIn("private service detail", status["training_readiness"]["summary"])

    def test_status_blocks_training_when_qlib_provider_is_unreadable(self) -> None:
        from api.v1.endpoints.research_status import (
            _NativeQlibProviderResolution,
            _apply_training_preflight,
        )

        status = {"trainable": True}
        native = _NativeQlibProviderResolution(
            authoritative=True,
            provider_name="canonical",
            error="Qlib Provider 不可用：quant_data 数据后端未通过完整性校验。",
        )

        _apply_training_preflight(status, native)

        self.assertFalse(status["trainable"])
        self.assertEqual(status["training_preflight"]["status"], "blocked")
        self.assertIn("Qlib Provider", status["training_readiness"]["summary"])

    def test_status_reports_independent_preflight_for_each_provider(self) -> None:
        from api.v1.endpoints.research_status import (
            _NativeQlibProviderResolution,
            build_research_status_payload,
        )

        readiness = QuantDataReadyResult(
            status="ready",
            validated_datasets=4,
            identities=tuple(
                QuantDataIdentity(
                    dataset_id=dataset_id,
                    snapshot_id=f"{dataset_id}-snapshot",
                    manifest_sha256=marker * 64,
                    content_sha256=marker * 64,
                    latest_trade_date="2026-08-07",
                    price_basis=price_basis,
                    data_kind=data_kind,
                )
                for dataset_id, marker, price_basis, data_kind in (
                    ("ashare_daily_qfq", "a", "qfq", "daily_bars"),
                    ("cn_index_daily", "b", "raw", "daily_bars"),
                    (
                        "pit_csi800",
                        "c",
                        "not_applicable",
                        "point_in_time_membership",
                    ),
                    (
                        "qlib_cleaning_plan",
                        "d",
                        "not_applicable",
                        "qlib_cleaning_plan",
                    ),
                )
            ),
            datasets=(),
        )
        canonical_lineage = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "canonical-snapshot",
            "manifest_sha256": "e" * 64,
            "content_sha256": "f" * 64,
            "descriptor_path": "runtime/qlib/provider/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "1" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-snapshot",
            "index_content_sha256": "2" * 64,
            "index_manifest_sha256": "3" * 64,
        }
        resolutions = {
            "canonical": _NativeQlibProviderResolution(
                authoritative=True,
                path=Path("/tmp/canonical-provider"),
                lineage=canonical_lineage,
                provider_name="canonical",
                calendar_end="2026-08-07",
            ),
            "cleaned": _NativeQlibProviderResolution(
                authoritative=True,
                path=Path("/tmp/cleaned-provider"),
                provider_name="cleaned",
                calendar_end="2026-08-07",
            ),
        }

        def resolve_provider(
            _dataset_id: str = "ashare_daily_qfq",
            *,
            provider: str = "canonical",
        ) -> _NativeQlibProviderResolution:
            return resolutions[provider]

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text("{}\n", encoding="utf-8")
            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_provider,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(readiness, canonical_lineage),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                payload = build_research_status_payload()

        self.assertTrue(payload["trainable"])
        self.assertEqual(payload["training_preflight"]["provider"], "canonical")
        self.assertEqual(
            payload["training_preflights"]["canonical"]["status"], "passed"
        )
        self.assertEqual(
            payload["training_preflights"]["cleaned"]["status"], "blocked"
        )
        self.assertIn(
            "不可变数据血缘",
            payload["training_preflights"]["cleaned"]["summary"],
        )

    def test_status_discards_historical_quality_fields_and_uses_live_preflight(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir) / "reports"
            reports_dir.mkdir()
            status_path = reports_dir / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-31T10:00:00+08:00",
                        "trainable": True,
                        "quality_report_id": "data-quality-0000000000000000",
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "Previous full scan passed.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            markdown = "# Quality\n\n- Stock universe: 5509\n"
            digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
            artifact = reports_dir / (
                f"data_quality_report_20260731_110000_{digest[:16]}.md"
            )
            artifact.write_text(markdown, encoding="utf-8")
            (reports_dir / "data_quality_report_latest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "report_id": f"data-quality-{digest[:16]}",
                        "publication_contract": (
                            "atomic_json_pointer_to_immutable_markdown_v1"
                        ),
                        "markdown_artifact": artifact.name,
                        "markdown_sha256": digest,
                        "scan_mode": "full",
                        "verification_status": "verified",
                        "trainable": True,
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_ready_provider,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(ready_training_data()),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                payload = build_research_status_payload()

        self.assertTrue(payload["trainable"])
        self.assertEqual(payload["training_readiness"]["status"], "trainable")
        self.assertEqual(payload["training_preflight"]["status"], "passed")
        self.assertIsNone(payload["quality_report_id"])
        self.assertIsNone(payload["quality_evidence_sha256"])
        self.assertEqual(payload["quality_scan"], {})
        self.assertEqual(payload["provider_quality"], {})
        self.assertEqual(payload["data_sources"], [])



    def test_status_allows_training_without_historical_quality_publication(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-31T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "Legacy optimistic state.",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_ready_provider,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(ready_training_data()),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                payload = build_research_status_payload()

        self.assertTrue(payload["trainable"])
        self.assertEqual(payload["training_readiness"]["status"], "trainable")
        self.assertEqual(payload["training_preflight"]["status"], "passed")

    def test_status_replaces_legacy_trainable_values_with_live_preflight(self) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "research_status_latest.json"
            for raw_trainable in ("false", "true", 1, None):
                with self.subTest(raw_trainable=raw_trainable):
                    status_path.write_text(
                        json.dumps(
                            {
                                "generated_at": "2026-07-31T10:00:00+08:00",
                                "trainable": raw_trainable,
                                "training_readiness": {
                                    "status": "trainable",
                                    "summary": "Malformed optimistic state.",
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    with (
                        patch(
                            "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                            status_path,
                        ),
                        patch(
                            "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                            side_effect=resolve_ready_provider,
                        ),
                        patch(
                            "quant_qmt_data.inspect_datasets",
                            return_value=local_training_catalog(ready_training_data()),
                        ),
                        patch(
                            "api.v1.endpoints.research_status._qlib_provider_from_path",
                            return_value={},
                        ),
                    ):
                        payload = build_research_status_payload()

                    self.assertIs(payload["trainable"], True)
                    self.assertEqual(payload["training_preflight"]["status"], "passed")





    def test_status_allows_cleaned_provider_when_live_preflight_passes(self) -> None:
        from api.v1.endpoints.research_status import build_research_status_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-31T10:00:00+08:00",
                        "trainable": False,
                        "training_readiness": {"status": "blocked"},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_ready_provider,
                ) as resolve_native,
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(ready_training_data()),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                payload = build_research_status_payload(
                    provider_dataset_id="qlib_ashare_cleaned_v2",
                    provider="cleaned",
                )

        self.assertEqual(resolve_native.call_count, 2)
        self.assertEqual(
            resolve_native.call_args_list[0].args,
            ("qlib_ashare_cleaned_v2",),
        )
        self.assertEqual(
            resolve_native.call_args_list[0].kwargs,
            {"provider": "cleaned"},
        )
        self.assertEqual(resolve_native.call_args_list[1].args, ())
        self.assertEqual(
            resolve_native.call_args_list[1].kwargs,
            {"provider": "canonical"},
        )
        self.assertTrue(payload["trainable"])
        self.assertEqual(payload["training_preflight"]["provider"], "cleaned")
        self.assertEqual(payload["training_preflight"]["status"], "passed")
        self.assertEqual(
            payload["qlib_provider"]["dataset_id"], "ashare_daily_qfq"
        )



    def test_status_redacts_mlflow_url_credentials_query_and_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            status_path.write_text(
                json.dumps(
                    {"generated_at": "2026-07-02T10:00:00+08:00", "trainable": True}
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch.dict(
                    os.environ,
                    {
                        "MLFLOW_URL": "http://fake_user:fake_pass@127.0.0.1:6106/?token=fake#frag"
                    },
                    clear=False,
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        mlflow = resp.json()["mlflow"]
        self.assertEqual(mlflow["url"], "http://127.0.0.1:6106/")
        self.assertEqual(mlflow["status"], "invalid")
        self.assertEqual(mlflow["role"], "experiment_raw_records")
        self.assertIsNone(mlflow["checked_at"])
        self.assertIn("username or password", mlflow["error"])
        self.assertNotIn("fake_pass", json.dumps(mlflow))
        self.assertNotIn("token=fake", json.dumps(mlflow))

    def test_status_marks_external_mlflow_as_unprobed_link(self) -> None:
        from api.v1.endpoints.research_status import _mlflow_payload

        with patch.dict(
            os.environ,
            {"MLFLOW_URL": "https://mlflow.example.com/experiments?token=fake"},
            clear=False,
        ):
            mlflow = _mlflow_payload()

        self.assertEqual(mlflow["url"], "https://mlflow.example.com/experiments")
        self.assertEqual(mlflow["status"], "linked")
        self.assertFalse(mlflow["health_checked"])
        self.assertIn("not probed", mlflow["error"])
        self.assertNotIn("token=fake", json.dumps(mlflow))

    def test_status_rejects_unsafe_mlflow_url_with_sanitized_display(self) -> None:
        from api.v1.endpoints.research_status import _mlflow_payload

        with patch.dict(
            os.environ,
            {"MLFLOW_URL": "http://127.0.0.1:6106/%2e%2e/admin"},
            clear=False,
        ):
            traversal = _mlflow_payload()
        with patch.dict(os.environ, {"MLFLOW_URL": "file:///tmp/mlflow"}, clear=False):
            non_http = _mlflow_payload()

        self.assertEqual(traversal["url"], "http://127.0.0.1:6106/%2e%2e/admin")
        self.assertEqual(traversal["status"], "invalid")
        self.assertIn("path traversal", traversal["error"])
        self.assertEqual(non_http["url"], "http://127.0.0.1:6006/")
        self.assertEqual(non_http["status"], "invalid")
        self.assertIn("http(s)", non_http["error"])

    def test_status_does_not_scan_untrusted_latest_training_provider_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            untrusted_provider = Path(temp_dir) / "outside" / "cn_data"
            untrusted_provider.mkdir(parents=True)
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "untrusted-provider-run",
                        "provider_uri": str(untrusted_provider),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch(
                    "api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT",
                    Path(temp_dir) / "trusted_qlib_root",
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    side_effect=AssertionError(
                        "untrusted provider must not be scanned"
                    ),
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "untrusted-provider-run")
        self.assertEqual(body["qlib_provider"]["path"], "/tmp/stale-cn-data")
        self.assertFalse(body["qlib_provider"]["latest_training_provider_trusted"])
        self.assertIn(
            "完整的 Qlib Provider 血缘",
            body["qlib_provider"]["provider_uri_warning"],
        )

    def test_status_does_not_scan_qlib_root_as_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            qlib_root = Path(temp_dir) / "quant_data" / "qlib"
            qlib_root.mkdir(parents=True)
            status_path = Path(temp_dir) / "research_status_latest.json"
            latest_training_path = Path(temp_dir) / "latest_training_result.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                        "qlib_provider": {
                            "path": "/tmp/stale-cn-data",
                            "calendar_days": 1,
                            "calendar_range": "2020-01-02 ~ 2020-01-02",
                            "instruments": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            latest_training_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "run_id": "root-provider-run",
                        "provider_uri": str(qlib_root),
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch("api.v1.endpoints.research_status.QLIB_PROVIDER_ROOT", qlib_root),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    side_effect=AssertionError(
                        "qlib root must not be scanned as a provider"
                    ),
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["latest_training"]["run_id"], "root-provider-run")
        self.assertFalse(body["qlib_provider"]["latest_training_provider_trusted"])
        self.assertIn(
            "完整的 Qlib Provider 血缘",
            body["qlib_provider"]["provider_uri_warning"],
        )

    def test_status_includes_latest_cash_control_backtest_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            cash_control_path = Path(temp_dir) / "cash_control_backtest_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            cash_control_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T09:00:00+08:00",
                        "mode": "read_only_cash_control_overlay",
                        "comparison": {
                            "baseline": {
                                "cumulative_return": 0.20,
                                "max_drawdown": -0.30,
                            },
                            "cash_control": {
                                "cumulative_return": 0.12,
                                "max_drawdown": -0.12,
                            },
                            "delta": {"cumulative_return": -0.08, "max_drawdown": 0.18},
                        },
                        "signal_summary": {
                            "latest_regime": "risk_off",
                            "latest_cash_ratio": 0.80,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(
            body["cash_control_backtest"]["mode"], "read_only_cash_control_overlay"
        )
        self.assertEqual(
            body["cash_control_backtest"]["signal_summary"]["latest_regime"], "risk_off"
        )
        self.assertEqual(
            body["cash_control_backtest"]["comparison"]["cash_control"]["max_drawdown"],
            -0.12,
        )

    def test_status_includes_latest_qlib_prediction_summary_when_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            predictions_path = Path(temp_dir) / "qlib_predictions_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-05T10:00:00+08:00",
                        "mode": "read_only_qlib_prediction_summary",
                        "run_id": "run_1",
                        "latest_date": "2026-07-02",
                        "top": [{"instrument": "SZ000002", "score": 0.25}],
                        "bottom": [{"instrument": "SZ000001", "score": -0.40}],
                        "summary": {"latest_rows": 3},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(
            body["qlib_predictions"]["mode"], "read_only_qlib_prediction_summary"
        )
        self.assertEqual(body["qlib_predictions"]["latest_date"], "2026-07-02")
        self.assertEqual(body["qlib_predictions"]["top"][0]["instrument"], "SZ000002")
        self.assertEqual(body["qlib_predictions"]["bottom"][0]["score"], -0.40)

    def test_status_includes_latest_candidate_pipeline_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            candidate_path = Path(temp_dir) / "candidate_pipeline_latest.json"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-15T18:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "batch_id": "candidate-alpha158-20260715_174713",
                        "provider_uri": "/private/provider",
                        "selected_models": ["catboost", "xgboost"],
                        "selected_combination": {
                            "model_id": "xgboost",
                            "strategy_id": "low_turnover_weekly",
                            "holdout": {
                                "annual_excess_return": 0.20,
                                "information_ratio": 1.5,
                                "max_drawdown": -0.12,
                            },
                        },
                        "holdout_verdict": {"status": "passed_research_gate"},
                        "models": [
                            {"id": "xgboost", "status": "completed"},
                            {
                                "id": "catboost",
                                "status": "failed",
                                "error": (
                                    "FileNotFoundError: /Users/cq/private/provider/model.bin; "
                                    "fallback=C:\\Users\\cq\\private\\model.bin"
                                ),
                            },
                        ],
                        "strategy_evaluations": [],
                        "artifacts": {"json": "/private/candidate.json"},
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()["candidate_pipeline"]
        self.assertEqual(body["selected_models"], ["catboost", "xgboost"])
        self.assertEqual(body["selected_combination"]["model_id"], "xgboost")
        self.assertEqual(body["holdout_verdict"]["status"], "passed_research_gate")
        self.assertEqual(
            body["models"][1]["error"],
            "FileNotFoundError: [path] fallback=[path]",
        )
        self.assertNotIn("/private/", str(body))
        self.assertNotIn("/Users/", str(body))
        self.assertNotIn("C:\\Users\\", str(body))

    def test_candidate_pipeline_ignores_legacy_dry_run_latest_pointer(self) -> None:
        from api.v1.endpoints.research_status import _candidate_pipeline_payload

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            (reports_dir / "candidate_pipeline_latest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "dry_run": True,
                        "models": [{"id": "lightgbm", "status": "completed"}],
                        "strategy_evaluations": [
                            {
                                "model_id": "lightgbm",
                                "strategy_id": "baseline_daily",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = _candidate_pipeline_payload(reports_dir)

        self.assertIsNone(payload)

    def test_status_returns_missing_payload_when_artifact_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            missing_path = Path(temp_dir) / "missing.json"

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    missing_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._resolve_native_qlib_provider",
                    side_effect=resolve_ready_provider,
                ),
                patch(
                    "quant_qmt_data.inspect_datasets",
                    return_value=local_training_catalog(ready_training_data()),
                ),
                patch(
                    "api.v1.endpoints.research_status._qlib_provider_from_path",
                    return_value={},
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["trainable"])
        self.assertEqual(body["training_readiness"]["status"], "trainable")
        self.assertEqual(body["training_preflight"]["status"], "passed")
        self.assertEqual(body["data_sources"], [])
        self.assertEqual(body["experiment_notebook"]["status"], "missing")
        self.assertEqual(body["experiment_notebook"]["runs"], [])
        self.assertIsNone(body["qlib_predictions"])
        self.assertIn("capability_matrix", body)
        self.assertIn("function_inventory", body)
        self.assertIn("china_astock_reuse", body)
        self.assertIn("acceptance_checklist", body)
        self.assertEqual(
            {item["id"] for item in body["capability_matrix"]},
            {
                "dsa_web_core",
                "quant_workflow_pipeline",
                "qlib_integration",
                "china_astock_quant_reuse",
                "external_tools_tonghuashun",
                "trading_execution",
            },
        )
        self.assertIn(
            "quant_workflow_jobs", {item["id"] for item in body["function_inventory"]}
        )
        self.assertIn(
            "event_backtest", {item["id"] for item in body["china_astock_reuse"]}
        )
        self.assertIn(
            "self_service_docs", {item["id"] for item in body["acceptance_checklist"]}
        )

    def test_status_includes_read_only_tonghuashun_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            app_path = Path(temp_dir) / "同花顺.app"
            app_path.mkdir()
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch(
                    "api.v1.endpoints.research_status.TONGHUASHUN_CANDIDATE_PATHS",
                    (app_path,),
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        tonghuashun = body["external_tools"]["tonghuashun"]
        self.assertTrue(tonghuashun["installed"])
        self.assertEqual(tonghuashun["status"], "available")
        self.assertEqual(tonghuashun["mode"], "read_only")
        self.assertEqual(tonghuashun["path"], str(app_path))
        self.assertIn("不自动下单", tonghuashun["restrictions"])
        self.assertEqual(
            body["external_tools"]["iphone_mirroring"]["status"], "deferred"
        )
        matrix_by_id = {item["id"]: item for item in body["capability_matrix"]}
        self.assertEqual(
            matrix_by_id["external_tools_tonghuashun"]["status"], "read_only"
        )
        self.assertIn(
            "本机应用存在检测", matrix_by_id["external_tools_tonghuashun"]["browser"]
        )
        self.assertIn(
            "不连接券商交易接口",
            matrix_by_id["external_tools_tonghuashun"]["restrictions"],
        )

    def test_capability_matrix_ignores_non_string_external_tool_restrictions(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _capability_matrix_payload

        matrix = _capability_matrix_payload(
            {
                "tonghuashun": {
                    "status": "available",
                    "installed": True,
                    "summary": "detected",
                    "capabilities": ["本机应用存在检测", {"unexpected": "object"}],
                    "restrictions": ["不自动下单", {"unexpected": "object"}, ""],
                }
            }
        )

        tonghuashun = {item["id"]: item for item in matrix}[
            "external_tools_tonghuashun"
        ]
        self.assertEqual(tonghuashun["browser"], "detected 能力：本机应用存在检测")
        self.assertEqual(tonghuashun["restrictions"], ["不自动下单"])

    def test_experiment_notebook_route_does_not_build_full_research_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text("{}", encoding="utf-8")
            (root / "experiment_runs.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "fast-notebook-run",
                        "experiment_name": "fast_notebook",
                        "status": "completed",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                status_path,
            ), patch(
                "api.v1.endpoints.research_status.build_research_status_payload",
                side_effect=AssertionError("full status must not be built"),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                response = client.get("/api/v1/research/experiments")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "available")
        self.assertEqual(body["total_runs"], 1)
        self.assertEqual(body["runs"][0]["run_id"], "fast-notebook-run")

    def test_experiment_notebook_reader_acquires_the_shared_file_guard(self) -> None:
        from api.v1.endpoints import research_status

        events: list[str] = []

        class RecordingGuard:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append("exit")

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            with patch.object(
                research_status,
                "_experiment_file_guard",
                side_effect=lambda _reports_dir: RecordingGuard(),
            ):
                payload = research_status._experiment_notebook_payload(reports_dir)

        self.assertEqual(payload["status"], "missing")
        self.assertEqual(events, ["enter", "exit"])

    def test_archive_route_uses_unlocked_reader_while_holding_shared_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text("{}", encoding="utf-8")
            (root / "experiment_runs.jsonl").write_text(
                json.dumps({"run_id": "archive-me"}) + "\n",
                encoding="utf-8",
            )
            notebook = {
                "status": "missing",
                "latest_note": "",
                "runs": [],
                "total_runs": 0,
                "invalid_lines": 0,
                "log_path": "experiment_log.md",
                "runs_path": "experiment_runs.jsonl",
            }

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._experiment_notebook_payload",
                    side_effect=AssertionError("nested experiment guard"),
                ),
                patch(
                    "api.v1.endpoints.research_status._experiment_notebook_payload_unlocked",
                    return_value=notebook,
                    create=True,
                ) as read_unlocked,
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                    raise_server_exceptions=False,
                )
                response = client.post(
                    "/api/v1/research/experiments/archive-me/archive"
                )

        self.assertEqual(response.status_code, 200)
        read_unlocked.assert_called_once_with(root)

    def test_permanent_delete_route_uses_unlocked_reader_while_holding_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            static_dir = root / "static"
            static_dir.mkdir()
            status_path = root / "research_status_latest.json"
            status_path.write_text("{}", encoding="utf-8")
            (root / "experiment_runs.jsonl").write_text(
                json.dumps({"run_id": "delete-me"}) + "\n",
                encoding="utf-8",
            )
            (root / "experiment_log.md").write_text(
                "<!-- experiment-run:delete-me -->\n"
                "delete me\n"
                "<!-- /experiment-run:delete-me -->\n",
                encoding="utf-8",
            )
            notebook = {
                "status": "missing",
                "latest_note": "",
                "runs": [],
                "total_runs": 0,
                "invalid_lines": 0,
                "log_path": "experiment_log.md",
                "runs_path": "experiment_runs.jsonl",
            }

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH",
                    status_path,
                ),
                patch(
                    "api.v1.endpoints.research_status._experiment_notebook_payload",
                    side_effect=AssertionError("nested experiment guard"),
                ),
                patch(
                    "api.v1.endpoints.research_status._experiment_notebook_payload_unlocked",
                    return_value=notebook,
                    create=True,
                ) as read_unlocked,
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                    raise_server_exceptions=False,
                )
                response = client.delete(
                    "/api/v1/research/experiments/delete-me/permanent?confirm=permanent"
                )

        self.assertEqual(response.status_code, 200)
        read_unlocked.assert_called_once_with(root)

    def test_archive_experiment_run_rewrites_active_log_and_archives_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            deleted_runs_path = Path(temp_dir) / "experiment_runs_deleted.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "abc123",
                                "experiment_name": "baseline_lgb_alpha158",
                                "status": "completed",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "run_id": "other456",
                                "experiment_name": "risk_overlay",
                                "status": "completed",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.post("/api/v1/research/experiments/abc123/archive")

            active_lines = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
            ]
            deleted_lines = [
                json.loads(line)
                for line in deleted_runs_path.read_text(encoding="utf-8").splitlines()
            ]
            lock_exists = (Path(temp_dir) / ".experiment.lock").exists()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(lock_exists)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual(body["experiment_notebook"]["runs"][0]["run_id"], "other456")

        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])

        self.assertEqual(deleted_lines[0]["run_id"], "abc123")
        self.assertEqual(deleted_lines[0]["deleted_from"], "experiment_runs.jsonl")
        self.assertIn("deleted_at", deleted_lines[0])

    def test_archive_experiment_run_restores_active_record_when_archive_write_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            deleted_runs_path = Path(temp_dir) / "experiment_runs_deleted.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_text = (
                json.dumps(
                    {"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"},
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps(
                    {"run_id": "other456", "experiment_name": "risk_overlay"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            runs_path.write_text(runs_text, encoding="utf-8")
            deleted_runs_path.mkdir()

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.post("/api/v1/research/experiments/abc123/archive")

            persisted_runs = runs_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 500)
        self.assertEqual(persisted_runs, runs_text)

    def test_archive_experiment_run_restores_snapshots_when_http_exception_is_raised(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _archive_experiment_run

        with tempfile.TemporaryDirectory() as temp_dir:
            reports_dir = Path(temp_dir)
            runs_path = reports_dir / "experiment_runs.jsonl"
            deleted_runs_path = reports_dir / "experiment_runs_deleted.jsonl"
            runs_text = (
                json.dumps(
                    {"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"}
                )
                + "\n"
            )
            deleted_runs_text = (
                json.dumps(
                    {"run_id": "old999", "deleted_from": "experiment_runs.jsonl"}
                )
                + "\n"
            )
            runs_path.write_text(runs_text, encoding="utf-8")
            deleted_runs_path.write_text(deleted_runs_text, encoding="utf-8")

            def mutate_then_fail(_reports_dir: Path, _run_id: str) -> dict[str, object]:
                runs_path.write_text("", encoding="utf-8")
                deleted_runs_path.write_text("", encoding="utf-8")
                raise HTTPException(status_code=503, detail="temporary archive failure")

            with patch(
                "api.v1.endpoints.research_status._remove_active_experiment_run",
                side_effect=mutate_then_fail,
            ):
                with self.assertRaises(HTTPException) as ctx:
                    _archive_experiment_run(reports_dir, "abc123")

            persisted_runs = runs_path.read_text(encoding="utf-8")
            persisted_deleted_runs = deleted_runs_path.read_text(encoding="utf-8")

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(persisted_runs, runs_text)
        self.assertEqual(persisted_deleted_runs, deleted_runs_text)

    def test_permanently_delete_experiment_run_removes_active_record_and_marked_note(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            deleted_runs_path = Path(temp_dir) / "experiment_runs_deleted.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "abc123",
                                "experiment_name": "baseline_lgb_alpha158",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"run_id": "other456", "experiment_name": "risk_overlay"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：应被永久删除。\n"
                "<!-- /experiment-run:abc123 -->\n\n"
                "<!-- experiment-run:other456 -->\n"
                "## 2026-07-04 risk_overlay\n\n"
                "- 结论：应保留。\n"
                "<!-- /experiment-run:other456 -->\n",
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.delete(
                    "/api/v1/research/experiments/abc123/permanent?confirm=permanent"
                )

            active_lines = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
            ]
            log_text = log_path.read_text(encoding="utf-8")
            lock_exists = (Path(temp_dir) / ".experiment.lock").exists()

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(lock_exists)
        body = resp.json()
        self.assertEqual(body["removed_run"]["run_id"], "abc123")
        self.assertEqual(body["experiment_notebook"]["total_runs"], 1)
        self.assertEqual([item["run_id"] for item in active_lines], ["other456"])
        self.assertNotIn("abc123", log_text)
        self.assertNotIn("应被永久删除", log_text)
        self.assertIn("other456", log_text)
        self.assertIn("应保留", log_text)
        self.assertFalse(deleted_runs_path.exists())

    def test_permanent_delete_requires_backend_confirmation_without_mutating_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_text = (
                json.dumps(
                    {"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_text = (
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：没有后端确认时必须保留。\n"
                "<!-- /experiment-run:abc123 -->\n"
            )
            runs_path.write_text(runs_text, encoding="utf-8")
            log_path.write_text(log_text, encoding="utf-8")

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.delete("/api/v1/research/experiments/abc123/permanent")

            persisted_runs = runs_path.read_text(encoding="utf-8")
            persisted_log = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 400)
        self.assertIn("confirm=permanent", resp.json()["message"])
        self.assertEqual(persisted_runs, runs_text)
        self.assertEqual(persisted_log, log_text)

    def test_permanent_delete_rejects_incomplete_note_block_without_removing_jsonl_record(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps(
                    {"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：这个 block 缺少结束标记。\n",
                encoding="utf-8",
            )

            with patch(
                "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                )
                resp = client.delete(
                    "/api/v1/research/experiments/abc123/permanent?confirm=permanent"
                )

            active_lines = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
            ]
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 409)
        self.assertEqual([item["run_id"] for item in active_lines], ["abc123"])
        self.assertIn("缺少结束标记", log_text)

    def test_permanent_delete_restores_jsonl_record_when_note_rewrite_fails(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            static_dir = Path(temp_dir) / "static"
            static_dir.mkdir()
            status_path = Path(temp_dir) / "research_status_latest.json"
            runs_path = Path(temp_dir) / "experiment_runs.jsonl"
            log_path = Path(temp_dir) / "experiment_log.md"
            status_path.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-02T10:00:00+08:00",
                        "trainable": True,
                        "training_readiness": {
                            "status": "trainable",
                            "summary": "ready",
                        },
                        "data_sources": [],
                    }
                ),
                encoding="utf-8",
            )
            runs_path.write_text(
                json.dumps(
                    {"run_id": "abc123", "experiment_name": "baseline_lgb_alpha158"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "# 实验记录本\n\n"
                "<!-- experiment-run:abc123 -->\n"
                "## 2026-07-03 baseline_lgb_alpha158\n\n"
                "- 结论：写入失败时应保留。\n"
                "<!-- /experiment-run:abc123 -->\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "api.v1.endpoints.research_status.RESEARCH_STATUS_PATH", status_path
                ),
                patch(
                    "api.v1.endpoints.research_status._remove_marked_experiment_note",
                    side_effect=OSError("simulated disk write failure"),
                ),
            ):
                client = TestClient(
                    create_app(static_dir=static_dir),
                    headers={"Origin": "http://127.0.0.1:8010"},
                    raise_server_exceptions=False,
                )
                resp = client.delete(
                    "/api/v1/research/experiments/abc123/permanent?confirm=permanent"
                )

            active_lines = [
                json.loads(line)
                for line in runs_path.read_text(encoding="utf-8").splitlines()
            ]
            log_text = log_path.read_text(encoding="utf-8")

        self.assertEqual(resp.status_code, 500)
        self.assertEqual([item["run_id"] for item in active_lines], ["abc123"])
        self.assertIn("写入失败时应保留", log_text)

    def test_restore_text_file_logs_warning_when_best_effort_rollback_fails(
        self,
    ) -> None:
        from api.v1.endpoints.research_status import _restore_text_file

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment_runs.jsonl"

            with (
                patch("pathlib.Path.write_text", side_effect=OSError("restore boom")),
                self.assertLogs(
                    "api.v1.endpoints.research_status", level="WARNING"
                ) as captured,
            ):
                _restore_text_file(path, True, "original content")

        self.assertTrue(
            any("rollback restore failed" in message for message in captured.output)
        )

    def test_status_paths_can_be_overridden_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            dsa_root = base / "dsa"
            projects_root = base / "workspace"
            workflow_root = base / "workflow"
            reports_root = base / "custom_reports"

            env = os.environ.copy()
            env.update(
                {
                    "DSA_ROOT": str(dsa_root),
                    "WORKING_PROJECTS_ROOT": str(projects_root),
                    "QUANT_WORKFLOW_ROOT": str(workflow_root),
                    "QUANT_WORKFLOW_REPORTS_ROOT": str(reports_root),
                }
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "import api.v1.endpoints.research_status as r; "
                        "print(json.dumps({"
                        "'dsa_root': str(r.DSA_ROOT), "
                        "'projects_root': str(r.PROJECTS_ROOT), "
                        "'workflow_root': str(r.QUANT_WORKFLOW_ROOT), "
                        "'reports_root': str(r.QUANT_WORKFLOW_REPORTS_ROOT), "
                        "'status_path': str(r.RESEARCH_STATUS_PATH)"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["dsa_root"], str(dsa_root))
        self.assertEqual(payload["projects_root"], str(projects_root))
        self.assertEqual(payload["workflow_root"], str(workflow_root))
        self.assertEqual(payload["reports_root"], str(reports_root))
        self.assertEqual(
            payload["status_path"], str(reports_root / "research_status_latest.json")
        )

    def test_default_status_paths_use_embedded_quant_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dsa_root = Path(temp_dir) / "daily_stock_analysis"
            dsa_root.mkdir(parents=True)

            env = os.environ.copy()
            env["DSA_ROOT"] = str(dsa_root)
            env.pop("WORKING_PROJECTS_ROOT", None)
            env.pop("QUANT_WORKFLOW_ROOT", None)
            env.pop("QUANT_WORKFLOW_REPORTS_ROOT", None)
            env.pop("QUANT_DATA_ROOT", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json; "
                        "from unittest.mock import patch; "
                        "patch('dotenv.load_dotenv', return_value=False).start(); "
                        "import api.v1.endpoints.research_status as r; "
                        "print(json.dumps({"
                        "'workflow_root': str(r.QUANT_WORKFLOW_ROOT), "
                        "'reports_root': str(r.QUANT_WORKFLOW_REPORTS_ROOT), "
                        "'quant_data_root': str(r.QUANT_DATA_ROOT), "
                        "'status_path': str(r.RESEARCH_STATUS_PATH)"
                        "}))"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["workflow_root"], str(dsa_root / "quant_workflow"))
        self.assertEqual(
            payload["reports_root"], str(dsa_root / "quant_workflow" / "reports")
        )
        self.assertEqual(payload["quant_data_root"], str(dsa_root.parent / "quant_qmt_data" / "data"))
        self.assertEqual(
            payload["status_path"],
            str(
                dsa_root / "quant_workflow" / "reports" / "research_status_latest.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
