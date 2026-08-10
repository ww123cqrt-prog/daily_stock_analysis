import importlib.util
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_data_quality.py"
    spec = importlib.util.spec_from_file_location("report_data_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_quality_publication(
    out_dir: Path,
    pointer_name: str = "data_quality_report_latest.json",
) -> tuple[dict, dict, bytes]:
    pointer = json.loads(
        (out_dir / pointer_name).read_text(encoding="utf-8")
    )
    evidence_bytes = (out_dir / pointer["evidence_artifact"]).read_bytes()
    evidence = json.loads(evidence_bytes.decode("utf-8"))
    return pointer, evidence, evidence_bytes


def write_prices(directory: Path, symbol: str) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=3),
            "open": [10.0, 10.2, 10.4],
            "high": [10.1, 10.3, 10.5],
            "low": [9.9, 10.1, 10.3],
            "close": [10.0, 10.2, 10.4],
            "volume": [1000, 1100, 1200],
        }
    )
    frame.to_parquet(directory / f"{symbol}.parquet", index=False)


def write_benchmark_features(
    provider: Path,
    *,
    symbols: tuple[str, ...] = ("SH000300", "SH000905", "SH000906"),
) -> None:
    for symbol_index, symbol in enumerate(symbols, start=1):
        feature_dir = provider / "features" / symbol.lower()
        feature_dir.mkdir(parents=True, exist_ok=True)
        for field in ("open", "high", "low", "close", "volume", "factor"):
            values = np.array(
                [
                    0.0,
                    1.0 if field == "factor" else 10.0 * symbol_index,
                    1.0 if field == "factor" else 10.0 * symbol_index + 1.0,
                ],
                dtype="<f4",
            )
            if field == "volume":
                values[1:] = [1000.0, 1100.0]
            values.tofile(feature_dir / f"{field}.day.bin")


def write_instrument_features(provider: Path, symbol: str) -> None:
    feature_dir = provider / "features" / symbol.lower()
    feature_dir.mkdir(parents=True, exist_ok=True)
    for field in ("open", "high", "low", "close", "volume", "factor"):
        np.array([0.0, 1.0, 2.0], dtype="<f4").tofile(
            feature_dir / f"{field}.day.bin"
        )


def write_direct_descriptor(
    provider: Path,
    *,
    cleaning_mode: str,
    stock_snapshot_id: str = "ashare-snapshot",
    index_snapshot_id: str = "index-snapshot",
) -> Path:
    provider.mkdir(parents=True, exist_ok=True)
    descriptor = provider / "PROVIDER_DESCRIPTOR.json"
    descriptor.write_text(
        json.dumps(
            {
                "dataset": {
                    "dataset_id": "ashare_daily_qfq",
                    "snapshot_id": stock_snapshot_id,
                    "manifest_sha256": "a" * 64,
                    "content_sha256": "b" * 64,
                },
                "index_dataset": {
                    "dataset_id": "cn_index_daily",
                    "snapshot_id": index_snapshot_id,
                    "manifest_sha256": "c" * 64,
                    "content_sha256": "d" * 64,
                },
                "cleaning": {"mode": cleaning_mode},
            }
        ),
        encoding="utf-8",
    )
    return descriptor


class ReportDataQualityTest(unittest.TestCase):
    def test_default_quality_report_fails_closed_when_qlib_provider_is_unavailable(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ashare_snapshot = root / "ashare"
            (ashare_snapshot / "bars").mkdir(parents=True)
            resolution = SimpleNamespace(snapshot_path=ashare_snapshot)
            resolved_ids: list[str] = []

            def resolve_only_formal_bars(dataset_id, **_kwargs):
                resolved_ids.append(dataset_id)
                if dataset_id == "ashare_daily_qfq":
                    return resolution
                self.fail(f"quality report must not fall back to {dataset_id}")

            with (
                patch.object(module, "QUANT_DATA", root),
                patch.object(module, "resolve_dataset", side_effect=resolve_only_formal_bars),
                patch.object(
                    module,
                    "qlib_workflow_provider_uri",
                    side_effect=RuntimeError("Qlib Provider unavailable"),
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(RuntimeError, "Qlib Provider unavailable"):
                    module.main(["--out-dir", str(root / "reports")])

        self.assertEqual(resolved_ids, ["ashare_daily_qfq"])

    def test_lowered_training_thresholds_are_not_authoritative(self):
        module = load_module()

        thresholds = module.build_training_threshold_contract(
            min_calendar_days=0,
            min_instruments=0,
            min_merged_instruments=0,
        )

        self.assertFalse(thresholds["authoritative"])
        self.assertEqual(thresholds["authoritative_min_calendar_days"], 252)
        self.assertEqual(thresholds["authoritative_min_instruments"], 100)

    def test_quality_contract_records_canonical_and_cleaned_provider_identities(self):
        module = load_module()

        def resolved(dataset_id: str, marker: str):
            return SimpleNamespace(
                dataset_id=dataset_id,
                catalog_entry={
                    "snapshot_id": f"{marker}-snapshot",
                    "snapshot_manifest_sha256": marker * 64,
                    "content_sha256": marker.upper() * 64,
                },
            )

        contract = module.build_quality_data_contract(
            ashare_resolution=resolved("ashare_daily_qfq", "a"),
            canonical_qlib_resolution=resolved("qlib_ashare_v2", "b"),
            cleaned_qlib_resolution=resolved("qlib_ashare_cleaned_v2", "c"),
            has_path_override=False,
        )

        self.assertEqual(contract["status"], "verified")
        self.assertEqual(
            contract["qlib_providers"]["canonical"]["dataset_id"],
            "qlib_ashare_v2",
        )
        self.assertEqual(
            contract["qlib_providers"]["cleaned"]["dataset_id"],
            "qlib_ashare_cleaned_v2",
        )
        self.assertEqual(
            contract["qlib_provider"], contract["qlib_providers"]["canonical"]
        )

    def test_provider_quality_preserves_validated_lineage_without_contract_branching(self):
        module = load_module()
        identity = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "historical-snapshot",
            "manifest_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        }

        record = module.build_provider_quality_record(
            provider_key="canonical",
            identity=identity,
            qlib_stats={"calendar_days": 0, "instruments": 0},
            merged_success=set(),
            issue_counts={},
            min_calendar_days=252,
            min_instruments=100,
            min_merged_instruments=100,
            scan_mode="full",
            verification_status="unverified",
            verification_block_reason="not verified",
        )

        self.assertEqual(record["provider_contract"], "immutable_snapshot_v2")

    def test_changed_provider_lineage_invalidates_quality_contract(self):
        module = load_module()
        expected = {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "snapshot-a",
            "manifest_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        }
        refreshed = {**expected, "snapshot_id": "snapshot-b"}
        contract = {"status": "verified", "reason": "stable"}

        module.bind_provider_lineage_state(
            contract,
            expected={"canonical": expected},
            refreshed={"canonical": refreshed},
        )

        self.assertEqual(contract["status"], "unverified")
        self.assertFalse(contract["provider_lineage_state"]["stable"])
        self.assertIn("Qlib Provider", contract["reason"])
        self.assertNotIn("direct", contract["reason"].lower())

    def test_sampled_scan_skips_second_full_content_revalidation(self):
        module = load_module()
        resolutions = {"ashare_daily_qfq": object()}

        with patch.object(
            module,
            "revalidate_native_snapshot_resolutions",
        ) as revalidate:
            status = module.revalidate_native_snapshot_content_for_scan(
                resolutions,
                scan_mode="sampled",
            )

        self.assertEqual(status, "skipped_sampled")
        revalidate.assert_not_called()

    def test_full_scan_keeps_second_full_content_revalidation(self):
        module = load_module()
        resolutions = {"ashare_daily_qfq": object()}

        with patch.object(
            module,
            "revalidate_native_snapshot_resolutions",
        ) as revalidate:
            status = module.revalidate_native_snapshot_content_for_scan(
                resolutions,
                scan_mode="full",
            )

        self.assertEqual(status, "verified")
        revalidate.assert_called_once_with(resolutions)

    def test_sampled_revalidation_status_keeps_snapshot_contract_unverified(self):
        module = load_module()
        state = {
            "ashare_daily_qfq": {
                "algorithm": "test",
                "sha256": "a" * 64,
            }
        }
        contract = {"status": "verified", "reason": "initial"}

        module.bind_snapshot_state(
            contract,
            before=state,
            after_scan=state,
            after_revalidation=state,
            fingerprint_failed=False,
            content_revalidation_status="skipped_sampled",
        )

        self.assertEqual(contract["status"], "unverified")
        self.assertFalse(contract["snapshot_state"]["stable"])
        self.assertFalse(contract["snapshot_state"]["content_revalidated"])
        self.assertEqual(
            contract["snapshot_state"]["content_revalidation_status"],
            "skipped_sampled",
        )
        self.assertIn("sampled", contract["reason"].lower())

    def test_main_records_provider_specific_quality_for_cleaned_snapshot(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            ashare_snapshot = root / "ashare-snapshot"
            canonical_snapshot = root / "canonical-snapshot"
            cleaned_snapshot = root / "cleaned-snapshot"
            index_snapshot = root / "index-snapshot"
            for snapshot in (
                ashare_snapshot,
                canonical_snapshot,
                cleaned_snapshot,
                index_snapshot,
            ):
                snapshot.mkdir(parents=True)
                (snapshot / "stable.bin").write_bytes(b"stable")
            descriptors = {
                "canonical": write_direct_descriptor(
                    canonical_snapshot,
                    cleaning_mode="canonical",
                ),
                "cleaned": write_direct_descriptor(
                    cleaned_snapshot,
                    cleaning_mode="cleaned",
                ),
            }
            identities = {
                "ashare_daily_qfq": SimpleNamespace(
                    dataset_id="ashare_daily_qfq",
                    snapshot_path=ashare_snapshot,
                    catalog_entry={
                        "snapshot_id": "ashare-snapshot",
                        "snapshot_manifest_sha256": "a" * 64,
                        "content_sha256": "b" * 64,
                    },
                ),
                "qlib_ashare_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_v2",
                    snapshot_path=canonical_snapshot,
                    catalog_entry={
                        "snapshot_id": "canonical-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                    },
                ),
                "qlib_ashare_cleaned_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_cleaned_v2",
                    snapshot_path=cleaned_snapshot,
                    catalog_entry={
                        "snapshot_id": "cleaned-snapshot",
                        "snapshot_manifest_sha256": "e" * 64,
                        "content_sha256": "f" * 64,
                    },
                ),
                "cn_index_daily": SimpleNamespace(
                    dataset_id="cn_index_daily",
                    snapshot_path=index_snapshot,
                    catalog_entry={
                        "snapshot_id": "index-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                        "price_basis": "derived",
                    },
                ),
            }
            summary = pd.DataFrame(
                {
                    "symbol": [f"{index:06d}" for index in range(100)],
                    "duplicate_dates": 0,
                    "null_dates": 0,
                    "bad_ohlc_rows": 0,
                    "abs_return_gt_50pct": 0,
                    "non_positive_volume": 0,
                    "missing_columns": "",
                }
            )

            def provider_stats(path: Path, **_kwargs) -> dict:
                is_cleaned = Path(path) == cleaned_snapshot
                return {
                    "calendar_valid": True,
                    "invalid_calendar_entries": 0,
                    "instrument_records_valid": True,
                    "invalid_instrument_records": 0,
                    "calendar_days": 251 if is_cleaned else 252,
                    "calendar_start": "2025-08-12",
                    "calendar_end": "2026-07-31",
                    "instruments": 100,
                    "feature_dirs": 103,
                    "instrument_feature_dirs": 100,
                    "missing_instrument_feature_dirs": 0,
                    "missing_instrument_feature_sample": "",
                    "complete_instrument_feature_dirs": 100,
                    "incomplete_instrument_feature_dirs": 0,
                    "missing_instrument_feature_files": 0,
                    "missing_instrument_feature_file_sample": "",
                    "invalid_instrument_feature_files": 0,
                    "invalid_instrument_feature_file_sample": "",
                    "misaligned_instrument_feature_dirs": 0,
                    "misaligned_instrument_feature_dir_sample": "",
                    "benchmark_feature_dirs": 3,
                    "unexpected_extra_feature_dirs": 0,
                    "unexpected_extra_feature_sample": "",
                    "benchmark_complete": True,
                    "benchmark_first_observation": "2025-08-12",
                    "benchmark_missing_days": 0,
                    "benchmark_details": {},
                    "size_mb": 1.0,
                }

            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=lambda dataset_id, **_kwargs: identities[dataset_id],
                ),
                patch.object(module, "QUANT_DATA", root),
                patch.object(
                    module,
                    "qlib_workflow_provider_uri",
                    side_effect=lambda provider: descriptors[provider].parent,
                ),
                patch.object(
                    module,
                    "source_stats",
                    side_effect=lambda name, _directory: module.SourceStats(
                        name=name,
                        files=100,
                        rows=300,
                        first_date="2024-01-02",
                        last_date="2026-07-31",
                        size_mb=1.0,
                    ),
                ),
                patch.object(
                    module,
                    "scan_merged_files",
                    return_value=(summary, pd.DataFrame()),
                ),
                patch.object(
                    module, "overlap_consistency", return_value=pd.DataFrame()
                ),
                patch.object(module, "qlib_provider_stats", side_effect=provider_stats),
                patch.object(
                    module,
                    "load_source_index",
                    return_value={f"{index:06d}" for index in range(100)},
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                result = module.main(["--out-dir", str(out_dir)])

            _pointer, quality, _evidence_bytes = load_quality_publication(out_dir)

        self.assertEqual(result, 0)
        self.assertTrue(quality["provider_quality"]["canonical"]["trainable"])
        self.assertFalse(quality["provider_quality"]["cleaned"]["trainable"])
        self.assertEqual(
            quality["provider_quality"]["cleaned"]["dataset_id"],
            "ashare_daily_qfq",
        )
        self.assertEqual(
            quality["provider_quality"]["cleaned"]["provider_contract"],
            "direct_parquet_v1",
        )
        self.assertEqual(
            quality["provider_quality"]["cleaned"]["calendar_days"], 251
        )
        self.assertTrue(
            quality["provider_quality"]["cleaned"]["failure_reasons"]
        )

    def test_native_snapshot_mutation_during_scan_cannot_publish_verified_evidence(
        self,
    ):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            ashare_snapshot = root / "ashare-snapshot"
            canonical_snapshot = root / "canonical-snapshot"
            cleaned_snapshot = root / "cleaned-snapshot"
            index_snapshot = root / "index-snapshot"
            for snapshot in (
                ashare_snapshot,
                canonical_snapshot,
                cleaned_snapshot,
                index_snapshot,
            ):
                snapshot.mkdir(parents=True)
                (snapshot / "initial.bin").write_bytes(b"initial")
            descriptors = {
                "canonical": write_direct_descriptor(
                    canonical_snapshot,
                    cleaning_mode="canonical",
                ),
                "cleaned": write_direct_descriptor(
                    cleaned_snapshot,
                    cleaning_mode="cleaned",
                ),
            }

            identities = {
                "ashare_daily_qfq": SimpleNamespace(
                    dataset_id="ashare_daily_qfq",
                    snapshot_path=ashare_snapshot,
                    catalog_entry={
                        "snapshot_id": "ashare-snapshot",
                        "snapshot_manifest_sha256": "a" * 64,
                        "content_sha256": "b" * 64,
                    },
                ),
                "qlib_ashare_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_v2",
                    snapshot_path=canonical_snapshot,
                    catalog_entry={
                        "snapshot_id": "canonical-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                    },
                ),
                "qlib_ashare_cleaned_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_cleaned_v2",
                    snapshot_path=cleaned_snapshot,
                    catalog_entry={
                        "snapshot_id": "cleaned-snapshot",
                        "snapshot_manifest_sha256": "e" * 64,
                        "content_sha256": "f" * 64,
                    },
                ),
                "cn_index_daily": SimpleNamespace(
                    dataset_id="cn_index_daily",
                    snapshot_path=index_snapshot,
                    catalog_entry={
                        "snapshot_id": "index-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                        "price_basis": "derived",
                    },
                ),
            }
            summary = pd.DataFrame(
                {
                    "symbol": [f"{index:06d}" for index in range(100)],
                    "duplicate_dates": 0,
                    "null_dates": 0,
                    "bad_ohlc_rows": 0,
                    "abs_return_gt_50pct": 0,
                    "non_positive_volume": 0,
                    "missing_columns": "",
                }
            )
            valid_provider_stats = {
                "calendar_days": 252,
                "calendar_start": "2025-08-12",
                "calendar_end": "2026-07-31",
                "instruments": 100,
                "feature_dirs": 103,
                "instrument_feature_dirs": 100,
                "missing_instrument_feature_dirs": 0,
                "complete_instrument_feature_dirs": 100,
                "incomplete_instrument_feature_dirs": 0,
                "missing_instrument_feature_files": 0,
                "missing_instrument_feature_file_sample": "",
                "invalid_instrument_feature_files": 0,
                "invalid_instrument_feature_file_sample": "",
                "benchmark_feature_dirs": 3,
                "unexpected_extra_feature_dirs": 0,
                "unexpected_extra_feature_sample": "",
                "benchmark_complete": True,
                "benchmark_first_observation": "2025-08-12",
                "benchmark_missing_days": 0,
                "benchmark_details": {},
                "size_mb": 1.0,
            }

            def mutate_during_scan(*_args, **_kwargs):
                (ashare_snapshot / "changed-during-scan.bin").write_bytes(b"changed")
                return summary, pd.DataFrame()

            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=lambda dataset_id, **_kwargs: identities[dataset_id],
                ),
                patch.object(module, "QUANT_DATA", root),
                patch.object(
                    module,
                    "qlib_workflow_provider_uri",
                    side_effect=lambda provider: descriptors[provider].parent,
                ),
                patch.object(
                    module,
                    "source_stats",
                    side_effect=lambda name, _directory: module.SourceStats(
                        name=name,
                        files=100,
                        rows=300,
                        first_date="2024-01-02",
                        last_date="2026-07-31",
                        size_mb=1.0,
                    ),
                ),
                patch.object(
                    module, "scan_merged_files", side_effect=mutate_during_scan
                ),
                patch.object(
                    module, "overlap_consistency", return_value=pd.DataFrame()
                ),
                patch.object(
                    module,
                    "qlib_provider_stats",
                    return_value=valid_provider_stats,
                ),
                patch.object(
                    module,
                    "load_source_index",
                    return_value={f"{index:06d}" for index in range(100)},
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                result = module.main(["--out-dir", str(out_dir)])

            _pointer, quality, _evidence_bytes = load_quality_publication(out_dir)

        self.assertEqual(result, 0)
        self.assertEqual(quality["data_contract"]["status"], "unverified")
        self.assertFalse(quality["data_contract"]["snapshot_state"]["stable"])
        self.assertEqual(quality["verification_status"], "unverified")
        self.assertFalse(quality["trainable"])
        self.assertFalse(quality["provider_quality"]["canonical"]["trainable"])
        self.assertIn("changed during", quality["data_contract"]["reason"].lower())

    def test_native_snapshot_mutation_after_resolution_requires_revalidation(
        self,
    ):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            ashare_snapshot = root / "ashare-snapshot"
            canonical_snapshot = root / "canonical-snapshot"
            cleaned_snapshot = root / "cleaned-snapshot"
            index_snapshot = root / "index-snapshot"
            for snapshot in (
                ashare_snapshot,
                canonical_snapshot,
                cleaned_snapshot,
                index_snapshot,
            ):
                snapshot.mkdir(parents=True)
                (snapshot / "initial.bin").write_bytes(b"initial")
            descriptors = {
                "canonical": write_direct_descriptor(
                    canonical_snapshot,
                    cleaning_mode="canonical",
                ),
                "cleaned": write_direct_descriptor(
                    cleaned_snapshot,
                    cleaning_mode="cleaned",
                ),
            }

            identities = {
                "ashare_daily_qfq": SimpleNamespace(
                    dataset_id="ashare_daily_qfq",
                    snapshot_path=ashare_snapshot,
                    catalog_entry={
                        "snapshot_id": "ashare-snapshot",
                        "snapshot_manifest_sha256": "a" * 64,
                        "content_sha256": "b" * 64,
                    },
                ),
                "qlib_ashare_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_v2",
                    snapshot_path=canonical_snapshot,
                    catalog_entry={
                        "snapshot_id": "canonical-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                    },
                ),
                "qlib_ashare_cleaned_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_cleaned_v2",
                    snapshot_path=cleaned_snapshot,
                    catalog_entry={
                        "snapshot_id": "cleaned-snapshot",
                        "snapshot_manifest_sha256": "e" * 64,
                        "content_sha256": "f" * 64,
                    },
                ),
                "cn_index_daily": SimpleNamespace(
                    dataset_id="cn_index_daily",
                    snapshot_path=index_snapshot,
                    catalog_entry={
                        "snapshot_id": "index-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                        "price_basis": "derived",
                    },
                ),
            }
            resolve_calls: dict[str, int] = {}

            def resolve_with_post_validation_mutation(dataset_id: str, **_kwargs):
                resolve_calls[dataset_id] = resolve_calls.get(dataset_id, 0) + 1
                if dataset_id == "ashare_daily_qfq" and resolve_calls[dataset_id] > 1:
                    raise module.DataContractError("content hash changed")
                return identities[dataset_id]

            summary = pd.DataFrame(
                {
                    "symbol": [f"{index:06d}" for index in range(100)],
                    "duplicate_dates": 0,
                    "null_dates": 0,
                    "bad_ohlc_rows": 0,
                    "abs_return_gt_50pct": 0,
                    "non_positive_volume": 0,
                    "missing_columns": "",
                }
            )
            valid_provider_stats = {
                "calendar_valid": True,
                "invalid_calendar_entries": 0,
                "instrument_records_valid": True,
                "invalid_instrument_records": 0,
                "calendar_days": 252,
                "calendar_start": "2025-08-12",
                "calendar_end": "2026-07-31",
                "instruments": 100,
                "feature_dirs": 103,
                "instrument_feature_dirs": 100,
                "missing_instrument_feature_dirs": 0,
                "complete_instrument_feature_dirs": 100,
                "incomplete_instrument_feature_dirs": 0,
                "missing_instrument_feature_files": 0,
                "missing_instrument_feature_file_sample": "",
                "invalid_instrument_feature_files": 0,
                "invalid_instrument_feature_file_sample": "",
                "misaligned_instrument_feature_dirs": 0,
                "misaligned_instrument_feature_dir_sample": "",
                "benchmark_feature_dirs": 3,
                "unexpected_extra_feature_dirs": 0,
                "unexpected_extra_feature_sample": "",
                "benchmark_complete": True,
                "benchmark_first_observation": "2025-08-12",
                "benchmark_missing_days": 0,
                "benchmark_details": {},
                "size_mb": 1.0,
            }

            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=resolve_with_post_validation_mutation,
                ),
                patch.object(module, "QUANT_DATA", root),
                patch.object(
                    module,
                    "qlib_workflow_provider_uri",
                    side_effect=lambda provider: descriptors[provider].parent,
                ),
                patch.object(
                    module,
                    "source_stats",
                    side_effect=lambda name, _directory: module.SourceStats(
                        name=name,
                        files=100,
                        rows=300,
                        first_date="2024-01-02",
                        last_date="2026-07-31",
                        size_mb=1.0,
                    ),
                ),
                patch.object(
                    module,
                    "scan_merged_files",
                    return_value=(summary, pd.DataFrame()),
                ),
                patch.object(
                    module, "overlap_consistency", return_value=pd.DataFrame()
                ),
                patch.object(
                    module,
                    "qlib_provider_stats",
                    return_value=valid_provider_stats,
                ),
                patch.object(
                    module,
                    "load_source_index",
                    return_value={f"{index:06d}" for index in range(100)},
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                result = module.main(["--out-dir", str(out_dir)])

            _pointer, quality, _evidence_bytes = load_quality_publication(out_dir)

        self.assertEqual(result, 0)
        self.assertGreater(resolve_calls["ashare_daily_qfq"], 1)
        self.assertEqual(quality["data_contract"]["status"], "unverified")
        self.assertFalse(quality["data_contract"]["snapshot_state"]["stable"])
        self.assertFalse(quality["trainable"])
        self.assertIn("content", quality["data_contract"]["reason"].lower())

    def test_source_index_normalizes_native_v2_exchange_prefixes(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("SH600000.parquet", "SZ000001.parquet", "830001.parquet"):
                (root / name).write_bytes(b"")

            symbols = module.load_source_index(root)

        self.assertEqual(symbols, {"600000", "000001", "830001"})

    def test_formal_provider_stats_do_not_silently_fallback_to_legacy_bins(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Qlib Provider"):
                module.qlib_provider_stats(provider)

    def test_incomplete_benchmark_is_a_training_readiness_blocker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text("SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8")
            feature = provider / "features" / "sh000300" / "close.day.bin"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(np.array([0.0, 3900.0], dtype="<f4").tobytes())

            stats = module.qlib_provider_stats(provider, compatibility=True)
            ready, detail = module.build_training_readiness(
                stats,
                {f"{index:06d}" for index in range(100)},
                {"duplicate_dates": 0, "null_dates": 0, "missing_columns": 0, "bad_ohlc_rows": 0},
            )

        self.assertFalse(stats["benchmark_complete"])
        self.assertFalse(ready)
        self.assertIn("benchmark", detail["summary"].lower())

    def test_fifth_formal_index_failure_is_reported_as_a_blocker(self):
        module = load_module()
        details = {
            symbol: {
                "complete": symbol != "SH000688",
                "missing_fields": [] if symbol != "SH000688" else ["close"],
                "missing_days": 0,
                "invalid_ohlc_dates": [],
            }
            for symbol in (
                "SH000300",
                "SH000905",
                "SH000906",
                "SZ399006",
                "SH000688",
            )
        }
        ready, result = module.build_training_readiness(
            {
                "calendar_valid": True,
                "invalid_calendar_entries": 0,
                "instrument_records_valid": True,
                "invalid_instrument_records": 0,
                "calendar_days": 252,
                "instruments": 100,
                "feature_dirs": 100,
                "instrument_feature_dirs": 100,
                "missing_instrument_feature_dirs": 0,
                "complete_instrument_feature_dirs": 100,
                "incomplete_instrument_feature_dirs": 0,
                "missing_instrument_feature_files": 0,
                "invalid_instrument_feature_files": 0,
                "misaligned_instrument_feature_dirs": 0,
                "benchmark_complete": False,
                "benchmark_details": details,
            },
            {f"{index:06d}" for index in range(100)},
            {
                "duplicate_dates": 0,
                "null_dates": 0,
                "missing_columns": 0,
                "bad_ohlc_rows": 0,
            },
        )

        self.assertFalse(ready)
        self.assertTrue(
            any("SH000688" in reason for reason in result["failure_reasons"])
        )

    def test_missing_instrument_feature_directories_block_training(self):
        module = load_module()
        ready, detail = module.build_training_readiness(
            {
                "calendar_days": 252,
                "instruments": 100,
                "feature_dirs": 3,
                "instrument_feature_dirs": 0,
                "missing_instrument_feature_dirs": 100,
                "benchmark_complete": True,
                "benchmark_missing_days": 0,
            },
            {f"{index:06d}" for index in range(100)},
            {
                "duplicate_dates": 0,
                "null_dates": 0,
                "missing_columns": 0,
                "bad_ohlc_rows": 0,
            },
        )

        self.assertFalse(ready)
        self.assertIn("feature", detail["summary"].lower())

    def test_missing_required_file_inside_instrument_directory_blocks_training(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_instrument_features(provider, "SZ000001")
            write_benchmark_features(provider)
            (provider / "features" / "sz000001" / "factor.day.bin").unlink()

            stats = module.qlib_provider_stats(provider, compatibility=True)
            ready, detail = module.build_training_readiness(
                stats,
                {"000001"},
                {
                    "duplicate_dates": 0,
                    "null_dates": 0,
                    "missing_columns": 0,
                    "bad_ohlc_rows": 0,
                },
                min_calendar_days=2,
                min_instruments=1,
                min_merged_instruments=1,
            )

        self.assertEqual(stats["complete_instrument_feature_dirs"], 0)
        self.assertEqual(stats["incomplete_instrument_feature_dirs"], 1)
        self.assertEqual(stats["missing_instrument_feature_files"], 1)
        self.assertIn(
            "sz000001/factor.day.bin",
            stats["missing_instrument_feature_file_sample"],
        )
        self.assertFalse(ready)
        self.assertIn("required feature files", detail["summary"].lower())

    def test_invalid_required_instrument_binary_files_block_training(self):
        module = load_module()
        invalid_payloads = {
            "empty": b"",
            "truncated": b"\x00\x00",
            "non_integer_start": np.array([0.5, 1.0], dtype="<f4").tobytes(),
            "outside_calendar": np.array([2.0, 1.0], dtype="<f4").tobytes(),
            "no_finite_values": np.array(
                [0.0, np.nan, np.nan], dtype="<f4"
            ).tobytes(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_instrument_features(provider, "SZ000001")
            write_benchmark_features(provider)
            factor_path = provider / "features" / "sz000001" / "factor.day.bin"

            for label, payload in invalid_payloads.items():
                with self.subTest(label=label):
                    factor_path.write_bytes(payload)
                    stats = module.qlib_provider_stats(provider, compatibility=True)
                    ready, detail = module.build_training_readiness(
                        stats,
                        {"000001"},
                        {
                            "duplicate_dates": 0,
                            "null_dates": 0,
                            "missing_columns": 0,
                            "bad_ohlc_rows": 0,
                        },
                        min_calendar_days=2,
                        min_instruments=1,
                        min_merged_instruments=1,
                    )
                    self.assertEqual(
                        stats["complete_instrument_feature_dirs"], 0
                    )
                    self.assertEqual(
                        stats["incomplete_instrument_feature_dirs"], 1
                    )
                    self.assertEqual(
                        stats["invalid_instrument_feature_files"], 1
                    )
                    self.assertIn(
                        "sz000001/factor.day.bin",
                        stats["invalid_instrument_feature_file_sample"],
                    )
                    self.assertFalse(ready)
                    self.assertIn(
                        "invalid required feature files",
                        detail["summary"].lower(),
                    )

    def test_required_instrument_binary_values_obey_field_contract(self):
        module = load_module()
        invalid_values = (
            ("mixed_infinite_price", "open", [0.0, 10.0, np.inf]),
            ("mixed_negative_infinite_price", "high", [0.0, 10.0, -np.inf]),
            ("negative_price", "close", [0.0, 10.0, -1.0]),
            ("negative_volume", "volume", [0.0, 1000.0, -1.0]),
            ("zero_factor", "factor", [0.0, 1.0, 0.0]),
            ("negative_factor", "factor", [0.0, 1.0, -1.0]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)

            for label, field, values in invalid_values:
                with self.subTest(label=label):
                    write_instrument_features(provider, "SZ000001")
                    np.array(values, dtype="<f4").tofile(
                        provider
                        / "features"
                        / "sz000001"
                        / f"{field}.day.bin"
                    )

                    stats = module.qlib_provider_stats(provider, compatibility=True)

                    self.assertEqual(stats["complete_instrument_feature_dirs"], 0)
                    self.assertEqual(stats["incomplete_instrument_feature_dirs"], 1)
                    self.assertEqual(stats["invalid_instrument_feature_files"], 1)
                    self.assertIn(
                        f"sz000001/{field}.day.bin",
                        stats["invalid_instrument_feature_file_sample"],
                    )

    def test_required_instrument_binary_rejects_invalid_ohlc_relationships(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_instrument_features(provider, "SZ000001")
            write_benchmark_features(provider)
            np.array([0.0, 0.5, 1.5], dtype="<f4").tofile(
                provider / "features" / "sz000001" / "high.day.bin"
            )

            stats = module.qlib_provider_stats(provider, compatibility=True)

        self.assertEqual(stats["complete_instrument_feature_dirs"], 0)
        self.assertEqual(stats["incomplete_instrument_feature_dirs"], 1)
        self.assertEqual(stats["invalid_instrument_feature_files"], 1)
        self.assertIn(
            "sz000001/ohlc_relationship",
            stats["invalid_instrument_feature_file_sample"],
        )

    def test_required_instrument_binary_allows_nan_with_valid_values(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_instrument_features(provider, "SZ000001")
            write_benchmark_features(provider)
            for field in ("open", "high", "low", "close", "volume", "factor"):
                value = 1.0 if field == "factor" else 10.0
                np.array([0.0, np.nan, value], dtype="<f4").tofile(
                    provider
                    / "features"
                    / "sz000001"
                    / f"{field}.day.bin"
                )

            stats = module.qlib_provider_stats(provider, compatibility=True)

        self.assertEqual(stats["complete_instrument_feature_dirs"], 1)
        self.assertEqual(stats["invalid_instrument_feature_files"], 0)

    def test_required_instrument_binary_rejects_low_interval_coverage(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text(
                "2026-07-15\n2026-07-16\n2026-07-17\n2026-07-20\n",
                encoding="utf-8",
            )
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-15\t2026-07-20\n", encoding="utf-8"
            )
            write_instrument_features(provider, "SZ000001")
            write_benchmark_features(provider)
            feature_dir = provider / "features" / "sz000001"
            for field in ("open", "high", "low", "close", "volume", "factor"):
                value = 1.0 if field == "factor" else 10.0
                np.array(
                    [0.0, value, np.nan, np.nan, np.nan], dtype="<f4"
                ).tofile(feature_dir / f"{field}.day.bin")

            stats = module.qlib_provider_stats(provider, compatibility=True)

        self.assertEqual(stats["complete_instrument_feature_dirs"], 0)
        self.assertEqual(stats["incomplete_instrument_feature_dirs"], 1)
        self.assertEqual(stats["invalid_instrument_feature_files"], 6)

    def test_required_instrument_binary_rejects_misaligned_field_masks(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text(
                "2026-07-15\n2026-07-16\n2026-07-17\n2026-07-20\n",
                encoding="utf-8",
            )
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-15\t2026-07-20\n", encoding="utf-8"
            )
            feature_dir = provider / "features" / "sz000001"
            feature_dir.mkdir(parents=True)
            for field in ("open", "high", "low", "close", "volume", "factor"):
                value = 1.0 if field == "factor" else 10.0
                values = [0.0, value, value, value, value]
                if field == "open":
                    values[2] = np.nan
                np.array(values, dtype="<f4").tofile(
                    feature_dir / f"{field}.day.bin"
                )
            write_benchmark_features(provider)

            stats = module.qlib_provider_stats(provider, compatibility=True)

        self.assertEqual(stats["complete_instrument_feature_dirs"], 0)
        self.assertEqual(stats["incomplete_instrument_feature_dirs"], 1)
        self.assertEqual(stats["misaligned_instrument_feature_dirs"], 1)
        self.assertIn(
            "sz000001", stats["misaligned_instrument_feature_dir_sample"]
        )

    def test_invalid_calendar_semantics_block_training(self):
        module = load_module()
        invalid_calendars = {
            "duplicate": "2026-07-17\n2026-07-17\n",
            "unordered": "2026-07-20\n2026-07-17\n",
            "invalid_iso": "2026-07-17\n2026-02-30\n",
        }
        for label, calendar_text in invalid_calendars.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "qlib"
                calendar = provider / "calendars" / "day.txt"
                calendar.parent.mkdir(parents=True)
                calendar.write_text(calendar_text, encoding="utf-8")
                instruments = provider / "instruments" / "all.txt"
                instruments.parent.mkdir(parents=True)
                instruments.write_text(
                    "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
                )

                stats = module.qlib_provider_stats(provider, compatibility=True)
                ready, detail = module.build_training_readiness(
                    stats,
                    {f"{index:06d}" for index in range(100)},
                    {
                        "duplicate_dates": 0,
                        "null_dates": 0,
                        "missing_columns": 0,
                        "bad_ohlc_rows": 0,
                    },
                )

                self.assertFalse(stats["calendar_valid"])
                self.assertGreater(stats["invalid_calendar_entries"], 0)
                self.assertFalse(ready)
                self.assertTrue(
                    any(
                        "calendar contract" in reason.lower()
                        for reason in detail["failure_reasons"]
                    )
                )

    def test_invalid_instrument_intervals_block_training(self):
        module = load_module()
        invalid_instruments = {
            "malformed": "SZ000001\t2026-07-17\n",
            "duplicate": (
                "SZ000001\t2026-07-17\t2026-07-20\n"
                "SZ000001\t2026-07-17\t2026-07-20\n"
            ),
            "reversed": "SZ000001\t2026-07-20\t2026-07-17\n",
            "outside_calendar": "SZ000001\t2026-07-16\t2026-07-20\n",
        }
        for label, instrument_text in invalid_instruments.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                provider = Path(temp_dir) / "qlib"
                calendar = provider / "calendars" / "day.txt"
                calendar.parent.mkdir(parents=True)
                calendar.write_text(
                    "2026-07-17\n2026-07-20\n", encoding="utf-8"
                )
                instruments = provider / "instruments" / "all.txt"
                instruments.parent.mkdir(parents=True)
                instruments.write_text(instrument_text, encoding="utf-8")

                stats = module.qlib_provider_stats(provider, compatibility=True)
                ready, detail = module.build_training_readiness(
                    stats,
                    {f"{index:06d}" for index in range(100)},
                    {
                        "duplicate_dates": 0,
                        "null_dates": 0,
                        "missing_columns": 0,
                        "bad_ohlc_rows": 0,
                    },
                )

                self.assertFalse(stats["instrument_records_valid"])
                self.assertGreater(stats["invalid_instrument_records"], 0)
                self.assertFalse(ready)
                self.assertTrue(
                    any(
                        "instrument contract" in reason.lower()
                        for reason in detail["failure_reasons"]
                    )
                )

    def test_all_three_benchmarks_and_six_fields_are_required(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)

            complete = module.qlib_provider_stats(provider, compatibility=True)
            (provider / "features" / "sh000906" / "volume.day.bin").unlink()
            incomplete = module.qlib_provider_stats(provider, compatibility=True)

        self.assertTrue(complete["benchmark_complete"])
        self.assertEqual(complete["benchmark_feature_dirs"], 3)
        self.assertEqual(set(complete["benchmark_details"]), set(module.BENCHMARK_SYMBOLS))
        self.assertFalse(incomplete["benchmark_complete"])
        self.assertIn(
            "volume", incomplete["benchmark_details"]["SH000906"]["missing_fields"]
        )

    def test_benchmark_features_reject_invalid_ohlc_relationships(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)
            write_instrument_features(provider, "SZ000001")
            np.array([0.0, 9.0, 11.0], dtype="<f4").tofile(
                provider / "features" / "sh000300" / "high.day.bin"
            )

            stats = module.qlib_provider_stats(provider, compatibility=True)
            ready, readiness = module.build_training_readiness(
                stats,
                {"000001"},
                {
                    "duplicate_dates": 0,
                    "null_dates": 0,
                    "missing_columns": 0,
                    "bad_ohlc_rows": 0,
                },
                min_calendar_days=1,
                min_instruments=1,
                min_merged_instruments=1,
            )

        self.assertFalse(stats["benchmark_complete"])
        self.assertIn(
            "ohlc_relationship",
            stats["benchmark_details"]["SH000300"]["missing_fields"],
        )
        self.assertEqual(
            stats["benchmark_details"]["SH000300"]["invalid_ohlc_dates"],
            ["2026-07-17"],
        )
        self.assertFalse(ready)
        self.assertIn(
            "SH000300 benchmark OHLC relationship is invalid on 2026-07-17",
            readiness["failure_reasons"],
        )
        self.assertNotIn("0 missing days", readiness["summary"])

    def test_benchmark_factor_must_cover_every_calendar_day(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir) / "qlib"
            calendar = provider / "calendars" / "day.txt"
            calendar.parent.mkdir(parents=True)
            calendar.write_text("2026-07-17\n2026-07-20\n", encoding="utf-8")
            instruments = provider / "instruments" / "all.txt"
            instruments.parent.mkdir(parents=True)
            instruments.write_text(
                "SZ000001\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)
            np.array([0.0, 1.0, np.nan], dtype="<f4").tofile(
                provider / "features" / "sh000300" / "factor.day.bin"
            )

            stats = module.qlib_provider_stats(provider, compatibility=True)

        self.assertFalse(stats["benchmark_complete"])
        self.assertEqual(stats["benchmark_details"]["SH000300"]["missing_days"], 1)
        self.assertIn(
            "factor", stats["benchmark_details"]["SH000300"]["missing_fields"]
        )

    def test_help_does_not_resolve_large_dataset_defaults(self):
        module = load_module()
        with patch.object(
            module,
            "resolve_dataset",
            side_effect=AssertionError("Native snapshots were resolved for --help"),
        ), patch.object(sys, "argv", ["report_data_quality.py", "--help"]):
            with self.assertRaises(SystemExit) as raised:
                module.main()

        self.assertEqual(raised.exception.code, 0)

    def test_scan_merged_files_respects_sample_limit(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            write_prices(data_dir, "000001")
            write_prices(data_dir, "000002")
            write_prices(data_dir, "000003")

            summary, issues = module.scan_merged_files(data_dir, sample_limit=1)

        self.assertEqual(summary["symbol"].tolist(), ["000001"])
        self.assertTrue(issues.empty)

    def test_scan_merged_files_records_missing_date_as_blocking_summary(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "open": [10.0],
                    "high": [10.1],
                    "low": [9.9],
                    "close": [10.0],
                    "volume": [1000.0],
                }
            ).to_parquet(data_dir / "000001.parquet", index=False)

            source = module.source_stats("ashare_daily_qfq", data_dir)
            summary, issues = module.scan_merged_files(data_dir)

        self.assertEqual(source.files, 1)
        self.assertEqual(source.rows, 0)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["missing_columns"], "date")
        self.assertEqual(summary.iloc[0]["bad_ohlc_rows"], 1)
        self.assertEqual(issues.iloc[0]["issue"], "missing_date_column")

    def test_overlap_consistency_compares_public_sources_against_baostock(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            baostock = base / "baostock"
            jqdata = base / "jqdata"
            eastmoney = base / "eastmoney"
            akshare = base / "akshare"
            for directory in [baostock, jqdata, eastmoney, akshare]:
                directory.mkdir()
                write_prices(directory, "000001")
            module.BAOSTOCK_DIR = baostock
            module.JQDATA_DIR = jqdata
            module.EASTMONEY_DIR = eastmoney
            module.AKSHARE_DIR = akshare

            overlap = module.overlap_consistency(sample_limit=10)

        self.assertCountEqual(overlap["source"].tolist(), ["jqdata", "eastmoney", "akshare"])
        self.assertTrue((overlap["overlap_days"] == 3).all())

    def test_write_research_status_custom_out_dir_does_not_touch_official_latest(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "isolated_reports"
            out_dir.mkdir()
            official_latest = base / "official_reports" / "research_status_latest.json"
            official_latest.parent.mkdir()
            official_latest.write_text('{"generated_at":"original"}', encoding="utf-8")
            (official_latest.parent / "latest_training_result.json").write_text(
                '{"run_id":"real-run"}',
                encoding="utf-8",
            )
            (out_dir / "latest_training_result.json").write_text(
                '{"run_id":"isolated-run"}',
                encoding="utf-8",
            )

            status_path = module.write_research_status(
                out_dir=out_dir,
                provider_dir=out_dir / "test-provider",
                source_rows=[
                    {
                        "name": "merged",
                        "files": 1,
                        "rows": 3,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-04",
                        "size_mb": 0.01,
                    }
                ],
                qlib_stats={
                    "calendar_days": 3,
                    "calendar_start": "2024-01-02",
                    "calendar_end": "2024-01-04",
                    "instruments": 1,
                    "feature_dirs": 1,
                    "size_mb": 0.01,
                },
                merged_success={"000001"},
                issue_counts={
                    "bad_ohlc_rows": 0,
                    "non_positive_volume": 0,
                    "abs_return_gt_50pct": 0,
                },
            )

            isolated_payload = json.loads(status_path.read_text(encoding="utf-8"))
            official_payload = json.loads(official_latest.read_text(encoding="utf-8"))

        self.assertEqual(status_path, out_dir / "research_status_latest.json")
        self.assertEqual(isolated_payload["data_sources"][0]["name"], "merged")
        self.assertEqual(isolated_payload["latest_training"]["run_id"], "isolated-run")
        self.assertEqual(official_payload, {"generated_at": "original"})

    def test_research_status_distinguishes_canonical_snapshot_from_audit_sources(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            status_path = module.write_research_status(
                out_dir=out_dir,
                provider_dir=out_dir / "test-provider",
                source_rows=[
                    {
                        "name": "baostock",
                        "files": 1,
                        "rows": 3,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-04",
                        "size_mb": 0.01,
                    },
                    {
                        "name": "ashare_daily_qfq",
                        "files": 1,
                        "rows": 3,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-04",
                        "size_mb": 0.01,
                    },
                ],
                qlib_stats={
                    "calendar_days": 3,
                    "calendar_start": "2024-01-02",
                    "calendar_end": "2024-01-04",
                    "instruments": 1,
                    "feature_dirs": 1,
                    "size_mb": 0.01,
                },
                merged_success={"000001"},
                issue_counts={
                    "bad_ohlc_rows": 0,
                    "non_positive_volume": 0,
                    "abs_return_gt_50pct": 0,
                },
            )

            payload = json.loads(status_path.read_text(encoding="utf-8"))

        sources = {item["name"]: item for item in payload["data_sources"]}
        self.assertEqual(sources["ashare_daily_qfq"]["dataset_id"], "ashare_daily_qfq")
        self.assertEqual(sources["ashare_daily_qfq"]["role"], "canonical")
        self.assertEqual(sources["ashare_daily_qfq"]["lifecycle"], "immutable_snapshot_v2")
        self.assertEqual(sources["baostock"]["dataset_id"], "ashare_baostock_qfq_legacy")
        self.assertEqual(sources["baostock"]["role"], "audit")
        self.assertEqual(sources["baostock"]["lifecycle"], "legacy_unknown")

    def test_main_writes_empty_report_when_stock_list_and_data_are_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            module.MERGED_DIR = root / "merged"
            module.QLIB_DIR = root / "qlib"
            argv = [
                "report_data_quality.py",
                "--out-dir",
                str(out_dir),
                "--overlap-sample",
                "5",
                "--merged-sample",
                "0",
            ]

            with patch.object(sys, "argv", argv):
                self.assertEqual(module.main(), 0)

            report = out_dir / "data_quality_report_latest.md"
            report_json = out_dir / "data_quality_report_latest.json"
            status = out_dir / "research_status_latest.json"

            self.assertTrue(report.exists())
            self.assertTrue(report_json.exists())
            self.assertTrue(status.exists())
            self.assertEqual(
                len(list(out_dir.glob("data_quality_ashare_qfq_summary_*.csv"))),
                1,
            )
            text = report.read_text(encoding="utf-8")
            pointer, structured, evidence_bytes = load_quality_publication(out_dir)
            status_payload = json.loads(status.read_text(encoding="utf-8"))
            immutable_markdown = out_dir / structured["markdown_artifact"]
            immutable_markdown_exists = immutable_markdown.exists()
            immutable_markdown_text = immutable_markdown.read_text(encoding="utf-8")

        self.assertIn("A-Share Data Quality Report", text)
        self.assertIn("Qlib Provider", text)
        self.assertNotIn("Direct Provider", text)
        self.assertNotIn("Direct Qlib Provider", text)
        self.assertIn("Stock universe: 0", text)
        self.assertEqual(pointer["schema_version"], 3)
        self.assertEqual(
            pointer["publication_contract"],
            "atomic_json_pointer_to_immutable_quality_evidence_v2",
        )
        self.assertNotIn("data_contract", pointer)
        self.assertEqual(structured["schema_version"], 3)
        self.assertEqual(
            structured["evidence_contract"],
            "immutable_data_quality_evidence_v2",
        )
        self.assertTrue(immutable_markdown_exists)
        self.assertNotEqual(immutable_markdown, report)
        self.assertEqual(immutable_markdown_text, text)
        self.assertRegex(pointer["report_id"], r"^data-quality-[0-9a-f]{16}$")
        self.assertEqual(
            hashlib.sha256(evidence_bytes).hexdigest(), pointer["evidence_sha256"]
        )
        self.assertEqual(status_payload["quality_report_id"], pointer["report_id"])
        self.assertEqual(
            status_payload["quality_evidence_sha256"],
            pointer["evidence_sha256"],
        )
        self.assertEqual(
            structured["markdown_sha256"],
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            structured["summary"],
            {
                "stock_universe": 0,
                "usable_instruments": 0,
                "total_rows": 0,
                "date_range": "",
                "source": "merged",
            },
        )

    def test_sampled_scan_is_partial_and_never_publishes_trainable_status(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            out_dir.mkdir()
            (out_dir / "data_quality_report_latest.json").write_text(
                '{"report_id":"formal-report"}', encoding="utf-8"
            )
            (out_dir / "data_quality_report_latest.md").write_text(
                "# Formal quality report\n", encoding="utf-8"
            )
            (out_dir / "research_status_latest.json").write_text(
                '{"trainable":true,"quality_report_id":"formal-report"}',
                encoding="utf-8",
            )
            bars = root / "bars"
            bars.mkdir()
            write_prices(bars, "000001")
            provider = root / "qlib"
            (provider / "calendars").mkdir(parents=True)
            (provider / "instruments").mkdir()
            (provider / "calendars" / "day.txt").write_text(
                "2026-07-17\n2026-07-20\n", encoding="utf-8"
            )
            (provider / "instruments" / "all.txt").write_text(
                "SH600000\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)
            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"

            result = module.main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--merged-dir",
                    str(bars),
                    "--qlib-dir",
                    str(provider),
                    "--merged-sample",
                    "1",
                    "--min-calendar-days",
                    "0",
                    "--min-instruments",
                    "0",
                ]
            )
            sampled_dir = out_dir / "test_runs" / "data_quality"
            report = (sampled_dir / "data_quality_report_sampled_latest.md").read_text(
                encoding="utf-8"
            )
            pointer, quality, evidence_bytes = load_quality_publication(
                sampled_dir,
                pointer_name="data_quality_report_sampled_latest.json",
            )
            status = json.loads(
                (sampled_dir / "research_status_sampled_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            formal_pointer = json.loads(
                (out_dir / "data_quality_report_latest.json").read_text(
                    encoding="utf-8"
                )
            )
            formal_status = json.loads(
                (out_dir / "research_status_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result, 0)
        self.assertEqual(formal_pointer["report_id"], "formal-report")
        self.assertTrue(formal_status["trainable"])
        self.assertEqual(quality.get("scan_mode"), "sampled")
        self.assertEqual(quality.get("coverage_status"), "partial")
        self.assertEqual(quality.get("verification_status"), "unverified")
        self.assertFalse(quality.get("trainable", True))
        self.assertEqual(
            hashlib.sha256(evidence_bytes).hexdigest(), pointer["evidence_sha256"]
        )
        self.assertNotIn("data_contract", pointer)
        self.assertEqual(status.get("quality_report_id"), pointer.get("report_id"))
        self.assertEqual(
            status.get("quality_evidence_sha256"), pointer.get("evidence_sha256")
        )
        self.assertIn("Scan mode: sampled", report)
        self.assertIn("Verification status: unverified", report)
        self.assertIn("Sampled report cannot authorize training", report)
        self.assertFalse(status["trainable"])
        self.assertEqual(status.get("quality_scan", {}).get("mode"), "sampled")
        self.assertEqual(status.get("quality_scan", {}).get("coverage"), "partial")
        self.assertEqual(
            status.get("quality_scan", {}).get("verification_status"),
            "unverified",
        )
        self.assertEqual(status["training_readiness"]["status"], "partial")

    def test_full_scan_with_explicit_paths_cannot_publish_verified_training(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            bars = root / "legacy-bars"
            bars.mkdir()
            write_prices(bars, "000001")
            provider = root / "legacy-qlib"
            (provider / "calendars").mkdir(parents=True)
            (provider / "instruments").mkdir()
            (provider / "calendars" / "day.txt").write_text(
                "2026-07-17\n2026-07-20\n", encoding="utf-8"
            )
            (provider / "instruments" / "all.txt").write_text(
                "SH600000\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(provider)
            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"

            result = module.main(
                [
                    "--out-dir",
                    str(out_dir),
                    "--merged-dir",
                    str(bars),
                    "--qlib-dir",
                    str(provider),
                    "--min-calendar-days",
                    "0",
                    "--min-instruments",
                    "0",
                ]
            )
            pointer, quality, evidence_bytes = load_quality_publication(out_dir)
            status = json.loads(
                (out_dir / "research_status_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result, 0)
        self.assertEqual(quality["scan_mode"], "full")
        self.assertEqual(quality["data_contract"]["status"], "unverified")
        self.assertEqual(quality["verification_status"], "unverified")
        self.assertFalse(quality["trainable"])
        self.assertEqual(
            hashlib.sha256(evidence_bytes).hexdigest(), pointer["evidence_sha256"]
        )
        self.assertNotIn("data_contract", pointer)
        self.assertEqual(status["quality_report_id"], pointer["report_id"])
        self.assertEqual(
            status["quality_evidence_sha256"], pointer["evidence_sha256"]
        )
        self.assertFalse(status["trainable"])

    def test_native_v2_with_lowered_thresholds_cannot_authorize_training(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"
            ashare_snapshot = root / "ashare-snapshot"
            bars = ashare_snapshot / "bars"
            bars.mkdir(parents=True)
            write_prices(bars, "000001")
            qlib_snapshot = root / "qlib-snapshot"
            (qlib_snapshot / "calendars").mkdir(parents=True)
            (qlib_snapshot / "instruments").mkdir()
            (qlib_snapshot / "calendars" / "day.txt").write_text(
                "2026-07-17\n2026-07-20\n", encoding="utf-8"
            )
            (qlib_snapshot / "instruments" / "all.txt").write_text(
                "SH600000\t2026-07-17\t2026-07-20\n", encoding="utf-8"
            )
            write_benchmark_features(qlib_snapshot)
            canonical_direct = root / "direct-canonical"
            cleaned_direct = root / "direct-cleaned"
            descriptors = {
                "canonical": write_direct_descriptor(
                    canonical_direct,
                    cleaning_mode="canonical",
                ),
                "cleaned": write_direct_descriptor(
                    cleaned_direct,
                    cleaning_mode="cleaned",
                ),
            }
            index_snapshot = root / "index-snapshot"
            index_snapshot.mkdir()
            (index_snapshot / "stable.parquet").write_bytes(b"stable")
            provider_stats = module.qlib_provider_stats(
                qlib_snapshot,
                compatibility=True,
            )
            module.STOCK_LIST = root / "missing_stock_lists.json"
            module.EMPTY_CODES = root / "missing_empty_codes.json"
            module.TIMEOUT_CODES = root / "missing_timeout_codes.json"
            module.BAOSTOCK_DIR = root / "baostock"
            module.JQDATA_DIR = root / "jqdata"
            module.EASTMONEY_DIR = root / "eastmoney"
            module.AKSHARE_DIR = root / "akshare"
            identities = {
                "ashare_daily_qfq": SimpleNamespace(
                    dataset_id="ashare_daily_qfq",
                    snapshot_path=ashare_snapshot,
                    catalog_entry={
                        "snapshot_id": "ashare-snapshot",
                        "snapshot_manifest_sha256": "a" * 64,
                        "content_sha256": "b" * 64,
                    },
                ),
                "qlib_ashare_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_v2",
                    snapshot_path=qlib_snapshot,
                    catalog_entry={
                        "snapshot_id": "qlib-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                    },
                ),
                "qlib_ashare_cleaned_v2": SimpleNamespace(
                    dataset_id="qlib_ashare_cleaned_v2",
                    snapshot_path=qlib_snapshot,
                    catalog_entry={
                        "snapshot_id": "cleaned-snapshot",
                        "snapshot_manifest_sha256": "e" * 64,
                        "content_sha256": "f" * 64,
                    },
                ),
                "cn_index_daily": SimpleNamespace(
                    dataset_id="cn_index_daily",
                    snapshot_path=index_snapshot,
                    catalog_entry={
                        "snapshot_id": "index-snapshot",
                        "snapshot_manifest_sha256": "c" * 64,
                        "content_sha256": "d" * 64,
                        "price_basis": "derived",
                    },
                ),
            }

            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=lambda dataset_id, **_kwargs: identities[dataset_id],
                ),
                patch.object(module, "QUANT_DATA", root),
                patch.object(
                    module,
                    "qlib_workflow_provider_uri",
                    side_effect=lambda provider: descriptors[provider].parent,
                ),
                patch.object(
                    module,
                    "qlib_provider_stats",
                    return_value=provider_stats,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                result = module.main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--min-calendar-days",
                        "0",
                        "--min-instruments",
                        "0",
                    ]
                )
            pointer, quality, _evidence_bytes = load_quality_publication(out_dir)
            status = json.loads(
                (out_dir / "research_status_latest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(result, 0)
        self.assertEqual(quality["data_contract"]["status"], "verified")
        self.assertFalse(quality["training_thresholds"]["authoritative"])
        self.assertEqual(quality["verification_status"], "unverified")
        self.assertFalse(quality["trainable"])
        self.assertIn(
            "below the fixed authoritative minimums",
            quality["training_readiness"]["summary"],
        )
        self.assertEqual(status["quality_report_id"], pointer["report_id"])
        self.assertFalse(status["trainable"])


if __name__ == "__main__":
    unittest.main()
