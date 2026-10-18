import importlib.util
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_paths.py"
    spec = importlib.util.spec_from_file_location("script_paths", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptPathsTest(unittest.TestCase):
    def test_default_roots_follow_workflow_location(self):
        module = load_module()
        workflow_root = Path(__file__).resolve().parents[1]
        dsa_root = workflow_root.parent

        self.assertEqual(module.workflow_root(), workflow_root)
        self.assertEqual(module.projects_root(), dsa_root.parent)
        sibling_quant_data = dsa_root.parent / "quant_data"
        expected_quant_data = sibling_quant_data if sibling_quant_data.exists() else dsa_root / "quant_data"
        self.assertEqual(module.quant_data_root(), expected_quant_data)
        expected_reports = workflow_root / "reports"
        self.assertEqual(module.reports_root(), expected_reports)

        expected_mlruns = workflow_root / "mlruns"
        self.assertEqual(module.mlruns_root(), expected_mlruns)

    def test_working_projects_override_moves_default_project_roots(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            projects_root = Path(temp_dir) / "WorkingProjects"
            with patch.dict(os.environ, {"WORKING_PROJECTS_ROOT": str(projects_root)}, clear=False):
                self.assertEqual(module.projects_root(), projects_root)
                self.assertEqual(module.workflow_root(), Path(__file__).resolve().parents[1])
                self.assertEqual(module.quant_data_root(), Path(__file__).resolve().parents[2] / "quant_data")
                self.assertEqual(module.china_astock_root(), projects_root / "china-astock-quant")

    def test_sibling_quant_data_is_used_when_repo_local_data_is_absent(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            sibling_data = temp_root / "quant_data"
            workflow_root.mkdir(parents=True)
            sibling_data.mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_DATA_ROOT",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.quant_data_root(), sibling_data)

    def test_archived_active_pointer_cannot_replace_a_missing_local_catalog(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ACTIVE_RELEASE_SET.json").write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"QUANT_DATA_ROOT": str(root)}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "catalog"):
                    module.qlib_direct_provider_descriptor()

    def test_repo_local_reports_and_mlruns_are_default_even_when_sibling_payload_exists(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            sibling_workflow = temp_root / "quant_workflow"
            sibling_reports = sibling_workflow / "reports"
            sibling_mlruns = sibling_workflow / "mlruns"
            workflow_root.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            sibling_mlruns.mkdir(parents=True)
            (sibling_reports / "latest_training_result.json").write_text("{}", encoding="utf-8")
            (sibling_mlruns / "0").mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_WORKFLOW_ROOT",
                        "QUANT_WORKFLOW_REPORTS_ROOT",
                        "QUANT_WORKFLOW_MLRUNS_DIR",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.reports_root(), workflow_root / "reports")
                self.assertEqual(module.mlruns_root(), workflow_root / "mlruns")

    def test_repo_local_reports_and_mlruns_win_when_both_payloads_exist(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dsa_root = temp_root / "daily_stock_analysis"
            workflow_root = dsa_root / "quant_workflow"
            repo_reports = workflow_root / "reports"
            repo_mlruns = workflow_root / "mlruns"
            sibling_reports = temp_root / "quant_workflow" / "reports"
            sibling_mlruns = temp_root / "quant_workflow" / "mlruns"
            repo_reports.mkdir(parents=True)
            repo_mlruns.mkdir(parents=True)
            sibling_reports.mkdir(parents=True)
            sibling_mlruns.mkdir(parents=True)
            (repo_reports / "research_status_latest.json").write_text("{}", encoding="utf-8")
            (repo_mlruns / "0").mkdir()
            (sibling_reports / "latest_training_result.json").write_text("{}", encoding="utf-8")
            (sibling_mlruns / "models").mkdir()

            with patch.object(module, "DEFAULT_DSA_ROOT", dsa_root), patch.object(
                module, "DEFAULT_WORKFLOW_ROOT", workflow_root
            ), patch.object(module, "DEFAULT_PROJECTS_ROOT", temp_root), patch.dict(
                os.environ,
                {
                    key: ""
                    for key in [
                        "WORKING_PROJECTS_ROOT",
                        "QUANT_WORKFLOW_ROOT",
                        "QUANT_WORKFLOW_REPORTS_ROOT",
                        "QUANT_WORKFLOW_MLRUNS_DIR",
                    ]
                },
                clear=False,
            ):
                self.assertEqual(module.reports_root(), repo_reports)
                self.assertEqual(module.mlruns_root(), repo_mlruns)

    def test_specific_env_overrides_take_precedence(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env = {
                "QUANT_WORKFLOW_ROOT": str(root / "workflow"),
                "QUANT_WORKFLOW_REPORTS_ROOT": str(root / "reports"),
                "QUANT_WORKFLOW_MLRUNS_DIR": str(root / "mlruns"),
                "QUANT_DATA_ROOT": str(root / "data"),
                "QLIB_PROVIDER_URI": str(root / "provider"),
                "QLIB_CLEANED_PROVIDER_URI": str(root / "provider-cleaned"),
                "QLIB_COMPATIBILITY_PROVIDER_URI": str(root / "provider-compat"),
                "QLIB_CLEANED_COMPATIBILITY_PROVIDER_URI": str(
                    root / "provider-cleaned-compat"
                ),
                "CHINA_ASTOCK_QUANT_ROOT": str(root / "astock"),
            }
            with patch.dict(os.environ, env, clear=False), patch.object(
                module,
                "_validate_native_provider_override",
                side_effect=lambda configured, **_kwargs: Path(configured),
            ):
                self.assertEqual(module.workflow_root(), root / "workflow")
                self.assertEqual(module.reports_root(), root / "reports")
                self.assertEqual(module.mlruns_root(), root / "mlruns")
                self.assertEqual(module.quant_data_root(), root / "data")
                self.assertEqual(module.qlib_provider_uri(), root / "provider")
                self.assertEqual(module.qlib_provider_uri("cleaned"), root / "provider-cleaned")
                self.assertEqual(
                    module.qlib_compatibility_provider_uri(), root / "provider-compat"
                )
                self.assertEqual(
                    module.qlib_compatibility_provider_uri("cleaned"),
                    root / "provider-cleaned-compat",
                )
                self.assertEqual(module.china_astock_root(), root / "astock")

    def test_native_provider_override_rejects_legacy_compatibility_path(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            legacy = root / "qlib" / "cn_data_baostock_jqdata"
            legacy.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "QUANT_DATA_ROOT": str(root),
                    "QLIB_PROVIDER_URI": str(legacy),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "legacy compatibility"):
                    module.qlib_provider_uri()

    def test_cleaned_provider_override_rejects_legacy_compatibility_path(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            legacy = root / "qlib" / "cn_data_baostock_jqdata_cleaned"
            legacy.mkdir(parents=True)
            with patch.dict(
                os.environ,
                {
                    "QUANT_DATA_ROOT": str(root),
                    "QLIB_CLEANED_PROVIDER_URI": str(legacy),
                },
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "legacy compatibility"):
                    module.qlib_provider_uri("cleaned")

    def test_workflow_provider_errors_use_public_provider_terminology(self):
        module = load_module()
        with patch.dict(
            os.environ,
            {
                "QLIB_PROVIDER_URI": "/legacy/provider",
                "QLIB_CLEANED_PROVIDER_URI": "",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError) as override_error:
                module.qlib_direct_provider_descriptor()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError) as workflow_error:
                module.qlib_workflow_init_config(provider_uri=Path(temp_dir))

        for message in (str(override_error.exception), str(workflow_error.exception)):
            self.assertIn("Qlib Provider", message)
            self.assertNotRegex(message, r"(?i)direct provider|native V2|\.bin")

    def test_workflow_provider_resolves_an_approved_opaque_lineage(self):
        module = load_module()
        pinned = Path("/quant_data/runtime/qlib_provider/canonical")
        identity = {
            "provider_contract": "direct_parquet_v1",
            "dataset_id": "ashare_daily_qfq",
            "snapshot_id": "ashare-snapshot",
            "content_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "descriptor_path": "runtime/qlib_provider/canonical/PROVIDER_DESCRIPTOR.json",
            "descriptor_sha256": "c" * 64,
            "cleaning_mode": "canonical",
            "index_dataset_id": "cn_index_daily",
            "index_snapshot_id": "index-snapshot",
            "index_content_sha256": "d" * 64,
            "index_manifest_sha256": "e" * 64,
        }

        with patch.object(
            module,
            "_resolve_approved_workflow_provider_uri",
            return_value=pinned,
            create=True,
        ) as resolver:
            resolved = module.qlib_workflow_provider_uri(
                "canonical",
                expected_provider_identity=identity,
            )

        self.assertEqual(resolved, pinned)
        resolver.assert_called_once_with("canonical", identity)

    def test_workflow_provider_accepts_an_append_only_successor(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            frozen_provider = root / "providers" / "frozen"
            current_provider = root / "providers" / "current"
            frozen_snapshot = root / "snapshots" / "frozen"
            current_snapshot = root / "snapshots" / "current"
            for provider in (frozen_provider, current_provider):
                provider.mkdir(parents=True)
                (provider / "PROVIDER_DESCRIPTOR.json").write_text(
                    "{}\n", encoding="utf-8"
                )
            for snapshot in (frozen_snapshot, current_snapshot):
                (snapshot / "bars").mkdir(parents=True)

            frozen_bar = frozen_snapshot / "bars" / "SH600000.parquet"
            current_bar = current_snapshot / "bars" / "SH600000.parquet"
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-08-07", "2026-08-08"]),
                    "close": [10.0, 10.5],
                }
            ).to_parquet(frozen_bar, index=False)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2026-08-07", "2026-08-08", "2026-08-10"]
                    ),
                    "close": [10.0, 10.5, 10.8],
                }
            ).to_parquet(current_bar, index=False)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            frozen_context = SimpleNamespace(
                snapshot_path=frozen_snapshot,
                file_index=pd.DataFrame(
                    [
                        {
                            "symbol": "SH600000",
                            "relative_path": "bars/SH600000.parquet",
                            "row_count": 2,
                            "first_date": "2026-08-07",
                            "last_date": "2026-08-08",
                            "sha256": digest(frozen_bar),
                        }
                    ]
                ),
                index_snapshot_path=None,
                index_file_index=None,
                calendar=pd.DatetimeIndex(pd.to_datetime(["2026-08-07", "2026-08-08"])),
                membership=pd.DataFrame(
                    {
                        "symbol": ["SH600000"],
                        "index_code": ["000300.XSHG"],
                        "start_date": pd.to_datetime(["2026-08-07"]),
                        "end_date": pd.to_datetime(["2026-08-08"]),
                    }
                ),
                cleaning={"mode": "canonical"},
            )
            current_context = SimpleNamespace(
                snapshot_path=current_snapshot,
                file_index=pd.DataFrame(
                    [
                        {
                            "symbol": "SH600000",
                            "relative_path": "bars/SH600000.parquet",
                            "row_count": 3,
                            "first_date": "2026-08-07",
                            "last_date": "2026-08-10",
                            "sha256": digest(current_bar),
                        }
                    ]
                ),
                index_snapshot_path=None,
                index_file_index=None,
                calendar=pd.DatetimeIndex(
                    pd.to_datetime(["2026-08-07", "2026-08-08", "2026-08-10"])
                ),
                membership=pd.DataFrame(
                    {
                        "symbol": ["SH600000"],
                        "index_code": ["000300.XSHG"],
                        "start_date": pd.to_datetime(["2026-08-07"]),
                        "end_date": pd.to_datetime(["2026-08-10"]),
                    }
                ),
                cleaning={"mode": "canonical"},
            )

            with patch.object(module, "quant_data_root", return_value=root), patch.object(
                module, "qlib_direct_provider_uri", return_value=current_provider
            ), patch.object(
                module,
                "_load_direct_provider_context",
                side_effect=[frozen_context, current_context],
                create=True,
            ):
                resolved = module.qlib_workflow_provider_uri(
                    "cleaned", append_only_from=frozen_provider
                )

            self.assertEqual(resolved, current_provider)

    def test_workflow_provider_rejects_a_rewritten_historical_row(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            frozen_provider = root / "providers" / "frozen"
            current_provider = root / "providers" / "current"
            frozen_snapshot = root / "snapshots" / "frozen"
            current_snapshot = root / "snapshots" / "current"
            for provider in (frozen_provider, current_provider):
                provider.mkdir(parents=True)
                (provider / "PROVIDER_DESCRIPTOR.json").write_text(
                    "{}\n", encoding="utf-8"
                )
            for snapshot in (frozen_snapshot, current_snapshot):
                (snapshot / "bars").mkdir(parents=True)

            frozen_bar = frozen_snapshot / "bars" / "SH600000.parquet"
            current_bar = current_snapshot / "bars" / "SH600000.parquet"
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-08-07", "2026-08-08"]),
                    "close": [10.0, 10.5],
                }
            ).to_parquet(frozen_bar, index=False)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2026-08-07", "2026-08-08", "2026-08-10"]
                    ),
                    "close": [9.9, 10.5, 10.8],
                }
            ).to_parquet(current_bar, index=False)

            def context(snapshot: Path, path: Path, rows: int, last: str):
                return SimpleNamespace(
                    snapshot_path=snapshot,
                    file_index=pd.DataFrame(
                        [
                            {
                                "symbol": "SH600000",
                                "relative_path": "bars/SH600000.parquet",
                                "row_count": rows,
                                "first_date": "2026-08-07",
                                "last_date": last,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        ]
                    ),
                    index_snapshot_path=None,
                    index_file_index=None,
                    calendar=pd.DatetimeIndex(
                        pd.to_datetime(
                            ["2026-08-07", "2026-08-08"]
                            if rows == 2
                            else ["2026-08-07", "2026-08-08", "2026-08-10"]
                        )
                    ),
                    membership=pd.DataFrame(
                        {
                            "symbol": ["SH600000"],
                            "index_code": ["000300.XSHG"],
                            "start_date": pd.to_datetime(["2026-08-07"]),
                            "end_date": pd.to_datetime([last]),
                        }
                    ),
                    cleaning={"mode": "canonical"},
                )

            with patch.object(module, "quant_data_root", return_value=root), patch.object(
                module, "qlib_direct_provider_uri", return_value=current_provider
            ), patch.object(
                module,
                "_load_direct_provider_context",
                side_effect=[
                    context(frozen_snapshot, frozen_bar, 2, "2026-08-08"),
                    context(current_snapshot, current_bar, 3, "2026-08-10"),
                ],
                create=True,
            ):
                with self.assertRaisesRegex(
                    module.QlibProviderHistoryError, "historical bars changed"
                ):
                    module.qlib_workflow_provider_uri(
                        "cleaned", append_only_from=frozen_provider
                    )

    def test_native_provider_override_accepts_validated_snapshot_path(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            snapshot = (
                root
                / "qlib"
                / "cn_data_ashare_v2"
                / "snapshots"
                / "qlib-native-001"
            )
            snapshot.mkdir(parents=True)
            manifest = {
                "dataset_id": "qlib_ashare_v2",
                "snapshot_id": snapshot.name,
                "role": "canonical",
                "status": "ready",
                "price_basis": "qfq",
                "benchmark_complete": True,
            }
            with patch.dict(
                os.environ,
                {
                    "QUANT_DATA_ROOT": str(root),
                    "QLIB_PROVIDER_URI": str(snapshot),
                },
                clear=False,
            ), patch.object(
                module,
                "validate_qlib_snapshot",
                return_value=manifest,
            ):
                self.assertEqual(module.qlib_provider_uri(), snapshot.resolve())

    def test_unknown_provider_name_is_rejected(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.qlib_provider_uri("unknown")

    def test_canonical_provider_resolves_native_v2_snapshot_once_per_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            first = root / "qlib" / "native" / "snapshots" / "snapshot-001"
            second = root / "qlib" / "native" / "snapshots" / "snapshot-002"
            resolved = [
                SimpleNamespace(
                    snapshot_path=first,
                    manifest={"benchmark_complete": True},
                ),
                SimpleNamespace(
                    snapshot_path=second,
                    manifest={"benchmark_complete": True},
                ),
            ]
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root), "QLIB_PROVIDER_URI": ""},
                clear=False,
            ), patch.object(module, "resolve_dataset", side_effect=resolved) as resolver:
                module._resolve_native_qlib.cache_clear()
                self.assertEqual(module.qlib_provider_uri(), first)
                self.assertEqual(module.qlib_provider_uri(), first)

            self.assertEqual(resolver.call_count, 1)
            resolver.assert_called_once_with(
                "qlib_ashare_v2", quant_data_root=root.resolve(), require_basis="qfq"
            )

    def test_canonical_provider_fails_closed_when_native_benchmarks_are_incomplete(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            resolved = SimpleNamespace(
                snapshot_path=root / "snapshot",
                manifest={"benchmark_complete": False},
            )
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root), "QLIB_PROVIDER_URI": ""},
                clear=False,
            ), patch.object(module, "resolve_dataset", return_value=resolved):
                module._resolve_native_qlib.cache_clear()
                with self.assertRaisesRegex(RuntimeError, "benchmarks are incomplete"):
                    module.qlib_provider_uri()

    def test_cleaned_provider_resolves_native_v2_snapshot_once_per_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            first = root / "qlib" / "cleaned" / "snapshots" / "snapshot-001"
            second = root / "qlib" / "cleaned" / "snapshots" / "snapshot-002"
            resolved = [
                SimpleNamespace(
                    snapshot_path=first,
                    manifest={"benchmark_complete": True},
                ),
                SimpleNamespace(
                    snapshot_path=second,
                    manifest={"benchmark_complete": True},
                ),
            ]
            with patch.dict(
                os.environ,
                {
                    "QUANT_DATA_ROOT": str(root),
                    "QLIB_CLEANED_PROVIDER_URI": "",
                },
                clear=False,
            ), patch.object(module, "resolve_dataset", side_effect=resolved) as resolver:
                self.assertEqual(module.qlib_provider_uri("cleaned"), first)
                self.assertEqual(module.qlib_provider_uri("cleaned"), first)

            self.assertEqual(resolver.call_count, 1)
            resolver.assert_called_once_with(
                "qlib_ashare_cleaned_v2",
                quant_data_root=root.resolve(),
                require_basis="qfq",
            )

    def test_ashare_qfq_bars_resolve_native_snapshot_once_per_process(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "quant_data"
            first = root / "parquet" / "native" / "snapshots" / "snapshot-001"
            second = root / "parquet" / "native" / "snapshots" / "snapshot-002"
            resolved = [
                SimpleNamespace(snapshot_path=first, manifest={}),
                SimpleNamespace(snapshot_path=second, manifest={}),
            ]
            with patch.dict(
                os.environ,
                {"QUANT_DATA_ROOT": str(root)},
                clear=False,
            ), patch.object(module, "resolve_dataset", side_effect=resolved) as resolver:
                module._resolve_native_ashare_qfq.cache_clear()
                self.assertEqual(module.ashare_qfq_bars_dir(), first / "bars")
                self.assertEqual(module.ashare_qfq_bars_dir(), first / "bars")

            self.assertEqual(resolver.call_count, 1)
            resolver.assert_called_once_with(
                "ashare_daily_qfq",
                quant_data_root=root.resolve(),
                require_basis="qfq",
            )


if __name__ == "__main__":
    unittest.main()
