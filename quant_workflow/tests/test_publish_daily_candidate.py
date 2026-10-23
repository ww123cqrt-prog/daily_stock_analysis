import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "publish_daily_candidate.py"
    spec = importlib.util.spec_from_file_location("publish_daily_candidate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_tree(path: Path, value: str) -> None:
    path.mkdir(parents=True)
    (path / "sentinel.txt").write_text(value, encoding="utf-8")


class PublishDailyCandidateTest(unittest.TestCase):
    def test_restore_manifest_fsyncs_restored_file_and_parent_directory(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quant_data" / "DATA_MANIFEST.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"new")

            with (
                patch.object(module.os, "fsync") as fsync,
                patch.object(
                    module, "_fsync_directory", wraps=module._fsync_directory
                ) as fsync_directory,
            ):
                module._restore_manifest(path, b"old")

            self.assertEqual(path.read_bytes(), b"old")
            fsync_directory.assert_called_once_with(path.parent)
            self.assertEqual(fsync.call_count, 1 if module.os.name == "nt" else 2)

    def test_restore_manifest_fsyncs_parent_after_deletion(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quant_data" / "DATA_MANIFEST.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"new")

            with (
                patch.object(module.os, "fsync") as fsync,
                patch.object(
                    module, "_fsync_directory", wraps=module._fsync_directory
                ) as fsync_directory,
            ):
                module._restore_manifest(path, None)

            self.assertFalse(path.exists())
            fsync_directory.assert_called_once_with(path.parent)
            self.assertEqual(fsync.call_count, 0 if module.os.name == "nt" else 1)

    def test_legacy_publisher_refuses_to_replace_v2_compatibility_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(module, "resolve_dataset", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "V2 canonical data is ready"):
                    module.publish_candidates(
                        candidates={},
                        targets={},
                        backup_root=root / "backups",
                        quant_data_root=root / "quant_data",
                        reports_root=root / "reports",
                    )

    def test_legacy_publisher_fails_closed_when_manifest_is_corrupt_but_v2_markers_exist(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            quant_data.mkdir()
            (quant_data / "DATA_MANIFEST.json").write_text(
                "{broken", encoding="utf-8"
            )
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                targets[name] = root / "official" / name
                write_tree(targets[name], f"v2-{name}")
            metadata = targets["merged_parquet"] / ".metadata"
            metadata.mkdir()
            (metadata / "COMPATIBILITY_MANIFEST.json").write_text(
                json.dumps({"contract": "v1_ashare_qfq_compatibility"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "V2 compatibility"):
                module.publish_candidates(
                    candidates={},
                    targets=targets,
                    backup_root=root / "backups",
                    quant_data_root=quant_data,
                    reports_root=root / "reports",
                )

    def test_corrupt_provider_manifest_is_still_v2_evidence(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            quant_data.mkdir()
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                targets[name] = root / "official" / name
                write_tree(targets[name], f"v2-{name}")
            (targets["cleaned_qlib"] / "PROVIDER_MANIFEST.json").write_text(
                "{broken", encoding="utf-8"
            )

            with patch.object(module, "resolve_dataset", side_effect=ValueError("bad catalog")):
                with self.assertRaisesRegex(RuntimeError, "V2 compatibility"):
                    module.publish_candidates(
                        candidates={},
                        targets=targets,
                        backup_root=root / "backups",
                        quant_data_root=quant_data,
                        reports_root=root / "reports",
                    )

    def test_cleaned_parquet_compatibility_marker_is_v2_evidence(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            quant_data.mkdir()
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                targets[name] = root / "official" / name
                write_tree(targets[name], f"v2-{name}")
            metadata = targets["cleaned_parquet"] / ".metadata"
            metadata.mkdir()
            (metadata / "COMPATIBILITY_MANIFEST.json").write_text(
                "{broken", encoding="utf-8"
            )

            with patch.object(module, "resolve_dataset", side_effect=ValueError("bad catalog")):
                with self.assertRaisesRegex(RuntimeError, "V2 compatibility"):
                    module.publish_candidates(
                        candidates={},
                        targets=targets,
                        backup_root=root / "backups",
                        quant_data_root=quant_data,
                        reports_root=root / "reports",
                    )

            for name in module.REQUIRED_ARTIFACTS:
                self.assertEqual(
                    (targets[name] / "sentinel.txt").read_text(encoding="utf-8"),
                    f"v2-{name}",
                )

    def test_manifest_failure_rolls_back_every_official_artifact(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {}
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                candidates[name] = root / "staging" / name
                targets[name] = root / "official" / name
                write_tree(candidates[name], f"new-{name}")
                write_tree(targets[name], f"old-{name}")

            def fail_manifest(**_kwargs):
                raise RuntimeError("manifest failed")

            with self.assertRaisesRegex(RuntimeError, "manifest failed"):
                module.publish_candidates(
                    candidates=candidates,
                    targets=targets,
                    backup_root=root / "backups",
                    quant_data_root=root / "quant_data",
                    reports_root=root / "reports",
                    sync_manifest_fn=fail_manifest,
                )

            for name in module.REQUIRED_ARTIFACTS:
                self.assertEqual((targets[name] / "sentinel.txt").read_text(encoding="utf-8"), f"old-{name}")
                self.assertEqual((candidates[name] / "sentinel.txt").read_text(encoding="utf-8"), f"new-{name}")

    def test_legacy_rollback_retries_transient_rename_failure(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {}
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                candidates[name] = root / "staging" / name
                targets[name] = root / "official" / name
                write_tree(candidates[name], f"new-{name}")
                write_tree(targets[name], f"old-{name}")

            original_replace = module.Path.replace
            failed_once = False

            def flaky_replace(path, target):
                nonlocal failed_once
                if (
                    path == targets["cleaned_qlib"]
                    and Path(target) == candidates["cleaned_qlib"]
                    and not failed_once
                ):
                    failed_once = True
                    raise OSError("transient rollback failure")
                return original_replace(path, target)

            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=module.DataContractError("no V2 dataset"),
                ),
                patch.object(module.Path, "replace", new=flaky_replace),
                self.assertRaisesRegex(RuntimeError, "manifest failed"),
            ):
                module.publish_candidates(
                    candidates=candidates,
                    targets=targets,
                    backup_root=root / "backups",
                    quant_data_root=root / "quant_data",
                    reports_root=root / "reports",
                    sync_manifest_fn=lambda **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("manifest failed")
                    ),
                )

            self.assertTrue(failed_once)
            for name in module.REQUIRED_ARTIFACTS:
                self.assertEqual(
                    (targets[name] / "sentinel.txt").read_text(encoding="utf-8"),
                    f"old-{name}",
                )
                self.assertEqual(
                    (candidates[name] / "sentinel.txt").read_text(encoding="utf-8"),
                    f"new-{name}",
                )

    def test_legacy_rollback_failure_reports_publish_and_rollback_errors(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {}
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                candidates[name] = root / "staging" / name
                targets[name] = root / "official" / name
                write_tree(candidates[name], f"new-{name}")
                write_tree(targets[name], f"old-{name}")

            original_replace = module.Path.replace

            def fail_rollback(path, target):
                if (
                    path == targets["cleaned_qlib"]
                    and Path(target) == candidates["cleaned_qlib"]
                ):
                    raise OSError("persistent rollback failure")
                return original_replace(path, target)

            with (
                patch.object(
                    module,
                    "resolve_dataset",
                    side_effect=module.DataContractError("no V2 dataset"),
                ),
                patch.object(module.Path, "replace", new=fail_rollback),
                self.assertRaises(RuntimeError) as raised,
            ):
                module.publish_candidates(
                    candidates=candidates,
                    targets=targets,
                    backup_root=root / "backups",
                    quant_data_root=root / "quant_data",
                    reports_root=root / "reports",
                    sync_manifest_fn=lambda **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("manifest failed")
                    ),
                )

            message = str(raised.exception)
            self.assertIn("RuntimeError", message)
            self.assertIn("OSError", message)
            self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_legacy_publisher_can_publish_when_v2_resolution_fails_without_v2_evidence(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {}
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                candidates[name] = root / "staging" / name
                targets[name] = root / "official" / name
                write_tree(candidates[name], f"new-{name}")

            def sync_manifest(*, quant_data_root, reports_root):
                del reports_root
                quant_data_root.mkdir(parents=True, exist_ok=True)
                path = quant_data_root / "DATA_MANIFEST.json"
                path.write_text("{}", encoding="utf-8")
                return path

            with patch.object(
                module,
                "resolve_dataset",
                side_effect=module.DataContractError("no V2 dataset"),
            ):
                result = module.publish_candidates(
                    candidates=candidates,
                    targets=targets,
                    backup_root=root / "backups",
                    quant_data_root=root / "quant_data",
                    reports_root=root / "reports",
                    sync_manifest_fn=sync_manifest,
                )

            self.assertEqual(result["status"], "published")
            for name in module.REQUIRED_ARTIFACTS:
                self.assertEqual(
                    (targets[name] / "sentinel.txt").read_text(encoding="utf-8"),
                    f"new-{name}",
                )

    def test_success_promotes_all_candidates_and_keeps_rollback_copy(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = {}
            targets = {}
            for name in module.REQUIRED_ARTIFACTS:
                candidates[name] = root / "staging" / name
                targets[name] = root / "official" / name
                write_tree(candidates[name], f"new-{name}")
                write_tree(targets[name], f"old-{name}")

            def sync_manifest(*, quant_data_root, reports_root):
                quant_data_root.mkdir(parents=True, exist_ok=True)
                path = quant_data_root / "DATA_MANIFEST.json"
                path.write_text(json.dumps({"reports_root": str(reports_root)}), encoding="utf-8")
                return path

            result = module.publish_candidates(
                candidates=candidates,
                targets=targets,
                backup_root=root / "backups",
                quant_data_root=root / "quant_data",
                reports_root=root / "reports",
                sync_manifest_fn=sync_manifest,
            )

            for name in module.REQUIRED_ARTIFACTS:
                self.assertEqual((targets[name] / "sentinel.txt").read_text(encoding="utf-8"), f"new-{name}")
                self.assertEqual((root / "backups" / name / "sentinel.txt").read_text(encoding="utf-8"), f"old-{name}")
            self.assertTrue(Path(result["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
