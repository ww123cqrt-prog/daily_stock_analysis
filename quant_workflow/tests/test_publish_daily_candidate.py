import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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
