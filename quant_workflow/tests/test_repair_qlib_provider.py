import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "repair_qlib_provider.py"
    spec = importlib.util.spec_from_file_location("repair_qlib_provider", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RepairQlibProviderTest(unittest.TestCase):
    def test_dry_run_reports_repairs_without_mutating_provider(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir)
            feature_dir = provider / "features" / "000001"
            feature_dir.mkdir(parents=True)
            inst_file = provider / "instruments" / "all.txt"
            inst_file.parent.mkdir()
            original = "000001\t2024-01-02\t2024-01-03\nSH000300\t2024-01-02\t2024-01-03\n"
            inst_file.write_text(original, encoding="utf-8")

            dir_changes = module.repair_feature_dirs(provider, dry_run=True)
            inst_changes = module.repair_instrument_file(inst_file, dry_run=True)

            self.assertEqual(dir_changes, [("000001", "SZ000001")])
            self.assertIn(("000001", "SZ000001"), inst_changes)
            self.assertIn(("SH000300", "<removed-benchmark>"), inst_changes)
            self.assertTrue(feature_dir.exists())
            self.assertFalse((provider / "features" / "sz000001").exists())
            self.assertEqual(inst_file.read_text(encoding="utf-8"), original)

    def test_repairs_feature_dirs_instruments_and_orphans(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            provider = Path(temp_dir)
            for name in ["000001", "600000", "123456", "sh000300"]:
                feature_dir = provider / "features" / name
                feature_dir.mkdir(parents=True)
                (feature_dir / "close.day.bin").write_bytes(b"bin")
            inst_dir = provider / "instruments"
            inst_dir.mkdir()
            (inst_dir / "all.txt").write_text(
                "000001\t2024-01-02\t2024-01-03\n"
                "600000\t2024-01-02\t2024-01-03\n"
                "SH000300\t2024-01-02\t2024-01-03\n",
                encoding="utf-8",
            )
            (inst_dir / "csi300.txt").write_text(
                "000001\t2024-01-02\t2024-01-03\n"
                "999999\t2024-01-02\t2024-01-03\n"
                "SH000300\t2024-01-02\t2024-01-03\n",
                encoding="utf-8",
            )

            dir_changes = module.repair_feature_dirs(provider, dry_run=False)
            all_changes = module.repair_instrument_file(inst_dir / "all.txt", dry_run=False)
            csi_changes = module.repair_instrument_file(inst_dir / "csi300.txt", dry_run=False)
            allowed = {sym for sym, _, _ in module.read_instruments(inst_dir / "all.txt")}
            filtered = module.filter_instrument_file(inst_dir / "csi300.txt", allowed, dry_run=False)
            orphans = module.remove_orphan_feature_dirs(provider, dry_run=False)

            all_text = (inst_dir / "all.txt").read_text(encoding="utf-8")
            csi_text = (inst_dir / "csi300.txt").read_text(encoding="utf-8")
            orphan_exists = (provider / "features" / "123456").exists()
            sz_feature_exists = (provider / "features" / "sz000001" / "close.day.bin").exists()
            sh_feature_exists = (provider / "features" / "sh600000" / "close.day.bin").exists()

        self.assertIn(("000001", "SZ000001"), dir_changes)
        self.assertIn(("600000", "SH600000"), dir_changes)
        self.assertIn(("000001", "SZ000001"), all_changes)
        self.assertIn(("SH000300", "<removed-benchmark>"), all_changes)
        self.assertIn(("999999", "<removed-not-in-all>"), filtered)
        self.assertIn("123456", orphans)
        self.assertIn("SZ000001\t2024-01-02\t2024-01-03", all_text)
        self.assertIn("SH600000\t2024-01-02\t2024-01-03", all_text)
        self.assertNotIn("SH000300", all_text)
        self.assertIn("SZ000001\t2024-01-02\t2024-01-03", csi_text)
        self.assertNotIn("999999", csi_text)
        self.assertFalse(orphan_exists)
        self.assertTrue(sz_feature_exists)
        self.assertTrue(sh_feature_exists)

    def test_missing_provider_dry_run_reports_noop_summary(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing_provider"
            buffer = io.StringIO()
            argv = ["repair_qlib_provider.py", "--provider", str(missing), "--dry-run"]

            with patch.object(sys, "argv", argv), redirect_stdout(buffer):
                self.assertEqual(module.main(), 0)

        output = buffer.getvalue()
        self.assertIn("DRY RUN: provider not found", output)
        self.assertIn("Feature directory renames: 0", output)
        self.assertIn("csi300.txt: 0 metadata changes", output)


if __name__ == "__main__":
    unittest.main()
