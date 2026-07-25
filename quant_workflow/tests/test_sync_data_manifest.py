import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_data_manifest.py"
    spec = importlib.util.spec_from_file_location("sync_data_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SyncDataManifestTest(unittest.TestCase):
    def test_sync_manifest_reflects_actual_cleaned_provider_calendar(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            reports = root / "quant_workflow" / "reports"
            cleaned_parquet = quant_data / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
            cleaned_provider = quant_data / "qlib" / "cn_data_baostock_jqdata_cleaned"
            for directory in (cleaned_parquet, cleaned_provider / "calendars", cleaned_provider / "instruments"):
                directory.mkdir(parents=True, exist_ok=True)
            (cleaned_parquet / "000001.parquet").write_bytes(b"fake")
            (cleaned_provider / "calendars" / "day.txt").write_text(
                "2026-07-08\n2026-07-09\n",
                encoding="utf-8",
            )
            (cleaned_provider / "instruments" / "all.txt").write_text(
                "SZ000001\t2026-07-08\t2026-07-09\n",
                encoding="utf-8",
            )
            reports.mkdir(parents=True)
            (reports / "data_quality_report_latest.md").write_text("# quality", encoding="utf-8")

            manifest_path = module.sync_data_manifest(quant_data_root=quant_data, reports_root=reports)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["official_entries"]["cleaned_parquet"]["file_count"], 1)
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["calendar_last"], "2026-07-09")
        self.assertEqual(manifest["official_entries"]["cleaned_qlib_provider"]["instruments_all"], 1)
        self.assertEqual(manifest["documented_inputs"]["canonical_qlib_provider"]["calendar_last"], "")
        self.assertIn("global_daily", manifest["documented_inputs"])
        self.assertIn("raw_global_daily", manifest["documented_inputs"])
        self.assertIn("raw_ashare_snapshots", manifest["documented_inputs"])
        self.assertIn("raw_ashare_news", manifest["documented_inputs"])

    def test_sync_preserves_robust_entries_and_replaces_stale_checkout_report_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quant_data = root / "quant_data"
            reports = root / "daily_stock_analysis" / "quant_workflow" / "reports"
            quant_data.mkdir()
            reports.mkdir(parents=True)
            (quant_data / "DATA_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "generated_at": "old",
                        "source_report": "/old/daily_stock_analysis-robust/quant_workflow/reports/data_hygiene_inventory_latest.json",
                        "official_entries": {
                            "robust_price_product": {"path": "/shared/immutable", "status": "do_not_move"}
                        },
                        "pending_entries": {"remote": {"status": "pending"}},
                    }
                ),
                encoding="utf-8",
            )

            path = module.sync_data_manifest(quant_data_root=quant_data, reports_root=reports)
            manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["official_entries"]["robust_price_product"]["path"], "/shared/immutable")
        self.assertEqual(manifest["pending_entries"]["remote"]["status"], "pending")
        self.assertEqual(manifest["source_report"], str(reports / "data_hygiene_inventory_latest.json"))
        self.assertNotIn("daily_stock_analysis-robust", json.dumps(manifest))


if __name__ == "__main__":
    unittest.main()
