import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ApplyDataCleaningTest(unittest.TestCase):
    def test_cleaning_plan_excludes_blockers_and_masks_review_events(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            stocks_csv = base / "stocks.csv"
            events_csv = base / "events.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "000001",
                        "severity": "blocker",
                        "cleaning_action": "exclude_symbol_from_training_until_repaired",
                    },
                    {
                        "symbol": "000002",
                        "severity": "review",
                        "cleaning_action": "review_and_winsorize_or_mask_event_dates",
                    },
                    {
                        "symbol": "000003",
                        "severity": "review",
                        "cleaning_action": "keep_but_mask_or_impute_volume",
                    },
                ]
            ).to_csv(stocks_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "symbol": "000002",
                        "issue_type": "abs_return_gt_50pct",
                        "severity": "review",
                        "sample_dates": "2024-01-03,2024-01-04",
                    },
                    {
                        "symbol": "000003",
                        "issue_type": "non_positive_volume",
                        "severity": "review",
                        "sample_dates": "2024-01-03",
                    },
                ]
            ).to_csv(events_csv, index=False)

            plan = module.load_cleaning_plan(stocks_csv, events_csv)

        self.assertIn("000001", plan.excluded_symbols)
        self.assertIn(pd.Timestamp("2024-01-03"), plan.event_dates_by_symbol["000002"])
        self.assertIn("000003", plan.volume_mask_symbols)

    def test_clean_frame_masks_event_rows_and_non_positive_volume_without_dropping_prices(self):
        module = load_script("apply_data_cleaning")
        plan = module.CleaningPlan(
            excluded_symbols=set(),
            volume_mask_symbols={"000003"},
            event_dates_by_symbol={"000003": {pd.Timestamp("2024-01-04")}},
        )
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "open": [10.0, 10.2, 20.0],
                "high": [10.5, 10.4, 20.5],
                "low": [9.8, 10.0, 19.8],
                "close": [10.1, 10.3, 20.2],
                "volume": [1000.0, 0.0, 2000.0],
                "factor": [1.0, 1.0, 1.0],
            }
        )

        cleaned, stats = module.clean_frame("000003", frame, plan)

        self.assertIsNotNone(cleaned)
        assert cleaned is not None
        self.assertEqual(len(cleaned), 3)
        volume_mask_date = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-03"), "volume"].iloc[0]
        event_open = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-04"), "open"].iloc[0]
        self.assertTrue(np.isnan(volume_mask_date))
        self.assertTrue(np.isnan(event_open))
        self.assertEqual(stats.masked_volume_rows, 1)
        self.assertEqual(stats.masked_event_rows, 1)

    def test_build_cleaned_parquet_skips_excluded_symbols_and_writes_summary(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "cleaned"
            src_dir.mkdir()
            frame = pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000.0, 0.0],
                }
            )
            frame.to_parquet(src_dir / "000001.parquet", index=False)
            frame.to_parquet(src_dir / "000002.parquet", index=False)
            plan = module.CleaningPlan(
                excluded_symbols={"000001"},
                volume_mask_symbols={"000002"},
                event_dates_by_symbol={},
            )

            summary = module.build_cleaned_parquet(src_dir, dst_dir, plan, overwrite=True)

            self.assertFalse((dst_dir / "000001.parquet").exists())
            self.assertTrue((dst_dir / "000002.parquet").exists())
            self.assertEqual(summary["excluded_symbols"], 1)
            self.assertEqual(summary["written_symbols"], 1)
            self.assertEqual(summary["masked_volume_rows"], 1)

    def test_build_cleaned_parquet_dry_run_does_not_write_outputs(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "cleaned"
            src_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                }
            ).to_parquet(src_dir / "000001.parquet", index=False)

            summary = module.build_cleaned_parquet(
                src_dir,
                dst_dir,
                module.CleaningPlan(),
                overwrite=True,
                dry_run=True,
            )

            self.assertFalse(dst_dir.exists())
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["would_write_symbols"], 1)
            self.assertEqual(summary["written_symbols"], 0)

    def test_build_cleaned_parquet_no_overwrite_preserves_existing_symbol_file(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "cleaned"
            src_dir.mkdir()
            dst_dir.mkdir()
            source = pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                }
            )
            existing = pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "open": [99.0],
                    "high": [99.0],
                    "low": [99.0],
                    "close": [99.0],
                    "volume": [99.0],
                }
            )
            source.to_parquet(src_dir / "000001.parquet", index=False)
            existing.to_parquet(dst_dir / "000001.parquet", index=False)

            summary = module.build_cleaned_parquet(
                src_dir,
                dst_dir,
                module.CleaningPlan(),
                overwrite=False,
            )

            preserved = pd.read_parquet(dst_dir / "000001.parquet")
            self.assertEqual(float(preserved["open"].iloc[0]), 99.0)
            self.assertEqual(summary["skipped_existing_symbols"], 1)
            self.assertEqual(summary["written_symbols"], 0)

    def test_copy_benchmark_features_preserves_index_without_polluting_instruments(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source_provider = base / "source"
            target_provider = base / "target"
            source_feature = source_provider / "features" / "sh000300"
            source_feature.mkdir(parents=True)
            (source_feature / "close.day.bin").write_bytes(np.array([0.0, 1.0, 2.0], dtype="<f4").tobytes())

            for provider in [source_provider, target_provider]:
                calendar = provider / "calendars" / "day.txt"
                calendar.parent.mkdir(parents=True, exist_ok=True)
                calendar.write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")

            instruments = target_provider / "instruments"
            instruments.mkdir(parents=True)
            (instruments / "all.txt").write_text("SH000300\t2024-01-02\t2024-01-03\nSZ000001\t2024-01-02\t2024-01-03\n", encoding="utf-8")
            (instruments / "csi300.txt").write_text("SH000300\t2024-01-02\t2024-01-03\nSZ000001\t2024-01-02\t2024-01-03\n", encoding="utf-8")

            copied = module.copy_benchmark_features(source_provider, target_provider, {"SH000300"})

            self.assertEqual(copied, ["SH000300"])
            self.assertTrue((target_provider / "features" / "sh000300" / "close.day.bin").exists())
            self.assertNotIn("SH000300", (target_provider / "instruments" / "all.txt").read_text(encoding="utf-8"))
            self.assertNotIn("SH000300", (target_provider / "instruments" / "csi300.txt").read_text(encoding="utf-8"))

    def test_copy_benchmark_features_reindexes_source_extra_day_by_calendar_date(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            target = base / "target"
            for root, text in [
                (source, "2026-07-17\n2026-07-20\n2026-07-21\n"),
                (target, "2026-07-17\n2026-07-21\n"),
            ]:
                path = root / "calendars" / "day.txt"
                path.parent.mkdir(parents=True)
                path.write_text(text, encoding="utf-8")
            feature = source / "features" / "sh000300" / "close.day.bin"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(np.array([0.0, 3900.0, 3910.0, 3920.0], dtype="<f4").tobytes())

            module.copy_benchmark_features(source, target, {"SH000300"})
            values = np.fromfile(target / "features" / "sh000300" / "close.day.bin", dtype="<f4")

        self.assertEqual(values.tolist(), [0.0, 3900.0, 3920.0])

    def test_copy_benchmark_features_rejects_target_day_missing_from_source_calendar(self):
        module = load_script("apply_data_cleaning")
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            target = base / "target"
            for root, text in [
                (source, "2026-07-17\n2026-07-20\n"),
                (target, "2026-07-17\n2026-07-20\n2026-07-21\n"),
            ]:
                path = root / "calendars" / "day.txt"
                path.parent.mkdir(parents=True)
                path.write_text(text, encoding="utf-8")
            feature = source / "features" / "sh000300" / "close.day.bin"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(np.array([0.0, 3900.0, 3910.0], dtype="<f4").tobytes())

            with self.assertRaisesRegex(ValueError, "target calendar dates missing"):
                module.copy_benchmark_features(source, target, {"SH000300"})


if __name__ == "__main__":
    unittest.main()
