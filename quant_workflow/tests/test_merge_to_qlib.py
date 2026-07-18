import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "merge_to_qlib.py"
    spec = importlib.util.spec_from_file_location("merge_to_qlib", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MergeToQlibTest(unittest.TestCase):
    def test_build_merged_parquet_no_overwrite_preserves_existing_output_file(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bs_dir = base / "bs"
            jq_dir = base / "jq"
            out_dir = base / "merged"
            bs_dir.mkdir()
            jq_dir.mkdir()
            out_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                }
            ).to_parquet(bs_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["1999-01-01"],
                    "open": [99.0],
                    "high": [99.0],
                    "low": [99.0],
                    "close": [99.0],
                    "volume": [99.0],
                }
            ).to_parquet(out_dir / "000001.parquet", index=False)

            count = module.build_merged_parquet(bs_dir, jq_dir, out_dir, overwrite=False)
            preserved = pd.read_parquet(out_dir / "000001.parquet")

        self.assertEqual(count, 1)
        self.assertEqual(preserved["close"].tolist(), [99.0])

    def test_merge_sources_keeps_base_and_only_fills_missing_dates(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bs_dir = base / "baostock"
            jq_dir = base / "jqdata"
            eastmoney_dir = base / "eastmoney"
            akshare_dir = base / "akshare"
            for directory in [bs_dir, jq_dir, eastmoney_dir, akshare_dir]:
                directory.mkdir()
            pd.DataFrame(
                {
                    "date": ["2026-07-01", "2026-07-02"],
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000.0, 1100.0],
                }
            ).to_parquet(bs_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-07-02", "2026-07-03"],
                    "open": [20.1, 20.2],
                    "high": [20.3, 20.4],
                    "low": [20.0, 20.1],
                    "close": [20.2, 20.3],
                    "volume": [9999.0, 1200.0],
                }
            ).to_parquet(jq_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-07-03", "2026-07-04"],
                    "open": [30.1, 30.2],
                    "high": [30.3, 30.4],
                    "low": [30.0, 30.1],
                    "close": [30.2, 30.3],
                    "volume": [9999.0, 1300.0],
                }
            ).to_parquet(eastmoney_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-07-04", "2026-07-05"],
                    "open": [40.1, 40.2],
                    "high": [40.3, 40.4],
                    "low": [40.0, 40.1],
                    "close": [40.2, 40.3],
                    "volume": [9999.0, 1400.0],
                }
            ).to_parquet(akshare_dir / "000001.parquet", index=False)

            merged = module.merge_one(
                "000001",
                bs_dir / "000001.parquet",
                jq_dir / "000001.parquet",
                eastmoney_dir / "000001.parquet",
                akshare_dir / "000001.parquet",
            )

        self.assertIsNotNone(merged)
        self.assertEqual(merged["date"].dt.strftime("%Y-%m-%d").tolist(), [
            "2026-07-01",
            "2026-07-02",
            "2026-07-03",
            "2026-07-04",
            "2026-07-05",
        ])
        # Base (baostock) is authoritative for the dates it covers (07-01, 07-02);
        # each later source fills ONLY the dates still missing, so its whole row
        # (close + volume together) lands for that new date without overwriting
        # any overlapping base date. 07-03 <- jqdata, 07-04 <- eastmoney,
        # 07-05 <- akshare.
        self.assertEqual(merged["close"].tolist(), [10.1, 10.2, 20.3, 30.3, 40.3])
        self.assertEqual(merged["volume"].tolist(), [1000.0, 1100.0, 1200.0, 1300.0, 1400.0])

    def test_build_merged_parquet_accepts_extra_source_directories_without_renaming_output(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            bs_dir = base / "baostock"
            jq_dir = base / "jqdata"
            eastmoney_dir = base / "eastmoney"
            out_dir = base / "merged_baostock_jqdata"
            for directory in [bs_dir, jq_dir, eastmoney_dir]:
                directory.mkdir()
            pd.DataFrame(
                {
                    "date": ["2026-07-01"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                }
            ).to_parquet(bs_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": ["2026-07-02"],
                    "open": [10.2],
                    "high": [10.4],
                    "low": [10.1],
                    "close": [10.3],
                    "volume": [1100.0],
                }
            ).to_parquet(eastmoney_dir / "000001.parquet", index=False)

            count = module.build_merged_parquet(
                bs_dir,
                jq_dir,
                out_dir,
                overwrite=True,
                extra_source_dirs=[("eastmoney", eastmoney_dir)],
            )
            merged = pd.read_parquet(out_dir / "000001.parquet")

        self.assertEqual(count, 1)
        self.assertEqual(out_dir.name, "merged_baostock_jqdata")
        self.assertEqual(merged["close"].tolist(), [10.1, 10.3])

    def test_convert_to_bin_no_overwrite_preserves_existing_feature_file(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "qlib"
            src_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                    "factor": [1.0],
                }
            ).to_parquet(src_dir / "000001.parquet", index=False)
            feature_dir = dst_dir / "features" / "sz000001"
            feature_dir.mkdir(parents=True)
            existing_close = feature_dir / "close.day.bin"
            existing_close.write_bytes(b"sentinel-close")

            result = module.convert_to_bin(src_dir, dst_dir, overwrite=False)

            self.assertEqual(result, 0)
            self.assertEqual(existing_close.read_bytes(), b"sentinel-close")
            self.assertTrue((dst_dir / "calendars" / "day.txt").exists())

    def test_smoke_fixture_runs_merge_and_convert_in_isolated_directory(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)

            result = module.run_smoke_fixture(base, overwrite=True)

            qlib_dir = base / "qlib"
            merged_dir = base / "merged"
            self.assertEqual(result, 0)
            self.assertTrue((merged_dir / "000001.parquet").exists())
            self.assertTrue((qlib_dir / "calendars" / "day.txt").exists())
            self.assertTrue((qlib_dir / "instruments" / "all.txt").exists())
            self.assertTrue((qlib_dir / "features" / "sz000001" / "close.day.bin").exists())

    def test_convert_to_bin_can_preserve_masked_volume_rows(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "qlib"
            src_dir.mkdir()
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000.0, np.nan],
                    "factor": [1.0, 1.0],
                }
            ).to_parquet(src_dir / "000001.parquet", index=False)

            result = module.convert_to_bin(
                src_dir,
                dst_dir,
                overwrite=True,
                dropna_fields=["open", "close", "high", "low"],
            )

            volume_bin = dst_dir / "features" / "sz000001" / "volume.day.bin"
            values = np.fromfile(volume_bin, dtype="<f")

        self.assertEqual(result, 0)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 1000.0)
        self.assertTrue(np.isnan(values[2]))

    def test_convert_to_bin_can_align_to_verified_external_calendar(self):
        module = load_module()
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            src_dir = base / "merged"
            dst_dir = base / "qlib"
            calendar_path = base / "verified_day.txt"
            src_dir.mkdir()
            calendar_path.write_text(
                "2019-12-31\n2020-01-02\n2020-01-03\n2020-01-06\n",
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "date": ["2020-01-02", "2020-01-03"],
                    "open": [10.0, 10.1],
                    "high": [10.2, 10.3],
                    "low": [9.9, 10.0],
                    "close": [10.1, 10.2],
                    "volume": [1000.0, 1100.0],
                    "factor": [1.0, 1.0],
                }
            ).to_parquet(src_dir / "000001.parquet", index=False)

            result = module.convert_to_bin(
                src_dir,
                dst_dir,
                overwrite=True,
                calendar_path=calendar_path,
            )

            close_bin = np.fromfile(
                dst_dir / "features" / "sz000001" / "close.day.bin",
                dtype="<f4",
            )
            self.assertEqual(result, 0)
            self.assertEqual(
                (dst_dir / "calendars" / "day.txt").read_bytes(),
                calendar_path.read_bytes(),
            )
            self.assertEqual(float(close_bin[0]), 1.0)
            np.testing.assert_allclose(close_bin[1:], [10.1, 10.2])


if __name__ == "__main__":
    unittest.main()
