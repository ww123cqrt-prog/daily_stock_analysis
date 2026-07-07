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


if __name__ == "__main__":
    unittest.main()
