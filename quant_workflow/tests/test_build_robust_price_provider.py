import importlib.util
import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MEMBERSHIP_HASH = "8be0b74b3e836e9073fc786937b5d9ece8e36494e530cbd07c6e5a608b681cbc"
CALENDAR = ["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"]


def load_script(name: str):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_price(path: Path, close_offset: float = 0.0) -> None:
    close = [10.1 + close_offset, 10.2 + close_offset]
    pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [value - 0.1 for value in close],
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1000.0, 1001.0],
        }
    ).to_parquet(path, index=False)


def imported_price_bundle(tmp_path: Path) -> Path:
    validator = load_script("validate_robust_price_bundle")
    importer = load_script("import_robust_price_bundle")
    source = tmp_path / "source"
    source.mkdir()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("000001\n600000\n", encoding="utf-8")
    write_price(source / "000001.parquet")
    write_price(source / "600000.parquet", close_offset=10.0)
    manifest = validator.build_bundle_manifest(
        price_dir=source,
        symbols_file=symbols_file,
        start_date="2020-01-01",
        end_date="2020-12-31",
        expected_symbol_count=2,
        source="baostock",
        adjustment="qfq",
        membership_snapshot_hash=MEMBERSHIP_HASH,
    )
    manifest_path = tmp_path / "source_manifest.json"
    validator.write_manifest(manifest_path, manifest)
    result = importer.import_price_bundle(
        source_dir=source,
        source_manifest=manifest_path,
        destination_root=tmp_path / "price_bundles",
        membership_symbols={"000001", "600000"},
        membership_snapshot_hash=MEMBERSHIP_HASH,
    )
    return Path(result["bundle_dir"])


def write_bin(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.asarray([0.0, *values], dtype="<f4").tofile(stream)


def benchmark_provider(tmp_path: Path) -> Path:
    provider = tmp_path / "benchmark_provider"
    calendar = provider / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True)
    calendar.write_text("\n".join(CALENDAR) + "\n", encoding="utf-8")
    instruments = provider / "instruments" / "all.txt"
    instruments.parent.mkdir(parents=True)
    instruments.write_text("", encoding="utf-8")
    for symbol_index, symbol in enumerate(("sh000300", "sh000905", "sh000906")):
        base = 100.0 + symbol_index * 10.0
        fields = {
            "open": [base, base + 1, base + 2, base + 3],
            "high": [base + 1, base + 2, base + 3, base + 4],
            "low": [base - 1, base, base + 1, base + 2],
            "close": [base + 0.5, base + 1.5, base + 2.5, base + 3.5],
            "volume": [1000.0, 1001.0, 1002.0, 1003.0],
            "factor": [1.0, 1.0, 1.0, 1.0],
        }
        for field, values in fields.items():
            write_bin(provider / "features" / symbol / f"{field}.day.bin", values)
    return provider


def test_builds_cleaned_provider_with_verified_benchmarks(tmp_path: Path) -> None:
    module = load_script("build_robust_price_provider")
    bundle = imported_price_bundle(tmp_path)
    benchmark = benchmark_provider(tmp_path)

    result = module.build_robust_price_provider(
        price_bundle_dir=bundle,
        benchmark_provider=benchmark,
        output_root=tmp_path / "products",
        required_start_date="2020-01-01",
    )

    product = Path(result["product_dir"])
    provider = Path(result["qlib_provider"])
    cleaned = Path(result["cleaned_parquet"])
    instruments = (provider / "instruments" / "all.txt").read_text(encoding="utf-8")
    cleaning_summary = json.loads(
        (product / "anomalies" / "cleaning_summary.json").read_text(
            encoding="utf-8"
        )
    )
    build_metadata = json.loads(
        (product / "BUILD.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "completed"
    assert result["reused"] is False
    assert len(list(cleaned.glob("*.parquet"))) == 2
    assert (provider / "calendars" / "day.txt").read_bytes() == (
        benchmark / "calendars" / "day.txt"
    ).read_bytes()
    assert "SH000300" not in instruments
    assert "SH000905" not in instruments
    assert "SH000906" not in instruments
    assert (provider / "features" / "sh000906" / "close.day.bin").is_file()
    assert (product / "anomalies" / "data_anomaly_stocks.csv").is_file()
    assert cleaning_summary["source_dir"] == "imported_price_bundle/prices"
    assert cleaning_summary["cleaned_dir"] == "cleaned_parquet"
    assert build_metadata["benchmark_result"]["provider_dir"] == "qlib_provider"
    assert ".tmp" not in json.dumps(cleaning_summary)
    assert ".tmp" not in json.dumps(build_metadata)
    assert not product.stat().st_mode & stat.S_IWUSR


def test_reuses_identical_immutable_product(tmp_path: Path) -> None:
    module = load_script("build_robust_price_provider")
    bundle = imported_price_bundle(tmp_path)
    benchmark = benchmark_provider(tmp_path)
    output_root = tmp_path / "products"

    first = module.build_robust_price_provider(
        price_bundle_dir=bundle,
        benchmark_provider=benchmark,
        output_root=output_root,
        required_start_date="2020-01-01",
    )
    second = module.build_robust_price_provider(
        price_bundle_dir=bundle,
        benchmark_provider=benchmark,
        output_root=output_root,
        required_start_date="2020-01-01",
    )

    assert second["reused"] is True
    assert second["product_dir"] == first["product_dir"]
