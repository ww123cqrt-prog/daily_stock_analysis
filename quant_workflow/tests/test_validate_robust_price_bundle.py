import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "validate_robust_price_bundle.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_robust_price_bundle", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_price(path: Path, dates: list[str], *, close_offset: float = 0.0) -> None:
    close = [10.1 + close_offset + index * 0.1 for index in range(len(dates))]
    pd.DataFrame(
        {
            "date": dates,
            "open": [value - 0.1 for value in close],
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [1000.0 + index for index in range(len(dates))],
        }
    ).to_parquet(path, index=False)


def build_valid_fixture(tmp_path: Path) -> tuple[Path, Path]:
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("000001\n600000\n", encoding="utf-8")
    write_price(price_dir / "000001.parquet", ["2020-01-02", "2020-01-03"])
    write_price(
        price_dir / "600000.parquet",
        ["2020-01-02", "2020-01-03"],
        close_offset=10.0,
    )
    return price_dir, symbols_file


def test_builds_and_reverifies_semantic_hash_manifest(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"

    manifest = module.build_bundle_manifest(
        price_dir=price_dir,
        symbols_file=symbols_file,
        start_date="2020-01-01",
        end_date="2020-12-31",
        expected_symbol_count=2,
        source="baostock",
        adjustment="qfq",
    )
    module.write_manifest(manifest_path, manifest)
    verified = module.verify_bundle_manifest(
        price_dir=price_dir,
        manifest_path=manifest_path,
        expected_source="baostock",
        expected_adjustment="qfq",
    )

    assert manifest["file_count"] == 2
    assert manifest["row_count"] == 4
    assert manifest["files"][0]["sha256"]
    assert verified["manifest_sha256"] == module.file_sha256(manifest_path)


def test_rejects_missing_and_unexpected_symbols(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    (price_dir / "600000.parquet").unlink()
    write_price(price_dir / "000002.parquet", ["2020-01-02"])

    with pytest.raises(module.PriceBundleError) as exc_info:
        module.build_bundle_manifest(
            price_dir=price_dir,
            symbols_file=symbols_file,
            start_date="2020-01-01",
            end_date="2020-12-31",
            expected_symbol_count=2,
            source="baostock",
            adjustment="qfq",
        )

    assert exc_info.value.reason_code == "symbol_set_mismatch"
    assert exc_info.value.details["missing_symbols"] == ["600000"]
    assert exc_info.value.details["unexpected_symbols"] == ["000002"]


def test_rejects_duplicate_dates_and_out_of_range_rows(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    write_price(
        price_dir / "000001.parquet",
        ["2019-12-31", "2020-01-02", "2020-01-02"],
    )

    with pytest.raises(module.PriceBundleError) as exc_info:
        module.build_bundle_manifest(
            price_dir=price_dir,
            symbols_file=symbols_file,
            start_date="2020-01-01",
            end_date="2020-12-31",
            expected_symbol_count=2,
            source="baostock",
            adjustment="qfq",
        )

    assert exc_info.value.reason_code in {
        "duplicate_dates",
        "date_out_of_range",
    }


def test_verify_rejects_tampered_parquet(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = module.build_bundle_manifest(
        price_dir=price_dir,
        symbols_file=symbols_file,
        start_date="2020-01-01",
        end_date="2020-12-31",
        expected_symbol_count=2,
        source="baostock",
        adjustment="qfq",
    )
    module.write_manifest(manifest_path, manifest)
    write_price(price_dir / "000001.parquet", ["2020-01-02"])

    with pytest.raises(module.PriceBundleError) as exc_info:
        module.verify_bundle_manifest(
            price_dir=price_dir,
            manifest_path=manifest_path,
            expected_source="baostock",
            expected_adjustment="qfq",
        )

    assert exc_info.value.reason_code == "file_hash_mismatch"


def test_verify_rejects_duplicate_manifest_symbols(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = module.build_bundle_manifest(
        price_dir=price_dir,
        symbols_file=symbols_file,
        start_date="2020-01-01",
        end_date="2020-12-31",
        expected_symbol_count=2,
        source="baostock",
        adjustment="qfq",
    )
    manifest["files"][1] = dict(manifest["files"][0])
    module.write_manifest(manifest_path, manifest)

    with pytest.raises(module.PriceBundleError) as exc_info:
        module.verify_bundle_manifest(
            price_dir=price_dir,
            manifest_path=manifest_path,
            expected_source="baostock",
            expected_adjustment="qfq",
        )

    assert exc_info.value.reason_code == "manifest_invalid"


def test_build_rejects_symlinked_parquet(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    target = price_dir / "600000.parquet"
    outside = tmp_path / "outside.parquet"
    target.replace(outside)
    target.symlink_to(outside)

    with pytest.raises(module.PriceBundleError) as exc_info:
        module.build_bundle_manifest(
            price_dir=price_dir,
            symbols_file=symbols_file,
            start_date="2020-01-01",
            end_date="2020-12-31",
            expected_symbol_count=2,
            source="baostock",
            adjustment="qfq",
        )

    assert exc_info.value.reason_code == "symlink_forbidden"


def test_main_writes_fail_closed_status_without_manifest(tmp_path: Path) -> None:
    module = load_module()
    price_dir, symbols_file = build_valid_fixture(tmp_path)
    (price_dir / "600000.parquet").unlink()
    manifest_path = tmp_path / "manifest.json"
    status_path = tmp_path / "status.json"

    return_code = module.main(
        [
            "--price-dir",
            str(price_dir),
            "--symbols-file",
            str(symbols_file),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-12-31",
            "--expected-symbol-count",
            "2",
            "--manifest-out",
            str(manifest_path),
            "--status-out",
            str(status_path),
        ]
    )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert return_code == 2
    assert status["status"] == "failed"
    assert status["reason_code"] == "symbol_set_mismatch"
    assert not manifest_path.exists()
