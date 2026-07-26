import importlib.util
import stat
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "import_robust_price_bundle.py"
MEMBERSHIP_HASH = "8be0b74b3e836e9073fc786937b5d9ece8e36494e530cbd07c6e5a608b681cbc"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "import_robust_price_bundle", SCRIPT_PATH
    )
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


def build_source(tmp_path: Path):
    validator = load_module().validator
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
    return source, manifest_path


def test_imports_to_hash_isolated_read_only_bundle(tmp_path: Path) -> None:
    module = load_module()
    source, manifest_path = build_source(tmp_path)
    destination_root = tmp_path / "bundles"

    result = module.import_price_bundle(
        source_dir=source,
        source_manifest=manifest_path,
        destination_root=destination_root,
        membership_symbols={"000001", "600000"},
        membership_snapshot_hash=MEMBERSHIP_HASH,
    )

    destination = Path(result["price_dir"])
    assert result["status"] == "completed"
    assert result["reused"] is False
    assert (destination / "000001.parquet").is_file()
    assert (destination.parent / "PRICE_BUNDLE_MANIFEST.json").is_file()
    assert (destination.parent / "IMPORT.json").is_file()
    assert not (destination / "000001.parquet").stat().st_mode & stat.S_IWUSR
    assert (source / "000001.parquet").stat().st_mode & stat.S_IWUSR


def test_reuses_same_verified_bundle(tmp_path: Path) -> None:
    module = load_module()
    source, manifest_path = build_source(tmp_path)
    destination_root = tmp_path / "bundles"

    first = module.import_price_bundle(
        source_dir=source,
        source_manifest=manifest_path,
        destination_root=destination_root,
        membership_symbols={"000001", "600000"},
        membership_snapshot_hash=MEMBERSHIP_HASH,
    )
    second = module.import_price_bundle(
        source_dir=source,
        source_manifest=manifest_path,
        destination_root=destination_root,
        membership_symbols={"000001", "600000"},
        membership_snapshot_hash=MEMBERSHIP_HASH,
    )

    assert second["reused"] is True
    assert second["price_dir"] == first["price_dir"]


def test_rejects_price_symbols_that_do_not_match_pit(tmp_path: Path) -> None:
    module = load_module()
    source, manifest_path = build_source(tmp_path)

    with pytest.raises(module.PriceBundleImportError) as exc_info:
        module.import_price_bundle(
            source_dir=source,
            source_manifest=manifest_path,
            destination_root=tmp_path / "bundles",
            membership_symbols={"000001", "600001"},
            membership_snapshot_hash=MEMBERSHIP_HASH,
        )

    assert exc_info.value.reason_code == "pit_symbol_set_mismatch"


def test_rejects_membership_hash_mismatch_before_copy(tmp_path: Path) -> None:
    module = load_module()
    source, manifest_path = build_source(tmp_path)

    with pytest.raises(module.PriceBundleImportError) as exc_info:
        module.import_price_bundle(
            source_dir=source,
            source_manifest=manifest_path,
            destination_root=tmp_path / "bundles",
            membership_symbols={"000001", "600000"},
            membership_snapshot_hash="0" * 64,
        )

    assert exc_info.value.reason_code == "membership_snapshot_hash_mismatch"
    assert not (tmp_path / "bundles").exists()
