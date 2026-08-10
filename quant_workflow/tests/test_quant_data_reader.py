from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def publish_raw_fixture(tmp_path: Path) -> tuple[Path, object]:
    contract = load_module("quant_data_contract")
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    snapshot = quant_data / "staging" / "run" / "raw-001"
    bars = snapshot / "bars"
    bars.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05"]),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10100.0, 11220.0, 12360.0],
            "source_id": ["test.raw"] * 3,
        }
    ).to_parquet(bars / "SH600000.parquet", index=False)
    contract.finalize_price_snapshot(
        snapshot,
        dataset_id="ashare_daily_raw",
        snapshot_id=snapshot.name,
        role="canonical",
        status="ready",
        source={"provider": "test", "interface": "fixture"},
        price_basis="raw",
        adjustment={"method": "none"},
        producer={"script": "test_quant_data_reader.py", "version": "1"},
        source_retrieved_at="2026-08-06T10:00:00+08:00",
    )
    contract.write_root_manifest(quant_data, {})
    contract.publish_snapshot_group(
        [(snapshot, quant_data / "parquet" / "ashare_1d" / "ashare_daily_raw")],
        quant_data_root=quant_data,
        update_root_catalog=True,
    )
    return quant_data, contract


def test_read_bars_uses_governed_index_and_inclusive_date_range(tmp_path: Path) -> None:
    quant_data, _ = publish_raw_fixture(tmp_path)
    reader = load_module("quant_data_reader")

    frame = reader.read_bars(
        "ashare_daily_raw",
        "600000.SH",
        quant_data_root=quant_data,
        price_basis="raw",
        fields=["close", "volume"],
        start="2026-08-04",
        end="2026-08-05",
    )

    assert frame.columns.tolist() == ["date", "close", "volume"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-04",
        "2026-08-05",
    ]
    assert frame["close"].tolist() == [10.2, 10.3]


def test_read_bars_rejects_unknown_fields_without_reading_another_path(
    tmp_path: Path,
) -> None:
    quant_data, _ = publish_raw_fixture(tmp_path)
    reader = load_module("quant_data_reader")

    with pytest.raises(reader.DataContractError, match="bar field is unavailable"):
        reader.read_bars(
            "ashare_daily_raw",
            "SH600000",
            quant_data_root=quant_data,
            price_basis="raw",
            fields=["secret"],
        )


@pytest.mark.parametrize("symbol", ["000001", "../../DATA_MANIFEST.json"])
def test_read_bars_rejects_unknown_or_unsafe_symbol(
    tmp_path: Path,
    symbol: str,
) -> None:
    quant_data, _ = publish_raw_fixture(tmp_path)
    reader = load_module("quant_data_reader")

    with pytest.raises(reader.DataContractError, match="bar symbol is unavailable"):
        reader.read_bars(
            "ashare_daily_raw",
            symbol,
            quant_data_root=quant_data,
            price_basis="raw",
        )


def test_build_update_universe_is_bound_to_the_frozen_snapshot(
    tmp_path: Path,
) -> None:
    quant_data, contract = publish_raw_fixture(tmp_path)
    reader = load_module("quant_data_reader")

    payload = reader.build_update_universe(
        "ashare_daily_raw",
        quant_data_root=quant_data,
        price_basis="raw",
    )

    resolved = contract.resolve_dataset(
        "ashare_daily_raw",
        quant_data_root=quant_data,
        require_basis="raw",
    )
    assert payload == {
        "schema_version": 1,
        "dataset_id": "ashare_daily_raw",
        "snapshot_id": resolved.manifest["snapshot_id"],
        "manifest_sha256": contract.sha256_file(
            resolved.snapshot_path / "DATASET_MANIFEST.json"
        ),
        "latest_trade_date": "2026-08-05",
        "symbol_count": 1,
        "all": ["SH600000"],
    }
