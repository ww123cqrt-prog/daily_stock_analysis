from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_workflow.scripts import build_cn_index_dataset as builder
from quant_workflow.scripts.quant_data_contract import validate_snapshot


SYMBOLS = ["SH000300", "SH000905", "SH000906", "SZ399006", "SH000688"]


def _write_feature(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray([0.0, *values], dtype="<f4").tofile(path)


def _native_fixture(tmp_path: Path) -> tuple[Path, list[str]]:
    native = tmp_path / "native"
    calendar = ["2026-01-02", "2026-01-05", "2026-01-06"]
    (native / "calendars").mkdir(parents=True)
    (native / "calendars" / "day.txt").write_text(
        "\n".join(calendar) + "\n",
        encoding="utf-8",
    )
    for offset, symbol in enumerate(SYMBOLS):
        close = [100.0 + offset, 101.0 + offset, 102.0 + offset]
        values = {
            "open": [value - 0.5 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 1.0 for value in close],
            "close": close,
            "volume": [1000.0, 1100.0, 1200.0],
            "factor": [1.0, 1.0, 1.0],
        }
        for field, field_values in values.items():
            _write_feature(
                native / "features" / symbol.lower() / f"{field}.day.bin",
                field_values,
            )
    (native / "PROVIDER_MANIFEST.json").write_text(
        json.dumps(
            {
                "benchmark_complete": True,
                "benchmarks": {
                    "required_symbols": SYMBOLS,
                    "source": {
                        "provider": "fixture",
                        "interface": "fixture-native",
                        "retrieved_at": "2026-01-07T08:00:00+08:00",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return native, calendar


def test_stage_index_dataset_converts_existing_native_evidence_to_parquet(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    native, calendar = _native_fixture(tmp_path)

    staged = builder.stage_index_dataset_from_native(
        quant_data_root=root,
        native_snapshot=native,
        run_id="index-fixture-001",
    )

    manifest = validate_snapshot(staged, verify_hashes=True)
    assert manifest["dataset_id"] == "cn_index_daily"
    assert manifest["price_basis"] == "derived"
    assert sorted(path.name for path in (staged / "bars").glob("*.parquet")) == sorted(
        f"{symbol}.parquet" for symbol in SYMBOLS
    )
    assert len(pd.read_parquet(staged / "bars" / "SH000300.parquet")) == len(calendar)
    assert not list(staged.rglob("*.bin"))
    assert not (staged / "features").exists()


def test_stage_index_dataset_rejects_incomplete_native_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    native, _ = _native_fixture(tmp_path)
    (native / "features" / "sh000688" / "close.day.bin").unlink()

    with pytest.raises(builder.IndexDatasetError, match="SH000688|incomplete"):
        builder.stage_index_dataset_from_native(
            quant_data_root=root,
            native_snapshot=native,
            run_id="index-fixture-002",
        )
