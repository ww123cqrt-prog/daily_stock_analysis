# -*- coding: utf-8 -*-
"""Tests for governed quant_data price snapshot resolution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.repositories.quant_price_repo import QuantPriceRepository


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_dataset(root: Path, *, status: str = "ready") -> None:
    dataset_root = root / "parquet" / "etf_1d" / "test_qfq"
    snapshot = dataset_root / "snapshots" / "snapshot-1"
    bars_dir = snapshot / "bars"
    bars_dir.mkdir(parents=True)

    index_rows = []
    for symbol, prices in (("SZ159915", [1.0, 1.1, 1.2]), ("SH513100", [2.0, 2.1, 2.2])):
        path = bars_dir / f"{symbol}.parquet"
        pd.DataFrame(
            {
                "symbol": [symbol] * 3,
                "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "open": prices,
                "close": prices,
                "adjustment": ["qfq"] * 3,
                "source_id": ["test.provider"] * 3,
            }
        ).to_parquet(path, index=False)
        index_rows.append(
            {
                "symbol": symbol,
                "relative_path": f"bars/{symbol}.parquet",
                "row_count": 3,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(index_rows).to_parquet(snapshot / "FILE_INDEX.parquet", index=False)

    snapshot_manifest = {
        "schema_version": 2,
        "dataset_id": "test_daily_qfq",
        "snapshot_id": "snapshot-1",
        "data_kind": "daily_bars",
        "role": "canonical",
        "status": status,
        "price_basis": "qfq",
        "file_index_path": "FILE_INDEX.parquet",
        "latest_trade_date": "2026-01-03",
        "source": {"provider": "test-provider"},
    }
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(snapshot_manifest), encoding="utf-8")
    manifest_hash = _sha256(manifest_path)
    (dataset_root / "LATEST.json").write_text(
        json.dumps(
            {
                "dataset_id": "test_daily_qfq",
                "snapshot_id": "snapshot-1",
                "snapshot_path": "parquet/etf_1d/test_qfq/snapshots/snapshot-1",
                "manifest_sha256": manifest_hash,
            }
        ),
        encoding="utf-8",
    )
    (root / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "datasets": {
                    "test_daily_qfq": {
                        **snapshot_manifest,
                        "dataset_root": "parquet/etf_1d/test_qfq",
                        "latest_pointer": "parquet/etf_1d/test_qfq/LATEST.json",
                        "snapshot_manifest_sha256": manifest_hash,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_loads_requested_symbols_from_a_ready_canonical_qfq_snapshot(tmp_path) -> None:
    _write_dataset(tmp_path)

    result = QuantPriceRepository(tmp_path).load(["159915", "513100"])

    assert result.symbols == ("SZ159915", "SH513100")
    assert result.dataset_id == "test_daily_qfq"
    assert result.snapshot_id == "snapshot-1"
    assert result.price_basis == "qfq"
    assert result.provider == "test-provider"
    assert len(result.bars) == 6


def test_rejects_a_dataset_that_is_not_ready(tmp_path) -> None:
    _write_dataset(tmp_path, status="blocked")

    with pytest.raises(ValueError, match="No ready canonical qfq dataset"):
        QuantPriceRepository(tmp_path).load(["159915", "513100"])


def test_rejects_symbols_missing_from_the_canonical_snapshot(tmp_path) -> None:
    _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="600519"):
        QuantPriceRepository(tmp_path).load(["600519"])
