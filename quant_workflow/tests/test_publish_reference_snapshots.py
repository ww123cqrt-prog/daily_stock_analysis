from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "publish_reference_snapshots.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("publish_reference_snapshots", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def seed_reference_inputs(quant_data: Path) -> None:
    pit = quant_data / "reference" / "pit_csi800"
    (pit / "daily").mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
            "index_code": ["000300.XSHG", "000300.XSHG"],
        }
    ).to_parquet(pit / "daily" / "2025.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "index_code": ["000300.XSHG"],
            "start_date": pd.to_datetime(["2025-01-02"]),
            "end_date": pd.to_datetime(["2025-01-03"]),
        }
    ).to_parquet(pit / "membership_intervals.parquet", index=False)
    (pit / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "baostock",
                "selected": "baostock",
                "date_range": {
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-03",
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )

    reference = quant_data / "reference"
    pd.DataFrame({"symbol": ["SH600000"]}).to_parquet(
        reference / "security_master.parquet", index=False
    )
    pd.DataFrame({"symbol": ["SH600000"]}).to_parquet(
        reference / "industry_intervals.parquet", index=False
    )
    pd.DataFrame({"symbol": ["SH600000"]}).to_parquet(
        reference / "st_status.parquet", index=False
    )
    pd.DataFrame({"symbol": ["SH600000"]}).to_parquet(
        reference / "suspension_status.parquet", index=False
    )

    membership_integrity = importlib.util.spec_from_file_location(
        "_robust_membership_integrity",
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "_robust_membership_integrity.py",
    )
    assert membership_integrity.loader is not None
    membership_module = importlib.util.module_from_spec(membership_integrity)
    membership_integrity.loader.exec_module(membership_module)
    _, membership_hash = membership_module.read_membership_snapshot(
        pit,
        start_date="2025-01-02",
        end_date="2025-01-03",
    )
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "baostock",
                "selected": "baostock",
                "date_range": {
                    "start_date": "2025-01-02",
                    "end_date": "2025-01-03",
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_hash,
                "build_run_id": "reference-test-001",
            }
        ),
        encoding="utf-8",
    )


def seed_fundamental_inputs(quant_data: Path) -> None:
    root = quant_data / "reference" / "fundamentals"
    (root / "daily").mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "report_date": pd.to_datetime(["2024-03-31"]),
            "source_available_date": pd.to_datetime(["2024-04-30"]),
            "effective_date": pd.to_datetime(["2024-05-06"]),
        }
    ).to_parquet(root / "fundamental_events.parquet", index=False)
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-05-06"]),
            "instrument": ["SH600000"],
            "effective_date": pd.to_datetime(["2024-05-06"]),
            "source_available_date": pd.to_datetime(["2024-04-30"]),
            "FUND_QUALITY_ROE": [0.1],
        }
    ).to_parquet(root / "daily" / "2024.parquet", index=False)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source = {
        "schema_version": 1,
        "status": "pass",
        "mixed_sources": False,
        "selected_source": "eastmoney_hsf10",
        "source_contract_version": 3,
        "effective_date_rule": "max_notice_or_update_plus_one_trading_day",
        "data_range": {"start": "2024-05-06", "end": "2024-05-06"},
        "coverage": {"passed": True, "minimum": 0.98},
        "artifacts": {
            "fundamental_events.parquet": digest(root / "fundamental_events.parquet"),
            "daily/2024.parquet": digest(root / "daily" / "2024.parquet"),
        },
    }
    (root / "FUNDAMENTAL_SOURCE.json").write_text(
        json.dumps(source), encoding="utf-8"
    )


def test_existing_pit_and_reference_are_published_as_resolvable_snapshots(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    seed_reference_inputs(quant_data)
    (quant_data / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    result = module.publish_existing_reference_snapshots(
        quant_data_root=quant_data,
        reports_root=tmp_path / "reports",
    )

    assert result["pit_csi800"]["status"] == "published"
    assert result["robust_reference"]["status"] == "published"
    assert (quant_data / "reference" / "pit_csi800" / "LATEST.json").is_file()
    assert (quant_data / "reference" / "LATEST.json").is_file()
    manifest = json.loads((quant_data / "DATA_MANIFEST.json").read_text())
    assert manifest["datasets"]["pit_csi800"]["latest_pointer"] == (
        "reference/pit_csi800/LATEST.json"
    )
    assert manifest["datasets"]["pit_csi800"]["artifact_index_path"] == (
        "ARTIFACT_INDEX.parquet"
    )
    assert "file_index_path" not in manifest["datasets"]["pit_csi800"]
    assert manifest["datasets"]["robust_reference"]["latest_pointer"] == (
        "reference/LATEST.json"
    )


def test_pit_fundamentals_are_registered_with_named_daily_component(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    seed_reference_inputs(quant_data)
    seed_fundamental_inputs(quant_data)
    (quant_data / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    result = module.publish_existing_reference_snapshots(
        quant_data_root=quant_data,
        reports_root=tmp_path / "reports",
    )

    assert result["ashare_fundamentals_pit"]["status"] == "published"
    manifest = json.loads((quant_data / "DATA_MANIFEST.json").read_text())
    entry = manifest["datasets"]["ashare_fundamentals_pit"]
    assert entry["role"] == "canonical"
    assert entry["status"] == "ready"
    assert entry["components"]["daily"]["path"] == "daily"
    assert entry["metadata"]["effective_date_rule"] == (
        "max_notice_or_update_plus_one_trading_day"
    )
