import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def publish_qfq_fixture(root: Path) -> None:
    contract = load_script("quant_data_contract")
    snapshot = root / "staging" / "price-run" / "qfq-source-001"
    bars = snapshot / "bars"
    factors = snapshot / "factors"
    bars.mkdir(parents=True)
    factors.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000.0, 1100.0],
            "factor": [1.0, 1.0],
            "source_id": ["baostock", "baostock"],
        }
    ).to_parquet(bars / "SH600000.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "factor": [1.0, 1.0],
        }
    ).to_parquet(factors / "SH600000.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "event_date": pd.to_datetime(["2026-07-23"]),
            "action_type": ["adjustment_factor_event"],
        }
    ).to_parquet(snapshot / "corporate_actions.parquet", index=False)
    contract.finalize_price_snapshot(
        snapshot,
        dataset_id="ashare_daily_qfq",
        snapshot_id="qfq-source-001",
        role="canonical",
        status="ready",
        source={"provider": "baostock", "interface": "raw_factor"},
        price_basis="qfq",
        adjustment={
            "method": "local_factor",
            "adjusted_at": "2026-07-25T23:00:00+08:00",
            "adjustment_as_of": "2026-07-24",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_factor_equals_1",
            "algorithm_version": "test-v1",
        },
        producer={"script": "test", "version": "1"},
        source_retrieved_at="2026-07-25T22:59:00+08:00",
    )
    dataset_root = root / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    published = contract.publish_snapshot(
        snapshot,
        dataset_root=dataset_root,
        quant_data_root=root,
    )
    contract.write_root_manifest(
        root,
        {
            "ashare_daily_qfq": contract.catalog_entry_from_snapshot(
                published,
                quant_data_root=root,
                dataset_root=dataset_root,
            )
        },
    )


def test_build_and_publish_qlib_v2_tracks_exact_qfq_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")

    published = module.build_qlib_v2(quant_data_root=root, publish=True)
    resolved = contract.resolve_dataset(
        "qlib_ashare_v2",
        quant_data_root=root,
        require_basis="qfq",
    )
    provider = json.loads(
        (published / "PROVIDER_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert published == resolved.snapshot_path
    assert resolved.manifest["latest_trade_date"] == "2026-07-24"
    assert provider["source_snapshot"]["snapshot_id"] == "qfq-source-001"
    assert (published / "features" / "sh600000" / "close.day.bin").is_file()
