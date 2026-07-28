import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BENCHMARK_SYMBOLS = ("SH000300", "SH000905", "SH000906")
BENCHMARK_FIELDS = ("open", "high", "low", "close", "volume", "factor")


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
            "factor": [0.5, 1.0],
            "source_id": ["baostock", "baostock"],
        }
    ).to_parquet(bars / "SH600000.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
            "factor": [0.5, 1.0],
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
    (root / "stock_lists.json").write_text(
        json.dumps(
            {
                "all": ["600000.XSHG"],
                "csi300": ["600000.XSHG"],
                "csi500": ["600000.XSHG"],
                "csi800": ["600000.XSHG"],
            }
        ),
        encoding="utf-8",
    )


def write_benchmark_provider(
    root: Path,
    *,
    source_snapshot: dict[str, str],
    calendar: tuple[str, ...] = ("2026-07-23", "2026-07-24"),
    missing: tuple[str, str, str] | None = None,
) -> Path:
    provider = root / "qlib" / "cn_data_baostock_jqdata"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(calendar) + "\n", encoding="utf-8"
    )
    (provider / "instruments" / "all.txt").write_text(
        f"SH600000\t{calendar[0]}\t{calendar[-1]}\n", encoding="utf-8"
    )
    for symbol_index, symbol in enumerate(BENCHMARK_SYMBOLS, start=1):
        feature_dir = provider / "features" / symbol.lower()
        feature_dir.mkdir(parents=True)
        for field in BENCHMARK_FIELDS:
            if field == "factor":
                values = np.ones(len(calendar), dtype="<f4")
            elif field == "volume":
                values = np.full(len(calendar), 1000.0 * symbol_index, dtype="<f4")
            else:
                values = np.arange(
                    10.0 * symbol_index,
                    10.0 * symbol_index + len(calendar),
                    dtype="<f4",
                )
            if missing == (symbol, field, calendar[-1]):
                values[-1] = np.nan
            np.concatenate((np.array([0.0], dtype="<f4"), values)).tofile(
                feature_dir / f"{field}.day.bin"
            )
    (provider / "PROVIDER_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "ready",
                "price_basis": "qfq",
                "source_snapshot": source_snapshot,
                "conversion": {"calendar_sha256": "filled-by-test-source"},
            }
        ),
        encoding="utf-8",
    )
    return provider


def benchmark_source_evidence() -> dict[str, str]:
    return {
        "provider": "tencent",
        "interface": "qt.gtimg.cn historical daily",
        "source_dataset_id": "qlib_ashare_merged_compatibility",
        "source_path": "qlib/cn_data_baostock_jqdata",
        "source_run_id": "v1-compat-test",
        "evidence_path": "runtime/v2_compatibility/runs/v1-compat-test/candidate_summary.json",
        "evidence_sha256": "b" * 64,
    }


def tencent_benchmark_frames(
    calendar: tuple[str, ...] = ("2026-07-23", "2026-07-24"),
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol_index, symbol in enumerate(BENCHMARK_SYMBOLS, start=1):
        base = float(symbol_index * 10)
        frames[symbol] = pd.DataFrame(
            {
                "open": [base + offset for offset in range(len(calendar))],
                "high": [base + offset + 0.2 for offset in range(len(calendar))],
                "low": [base + offset - 0.1 for offset in range(len(calendar))],
                "close": [base + offset + 0.1 for offset in range(len(calendar))],
                "volume": [1000.0 * symbol_index] * len(calendar),
            },
            index=list(calendar),
        )
    return frames


def test_build_and_publish_qlib_v2_tracks_exact_qfq_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    qfq = contract.resolve_dataset("ashare_daily_qfq", quant_data_root=root)
    benchmark_provider = write_benchmark_provider(
        root,
        source_snapshot=dict(qfq.manifest["upstream_snapshots"][0])
        if qfq.manifest.get("upstream_snapshots")
        else {
            "dataset_id": qfq.dataset_id,
            "snapshot_id": qfq.manifest["snapshot_id"],
            "content_sha256": qfq.manifest["content_sha256"],
            "price_basis": qfq.manifest["price_basis"],
        },
    )

    published = module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=benchmark_provider,
        benchmark_source=benchmark_source_evidence(),
    )
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
    assert resolved.manifest["benchmark_complete"] is True
    assert provider["benchmark_complete"] is True
    assert resolved.catalog_entry["benchmark_complete"] is True
    assert set(resolved.manifest["benchmarks"]["symbols"]) == set(BENCHMARK_SYMBOLS)
    assert resolved.manifest["benchmarks"] == provider["benchmarks"]
    assert resolved.manifest["benchmarks"]["source"]["provider"] == "tencent"
    assert len(resolved.manifest["benchmarks"]["calendar_sha256"]) == 64
    for symbol in BENCHMARK_SYMBOLS:
        benchmark = resolved.manifest["benchmarks"]["symbols"][symbol]
        assert benchmark["coverage"] == {
            "start": "2026-07-23",
            "end": "2026-07-24",
            "observations": 2,
        }
        assert set(benchmark["features"]) == set(BENCHMARK_FIELDS)
        for field, evidence in benchmark["features"].items():
            assert evidence["relative_path"] == (
                f"features/{symbol.lower()}/{field}.day.bin"
            )
            assert len(evidence["sha256"]) == 64
    factor_values = np.fromfile(
        published / "features" / "sh600000" / "factor.day.bin", dtype="<f4"
    )
    assert factor_values[1:].tolist() == [1.0, 1.0]
    assert provider["conversion"]["factor_semantics"] == "unit_factor_prices_already_qfq"
    assert (published / "instruments" / "csi300.txt").read_text(
        encoding="utf-8"
    ) == "SH600000\t2026-07-23\t2026-07-24\n"
    assert set(resolved.manifest["instrument_pools"]["pools"]) == {
        "csi300",
        "csi500",
        "csi800",
    }
    assert resolved.manifest["instrument_pools"] == provider["instrument_pools"]


def _qfq_source_contract(contract, root: Path) -> dict[str, str]:
    qfq = contract.resolve_dataset("ashare_daily_qfq", quant_data_root=root)
    return {
        "dataset_id": qfq.dataset_id,
        "snapshot_id": str(qfq.manifest["snapshot_id"]),
        "content_sha256": str(qfq.manifest["content_sha256"]),
        "price_basis": str(qfq.manifest["price_basis"]),
    }


def test_build_rejects_benchmark_provider_with_different_calendar(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
        calendar=("2026-07-23", "2026-07-25"),
    )

    with pytest.raises(module.DataContractError, match="calendar"):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=provider,
            benchmark_source=benchmark_source_evidence(),
        )


def test_default_build_regenerates_benchmarks_when_compatibility_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    stale_source = _qfq_source_contract(contract, root)
    stale_source["snapshot_id"] = "older-qfq-snapshot"
    write_benchmark_provider(root, source_snapshot=stale_source)
    calls: list[tuple[str, str]] = []

    def fake_download(start_date: str, end_date: str, **_kwargs):
        calls.append((start_date, end_date))
        return tencent_benchmark_frames()

    monkeypatch.setattr(module, "download_tencent_benchmarks", fake_download)

    snapshot = module.build_qlib_v2(quant_data_root=root)
    manifest = json.loads(
        (snapshot / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    source = manifest["benchmarks"]["source"]
    evidence_path = root / source["evidence_path"]

    assert calls == [("2026-07-23", "2026-07-24")]
    assert source["provider"] == "tencent"
    assert source["source_dataset_id"] == "tencent_index_daily"
    assert source["source_manifest_sha256"] == module.sha256_file(evidence_path)
    assert source["evidence_sha256"] == module.sha256_file(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "complete"
    assert evidence["calendar_sha256"] == manifest["benchmarks"]["calendar_sha256"]
    assert set(evidence["symbols"]) == set(BENCHMARK_SYMBOLS)
    for symbol in BENCHMARK_SYMBOLS:
        assert evidence["symbols"][symbol]["coverage"]["end"] == "2026-07-24"
        assert (snapshot / "features" / symbol.lower() / "close.day.bin").is_file()


def test_build_rejects_missing_required_benchmark_date(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
        missing=("SH000905", "close", "2026-07-24"),
    )

    with pytest.raises(module.DataContractError, match="SH000905.*close.*required"):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=provider,
            benchmark_source=benchmark_source_evidence(),
        )


def test_failed_rebuild_keeps_published_native_pointer_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=provider,
        benchmark_source=benchmark_source_evidence(),
    )
    pointer = root / "qlib" / "cn_data_ashare_v2" / "LATEST.json"
    before = pointer.read_bytes()
    raw = np.fromfile(
        provider / "features" / "sh000906" / "open.day.bin", dtype="<f4"
    )
    raw[-1] = np.nan
    raw.tofile(provider / "features" / "sh000906" / "open.day.bin")

    with pytest.raises(module.DataContractError, match="SH000906.*open.*required"):
        module.build_qlib_v2(
            quant_data_root=root,
            publish=True,
            benchmark_provider=provider,
            benchmark_source=benchmark_source_evidence(),
        )

    assert pointer.read_bytes() == before


def test_qlib_benchmark_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    snapshot = module.build_qlib_v2(
        quant_data_root=root,
        benchmark_provider=provider,
        benchmark_source=benchmark_source_evidence(),
    )
    feature = snapshot / "features" / "sh000300" / "close.day.bin"
    feature.write_bytes(feature.read_bytes() + b"tamper")

    with pytest.raises(contract.DataContractError, match="benchmark|Qlib file"):
        contract.validate_qlib_snapshot(snapshot, verify_hashes=True)
