import hashlib
import importlib.util
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BENCHMARK_SYMBOLS = (
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
)
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
    invalid_ohlc: tuple[str, str] | None = None,
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
            if invalid_ohlc == (symbol, calendar[-1]) and field == "low":
                values[-1] += 5.0
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


def benchmark_source_evidence(root: Path) -> dict[str, str]:
    run_root = root / "runtime/v2_compatibility/runs/v1-compat-test"
    relative = run_root.relative_to(root) / "candidate_summary.json"
    evidence_path = root / relative
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    provider = root / "qlib" / "cn_data_baostock_jqdata"
    provider_manifest = json.loads(
        (provider / "PROVIDER_MANIFEST.json").read_text(encoding="utf-8")
    )
    source_snapshot = provider_manifest["source_snapshot"]
    evidence_path.write_text(
        json.dumps(
            {
                "status": "candidate_ready",
                "run_id": "v1-compat-test",
                "source_snapshot": source_snapshot,
                "benchmark": {"status": "completed", "source": "tencent"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "publish_summary.json").write_text(
        json.dumps(
            {
                "status": "published",
                "run_id": "v1-compat-test",
                "published_at": "2026-07-25T23:30:00+08:00",
                "source_snapshot": source_snapshot,
                "targets": {"merged_qlib": str(provider)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "provider": "tencent",
        "interface": "qt.gtimg.cn historical daily",
        "source_dataset_id": "qlib_ashare_merged_compatibility",
        "source_path": "qlib/cn_data_baostock_jqdata",
        "source_run_id": "v1-compat-test",
        "evidence_path": relative.as_posix(),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
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
        benchmark_source=benchmark_source_evidence(root),
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
    assert provider["producer"] == {
        "script": "build_qlib_v2.py",
        "version": "10",
    }
    assert (published / "features" / "sh600000" / "close.day.bin").is_file()
    assert resolved.manifest["benchmark_complete"] is True
    assert provider["benchmark_complete"] is True
    assert resolved.catalog_entry["benchmark_complete"] is True
    assert set(resolved.manifest["benchmarks"]["symbols"]) == set(BENCHMARK_SYMBOLS)
    assert resolved.manifest["benchmarks"]["required_symbols"] == list(
        BENCHMARK_SYMBOLS
    )
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
            benchmark_source=benchmark_source_evidence(root),
        )


def test_explicit_build_rejects_invalid_benchmark_ohlc(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
        invalid_ohlc=("SH000300", "2026-07-24"),
    )

    with pytest.raises(
        module.DataContractError,
        match=r"SH000300.*OHLC.*2026-07-24",
    ):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=provider,
            benchmark_source=benchmark_source_evidence(root),
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


def test_default_build_reconciles_invalid_tencent_ohlc_with_eastmoney(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    stale_source = _qfq_source_contract(contract, root)
    stale_source["snapshot_id"] = "older-qfq-snapshot"
    write_benchmark_provider(root, source_snapshot=stale_source)

    tencent_frames = tencent_benchmark_frames()
    for frame in tencent_frames.values():
        frame.loc["2026-07-24", "low"] = frame.loc["2026-07-24", "high"] + 5.0
    eastmoney_frames = tencent_benchmark_frames()
    for symbol, frame in eastmoney_frames.items():
        eastmoney_frames[symbol] = frame.drop(index="2026-07-23")
    calls: list[str] = []

    def fake_tencent(*_args, **_kwargs):
        calls.append("tencent")
        return tencent_frames

    def fake_eastmoney(*_args, **_kwargs):
        calls.append("eastmoney")
        return eastmoney_frames

    monkeypatch.setattr(module, "download_tencent_benchmarks", fake_tencent)
    monkeypatch.setattr(
        module,
        "download_eastmoney_benchmarks",
        fake_eastmoney,
        raising=False,
    )

    snapshot = module.build_qlib_v2(quant_data_root=root)
    manifest = json.loads(
        (snapshot / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    source = manifest["benchmarks"]["source"]
    evidence = json.loads((root / source["evidence_path"]).read_text(encoding="utf-8"))

    assert calls == ["tencent", "eastmoney"]
    assert source["provider"] == "tencent+eastmoney"
    assert source["source_dataset_id"] == "tencent_eastmoney_reconciled_index_daily"
    assert evidence["source_attempts"] == [
        {
            "provider": "tencent",
            "status": "requires_correction",
            "reason": "SH000300 benchmark OHLC relationship is invalid on 2026-07-24",
        },
        {"provider": "eastmoney", "status": "correction_source"},
    ]
    assert evidence["corrections"] == [
        {
            "date": "2026-07-24",
            "fields": ["open", "high", "low", "close", "volume"],
            "source": "eastmoney",
            "symbol": symbol,
        }
        for symbol in BENCHMARK_SYMBOLS
    ]
    for symbol in BENCHMARK_SYMBOLS:
        inputs = evidence["symbols"][symbol]["input_artifacts"]
        assert set(inputs) == {"tencent", "eastmoney"}
        for input_evidence in inputs.values():
            input_path = root / input_evidence["raw_path"]
            assert input_path.is_file()
            assert input_evidence["raw_sha256"] == module.sha256_file(input_path)
        raw = np.fromfile(
            snapshot / "features" / symbol.lower() / "low.day.bin", dtype="<f4"
        )
        np.testing.assert_allclose(
            raw[1:],
            eastmoney_frames[symbol]["low"]
            .reindex(["2026-07-23", "2026-07-24"])
            .fillna(tencent_frames[symbol]["low"])
            .to_numpy(dtype=float),
        )


def test_default_rebuild_reuses_current_native_benchmarks_without_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    compatibility_source = benchmark_source_evidence(root)
    first = module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=compatibility,
        benchmark_source=compatibility_source,
    )
    shutil.rmtree(compatibility)

    def reject_download(*_args, **_kwargs):
        pytest.fail("current native benchmarks should be reused without a download")

    monkeypatch.setattr(module, "download_tencent_benchmarks", reject_download)
    second = module.build_qlib_v2(quant_data_root=root)
    manifest = json.loads(
        (second / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert second != first
    source = manifest["benchmarks"]["source"]
    assert source["source_dataset_id"] == "qlib_ashare_v2"
    assert source["source_path"] == first.relative_to(root).as_posix()
    assert source["origin_provider"] == "tencent"
    assert source["origin_interface"] == "qt.gtimg.cn historical daily"
    assert source["origin_source_dataset_id"] == (
        "qlib_ashare_merged_compatibility"
    )
    assert source["origin_evidence_sha256"] == compatibility_source["evidence_sha256"]


def test_explicit_native_benchmark_provider_discovers_its_own_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    first = module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=compatibility,
        benchmark_source=benchmark_source_evidence(root),
    )
    shutil.rmtree(compatibility)

    second = module.build_qlib_v2(
        quant_data_root=root,
        benchmark_provider=first,
    )
    manifest = json.loads(
        (second / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )

    source = manifest["benchmarks"]["source"]
    assert source["source_dataset_id"] == "qlib_ashare_v2"
    assert source["source_path"] == first.relative_to(root).as_posix()
    assert source["source_run_id"] == first.name
    assert source["origin_provider"] == "tencent"


def test_explicit_native_benchmark_provider_rejects_tampered_manifests(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    compatibility_source = benchmark_source_evidence(root)
    published = module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=compatibility,
        benchmark_source=compatibility_source,
    )
    shutil.rmtree(compatibility)
    supplied_source = dict(compatibility_source)
    supplied_source.update(
        {
            "source_dataset_id": "forged_native_source",
            "source_path": published.relative_to(root).as_posix(),
            "source_run_id": published.name,
        }
    )
    for filename in ("DATASET_MANIFEST.json", "PROVIDER_MANIFEST.json"):
        path = published / filename
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["benchmarks"]["source"]["origin_provider"] = "forged"
        path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(module.DataContractError, match="size mismatch|hash mismatch"):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=published,
            benchmark_source=supplied_source,
        )


def test_explicit_native_benchmark_provider_must_be_published(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    staging = module.build_qlib_v2(
        quant_data_root=root,
        benchmark_provider=compatibility,
        benchmark_source=benchmark_source_evidence(root),
    )
    supplied_source = benchmark_source_evidence(root)
    supplied_source.update(
        {
            "source_dataset_id": "forged_staging_source",
            "source_path": staging.relative_to(root).as_posix(),
            "source_run_id": staging.name,
        }
    )

    with pytest.raises(
        module.DataContractError,
        match="published canonical snapshot",
    ):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=staging,
            benchmark_source=supplied_source,
        )


def test_native_provider_without_dataset_manifest_cannot_pose_as_compatibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    staging = module.build_qlib_v2(
        quant_data_root=root,
        benchmark_provider=compatibility,
        benchmark_source=benchmark_source_evidence(root),
    )
    (staging / "DATASET_MANIFEST.json").unlink()
    forged_evidence_path = root / "runtime/forged-native-evidence.json"
    forged_evidence_path.parent.mkdir(parents=True, exist_ok=True)
    forged_evidence_path.write_text('{"status":"forged"}', encoding="utf-8")
    forged_source = {
        "provider": "forged",
        "interface": "forged compatibility benchmark",
        "source_dataset_id": "qlib_ashare_merged_compatibility",
        "source_path": staging.relative_to(root).as_posix(),
        "source_run_id": "forged-run",
        "evidence_path": forged_evidence_path.relative_to(root).as_posix(),
        "evidence_sha256": hashlib.sha256(
            forged_evidence_path.read_bytes()
        ).hexdigest(),
    }

    with pytest.raises(
        module.DataContractError,
        match="no published benchmark provenance",
    ):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=staging,
            benchmark_source=forged_source,
        )


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
            benchmark_source=benchmark_source_evidence(root),
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
        benchmark_source=benchmark_source_evidence(root),
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
            benchmark_source=benchmark_source_evidence(root),
        )

    assert pointer.read_bytes() == before


@pytest.mark.parametrize("failure_stage", ["conversion", "benchmark", "publication"])
def test_failed_build_removes_only_its_staging_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    unrelated = root / "staging" / "keep-this-run" / "artifact.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"keep")

    if failure_stage == "conversion":
        def fail_conversion(_source, target, **_kwargs):
            Path(target).mkdir(parents=True, exist_ok=True)
            (Path(target) / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("injected conversion failure")

        monkeypatch.setattr(module, "convert_to_bin", fail_conversion)
    elif failure_stage == "benchmark":
        monkeypatch.setattr(
            module,
            "_import_benchmarks",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected benchmark failure")
            ),
        )
    else:
        monkeypatch.setattr(
            module,
            "publish_qlib_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected publication failure")
            ),
        )

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        module.build_qlib_v2(
            quant_data_root=root,
            publish=True,
            benchmark_provider=provider,
            benchmark_source=benchmark_source_evidence(root),
        )

    assert unrelated.read_bytes() == b"keep"
    assert list((root / "staging").glob("qlib-v2-*")) == []


@pytest.mark.parametrize("failure", ["missing", "hash_mismatch"])
def test_build_rejects_unverifiable_benchmark_source_evidence(
    tmp_path: Path,
    failure: str,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    provider = write_benchmark_provider(
        root,
        source_snapshot=_qfq_source_contract(contract, root),
    )
    source = benchmark_source_evidence(root)
    evidence_path = root / source["evidence_path"]
    if failure == "missing":
        evidence_path.unlink()
    else:
        evidence_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(
        module.DataContractError,
        match=(
            "evidence is unavailable|evidence hash mismatch|"
            "no published benchmark provenance"
        ),
    ):
        module.build_qlib_v2(
            quant_data_root=root,
            benchmark_provider=provider,
            benchmark_source=source,
        )


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
        benchmark_source=benchmark_source_evidence(root),
    )
    feature = snapshot / "features" / "sh000300" / "close.day.bin"
    feature.write_bytes(feature.read_bytes() + b"tamper")

    with pytest.raises(contract.DataContractError, match="benchmark|Qlib file"):
        contract.validate_qlib_snapshot(snapshot, verify_hashes=True)


def test_cleaned_v2_builds_without_compatibility_provider_and_records_cleaning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    canonical_module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    source_contract = _qfq_source_contract(contract, root)
    compatibility_provider = write_benchmark_provider(
        root,
        source_snapshot=source_contract,
    )
    canonical_snapshot = canonical_module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=compatibility_provider,
        benchmark_source=benchmark_source_evidence(root),
    )
    canonical_manifest = contract.validate_dataset_snapshot(canonical_snapshot)
    canonical_identity = {
        field: str(canonical_manifest[field])
        for field in ("dataset_id", "snapshot_id", "content_sha256", "price_basis")
    }
    shutil.rmtree(compatibility_provider)

    cleaned_script = SCRIPTS / "build_cleaned_qlib_v2.py"
    assert cleaned_script.is_file(), "native cleaned V2 builder is missing"
    cleaned_module = load_script("build_cleaned_qlib_v2")
    published = cleaned_module.build_cleaned_qlib_v2(
        quant_data_root=root,
        publish=True,
    )
    resolved = contract.resolve_dataset(
        "qlib_ashare_cleaned_v2",
        quant_data_root=root,
        require_basis="qfq",
    )

    assert published == resolved.snapshot_path
    assert resolved.manifest["benchmark_complete"] is True
    assert resolved.manifest["upstream_snapshots"] == [source_contract]
    assert resolved.manifest["dependency_snapshots"] == [canonical_identity]
    transformation = resolved.manifest["transformation"]
    assert transformation["name"] == "anomaly_masking"
    assert transformation["algorithm"]["version"] == "2"
    assert transformation["source_snapshot"] == source_contract
    assert transformation["summary"]["source_symbols"] == 1
    assert transformation["summary"]["output_symbols"] == 1
    assert set(transformation["artifacts"]) == {
        "anomaly_stocks",
        "anomaly_events",
        "cleaning_strategy",
        "cleaning_summary",
    }
    provider_manifest = json.loads(
        (published / "PROVIDER_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert resolved.manifest["producer"]["version"] == "6"
    assert provider_manifest["producer"] == {
        "script": "build_cleaned_qlib_v2.py",
        "version": "6",
    }
    assert provider_manifest["transformation"] == transformation
    assert provider_manifest["dependency_snapshots"] == [canonical_identity]
    assert (
        resolved.manifest["benchmarks"]["source"]["source_dataset_id"]
        == "qlib_ashare_v2"
    )
    benchmark_source = resolved.manifest["benchmarks"]["source"]
    assert benchmark_source["source_dataset_id"] == "qlib_ashare_v2"
    assert benchmark_source["source_path"].startswith(
        "qlib/cn_data_ashare_v2/snapshots/"
    )
    assert benchmark_source["origin_source_path"] == (
        "qlib/cn_data_baostock_jqdata"
    )

    artifact = published / transformation["artifacts"]["cleaning_summary"]["relative_path"]
    artifact.write_text("tampered", encoding="utf-8")
    with pytest.raises(contract.DataContractError, match="transformation artifact"):
        contract.validate_qlib_snapshot(published, verify_hashes=True)


@pytest.mark.parametrize(
    ("changed_dataset", "expected_error"),
    [
        ("ashare_daily_qfq", "qfq snapshot changed during cleaned build"),
        ("qlib_ashare_v2", "canonical Qlib changed during cleaned build"),
    ],
)
def test_cleaned_publish_rejects_input_change_inside_publication_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_dataset: str,
    expected_error: str,
) -> None:
    root = tmp_path / "quant_data"
    publish_qfq_fixture(root)
    canonical_module = load_script("build_qlib_v2")
    contract = load_script("quant_data_contract")
    source_contract = _qfq_source_contract(contract, root)
    compatibility = write_benchmark_provider(
        root,
        source_snapshot=source_contract,
    )
    canonical_module.build_qlib_v2(
        quant_data_root=root,
        publish=True,
        benchmark_provider=compatibility,
        benchmark_source=benchmark_source_evidence(root),
    )
    shutil.rmtree(compatibility)

    cleaned_module = load_script("build_cleaned_qlib_v2")
    real_resolve = cleaned_module.resolve_dataset
    resolutions: dict[str, int] = {}
    lock_state = {"held": False}

    @contextmanager
    def tracked_publication_lock(_root: Path):
        assert lock_state["held"] is False
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    def resolve_with_concurrent_input(dataset_id: str, **kwargs):
        resolved = real_resolve(dataset_id, **kwargs)
        resolutions[dataset_id] = resolutions.get(dataset_id, 0) + 1
        if resolutions[dataset_id] == 1:
            return resolved
        assert lock_state["held"] is True
        if dataset_id != changed_dataset:
            return resolved
        changed_manifest = dict(resolved.manifest)
        changed_manifest["snapshot_id"] = f"{dataset_id}-concurrently-published"
        changed_manifest["content_sha256"] = "f" * 64
        return SimpleNamespace(
            dataset_id=resolved.dataset_id,
            manifest=changed_manifest,
            snapshot_path=resolved.snapshot_path,
        )

    monkeypatch.setattr(
        cleaned_module,
        "resolve_dataset",
        resolve_with_concurrent_input,
    )
    monkeypatch.setattr(
        cleaned_module,
        "dataset_publication_lock",
        tracked_publication_lock,
    )

    with pytest.raises(
        contract.DataContractError,
        match=expected_error,
    ):
        cleaned_module.build_cleaned_qlib_v2(
            quant_data_root=root,
            publish=True,
        )
    assert resolutions[changed_dataset] >= 2
    assert lock_state["held"] is False
    assert not (root / "qlib" / "cn_data_ashare_cleaned_v2" / "LATEST.json").exists()
