from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from quant_workflow import quant_data_qlib_provider as direct
from quant_workflow.scripts import quant_data_contract as contract


def _write_symbol(
    snapshot: Path,
    symbol: str,
    dates: list[str],
    closes: list[float],
    volumes: list[float] | None = None,
) -> None:
    close = np.asarray(closes, dtype=float)
    volume = (
        np.asarray(volumes, dtype=float)
        if volumes is not None
        else np.arange(1, len(dates) + 1, dtype=float) * 100.0
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "preclose": close - 0.2,
            "volume": volume,
            "amount": close * volume,
            "turn": np.arange(1, len(dates) + 1, dtype=float),
            "pctChg": np.arange(1, len(dates) + 1, dtype=float) / 10.0,
            "tradestatus": 1,
            "isST": 0,
            "symbol": symbol,
            "source_id": "fixture.raw_x_factor",
            "factor": np.linspace(0.8, 1.0, len(dates)),
        }
    )
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "factor": np.linspace(0.8, 1.0, len(dates)),
        }
    )
    (snapshot / "bars").mkdir(parents=True, exist_ok=True)
    (snapshot / "factors").mkdir(parents=True, exist_ok=True)
    bars.to_parquet(snapshot / "bars" / f"{symbol}.parquet", index=False)
    factors.to_parquet(snapshot / "factors" / f"{symbol}.parquet", index=False)


def _stage_qfq_snapshot(
    quant_data_root: Path,
    snapshot_id: str,
    *,
    sh_close: float,
    sh_volumes: list[float] | None = None,
) -> Path:
    snapshot = quant_data_root / "staging" / snapshot_id / snapshot_id
    _write_symbol(
        snapshot,
        "SH600000",
        ["2026-01-02", "2026-01-06"],
        [sh_close, sh_close + 1.0],
        sh_volumes,
    )
    _write_symbol(
        snapshot,
        "SZ000001",
        ["2026-01-02", "2026-01-05", "2026-01-06"],
        [20.0, 21.0, 22.0],
    )
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "ex_date": pd.to_datetime(["2025-12-31"]),
            "action_type": ["dividend"],
            "ratio": [1.0],
            "source": ["fixture"],
        }
    ).to_parquet(snapshot / "corporate_actions.parquet", index=False)
    contract.finalize_price_snapshot(
        snapshot,
        dataset_id="ashare_daily_qfq",
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={"provider": "fixture", "interface": "raw_plus_factor"},
        price_basis="qfq",
        adjustment={
            "method": "provider_native_paired",
            "adjusted_at": "2026-01-07T08:00:00+08:00",
            "adjustment_as_of": "2026-01-06",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_valid_trade_date_factor_equals_1",
            "algorithm_version": "fixture-v1",
            "paired_raw_snapshot_id": f"raw-{snapshot_id}",
        },
        producer={"script": "test_quant_data_qlib_provider.py", "version": "1"},
        source_retrieved_at="2026-01-07T07:59:00+08:00",
    )
    return snapshot


def _publish_qfq(
    quant_data_root: Path,
    snapshot_id: str,
    *,
    sh_close: float,
    update_catalog: bool,
    sh_volumes: list[float] | None = None,
) -> Path:
    dataset_root = (
        quant_data_root / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    )
    staged = _stage_qfq_snapshot(
        quant_data_root,
        snapshot_id,
        sh_close=sh_close,
        sh_volumes=sh_volumes,
    )
    if update_catalog:
        return contract.publish_snapshot_group(
            [(staged, dataset_root)],
            quant_data_root=quant_data_root,
            update_root_catalog=True,
        )[0]
    published = contract.publish_snapshot(
        staged,
        dataset_root=dataset_root,
        quant_data_root=quant_data_root,
    )
    contract.write_root_manifest(
        quant_data_root,
        {
            "ashare_daily_qfq": contract.catalog_entry_from_snapshot(
                published,
                quant_data_root=quant_data_root,
                dataset_root=dataset_root,
            )
        },
    )
    return published


def _stage_index_snapshot(
    quant_data_root: Path,
    snapshot_id: str,
) -> Path:
    snapshot = quant_data_root / "staging" / snapshot_id / snapshot_id
    bars_dir = snapshot / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    for symbol, closes in {
        "SH000300": [100.0, 101.0, 102.0],
        "SH000905": [200.0, 201.0, 202.0],
        "SH000906": [300.0, 301.0, 302.0],
        "SZ399006": [400.0, 401.0, 402.0],
        "SH000688": [500.0, 501.0, 502.0],
    }.items():
        close = np.asarray(closes, dtype=float)
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": [1000.0, 1100.0, 1200.0],
                "source_id": "fixture.index",
            }
        )
        frame.to_parquet(bars_dir / f"{symbol}.parquet", index=False)
    contract.finalize_price_snapshot(
        snapshot,
        dataset_id="cn_index_daily",
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={"provider": "fixture", "interface": "index_parquet"},
        price_basis="derived",
        adjustment={"method": "none"},
        producer={"script": "test_quant_data_qlib_provider.py", "version": "1"},
        source_retrieved_at="2026-01-07T07:59:00+08:00",
    )
    return snapshot


def _publish_index(
    quant_data_root: Path,
    snapshot_id: str = "index-001",
) -> Path:
    staged = _stage_index_snapshot(quant_data_root, snapshot_id)
    dataset_root = quant_data_root / "parquet" / "index_1d" / "cn_index_daily"
    return contract.publish_snapshot_group(
        [(staged, dataset_root)],
        quant_data_root=quant_data_root,
        update_root_catalog=True,
    )[0]


def _membership_frame(*, swapped: bool = False) -> pd.DataFrame:
    index_codes = (
        ["000905.XSHG", "000300.XSHG"]
        if swapped
        else ["000300.XSHG", "000905.XSHG"]
    )
    return pd.DataFrame(
        {
            "symbol": ["SH600000", "SZ000001"],
            "index_code": index_codes,
            "start_date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "end_date": pd.to_datetime(["2026-01-06", "2026-01-06"]),
        }
    )


def _publish_membership(
    quant_data_root: Path,
    *,
    swapped: bool = False,
) -> tuple[Path, str]:
    snapshot_id = "pit-csi800-002" if swapped else "pit-csi800-001"
    pit_dir = quant_data_root / "staging" / snapshot_id / snapshot_id
    membership = _membership_frame(swapped=swapped)
    intervals = pit_dir / "membership_intervals.parquet"
    daily = pit_dir / "daily" / "2026.parquet"
    daily.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(intervals, index=False)
    membership.assign(date=pd.Timestamp("2026-01-02")).to_parquet(
        daily,
        index=False,
    )
    source = {
        "requested": "baostock",
        "selected": "baostock",
        "fallback_reason": None,
        "indices": {
            "csi300": "000300.XSHG",
            "csi500": "000905.XSHG",
        },
        "date_range": {
            "start_date": "2026-01-02",
            "end_date": "2026-01-06",
        },
    }
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(source),
        encoding="utf-8",
    )
    contract.finalize_artifact_snapshot(
        pit_dir,
        dataset_id="pit_csi800",
        snapshot_id=snapshot_id,
        data_kind="point_in_time_membership",
        role="canonical",
        status="ready",
        source={"provider": "fixture", "interface": "membership_builder"},
        price_basis="not_applicable",
        producer={"script": "test_quant_data_qlib_provider.py", "version": "1"},
        source_retrieved_at="2026-01-07T08:00:00+08:00",
        date_range={"start": "2026-01-02", "end": "2026-01-06"},
        components={
            "daily": {"kind": "partitioned_parquet", "path": "daily"},
            "membership_intervals": {
                "kind": "parquet",
                "path": "membership_intervals.parquet",
            },
            "source_manifest": {"kind": "json", "path": "MEMBERSHIP_SOURCE.json"},
        },
    )
    published = contract.publish_snapshot_group(
        [(pit_dir, quant_data_root / "reference" / "pit_csi800")],
        quant_data_root=quant_data_root,
        update_root_catalog=True,
    )[0]
    manifest = contract.validate_dataset_snapshot(published)
    return published / intervals.relative_to(pit_dir), str(manifest["content_sha256"])


def _publish_cleaning_plan(
    quant_data_root: Path,
    *,
    volume_mask_symbols: list[str] | None = None,
    event_dates: dict[str, list[str]] | None = None,
) -> Path:
    snapshot_id = "qlib-cleaning-plan-001"
    staged = quant_data_root / "staging" / snapshot_id / snapshot_id
    staged.mkdir(parents=True)
    plan_path = staged / "CLEANING_PLAN.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "fixture.cleaning",
                "volume_mask_symbols": volume_mask_symbols or [],
                "excluded_symbols": [],
                "event_dates": (
                    {"SH600000": ["2026-01-06"]}
                    if event_dates is None
                    else event_dates
                ),
                "event_mask_fields": [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "turn",
                    "pctChg",
                ],
            }
        ),
        encoding="utf-8",
    )
    upstream = contract.resolve_dataset(
        "ashare_daily_qfq", quant_data_root=quant_data_root, require_basis="qfq"
    )
    contract.finalize_artifact_snapshot(
        staged,
        dataset_id="qlib_cleaning_plan",
        snapshot_id=snapshot_id,
        data_kind="qlib_cleaning_plan",
        role="canonical",
        status="ready",
        source={"provider": "fixture", "interface": "cleaning_plan_builder"},
        price_basis="not_applicable",
        producer={"script": "test_quant_data_qlib_provider.py", "version": "1"},
        source_retrieved_at="2026-01-07T08:00:00+08:00",
        date_range={"start": "2026-01-02", "end": "2026-01-06"},
        components={"plan": {"kind": "json", "path": "CLEANING_PLAN.json"}},
        upstream_snapshots=[
            {
                field: upstream.manifest[field]
                for field in (
                    "dataset_id",
                    "snapshot_id",
                    "content_sha256",
                    "price_basis",
                )
            }
        ],
    )
    published = contract.publish_snapshot_group(
        [(staged, quant_data_root / "reference" / "qlib_cleaning_plan")],
        quant_data_root=quant_data_root,
        update_root_catalog=True,
    )[0]
    return published / "CLEANING_PLAN.json"


@pytest.fixture()
def direct_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    quant_data_root = tmp_path / "quant_data"
    snapshot = _publish_qfq(
        quant_data_root,
        "qfq-001",
        sh_close=10.0,
        update_catalog=False,
    )
    _publish_index(quant_data_root)
    membership, _ = _publish_membership(quant_data_root)
    descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
        index_dataset_id="cn_index_daily",
    )
    return quant_data_root, snapshot, descriptor


def test_descriptor_contains_metadata_and_calendar_but_no_market_copy(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    _, _, descriptor = direct_fixture
    provider_dir = descriptor.parent
    files = sorted(
        path.relative_to(provider_dir).as_posix()
        for path in provider_dir.rglob("*")
        if path.is_file()
    )

    assert files == [
        "CALENDAR.parquet",
        "MEMBERSHIP_INTERVALS.parquet",
        "PROVIDER_DESCRIPTOR.json",
    ]
    assert not list(provider_dir.rglob("*.bin"))
    assert not (provider_dir / "features").exists()
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    assert payload["storage_mode"] == "direct_parquet"
    assert payload["dataset"]["dataset_id"] == "ashare_daily_qfq"
    assert payload["dataset"]["snapshot_id"] == "qfq-001"
    assert payload["membership"]["dataset_id"] == "pit_csi800"
    assert payload["membership"]["snapshot_id"] == "pit-csi800-001"
    assert payload["membership"]["content_sha256"]
    assert payload["membership"]["manifest_sha256"]


def test_discovers_governed_root_without_requiring_quant_data_directory_name(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    renamed_root = quant_data_root.with_name("governed-market-lake")
    quant_data_root.rename(renamed_root)
    renamed_descriptor = renamed_root / descriptor.relative_to(quant_data_root)

    assert direct.discover_quant_data_root(renamed_descriptor) == renamed_root.resolve()


def test_direct_qlib_config_is_yaml_serializable_and_names_custom_provider(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture

    config = direct.qlib_direct_init_config(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
        kernels=1,
    )

    yaml.safe_dump(config, sort_keys=False)
    assert config["provider"] == "LocalProvider"
    assert config["provider_uri"] == str(descriptor.parent)


def test_build_rejects_membership_missing_from_governed_catalog(
    tmp_path: Path,
) -> None:
    quant_data_root = tmp_path / "quant_data"
    _publish_qfq(
        quant_data_root,
        "qfq-001",
        sh_close=10.0,
        update_catalog=False,
    )
    membership = quant_data_root / "reference" / "pit_csi800" / "membership_intervals.parquet"
    membership.parent.mkdir(parents=True, exist_ok=True)
    _membership_frame().to_parquet(membership, index=False)

    with pytest.raises(direct.DirectProviderError, match="formal|registered"):
        direct.build_direct_provider_descriptor(
            quant_data_root=quant_data_root,
            membership_path=membership,
        )


def test_build_ignores_legacy_writer_marker_outside_immutable_snapshot(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, _ = direct_fixture
    marker = (
        quant_data_root
        / "reference"
        / "pit_csi800"
        / ".build_pit_csi800.in-progress.json"
    )
    marker.write_text("{}", encoding="utf-8")

    assert direct.build_direct_provider_descriptor(quant_data_root=quant_data_root)


def test_existing_descriptor_does_not_revalidate_every_bar_file(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_data_root, _, descriptor = direct_fixture

    def unexpected_full_validation(*args, **kwargs):
        raise AssertionError("existing descriptor must use the fast identity check")

    monkeypatch.setattr(direct, "resolve_dataset", unexpected_full_validation)
    reused = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
    )

    assert reused == descriptor


def test_calendar_fields_and_missing_dates_are_mapped_without_forward_fill(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    calendar = direct.QuantDataCalendarProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )
    feature = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    assert list(calendar.load_calendar("day", False)) == list(
        pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    )
    close = feature.feature("SH600000", "$close", 0, 2, "day")
    volume = feature.feature("SH600000", "$volume", 0, 2, "day")
    amount = feature.feature("SH600000", "$amount", 0, 2, "day")
    vwap = feature.feature("SH600000", "$vwap", 0, 2, "day")
    factor = feature.feature("SH600000", "$factor", 0, 2, "day")

    assert close.iloc[0] == pytest.approx(10.0)
    assert np.isnan(close.iloc[1])
    assert close.iloc[2] == pytest.approx(11.0)
    assert np.isnan(volume.iloc[1])
    assert amount.iloc[0] == pytest.approx(close.iloc[0] * volume.iloc[0])
    assert vwap.iloc[0] == pytest.approx(close.iloc[0])
    assert factor.tolist()[0] == pytest.approx(1.0)
    assert np.isnan(factor.iloc[1])
    assert factor.tolist()[2] == pytest.approx(1.0)


def test_direct_provider_reads_formal_index_dataset(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    feature = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    close = feature.feature("SH000300", "$close", 0, 2, "day")
    volume = feature.feature("SH000300", "$volume", 0, 2, "day")

    assert close.tolist() == pytest.approx([100.0, 101.0, 102.0])
    assert volume.tolist() == pytest.approx([1000.0, 1100.0, 1200.0])
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    assert payload["index_dataset"]["dataset_id"] == "cn_index_daily"
    assert payload["index_dataset"]["required_symbols"] == [
        "SH000300",
        "SH000905",
        "SH000906",
        "SZ399006",
        "SH000688",
    ]


def test_cleaned_direct_provider_applies_frozen_cleaning_plan(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, _ = direct_fixture
    _publish_cleaning_plan(quant_data_root)
    descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
        index_dataset_id="cn_index_daily",
        cleaning_mode="cleaned",
    )
    feature = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    close = feature.feature("SH600000", "$close", 0, 2, "day")
    volume = feature.feature("SH600000", "$volume", 0, 2, "day")

    assert close.iloc[0] == pytest.approx(10.0)
    assert np.isnan(close.iloc[2])
    assert np.isnan(volume.iloc[2])
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    assert payload["cleaning"]["mode"] == "cleaned"
    assert payload["cleaning"]["plan_sha256"]
    assert payload["cleaning"]["dataset_id"] == "qlib_cleaning_plan"
    assert payload["cleaning"]["snapshot_id"] == "qlib-cleaning-plan-001"
    assert payload["cleaning"]["content_sha256"]
    assert payload["cleaning"]["manifest_sha256"]


def test_cleaned_direct_provider_excludes_rows_with_masked_required_fields(
    tmp_path: Path,
) -> None:
    quant_data_root = tmp_path / "quant_data"
    _publish_qfq(
        quant_data_root,
        "qfq-zero-volume",
        sh_close=10.0,
        sh_volumes=[0.0, 200.0],
        update_catalog=False,
    )
    _publish_index(quant_data_root)
    _publish_membership(quant_data_root)
    _publish_cleaning_plan(
        quant_data_root,
        volume_mask_symbols=["SH600000"],
        event_dates={},
    )
    descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
        index_dataset_id="cn_index_daily",
        cleaning_mode="cleaned",
    )
    feature = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    close = feature.feature("SH600000", "$close", 0, 2, "day")
    volume = feature.feature("SH600000", "$volume", 0, 2, "day")
    factor = feature.feature("SH600000", "$factor", 0, 2, "day")

    assert np.isnan(close.iloc[0])
    assert np.isnan(volume.iloc[0])
    assert np.isnan(factor.iloc[0])
    assert close.iloc[2] == pytest.approx(11.0)
    assert volume.iloc[2] == pytest.approx(200.0)
    assert factor.iloc[2] == pytest.approx(1.0)


def test_all_and_csi_pools_keep_historical_membership_intervals(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    provider = direct.QuantDataInstrumentProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    all_pool = provider.list_instruments(
        provider.instruments("all"),
        start_time="2026-01-02",
        end_time="2026-01-06",
    )
    csi300 = provider.list_instruments(
        provider.instruments("csi300"),
        start_time="2026-01-02",
        end_time="2026-01-06",
        as_list=True,
    )
    csi500 = provider.list_instruments(
        provider.instruments("csi500"),
        start_time="2026-01-02",
        end_time="2026-01-06",
        as_list=True,
    )
    csi800 = provider.list_instruments(
        provider.instruments("csi800"),
        start_time="2026-01-02",
        end_time="2026-01-06",
        as_list=True,
    )

    assert set(all_pool) == {"SH600000", "SZ000001"}
    assert csi300 == ["SH600000"]
    assert csi500 == ["SZ000001"]
    assert csi800 == ["SH600000", "SZ000001"]
    with pytest.raises(direct.DirectProviderError, match="coverage ends"):
        provider.list_instruments(
            provider.instruments("csi800"),
            start_time="2026-01-02",
            end_time="2026-01-07",
        )


def test_qlib_market_lookup_without_dates_does_not_use_index_calendar_start(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    provider = direct.QuantDataInstrumentProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    # Qlib first asks for a market definition without dates.  Its calendar may
    # begin before the historical membership source, so this lookup must keep
    # the frozen intervals and defer coverage checks to explicit date ranges.
    object.__setattr__(
        provider._context,
        "calendar",
        pd.DatetimeIndex(["2025-01-01", "2026-01-06"]),
    )
    assert provider.list_instruments(
        provider.instruments("csi800"),
        as_list=True,
    ) == ["SH600000", "SZ000001"]


def test_provider_locks_snapshot_even_after_latest_pointer_advances(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, old_snapshot, descriptor = direct_fixture
    _publish_qfq(
        quant_data_root,
        "qfq-002",
        sh_close=99.0,
        update_catalog=True,
    )

    locked = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )
    assert locked.feature("SH600000", "$close", 0, 0, "day").iloc[0] == pytest.approx(10.0)
    assert locked.snapshot_path == old_snapshot

    new_descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
    )
    current = direct.QuantDataFeatureProvider(
        descriptor_path=new_descriptor,
        quant_data_root=quant_data_root,
    )
    assert current.feature("SH600000", "$close", 0, 0, "day").iloc[0] == pytest.approx(99.0)


def test_descriptor_rejects_index_pointer_race(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_data_root, _, _ = direct_fixture
    old_entry, old_snapshot = direct._current_dataset_identity(
        quant_data_root,
        "cn_index_daily",
        expected_price_basis="derived",
    )
    _publish_index(quant_data_root, snapshot_id="index-002")
    original_identity = direct._current_dataset_identity

    def stale_index_identity(root, dataset_id, *, expected_price_basis):
        if dataset_id == "cn_index_daily":
            return old_entry, old_snapshot
        return original_identity(
            root,
            dataset_id,
            expected_price_basis=expected_price_basis,
        )

    monkeypatch.setattr(direct, "_current_dataset_identity", stale_index_identity)

    with pytest.raises(
        direct.DirectProviderError,
        match="formal index dataset changed during descriptor creation",
    ):
        direct.build_direct_provider_descriptor(
            quant_data_root=quant_data_root,
            output_root="runtime/race-test",
        )


def test_paths_helper_refreshes_after_latest_snapshot_advances(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_data_root, _, first_descriptor = direct_fixture
    from quant_workflow.scripts import _paths

    monkeypatch.setenv("QUANT_DATA_ROOT", str(quant_data_root))
    clear_cache = getattr(_paths._resolve_direct_qlib_descriptor, "cache_clear", None)
    if clear_cache is not None:
        clear_cache()
    assert _paths.qlib_direct_provider_descriptor() == first_descriptor

    _publish_qfq(
        quant_data_root,
        "qfq-002",
        sh_close=99.0,
        update_catalog=True,
    )

    second_descriptor = _paths.qlib_direct_provider_descriptor()
    assert second_descriptor != first_descriptor
    assert "qfq-002" in second_descriptor.parts


def test_workflow_provider_entrypoint_uses_direct_descriptor(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import _paths

    monkeypatch.setenv("QUANT_DATA_ROOT", str(quant_data_root))
    clear_cache = getattr(_paths._resolve_direct_qlib_descriptor, "cache_clear", None)
    if clear_cache is not None:
        clear_cache()

    assert _paths.qlib_workflow_provider_descriptor() == descriptor
    assert _paths.qlib_workflow_provider_uri() == descriptor.parent

    config = _paths.qlib_workflow_init_config(kernels=1)
    assert config["provider_uri"] == str(descriptor.parent)
    assert config["provider"] == "LocalProvider"


def test_membership_versions_get_distinct_frozen_descriptors(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, first_descriptor = direct_fixture
    old_provider = direct.QuantDataInstrumentProvider(
        descriptor_path=first_descriptor,
        quant_data_root=quant_data_root,
    )
    _publish_membership(quant_data_root, swapped=True)

    second_descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
    )
    new_provider = direct.QuantDataInstrumentProvider(
        descriptor_path=second_descriptor,
        quant_data_root=quant_data_root,
    )

    assert second_descriptor != first_descriptor
    assert old_provider.list_instruments(
        old_provider.instruments("csi300"),
        start_time="2026-01-02",
        end_time="2026-01-06",
        as_list=True,
    ) == ["SH600000"]
    assert new_provider.list_instruments(
        new_provider.instruments("csi300"),
        start_time="2026-01-02",
        end_time="2026-01-06",
        as_list=True,
    ) == ["SZ000001"]


def test_descriptor_rejects_frozen_membership_manifest_tamper(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    manifest_path = (
        quant_data_root
        / payload["membership"]["snapshot_path"]
        / "DATASET_MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latest_trade_date"] = "2026-01-05"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(direct.DirectProviderError, match="manifest.*hash mismatch"):
        direct.QuantDataInstrumentProvider(
            descriptor_path=descriptor,
            quant_data_root=quant_data_root,
        )


def test_descriptor_rejects_frozen_membership_component_tamper(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    source_path = quant_data_root / payload["membership"]["source_path"]
    _membership_frame(swapped=True).to_parquet(source_path, index=False)

    with pytest.raises(direct.DirectProviderError, match="source artifact hash mismatch"):
        direct.QuantDataInstrumentProvider(
            descriptor_path=descriptor,
            quant_data_root=quant_data_root,
        )


def test_descriptor_path_rejects_symlinked_directory(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    alias = quant_data_root / "runtime" / "qlib_direct" / "provider-alias"
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(descriptor.parent, target_is_directory=True)

    with pytest.raises(direct.DirectProviderError, match="symlink"):
        direct.QuantDataFeatureProvider(
            descriptor_path=alias,
            quant_data_root=quant_data_root,
        )


def test_feature_access_rejects_symlinked_bar_file(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, snapshot, descriptor = direct_fixture
    bar_path = snapshot / "bars" / "SH600000.parquet"
    relocated = snapshot / "bars" / "SH600000.real.parquet"
    os.replace(bar_path, relocated)
    bar_path.symlink_to(relocated)
    provider = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )

    with pytest.raises(direct.DirectProviderError, match="symlink"):
        provider.feature("SH600000", "$close", 0, 0, "day")


def test_access_rejects_a_bar_file_changed_after_descriptor_creation(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, snapshot, descriptor = direct_fixture
    bar_path = snapshot / "bars" / "SH600000.parquet"
    changed = pd.read_parquet(bar_path)
    changed.loc[0, "close"] = 999.0
    changed.to_parquet(bar_path, index=False)

    provider = direct.QuantDataFeatureProvider(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
    )
    with pytest.raises(direct.DirectProviderError, match="hash mismatch"):
        provider.feature("SH600000", "$close", 0, 0, "day")


def test_real_qlib_features_use_direct_provider_without_disk_cache(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    direct.init_qlib_from_quant_data(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
        kernels=1,
    )
    from qlib.config import C
    from qlib.data import D

    frame = D.features(
        ["SH600000"],
        ["$close", "$volume", "$amount", "$vwap", "$factor"],
        start_time="2026-01-02",
        end_time="2026-01-06",
    )

    assert frame.loc[("SH600000", pd.Timestamp("2026-01-02")), "$close"] == pytest.approx(10.0)
    assert np.isnan(frame.loc[("SH600000", pd.Timestamp("2026-01-05")), "$close"])
    assert frame.loc[("SH600000", pd.Timestamp("2026-01-06")), "$factor"] == pytest.approx(1.0)
    assert C.expression_cache is None
    assert C.dataset_cache is None
    assert C.default_disk_cache == 0


def test_real_qlib_features_support_multiple_worker_processes(
    direct_fixture: tuple[Path, Path, Path],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    direct.init_qlib_from_quant_data(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root,
        kernels=2,
    )
    from qlib.data import D

    frame = D.features(
        ["SH600000", "SZ000001"],
        ["$close", "Ref($close,1)", "Mean($volume,2)"],
        start_time="2026-01-02",
        end_time="2026-01-06",
    )

    assert frame.index.get_level_values("instrument").unique().tolist() == [
        "SH600000",
        "SZ000001",
    ]
    assert frame.loc[("SZ000001", pd.Timestamp("2026-01-05")), "$close"] == pytest.approx(21.0)


def test_cli_smoke_empty_returns_failure(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli
    from qlib.data import D

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start="2026-01-02",
            smoke_end="2026-01-02",
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)
    monkeypatch.setattr(
        D,
        "features",
        lambda *_, **__: pd.DataFrame(),
        raising=False,
    )

    return_code = cli.main()
    result = json.loads(capsys.readouterr().out)

    assert return_code == 1
    assert result["status"] == "smoke_failed"
    assert result["smoke_test"] == "empty"


def test_cli_does_not_expose_legacy_snapshot_path_overrides() -> None:
    from quant_workflow.scripts import build_qlib_direct_provider as cli

    args = cli.parse_args([])
    assert not hasattr(args, "membership_path")
    assert not hasattr(args, "cleaning_plan_path")

    with pytest.raises(SystemExit):
        cli.parse_args(["--membership-path", "/tmp/legacy.parquet"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--cleaning-plan-path", "/tmp/legacy.json"])


def test_cli_smoke_samples_multiple_historical_dates(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli
    from qlib.data import D

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start=None,
            smoke_end=None,
            smoke_samples=3,
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)

    def features(symbols, fields, *, start_time, end_time):
        calls.append((str(symbols[0]), str(start_time), str(end_time)))
        return pd.DataFrame({field: [1.0] for field in fields})

    monkeypatch.setattr(D, "features", features, raising=False)

    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)

    assert result["smoke_test"] == "pass"
    assert result["smoke_sample_count"] == 2
    assert len(calls) == 2 + len(direct.REQUIRED_INDEX_SYMBOLS) * 3
    stock_dates = {start for symbol, start, _ in calls if symbol == "SH600000"}
    assert stock_dates == {"2026-01-02", "2026-01-06"}


def test_cli_smoke_default_regression_uses_file_index_dates_and_five_indices(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli
    from qlib.data import D

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
            cleaning_plan_path=None,
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start=None,
            smoke_end=None,
            smoke_samples=5,
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)

    def features(symbols, fields, *, start_time, end_time):
        calls.append((str(symbols[0]), str(start_time), str(end_time)))
        return pd.DataFrame({field: [1.0] for field in fields})

    monkeypatch.setattr(D, "features", features, raising=False)

    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)

    assert result["smoke_test"] == "pass"
    # SH600000 has two actual rows; the suspended 2026-01-05 date is not sampled.
    assert result["smoke_sample_count"] == 2
    # Stock symbol (2 dates) + 5 indices (3 dates each) = 17 total calls.
    expected_calls = 2 + 5 * 3
    assert len(calls) == expected_calls
    # Verify all 5 required index symbols are present in calls
    called_symbols = {sym for sym, _, _ in calls}
    assert called_symbols == {"SH600000", *direct.REQUIRED_INDEX_SYMBOLS}
    # Verify smoke_checks structure contains symbol, start, end
    assert len(result["smoke_checks"]) == expected_calls
    assert all("symbol" in c and "start" in c and "end" in c for c in result["smoke_checks"])


def test_cli_smoke_symbol_missing_from_file_index_raises_error(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
            cleaning_plan_path=None,
            membership_path=None,
            output_root=None,
            smoke_symbol="SH999999",  # Missing symbol
            smoke_start=None,
            smoke_end=None,
            smoke_samples=5,
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)

    with pytest.raises(ValueError, match="smoke symbol is missing from its file index"):
        cli.main()


def test_cli_smoke_explicit_date_range_checks_indices_and_stock(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli
    from qlib.data import D

    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
            cleaning_plan_path=None,
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start="2026-01-02",
            smoke_end="2026-01-02",
            smoke_samples=5,
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)

    def features(symbols, fields, *, start_time, end_time):
        calls.append((str(symbols[0]), str(start_time), str(end_time)))
        return pd.DataFrame({field: [1.0] for field in fields})

    monkeypatch.setattr(D, "features", features, raising=False)

    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)

    assert result["smoke_test"] == "pass"
    # Stock + 5 indices on single explicit date = 6 calls
    assert len(calls) == 6
    called_symbols = {sym for sym, _, _ in calls}
    assert called_symbols == {"SH600000", *direct.REQUIRED_INDEX_SYMBOLS}
    assert all(start == "2026-01-02" and end == "2026-01-02" for _, start, end in calls)


def test_cli_smoke_rejects_partially_null_required_fields(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    from quant_workflow.scripts import build_qlib_direct_provider as cli
    from qlib.data import D

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
            cleaning_plan_path=None,
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start="2026-01-02",
            smoke_end="2026-01-02",
            smoke_samples=5,
            skip_smoke=False,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)
    monkeypatch.setattr(cli, "init_qlib_from_quant_data", lambda **_: None)

    def features(_symbols, fields, **_kwargs):
        values = {field: [1.0] for field in fields}
        values[fields[-1]] = [float("nan")]
        return pd.DataFrame(values)

    monkeypatch.setattr(D, "features", features, raising=False)

    assert cli.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "smoke_failed"
    assert result["smoke_failures"]
    assert {failure["reason"] for failure in result["smoke_failures"]} == {
        "missing_or_null_required_field"
    }


def test_cli_rejects_bin_file_inside_direct_metadata_directory(
    direct_fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    quant_data_root, _, descriptor = direct_fixture
    (descriptor.parent / "unexpected.day.bin").write_bytes(b"not-direct-metadata")
    from quant_workflow.scripts import build_qlib_direct_provider as cli

    monkeypatch.setattr(
        cli,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=quant_data_root,
            dataset_id="ashare_daily_qfq",
            index_dataset_id="cn_index_daily",
            cleaning_mode="canonical",
            cleaning_plan_path=None,
            membership_path=None,
            output_root=None,
            smoke_symbol="SH600000",
            smoke_start=None,
            smoke_end=None,
            smoke_samples=5,
            skip_smoke=True,
        ),
    )
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: descriptor)

    assert cli.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid_storage"
    assert result["market_data_bin_count"] == 1
