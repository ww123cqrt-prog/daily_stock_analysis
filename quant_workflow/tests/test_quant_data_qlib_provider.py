from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_workflow import quant_data_qlib_provider as direct
from quant_workflow.scripts import _robust_membership_integrity as membership_integrity
from quant_workflow.scripts import quant_data_contract as contract


def _write_symbol(
    snapshot: Path,
    symbol: str,
    dates: list[str],
    closes: list[float],
) -> None:
    close = np.asarray(closes, dtype=float)
    volume = np.arange(1, len(dates) + 1, dtype=float) * 100.0
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
) -> Path:
    snapshot = quant_data_root / "staging" / snapshot_id / snapshot_id
    _write_symbol(
        snapshot,
        "SH600000",
        ["2026-01-02", "2026-01-06"],
        [sh_close, sh_close + 1.0],
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
) -> Path:
    dataset_root = (
        quant_data_root / "parquet" / "ashare_1d" / "ashare_daily_qfq"
    )
    staged = _stage_qfq_snapshot(
        quant_data_root,
        snapshot_id,
        sh_close=sh_close,
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
    pit_dir = quant_data_root / "reference" / "pit_csi800"
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
    with membership_integrity.membership_read_lock(pit_dir) as locked:
        locked.ensure_transaction_clean()
        snapshot_hash = locked.fingerprint()

    manifest_path = quant_data_root / "DATA_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    official_entries = dict(manifest.get("official_entries") or {})
    official_entries["pit_csi800"] = {
        "path": "reference/pit_csi800",
        "role": "Official point-in-time CSI800 membership snapshot.",
        "status": "do_not_move",
        "date_range": source["date_range"],
        "requested_source": source["requested"],
        "selected_source": source["selected"],
        "membership_snapshot_hash": snapshot_hash,
    }
    manifest["official_entries"] = official_entries
    contract.write_root_manifest(
        quant_data_root,
        manifest["datasets"],
        existing=manifest,
    )
    return intervals, snapshot_hash


@pytest.fixture()
def direct_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    quant_data_root = tmp_path / "quant_data"
    snapshot = _publish_qfq(
        quant_data_root,
        "qfq-001",
        sh_close=10.0,
        update_catalog=False,
    )
    membership, _ = _publish_membership(quant_data_root)
    descriptor = direct.build_direct_provider_descriptor(
        quant_data_root=quant_data_root,
        membership_path=membership,
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


def test_build_rejects_membership_missing_from_official_manifest(
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

    with pytest.raises(direct.DirectProviderError, match="official|registered"):
        direct.build_direct_provider_descriptor(
            quant_data_root=quant_data_root,
            membership_path=membership,
        )


def test_build_rejects_membership_while_writer_is_in_progress(
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

    with pytest.raises(direct.DirectProviderError, match="build_in_progress"):
        direct.build_direct_provider_descriptor(quant_data_root=quant_data_root)


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
        membership_path=(
            quant_data_root / "reference" / "pit_csi800" / "membership_intervals.parquet"
        ),
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
        membership_path=(
            quant_data_root / "reference" / "pit_csi800" / "membership_intervals.parquet"
        ),
    )
    current = direct.QuantDataFeatureProvider(
        descriptor_path=new_descriptor,
        quant_data_root=quant_data_root,
    )
    assert current.feature("SH600000", "$close", 0, 0, "day").iloc[0] == pytest.approx(99.0)


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
