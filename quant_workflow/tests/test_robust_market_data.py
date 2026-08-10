from __future__ import annotations

import importlib.util
import fcntl
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robust_market_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("robust_market_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    from _robust_membership_integrity import membership_read_lock

    dates = pd.bdate_range("2025-01-02", periods=4)
    pit_dir = root / "pit" / "daily"
    pit_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": [dates[1], dates[2], dates[2], dates[3]],
            "symbol": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
            "index_code": ["000300.XSHG", "000300.XSHG", "000905.XSHG", "000905.XSHG"],
        }
    ).to_parquet(pit_dir / "2025.parquet", index=False)
    pit_root = root / "pit"
    pd.DataFrame(
        {
            "symbol": ["SH600000", "SZ000001"],
            "index_code": ["000300.XSHG", "000905.XSHG"],
            "start_date": [dates[1], dates[2]],
            "end_date": [dates[2], dates[3]],
        }
    ).to_parquet(pit_root / "membership_intervals.parquet", index=False)
    (pit_root / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": str(dates[0].date()),
                    "end_date": str(dates[-1].date()),
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )
    with membership_read_lock(pit_root) as snapshot:
        membership_snapshot_hash = snapshot.fingerprint()
    price_dir = root / "prices"
    price_dir.mkdir()
    for symbol, code, base in [
        ("SH600000", "600000", 10.0),
        ("SZ000001", "000001", 20.0),
    ]:
        pd.DataFrame(
            {
                "date": dates,
                "open": [base, base + 1, base + 2, base + 3],
                "close": [base + 0.5, base + 1.5, base + 2.5, base + 3.5],
                "high": [base + 1, base + 2, base + 3, base + 4],
                "low": [base - 1, base, base + 1, base + 2],
                "volume": [1000.0] * 4,
            }
        ).to_parquet(price_dir / f"{code}.parquet", index=False)
    reference = root / "reference"
    reference.mkdir()
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": str(dates[0].date()),
                    "end_date": str(dates[-1].date()),
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_snapshot_hash,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "symbol": ["SH600000", "SZ000001"],
            "list_date": ["1999-01-01", "1991-01-01"],
            "delist_date": [None, None],
            "board": ["main", "main"],
        }
    ).to_parquet(reference / "security_master.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000", "SZ000001"],
            "industry_code": ["old_bank", "new_bank", "finance"],
            "start_date": [dates[0], dates[2], dates[0]],
            "end_date": [dates[1], dates[3], dates[3]],
        }
    ).to_parquet(reference / "industry_intervals.parquet", index=False)
    keys = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SZ000001"]], names=["date", "symbol"]
    ).to_frame(index=False)
    st = keys.copy()
    st["is_st"] = False
    st.loc[(st["date"] == dates[2]) & (st["symbol"] == "SZ000001"), "is_st"] = True
    st.to_parquet(reference / "st_status.parquet", index=False)
    suspension = keys.copy()
    suspension["is_suspended"] = False
    suspension.loc[
        (suspension["date"] == dates[3]) & (suspension["symbol"] == "SZ000001"),
        "is_suspended",
    ] = True
    suspension.to_parquet(reference / "suspension_status.parquet", index=False)
    return root / "pit", price_dir, reference


def test_market_panel_uses_only_pit_member_days_and_historical_metadata(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)

    panel = module.load_market_panel(
        pit_dir=pit_dir,
        price_dir=price_dir,
        reference_dir=reference,
        start_date="2025-01-03",
        end_date="2025-01-07",
    )

    assert list(panel[["date", "symbol"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2025-01-03"), "SH600000"),
        (pd.Timestamp("2025-01-06"), "SH600000"),
        (pd.Timestamp("2025-01-06"), "SZ000001"),
        (pd.Timestamp("2025-01-07"), "SZ000001"),
    ]
    sh_old = panel[
        (panel["date"] == pd.Timestamp("2025-01-03")) & (panel["symbol"] == "SH600000")
    ].iloc[0]
    sh_new = panel[
        (panel["date"] == pd.Timestamp("2025-01-06")) & (panel["symbol"] == "SH600000")
    ].iloc[0]
    sz_st = panel[
        (panel["date"] == pd.Timestamp("2025-01-06")) & (panel["symbol"] == "SZ000001")
    ].iloc[0]
    sz_paused = panel[
        (panel["date"] == pd.Timestamp("2025-01-07")) & (panel["symbol"] == "SZ000001")
    ].iloc[0]
    assert sh_old["industry"] == "old_bank"
    assert sh_new["industry"] == "new_bank"
    assert bool(sz_st["is_st"]) is True
    assert bool(sz_paused["suspended"]) is True
    assert sh_old["previous_close"] == 10.5
    assert sh_old["list_date"] == pd.Timestamp("1999-01-01")
    assert sh_old["listing_trading_day"] > 5
    assert bool(sh_old["listing_history_complete"]) is False
    assert panel.attrs["reference_source_metadata"]["selected"] == "baostock"
    assert (
        panel.attrs["reference_source_metadata"]["fallback_reason"]
        == "jqdata_permission_denied"
    )
    assert len(panel.attrs["reference_snapshot_hash"]) == 64
    assert panel.attrs["membership_source_metadata"]["selected"] == "baostock"
    assert (
        panel.attrs["membership_source_metadata"]["fallback_reason"]
        == "jqdata_permission_denied"
    )
    assert len(panel.attrs["membership_snapshot_hash"]) == 64


def test_market_panel_allows_missing_prices_on_verified_suspension_day(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    path = price_dir / "000001.parquet"
    prices = pd.read_parquet(path)
    paused = prices["date"].eq(pd.Timestamp("2025-01-07"))
    prices.loc[paused, ["open", "close", "high", "low"]] = np.nan
    prices.to_parquet(path, index=False)

    panel = module.load_market_panel(
        pit_dir=pit_dir,
        price_dir=price_dir,
        reference_dir=reference,
        start_date="2025-01-03",
        end_date="2025-01-07",
    )

    row = panel[
        (panel["date"] == pd.Timestamp("2025-01-07")) & (panel["symbol"] == "SZ000001")
    ].iloc[0]
    assert bool(row["suspended"]) is True
    assert pd.isna(row["open"])


def test_market_panel_rejects_price_history_missing_listing_start(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    path = price_dir / "600000.parquet"
    prices = pd.read_parquet(path).iloc[1:].reset_index(drop=True)
    prices.to_parquet(path, index=False)
    security_path = reference / "security_master.parquet"
    security = pd.read_parquet(security_path)
    security["list_date"] = pd.to_datetime(security["list_date"])
    security.loc[security["symbol"] == "SH600000", "list_date"] = pd.Timestamp(
        "2025-01-02"
    )
    security.to_parquet(security_path, index=False)

    with pytest.raises(ValueError, match="listing date"):
        module.load_market_panel(
            pit_dir=pit_dir,
            price_dir=price_dir,
            reference_dir=reference,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )


def test_reference_read_lock_blocks_writer_after_integrity_check(
    tmp_path: Path,
) -> None:
    module = load_module()
    _, _, reference = build_fixture(tmp_path)
    lock_path = reference.parent / f".{reference.name}.download_robust_reference.lock"
    writer_attempting = threading.Event()
    writer_acquired = threading.Event()

    def writer() -> None:
        with lock_path.open("a+b") as handle:
            writer_attempting.set()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            writer_acquired.set()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    thread = threading.Thread(target=writer)
    with module.reference_read_lock(reference) as snapshot:
        snapshot.ensure_transaction_clean()
        snapshot.validate_source(start_date="2025-01-03", end_date="2025-01-07")
        thread.start()
        assert writer_attempting.wait(timeout=1.0)
        assert writer_acquired.wait(timeout=0.1) is False
        assert len(snapshot.fingerprint()) == 64

    assert writer_acquired.wait(timeout=1.0)
    thread.join(timeout=1.0)
    assert thread.is_alive() is False


def test_market_panel_rejects_dirty_reference_transaction_before_any_parquet_read(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    (reference / ".download_robust_reference.transaction.json").write_text(
        '{"state": "prepared"}\n', encoding="utf-8"
    )
    parquet_reads: list[object] = []

    def reject_parquet_read(*args, **kwargs):
        parquet_reads.append(args[0] if args else None)
        raise AssertionError("no parquet may be read while the journal exists")

    monkeypatch.setattr(module.pd, "read_parquet", reject_parquet_read)

    with pytest.raises(ValueError, match="reference transaction"):
        module.load_market_panel(
            pit_dir=pit_dir,
            price_dir=price_dir,
            reference_dir=reference,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )

    assert parquet_reads == []


def test_market_panel_rejects_reference_source_range_before_reading_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    source_path = reference / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["date_range"]["end_date"] = "2025-01-06"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    parquet_reads: list[object] = []
    monkeypatch.setattr(
        module.pd,
        "read_parquet",
        lambda *args, **kwargs: parquet_reads.append(args[0] if args else None),
    )

    with pytest.raises(ValueError, match="reference source range"):
        module.load_market_panel(
            pit_dir=pit_dir,
            price_dir=price_dir,
            reference_dir=reference,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )

    assert parquet_reads == []


def test_market_panel_rejects_reference_from_different_membership_snapshot(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    source_path = reference / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["membership_snapshot_hash"] = "f" * 64
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="different membership snapshot"):
        module.load_market_panel(
            pit_dir=pit_dir,
            price_dir=price_dir,
            reference_dir=reference,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )


def test_reference_fingerprint_rejects_symlinked_snapshot_file(tmp_path: Path) -> None:
    module = load_module()
    _, _, reference = build_fixture(tmp_path)
    external = tmp_path / "external-security.parquet"
    target = reference / "security_master.parquet"
    target.replace(external)
    target.symlink_to(external)

    with pytest.raises(ValueError, match="snapshot files are incomplete"):
        with module.reference_read_lock(reference) as snapshot:
            snapshot.fingerprint()


@pytest.mark.parametrize("changed_snapshot", ["membership", "reference"])
def test_market_panel_rechecks_snapshots_before_return(
    tmp_path: Path, monkeypatch, changed_snapshot: str
) -> None:
    module = load_module()
    pit_dir, price_dir, reference = build_fixture(tmp_path)
    original_industry_lookup = module._industry_for_members
    changed = False

    def changing_industry_lookup(members, intervals):
        nonlocal changed
        result = original_industry_lookup(members, intervals)
        if not changed:
            path = (
                pit_dir / "daily" / "2025.parquet"
                if changed_snapshot == "membership"
                else reference / "security_master.parquet"
            )
            path.write_bytes(path.read_bytes() + b"changed-after-read")
            changed = True
        return result

    monkeypatch.setattr(module, "_industry_for_members", changing_industry_lookup)

    with pytest.raises(ValueError, match=f"{changed_snapshot} snapshot changed"):
        module.load_market_panel(
            pit_dir=pit_dir,
            price_dir=price_dir,
            reference_dir=reference,
            start_date="2025-01-03",
            end_date="2025-01-07",
        )


def test_qlib_feature_reader_uses_the_initialized_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    provider.mkdir()
    dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    feature_index = pd.MultiIndex.from_arrays(
        [["SH000300"] * len(dates), dates],
        names=["instrument", "datetime"],
    )
    feature_frame = pd.DataFrame(
        {"$open": [100.0, 101.0, 102.0]},
        index=feature_index,
    )
    calls: list[tuple[object, ...]] = []

    class FakeDataApi:
        @staticmethod
        def features(*args, **kwargs):
            calls.append((args, kwargs))
            return feature_frame

    qlib_module = ModuleType("qlib")
    qlib_data_module = ModuleType("qlib.data")
    qlib_data_module.D = FakeDataApi
    qlib_module.data = qlib_data_module
    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_module)
    initialized: list[Path] = []
    monkeypatch.setattr(
        module,
        "init_qlib_workflow",
        lambda *, provider_uri: initialized.append(Path(provider_uri)),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "provider_calendar_days",
        lambda _provider: [value.date().isoformat() for value in dates],
        raising=False,
    )

    series = module.read_qlib_feature(provider, "SH000300", "open")

    assert initialized == [provider]
    assert calls == [
        (
            (["SH000300"], ["$open"]),
            {
                "start_time": "2025-01-03",
                "end_time": "2025-01-07",
                "freq": "day",
            },
        )
    ]
    assert list(series.index) == list(dates)
    assert series.tolist() == [100.0, 101.0, 102.0]


def test_regime_history_is_unchanged_when_only_future_rows_change() -> None:
    module = load_module()
    dates = pd.bdate_range("2024-01-02", periods=280)
    symbols = [f"SH600{i:03d}" for i in range(120)]
    index = pd.MultiIndex.from_product(
        [dates, symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    index["close"] = (
        np.tile(np.linspace(10.0, 20.0, len(dates)), len(symbols))
        .reshape(len(symbols), len(dates))
        .T.ravel()
    )
    panel = index.copy()

    first = module.build_market_regimes(panel)
    changed = panel.copy()
    cutoff = dates[-20]
    changed.loc[changed["date"] > cutoff, "close"] *= 0.2
    second = module.build_market_regimes(changed)

    assert first.loc[first.index <= cutoff].equals(second.loc[second.index <= cutoff])


def test_regime_history_ignores_symbols_that_only_appear_in_the_future() -> None:
    module = load_module()
    dates = pd.bdate_range("2024-01-02", periods=120)
    existing_symbols = [f"SH600{i:03d}" for i in range(100)]
    existing = pd.MultiIndex.from_product(
        [dates, existing_symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    existing["close"] = np.repeat(
        np.linspace(10.0, 20.0, len(dates)), len(existing_symbols)
    )
    cutoff = dates[-20]
    future_symbols = [f"SZ300{i:03d}" for i in range(100)]
    future = pd.MultiIndex.from_product(
        [dates[dates > cutoff], future_symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    future["close"] = 10.0

    baseline = module.build_market_regimes(existing)
    with_future_members = module.build_market_regimes(
        pd.concat([existing, future], ignore_index=True)
    )

    pd.testing.assert_series_equal(
        baseline.loc[baseline.index <= cutoff],
        with_future_members.loc[with_future_members.index <= cutoff],
    )
