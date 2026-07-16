from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robust_market_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("robust_market_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
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


def test_qlib_binary_reader_uses_calendar_start_offset(tmp_path: Path) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "features" / "sh000300").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n",
        encoding="utf-8",
    )
    np.asarray([1.0, 100.0, 101.0, 102.0], dtype="<f4").tofile(
        provider / "features" / "sh000300" / "open.day.bin"
    )

    series = module.read_qlib_feature(provider, "SH000300", "open")

    assert list(series.index) == list(
        pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"])
    )
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
