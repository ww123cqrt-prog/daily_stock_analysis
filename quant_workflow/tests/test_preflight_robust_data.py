from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "preflight_robust_data.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("preflight_robust_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_price(path: Path, dates: list[str], close: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": [close] * len(dates),
            "high": [close] * len(dates),
            "low": [close] * len(dates),
            "close": [close] * len(dates),
            "volume": [1000.0] * len(dates),
        }
    ).to_parquet(path, index=False)


def write_reference_tables(root: Path, dates: list[str], symbols: list[str]) -> None:
    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": symbols,
            "list_date": ["2010-01-01"] * len(symbols),
            "delist_date": [None] * len(symbols),
            "board": ["main"] * len(symbols),
        }
    ).to_parquet(reference / "security_master.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": symbols,
            "industry_code": ["bank"] * len(symbols),
            "start_date": [dates[0]] * len(symbols),
            "end_date": [dates[-1]] * len(symbols),
        }
    ).to_parquet(reference / "industry_intervals.parquet", index=False)
    daily = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    daily.assign(is_st=False).to_parquet(reference / "st_status.parquet", index=False)
    daily.assign(is_suspended=False).to_parquet(
        reference / "suspension_status.parquet", index=False
    )


def write_qlib_benchmark(provider: Path, symbol: str, periods: int) -> None:
    feature_dir = provider / "features" / symbol.lower()
    feature_dir.mkdir(parents=True, exist_ok=True)
    for field in ["open", "high", "low", "close", "volume"]:
        values = [1000.0] * periods if field == "volume" else [100.0] * periods
        np.asarray([0.0, *values], dtype="<f4").tofile(feature_dir / f"{field}.day.bin")


def build_small_valid_fixture(root: Path) -> tuple[Path, Path, Path]:
    dates = ["2026-01-05", "2026-01-06"]
    symbols = ["SH600000", "SZ000001"]
    pit_dir = root / "reference" / "pit_csi800"
    (pit_dir / "daily").mkdir(parents=True)
    membership = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    membership["index_code"] = ["000300.XSHG", "000905.XSHG"] * len(dates)
    membership.to_parquet(pit_dir / "daily" / "2026.parquet", index=False)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "date_range": {
                    "start_date": dates[0],
                    "end_date": dates[-1],
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )

    price_dir = root / "parquet" / "ashare_1d" / "cleaned"
    for symbol in symbols:
        write_price(price_dir / f"{symbol[-6:]}.parquet", dates)
    write_reference_tables(root, dates, symbols)
    calendar_path = root / "qlib" / "provider" / "calendars" / "day.txt"
    calendar_path.parent.mkdir(parents=True)
    calendar_path.write_text("\n".join(dates) + "\n", encoding="utf-8")
    provider = calendar_path.parent.parent
    for benchmark in ["SH000300", "SH000905", "SH000906"]:
        write_qlib_benchmark(provider, benchmark, len(dates))
    return pit_dir, price_dir, provider


def test_symbol_file_lookup_matches_real_numeric_parquet_names(tmp_path: Path) -> None:
    module = load_module()
    price_dir = tmp_path / "prices"
    write_price(price_dir / "600000.parquet", ["2026-01-05"])

    assert (
        module._find_symbol_file(price_dir, "SH600000") == price_dir / "600000.parquet"
    )


def test_preflight_passes_complete_fixture_with_configurable_expected_members(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
        membership_coverage_min=0.99,
        ohlcv_coverage_min=0.98,
    )

    assert result["status"] == "ready"
    assert result["summary"]["membership_coverage"] == 1.0
    assert result["summary"]["ohlcv_coverage"] == 1.0
    assert all(check["passed"] for check in result["checks"])


def test_preflight_blocks_membership_without_valid_source_provenance(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    (pit_dir / "MEMBERSHIP_SOURCE.json").unlink()

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
    )

    check = next(
        item for item in result["checks"] if item["id"] == "membership_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["passed"] is False
    assert check["details"]["reason_code"] == "membership_source_missing"


def test_preflight_blocks_when_benchmark_and_metadata_are_missing(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    for path in (provider / "features" / "sh000905").iterdir():
        path.unlink()
    (provider / "features" / "sh000905").rmdir()
    (tmp_path / "reference" / "st_status.parquet").unlink()

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
    )

    assert result["status"] == "blocked_data"
    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert "required_benchmarks" in failed
    assert "reference_metadata" in failed


def test_preflight_blocks_when_daily_metadata_has_member_day_gap(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    path = tmp_path / "reference" / "st_status.parquet"
    st = pd.read_parquet(path).iloc[1:].copy()
    st.to_parquet(path, index=False)

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
    )

    check = next(
        item for item in result["checks"] if item["id"] == "reference_metadata"
    )
    assert result["status"] == "blocked_data"
    assert check["passed"] is False
    assert check["details"]["member_day_gaps"]["st_status"] == 1


def test_preflight_blocks_duplicate_and_future_membership_without_reading_future_prices(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    year_path = pit_dir / "daily" / "2026.parquet"
    membership = pd.read_parquet(year_path)
    duplicate = membership.iloc[[0]].copy()
    future = membership.iloc[[0]].copy()
    future["date"] = pd.Timestamp("2026-01-07")
    pd.concat([membership, duplicate, future], ignore_index=True).to_parquet(
        year_path, index=False
    )

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["status"] == "blocked_data"
    assert "membership_duplicates" in failed
    assert "future_dates" in failed


def test_preflight_writes_machine_readable_blocked_report(tmp_path: Path) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    for path in (provider / "features" / "sh000906").iterdir():
        path.unlink()
    (provider / "features" / "sh000906").rmdir()
    out_dir = tmp_path / "reports" / "robust_strategy"

    return_code = module.main(
        [
            "--pit-dir",
            str(pit_dir),
            "--price-dir",
            str(price_dir),
            "--provider-dir",
            str(provider),
            "--reference-dir",
            str(tmp_path / "reference"),
            "--start-date",
            "2026-01-05",
            "--end-date",
            "2026-01-06",
            "--as-of-date",
            "2026-01-06",
            "--expected-members-per-day",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )

    payload = json.loads(
        (out_dir / "preflight_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 2
    assert payload["status"] == "blocked_data"
    assert (out_dir / "preflight_latest.md").is_file()
