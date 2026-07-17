from __future__ import annotations

import importlib.util
import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
    from _robust_membership_integrity import membership_read_lock

    reference = root / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    with membership_read_lock(reference / "pit_csi800") as snapshot:
        membership_snapshot_hash = snapshot.fingerprint()
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": dates[0],
                    "end_date": dates[-1],
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_snapshot_hash,
            }
        ),
        encoding="utf-8",
    )
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
    membership.to_parquet(pit_dir / "membership_intervals.parquet", index=False)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
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
    assert len(result["membership_snapshot_hash"]) == 64
    assert result["reference_source"]["selected"] == "baostock"
    assert result["reference_source"]["industry_classification"] == "CSRC"
    assert len(result["reference_snapshot_hash"]) == 64
    assert all(check["passed"] for check in result["checks"])


def test_preflight_counts_verified_suspension_without_ohlcv_as_covered(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    price_path = price_dir / "000001.parquet"
    prices = pd.read_parquet(price_path)
    prices = prices[prices["date"] != pd.Timestamp("2026-01-06")]
    prices.to_parquet(price_path, index=False)
    suspension_path = tmp_path / "reference" / "suspension_status.parquet"
    suspension = pd.read_parquet(suspension_path)
    suspension.loc[
        suspension["date"].eq(pd.Timestamp("2026-01-06"))
        & suspension["symbol"].eq("SZ000001"),
        "is_suspended",
    ] = True
    suspension.to_parquet(suspension_path, index=False)

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=tmp_path / "reference",
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
        ohlcv_coverage_min=1.0,
    )

    assert result["status"] == "ready"
    assert result["summary"]["ohlcv_coverage"] == 1.0


def test_preflight_fails_fast_for_active_membership_writer_without_reading_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    parquet_reads: list[object] = []
    monkeypatch.setattr(
        module.pd,
        "read_parquet",
        lambda *args, **kwargs: parquet_reads.append(args[0] if args else None),
    )
    lock_path = pit_dir / ".build_pit_csi800.lock"
    started = time.monotonic()
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
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
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    assert time.monotonic() - started < 0.5
    assert result["status"] == "blocked_data"
    assert result["summary"]["failed_checks"] == ["membership_snapshot_integrity"]
    assert result["checks"][0]["details"]["reason_code"] == "membership_writer_active"
    assert parquet_reads == []


def test_preflight_rejects_membership_journal_without_reading_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    (pit_dir / ".build_pit_csi800.transaction.json").write_text(
        '{"state": "prepared"}\n', encoding="utf-8"
    )
    parquet_reads: list[object] = []
    monkeypatch.setattr(
        module.pd,
        "read_parquet",
        lambda *args, **kwargs: parquet_reads.append(args[0] if args else None),
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

    assert result["status"] == "blocked_data"
    assert (
        result["checks"][0]["details"]["reason_code"]
        == "membership_transaction_incomplete"
    )
    assert parquet_reads == []


def test_membership_fingerprint_changes_when_daily_partition_byte_changes(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, _, _ = build_small_valid_fixture(tmp_path)
    with module.membership_read_lock(pit_dir) as snapshot:
        first = snapshot.fingerprint()
    daily_path = pit_dir / "daily" / "2026.parquet"
    daily_path.write_bytes(daily_path.read_bytes() + b"changed")
    with module.membership_read_lock(pit_dir) as snapshot:
        second = snapshot.fingerprint()

    assert first != second


@pytest.mark.parametrize(
    ("requested", "selected", "fallback_reason"),
    [
        ("jqdata", "baostock", None),
        ("baostock", "jqdata", None),
        ("auto", "jqdata", "jqdata_permission_denied"),
        ("auto", "baostock", "permission_denied"),
    ],
)
def test_preflight_blocks_invalid_membership_source_cross_field_combinations(
    tmp_path: Path,
    requested: str,
    selected: str,
    fallback_reason: str | None,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = pit_dir / "MEMBERSHIP_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["requested"] = requested
    source["selected"] = selected
    if fallback_reason is None:
        source.pop("fallback_reason", None)
    else:
        source["fallback_reason"] = fallback_reason
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
    assert check["details"]["reason_code"] == "membership_source_invalid"
    assert result["membership_snapshot_hash"] is None


def test_preflight_blocks_dirty_reference_transaction_without_reading_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    reference = tmp_path / "reference"
    (reference / ".download_robust_reference.transaction.json").write_text(
        '{"state": "prepared"}\n', encoding="utf-8"
    )
    parquet_reads: list[object] = []

    def reject_parquet_read(*args, **kwargs):
        parquet_reads.append(args[0] if args else None)
        raise AssertionError("no parquet may be read while the journal exists")

    monkeypatch.setattr(module.pd, "read_parquet", reject_parquet_read)

    result = module.run_preflight(
        pit_dir=pit_dir,
        price_dir=price_dir,
        provider_dir=provider,
        reference_dir=reference,
        start_date="2026-01-05",
        end_date="2026-01-06",
        as_of_date="2026-01-06",
        expected_members_per_day=2,
    )

    check = next(
        item for item in result["checks"] if item["id"] == "reference_transaction_clean"
    )
    assert result["status"] == "blocked_data"
    assert check["passed"] is False
    assert check["details"]["reason_code"] == "reference_transaction_incomplete"
    assert parquet_reads == []


def test_preflight_blocks_missing_reference_provenance(tmp_path: Path) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    (tmp_path / "reference" / "REFERENCE_SOURCE.json").unlink()

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_source_missing"
    assert result["reference_source"] == {}


def test_preflight_blocks_reference_provenance_with_insufficient_range(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["date_range"]["end_date"] = "2026-01-05"
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_source_range_incomplete"


def test_preflight_blocks_reference_bound_to_different_membership_snapshot(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["membership_snapshot_hash"] = "f" * 64
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_membership_snapshot_mismatch"


def test_preflight_blocks_illegal_reference_source_and_unsafe_fallback(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["selected"] = "unapproved"
    source["fallback_reason"] = "api_key=secret"
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_source_invalid"
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(
    ("requested", "selected", "fallback_reason"),
    [
        ("jqdata", "baostock", None),
        ("jqdata", "jqdata", "permission_denied"),
        ("baostock", "jqdata", None),
        ("baostock", "baostock", "permission_denied"),
        ("auto", "jqdata", "permission_denied"),
        ("auto", "baostock", "unknown_reason"),
    ],
)
def test_preflight_blocks_invalid_reference_source_cross_field_combinations(
    tmp_path: Path,
    requested: str,
    selected: str,
    fallback_reason: str | None,
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["requested"] = requested
    source["selected"] = selected
    if fallback_reason is None:
        source.pop("fallback_reason", None)
    else:
        source["fallback_reason"] = fallback_reason
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_source_invalid"
    assert result["reference_snapshot_hash"] is None


@pytest.mark.parametrize(
    "fallback_reason",
    [
        "jqdata_auth_failed",
        "jqdata_sdk_unavailable",
        "jqdata_permission_denied",
        "jqdata_quota_exceeded",
    ],
)
def test_preflight_accepts_downloader_auto_baostock_fallback_reasons(
    tmp_path: Path, fallback_reason: str
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["fallback_reason"] = fallback_reason
    source_path.write_text(json.dumps(source), encoding="utf-8")

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

    assert result["status"] == "ready"
    assert result["reference_source"]["fallback_reason"] == fallback_reason


@pytest.mark.parametrize(
    "fallback_reason",
    [
        "auth_failed",
        "sdk_unavailable",
        "permission_denied",
        "quota_exceeded",
        "unknown_reason",
    ],
)
def test_preflight_rejects_unprefixed_or_unknown_auto_baostock_fallback_reasons(
    tmp_path: Path, fallback_reason: str
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["fallback_reason"] = fallback_reason
    source_path.write_text(json.dumps(source), encoding="utf-8")

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
        item for item in result["checks"] if item["id"] == "reference_provenance"
    )
    assert result["status"] == "blocked_data"
    assert check["details"]["reason_code"] == "reference_source_invalid"


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
    from _robust_membership_integrity import membership_read_lock

    with membership_read_lock(pit_dir) as snapshot:
        membership_snapshot_hash = snapshot.fingerprint()
    source_path = tmp_path / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["membership_snapshot_hash"] = membership_snapshot_hash
    source_path.write_text(json.dumps(source), encoding="utf-8")

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


def test_preflight_rechecks_membership_snapshot_before_return(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    original_metadata_check = module._metadata_check
    changed = False

    def changing_metadata_check(reference_dir, selected):
        nonlocal changed
        result = original_metadata_check(reference_dir, selected)
        if not changed:
            daily_path = pit_dir / "daily" / "2026.parquet"
            daily_path.write_bytes(daily_path.read_bytes() + b"changed-after-read")
            changed = True
        return result

    monkeypatch.setattr(module, "_metadata_check", changing_metadata_check)

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

    stability = next(
        item for item in result["checks"] if item["id"] == "snapshot_stability"
    )
    assert result["status"] == "blocked_data"
    assert stability["passed"] is False
    assert stability["details"]["reason_code"] == "membership_snapshot_changed"


def test_preflight_rechecks_reference_snapshot_before_return(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir, price_dir, provider = build_small_valid_fixture(tmp_path)
    original_lock = module.reference_read_lock
    completed_locks = 0

    @contextmanager
    def changing_reference_lock(reference_dir):
        nonlocal completed_locks
        with original_lock(reference_dir) as snapshot:
            yield snapshot
        completed_locks += 1
        if completed_locks == 3:
            path = Path(reference_dir) / "security_master.parquet"
            path.write_bytes(path.read_bytes() + b"changed-after-fingerprint")

    monkeypatch.setattr(module, "reference_read_lock", changing_reference_lock)

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

    stability = next(
        item for item in result["checks"] if item["id"] == "snapshot_stability"
    )
    assert result["status"] == "blocked_data"
    assert stability["passed"] is False
    assert stability["details"]["reason_code"] == "reference_snapshot_changed"


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
