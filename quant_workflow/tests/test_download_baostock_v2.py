import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_baostock_v2.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("download_baostock_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-07-14", "2025-07-15", "2026-07-15", "2026-07-16"]
            ),
            "open": [10.0, 10.2, 12.0, 11.5],
            "high": [10.4, 10.5, 12.2, 11.8],
            "low": [9.8, 10.0, 11.8, 11.3],
            "close": [10.1, 10.3, 12.1, 11.6],
            "preclose": [9.9, 10.1, 12.0, 12.1],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "amount": [10100.0, 11330.0, 14520.0, 15080.0],
        }
    )


def factor_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["sh.600000", "sh.600000"],
            "dividOperateDate": ["2025-07-14", "2026-07-16"],
            "foreAdjustFactor": [0.95, 1.0],
            "backAdjustFactor": [10.0, 10.5],
            "adjustFactor": [10.0, 10.5],
        }
    )


class FakePagedResult:
    def __init__(
        self,
        *,
        pages: list[list[list[str]]],
        repeat_last_page: bool = False,
        fail_after_full_page: bool = False,
    ) -> None:
        self.fields = ["date"]
        self.error_code = "0"
        self.error_msg = ""
        self.per_page_count = 2
        self.cur_page_num = "1"
        self.cur_row_num = 0
        self.data = pages[0]
        self._pages = pages
        self._page_index = 0
        self._repeat_last_page = repeat_last_page
        self._fail_after_full_page = fail_after_full_page
        self._next_calls = 0

    def next(self) -> bool:
        self._next_calls += 1
        if self._next_calls > 10:
            raise RuntimeError("test safety limit reached")
        if self.cur_row_num < len(self.data):
            return True
        if self._fail_after_full_page:
            return False
        if self._page_index + 1 < len(self._pages):
            self._page_index += 1
            self.cur_page_num = str(self._page_index + 1)
            self.data = self._pages[self._page_index]
            self.cur_row_num = 0
            return bool(self.data)
        if self._repeat_last_page:
            self.cur_row_num = 0
            return True
        self.data = []
        self.cur_row_num = 0
        return False

    def get_row_data(self) -> list[str]:
        row = self.data[self.cur_row_num]
        self.cur_row_num += 1
        return row


def test_collect_result_accepts_normal_final_short_page() -> None:
    module = load_module()
    result = FakePagedResult(
        pages=[[["2026-07-22"], ["2026-07-23"]], [["2026-07-24"]]],
    )

    frame = module.collect_result(result)

    assert frame["date"].tolist() == ["2026-07-22", "2026-07-23", "2026-07-24"]


def test_atomic_json_retries_temporary_reader_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    destination = tmp_path / "progress.json"
    real_replace = module.os.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "file is temporarily held by a reader")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", flaky_replace)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    module.atomic_json(destination, {"status": "running"})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "running"
    }


def test_collect_result_rejects_repeated_provider_page() -> None:
    module = load_module()
    result = FakePagedResult(
        pages=[[["2026-07-22"], ["2026-07-23"]]],
        repeat_last_page=True,
    )

    with pytest.raises(module.BaostockV2Error, match="repeated result page"):
        module.collect_result(result)


def test_collect_result_rejects_truncated_full_page() -> None:
    module = load_module()
    result = FakePagedResult(
        pages=[[["2026-07-22"], ["2026-07-23"]]],
        fail_after_full_page=True,
    )

    with pytest.raises(module.BaostockV2Error, match="ended after a full page"):
        module.collect_result(result)


def test_expand_daily_factors_uses_factor_effective_on_event_date() -> None:
    module = load_module()
    factors = module.expand_daily_factors(raw_frame(), factor_events())

    assert factors["date"].tolist() == raw_frame()["date"].tolist()
    assert factors["factor"].tolist() == [0.95, 0.95, 0.95, 1.0]
    assert factors["factor_source"].eq("baostock_foreAdjustFactor").all()


def test_expand_daily_factors_accepts_parquet_microsecond_dates() -> None:
    module = load_module()
    raw = raw_frame()
    raw["date"] = raw["date"].astype("datetime64[us]")

    factors = module.expand_daily_factors(raw, factor_events())

    assert factors["date"].tolist() == raw["date"].tolist()
    assert factors["factor"].tolist() == [0.95, 0.95, 0.95, 1.0]


def test_expand_daily_factors_rejects_uncovered_early_bars() -> None:
    module = load_module()
    events = factor_events().iloc[1:].copy()

    with pytest.raises(module.BaostockV2Error, match="do not cover earliest raw bar"):
        module.expand_daily_factors(raw_frame(), events)


def test_expand_daily_factors_allows_bounded_listing_boundary_gap() -> None:
    module = load_module()
    raw = raw_frame().copy()
    raw.loc[raw.index[0], "date"] = pd.Timestamp("2025-07-13")

    factors = module.expand_daily_factors(raw, factor_events())

    assert factors.loc[0, "factor"] == 1.0
    assert factors.loc[0, "factor_source"] == "baostock_listing_gap_identity"
    assert bool(factors.loc[0, "factor_backfilled_before_first_event"])
    assert not factors.loc[1:, "factor_backfilled_before_first_event"].any()


def test_expand_daily_factors_uses_same_provider_pair_for_long_gap() -> None:
    module = load_module()
    raw = raw_frame()
    events = factor_events().iloc[1:].copy()
    provider_qfq = raw.copy()
    for column in module.PRICE_COLUMNS:
        provider_qfq.loc[provider_qfq.index[:3], column] *= 0.8

    factors = module.expand_daily_factors(
        raw,
        events,
        provider_qfq=provider_qfq,
    )

    assert factors["factor"].tolist() == [0.8, 0.8, 0.8, 1.0]
    assert factors.loc[:2, "factor_paired_fallback"].all()
    assert not bool(factors.loc[3, "factor_paired_fallback"])
    assert (
        factors.loc[:2, "factor_source"]
        == "baostock_raw_qfq_pair_fallback"
    ).all()


def test_expand_daily_factors_normalizes_provider_verified_non_unit_terminal_factor() -> None:
    module = load_module()
    raw = raw_frame()
    events = factor_events()
    events.loc[events.index[-1], "foreAdjustFactor"] = 1.034063
    provider_qfq = raw.copy()
    expected_factors = pd.Series(
        [0.95 / 1.034063, 0.95 / 1.034063, 0.95 / 1.034063, 1.0]
    )
    for column in module.PRICE_COLUMNS:
        provider_qfq[column] = provider_qfq[column] * expected_factors

    factors = module.expand_daily_factors(
        raw,
        events,
        provider_qfq=provider_qfq,
    )

    assert factors["factor"].tolist() == pytest.approx(expected_factors.tolist())
    assert factors["factor_source"].eq(
        "baostock_foreAdjustFactor_terminal_normalized"
    ).all()
    assert factors["factor_terminal_normalized"].all()


def test_expand_daily_factors_uses_last_traded_factor_before_future_event() -> None:
    module = load_module()
    raw = raw_frame().iloc[:3].copy()
    provider_qfq = raw.copy()
    for column in module.PRICE_COLUMNS:
        provider_qfq[column] = provider_qfq[column] * 0.95

    factors = module.expand_daily_factors(
        raw,
        factor_events(),
        provider_qfq=provider_qfq,
    )

    assert factors["factor"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert factors["factor_source"].eq(
        "baostock_foreAdjustFactor_terminal_normalized"
    ).all()
    assert factors["factor_terminal_normalized"].all()


def test_expand_daily_factors_normalizes_non_unit_provider_qfq_terminal_factor() -> None:
    module = load_module()
    raw = raw_frame()
    events = factor_events()
    events.loc[events.index[-1], "foreAdjustFactor"] = 1.034063
    provider_qfq = raw.copy()
    provider_native_factors = pd.Series([0.95, 0.95, 0.95, 1.034063])
    for column in module.PRICE_COLUMNS:
        provider_qfq[column] = provider_qfq[column] * provider_native_factors

    factors = module.expand_daily_factors(
        raw,
        events,
        provider_qfq=provider_qfq,
    )

    expected = provider_native_factors / provider_native_factors.iloc[-1]
    assert factors["factor"].tolist() == pytest.approx(expected.tolist())
    assert factors["factor_terminal_normalized"].all()


def test_expand_daily_factors_rejects_unverified_non_unit_terminal_factor() -> None:
    module = load_module()
    events = factor_events()
    events.loc[events.index[-1], "foreAdjustFactor"] = 1.034063

    with pytest.raises(module.BaostockV2Error, match="requires provider qfq verification"):
        module.expand_daily_factors(raw_frame(), events)


def test_paired_factor_derivation_rejects_inconsistent_ohlc_ratios() -> None:
    module = load_module()
    raw = raw_frame()
    provider_qfq = raw.copy()
    provider_qfq["open"] *= 0.8
    provider_qfq["close"] *= 0.7

    with pytest.raises(module.BaostockV2Error, match="paired qfq factors disagree"):
        module.derive_paired_daily_factors(raw, provider_qfq)


def test_build_local_qfq_adjusts_prices_but_not_volume_or_amount() -> None:
    module = load_module()
    raw = raw_frame()
    factors = module.expand_daily_factors(raw, factor_events())

    qfq = module.build_local_qfq(raw, factors)

    assert qfq["close"].tolist() == [9.595, 9.785, 11.495, 11.6]
    assert qfq["volume"].tolist() == raw["volume"].tolist()
    assert qfq["amount"].tolist() == raw["amount"].tolist()
    assert qfq["source_id"].eq("baostock_raw_x_foreAdjustFactor").all()


def test_compare_provider_qfq_accepts_rounding_and_rejects_scale_drift() -> None:
    module = load_module()
    raw = raw_frame()
    factors = module.expand_daily_factors(raw, factor_events())
    local = module.build_local_qfq(raw, factors)
    provider = local.copy()
    provider["close"] = provider["close"] + 0.0001

    result = module.compare_provider_qfq(local, provider)
    assert result["max_absolute_error"] < 0.001

    constant_scaled = local.copy()
    for column in module.PRICE_COLUMNS:
        constant_scaled[column] = constant_scaled[column] * 1.034063
    scaled_result = module.compare_provider_qfq(local, constant_scaled)
    assert scaled_result["provider_terminal_scale"] == pytest.approx(
        1 / 1.034063
    )
    assert scaled_result["max_absolute_error"] < 1e-10

    provider.loc[0, "close"] = provider.loc[0, "close"] + 0.1
    with pytest.raises(module.BaostockV2Error, match="provider qfq mismatch"):
        module.compare_provider_qfq(local, provider)


def test_normalize_factor_events_requires_complete_provider_fields() -> None:
    module = load_module()
    incomplete = factor_events().drop(columns=["foreAdjustFactor"])

    with pytest.raises(module.BaostockV2Error, match="factor fields are missing"):
        module.normalize_factor_events(incomplete, "SH600000")


def test_qfq_audit_always_includes_recent_factor_events() -> None:
    module = load_module()
    events = factor_events().copy()
    events.loc[events.index[-1], "dividOperateDate"] = "2026-07-20"

    assert module.needs_provider_qfq_audit(
        "SH600000",
        events,
        audit_since="2026-07-18",
        sample_modulus=100,
    )

    assert module.needs_provider_qfq_audit(
        "SH600000",
        factor_events(),
        audit_since="2026-07-18",
        sample_modulus=100,
        factor_backfill_used=True,
    )


def test_qfq_audit_always_includes_non_unit_terminal_factor() -> None:
    module = load_module()
    events = factor_events()
    events.loc[events.index[-1], "foreAdjustFactor"] = 1.034063

    assert module.needs_provider_qfq_audit(
        "SH600786",
        events,
        audit_since="2026-07-18",
        sample_modulus=100,
    )


def test_legacy_comparison_accepts_constant_rescaling(tmp_path: Path) -> None:
    module = load_module()
    new_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    new_dir.mkdir()
    legacy_dir.mkdir()
    dates = pd.to_datetime(["2026-07-21", "2026-07-22", "2026-07-23"])
    pd.DataFrame(
        {"date": dates, "close": [5.0, 5.5, 6.0]}
    ).to_parquet(new_dir / "SZ000001.parquet", index=False)
    pd.DataFrame(
        {"date": dates, "close": [10.0, 11.0, 12.0]}
    ).to_parquet(legacy_dir / "000001.parquet", index=False)

    summary, details = module.compare_legacy_qfq_directory(
        qfq_bars_dir=new_dir,
        legacy_qfq_dir=legacy_dir,
    )

    assert summary["status"] == "pass"
    assert summary["constant_scale_shift_symbols"] == 1
    assert summary["ratio_stability_rate"] == 1.0
    assert details.loc[0, "median_price_ratio"] == pytest.approx(0.5)


def test_legacy_comparison_rejects_mid_history_scale_drift(tmp_path: Path) -> None:
    module = load_module()
    new_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    new_dir.mkdir()
    legacy_dir.mkdir()
    dates = pd.to_datetime(["2026-07-21", "2026-07-22", "2026-07-23"])
    pd.DataFrame(
        {"date": dates, "close": [5.0, 7.0, 6.0]}
    ).to_parquet(new_dir / "SZ000001.parquet", index=False)
    pd.DataFrame(
        {"date": dates, "close": [10.0, 11.0, 12.0]}
    ).to_parquet(legacy_dir / "000001.parquet", index=False)

    summary, details = module.compare_legacy_qfq_directory(
        qfq_bars_dir=new_dir,
        legacy_qfq_dir=legacy_dir,
    )

    assert summary["status"] == "failed"
    assert summary["failed_gates"] == ["unexplained_price_path"]
    assert not bool(details.loc[0, "ratio_stable"])


def test_legacy_comparison_explains_corrupt_legacy_incremental_splice(
    tmp_path: Path,
) -> None:
    module = load_module()
    new_dir = tmp_path / "new"
    legacy_dir = tmp_path / "legacy"
    new_dir.mkdir()
    legacy_dir.mkdir()
    dates = pd.to_datetime(
        ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"]
    )
    pd.DataFrame(
        {
            "date": dates,
            "close": [5.0, 5.5, 6.0, 6.5],
        }
    ).to_parquet(new_dir / "SZ000001.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "close": [10.0, 11.0, 6.0, 6.5],
            "pctChg": [0.0, 10.0, 9.0909, 8.3333],
        }
    ).to_parquet(legacy_dir / "000001.parquet", index=False)

    summary, details = module.compare_legacy_qfq_directory(
        qfq_bars_dir=new_dir,
        legacy_qfq_dir=legacy_dir,
    )

    assert summary["status"] == "pass"
    assert summary["explained_legacy_scale_drift_symbols"] == 1
    assert summary["unexplained_price_path_symbols"] == 0
    assert bool(details.loc[0, "legacy_scale_drift_explained"])


def test_source_retrieval_window_includes_resumed_cached_files(
    tmp_path: Path,
) -> None:
    module = load_module()
    first = tmp_path / "first.parquet"
    last = tmp_path / "BUNDLE_MANIFEST.json"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    os.utime(first, (1_700_000_000, 1_700_000_000))
    os.utime(last, (1_700_003_600, 1_700_003_600))

    window = module.source_retrieval_window([first, last])

    assert window["started_at"] < window["completed_at"]
    assert window["started_at"].endswith(("+08:00", "Z", "+00:00"))


def test_resume_accepts_valid_empty_factor_event_file(tmp_path: Path) -> None:
    module = load_module()
    symbol = "SZ000001"
    module.atomic_parquet(raw_frame(), tmp_path / "raw" / f"{symbol}.parquet")
    empty_events = pd.DataFrame(columns=module.FACTOR_COLUMNS)
    module.atomic_parquet(
        empty_events,
        tmp_path / "factor_events" / f"{symbol}.parquet",
    )

    assert module._symbol_source_files_complete(tmp_path, symbol)


def test_build_snapshot_records_provider_verified_non_unit_terminal_factor(
    tmp_path: Path,
) -> None:
    module = load_module()
    bundle = tmp_path / "bundle"
    raw = raw_frame()
    events = factor_events()
    events.loc[events.index[-1], "foreAdjustFactor"] = 1.034063
    provider_qfq = raw.copy()
    expected_factors = pd.Series(
        [0.95 / 1.034063, 0.95 / 1.034063, 0.95 / 1.034063, 1.0]
    )
    for column in module.PRICE_COLUMNS:
        provider_qfq[column] = provider_qfq[column] * expected_factors
    raw_path = bundle / "raw" / "SH600786.parquet"
    event_path = bundle / "factor_events" / "SH600786.parquet"
    qfq_path = bundle / "qfq_provider" / "SH600786.parquet"
    module.atomic_parquet(raw, raw_path)
    module.atomic_parquet(events, event_path)
    module.atomic_parquet(provider_qfq, qfq_path)
    index = pd.DataFrame(
        [
            {
                "symbol": "SH600786",
                "raw_path": "raw/SH600786.parquet",
                "factor_events_path": "factor_events/SH600786.parquet",
                "qfq_provider_path": "qfq_provider/SH600786.parquet",
                "raw_sha256": module.sha256_file(raw_path),
                "factor_events_sha256": module.sha256_file(event_path),
                "qfq_provider_sha256": module.sha256_file(qfq_path),
            }
        ]
    )
    module.atomic_parquet(index, bundle / "BUNDLE_FILE_INDEX.parquet")
    module.atomic_json(
        bundle / "BUNDLE_MANIFEST.json",
        {
            "status": "ready",
            "completed_symbols": 1,
            "index_path": "BUNDLE_FILE_INDEX.parquet",
            "requested_through": "2026-07-24",
            "source_retrieved_at": "2026-07-24T18:00:00+08:00",
        },
    )

    _, qfq_stage = module.build_snapshots_from_bundle(
        bundle_dir=bundle,
        quant_data_root=tmp_path / "quant_data",
    )
    manifest = json.loads(
        (qfq_stage / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )

    adjustment = manifest["adjustment"]
    assert adjustment["normalization_rule"] == (
        "foreAdjustFactor_divided_by_terminal_when_non_unit; "
        "latest_valid_trade_date_factor_equals_1; "
        "provider_qfq_constant_scale_normalized_to_latest_valid_trade_date; "
        "same_provider_qfq_full_history_verified"
    )
    assert adjustment["non_unit_terminal_factor_symbols"] == 1
    assert adjustment["algorithm_version"] == "baostock-factor-evidence-v3"
    qfq_bars = pd.read_parquet(qfq_stage / "bars" / "SH600786.parquet")
    assert qfq_bars.sort_values("date")["close"].iloc[-1] == pytest.approx(
        raw.sort_values("date")["close"].iloc[-1]
    )


def test_build_snapshot_rejects_bundle_index_outside_bundle(tmp_path: Path) -> None:
    module = load_module()
    bundle = tmp_path / "bundle"
    outside_index = tmp_path / "outside-index.parquet"
    pd.DataFrame([{"symbol": "SH600786"}]).to_parquet(outside_index, index=False)
    module.atomic_json(
        bundle / "BUNDLE_MANIFEST.json",
        {
            "status": "ready",
            "completed_symbols": 1,
            "index_path": "../outside-index.parquet",
        },
    )

    with pytest.raises(module.BaostockV2Error, match="inside bundle"):
        module.build_snapshots_from_bundle(
            bundle_dir=bundle,
            quant_data_root=tmp_path / "quant_data",
        )


def test_incremental_bundle_keeps_empty_symbols_and_queries_full_factor_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "qfq-001",
                "manifest_sha256": "a" * 64,
                "latest_trade_date": "2026-08-04",
                "symbol_count": 2,
                "all": ["600000.XSHG", "000001.XSHE"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bundle"
    calls: list[tuple[str, str, str, str | None]] = []

    class LoginResult:
        error_code = "0"
        error_msg = ""

    class Provider:
        def login(self):
            return LoginResult()

        def logout(self):
            return LoginResult()

        def query_history_k_data_plus(self):
            raise AssertionError("query is intercepted")

        def query_adjust_factor(self):
            raise AssertionError("query is intercepted")

    provider = Provider()
    fields = module.HISTORY_FIELDS.split(",")

    def fake_query(_gate, function, **kwargs):
        calls.append(
            (
                function.__name__,
                str(kwargs["code"]),
                str(kwargs["start_date"]),
                kwargs.get("adjustflag"),
            )
        )
        if function.__name__ == "query_adjust_factor":
            return pd.DataFrame(columns=module.FACTOR_COLUMNS)
        if kwargs["code"] == "sz.000001":
            return pd.DataFrame(columns=fields)
        row = {field: "" for field in fields}
        row.update(
            {
                "date": "2026-08-04",
                "code": kwargs["code"],
                "open": "10.0",
                "high": "10.2",
                "low": "9.9",
                "close": "10.1",
                "preclose": "10.0",
                "volume": "1000",
                "amount": "10100",
                "turn": "0.1",
                "pctChg": "1.0",
                "tradestatus": "1",
                "isST": "0",
            }
        )
        return pd.DataFrame([row], columns=fields)

    monkeypatch.setattr(module, "_query_with_retry", fake_query)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    manifest_path = module.download_bundle(
        universe_file=universe,
        output_dir=output,
        through="2026-08-07",
        start="2026-07-27",
        factor_start="1990-01-01",
        mode="incremental",
        base_latest_trade_date="2026-08-04",
        request_interval=0.2,
        request_budget=20,
        retries=1,
        provider=provider,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = pd.read_parquet(output / "BUNDLE_FILE_INDEX.parquet")
    assert manifest["bundle_mode"] == "incremental"
    assert manifest["base_latest_trade_date"] == "2026-08-04"
    assert manifest["requested_start"] == "2026-07-27"
    assert manifest["factor_requested_start"] == "1990-01-01"
    assert manifest["universe"] == {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-001",
        "manifest_sha256": "a" * 64,
        "latest_trade_date": "2026-08-04",
        "symbol_count": 2,
        "universe_file_sha256": module.sha256_file(universe),
    }
    assert manifest["completed_symbols"] == 2
    assert manifest["empty_symbols"] == 1
    assert index["symbol"].tolist() == ["SH600000", "SZ000001"]
    assert index.set_index("symbol").loc["SZ000001", "raw_rows"] == 0
    factor_calls = [call for call in calls if call[0] == "query_adjust_factor"]
    assert factor_calls == [
        ("query_adjust_factor", "sh.600000", "1990-01-01", None),
        ("query_adjust_factor", "sz.000001", "1990-01-01", None),
    ]


def test_incremental_bundle_rejects_unbound_universe_before_provider_login(
    tmp_path: Path,
) -> None:
    module = load_module()
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps({"all": ["600000.XSHG"]}),
        encoding="utf-8",
    )

    class Provider:
        def login(self):
            raise AssertionError("provider login must not run")

    with pytest.raises(module.BaostockV2Error, match="frozen base snapshot"):
        module.download_bundle(
            universe_file=universe,
            output_dir=tmp_path / "bundle",
            through="2026-08-07",
            start="2026-07-27",
            factor_start="1990-01-01",
            mode="incremental",
            base_latest_trade_date="2026-08-04",
            request_interval=0.2,
            request_budget=20,
            retries=1,
            provider=Provider(),
        )


def test_download_cli_forwards_incremental_bundle_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    calls: list[dict[str, object]] = []
    manifest = tmp_path / "bundle" / "BUNDLE_MANIFEST.json"

    def fake_download_bundle(**kwargs):
        calls.append(kwargs)
        return manifest

    monkeypatch.setattr(module, "download_bundle", fake_download_bundle)

    result = module.main(
        [
            "download",
            "--universe-file",
            str(tmp_path / "universe.json"),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--through",
            "2026-08-07",
            "--start",
            "2026-07-27",
            "--mode",
            "incremental",
            "--factor-start",
            "1990-01-01",
            "--base-latest-trade-date",
            "2026-08-04",
        ]
    )

    assert result == 0
    assert calls[0]["mode"] == "incremental"
    assert calls[0]["factor_start"] == "1990-01-01"
    assert calls[0]["base_latest_trade_date"] == "2026-08-04"
