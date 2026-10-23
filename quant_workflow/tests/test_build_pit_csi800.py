from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_pit_csi800.py"
CSI300 = "000300.XSHG"
CSI500 = "000905.XSHG"


def load_module():
    spec = importlib.util.spec_from_file_location("build_pit_csi800", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_members() -> dict[str, list[str]]:
    return {
        CSI300: [f"{600000 + offset:06d}.XSHG" for offset in range(300)],
        CSI500: [f"{1 + offset:06d}.XSHE" for offset in range(500)],
    }


def normalized_daily_membership(dates: list[str]) -> pd.DataFrame:
    records: list[tuple[pd.Timestamp, str, str]] = []
    for value in dates:
        timestamp = pd.Timestamp(value)
        records.extend(
            (timestamp, f"SH{600000 + offset:06d}", CSI300) for offset in range(300)
        )
        records.extend(
            (timestamp, f"SZ{1 + offset:06d}", CSI500) for offset in range(500)
        )
    return pd.DataFrame(records, columns=["date", "symbol", "index_code"])


class FakeProvider:
    source_name = "test_fixture"

    def __init__(
        self,
        trading_days: list[str],
        members: dict[str, list[str]] | None = None,
        *,
        member_error: Exception | None = None,
        member_errors_by_date: dict[str, Exception] | None = None,
    ) -> None:
        self.trading_days = trading_days
        self.members = valid_members() if members is None else members
        self.member_error = member_error
        self.member_errors_by_date = member_errors_by_date or {}
        self.calendar_calls: list[tuple[date, date]] = []
        self.member_calls: list[tuple[str, date]] = []

    def get_trading_days(self, start_date: date, end_date: date):
        self.calendar_calls.append((start_date, end_date))
        return [
            value
            for value in self.trading_days
            if start_date <= date.fromisoformat(value) <= end_date
        ]

    def get_index_members(self, index_code: str, on_date: date):
        self.member_calls.append((index_code, on_date))
        dated_error = self.member_errors_by_date.get(on_date.isoformat())
        if dated_error is not None:
            raise dated_error
        if self.member_error is not None:
            raise self.member_error
        return list(self.members[index_code])


class ConcurrentFakeProvider(FakeProvider):
    def __init__(
        self,
        trading_days: list[str],
        *,
        slow_date: str | None,
        slow_started,
        release_slow,
    ) -> None:
        super().__init__(trading_days)
        self.slow_date = date.fromisoformat(slow_date) if slow_date else None
        self.slow_started = slow_started
        self.release_slow = release_slow

    def get_index_members(self, index_code: str, on_date: date):
        if on_date == self.slow_date and index_code == CSI300:
            self.slow_started.set()
            if not self.release_slow.wait(timeout=10):
                raise TimeoutError("test did not release slow provider")
        return super().get_index_members(index_code, on_date)


class CalendarResponseProvider(FakeProvider):
    def __init__(
        self,
        responses: dict[tuple[date, date], list[str]],
    ) -> None:
        super().__init__([])
        self.responses = responses

    def get_trading_days(self, start_date: date, end_date: date):
        self.calendar_calls.append((start_date, end_date))
        return list(self.responses.get((start_date, end_date), []))


def run_concurrent_build(
    output_dir: str,
    status_dir: str,
    start_date: str,
    end_date: str,
    slow_date: str | None,
    slow_started,
    release_slow,
    finished,
    result_queue,
) -> None:
    module = load_module()
    provider = ConcurrentFakeProvider(
        ["2025-01-02", "2025-01-03", "2025-01-06"],
        slow_date=slow_date,
        slow_started=slow_started,
        release_slow=release_slow,
    )
    try:
        return_code = module.main(
            [
                "--start-date",
                start_date,
                "--end-date",
                end_date,
                "--output-dir",
                output_dir,
                "--status-out-dir",
                status_dir,
            ],
            provider=provider,
            today=date(2025, 1, 6),
        )
        result_queue.put(return_code)
    finally:
        finished.set()


def run_status_order_build(
    output_dir: str,
    status_dir: str,
    requested_date: str,
    delay_after_build: bool,
    build_returned,
    release_return,
    finished,
    result_queue,
) -> None:
    module = load_module()
    if delay_after_build:
        original_build = module.build_pit_csi800

        def delayed_build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            build_returned.set()
            if not release_return.wait(timeout=10):
                raise TimeoutError("test did not release delayed build")
            return result

        module.build_pit_csi800 = delayed_build
    provider = FakeProvider(["2025-01-02", "2025-01-03"])
    try:
        return_code = module.main(
            [
                "--start-date",
                requested_date,
                "--end-date",
                requested_date,
                "--output-dir",
                output_dir,
                "--status-out-dir",
                status_dir,
            ],
            provider=provider,
            today=date(2025, 1, 3),
        )
        result_queue.put(return_code)
    finally:
        finished.set()


def run_main(
    module,
    tmp_path: Path,
    provider: FakeProvider,
    *,
    start_date: str = "2025-01-02",
    end_date: str = "2025-01-03",
    extra_args: list[str] | None = None,
) -> tuple[int, Path, Path]:
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    args = [
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--output-dir",
        str(output_dir),
        "--status-out-dir",
        str(status_dir),
    ]
    args.extend(extra_args or [])
    return_code = module.main(args, provider=provider, today=date(2025, 12, 31))
    return return_code, output_dir, status_dir


def test_builds_normalized_daily_membership_intervals_and_separate_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])

    return_code, output_dir, status_dir = run_main(module, tmp_path, provider)

    daily = pd.read_parquet(output_dir / "daily" / "2025.parquet")
    intervals = pd.read_parquet(output_dir / "membership_intervals.parquet")
    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert len(daily) == 1600
    assert pd.api.types.is_datetime64_any_dtype(daily["date"])
    assert daily.groupby(["date", "index_code"]).size().unique().tolist() == [300, 500]
    assert daily.groupby("date")["symbol"].nunique().eq(800).all()
    assert {"SH600000", "SZ000001"}.issubset(set(daily["symbol"]))
    assert len(intervals) == 800
    assert intervals["start_date"].eq(pd.Timestamp("2025-01-02")).all()
    assert intervals["end_date"].eq(pd.Timestamp("2025-01-03")).all()
    assert provider.member_calls == [
        (CSI300, date(2025, 1, 2)),
        (CSI500, date(2025, 1, 2)),
        (CSI300, date(2025, 1, 3)),
        (CSI500, date(2025, 1, 3)),
    ]
    assert status["status"] == "completed"
    assert status["date_range"] == {
        "start_date": "2025-01-02",
        "end_date": "2025-01-03",
    }
    assert status["completed_date_count"] == 2
    assert status["missing_summary"]["count"] == 0
    assert status["anomaly_summary"]["count"] == 0
    assert status["membership_source"]["selected"] == "test_fixture"
    assert not (output_dir / module.BUILD_IN_PROGRESS_FILENAME).exists()
    source = json.loads(
        (output_dir / "MEMBERSHIP_SOURCE.json").read_text(encoding="utf-8")
    )
    assert source["selected"] == "test_fixture"
    assert source["date_range"] == {
        "start_date": "2025-01-02",
        "end_date": "2025-01-03",
    }
    assert not (output_dir / "build_status_latest.json").exists()
    assert not (status_dir / "daily").exists()
    assert not (status_dir / "membership_intervals.parquet").exists()


def test_baostock_provider_reads_historical_calendar_and_members() -> None:
    module = load_module()

    class Result:
        error_code = "0"
        error_msg = "success"

        def __init__(self, rows):
            self.rows = list(rows)
            self.offset = 0

        def next(self):
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def __init__(self):
            self.member_calls = []
            self.logged_out = False

        def login(self):
            return type("Login", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self):
            self.logged_out = True

        def query_trade_dates(self, *, start_date, end_date):
            assert (start_date, end_date) == ("2014-01-01", "2014-01-03")
            return Result(
                [
                    ["2014-01-01", "0"],
                    ["2014-01-02", "1"],
                    ["2014-01-03", "1"],
                ]
            )

        def query_hs300_stocks(self, *, date):
            self.member_calls.append(("csi300", date))
            return Result([["2013-12-30", "sh.600000", "name"]])

        def query_zz500_stocks(self, *, date):
            self.member_calls.append(("csi500", date))
            return Result([["2013-12-30", "sz.000001", "name"]])

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    days = provider.get_trading_days(date(2014, 1, 1), date(2014, 1, 3))
    csi300 = provider.get_index_members(CSI300, date(2014, 1, 2))
    csi500 = provider.get_index_members(CSI500, date(2014, 1, 2))
    provider.close()

    assert days == [date(2014, 1, 2), date(2014, 1, 3)]
    assert csi300 == ["SH600000"]
    assert csi500 == ["SZ000001"]
    assert fake.member_calls == [
        ("csi300", "2014-01-02"),
        ("csi500", "2014-01-02"),
    ]
    assert fake.logged_out is True


def test_baostock_provider_reconnects_after_a_failed_membership_request() -> None:
    module = load_module()

    class Result:
        error_msg = "test response"

        def __init__(self, rows, *, error_code="0"):
            self.error_code = error_code
            self.rows = list(rows)
            self.offset = 0

        def next(self):
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0", "error_msg": "success"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            assert date == "2014-07-09"
            self.member_calls += 1
            if self.member_calls == 1:
                return Result([], error_code="10002007")
            return Result([["2014-07-07", "sz.000001", "name"]])

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    members = provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert members == ["SZ000001"]
    assert fake.member_calls == 2
    assert fake.login_calls == 2
    assert fake.logout_calls == 2


def test_baostock_provider_retries_when_query_raises_connection_error() -> None:
    module = load_module()

    class Result:
        error_code = "0"

        def __init__(self, rows):
            self.rows = list(rows)
            self.offset = 0

        def next(self):
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            assert date == "2014-07-09"
            self.member_calls += 1
            if self.member_calls == 1:
                raise ConnectionError("private transport details")
            return Result([["2014-07-07", "sz.000001", "name"]])

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    members = provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert members == ["SZ000001"]
    assert fake.member_calls == 2
    assert fake.login_calls == 2
    assert fake.logout_calls == 2


def test_baostock_provider_retries_when_error_code_changes_during_iteration() -> None:
    module = load_module()

    class Result:
        def __init__(self, rows, *, fail_during_iteration=False):
            self.error_code = "0"
            self.rows = list(rows)
            self.offset = 0
            self.fail_during_iteration = fail_during_iteration

        def next(self):
            if self.fail_during_iteration:
                self.fail_during_iteration = False
                self.error_code = "10002007"
                return False
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            assert date == "2014-07-09"
            self.member_calls += 1
            if self.member_calls == 1:
                return Result([], fail_during_iteration=True)
            return Result([["2014-07-07", "sz.000001", "name"]])

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    members = provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert members == ["SZ000001"]
    assert fake.member_calls == 2
    assert fake.login_calls == 2
    assert fake.logout_calls == 2


@pytest.mark.parametrize("failure_stage", ["next", "get_row_data"])
def test_baostock_provider_retries_transport_errors_during_iteration(
    failure_stage: str,
) -> None:
    module = load_module()

    class Result:
        error_code = "0"

        def __init__(self, rows, *, transport_failure=None):
            self.rows = list(rows)
            self.offset = 0
            self.transport_failure = transport_failure

        def next(self):
            if self.transport_failure == "next":
                raise ConnectionError("private transport details")
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            if self.transport_failure == "get_row_data":
                raise ConnectionError("private transport details")
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            assert date == "2014-07-09"
            self.member_calls += 1
            return Result(
                [["2014-07-07", "sz.000001", "name"]],
                transport_failure=failure_stage if self.member_calls == 1 else None,
            )

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    members = provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert members == ["SZ000001"]
    assert fake.member_calls == 2
    assert fake.login_calls == 2
    assert fake.logout_calls == 2


@pytest.mark.parametrize("failure_stage", ["query", "next", "get_row_data"])
def test_baostock_provider_does_not_retry_non_network_exceptions(
    failure_stage: str,
) -> None:
    module = load_module()

    class Result:
        error_code = "0"

        def __init__(self):
            self.offset = 0

        def next(self):
            if failure_stage == "next":
                raise ValueError("malformed provider data")
            self.offset += 1
            return True

        def get_row_data(self):
            if failure_stage == "get_row_data":
                raise ValueError("malformed provider data")
            return ["2014-07-07", "sz.000001", "name"]

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            assert date == "2014-07-09"
            self.member_calls += 1
            if failure_stage == "query":
                raise ValueError("malformed provider data")
            return Result()

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    with pytest.raises(ValueError, match="malformed provider data"):
        provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert fake.member_calls == 1
    assert fake.login_calls == 1
    assert fake.logout_calls == 1


def test_baostock_provider_does_not_retry_a_non_transport_error() -> None:
    module = load_module()

    class Result:
        error_code = "10001001"
        error_msg = "invalid parameter"

    class FakeBaostock:
        def __init__(self):
            self.login_calls = 0
            self.logout_calls = 0
            self.member_calls = 0

        def login(self):
            self.login_calls += 1
            return type("Login", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self):
            self.logout_calls += 1
            return type("Logout", (), {"error_code": "0", "error_msg": "success"})()

        def query_hs300_stocks(self, *, date):
            raise AssertionError("the CSI300 endpoint must not be called")

        def query_zz500_stocks(self, *, date):
            self.member_calls += 1
            return Result()

    fake = FakeBaostock()
    provider = module.BaostockProvider(fake)

    with pytest.raises(module.ProviderQueryError, match="baostock_fetch_failed"):
        provider.get_index_members(CSI500, date(2014, 7, 9))
    provider.close()

    assert fake.member_calls == 1
    assert fake.login_calls == 1
    assert fake.logout_calls == 1


def test_auto_source_falls_back_to_baostock_after_jqdata_permission_denial() -> None:
    module = load_module()

    class FakeJQData:
        def auth(self, username, password):
            assert username == "user"
            assert password == "password"

        def is_auth(self):
            return True

        def get_index_stocks(self, index_code, *, date):
            raise PermissionError("historical membership denied")

    class Result:
        error_code = "0"
        error_msg = "success"

        def __init__(self, rows):
            self.rows = list(rows)
            self.offset = 0

        def next(self):
            if self.offset >= len(self.rows):
                return False
            self.offset += 1
            return True

        def get_row_data(self):
            return self.rows[self.offset - 1]

    class FakeBaostock:
        def login(self):
            return type("Login", (), {"error_code": "0", "error_msg": "success"})()

        def logout(self):
            return None

        def query_hs300_stocks(self, *, date):
            return Result(
                [
                    ["2013-12-30", f"sh.{600000 + offset:06d}", "name"]
                    for offset in range(300)
                ]
            )

        def query_zz500_stocks(self, *, date):
            return Result(
                [
                    ["2013-12-30", f"sz.{1 + offset:06d}", "name"]
                    for offset in range(500)
                ]
            )

    provider, metadata = module._create_membership_provider(
        requested_source="auto",
        probe_date=date(2014, 1, 2),
        jq_module=FakeJQData(),
        baostock_module=FakeBaostock(),
        environ={"JQ_USERNAME": "user", "JQ_PASSWORD": "password"},
    )

    assert isinstance(provider, module.BaostockProvider)
    assert metadata == {
        "requested": "auto",
        "selected": "baostock",
        "fallback_reason": "jqdata_permission_denied",
    }
    provider.close()


def test_existing_membership_cannot_be_extended_with_a_different_source(
    tmp_path: Path,
) -> None:
    module = load_module()
    first = FakeProvider(["2025-01-02", "2025-01-03"])
    first.source_name = "jqdata"
    return_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        first,
        start_date="2025-01-02",
        end_date="2025-01-02",
    )
    assert return_code == 0
    daily_before = (output_dir / "daily" / "2025.parquet").read_bytes()
    source_before = (output_dir / "MEMBERSHIP_SOURCE.json").read_bytes()

    second = FakeProvider(["2025-01-02", "2025-01-03"])
    second.source_name = "baostock"
    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=second,
        today=date(2025, 12, 31),
    )
    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )

    assert return_code == 2
    assert status["reason_code"] == "membership_source_mismatch"
    assert second.member_calls == []
    assert (output_dir / "daily" / "2025.parquet").read_bytes() == daily_before
    assert (output_dir / "MEMBERSHIP_SOURCE.json").read_bytes() == source_before


def test_default_output_and_status_directory_follow_quant_data_root(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    monkeypatch.setattr(module, "quant_data_root", lambda: quant_data)
    provider = FakeProvider(["2025-01-02"])

    return_code = module.main(
        ["--start-date", "2025-01-02", "--end-date", "2025-01-02"],
        provider=provider,
        today=date(2025, 12, 31),
    )

    output_dir = quant_data / "reference" / "pit_csi800"
    assert return_code == 0
    assert (output_dir / "daily" / "2025.parquet").is_file()
    assert (output_dir / "membership_intervals.parquet").is_file()
    assert (output_dir / "build_status_latest.json").is_file()


def test_blocks_when_range_calendar_omits_authoritative_trading_day(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = CalendarResponseProvider(
        {
            (date(2025, 1, 2), date(2025, 1, 3)): ["2025-01-02"],
            (date(2025, 1, 1), date(2025, 1, 3)): [
                "2025-01-02",
                "2025-01-03",
            ],
        }
    )

    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "calendar_inconsistent"
    assert status["expected_date_count"] == 2
    assert provider.member_calls == []
    assert not (output_dir / "daily").exists()
    assert not (output_dir / "membership_intervals.parquet").exists()


def test_blocks_when_complete_intermediate_year_calendar_is_empty(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = CalendarResponseProvider(
        {
            (date(2023, 12, 29), date(2025, 1, 2)): [
                "2023-12-29",
                "2025-01-02",
            ],
            (date(2023, 1, 1), date(2023, 12, 31)): ["2023-12-29"],
            (date(2024, 1, 1), date(2024, 12, 31)): [],
            (date(2025, 1, 1), date(2025, 1, 2)): ["2025-01-02"],
        }
    )
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"

    return_code = module.main(
        [
            "--start-date",
            "2023-12-29",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 2),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "calendar_inconsistent"
    assert provider.member_calls == []
    assert not (output_dir / "daily").exists()
    assert not (output_dir / "membership_intervals.parquet").exists()


def test_blocks_when_constituent_count_falls_below_99_percent(tmp_path: Path) -> None:
    module = load_module()
    members = valid_members()
    members[CSI300] = members[CSI300][:-4]
    provider = FakeProvider(["2025-01-02"], members)

    return_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        provider,
        end_date="2025-01-02",
    )

    status_text = (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    status = json.loads(status_text)
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "membership_validation_failed"
    assert status["completed_date_count"] == 0
    assert status["missing_summary"]["count"] == 1
    assert status["anomaly_summary"]["items"][0]["csi300_count"] == 296
    assert not (output_dir / "daily" / "2025.parquet").exists()
    assert "Traceback" not in status_text


def test_accepts_temporary_499_member_csi500_gap_without_imputation(
    tmp_path: Path,
) -> None:
    module = load_module()
    members = valid_members()
    members[CSI500] = members[CSI500][:-1]
    provider = FakeProvider(["2019-01-07"], members)

    return_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        provider,
        start_date="2019-01-07",
        end_date="2019-01-07",
    )

    daily = pd.read_parquet(output_dir / "daily" / "2019.parquet")
    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert len(daily) == 799
    assert daily.groupby("index_code").size().to_dict() == {
        CSI300: 300,
        CSI500: 499,
    }
    assert status["status"] == "completed"
    assert status["anomaly_summary"]["count"] == 1
    assert status["anomaly_summary"]["items"][0]["reason_code"] == (
        "membership_count_below_nominal"
    )


def test_blocks_when_csi300_and_csi500_overlap(tmp_path: Path) -> None:
    module = load_module()
    members = valid_members()
    members[CSI500][0] = members[CSI300][0]
    provider = FakeProvider(["2025-01-02"], members)

    return_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        provider,
        end_date="2025-01-02",
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    anomaly = status["anomaly_summary"]["items"][0]
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "membership_validation_failed"
    assert anomaly["overlap_count"] == 1
    assert anomaly["union_count"] == 799
    assert not (output_dir / "daily" / "2025.parquet").exists()


def test_historical_permission_error_writes_sanitized_blocked_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider(
        ["2025-01-02"],
        member_error=PermissionError(
            "JQData permission denied for account private-user with password private-secret"
        ),
    )

    return_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        provider,
        end_date="2025-01-02",
    )

    status_text = (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    status = json.loads(status_text)
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "jqdata_permission_denied"
    assert status["completed_date_count"] == 0
    assert status["missing_summary"]["count"] == 1
    assert "private-user" not in status_text
    assert "private-secret" not in status_text
    assert "permission denied" not in status_text.lower()
    assert "Traceback" not in status_text
    assert not (output_dir / "build_status_latest.json").exists()


def test_skips_an_existing_complete_year_without_refetching_members(
    tmp_path: Path,
) -> None:
    module = load_module()
    first_provider = FakeProvider(["2025-01-02"])
    first_code, output_dir, status_dir = run_main(
        module,
        tmp_path,
        first_provider,
        end_date="2025-01-02",
    )
    daily_path = output_dir / "daily" / "2025.parquet"
    original_bytes = daily_path.read_bytes()
    second_provider = FakeProvider(["2025-01-02"])

    second_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=second_provider,
        today=date(2025, 12, 31),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert first_code == second_code == 0
    assert len(first_provider.member_calls) == 2
    assert second_provider.member_calls == []
    assert daily_path.read_bytes() == original_bytes
    assert status["skipped_years"] == [2025]
    assert status["completed_date_count"] == 1


def test_partial_rerun_accepts_an_existing_partition_superset(tmp_path: Path) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_path = output_dir / "daily" / "2025.parquet"
    daily_path.parent.mkdir(parents=True)
    normalized_daily_membership(["2025-01-02", "2025-01-03"]).to_parquet(
        daily_path,
        index=False,
    )
    original_bytes = daily_path.read_bytes()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert status["status"] == "completed"
    assert status["completed_date_count"] == 1
    assert status["skipped_years"] == [2025]
    assert daily_path.read_bytes() == original_bytes
    assert provider.member_calls == []
    assert provider.calendar_calls == [
        (date(2025, 1, 3), date(2025, 1, 3)),
        (date(2025, 1, 1), date(2025, 1, 3)),
    ]


def test_partial_rerun_accepts_other_authoritative_year_partitions(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True)
    path_2024 = daily_dir / "2024.parquet"
    path_2025 = daily_dir / "2025.parquet"
    normalized_daily_membership(["2024-12-30", "2024-12-31"]).to_parquet(
        path_2024,
        index=False,
    )
    normalized_daily_membership(["2025-01-02", "2025-01-03"]).to_parquet(
        path_2025,
        index=False,
    )
    original_2024 = path_2024.read_bytes()
    original_2025 = path_2025.read_bytes()
    provider = FakeProvider(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert status["status"] == "completed"
    assert path_2024.read_bytes() == original_2024
    assert path_2025.read_bytes() == original_2025
    assert provider.member_calls == []
    assert provider.calendar_calls == [
        (date(2025, 1, 3), date(2025, 1, 3)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 3)),
    ]


def test_partial_rerun_fetches_only_missing_dates_and_preserves_other_dates(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_path = output_dir / "daily" / "2025.parquet"
    daily_path.parent.mkdir(parents=True)
    existing = normalized_daily_membership(["2025-01-02", "2025-01-03"])
    existing.to_parquet(daily_path, index=False)
    provider = FakeProvider(["2025-01-02", "2025-01-03", "2025-01-06"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-06",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 6),
    )

    daily = pd.read_parquet(daily_path)
    preserved = daily.loc[daily["date"] == pd.Timestamp("2025-01-02")].reset_index(
        drop=True
    )
    original = existing.loc[existing["date"] == pd.Timestamp("2025-01-02")].reset_index(
        drop=True
    )
    assert return_code == 0
    assert sorted(daily["date"].dt.date.unique()) == [
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
    ]
    pd.testing.assert_frame_equal(preserved, original)
    assert provider.member_calls == [
        (CSI300, date(2025, 1, 6)),
        (CSI500, date(2025, 1, 6)),
    ]
    assert provider.calendar_calls == [
        (date(2025, 1, 3), date(2025, 1, 6)),
        (date(2025, 1, 1), date(2025, 1, 6)),
    ]


def test_force_refetches_an_existing_complete_year(tmp_path: Path) -> None:
    module = load_module()
    first_provider = FakeProvider(["2025-01-02"])
    _, output_dir, status_dir = run_main(
        module,
        tmp_path,
        first_provider,
        end_date="2025-01-02",
    )
    forced_provider = FakeProvider(["2025-01-02"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
            "--force",
        ],
        provider=forced_provider,
        today=date(2025, 12, 31),
    )

    assert return_code == 0
    assert len(forced_provider.member_calls) == 2


def test_later_year_failure_keeps_completed_year_checkpoint_but_not_intervals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True)
    path_2024 = daily_dir / "2024.parquet"
    path_2025 = daily_dir / "2025.parquet"
    intervals_path = output_dir / "membership_intervals.parquet"
    normalized_daily_membership(["2024-12-30"]).to_parquet(path_2024, index=False)
    normalized_daily_membership(["2025-01-02"]).to_parquet(path_2025, index=False)
    pd.DataFrame(
        {
            "symbol": ["sentinel"],
            "index_code": [CSI300],
            "start_date": [pd.Timestamp("2020-01-01")],
            "end_date": [pd.Timestamp("2020-01-01")],
        }
    ).to_parquet(intervals_path, index=False)
    original_2024 = path_2024.read_bytes()
    original_2025 = path_2025.read_bytes()
    original_intervals = intervals_path.read_bytes()
    published_statuses: list[dict] = []
    commit = module._commit_file_transaction

    def record_commit(parquet_payloads, json_payloads, *, output_dir):
        published_statuses.extend(
            payload
            for target, payload in json_payloads.items()
            if target.name == module.STATUS_FILENAME
        )
        return commit(parquet_payloads, json_payloads, output_dir=output_dir)

    monkeypatch.setattr(module, "_commit_file_transaction", record_commit)
    provider = FakeProvider(
        ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"],
        member_errors_by_date={
            "2025-01-03": PermissionError("historical membership denied")
        },
    )

    return_code = module.main(
        [
            "--start-date",
            "2024-12-31",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "jqdata_permission_denied"
    assert path_2024.read_bytes() != original_2024
    assert sorted(pd.read_parquet(path_2024)["date"].dt.date.unique()) == [
        date(2024, 12, 30),
        date(2024, 12, 31),
    ]
    assert path_2025.read_bytes() == original_2025
    assert intervals_path.read_bytes() == original_intervals
    assert status["completed_date_count"] == 2
    assert (output_dir / module.BUILD_IN_PROGRESS_FILENAME).is_file()
    assert any(
        item["status"] == "running"
        and item["completed_date_count"] == 1
        and item["expected_date_count"] == 3
        for item in published_statuses
    )


def test_concurrent_incremental_builds_do_not_lose_dates(tmp_path: Path) -> None:
    output_dir = tmp_path / "pit"
    daily_path = output_dir / "daily" / "2025.parquet"
    daily_path.parent.mkdir(parents=True)
    normalized_daily_membership(["2025-01-02"]).to_parquet(daily_path, index=False)
    context = multiprocessing.get_context("spawn")
    slow_started = context.Event()
    release_slow = context.Event()
    slow_finished = context.Event()
    fast_finished = context.Event()
    fast_started = context.Event()
    result_queue = context.Queue()
    slow_process = context.Process(
        target=run_concurrent_build,
        args=(
            str(output_dir),
            str(tmp_path / "status-slow"),
            "2025-01-03",
            "2025-01-03",
            "2025-01-03",
            slow_started,
            release_slow,
            slow_finished,
            result_queue,
        ),
    )
    fast_process = context.Process(
        target=run_concurrent_build,
        args=(
            str(output_dir),
            str(tmp_path / "status-fast"),
            "2025-01-06",
            "2025-01-06",
            None,
            fast_started,
            release_slow,
            fast_finished,
            result_queue,
        ),
    )

    try:
        slow_process.start()
        assert slow_started.wait(timeout=5)
        fast_process.start()
        lock_path = output_dir / ".build_pit_csi800.lock"
        if not lock_path.exists():
            assert fast_finished.wait(timeout=5)
        release_slow.set()
        slow_process.join(timeout=10)
        fast_process.join(timeout=10)
        assert slow_process.exitcode == 0
        assert fast_process.exitcode == 0
        assert sorted([result_queue.get(timeout=2), result_queue.get(timeout=2)]) == [
            0,
            0,
        ]
    finally:
        release_slow.set()
        for process in (slow_process, fast_process):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)

    daily = pd.read_parquet(daily_path)
    assert sorted(daily["date"].dt.date.unique()) == [
        date(2025, 1, 2),
        date(2025, 1, 3),
        date(2025, 1, 6),
    ]


def test_concurrent_requests_cannot_publish_stale_latest_status(tmp_path: Path) -> None:
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "shared-status"
    context = multiprocessing.get_context("spawn")
    old_build_returned = context.Event()
    release_old = context.Event()
    old_finished = context.Event()
    new_finished = context.Event()
    new_build_returned = context.Event()
    result_queue = context.Queue()
    old_process = context.Process(
        target=run_status_order_build,
        args=(
            str(output_dir),
            str(status_dir),
            "2025-01-02",
            True,
            old_build_returned,
            release_old,
            old_finished,
            result_queue,
        ),
    )
    new_process = context.Process(
        target=run_status_order_build,
        args=(
            str(output_dir),
            str(status_dir),
            "2025-01-03",
            False,
            new_build_returned,
            release_old,
            new_finished,
            result_queue,
        ),
    )

    try:
        old_process.start()
        assert old_build_returned.wait(timeout=5)
        new_process.start()
        assert new_finished.wait(timeout=10)
        release_old.set()
        old_process.join(timeout=10)
        new_process.join(timeout=10)
        assert old_process.exitcode == 0
        assert new_process.exitcode == 0
        assert sorted([result_queue.get(timeout=2), result_queue.get(timeout=2)]) == [
            0,
            0,
        ]
    finally:
        release_old.set()
        for process in (old_process, new_process):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert status["date_range"] == {
        "start_date": "2025-01-03",
        "end_date": "2025-01-03",
    }


def test_status_write_failure_returns_nonzero_without_committing_data(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_blocker = tmp_path / "status-is-a-file"
    status_blocker.write_text("not a directory", encoding="utf-8")
    provider = FakeProvider(["2025-01-02"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_blocker),
        ],
        provider=provider,
        today=date(2025, 1, 2),
    )

    assert return_code != 0
    assert not (output_dir / "daily" / "2025.parquet").exists()
    assert not (output_dir / "membership_intervals.parquet").exists()


def test_existing_date_below_99_percent_blocks_before_fetch_or_write(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_path = output_dir / "daily" / "2025.parquet"
    intervals_path = output_dir / "membership_intervals.parquet"
    daily_path.parent.mkdir(parents=True)
    incomplete = normalized_daily_membership(["2025-01-02"]).iloc[:-9]
    incomplete.to_parquet(daily_path, index=False)
    pd.DataFrame({"sentinel": [1]}).to_parquet(intervals_path, index=False)
    original_daily = daily_path.read_bytes()
    original_intervals = intervals_path.read_bytes()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "existing_partition_invalid"
    assert provider.member_calls == []
    assert daily_path.read_bytes() == original_daily
    assert intervals_path.read_bytes() == original_intervals


def test_blocks_without_overwriting_an_invalid_existing_year(tmp_path: Path) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_path = output_dir / "daily" / "2025.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame({"bad_column": ["do-not-overwrite"]}).to_parquet(
        daily_path, index=False
    )
    original_bytes = daily_path.read_bytes()
    provider = FakeProvider(["2025-01-02"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 12, 31),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "existing_partition_invalid"
    assert daily_path.read_bytes() == original_bytes
    assert provider.member_calls == []


def test_blocks_existing_year_with_date_outside_authoritative_calendar(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_path = output_dir / "daily" / "2025.parquet"
    daily_path.parent.mkdir(parents=True)
    normalized_daily_membership(["2025-01-02", "2025-01-03", "2025-01-04"]).to_parquet(
        daily_path, index=False
    )
    original_bytes = daily_path.read_bytes()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 12, 31),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "existing_partition_invalid"
    anomaly = status["anomaly_summary"]["items"][0]
    assert anomaly["stage"] == "existing_partition"
    assert anomaly["year"] == 2025
    assert anomaly["invalid_date_count"] == 1
    assert daily_path.read_bytes() == original_bytes
    assert provider.member_calls == []
    assert not (output_dir / "membership_intervals.parquet").exists()


def test_blocks_date_stored_in_the_wrong_year_partition(tmp_path: Path) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True)
    normalized_daily_membership(["2024-12-31"]).to_parquet(
        daily_dir / "2024.parquet",
        index=False,
    )
    wrong_year_path = daily_dir / "2025.parquet"
    normalized_daily_membership(["2024-12-31", "2025-01-02", "2025-01-03"]).to_parquet(
        wrong_year_path, index=False
    )
    original_bytes = wrong_year_path.read_bytes()
    provider = FakeProvider(["2024-12-31", "2025-01-02", "2025-01-03"])

    return_code = module.main(
        [
            "--start-date",
            "2024-12-31",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 12, 31),
    )

    status = json.loads(
        (status_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "existing_partition_invalid"
    anomaly = status["anomaly_summary"]["items"][0]
    assert anomaly["stage"] == "existing_partition"
    assert anomaly["year"] == 2025
    assert anomaly["invalid_date_count"] == 1
    assert wrong_year_path.read_bytes() == original_bytes
    assert provider.member_calls == []


def test_compresses_membership_into_trading_day_intervals() -> None:
    module = load_module()
    frame = pd.DataFrame(
        [
            ("2025-01-05", "SH600000", CSI300),
            ("2025-01-06", "SH600000", CSI300),
            ("2025-01-08", "SH600000", CSI300),
            ("2025-01-05", "SZ000001", CSI500),
            ("2025-01-06", "SZ000001", CSI500),
            ("2025-01-07", "SZ000001", CSI500),
            ("2025-01-08", "SZ000001", CSI500),
        ],
        columns=["date", "symbol", "index_code"],
    )

    intervals = module.compress_membership_intervals(
        frame,
        authoritative_dates=[
            date(2025, 1, 5),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
        ],
    )

    sh_intervals = intervals.loc[intervals["symbol"] == "SH600000"].reset_index(
        drop=True
    )
    sz_intervals = intervals.loc[intervals["symbol"] == "SZ000001"].reset_index(
        drop=True
    )
    assert list(
        sh_intervals[["start_date", "end_date"]].itertuples(index=False, name=None)
    ) == [
        (pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-06")),
        (pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-08")),
    ]
    assert list(
        sz_intervals[["start_date", "end_date"]].itertuples(index=False, name=None)
    ) == [(pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-08"))]


def test_compression_splits_across_missing_authoritative_trading_day() -> None:
    module = load_module()
    frame = pd.DataFrame(
        [
            ("2025-01-02", "SH600000", CSI300),
            ("2025-01-06", "SH600000", CSI300),
        ],
        columns=["date", "symbol", "index_code"],
    )

    intervals = module.compress_membership_intervals(
        frame,
        authoritative_dates=[
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 6),
        ],
    )

    assert list(
        intervals[["start_date", "end_date"]].itertuples(index=False, name=None)
    ) == [
        (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02")),
        (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-01-06")),
    ]


def test_build_queries_calendar_for_missing_year_between_partitions(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    status_dir = tmp_path / "status"
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True)
    normalized_daily_membership(["2023-12-29"]).to_parquet(
        daily_dir / "2023.parquet",
        index=False,
    )
    normalized_daily_membership(["2025-01-02"]).to_parquet(
        daily_dir / "2025.parquet",
        index=False,
    )
    provider = FakeProvider(["2023-12-29", "2024-01-02", "2025-01-02"])

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=provider,
        today=date(2025, 1, 2),
    )

    intervals = pd.read_parquet(output_dir / "membership_intervals.parquet")
    sample = intervals.loc[
        (intervals["symbol"] == "SH600000") & (intervals["index_code"] == CSI300)
    ]
    assert return_code == 0
    assert list(
        sample[["start_date", "end_date"]].itertuples(index=False, name=None)
    ) == [
        (pd.Timestamp("2023-12-29"), pd.Timestamp("2023-12-29")),
        (pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02")),
    ]
    assert provider.calendar_calls == [
        (date(2025, 1, 2), date(2025, 1, 2)),
        (date(2023, 1, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 2)),
    ]


def test_parquet_transaction_rolls_back_installed_files_on_commit_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    daily_path = tmp_path / "daily" / "2025.parquet"
    intervals_path = tmp_path / "membership_intervals.parquet"
    daily_path.parent.mkdir(parents=True)
    normalized_daily_membership(["2025-01-02"]).to_parquet(daily_path, index=False)
    pd.DataFrame({"sentinel": [1]}).to_parquet(intervals_path, index=False)
    original_daily = daily_path.read_bytes()
    original_intervals = intervals_path.read_bytes()
    real_replace = module.os.replace

    def fail_interval_install(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == intervals_path and source_path.name.endswith(".stage"):
            raise OSError("injected commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_interval_install)

    with pytest.raises(OSError, match="injected commit failure"):
        module._commit_parquet_transaction(
            {
                daily_path: normalized_daily_membership(["2025-01-02", "2025-01-03"]),
                intervals_path: pd.DataFrame(
                    {
                        "symbol": ["SH600000"],
                        "index_code": [CSI300],
                        "start_date": [pd.Timestamp("2025-01-02")],
                        "end_date": [pd.Timestamp("2025-01-03")],
                    }
                ),
            }
        )

    assert daily_path.read_bytes() == original_daily
    assert intervals_path.read_bytes() == original_intervals
    assert not any(
        path.name.endswith((".stage", ".backup")) for path in tmp_path.rglob("*")
    )


def test_windows_fsync_keeps_file_flush_without_directory_descriptors(tmp_path, monkeypatch):
    module = load_module()
    import quant_data_contract as contract

    path = tmp_path / "file.bin"
    path.write_bytes(b"unchanged")
    opened, flushed = [], []
    original_open, original_fsync = os.open, os.fsync

    def open_file(target, flags, *args, **kwargs):
        assert Path(target).is_file(), "Windows must not open a directory descriptor"
        assert flags & os.O_RDWR
        opened.append(Path(target))
        return original_open(target, flags, *args, **kwargs)

    def flush_file(fd):
        flushed.append(fd)
        return original_fsync(fd)

    fake_os = SimpleNamespace(
        **{**vars(os), "name": "nt", "open": open_file, "fsync": flush_file}
    )
    monkeypatch.setattr(module, "os", fake_os)
    monkeypatch.setattr(contract, "os", fake_os)
    module._fsync_file(path)
    module._fsync_directory(tmp_path)
    assert opened == [path]
    assert len(flushed) == 1
    assert path.read_bytes() == b"unchanged"


def test_file_fsync_error_is_not_suppressed(tmp_path, monkeypatch):
    module = load_module()
    path = tmp_path / "file.bin"
    path.write_bytes(b"unchanged")

    def fail_flush(_fd):
        raise OSError("file flush failed")

    monkeypatch.setattr(module, "os", SimpleNamespace(**{**vars(os), "fsync": fail_flush}))
    with pytest.raises(OSError, match="file flush failed"):
        module._fsync_file(path)
    assert path.read_bytes() == b"unchanged"


def test_membership_publish_fsyncs_before_and_after_install(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    daily_path = tmp_path / "daily" / "2025.parquet"
    source_path = tmp_path / module.SOURCE_FILENAME
    events: list[tuple[str, str]] = []
    real_fsync = module.os.fsync
    real_replace = module.os.replace

    def record_fsync(descriptor):
        events.append(("fsync", str(descriptor)))
        return real_fsync(descriptor)

    def record_replace(source, destination):
        destination_path = Path(destination)
        if destination_path.name == module.TRANSACTION_JOURNAL_FILENAME:
            events.append(("journal", destination_path.name))
        elif Path(source).name.endswith(".stage"):
            events.append(("install", destination_path.name))
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(module.os, "replace", record_replace)

    module._commit_file_transaction(
        {daily_path: normalized_daily_membership(["2025-01-02"])},
        {source_path: {"selected": "baostock"}},
        output_dir=tmp_path,
    )

    journal_events = [
        index for index, event in enumerate(events) if event[0] == "journal"
    ]
    install_events = [
        index for index, event in enumerate(events) if event[0] == "install"
    ]
    fsync_events = [index for index, event in enumerate(events) if event[0] == "fsync"]
    assert len(journal_events) == 2
    assert install_events
    assert any(index < journal_events[0] for index in fsync_events)
    assert journal_events[0] < min(install_events)
    assert any(
        max(install_events) < index < journal_events[1] for index in fsync_events
    )


def test_next_build_recovers_crashed_transaction_journal(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    daily_path = output_dir / "daily" / "2025.parquet"
    intervals_path = output_dir / "membership_intervals.parquet"
    daily_path.parent.mkdir(parents=True)
    original_daily_frame = normalized_daily_membership(["2025-01-02"])
    original_daily_frame.to_parquet(daily_path, index=False)
    original_intervals_frame = module.compress_membership_intervals(
        original_daily_frame,
        authoritative_dates=[date(2025, 1, 2)],
    )
    original_intervals_frame.to_parquet(intervals_path, index=False)
    original_daily = daily_path.read_bytes()
    original_intervals = intervals_path.read_bytes()
    replacement_daily = normalized_daily_membership(["2025-01-02", "2025-01-03"])
    replacement_intervals = module.compress_membership_intervals(
        replacement_daily,
        authoritative_dates=[date(2025, 1, 2), date(2025, 1, 3)],
    )

    class SimulatedProcessCrash(BaseException):
        pass

    real_replace = module.os.replace
    stage_install_count = 0

    def crash_after_first_install(source, destination):
        nonlocal stage_install_count
        source_path = Path(source)
        if source_path.name.endswith(".stage"):
            stage_install_count += 1
            if stage_install_count == 2:
                raise SimulatedProcessCrash("simulated process exit")
        return real_replace(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_after_first_install)
        with pytest.raises(SimulatedProcessCrash, match="simulated process exit"):
            module._commit_parquet_transaction(
                {
                    daily_path: replacement_daily,
                    intervals_path: replacement_intervals,
                }
            )

    journal_path = output_dir / module.TRANSACTION_JOURNAL_FILENAME
    assert journal_path.is_file()
    assert daily_path.read_bytes() != original_daily
    assert intervals_path.read_bytes() == original_intervals

    return_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-02",
            "--output-dir",
            str(output_dir),
        ],
        provider=FakeProvider(["2025-01-02"]),
        today=date(2025, 1, 2),
    )

    assert return_code == 0
    assert daily_path.read_bytes() == original_daily
    assert intervals_path.read_bytes() == original_intervals
    assert not journal_path.exists()


def test_transaction_recovery_is_reentrant_after_crash_during_rollback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    daily_path = output_dir / "daily" / "2025.parquet"
    intervals_path = output_dir / "membership_intervals.parquet"
    daily_path.parent.mkdir(parents=True)
    original_daily_frame = normalized_daily_membership(["2025-01-02"])
    original_daily_frame.to_parquet(daily_path, index=False)
    original_intervals_frame = module.compress_membership_intervals(
        original_daily_frame,
        authoritative_dates=[date(2025, 1, 2)],
    )
    original_intervals_frame.to_parquet(intervals_path, index=False)
    original_daily = daily_path.read_bytes()
    original_intervals = intervals_path.read_bytes()
    replacement_daily = normalized_daily_membership(["2025-01-02", "2025-01-03"])
    replacement_intervals = module.compress_membership_intervals(
        replacement_daily,
        authoritative_dates=[date(2025, 1, 2), date(2025, 1, 3)],
    )

    class SimulatedProcessCrash(BaseException):
        pass

    real_replace = module.os.replace
    stage_install_count = 0

    def crash_during_install(source, destination):
        nonlocal stage_install_count
        is_stage_install = Path(source).name.endswith(".stage")
        result = real_replace(source, destination)
        if is_stage_install:
            stage_install_count += 1
            if stage_install_count == 2:
                raise SimulatedProcessCrash("install interrupted")
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_during_install)
        with pytest.raises(SimulatedProcessCrash, match="install interrupted"):
            module._commit_parquet_transaction(
                {
                    daily_path: replacement_daily,
                    intervals_path: replacement_intervals,
                }
            )

    restored_entry_count = 0

    def crash_after_first_restore(source, destination):
        nonlocal restored_entry_count
        result = real_replace(source, destination)
        source_name = Path(source).name
        if source_name.endswith(".backup") or source_name.endswith(".rollback"):
            restored_entry_count += 1
            if restored_entry_count == 1:
                raise SimulatedProcessCrash("recovery interrupted")
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_after_first_restore)
        with pytest.raises(SimulatedProcessCrash, match="recovery interrupted"):
            module._recover_transaction(output_dir)

    journal_path = output_dir / module.TRANSACTION_JOURNAL_FILENAME
    assert journal_path.is_file()

    module._recover_transaction(output_dir)

    assert daily_path.read_bytes() == original_daily
    assert intervals_path.read_bytes() == original_intervals
    assert not journal_path.exists()


def test_recovery_rejects_crafted_internal_entry_without_deleting_target(
    tmp_path: Path,
) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    output_dir.mkdir()
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    transaction_id = "a" * 32
    journal_path = output_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.write_text(
        json.dumps(
            {
                "transaction_id": transaction_id,
                "state": "prepared",
                "entries": [
                    {
                        "target": str(unrelated),
                        "stage": str(output_dir / ".wrong.stage"),
                        "backup": str(output_dir / ".wrong.backup"),
                        "had_original": False,
                        "installed_sha256": "0" * 64,
                        "transaction_id": transaction_id,
                        "external_owner_path": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.BuildBlocked) as exc_info:
        module._recover_transaction(output_dir)

    assert exc_info.value.reason_code == "transaction_recovery_failed"
    assert unrelated.read_text(encoding="utf-8") == "keep me"
    assert journal_path.is_file()


def test_recovery_rejects_symlinked_transaction_journal(tmp_path: Path) -> None:
    module = load_module()
    output_dir = tmp_path / "pit"
    output_dir.mkdir()
    target = output_dir / "membership_intervals.parquet"
    transaction_id = "b" * 32
    stage = target.with_name(f".{target.name}.{transaction_id}.stage")
    backup = target.with_name(f".{target.name}.{transaction_id}.backup")
    external_journal = tmp_path / "external-journal.json"
    external_journal.write_text(
        json.dumps(
            {
                "version": module.TRANSACTION_JOURNAL_VERSION,
                "transaction_id": transaction_id,
                "state": "committed",
                "entries": [
                    {
                        "target": str(target),
                        "stage": str(stage),
                        "backup": str(backup),
                        "had_original": False,
                        "installed_sha256": "0" * 64,
                        "transaction_id": transaction_id,
                        "external_owner_path": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal_path = output_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.symlink_to(external_journal)

    with pytest.raises(module.BuildBlocked) as exc_info:
        module._recover_transaction(output_dir)

    assert exc_info.value.reason_code == "transaction_recovery_failed"
    assert journal_path.is_symlink()
    assert external_journal.is_file()


def test_recovery_preserves_external_status_replaced_by_later_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    output_a = tmp_path / "pit-a"
    output_b = tmp_path / "pit-b"
    status_path = tmp_path / "shared-status" / module.STATUS_FILENAME
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"transaction": "before-a"}\n', encoding="utf-8")

    class SimulatedProcessCrash(BaseException):
        pass

    real_replace = module.os.replace

    def crash_after_status_install(source, destination):
        result = real_replace(source, destination)
        if Path(source).name.endswith(".stage"):
            raise SimulatedProcessCrash("transaction A interrupted")
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_after_status_install)
        with pytest.raises(SimulatedProcessCrash, match="transaction A interrupted"):
            module._commit_file_transaction(
                {},
                {status_path: {"transaction": "a"}},
                output_dir=output_a,
            )

    journal_path = output_a / module.TRANSACTION_JOURNAL_FILENAME
    journal_payload = json.loads(journal_path.read_text(encoding="utf-8"))
    return_code = module.main(
        [
            "--start-date",
            "2025-01-03",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_b),
            "--status-out-dir",
            str(status_path.parent),
        ],
        provider=FakeProvider(["2025-01-03"]),
        today=date(2025, 1, 3),
    )
    status_b_bytes = status_path.read_bytes()

    module._recover_transaction(
        output_a,
        allowed_external_targets=(status_path,),
    )

    assert return_code == 0
    assert status_path.read_bytes() == status_b_bytes
    assert all(entry.get("installed_sha256") for entry in journal_payload["entries"])
    assert not journal_path.exists()


def test_recovery_preserves_byte_identical_external_status_owned_by_later_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    output_a = tmp_path / "pit-a"
    output_b = tmp_path / "pit-b"
    status_path = tmp_path / "shared-status" / module.STATUS_FILENAME
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"transaction": "before-a"}\n', encoding="utf-8")

    class SimulatedProcessCrash(BaseException):
        pass

    real_replace = module.os.replace

    def crash_after_status_install(source, destination):
        result = real_replace(source, destination)
        if Path(destination) == status_path:
            raise SimulatedProcessCrash("transaction A interrupted")
        return result

    payload = {"transaction": "same-bytes"}
    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_after_status_install)
        with pytest.raises(SimulatedProcessCrash, match="transaction A interrupted"):
            module._commit_file_transaction(
                {}, {status_path: payload}, output_dir=output_a
            )

    module._commit_file_transaction({}, {status_path: payload}, output_dir=output_b)
    later_bytes = status_path.read_bytes()

    module._recover_transaction(
        output_a,
        allowed_external_targets=(status_path,),
    )

    assert status_path.read_bytes() == later_bytes
    assert b"same-bytes" in later_bytes


def test_failure_status_transfers_owner_before_old_transaction_recovers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    output_a = tmp_path / "pit-a"
    status_path = tmp_path / "shared-status" / module.STATUS_FILENAME
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"transaction": "before-a"}\n', encoding="utf-8")

    class SimulatedProcessCrash(BaseException):
        pass

    real_replace = module.os.replace

    def crash_after_status_install(source, destination):
        result = real_replace(source, destination)
        if Path(destination) == status_path:
            raise SimulatedProcessCrash("transaction A interrupted")
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(module.os, "replace", crash_after_status_install)
        with pytest.raises(SimulatedProcessCrash):
            module._commit_file_transaction(
                {},
                {status_path: {"status": "completed", "transaction": "a"}},
                output_dir=output_a,
            )

    blocked = {"status": "blocked_data", "reason_code": "jqdata_permission_denied"}
    assert module._safe_write_status(blocked, status_path) is True
    blocked_bytes = status_path.read_bytes()

    module._recover_transaction(
        output_a,
        allowed_external_targets=(status_path,),
    )

    assert status_path.read_bytes() == blocked_bytes


def test_noop_build_preserves_valid_intervals_bytes_and_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])
    return_code, output_dir, status_dir = run_main(module, tmp_path, provider)
    intervals_path = output_dir / "membership_intervals.parquet"
    original_bytes = intervals_path.read_bytes()
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(intervals_path, ns=(fixed_mtime_ns, fixed_mtime_ns))

    real_compress = module.compress_membership_intervals
    compression_calls = 0

    def track_recompression(*args, **kwargs):
        nonlocal compression_calls
        compression_calls += 1
        return real_compress(*args, **kwargs)

    monkeypatch.setattr(module, "compress_membership_intervals", track_recompression)
    second_provider = FakeProvider(["2025-01-02", "2025-01-03"])
    second_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=second_provider,
        today=date(2025, 12, 31),
    )

    assert return_code == second_code == 0
    assert intervals_path.read_bytes() == original_bytes
    assert intervals_path.stat().st_mtime_ns == fixed_mtime_ns
    assert compression_calls == 1
    assert second_provider.member_calls == []


def test_noop_build_rebuilds_schema_valid_but_truncated_intervals(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider(["2025-01-02", "2025-01-03"])
    return_code, output_dir, status_dir = run_main(module, tmp_path, provider)
    intervals_path = output_dir / "membership_intervals.parquet"
    expected = pd.read_parquet(intervals_path)
    truncated = expected.iloc[:-1].copy()
    truncated.to_parquet(intervals_path, index=False)
    truncated_bytes = intervals_path.read_bytes()

    second_code = module.main(
        [
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
            "--output-dir",
            str(output_dir),
            "--status-out-dir",
            str(status_dir),
        ],
        provider=FakeProvider(["2025-01-02", "2025-01-03"]),
        today=date(2025, 12, 31),
    )

    assert return_code == second_code == 0
    assert intervals_path.read_bytes() != truncated_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(intervals_path), expected)


def test_rejects_future_dates_before_calling_provider(tmp_path: Path) -> None:
    module = load_module()
    today = date(2025, 1, 2)
    future = today + timedelta(days=1)
    provider = FakeProvider([])
    output_dir = tmp_path / "pit"

    return_code = module.main(
        [
            "--start-date",
            future.isoformat(),
            "--end-date",
            future.isoformat(),
            "--output-dir",
            str(output_dir),
        ],
        provider=provider,
        today=today,
    )

    status = json.loads(
        (output_dir / "build_status_latest.json").read_text(encoding="utf-8")
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "future_date_not_allowed"
    assert status["completed_date_count"] == 0
    assert provider.calendar_calls == []
    assert provider.member_calls == []
    assert not (output_dir / "daily").exists()
