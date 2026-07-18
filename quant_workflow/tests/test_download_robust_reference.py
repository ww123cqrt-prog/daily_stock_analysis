from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import threading
import tracemalloc
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_robust_reference.py"
)
TABLE_NAMES = (
    "security_master.parquet",
    "industry_intervals.parquet",
    "st_status.parquet",
    "suspension_status.parquet",
)
SOURCE_FILENAME = "REFERENCE_SOURCE.json"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "download_robust_reference", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeProvider:
    def __init__(self) -> None:
        self.security = {
            "600000.XSHG": SimpleNamespace(start_date=date(2000, 1, 1), end_date=None),
            "300001.XSHE": SimpleNamespace(start_date=date(2010, 1, 1), end_date=None),
            "430001.XBSE": SimpleNamespace(start_date=date(2012, 1, 1), end_date=None),
        }
        self.industries = {
            ("2025-01-02", "600000.XSHG"): "801010",
            ("2025-01-02", "300001.XSHE"): "801020",
            ("2025-01-02", "430001.XBSE"): "801030",
            ("2025-01-03", "600000.XSHG"): "801040",
            ("2025-01-03", "300001.XSHE"): "801020",
        }
        self.st = {
            ("2025-01-02", "600000.XSHG"): False,
            ("2025-01-02", "300001.XSHE"): True,
            ("2025-01-02", "430001.XBSE"): False,
            ("2025-01-03", "600000.XSHG"): False,
            ("2025-01-03", "300001.XSHE"): False,
        }
        self.paused = {
            ("2025-01-02", "600000.XSHG"): 0,
            ("2025-01-02", "300001.XSHE"): 0,
            ("2025-01-02", "430001.XBSE"): 1,
            ("2025-01-03", "600000.XSHG"): 0,
            ("2025-01-03", "300001.XSHE"): 0,
        }
        self.calls: list[tuple[object, ...]] = []
        self.error: Exception | None = None
        self.industry_error: Exception | None = None
        self.industry_errors: list[Exception | None] = []

    def get_security_info(self, code: str):
        self.calls.append(("security", code))
        if self.error is not None:
            raise self.error
        return self.security.get(code)

    def get_industry(self, codes: list[str], date: str):
        self.calls.append(("industry", tuple(codes), date))
        if self.industry_errors:
            error = self.industry_errors.pop(0)
            if error is not None:
                raise error
        if self.industry_error is not None:
            raise self.industry_error
        if self.error is not None:
            raise self.error
        return {
            code: {"sw_l1": {"industry_code": self.industries[(date, code)]}}
            for code in codes
            if (date, code) in self.industries
        }

    def get_extras(
        self,
        name: str,
        codes: list[str],
        start_date: str,
        end_date: str,
        df: bool = True,
    ):
        assert name == "is_st"
        assert df is True
        self.calls.append(("st", tuple(codes), start_date, end_date))
        if self.error is not None:
            raise self.error
        days = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame(
            {
                code: [
                    self.st.get((day.date().isoformat(), code), pd.NA) for day in days
                ]
                for code in codes
            },
            index=days,
        )

    def get_price(self, codes: list[str], **kwargs):
        assert kwargs["fields"] == ["paused"]
        assert kwargs["skip_paused"] is False
        assert kwargs["panel"] is False
        self.calls.append(
            ("paused", tuple(codes), kwargs["start_date"], kwargs["end_date"])
        )
        if self.error is not None:
            raise self.error
        rows = []
        for day in pd.date_range(kwargs["start_date"], kwargs["end_date"], freq="D"):
            for code in codes:
                key = (day.date().isoformat(), code)
                if key in self.paused:
                    rows.append({"time": day, "code": code, "paused": self.paused[key]})
        return pd.DataFrame(rows, columns=["time", "code", "paused"])


class FakeBaostockResponse:
    def __init__(
        self,
        fields: list[str] | tuple[str, ...] = (),
        rows: list[list[str]] | None = None,
        *,
        error_code: str = "0",
        error_msg: str = "",
    ) -> None:
        self.fields = list(fields)
        self.rows = [] if rows is None else rows
        self.error_code = error_code
        self.error_msg = error_msg
        self._offset = 0

    def next(self) -> bool:
        return self._offset < len(self.rows)

    def get_row_data(self) -> list[str]:
        row = self.rows[self._offset]
        self._offset += 1
        return row


class FakeBaostockProvider:
    basic_fields = ("code", "code_name", "ipoDate", "outDate", "type", "status")
    industry_fields = (
        "updateDate",
        "code",
        "code_name",
        "industry",
        "industryClassification",
    )
    history_fields = ("date", "tradestatus", "isST")

    def __init__(self) -> None:
        self.basic = [
            ["sh.600000", "PFB", "2000-01-01", "", "1", "1"],
            ["sz.300001", "TEST-GROWTH", "2010-01-01", "", "1", "1"],
            ["bj.430001", "TEST-BJ", "2012-01-01", "", "1", "1"],
        ]
        self.industries = {
            "2025-01-02": [
                ["2025-01-02", "sh.600000", "PFB", "Bank", "CSRC"],
                ["2025-01-02", "sz.300001", "TEST-GROWTH", "Software", "CSRC"],
                ["2025-01-02", "bj.430001", "TEST-BJ", "Machinery", "CSRC"],
            ],
            "2025-01-03": [
                ["2025-01-03", "sh.600000", "PFB", "Finance", "CSRC"],
                ["2025-01-03", "sz.300001", "TEST-GROWTH", "Software", "CSRC"],
            ],
        }
        self.history = {
            "sh.600000": [
                ["2025-01-02", "1", "0"],
                ["2025-01-03", "1", "0"],
            ],
            "sz.300001": [
                ["2025-01-02", "1", "1"],
                ["2025-01-03", "1", "0"],
            ],
            "bj.430001": [["2025-01-02", "0", "0"]],
        }
        self.failures: dict[str, list[tuple[str, str]]] = {}
        self.exceptions: dict[str, list[Exception]] = {}
        self.calls: list[tuple[object, ...]] = []
        self.login_count = 0
        self.logout_count = 0

    def _response(
        self, name: str, fields: tuple[str, ...], rows: list[list[str]]
    ) -> FakeBaostockResponse:
        exceptions = self.exceptions.get(name, [])
        if exceptions:
            raise exceptions.pop(0)
        failures = self.failures.get(name, [])
        if failures:
            error_code, error_msg = failures.pop(0)
            return FakeBaostockResponse(error_code=error_code, error_msg=error_msg)
        return FakeBaostockResponse(fields, rows)

    def login(self):
        self.login_count += 1
        self.calls.append(("login",))
        return self._response("login", (), [])

    def logout(self):
        self.logout_count += 1
        self.calls.append(("logout",))
        return FakeBaostockResponse()

    def query_stock_basic(self, code: str = "", code_name: str = ""):
        self.calls.append(("basic", code, code_name))
        return self._response("basic", self.basic_fields, self.basic)

    def query_stock_industry(self, code: str = "", date: str = ""):
        self.calls.append(("industry", code, date))
        return self._response(
            "industry", self.industry_fields, self.industries.get(date, [])
        )

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ):
        self.calls.append(
            ("history", code, fields, start_date, end_date, frequency, adjustflag)
        )
        return self._response(
            "history", self.history_fields, self.history.get(code, [])
        )


class LargeBaostockProvider:
    industry_fields = FakeBaostockProvider.industry_fields
    history_fields = FakeBaostockProvider.history_fields

    def __init__(self, module, dates: list[str], symbols: list[str]) -> None:
        self.module = module
        self.dates = dates
        self.symbols = symbols

    def query_stock_industry(self, code: str = "", date: str = ""):
        rows = [
            [
                date,
                self.module.qlib_to_baostock_symbol(symbol),
                symbol,
                "Industry",
                "CSRC",
            ]
            for symbol in self.symbols
        ]
        return FakeBaostockResponse(self.industry_fields, rows)

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
    ):
        rows = [
            [
                day,
                "0" if offset % 17 == 0 else "1",
                "1" if offset % 19 == 0 else "0",
            ]
            for offset, day in enumerate(self.dates)
        ]
        return FakeBaostockResponse(self.history_fields, rows)


def write_pit(pit_dir: Path, rows: list[tuple[str, str]]) -> None:
    daily = pit_dir / "daily"
    daily.mkdir(parents=True)
    membership = pd.DataFrame(rows, columns=["date", "symbol"])
    membership.assign(index_code="000906.XSHG").to_parquet(
        daily / "2025.parquet",
        index=False,
    )
    membership.to_parquet(pit_dir / "membership_intervals.parquet", index=False)
    dates = sorted(row[0] for row in rows)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "jqdata",
                "selected": "jqdata",
                "date_range": {
                    "start_date": dates[0],
                    "end_date": dates[-1],
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        )
    )


def valid_rows() -> list[tuple[str, str]]:
    return [
        ("2025-01-02", "SH600000"),
        ("2025-01-02", "SZ300001"),
        ("2025-01-02", "BJ430001"),
        ("2025-01-03", "SH600000"),
        ("2025-01-03", "SZ300001"),
    ]


def run_main(
    module,
    tmp_path: Path,
    provider: FakeProvider | None = None,
    *,
    baostock_provider: FakeBaostockProvider | None = None,
    reference_source: str | None = None,
    today: date = date(2025, 1, 3),
):
    pit_dir = tmp_path / "pit"
    reference_dir = tmp_path / "reference"
    status_dir = tmp_path / "status"
    if not (pit_dir / "daily").exists():
        write_pit(pit_dir, valid_rows())
    argv = [
        "--pit-dir",
        str(pit_dir),
        "--reference-dir",
        str(reference_dir),
        "--status-out-dir",
        str(status_dir),
        "--start-date",
        "2025-01-02",
        "--end-date",
        "2025-01-03",
    ]
    if reference_source is not None:
        argv.extend(["--reference-source", reference_source])
    main_kwargs = {"provider": provider, "today": today}
    if baostock_provider is not None:
        main_kwargs["baostock_provider"] = baostock_provider
    return_code = module.main(argv, **main_kwargs)
    return return_code, reference_dir, status_dir


def seed_old_tables(reference_dir: Path) -> dict[str, bytes]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    payloads = {name: f"old:{name}".encode() for name in TABLE_NAMES}
    for name, payload in payloads.items():
        (reference_dir / name).write_bytes(payload)
    return payloads


def seed_old_source(reference_dir: Path) -> bytes:
    reference_dir.mkdir(parents=True, exist_ok=True)
    payload = b'{"selected":"old-provider"}\n'
    (reference_dir / SOURCE_FILENAME).write_bytes(payload)
    return payload


def assert_old_tables_unchanged(reference_dir: Path, old: dict[str, bytes]) -> None:
    assert {name: (reference_dir / name).read_bytes() for name in TABLE_NAMES} == old


def test_builds_all_tables_and_compresses_industry_changes_and_membership_breaks(
    tmp_path: Path,
) -> None:
    module = load_module()

    return_code, reference_dir, status_dir = run_main(module, tmp_path, FakeProvider())

    security = pd.read_parquet(reference_dir / "security_master.parquet")
    industry = pd.read_parquet(reference_dir / "industry_intervals.parquet")
    st = pd.read_parquet(reference_dir / "st_status.parquet")
    suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    source = json.loads((reference_dir / SOURCE_FILENAME).read_text())
    with module.membership_read_lock(tmp_path / "pit") as snapshot:
        expected_membership_hash = snapshot.fingerprint()
    assert return_code == 0
    assert security.to_dict("records") == [
        {
            "symbol": "BJ430001",
            "list_date": pd.Timestamp("2012-01-01"),
            "delist_date": pd.NaT,
            "board": "other",
        },
        {
            "symbol": "SH600000",
            "list_date": pd.Timestamp("2000-01-01"),
            "delist_date": pd.NaT,
            "board": "main",
        },
        {
            "symbol": "SZ300001",
            "list_date": pd.Timestamp("2010-01-01"),
            "delist_date": pd.NaT,
            "board": "growth",
        },
    ]
    assert industry.to_dict("records") == [
        {
            "symbol": "BJ430001",
            "industry_code": "801030",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-02"),
        },
        {
            "symbol": "SH600000",
            "industry_code": "801010",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-02"),
        },
        {
            "symbol": "SH600000",
            "industry_code": "801040",
            "start_date": pd.Timestamp("2025-01-03"),
            "end_date": pd.Timestamp("2025-01-03"),
        },
        {
            "symbol": "SZ300001",
            "industry_code": "801020",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-03"),
        },
    ]
    assert len(st) == len(suspension) == len(valid_rows())
    assert st.loc[st["symbol"].eq("SZ300001"), "is_st"].tolist() == [True, False]
    assert suspension.loc[
        suspension["symbol"].eq("BJ430001"), "is_suspended"
    ].tolist() == [True]
    assert status["status"] == "completed"
    assert status["reason_code"] == "completed"
    assert status["member_day_count"] == 5
    assert status["date_range"] == {
        "start_date": "2025-01-02",
        "end_date": "2025-01-03",
    }
    assert status["requested"] == "auto"
    assert status["selected"] == "jqdata"
    assert status["industry_classification"] == "sw_l1"
    assert status["membership_snapshot_hash"] == expected_membership_hash
    assert len(expected_membership_hash) == 64
    assert source == {
        "date_range": {
            "start_date": "2025-01-02",
            "end_date": "2025-01-03",
        },
        "industry_classification": "sw_l1",
        "membership_snapshot_hash": expected_membership_hash,
        "requested": "auto",
        "selected": "jqdata",
    }


def test_jqdata_keeps_post_delist_index_days_and_marks_them_untradable(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider()
    provider.security["600000.XSHG"] = SimpleNamespace(
        start_date=date(2000, 1, 1),
        end_date=date(2025, 1, 2),
    )
    del provider.st[("2025-01-03", "600000.XSHG")]
    del provider.paused[("2025-01-03", "600000.XSHG")]
    provider.st[("2025-01-02", "600000.XSHG")] = True
    provider.paused[("2025-01-02", "600000.XSHG")] = True

    return_code, reference_dir, _ = run_main(module, tmp_path, provider)

    security = pd.read_parquet(reference_dir / "security_master.parquet")
    st = pd.read_parquet(reference_dir / "st_status.parquet")
    suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
    security_row = security.loc[security["symbol"] == "SH600000"].iloc[0]
    post_delist_st = st.loc[
        (st["symbol"] == "SH600000")
        & (st["date"] == pd.Timestamp("2025-01-03")),
        "is_st",
    ]
    post_delist_suspension = suspension.loc[
        (suspension["symbol"] == "SH600000")
        & (suspension["date"] == pd.Timestamp("2025-01-03")),
        "is_suspended",
    ]

    assert return_code == 0
    assert security_row["delist_date"] == pd.Timestamp("2025-01-02")
    assert post_delist_st.tolist() == [True]
    assert post_delist_suspension.tolist() == [True]


def test_cli_reference_source_defaults_to_auto_and_rejects_unknown_value() -> None:
    module = load_module()
    parser = module._argument_parser()

    assert parser.parse_args([]).reference_source == "auto"
    for source in ("auto", "jqdata", "baostock"):
        assert (
            parser.parse_args(["--reference-source", source]).reference_source == source
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--reference-source", "mixed"])


def test_auto_falls_back_as_a_whole_when_jqdata_industry_probe_is_denied(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    jq_provider = FakeProvider()
    secret = "jq-user@example.com:permission-detail"
    jq_provider.industry_error = PermissionError(f"no privilege: {secret}")
    baostock_provider = FakeBaostockProvider()

    return_code, reference_dir, status_dir = run_main(
        module,
        tmp_path,
        jq_provider,
        baostock_provider=baostock_provider,
    )

    captured = capsys.readouterr()
    source_text = (reference_dir / SOURCE_FILENAME).read_text()
    status_text = (status_dir / "download_robust_reference_latest.json").read_text()
    source = json.loads(source_text)
    status = json.loads(status_text)
    assert return_code == 0
    assert [call[0] for call in jq_provider.calls] == ["security", "industry"]
    assert baostock_provider.login_count == 1
    assert baostock_provider.logout_count == 1
    assert source["requested"] == status["requested"] == "auto"
    assert source["selected"] == status["selected"] == "baostock"
    assert (
        source["fallback_reason"]
        == status["fallback_reason"]
        == ("jqdata_permission_denied")
    )
    assert source["industry_classification"] == "CSRC"
    assert secret not in source_text + status_text + captured.out + captured.err


def test_auto_discards_jqdata_build_when_permission_fails_after_probe(
    tmp_path: Path,
) -> None:
    module = load_module()
    jq_provider = FakeProvider()
    jq_provider.industry_errors = [
        None,
        PermissionError("full build has no industry privilege"),
    ]
    baostock_provider = FakeBaostockProvider()

    return_code, reference_dir, status_dir = run_main(
        module,
        tmp_path,
        jq_provider,
        baostock_provider=baostock_provider,
    )

    source = json.loads((reference_dir / SOURCE_FILENAME).read_text())
    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    security = pd.read_parquet(reference_dir / "security_master.parquet")
    assert return_code == 0
    assert [call[0] for call in jq_provider.calls].count("industry") == 2
    assert source["selected"] == status["selected"] == "baostock"
    assert source["fallback_reason"] == "jqdata_permission_denied"
    assert security["symbol"].tolist() == ["BJ430001", "SH600000", "SZ300001"]
    assert baostock_provider.login_count == baostock_provider.logout_count == 1


def test_baostock_builds_all_four_tables_from_historical_pit_data(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeBaostockProvider()

    return_code, reference_dir, status_dir = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    security = pd.read_parquet(reference_dir / "security_master.parquet")
    industry = pd.read_parquet(reference_dir / "industry_intervals.parquet")
    st = pd.read_parquet(reference_dir / "st_status.parquet")
    suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
    source = json.loads((reference_dir / SOURCE_FILENAME).read_text())
    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code == 0
    assert security["symbol"].tolist() == ["BJ430001", "SH600000", "SZ300001"]
    assert industry.to_dict("records") == [
        {
            "symbol": "BJ430001",
            "industry_code": "csrc:machinery",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-02"),
        },
        {
            "symbol": "SH600000",
            "industry_code": "csrc:bank",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-02"),
        },
        {
            "symbol": "SH600000",
            "industry_code": "csrc:finance",
            "start_date": pd.Timestamp("2025-01-03"),
            "end_date": pd.Timestamp("2025-01-03"),
        },
        {
            "symbol": "SZ300001",
            "industry_code": "csrc:software",
            "start_date": pd.Timestamp("2025-01-02"),
            "end_date": pd.Timestamp("2025-01-03"),
        },
    ]
    assert st["is_st"].tolist() == [False, False, True, False, False]
    assert suspension["is_suspended"].tolist() == [True, False, False, False, False]
    assert source == {
        "date_range": {
            "start_date": "2025-01-02",
            "end_date": "2025-01-03",
        },
        "industry_classification": "CSRC",
        "membership_snapshot_hash": status["membership_snapshot_hash"],
        "requested": "baostock",
        "selected": "baostock",
    }
    assert len(source["membership_snapshot_hash"]) == 64
    assert status["selected"] == "baostock"
    assert provider.login_count == provider.logout_count == 1
    assert ("industry", "", "2025-01-02") in provider.calls
    assert ("industry", "", "2025-01-03") in provider.calls
    history_calls = [call for call in provider.calls if call[0] == "history"]
    assert len(history_calls) == 3
    assert all(
        call[2:]
        == (
            "date,tradestatus,isST",
            "2025-01-02",
            "2025-01-03",
            "d",
            "3",
        )
        for call in history_calls
    )


def test_baostock_nonzero_response_relogs_and_retries_once(tmp_path: Path) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.failures["basic"] = [("10002007", "temporary socket detail")]

    return_code, reference_dir, _ = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    assert return_code == 0
    assert (reference_dir / SOURCE_FILENAME).is_file()
    assert [call[0] for call in provider.calls].count("basic") == 2
    assert provider.login_count == 2
    assert provider.logout_count == 2


def test_baostock_business_error_does_not_relogin_or_retry(tmp_path: Path) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.failures["basic"] = [("10001001", "invalid business parameter")]

    return_code, reference_dir, status_dir = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "baostock_query_failed"
    assert [call[0] for call in provider.calls].count("basic") == 1
    assert provider.login_count == provider.logout_count == 1
    assert not (reference_dir / SOURCE_FILENAME).exists()


@pytest.mark.parametrize(
    ("update_date", "reason_code"),
    [
        ("2025-01-03", "future_industry_update_date"),
        ("not-a-date", "invalid_industry_update_date"),
    ],
)
def test_baostock_industry_update_date_must_be_valid_as_of_requested_day(
    tmp_path: Path, update_date: str, reason_code: str
) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.industries["2025-01-02"][0][0] = update_date

    return_code, reference_dir, status_dir = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == reason_code
    assert not (reference_dir / SOURCE_FILENAME).exists()


def test_baostock_keeps_post_delist_index_days_and_marks_them_untradable(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.basic[0] = [
        "sh.600000",
        "PFB",
        "2000-01-01",
        "2025-01-02",
        "1",
        "0",
    ]
    provider.history["sh.600000"] = [["2025-01-02", "0", "1"]]

    return_code, reference_dir, _ = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    security = pd.read_parquet(reference_dir / "security_master.parquet")
    st = pd.read_parquet(reference_dir / "st_status.parquet")
    suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
    security_row = security.loc[security["symbol"] == "SH600000"].iloc[0]
    post_delist_st = st.loc[
        (st["symbol"] == "SH600000")
        & (st["date"] == pd.Timestamp("2025-01-03")),
        "is_st",
    ]
    post_delist_suspension = suspension.loc[
        (suspension["symbol"] == "SH600000")
        & (suspension["date"] == pd.Timestamp("2025-01-03")),
        "is_suspended",
    ]

    assert return_code == 0
    assert security_row["delist_date"] == pd.Timestamp("2025-01-02")
    assert post_delist_st.tolist() == [True]
    assert post_delist_suspension.tolist() == [True]


def test_baostock_marks_missing_delist_day_untradable(tmp_path: Path) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.basic[0] = [
        "sh.600000",
        "PFB",
        "2000-01-01",
        "2025-01-03",
        "1",
        "0",
    ]
    provider.history["sh.600000"] = [["2025-01-02", "0", "1"]]

    return_code, reference_dir, _ = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    st = pd.read_parquet(reference_dir / "st_status.parquet")
    suspension = pd.read_parquet(reference_dir / "suspension_status.parquet")
    delist_day_st = st.loc[
        (st["symbol"] == "SH600000")
        & (st["date"] == pd.Timestamp("2025-01-03")),
        "is_st",
    ]
    delist_day_suspension = suspension.loc[
        (suspension["symbol"] == "SH600000")
        & (suspension["date"] == pd.Timestamp("2025-01-03")),
        "is_suspended",
    ]

    assert return_code == 0
    assert delist_day_st.tolist() == [True]
    assert delist_day_suspension.tolist() == [True]


def test_baostock_100k_member_days_stay_memory_bounded_and_exact() -> None:
    module = load_module()
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    symbols = [f"SH{600000 + offset:06d}" for offset in range(200)]
    membership = pd.MultiIndex.from_product(
        [dates, symbols], names=["date", "symbol"]
    ).to_frame(index=False)
    provider = LargeBaostockProvider(
        module,
        [timestamp.date().isoformat() for timestamp in dates],
        symbols,
    )
    adapter = module.BaostockAdapter(provider)

    tracemalloc.start()
    daily_industry = adapter._fetch_daily_industries(membership)
    st, suspension = adapter._fetch_status_tables(membership)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_bytes < 50 * 1024 * 1024
    pd.testing.assert_frame_equal(daily_industry.loc[:, ["date", "symbol"]], membership)
    assert daily_industry["industry_code"].eq("csrc:industry").all()
    expected_st = np.repeat(np.arange(len(dates)) % 19 == 0, len(symbols))
    expected_suspended = np.repeat(np.arange(len(dates)) % 17 == 0, len(symbols))
    assert np.array_equal(st["is_st"].to_numpy(), expected_st)
    assert np.array_equal(suspension["is_suspended"].to_numpy(), expected_suspended)
    assert len(st) == len(suspension) == len(membership) == 100_000


def test_baostock_reuses_valid_industry_partition_cache(tmp_path: Path) -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    cache_dir = tmp_path / "cache"
    first_provider = FakeBaostockProvider()
    first_adapter = module.BaostockAdapter(first_provider, cache_dir=cache_dir)

    expected = first_adapter._fetch_daily_industries(membership)

    second_provider = FakeBaostockProvider()
    second_provider.exceptions["industry"] = [
        RuntimeError("cache should avoid provider access")
    ]
    second_adapter = module.BaostockAdapter(second_provider, cache_dir=cache_dir)
    actual = second_adapter._fetch_daily_industries(membership)

    pd.testing.assert_frame_equal(actual, expected)
    assert not [call for call in second_provider.calls if call[0] == "industry"]
    assert second_adapter.industry_classification == "CSRC"


def test_baostock_realigns_reordered_industry_partition_cache(
    tmp_path: Path,
) -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "symbol": ["SH600000", "SZ000001"],
        }
    )
    cache_dir = tmp_path / "cache"
    adapter = module.BaostockAdapter(FakeBaostockProvider(), cache_dir=cache_dir)
    cache_path = cache_dir / "industry" / "2025.parquet"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "symbol": ["SZ000001", "SH600000"],
            "industry_code": ["CSRC:Bank", "CSRC:Technology"],
            "industry_classification": ["CSRC", "CSRC"],
        }
    ).to_parquet(cache_path, index=False)

    actual = adapter._fetch_daily_industries(membership)

    assert actual[["date", "symbol"]].equals(membership)
    assert actual["industry_code"].tolist() == ["CSRC:Technology", "CSRC:Bank"]


def test_baostock_rejects_empty_industry_code_in_cache(tmp_path: Path) -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "symbol": ["SH600000"],
        }
    )
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "industry" / "2025.parquet"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": membership["date"],
            "symbol": membership["symbol"],
            "industry_code": [""],
            "industry_classification": ["CSRC"],
        }
    ).to_parquet(cache_path, index=False)
    adapter = module.BaostockAdapter(FakeBaostockProvider(), cache_dir=cache_dir)

    with pytest.raises(module.BlockedData, match="invalid_reference_cache"):
        adapter._fetch_daily_industries(membership)


def test_baostock_reuses_valid_symbol_status_cache(tmp_path: Path) -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    security = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "list_date": pd.to_datetime(["2000-01-01"]),
            "delist_date": [pd.NaT],
            "board": ["main"],
        }
    )
    cache_dir = tmp_path / "cache"
    first_provider = FakeBaostockProvider()
    first_adapter = module.BaostockAdapter(first_provider, cache_dir=cache_dir)

    expected_st, expected_suspension = first_adapter._fetch_status_tables(
        membership, security
    )

    second_provider = FakeBaostockProvider()
    second_provider.exceptions["history"] = [
        RuntimeError("cache should avoid provider access")
    ]
    second_adapter = module.BaostockAdapter(second_provider, cache_dir=cache_dir)
    actual_st, actual_suspension = second_adapter._fetch_status_tables(
        membership, security
    )

    pd.testing.assert_frame_equal(actual_st, expected_st)
    pd.testing.assert_frame_equal(actual_suspension, expected_suspension)
    assert not [call for call in second_provider.calls if call[0] == "history"]


def test_baostock_rejects_status_cache_for_different_symbol(
    tmp_path: Path,
) -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    security = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "list_date": pd.to_datetime(["2000-01-01"]),
            "delist_date": [pd.NaT],
            "board": ["main"],
        }
    )
    cache_dir = tmp_path / "cache"
    cache_path = cache_dir / "status" / "SH600000.parquet"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["SZ000001", "SZ000001"],
            "date": membership["date"],
            "is_st": [False, True],
            "is_suspended": [False, False],
        }
    ).to_parquet(cache_path, index=False)
    adapter = module.BaostockAdapter(FakeBaostockProvider(), cache_dir=cache_dir)

    with pytest.raises(module.BlockedData, match="invalid_reference_cache"):
        adapter._fetch_status_tables(membership, security)


def test_baostock_missing_member_day_fails_closed_and_preserves_old_source(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    provider.history["sh.600000"] = [["2025-01-02", "1", "0"]]
    reference_dir = tmp_path / "reference"
    old_tables = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)

    return_code, _, status_dir = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "missing_st_member_day"
    assert status["requested"] == status["selected"] == "baostock"
    assert_old_tables_unchanged(reference_dir, old_tables)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source
    assert provider.logout_count == 1


@pytest.mark.parametrize("missing_kind", ["industry", "st", "paused"])
def test_missing_member_day_blocks_and_preserves_all_old_tables(
    tmp_path: Path, missing_kind: str
) -> None:
    module = load_module()
    provider = FakeProvider()
    if missing_kind == "industry":
        del provider.industries[("2025-01-03", "600000.XSHG")]
    elif missing_kind == "st":
        del provider.st[("2025-01-03", "600000.XSHG")]
    else:
        del provider.paused[("2025-01-03", "600000.XSHG")]
    reference_dir = tmp_path / "reference"
    old = seed_old_tables(reference_dir)

    return_code, _, status_dir = run_main(module, tmp_path, provider)

    status_text = (status_dir / "download_robust_reference_latest.json").read_text()
    status = json.loads(status_text)
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == f"missing_{missing_kind}_member_day"
    assert "600000.XSHG" not in status_text
    assert_old_tables_unchanged(reference_dir, old)


def test_missing_security_info_blocks_and_preserves_all_old_tables(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider()
    provider.security["600000.XSHG"] = None
    reference_dir = tmp_path / "reference"
    old = seed_old_tables(reference_dir)

    return_code, _, status_dir = run_main(module, tmp_path, provider)

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "missing_security_master_member"
    assert_old_tables_unchanged(reference_dir, old)


def test_provider_permission_error_is_sanitized(tmp_path: Path, capsys) -> None:
    module = load_module()
    provider = FakeProvider()
    secret = "user@example.com:password-should-never-leak"
    provider.error = PermissionError(f"no privilege for {secret}")

    return_code, reference_dir, status_dir = run_main(module, tmp_path, provider)

    captured = capsys.readouterr()
    status_text = (status_dir / "download_robust_reference_latest.json").read_text()
    assert return_code != 0
    assert json.loads(status_text)["reason_code"] == "jqdata_permission_denied"
    assert secret not in status_text + captured.out + captured.err
    assert not reference_dir.exists()


def test_membership_writer_active_blocks_before_parquet_or_provider(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    write_pit(pit_dir, valid_rows())
    provider = FakeProvider()
    read_calls = 0
    real_read_parquet = module.pd.read_parquet

    def record_read_parquet(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(module.pd, "read_parquet", record_read_parquet)
    lock_handle = (pit_dir / ".build_pit_csi800.lock").open("a+b")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    try:
        return_code, _, status_dir = run_main(module, tmp_path, provider)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "membership_writer_active"
    assert status["membership_snapshot_hash"] is None
    assert read_calls == 0
    assert provider.calls == []


def test_membership_journal_blocks_before_parquet_or_provider(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    write_pit(pit_dir, valid_rows())
    (pit_dir / ".build_pit_csi800.transaction.json").write_text("{}")
    provider = FakeProvider()
    read_calls = 0
    real_read_parquet = module.pd.read_parquet

    def record_read_parquet(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return real_read_parquet(*args, **kwargs)

    monkeypatch.setattr(module.pd, "read_parquet", record_read_parquet)

    return_code, _, status_dir = run_main(module, tmp_path, provider)

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "membership_transaction_incomplete"
    assert status["membership_snapshot_hash"] is None
    assert read_calls == 0
    assert provider.calls == []


def test_membership_source_must_cover_requested_range_before_provider(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    write_pit(pit_dir, valid_rows())
    source_path = pit_dir / "MEMBERSHIP_SOURCE.json"
    source = json.loads(source_path.read_text())
    source["date_range"]["end_date"] = "2025-01-02"
    source_path.write_text(json.dumps(source))
    provider = FakeProvider()

    return_code, _, status_dir = run_main(module, tmp_path, provider)

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "membership_source_range_incomplete"
    assert status["membership_snapshot_hash"] is None
    assert provider.calls == []


def test_membership_shared_lock_covers_read_but_not_provider_network(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    write_pit(pit_dir, valid_rows())
    lock_path = pit_dir / ".build_pit_csi800.lock"
    observed = {"read_locked": False, "provider_unlocked": False}
    real_read_parquet = module.pd.read_parquet

    def exclusive_lock_available() -> bool:
        handle = lock_path.open("a+b")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return True
        finally:
            handle.close()

    def record_read_parquet(*args, **kwargs):
        observed["read_locked"] = not exclusive_lock_available()
        return real_read_parquet(*args, **kwargs)

    class LockCheckingProvider(FakeProvider):
        def get_security_info(self, code: str):
            observed["provider_unlocked"] = exclusive_lock_available()
            return super().get_security_info(code)

    monkeypatch.setattr(module.pd, "read_parquet", record_read_parquet)

    return_code, _, _ = run_main(
        module,
        tmp_path,
        LockCheckingProvider(),
        reference_source="jqdata",
    )

    assert return_code == 0
    assert observed == {"read_locked": True, "provider_unlocked": True}


def test_publish_rejects_membership_change_during_reference_download(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    reference_dir = tmp_path / "reference"
    status_dir = tmp_path / "status"
    write_pit(pit_dir, valid_rows())
    old_tables = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)

    class MembershipMutatingProvider(FakeProvider):
        def get_price(self, codes: list[str], **kwargs):
            result = super().get_price(codes, **kwargs)
            source_path = pit_dir / "MEMBERSHIP_SOURCE.json"
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            source_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            return result

    return_code = module.main(
        [
            "--pit-dir",
            str(pit_dir),
            "--reference-dir",
            str(reference_dir),
            "--status-out-dir",
            str(status_dir),
            "--reference-source",
            "jqdata",
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
        ],
        provider=MembershipMutatingProvider(),
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "membership_snapshot_changed"
    assert_old_tables_unchanged(reference_dir, old_tables)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source


@pytest.mark.parametrize(
    ("rows", "today", "reason_code"),
    [
        (
            valid_rows() + [("2025-01-03", "SH600000")],
            date(2025, 1, 3),
            "duplicate_pit_member_day",
        ),
        (
            valid_rows() + [("2025-01-04", "SH600000")],
            date(2025, 1, 3),
            "future_pit_date",
        ),
    ],
)
def test_rejects_duplicate_or_future_pit_before_provider_calls(
    tmp_path: Path,
    rows: list[tuple[str, str]],
    today: date,
    reason_code: str,
) -> None:
    module = load_module()
    write_pit(tmp_path / "pit", rows)
    provider = FakeProvider()

    return_code, _, status_dir = run_main(module, tmp_path, provider, today=today)

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == reason_code
    assert provider.calls == []


def test_atomic_publish_failure_rolls_back_all_four_old_tables(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    old = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)
    real_replace = os.replace
    installed = 0

    def fail_after_second_install(source, destination):
        nonlocal installed
        result = real_replace(source, destination)
        if Path(source).name.endswith(".stage"):
            installed += 1
            if installed == 2:
                raise OSError("sensitive filesystem detail")
        return result

    monkeypatch.setattr(module.os, "replace", fail_after_second_install)

    return_code, _, status_dir = run_main(module, tmp_path, FakeProvider())

    status_text = (status_dir / "download_robust_reference_latest.json").read_text()
    assert return_code != 0
    assert json.loads(status_text)["reason_code"] == "atomic_publish_failed"
    assert "sensitive filesystem detail" not in status_text
    assert_old_tables_unchanged(reference_dir, old)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source
    assert not list(reference_dir.glob(".*.stage"))
    assert not list(reference_dir.glob(".*.backup"))


def test_publish_fsync_order_precedes_prepared_and_committed_journals(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    seed_old_tables(reference_dir)
    seed_old_source(reference_dir)
    status_path = reference_dir / "download_robust_reference_latest.json"
    status_path.write_text('{"status":"old"}\n')
    tables = {
        "security_master.parquet": pd.DataFrame(
            [["SH600000", pd.Timestamp("2000-01-01"), pd.NaT, "main"]],
            columns=("symbol", "list_date", "delist_date", "board"),
        ),
        "industry_intervals.parquet": pd.DataFrame(
            [
                [
                    "SH600000",
                    "csrc:bank",
                    pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-02"),
                ]
            ],
            columns=("symbol", "industry_code", "start_date", "end_date"),
        ),
        "st_status.parquet": pd.DataFrame(
            [[pd.Timestamp("2025-01-02"), "SH600000", False]],
            columns=("date", "symbol", "is_st"),
        ),
        "suspension_status.parquet": pd.DataFrame(
            [[pd.Timestamp("2025-01-02"), "SH600000", False]],
            columns=("date", "symbol", "is_suspended"),
        ),
    }
    events: list[tuple[str, str]] = []
    real_fsync_file = module._fsync_file
    real_fsync_directories = module._fsync_directories
    real_write_journal = module._write_transaction_journal
    real_replace = module.os.replace

    def record_fsync_file(path):
        result = real_fsync_file(path)
        if Path(path).name.endswith(".stage"):
            events.append(("stage_fsync", Path(path).name))
        return result

    def record_fsync_directories(paths):
        result = real_fsync_directories(paths)
        events.append(("directory_fsync", "completed"))
        return result

    def record_journal(payload, path):
        result = real_write_journal(payload, path)
        events.append((f"{payload['state']}_journal", "durable"))
        return result

    def record_replace(source, destination):
        result = real_replace(source, destination)
        if Path(source).name.endswith(".stage"):
            events.append(("replace", Path(destination).name))
        return result

    monkeypatch.setattr(module, "_fsync_file", record_fsync_file)
    monkeypatch.setattr(module, "_fsync_directories", record_fsync_directories)
    monkeypatch.setattr(module, "_write_transaction_journal", record_journal)
    monkeypatch.setattr(module.os, "replace", record_replace)

    module._publish_transaction(
        tables,
        reference_dir,
        {"selected": "baostock"},
        reference_dir / SOURCE_FILENAME,
        {"status": "completed"},
    )

    stage_fsyncs = [
        index for index, event in enumerate(events) if event[0] == "stage_fsync"
    ]
    replaces = [index for index, event in enumerate(events) if event[0] == "replace"]
    directory_fsyncs = [
        index for index, event in enumerate(events) if event[0] == "directory_fsync"
    ]
    prepared = next(
        index for index, event in enumerate(events) if event[0] == "prepared_journal"
    )
    committed = next(
        index for index, event in enumerate(events) if event[0] == "committed_journal"
    )
    assert len(stage_fsyncs) == len(replaces) == 6
    assert max(stage_fsyncs) < prepared < min(replaces)
    assert any(index < prepared for index in directory_fsyncs)
    post_replace_fsync = next(
        index for index in directory_fsyncs if index > max(replaces)
    )
    assert max(replaces) < post_replace_fsync < committed


def test_main_recovers_prepared_publish_before_any_provider_call(
    tmp_path: Path,
) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    status_dir = tmp_path / "status"
    old_tables = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)
    targets = [reference_dir / name for name in TABLE_NAMES] + [
        reference_dir / SOURCE_FILENAME
    ]
    transaction_id = "a" * 32
    entries = []
    for offset, target in enumerate(targets):
        stage = reference_dir / f".{target.name}.{transaction_id}.stage"
        backup = reference_dir / f".{target.name}.{transaction_id}.backup"
        backup.write_bytes(target.read_bytes())
        stage.write_bytes(f"new:{target.name}".encode())
        if offset < 2:
            target.write_bytes(stage.read_bytes())
            stage.unlink()
        entries.append(
            {
                "target": str(target),
                "stage": str(stage),
                "backup": str(backup),
                "had_original": True,
            }
        )
    journal_path = reference_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.write_text(
        json.dumps(
            {
                "version": 1,
                "transaction_id": transaction_id,
                "state": "prepared",
                "entries": entries,
            }
        )
    )
    provider = FakeProvider()

    return_code = module.main(
        [
            "--pit-dir",
            str(tmp_path / "missing-pit"),
            "--reference-dir",
            str(reference_dir),
            "--status-out-dir",
            str(status_dir),
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["status"] == "blocked_data"
    assert status["reason_code"] == "membership_snapshot_invalid"
    assert provider.calls == []
    assert_old_tables_unchanged(reference_dir, old_tables)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source
    assert not journal_path.exists()
    for entry in entries:
        assert not Path(entry["stage"]).exists()
        assert not Path(entry["backup"]).exists()


def test_recovery_rejects_crafted_cleanup_path_without_deleting_canonical_file(
    tmp_path: Path,
) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    seed_old_tables(reference_dir)
    seed_old_source(reference_dir)
    target = reference_dir / "security_master.parquet"
    unrelated_canonical = reference_dir / "industry_intervals.parquet"
    original = unrelated_canonical.read_bytes()
    transaction_id = "a" * 32
    journal_path = reference_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.write_text(
        json.dumps(
            {
                "version": module.TRANSACTION_JOURNAL_VERSION,
                "transaction_id": transaction_id,
                "state": "committed",
                "entries": [
                    {
                        "target": str(target),
                        "stage": str(unrelated_canonical),
                        "backup": None,
                        "had_original": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    allowed = [reference_dir / name for name in TABLE_NAMES] + [
        reference_dir / SOURCE_FILENAME,
        reference_dir / module.STATUS_FILENAME,
    ]

    with pytest.raises(module.BlockedData) as exc_info:
        module._recover_transaction(journal_path, allowed)

    assert exc_info.value.reason_code == "transaction_recovery_failed"
    assert unrelated_canonical.read_bytes() == original
    assert journal_path.is_file()


def test_recovery_rejects_symlinked_backup_without_overwriting_target(
    tmp_path: Path,
) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    seed_old_tables(reference_dir)
    seed_old_source(reference_dir)
    target = reference_dir / "security_master.parquet"
    external = reference_dir / "industry_intervals.parquet"
    original_target = target.read_bytes()
    original_external = external.read_bytes()
    transaction_id = "b" * 32
    stage = target.with_name(f".{target.name}.{transaction_id}.stage")
    backup = target.with_name(f".{target.name}.{transaction_id}.backup")
    backup.symlink_to(external)
    journal_path = reference_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.write_text(
        json.dumps(
            {
                "version": module.TRANSACTION_JOURNAL_VERSION,
                "transaction_id": transaction_id,
                "state": "prepared",
                "entries": [
                    {
                        "target": str(target),
                        "stage": str(stage),
                        "backup": str(backup),
                        "had_original": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.BlockedData) as exc_info:
        module._recover_transaction(journal_path, [target])

    assert exc_info.value.reason_code == "transaction_recovery_failed"
    assert target.read_bytes() == original_target
    assert external.read_bytes() == original_external
    assert journal_path.is_file()


def test_recovery_rejects_symlinked_transaction_journal(tmp_path: Path) -> None:
    module = load_module()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    target = reference_dir / "security_master.parquet"
    transaction_id = "c" * 32
    stage = target.with_name(f".{target.name}.{transaction_id}.stage")
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
                        "backup": None,
                        "had_original": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal_path = reference_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal_path.symlink_to(external_journal)

    with pytest.raises(module.BlockedData) as exc_info:
        module._recover_transaction(journal_path, [target])

    assert exc_info.value.reason_code == "transaction_recovery_failed"
    assert journal_path.is_symlink()
    assert external_journal.is_file()


def test_crash_recovery_does_not_depend_on_changed_external_status_dir(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    reference_dir = tmp_path / "reference"
    status_dir_a = tmp_path / "status-a"
    status_dir_b = tmp_path / "status-b"
    write_pit(pit_dir, valid_rows())
    old_tables = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)
    canonical_status = reference_dir / "download_robust_reference_latest.json"
    old_canonical_status = b'{"status":"old-canonical"}\n'
    canonical_status.write_bytes(old_canonical_status)
    real_replace = module.os.replace
    crashed = False

    def crash_after_first_target_replace(source, destination):
        nonlocal crashed
        result = real_replace(source, destination)
        if Path(source).name.endswith(".stage") and not crashed:
            crashed = True
            raise SystemExit("simulated publish crash")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", crash_after_first_target_replace)
        with pytest.raises(SystemExit, match="simulated publish crash"):
            module.main(
                [
                    "--pit-dir",
                    str(pit_dir),
                    "--reference-dir",
                    str(reference_dir),
                    "--status-out-dir",
                    str(status_dir_a),
                    "--reference-source",
                    "jqdata",
                    "--start-date",
                    "2025-01-02",
                    "--end-date",
                    "2025-01-03",
                ],
                provider=FakeProvider(),
                today=date(2025, 1, 3),
            )

    journal_path = reference_dir / module.TRANSACTION_JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text())
    assert all(
        Path(entry["target"]).parent == reference_dir for entry in journal["entries"]
    )
    provider = FakeProvider()
    return_code = module.main(
        [
            "--pit-dir",
            str(tmp_path / "missing-pit"),
            "--reference-dir",
            str(reference_dir),
            "--status-out-dir",
            str(status_dir_b),
            "--start-date",
            "2025-01-02",
            "--end-date",
            "2025-01-03",
        ],
        provider=provider,
        today=date(2025, 1, 3),
    )

    external_status = json.loads(
        (status_dir_b / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert external_status["reason_code"] == "membership_snapshot_invalid"
    assert provider.calls == []
    assert_old_tables_unchanged(reference_dir, old_tables)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source
    assert canonical_status.read_bytes() == old_canonical_status
    assert not journal_path.exists()


def test_writer_lock_is_released_during_build_and_reacquired_for_publish(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    reference_dir = tmp_path / "reference"
    status_dir = tmp_path / "status"
    write_pit(pit_dir, valid_rows())
    build_started = threading.Event()
    allow_build = threading.Event()
    publish_started = threading.Event()
    result: dict[str, object] = {}

    class BlockingProvider(FakeProvider):
        def get_security_info(self, code: str):
            if not build_started.is_set():
                build_started.set()
                if not allow_build.wait(timeout=5):
                    raise RuntimeError("build test timeout")
            return super().get_security_info(code)

    real_publish = module._publish_transaction

    def record_publish(*args, **kwargs):
        publish_started.set()
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(module, "_publish_transaction", record_publish)

    def run_writer() -> None:
        try:
            result["return_code"] = module.main(
                [
                    "--pit-dir",
                    str(pit_dir),
                    "--reference-dir",
                    str(reference_dir),
                    "--status-out-dir",
                    str(status_dir),
                    "--reference-source",
                    "jqdata",
                    "--start-date",
                    "2025-01-02",
                    "--end-date",
                    "2025-01-03",
                ],
                provider=BlockingProvider(),
                today=date(2025, 1, 3),
            )
        except BaseException as exc:
            result["error"] = exc

    writer = threading.Thread(target=run_writer, daemon=True)
    writer.start()
    assert build_started.wait(timeout=2)
    lock_path = (
        reference_dir.parent / f".{reference_dir.name}{module.WRITER_LOCK_FILENAME}"
    )
    shared_handle = lock_path.open("a+b")
    shared_acquired = False
    publish_during_shared_lock = False
    try:
        try:
            fcntl.flock(shared_handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            shared_acquired = True
        except BlockingIOError:
            shared_acquired = False
        allow_build.set()
        publish_during_shared_lock = publish_started.wait(timeout=0.2)
    finally:
        allow_build.set()
        if shared_acquired:
            fcntl.flock(shared_handle.fileno(), fcntl.LOCK_UN)
        shared_handle.close()
    assert publish_started.wait(timeout=2)
    writer.join(timeout=5)

    assert shared_acquired
    assert not publish_during_shared_lock
    assert not writer.is_alive()
    assert "error" not in result
    assert result["return_code"] == 0


def test_baostock_failure_uses_safe_reason_and_never_overwrites_old_source(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    provider = FakeBaostockProvider()
    secret = "socket://private-host?token=never-leak"
    provider.exceptions["basic"] = [RuntimeError(secret)]
    reference_dir = tmp_path / "reference"
    old = seed_old_tables(reference_dir)
    old_source = seed_old_source(reference_dir)

    return_code, _, status_dir = run_main(
        module,
        tmp_path,
        baostock_provider=provider,
        reference_source="baostock",
    )

    captured = capsys.readouterr()
    status_text = (status_dir / "download_robust_reference_latest.json").read_text()
    status = json.loads(status_text)
    assert return_code != 0
    assert status["reason_code"] == "baostock_fetch_failed"
    assert status["requested"] == status["selected"] == "baostock"
    assert secret not in status_text + captured.out + captured.err
    assert_old_tables_unchanged(reference_dir, old)
    assert (reference_dir / SOURCE_FILENAME).read_bytes() == old_source
    assert provider.logout_count == 1


def test_real_future_delist_date_blocks_instead_of_being_silently_cleared(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = FakeProvider()
    provider.security["600000.XSHG"] = SimpleNamespace(
        start_date=date(2000, 1, 1),
        end_date=date(2025, 1, 4),
    )
    reference_dir = tmp_path / "reference"
    old = seed_old_tables(reference_dir)

    return_code, _, status_dir = run_main(
        module, tmp_path, provider, today=date(2025, 1, 3)
    )

    status = json.loads(
        (status_dir / "download_robust_reference_latest.json").read_text()
    )
    assert return_code != 0
    assert status["reason_code"] == "future_security_master_date"
    assert_old_tables_unchanged(reference_dir, old)


def test_final_validation_rejects_industry_gap_overlap_or_future_interval() -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    base = {
        "security_master.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "list_date": pd.to_datetime(["2000-01-01"]),
                "delist_date": [pd.NaT],
                "board": ["main"],
            }
        ),
        "industry_intervals.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "industry_code": ["801010"],
                "start_date": pd.to_datetime(["2025-01-02"]),
                "end_date": pd.to_datetime(["2025-01-02"]),
            }
        ),
        "st_status.parquet": membership.assign(is_st=False),
        "suspension_status.parquet": membership.assign(is_suspended=False),
    }

    with pytest.raises(module.BlockedData, match="industry"):
        module._validate_tables(base, membership, date(2025, 1, 3))

    future = dict(base)
    future["industry_intervals.parquet"] = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "industry_code": ["801010"],
            "start_date": pd.to_datetime(["2025-01-02"]),
            "end_date": pd.to_datetime(["2025-01-04"]),
        }
    )
    with pytest.raises(module.BlockedData, match="future"):
        module._validate_tables(future, membership, date(2025, 1, 3))


def test_final_validation_rejects_empty_industry_code() -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "symbol": ["SH600000"],
        }
    )
    tables = {
        "security_master.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "list_date": pd.to_datetime(["2000-01-01"]),
                "delist_date": [pd.NaT],
                "board": ["main"],
            }
        ),
        "industry_intervals.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "industry_code": [""],
                "start_date": membership["date"],
                "end_date": membership["date"],
            }
        ),
        "st_status.parquet": membership.assign(is_st=False),
        "suspension_status.parquet": membership.assign(is_suspended=False),
    }

    with pytest.raises(module.BlockedData, match="industry"):
        module._validate_tables(tables, membership, date(2025, 1, 2))


def test_final_validation_rejects_unbounded_post_delist_membership() -> None:
    module = load_module()
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-06-29"]),
            "symbol": ["SH600000", "SH600000"],
        }
    )
    tables = {
        "security_master.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "list_date": pd.to_datetime(["2000-01-01"]),
                "delist_date": pd.to_datetime(["2025-01-02"]),
                "board": ["main"],
            }
        ),
        "industry_intervals.parquet": pd.DataFrame(
            {
                "symbol": ["SH600000"],
                "industry_code": ["801010"],
                "start_date": pd.to_datetime(["2025-01-02"]),
                "end_date": pd.to_datetime(["2025-06-29"]),
            }
        ),
        "st_status.parquet": membership.assign(is_st=False),
        "suspension_status.parquet": membership.assign(is_suspended=True),
    }

    with pytest.raises(module.BlockedData, match="security_master"):
        module._validate_tables(tables, membership, date(2025, 6, 29))
