from __future__ import annotations

import importlib.util
import json
import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

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

    def get_security_info(self, code: str):
        self.calls.append(("security", code))
        if self.error is not None:
            raise self.error
        return self.security.get(code)

    def get_industry(self, codes: list[str], date: str):
        self.calls.append(("industry", tuple(codes), date))
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


def write_pit(pit_dir: Path, rows: list[tuple[str, str]]) -> None:
    daily = pit_dir / "daily"
    daily.mkdir(parents=True)
    pd.DataFrame(rows, columns=["date", "symbol"]).assign(
        index_code="000906.XSHG"
    ).to_parquet(
        daily / "2025.parquet",
        index=False,
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
    module, tmp_path: Path, provider: FakeProvider, *, today: date = date(2025, 1, 3)
):
    pit_dir = tmp_path / "pit"
    reference_dir = tmp_path / "reference"
    status_dir = tmp_path / "status"
    if not (pit_dir / "daily").exists():
        write_pit(pit_dir, valid_rows())
    return_code = module.main(
        [
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
        ],
        provider=provider,
        today=today,
    )
    return return_code, reference_dir, status_dir


def seed_old_tables(reference_dir: Path) -> dict[str, bytes]:
    reference_dir.mkdir(parents=True, exist_ok=True)
    payloads = {name: f"old:{name}".encode() for name in TABLE_NAMES}
    for name, payload in payloads.items():
        (reference_dir / name).write_bytes(payload)
    return payloads


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
    assert not list(reference_dir.glob(".*.stage"))
    assert not list(reference_dir.glob(".*.backup"))


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
