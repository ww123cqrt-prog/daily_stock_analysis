from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_robust_fundamentals.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "download_robust_fundamentals", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, pages: dict[int, dict]) -> None:
        self.pages = pages
        self.calls: list[dict] = []

    def get(self, _url: str, *, params: dict, timeout: float):
        assert timeout > 0
        self.calls.append(dict(params))
        page = params.get("p", params.get("pageNumber"))
        return FakeResponse(self.pages[int(page)])


def test_fetch_statement_batch_reads_every_declared_page() -> None:
    module = load_module()
    session = FakeSession(
        {
            1: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {"SECUCODE": "600000.SH", "REPORT_DATE": "2025-12-31"}
                    ],
                },
            },
            2: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {"SECUCODE": "000001.SZ", "REPORT_DATE": "2026-03-31"}
                    ],
                },
            },
        }
    )

    frame = module.fetch_statement_batch(
        session,
        statement="income",
        symbols=["SH600000", "SZ000001"],
        start_report_date="2010-01-01",
        end_report_date="2025-12-31",
        timeout=5.0,
    )

    assert frame["SECUCODE"].tolist() == ["600000.SH"]
    assert [call["p"] for call in session.calls] == ["1", "2"]
    assert session.calls[0]["sr"] == "1,1"
    assert "REPORT_DATE" not in session.calls[0]["filter"]


def test_cache_key_changes_with_source_contract() -> None:
    module = load_module()

    first = module.batch_cache_key(
        statement="income",
        symbols=["SH600000"],
        start_report_date="2010-01-01",
        end_report_date="2025-12-31",
        source_contract_version=1,
    )
    second = module.batch_cache_key(
        statement="income",
        symbols=["SH600000"],
        start_report_date="2010-01-01",
        end_report_date="2025-12-31",
        source_contract_version=2,
    )

    assert first != second


def test_fetch_summary_period_reads_all_pages_and_adds_update_date() -> None:
    module = load_module()
    session = FakeSession(
        {
            1: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {
                            "SECUCODE": "600000.SH",
                            "REPORT_DATE": "2022-12-31",
                            "NOTICE_DATE": "2024-04-30",
                            "TOTAL_ASSETS": 100.0,
                            "TOTAL_LIABILITIES": 60.0,
                            "TOTAL_EQUITY": 40.0,
                        }
                    ],
                },
            },
            2: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {
                            "SECUCODE": "000001.SZ",
                            "REPORT_DATE": "2022-12-31",
                            "NOTICE_DATE": "2023-04-25",
                            "TOTAL_ASSETS": 200.0,
                            "TOTAL_LIABILITIES": 100.0,
                            "TOTAL_EQUITY": 100.0,
                        }
                    ],
                },
            },
        }
    )

    frame = module.fetch_summary_period(
        session,
        statement="balance",
        report_date="2022-12-31",
        timeout=5.0,
    )

    assert len(frame) == 2
    assert frame["UPDATE_DATE"].equals(frame["NOTICE_DATE"])
    assert frame["TOTAL_PARENT_EQUITY"].equals(frame["TOTAL_EQUITY"])
    assert [call["pageNumber"] for call in session.calls] == ["1", "2"]


def test_fetch_income_period_reads_every_page() -> None:
    module = load_module()
    session = FakeSession(
        {
            1: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {
                            "SECUCODE": "600000.SH",
                            "REPORT_DATE": "2022-12-31",
                            "NOTICE_DATE": "2023-03-20",
                            "UPDATE_DATE": "2024-04-30",
                        }
                    ],
                },
            },
            2: {
                "success": True,
                "message": "ok",
                "result": {
                    "pages": 2,
                    "count": 2,
                    "data": [
                        {
                            "SECUCODE": "000001.SZ",
                            "REPORT_DATE": "2022-12-31",
                            "NOTICE_DATE": "2023-04-25",
                            "UPDATE_DATE": "2023-04-25",
                        }
                    ],
                },
            },
        }
    )

    frame = module.fetch_income_period(
        session, report_date="2022-12-31", timeout=5.0
    )

    assert frame["SECUCODE"].tolist() == ["600000.SH", "000001.SZ"]
    assert [call["p"] for call in session.calls] == ["1", "2"]


def test_load_member_market_rows_excludes_nonmember_dates(tmp_path: Path) -> None:
    module = load_module()
    pd = __import__("pandas")
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "close": [10.0, 11.0],
            "volume": [100.0, 110.0],
            "amount": [1_000.0, 1_210.0],
            "turn": [1.0, 1.1],
        }
    ).to_parquet(tmp_path / "600000.parquet", index=False)
    membership = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-03"]), "symbol": ["SH600000"]}
    )

    result = module.load_member_market_rows(membership, price_dir=tmp_path)

    assert result[["date", "symbol"]].to_dict("records") == [
        {"date": pd.Timestamp("2025-01-03"), "symbol": "SH600000"}
    ]


def test_build_product_writes_events_daily_partitions_and_source_contract(
    tmp_path: Path,
) -> None:
    module = load_module()
    pd = __import__("pandas")
    common = {
        "SECUCODE": ["600000.SH"],
        "REPORT_DATE": ["2022-12-31"],
        "NOTICE_DATE": ["2024-04-30"],
        "UPDATE_DATE": ["2024-04-30"],
    }
    balance = pd.DataFrame(
        {
            **common,
            "TOTAL_ASSETS": [1_000.0],
            "TOTAL_LIABILITIES": [600.0],
            "TOTAL_PARENT_EQUITY": [400.0],
            "SHARE_CAPITAL": [100.0],
        }
    )
    income = pd.DataFrame(
        {
            **common,
            "TOTAL_OPERATE_INCOME": [300.0],
            "TOTAL_OPERATE_INCOME_YOY": [20.0],
            "PARENT_NETPROFIT": [40.0],
            "PARENT_NETPROFIT_YOY": [10.0],
            "BASIC_EPS": [0.4],
        }
    )
    cashflow = pd.DataFrame({**common, "NETCASH_OPERATE": [80.0]})
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-05-06")],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1_000.0],
            "turn": [10.0],
        }
    ).to_parquet(price_dir / "600000.parquet", index=False)
    output = tmp_path / "fundamentals"

    result = module.build_fundamental_product(
        raw_statements={
            "balance": balance,
            "income": income,
            "cashflow": cashflow,
        },
        calendar=pd.DatetimeIndex(pd.to_datetime(["2024-04-30", "2024-05-06"])),
        membership=pd.DataFrame(
            {"date": [pd.Timestamp("2024-05-06")], "symbol": ["SH600000"]}
        ),
        price_dir=price_dir,
        output_dir=output,
        coverage_minimum=1.0,
        source_metadata={"selected_source": "test"},
    )

    assert result["status"] == "pass"
    assert (output / "fundamental_events.parquet").is_file()
    assert (output / "daily" / "2024.parquet").is_file()
    source = json.loads((output / "FUNDAMENTAL_SOURCE.json").read_text())
    assert source["effective_date_rule"] == "max_notice_or_update_plus_one_trading_day"
    assert source["coverage"]["passed"] is True


def test_report_periods_end_at_frozen_2025_boundary() -> None:
    module = load_module()

    periods = module.report_periods("2025-01-01", "2025-12-31")

    assert periods == ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def test_period_cache_reuses_complete_parquet(tmp_path: Path) -> None:
    module = load_module()
    pd = __import__("pandas")
    calls = 0

    def fetcher():
        nonlocal calls
        calls += 1
        return pd.DataFrame({"SECUCODE": ["600000.SH"], "value": [calls]})

    first = module.load_or_fetch_period(
        cache_dir=tmp_path,
        statement="income",
        report_date="2022-12-31",
        fetcher=fetcher,
    )
    second = module.load_or_fetch_period(
        cache_dir=tmp_path,
        statement="income",
        report_date="2022-12-31",
        fetcher=fetcher,
    )

    assert calls == 1
    assert first.equals(second)


def test_missing_historical_members_are_supplemented_by_symbol_batches() -> None:
    module = load_module()
    pd = __import__("pandas")
    base = pd.DataFrame(
        {"SECUCODE": ["600000.SH"], "REPORT_DATE": ["2022-12-31"]}
    )
    calls: list[list[str]] = []

    def fetcher(symbols):
        calls.append(list(symbols))
        return pd.DataFrame(
            {
                "SECUCODE": ["600005.SH"],
                "REPORT_DATE": ["2014-12-31"],
            }
        )

    result, supplemented = module.supplement_missing_symbols(
        base,
        expected_symbols=["SH600000", "SH600005"],
        fetcher=fetcher,
        batch_size=20,
    )

    assert calls == [["SH600005"]]
    assert supplemented == ["SH600005"]
    assert sorted(result["SECUCODE"].unique()) == ["600000.SH", "600005.SH"]


def test_symbol_batch_cache_reuses_complete_parquet(tmp_path: Path) -> None:
    module = load_module()
    pd = __import__("pandas")
    calls = 0

    def fetcher():
        nonlocal calls
        calls += 1
        return pd.DataFrame({"SECUCODE": ["600005.SH"], "value": [calls]})

    kwargs = {
        "cache_dir": tmp_path,
        "statement": "income",
        "symbols": ["SH600005"],
        "start_report_date": "2010-01-01",
        "end_report_date": "2025-12-31",
        "fetcher": fetcher,
    }
    first = module.load_or_fetch_symbol_batch(**kwargs)
    second = module.load_or_fetch_symbol_batch(**kwargs)

    assert calls == 1
    assert first.equals(second)


def test_goal1_download_rejects_any_2026_boundary() -> None:
    module = load_module()

    with pytest.raises(ValueError, match="2026"):
        module.validate_goal1_dates(
            data_start="2014-01-02",
            data_end="2026-01-02",
            report_start="2010-01-01",
            report_end="2025-12-31",
        )
