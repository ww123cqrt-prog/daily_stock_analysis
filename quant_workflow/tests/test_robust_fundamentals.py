from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_fundamentals import (
    FACTOR_COLUMNS,
    build_daily_fundamental_features,
    build_fundamental_events,
    fundamental_coverage_report,
    prepare_statement_frame,
)


def _raw_statement(
    *,
    notice_date: str,
    update_date: str,
    statement: str,
) -> pd.DataFrame:
    common = {
        "SECUCODE": ["600000.SH"],
        "REPORT_DATE": ["2022-12-31"],
        "NOTICE_DATE": [notice_date],
        "UPDATE_DATE": [update_date],
        "REPORT_TYPE": ["annual"],
    }
    if statement == "balance":
        common.update(
            {
                "TOTAL_ASSETS": [1_000.0],
                "TOTAL_LIABILITIES": [600.0],
                "TOTAL_PARENT_EQUITY": [400.0],
                "SHARE_CAPITAL": [100.0],
            }
        )
    elif statement == "income":
        common.update(
            {
                "TOTAL_OPERATE_INCOME": [300.0],
                "TOTAL_OPERATE_INCOME_YOY": [20.0],
                "PARENT_NETPROFIT": [40.0],
                "PARENT_NETPROFIT_YOY": [10.0],
                "BASIC_EPS": [0.4],
            }
        )
    elif statement == "cashflow":
        common.update({"NETCASH_OPERATE": [80.0]})
    else:  # pragma: no cover - test helper contract
        raise AssertionError(statement)
    return pd.DataFrame(common)


def test_prepare_statement_preserves_notice_and_delays_revised_values() -> None:
    prepared = prepare_statement_frame(
        _raw_statement(
            notice_date="2023-03-20",
            update_date="2024-04-30",
            statement="income",
        ),
        statement="income",
    )

    row = prepared.iloc[0]
    assert row["notice_date"] == pd.Timestamp("2023-03-20")
    assert row["update_date"] == pd.Timestamp("2024-04-30")
    assert row["source_available_date"] == pd.Timestamp("2024-04-30")
    assert bool(row["revision_delayed"]) is True


def test_fundamental_event_becomes_available_on_next_trading_day() -> None:
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-04-30", "2024-05-06", "2024-05-07"]))
    balance = prepare_statement_frame(
        _raw_statement(
            notice_date="2023-03-20",
            update_date="2024-04-30",
            statement="balance",
        ),
        statement="balance",
    )
    income = prepare_statement_frame(
        _raw_statement(
            notice_date="2023-03-20",
            update_date="2024-04-30",
            statement="income",
        ),
        statement="income",
    )
    cashflow = prepare_statement_frame(
        _raw_statement(
            notice_date="2023-03-20",
            update_date="2024-04-30",
            statement="cashflow",
        ),
        statement="cashflow",
    )

    events = build_fundamental_events(
        balance=balance,
        income=income,
        cashflow=cashflow,
        calendar=calendar,
    )

    assert events.loc[0, "effective_date"] == pd.Timestamp("2024-05-06")
    assert events.loc[0, "symbol"] == "SH600000"


def test_daily_features_never_use_event_before_effective_date() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "report_date": [pd.Timestamp("2022-12-31")],
            "effective_date": [pd.Timestamp("2024-05-06")],
            "total_assets": [1_000.0],
            "total_liabilities": [600.0],
            "total_parent_equity": [400.0],
            "share_capital": [100.0],
            "total_operate_income": [300.0],
            "revenue_yoy": [0.20],
            "parent_netprofit": [40.0],
            "netprofit_yoy": [0.10],
            "basic_eps": [0.4],
            "netcash_operate": [80.0],
            "annualization_factor": [1.0],
        }
    )
    market_rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-30", "2024-05-06", "2024-05-07"]),
            "symbol": ["SH600000"] * 3,
            "close": [10.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0],
            "amount": [1_000.0, 1_000.0, 1_000.0],
        }
    )

    features = build_daily_fundamental_features(events, market_rows)

    before = features.loc[features["datetime"] == pd.Timestamp("2024-04-30")]
    after = features.loc[features["datetime"] == pd.Timestamp("2024-05-06")]
    assert before[list(FACTOR_COLUMNS)].isna().all(axis=None)
    assert np.isclose(after.iloc[0]["FUND_QUALITY_ROE"], 0.10)
    assert np.isclose(after.iloc[0]["FUND_GROWTH_REVENUE_YOY"], 0.20)
    assert np.isclose(after.iloc[0]["FUND_CASHFLOW_TO_ASSETS"], 0.08)
    assert np.isclose(after.iloc[0]["FUND_LEVERAGE_DEBT_TO_ASSETS"], 0.60)
    assert np.isclose(after.iloc[0]["FUND_VALUE_EARNINGS_YIELD"], 0.04)
    assert np.isclose(after.iloc[0]["FUND_VALUE_BOOK_TO_MARKET"], 0.40)


def test_coverage_gate_requires_every_registered_family_at_98_percent() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02"] * 100),
            "instrument": [f"SH60{number:04d}" for number in range(100)],
            **{column: np.ones(100) for column in FACTOR_COLUMNS},
        }
    )
    frame.loc[0:2, "FUND_CASHFLOW_TO_ASSETS"] = np.nan

    report = fundamental_coverage_report(frame, minimum=0.98)

    assert report["families"]["cash_flow"]["coverage"] == 0.97
    assert report["passed"] is False
    assert report["reason_code"] == "fundamental_coverage_below_threshold"


def test_valuation_falls_back_to_turnover_implied_float_market_cap() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "report_date": [pd.Timestamp("2024-03-31")],
            "effective_date": [pd.Timestamp("2024-05-06")],
            "total_assets": [1_000.0],
            "total_liabilities": [600.0],
            "total_parent_equity": [400.0],
            "share_capital": [np.nan],
            "total_operate_income": [300.0],
            "revenue_yoy": [0.20],
            "parent_netprofit": [10.0],
            "netprofit_yoy": [0.10],
            "basic_eps": [0.10],
            "netcash_operate": [20.0],
            "annualization_factor": [4.0],
        }
    )
    market_rows = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-05-06")],
            "symbol": ["SH600000"],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1_000.0],
            "turn": [10.0],
        }
    )

    result = build_daily_fundamental_features(events, market_rows).iloc[0]

    assert np.isclose(result["FUND_VALUE_EARNINGS_YIELD"], 0.04)
    assert np.isclose(result["FUND_VALUE_BOOK_TO_MARKET"], 0.04)


def test_coverage_gate_passes_at_exact_threshold() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-02"] * 100),
            "instrument": [f"SH60{number:04d}" for number in range(100)],
            **{column: np.ones(100) for column in FACTOR_COLUMNS},
        }
    )
    frame.loc[0:1, "FUND_GROWTH_NETPROFIT_YOY"] = np.nan

    report = fundamental_coverage_report(frame, minimum=0.98)

    assert report["families"]["growth"]["coverage"] == 0.98
    assert report["passed"] is True
