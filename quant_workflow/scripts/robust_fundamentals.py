"""Leakage-safe transformations for historical fundamental research data."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


STATEMENT_FIELDS: Mapping[str, Mapping[str, str]] = {
    "balance": {
        "TOTAL_ASSETS": "total_assets",
        "TOTAL_LIABILITIES": "total_liabilities",
        "TOTAL_PARENT_EQUITY": "total_parent_equity",
        "SHARE_CAPITAL": "share_capital",
    },
    "income": {
        "TOTAL_OPERATE_INCOME": "total_operate_income",
        "TOTAL_OPERATE_INCOME_YOY": "revenue_yoy",
        "PARENT_NETPROFIT": "parent_netprofit",
        "PARENT_NETPROFIT_YOY": "netprofit_yoy",
        "BASIC_EPS": "basic_eps",
    },
    "cashflow": {
        "NETCASH_OPERATE": "netcash_operate",
    },
}

FACTOR_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "valuation": (
        "FUND_VALUE_EARNINGS_YIELD",
        "FUND_VALUE_BOOK_TO_MARKET",
    ),
    "quality": ("FUND_QUALITY_ROE",),
    "growth": (
        "FUND_GROWTH_REVENUE_YOY",
        "FUND_GROWTH_NETPROFIT_YOY",
    ),
    "cash_flow": ("FUND_CASHFLOW_TO_ASSETS",),
    "leverage": (
        "FUND_LEVERAGE_DEBT_TO_ASSETS",
        "FUND_LEVERAGE_EQUITY_TO_ASSETS",
    ),
}
FACTOR_COLUMNS = tuple(
    column for columns in FACTOR_FAMILIES.values() for column in columns
)


def _normalize_symbol(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if len(text) == 9 and text[6:] in {".SH", ".SZ", ".BJ"} and text[:6].isdigit():
        return f"{text[7:]}{text[:6]}"
    if len(text) == 8 and text[:2] in {"SH", "SZ", "BJ"} and text[2:].isdigit():
        return text
    return None


def prepare_statement_frame(frame: pd.DataFrame, *, statement: str) -> pd.DataFrame:
    """Normalize one statement while preserving both publication timestamps.

    EastMoney exposes the original notice date and the last update date but only
    returns the current values. A revised row therefore cannot become available
    before the later of those two dates.
    """
    if statement not in STATEMENT_FIELDS:
        raise ValueError(f"unknown fundamental statement: {statement}")
    required = {
        "SECUCODE",
        "REPORT_DATE",
        "NOTICE_DATE",
        "UPDATE_DATE",
        *STATEMENT_FIELDS[statement].keys(),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{statement} statement is missing columns: {', '.join(missing)}")
    selected = frame.loc[:, sorted(required | {"REPORT_TYPE"} & set(frame.columns))].copy()
    selected["symbol"] = selected["SECUCODE"].map(_normalize_symbol)
    selected["report_date"] = pd.to_datetime(
        selected["REPORT_DATE"], errors="coerce"
    ).dt.normalize()
    selected["notice_date"] = pd.to_datetime(
        selected["NOTICE_DATE"], errors="coerce"
    ).dt.normalize()
    selected["update_date"] = pd.to_datetime(
        selected["UPDATE_DATE"], errors="coerce"
    ).dt.normalize()
    selected["update_date"] = selected["update_date"].fillna(selected["notice_date"])
    selected["source_available_date"] = selected[
        ["notice_date", "update_date"]
    ].max(axis=1)
    selected["revision_delayed"] = selected["update_date"].gt(
        selected["notice_date"]
    )
    for raw, normalized in STATEMENT_FIELDS[statement].items():
        values = pd.to_numeric(selected[raw], errors="coerce")
        if normalized in {"revenue_yoy", "netprofit_yoy"}:
            values = values / 100.0
        selected[normalized] = values
    selected = selected.dropna(
        subset=["symbol", "report_date", "notice_date", "source_available_date"]
    )
    output_columns = [
        "symbol",
        "report_date",
        "notice_date",
        "update_date",
        "source_available_date",
        "revision_delayed",
        *STATEMENT_FIELDS[statement].values(),
    ]
    selected = selected.loc[:, output_columns].sort_values(
        ["symbol", "report_date", "source_available_date"], kind="stable"
    )
    return selected.drop_duplicates(["symbol", "report_date"], keep="last").reset_index(
        drop=True
    )


def _next_trading_day(
    values: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    days = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
    days = days.dropna().normalize().sort_values().unique()
    if days.empty:
        raise ValueError("fundamental effective-date calendar is empty")
    available = pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[ns]")
    positions = days.searchsorted(available, side="right")
    result = np.full(len(available), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions < len(days)
    result[valid] = days.to_numpy(dtype="datetime64[ns]")[positions[valid]]
    return pd.Series(pd.to_datetime(result), index=values.index)


def build_fundamental_events(
    *,
    balance: pd.DataFrame,
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Combine complete statement sets into conservative PIT events."""

    def renamed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        metadata = {
            "notice_date": f"{prefix}_notice_date",
            "update_date": f"{prefix}_update_date",
            "source_available_date": f"{prefix}_source_available_date",
            "revision_delayed": f"{prefix}_revision_delayed",
        }
        return frame.rename(columns=metadata)

    merged = renamed(balance, "balance").merge(
        renamed(income, "income"), on=["symbol", "report_date"], how="inner"
    )
    merged = merged.merge(
        renamed(cashflow, "cashflow"), on=["symbol", "report_date"], how="inner"
    )
    available_columns = [
        "balance_source_available_date",
        "income_source_available_date",
        "cashflow_source_available_date",
    ]
    merged["source_available_date"] = merged[available_columns].max(axis=1)
    merged["effective_date"] = _next_trading_day(
        merged["source_available_date"], calendar
    )
    merged["revision_delayed"] = merged[
        [
            "balance_revision_delayed",
            "income_revision_delayed",
            "cashflow_revision_delayed",
        ]
    ].any(axis=1)
    months = merged["report_date"].dt.month.map({3: 3, 6: 6, 9: 9, 12: 12})
    merged["annualization_factor"] = 12.0 / pd.to_numeric(months, errors="coerce")
    merged = merged.dropna(subset=["effective_date", "annualization_factor"])
    return merged.sort_values(
        ["symbol", "effective_date", "report_date"], kind="stable"
    ).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce")
    result = top / bottom.where(bottom.abs() > 0)
    return result.replace([np.inf, -np.inf], np.nan)


def build_daily_fundamental_features(
    events: pd.DataFrame, market_rows: pd.DataFrame
) -> pd.DataFrame:
    """As-of join statement events onto market rows and calculate five families."""
    required_market = {"date", "symbol", "close", "volume", "amount"}
    missing_market = sorted(required_market - set(market_rows.columns))
    if missing_market:
        raise ValueError(f"market rows are missing columns: {', '.join(missing_market)}")
    market = market_rows.copy()
    market["datetime"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["instrument"] = market["symbol"].map(_normalize_symbol)
    market = market.dropna(subset=["datetime", "instrument"]).sort_values(
        ["instrument", "datetime"], kind="stable"
    )
    normalized_events = events.copy()
    normalized_events["symbol"] = normalized_events["symbol"].map(_normalize_symbol)
    normalized_events["effective_date"] = pd.to_datetime(
        normalized_events["effective_date"], errors="coerce"
    ).dt.normalize()
    normalized_events = normalized_events.dropna(subset=["symbol", "effective_date"])

    joined_blocks: list[pd.DataFrame] = []
    for instrument, rows in market.groupby("instrument", sort=False):
        candidates = normalized_events.loc[normalized_events["symbol"] == instrument]
        rows = rows.sort_values("datetime", kind="stable")
        if candidates.empty:
            joined = rows.copy()
            for column in normalized_events.columns:
                if column not in joined.columns:
                    joined[column] = np.nan
        else:
            joined = pd.merge_asof(
                rows,
                candidates.sort_values("effective_date", kind="stable"),
                left_on="datetime",
                right_on="effective_date",
                direction="backward",
                allow_exact_matches=True,
            )
        joined_blocks.append(joined)
    joined = pd.concat(joined_blocks, ignore_index=True) if joined_blocks else market.copy()
    vwap = _safe_ratio(joined["amount"], joined["volume"]).fillna(
        pd.to_numeric(joined["close"], errors="coerce")
    )
    reported_share_capital = pd.to_numeric(
        joined.get("share_capital"), errors="coerce"
    )
    eps = pd.to_numeric(joined.get("basic_eps"), errors="coerce")
    annualization = pd.to_numeric(joined.get("annualization_factor"), errors="coerce")
    annual_profit = pd.to_numeric(joined.get("parent_netprofit"), errors="coerce") * annualization
    inferred_share_capital = _safe_ratio(annual_profit, eps * annualization)
    inferred_share_capital = inferred_share_capital.where(inferred_share_capital > 0)
    market_cap = vwap * reported_share_capital
    if "turn" in joined.columns:
        turnover_market_cap = _safe_ratio(
            pd.to_numeric(joined["amount"], errors="coerce"),
            pd.to_numeric(joined["turn"], errors="coerce") / 100.0,
        )
        market_cap = market_cap.fillna(turnover_market_cap)
    market_cap = market_cap.fillna(vwap * inferred_share_capital)
    annual_cashflow = pd.to_numeric(joined.get("netcash_operate"), errors="coerce") * annualization
    assets = pd.to_numeric(joined.get("total_assets"), errors="coerce")
    equity = pd.to_numeric(joined.get("total_parent_equity"), errors="coerce")
    liabilities = pd.to_numeric(joined.get("total_liabilities"), errors="coerce")
    per_share_earnings_yield = _safe_ratio(
        pd.to_numeric(joined.get("basic_eps"), errors="coerce") * annualization,
        vwap,
    )
    joined["FUND_VALUE_EARNINGS_YIELD"] = per_share_earnings_yield.fillna(
        _safe_ratio(annual_profit, market_cap)
    )
    joined["FUND_VALUE_BOOK_TO_MARKET"] = _safe_ratio(equity, market_cap)
    joined["FUND_QUALITY_ROE"] = _safe_ratio(annual_profit, equity)
    joined["FUND_GROWTH_REVENUE_YOY"] = pd.to_numeric(
        joined.get("revenue_yoy"), errors="coerce"
    )
    joined["FUND_GROWTH_NETPROFIT_YOY"] = pd.to_numeric(
        joined.get("netprofit_yoy"), errors="coerce"
    )
    joined["FUND_CASHFLOW_TO_ASSETS"] = _safe_ratio(annual_cashflow, assets)
    joined["FUND_LEVERAGE_DEBT_TO_ASSETS"] = _safe_ratio(liabilities, assets)
    joined["FUND_LEVERAGE_EQUITY_TO_ASSETS"] = _safe_ratio(equity, assets)
    metadata = [
        column
        for column in ["report_date", "effective_date", "source_available_date"]
        if column in joined.columns
    ]
    return joined.loc[:, ["datetime", "instrument", *FACTOR_COLUMNS, *metadata]].sort_values(
        ["datetime", "instrument"], kind="stable"
    ).reset_index(drop=True)


def fundamental_coverage_report(
    features: pd.DataFrame, *, minimum: float
) -> dict[str, Any]:
    threshold = float(minimum)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("fundamental coverage minimum must be in (0, 1]")
    total = int(len(features))
    families: dict[str, dict[str, Any]] = {}
    for family, columns in FACTOR_FAMILIES.items():
        numeric = features.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
        count = int(valid.sum())
        coverage = count / total if total else 0.0
        families[family] = {
            "columns": list(columns),
            "valid_rows": count,
            "total_rows": total,
            "coverage": coverage,
            "passed": coverage >= threshold,
        }
    passed = total > 0 and all(item["passed"] for item in families.values())
    return {
        "minimum": threshold,
        "total_rows": total,
        "families": families,
        "passed": passed,
        "reason_code": None if passed else "fundamental_coverage_below_threshold",
    }
