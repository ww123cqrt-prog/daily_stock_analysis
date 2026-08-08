"""Shared minute-bar schema for limit-up data production and research."""

from __future__ import annotations

import pandas as pd


MINUTE_COLUMNS = {
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "high_limit",
    "low_limit",
    "pre_close",
}
PRICE_TOLERANCE = 0.005


def normalize_minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one chronological trading day of validated minute bars."""

    missing = MINUTE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"minute bars are missing columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    numeric = sorted(MINUTE_COLUMNS - {"datetime"})
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["datetime", *numeric]].isna().any().any():
        raise ValueError("minute bars contain invalid values")
    result = result.sort_values("datetime", kind="stable").reset_index(drop=True)
    if result["datetime"].duplicated().any():
        raise ValueError("minute bars contain duplicate timestamps")
    dates = result["datetime"].dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError("minute bars must contain exactly one trading day")
    prices = ["open", "high", "low", "close", "high_limit", "pre_close"]
    if (result[prices] <= 0).any().any():
        raise ValueError("minute bars contain nonpositive prices")
    if (result[["volume", "money"]] < 0).any().any():
        raise ValueError("minute bars contain negative volume or money")
    return result
