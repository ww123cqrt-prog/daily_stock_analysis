"""Deterministic JQData SDK stub for managed browser smoke tests."""

from __future__ import annotations

from typing import Any

import pandas as pd


_authenticated = False
_auth_calls: list[tuple[str, str]] = []


def auth(username: str, password: str) -> None:
    global _authenticated
    _auth_calls.append((username, password))
    _authenticated = bool(username and password)


def is_auth() -> bool:
    return _authenticated


def get_query_count() -> dict[str, int]:
    return {"spare": 1000, "total": 1000}


def get_index_stocks(code: str, date: str | None = None) -> list[str]:
    if code == "000905.XSHG":
        return ["000905.XSHG", "000001.XSHE", "600519.XSHG"]
    return ["000001.XSHE", "600519.XSHG", "300750.XSHE"]


def get_all_securities(types: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(index=["000001.XSHE", "600519.XSHG", "300750.XSHE", "000905.XSHG"])


def get_price(
    code: str,
    start_date: str,
    end_date: str,
    frequency: str,
    skip_paused: bool,
    fq: str,
) -> pd.DataFrame:
    del code, start_date, end_date, frequency, skip_paused, fq
    return pd.DataFrame(
        {
            "date": ["2025-03-03", "2025-03-04"],
            "open": [10.0, 10.2],
            "close": [10.1, 10.3],
            "high": [10.4, 10.5],
            "low": [9.8, 10.0],
            "volume": [1000.0, 1100.0],
        }
    )


def __getattr__(name: str) -> Any:
    raise AttributeError(f"Managed smoke JQData stub does not implement {name!r}.")
