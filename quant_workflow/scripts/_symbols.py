"""Shared stock symbol helpers for local Qlib workflow scripts."""

from __future__ import annotations


SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301", "302")
BJ_PREFIXES = ("4", "8")
EXCHANGE_PREFIXES = ("SH", "SZ", "BJ")
KNOWN_SUFFIXES = (".XSHE", ".SZ", ".XSHG", ".SH")


def normalize_symbol(
    symbol: object,
    *,
    strip_any_suffix: bool = False,
    strip_prefix: bool = False,
) -> str:
    """Normalize a stock symbol without assigning an exchange."""
    value = str(symbol).strip().upper()
    if strip_any_suffix and "." in value:
        value = value.split(".", 1)[0]
    elif value.endswith(KNOWN_SUFFIXES):
        value = value.split(".", 1)[0]
    if strip_prefix and value.startswith(EXCHANGE_PREFIXES):
        value = value[2:]
    return value


def to_qlib_name(
    symbol: object,
    *,
    default_exchange: str | None = None,
    strip_any_suffix: bool = False,
    strip_prefix: bool = False,
) -> str:
    """Convert a stock symbol to Qlib's SH/SZ/BJ-prefixed instrument name."""
    value = normalize_symbol(
        symbol,
        strip_any_suffix=strip_any_suffix,
        strip_prefix=strip_prefix,
    )
    if value.startswith(EXCHANGE_PREFIXES):
        return value
    if value.startswith(SH_PREFIXES):
        return f"SH{value}"
    if value.startswith(SZ_PREFIXES):
        return f"SZ{value}"
    if value.startswith(BJ_PREFIXES):
        return f"BJ{value}"
    if default_exchange:
        return f"{default_exchange.upper()}{value}"
    return value
