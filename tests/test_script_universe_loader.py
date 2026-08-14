# -*- coding: utf-8 -*-
"""Unit tests for universe pool symbol normalization."""

from src.services.script_universe_loader import _normalize_ashare_symbol


def test_normalize_ashare_symbol_adds_exchange_prefix():
    assert _normalize_ashare_symbol("600519") == "SH600519"
    assert _normalize_ashare_symbol("000001") == "SZ000001"
    assert _normalize_ashare_symbol("300750") == "SZ300750"
    assert _normalize_ashare_symbol("688981") == "SH688981"


def test_normalize_ashare_symbol_keeps_prefixed_and_lowercase():
    assert _normalize_ashare_symbol("sh600519") == "SH600519"
    assert _normalize_ashare_symbol("SZ000001") == "SZ000001"


def test_normalize_ashare_symbol_passes_unknown_forms_through():
    assert _normalize_ashare_symbol("159915") == "159915"


from src.services.script_universe_loader import _pool_cache_key


def test_pool_cache_key_includes_custom_symbols() -> None:
    """C-1: different custom symbol lists must never share a cached bundle."""
    base = _pool_cache_key("custom", "snap", "2024-01-01", "2024-06-30")
    with_a = _pool_cache_key(
        "custom", "snap", "2024-01-01", "2024-06-30",
        custom_symbols=["600519"],
    )
    with_b = _pool_cache_key(
        "custom", "snap", "2024-01-01", "2024-06-30",
        custom_symbols=["000001"],
    )
    assert with_a != base
    assert with_a != with_b
    # Symbol order must not matter.
    assert with_a == _pool_cache_key(
        "custom", "snap", "2024-01-01", "2024-06-30",
        custom_symbols=["600519"],
    )


def test_pool_cache_key_includes_membership_sha() -> None:
    """M-4: membership reference edits must invalidate the cache."""
    base = _pool_cache_key("csi800", "snap", None, None)
    assert base != _pool_cache_key(
        "csi800", "snap", None, None, membership_sha256="a" * 64
    )
    assert base != _pool_cache_key(
        "csi800", "snap", None, None, membership_sha256="b" * 64
    )


def test_pool_cache_key_includes_secondary_etf_snapshot() -> None:
    """P1: an ETF republish must invalidate mixed custom-pool caches even
    when the A-share snapshot id is unchanged."""
    base = _pool_cache_key(
        "custom", "ashare-snap", "2024-01-01", "2024-06-30",
        custom_symbols=["600519", "159915"],
    )
    with_etf_a = _pool_cache_key(
        "custom", "ashare-snap", "2024-01-01", "2024-06-30",
        custom_symbols=["600519", "159915"],
        secondary_snapshot_id="etf-snap-1",
    )
    with_etf_b = _pool_cache_key(
        "custom", "ashare-snap", "2024-01-01", "2024-06-30",
        custom_symbols=["600519", "159915"],
        secondary_snapshot_id="etf-snap-2",
    )
    assert with_etf_a != base
    assert with_etf_a != with_etf_b
