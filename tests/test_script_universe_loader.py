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


def test_explicit_legacy_loader_reads_strict_production_v2_service(monkeypatch, tmp_path) -> None:
    import json

    import httpx
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from quant_workflow.scripts.quant_data_service_client import QuantDataServiceClient
    from src.services import production_v2_market_loader
    from src.services.script_universe_loader import load_market_view

    release = {
        "authority": "active", "dataset_set_version": "production.v2",
        "generation": 10, "pointer_sha256": "a" * 64,
        "release_set_sha256": "b" * 64,
    }

    def metadata(name: str) -> dict:
        return {
            **release, "dataset_id": name, "module_id": name,
            "module_release_sha256": ("c" if name == "cn_etf_daily" else "d") * 64,
            "content_manifest_sha256": "e" * 64,
            "requested_through": "2026-08-21",
            "component_identities": {"qfq": "f" * 64},
            "availability": {"cn_etf_daily": "AVAILABLE", "ashare_daily_raw": "AVAILABLE"},
        }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={
                **release, "status": "ready", "validated_datasets": 2,
                "datasets": [
                    {"dataset_id": name, "module_id": name, "status": "ready",
                     "availability": "AVAILABLE"}
                    for name in ("ashare_daily_raw", "cn_etf_daily")
                ],
            })
        if request.url.path.startswith("/v1/datasets/"):
            name = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={
                "metadata": metadata(name),
                "coverage": {"symbols": ["SH513100", "SZ159915"] if name == "cn_etf_daily" else []},
            })
        assert request.url.path == "/v1/bars/query"
        body = json.loads(request.content)
        assert body["dataset_id"] == "cn_etf_daily"
        assert body["mode"] == "strict"
        assert body["symbols"] == ["SH513100", "SZ159915"]
        table = pa.table({
            "symbol": ["SH513100", "SZ159915", "SH513100", "SZ159915"],
            "date": ["2026-08-20", "2026-08-20", "2026-08-21", "2026-08-21"],
            "open": [2.2, 3.1, 2.3, 3.2],
            "close": [2.25, 3.15, 2.35, 3.25],
            "volume": [100.0, 200.0, 110.0, 220.0],
        })
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return httpx.Response(200, content=sink.getvalue().to_pybytes(), headers={
            "X-Quant-Data-Metadata": json.dumps({
                **metadata("cn_etf_daily"), "basis": "qfq",
                "staleness": {
                    "mode": "strict", "complete": True, "uncovered": [],
                    "requested_through": "2026-08-21",
                    "summary": {"included_symbol_count": 2, "excluded_symbol_count": 0},
                },
            }),
        })

    monkeypatch.setattr(
        production_v2_market_loader, "QuantDataServiceClient",
        lambda **_kwargs: QuantDataServiceClient(transport=httpx.MockTransport(handler)),
    )
    view, meta = production_v2_market_loader.load_native_market_view(
        "custom", custom_symbols=["159915", "513100"],
        start="2026-08-20", end="2026-08-21", cache_dir=tmp_path,
    )
    assert meta["release_set_sha256"] == "b" * 64
    assert meta["dataset_id"] == "cn_etf_daily"
    assert view.close_at("SH513100", 1) == 2.35
    assert view.volume_at("SZ159915", 1) == 220.0
