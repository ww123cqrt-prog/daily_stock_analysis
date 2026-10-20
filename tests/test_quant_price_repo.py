# -*- coding: utf-8 -*-
"""Tests for the DSA quant_data service price repository."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from quant_workflow.scripts.quant_data_service_client import (
    NativeDatasetIdentity,
    NativeDatasetResult,
    NativeReleaseIdentity,
    QuantDataClientError,
    QuantDataServiceClient,
)
from src.repositories.quant_price_repo import QuantPriceRepository


@pytest.mark.parametrize("symbol,expected", [
    ("920748", "BJ920748"), ("BJ920748", "BJ920748"), ("920748.BJ", "BJ920748"),
    ("600519", "SH600519"), ("600519.SH", "SH600519"), ("000001", "SZ000001"),
    ("159915", "SZ159915"), ("900901", "SH900901"), ("830799", "BJ830799"),
])
def test_price_symbols_share_exchange_rules(symbol, expected):
    from src.repositories.quant_price_repo import normalize_market_symbol
    assert normalize_market_symbol(symbol) == expected


def _arrow_response(table: pa.Table, identity: dict) -> httpx.Response:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return httpx.Response(
        200,
        content=sink.getvalue().to_pybytes(),
        headers={"X-Quant-Data-Metadata": json.dumps(identity)},
    )


def _client(handler) -> QuantDataServiceClient:
    return QuantDataServiceClient(
        base_url="http://127.0.0.1:8766",
        timeout=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_does_not_fallback_when_the_data_service_is_unavailable() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "snapshot_changed",
                    "message": "snapshot changed",
                }
            },
        )

    with pytest.raises(QuantDataClientError, match="snapshot changed") as error:
        QuantPriceRepository(client=_client(handler)).load(
            ["159915"], start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
        )

    assert error.value.code == "snapshot_changed"
    assert calls == 1


def test_requires_bounded_dates_before_contacting_service() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        raise AssertionError("service must not be called")

    with pytest.raises(ValueError, match="start and end date"):
        QuantPriceRepository(client=_client(handler)).load(["159915"])
    assert called is False


def test_rejects_duplicate_symbols_before_calling_the_service() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        raise AssertionError("service must not be called")

    with pytest.raises(ValueError, match="duplicates"):
        QuantPriceRepository(client=_client(handler)).load(
            ["600519", "SH600519"],
            start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
        )

    assert called is False


def test_loads_mixed_symbols_from_active_v2_with_strict_coverage() -> None:
    release = {
        "authority": "active", "dataset_set_version": "production.v2",
        "generation": 10, "pointer_sha256": "a" * 64,
        "release_set_sha256": "b" * 64,
    }
    symbols_by_module = {
        "cn_etf_daily": ["SZ159915"],
        "ashare_daily_raw": ["SH600519"],
    }
    requested = []

    def metadata(module: str) -> dict:
        return {
            **release, "dataset_id": module, "module_id": module,
            "module_release_sha256": "c" * 64,
            "content_manifest_sha256": "d" * 64,
            "requested_through": "2026-08-21",
            "component_identities": {"qfq": "e" * 64},
            "availability": {module: "AVAILABLE"},
        }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={
                **release, "status": "ready", "validated_datasets": 2,
                "datasets": [
                    {"dataset_id": module, "module_id": module, "status": "ready",
                     "availability": "AVAILABLE"}
                    for module in symbols_by_module
                ],
            })
        if request.url.path.startswith("/v1/datasets/"):
            module = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json={
                "metadata": metadata(module),
                "coverage": {"symbols": symbols_by_module[module]},
            })
        assert request.url.path == "/v1/bars/query"
        body = json.loads(request.content)
        requested.append(body)
        assert body["mode"] == "strict"
        assert body["start"] == "2026-08-20"
        assert body["end"] == "2026-08-21"
        assert body["requested_through"] == "2026-08-21"
        module = body["dataset_id"]
        symbol = symbols_by_module[module][0]
        return _arrow_response(
            pa.table({
                "symbol": [symbol], "date": ["2026-08-21"],
                "open": [1.0], "close": [1.1],
            }),
            {
                **metadata(module), "basis": "qfq",
                "staleness": {
                    "mode": "strict", "complete": True, "uncovered": [],
                    "requested_through": "2026-08-21",
                    "summary": {"included_symbol_count": 1, "excluded_symbol_count": 0},
                },
            },
        )

    result = QuantPriceRepository(client=_client(handler)).load(
        ["159915", "600519"],
        start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
    )

    assert {item["dataset_id"] for item in requested} == set(symbols_by_module)
    assert result.dataset_set_version == "production.v2"
    assert result.release_set_sha256 == "b" * 64
    assert result.snapshot_id is None
    assert {bar.symbol for bar in result.bars} == {"SZ159915", "SH600519"}


def test_strict_coverage_conflict_is_preserved() -> None:
    release = NativeReleaseIdentity(10, "a" * 64, "b" * 64)
    dataset = NativeDatasetIdentity(
        "cn_etf_daily", release, "c" * 64, "d" * 64,
        "2026-08-21", {"qfq": "e" * 64},
    )

    def reject(*_args, **_kwargs):
        raise QuantDataClientError(
            "strict coverage unavailable", code="stale_coverage", status_code=409,
        )

    client = SimpleNamespace(
        native_ready=lambda: release,
        native_dataset=lambda *_args, **_kwargs: NativeDatasetResult(dataset, ("SZ159915",)),
        native_bars_batch=reject,
    )
    with pytest.raises(QuantDataClientError) as error:
        QuantPriceRepository(client=client).load(
            ["159915"], start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
        )
    assert error.value.code == "stale_coverage"
    assert error.value.status_code == 409


def test_rejects_symbols_missing_from_active_modules() -> None:
    release = NativeReleaseIdentity(10, "a" * 64, "b" * 64)
    dataset = NativeDatasetIdentity(
        "cn_etf_daily", release, "c" * 64, "d" * 64,
        "2026-08-21", {"qfq": "e" * 64},
    )
    client = SimpleNamespace(
        native_ready=lambda: release,
        native_dataset=lambda *_args, **_kwargs: NativeDatasetResult(dataset, ()),
    )
    with pytest.raises(ValueError, match="SH600519"):
        QuantPriceRepository(client=client).load(
            ["600519"], start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
        )


def test_rejects_invalid_rows_in_strict_response() -> None:
    release = NativeReleaseIdentity(10, "a" * 64, "b" * 64)
    dataset = NativeDatasetIdentity(
        "cn_etf_daily", release, "c" * 64, "d" * 64,
        "2026-08-21", {"qfq": "e" * 64},
    )
    frame = pd.DataFrame({
        "symbol": ["SZ159915"], "date": ["2026-08-21"],
        "open": [None], "close": [1.1],
    })
    with pytest.raises(ValueError, match="invalid price bars"):
        QuantPriceRepository._normalize_result(
            ("SZ159915",), frame, release, {"cn_etf_daily": dataset},
        )


def test_large_symbol_list_is_queried_in_bounded_batches() -> None:
    release = NativeReleaseIdentity(10, "a" * 64, "b" * 64)
    dataset = NativeDatasetIdentity(
        "cn_etf_daily", release, "c" * 64, "d" * 64,
        "2026-08-21", {"qfq": "e" * 64},
    )
    codes = [f"159{index:03d}" for index in range(55)]
    symbols = tuple(f"SZ{code}" for code in codes)
    chunks = []

    def read_bars(_identity, selected, **_kwargs):
        chunks.append(tuple(selected))
        return SimpleNamespace(frame=pd.DataFrame({
            "symbol": selected, "date": ["2026-08-21"] * len(selected),
            "open": [1.0] * len(selected), "close": [1.1] * len(selected),
        }))

    client = SimpleNamespace(
        native_ready=lambda: release,
        native_dataset=lambda *_args, **_kwargs: NativeDatasetResult(dataset, symbols),
        native_bars_batch=read_bars,
    )
    result = QuantPriceRepository(client=client).load(
        codes, start_date=date(2026, 8, 20), end_date=date(2026, 8, 21),
    )
    assert [len(chunk) for chunk in chunks] == [50, 5]
    assert len(result.bars) == 55
