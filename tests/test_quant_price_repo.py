# -*- coding: utf-8 -*-
"""Tests for the DSA quant_data service price repository."""

from __future__ import annotations

import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from quant_workflow.scripts.quant_data_service_client import QuantDataServiceClient
from src.repositories.quant_price_repo import QuantPriceRepository


def _identity(
    dataset_id: str,
    *,
    data_kind: str,
    snapshot_id: str = "snapshot-1",
) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "manifest_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "latest_trade_date": "2026-08-07",
        "price_basis": "qfq",
        "data_kind": data_kind,
    }


def _arrow_response(table: pa.Table, identity: dict[str, str]) -> httpx.Response:
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


def test_loads_etfs_from_the_formal_paired_daily_dataset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/bars/query"
        payload = json.loads(request.content)
        assert payload["dataset_id"] == "cn_etf_daily"
        assert payload["symbols"] == ["SZ159915", "SH513100"]
        assert payload["basis"] == "qfq"
        assert payload["fields"] == ["open", "close"]
        return _arrow_response(
            pa.table(
                {
                    "symbol": ["SZ159915", "SH513100"],
                    "date": ["2026-08-07", "2026-08-07"],
                    "open": [1.0, 2.0],
                    "close": [1.1, 2.1],
                }
            ),
            _identity("cn_etf_daily", data_kind="paired_daily_bars"),
        )

    result = QuantPriceRepository(client=_client(handler)).load(["159915", "513100"])

    assert result.symbols == ("SZ159915", "SH513100")
    assert result.dataset_id == "cn_etf_daily"
    assert result.snapshot_id == "snapshot-1"
    assert result.price_basis == "qfq"
    assert result.provider == "quant_data"
    assert result.latest_trade_date == "2026-08-07"
    assert result.manifest_sha256 == "a" * 64
    assert result.content_sha256 == "b" * 64
    assert result.data_kind == "paired_daily_bars"
    assert len(result.bars) == 2


def test_tries_ashare_only_when_symbol_is_not_in_the_etf_dataset() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requested.append(str(payload["dataset_id"]))
        if payload["dataset_id"] == "cn_etf_daily":
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "symbol_unavailable",
                        "message": "bar symbol is unavailable",
                    }
                },
            )
        return _arrow_response(
            pa.table(
                {
                    "symbol": ["SH600519"],
                    "date": ["2026-08-04"],
                    "open": [1400.0],
                    "close": [1410.0],
                }
            ),
            _identity("ashare_daily_qfq", data_kind="daily_bars", snapshot_id="ashare-1"),
        )

    result = QuantPriceRepository(client=_client(handler)).load(["600519"])

    assert requested == ["cn_etf_daily", "ashare_daily_qfq"]
    assert result.dataset_id == "ashare_daily_qfq"
    assert result.snapshot_id == "ashare-1"


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

    with pytest.raises(ValueError, match="quant_data data backend"):
        QuantPriceRepository(client=_client(handler)).load(["159915"])

    assert calls == 1


def test_rejects_symbols_missing_from_all_formal_qfq_datasets() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "symbol_unavailable",
                    "message": "bar symbol is unavailable",
                }
            },
        )

    with pytest.raises(ValueError, match="600519"):
        QuantPriceRepository(client=_client(handler)).load(["600519"])


def test_rejects_duplicate_symbols_before_calling_the_service() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        raise AssertionError("service must not be called")

    with pytest.raises(ValueError, match="duplicates"):
        QuantPriceRepository(client=_client(handler)).load(["600519", "SH600519"])

    assert called is False
