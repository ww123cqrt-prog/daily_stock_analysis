from __future__ import annotations

import io
import json

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from quant_workflow.scripts.quant_data_service_client import (
    NativeDatasetIdentity,
    NativeReleaseIdentity,
    QuantDataClientError,
    QuantDataIdentityError,
    QuantDataServiceClient,
    QuantDataSnapshotChanged,
)


def _identity(*, dataset_id: str = "cn_etf_daily", snapshot_id: str = "snapshot-1") -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "manifest_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "latest_trade_date": "2026-08-07",
        "price_basis": "qfq",
        "data_kind": "paired_daily_bars",
    }


def _arrow_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _client(handler) -> QuantDataServiceClient:
    return QuantDataServiceClient(
        base_url="http://127.0.0.1:8766",
        transport=httpx.MockTransport(handler),
        timeout=1.0,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8766",
        "http://quant-data.example:8766",
        "http://user:password@127.0.0.1:8766",
        "http://127.0.0.1:8766/v1",
    ],
)
def test_client_rejects_non_local_or_credentialed_service_urls(url: str) -> None:
    with pytest.raises(ValueError, match="local quant_data service URL"):
        QuantDataServiceClient(base_url=url)


def test_query_bars_batch_decodes_arrow_and_returns_complete_identity() -> None:
    table = pa.table(
        {
            "symbol": ["SZ159915", "SH513100"],
            "date": ["2026-08-07", "2026-08-07"],
            "open": [1.2, 2.3],
            "close": [1.3, 2.4],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/bars/query"
        assert json.loads(request.content) == {
            "dataset_id": "cn_etf_daily",
            "symbols": ["SZ159915", "SH513100"],
            "basis": "qfq",
            "fields": ["open", "close"],
            "start": "2026-08-07",
            "end": "2026-08-07",
        }
        return httpx.Response(
            200,
            content=_arrow_bytes(table),
            headers={
                "Content-Type": "application/vnd.apache.arrow.stream",
                "X-Quant-Data-Metadata": json.dumps(_identity()),
            },
        )

    result = _client(handler).query_bars_batch(
        "cn_etf_daily",
        ["SZ159915", "SH513100"],
        basis="qfq",
        fields=["open", "close"],
        start="2026-08-07",
        end="2026-08-07",
    )

    assert result.frame.columns.tolist() == ["symbol", "date", "open", "close"]
    assert result.frame["close"].tolist() == [1.3, 2.4]
    assert result.identity.as_dict() == _identity()


def test_partitioned_artifact_query_locks_identity_across_requests() -> None:
    table = pa.table(
        {
            "date": ["2025-01-02"],
            "symbol": ["SH600000"],
            "index_code": ["SH000300"],
        }
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/artifacts/pit_csi800/daily"
        assert request.url.params["start"] == "2025-01-02"
        assert request.url.params["end"] == "2025-01-03"
        if calls == 1:
            assert request.url.params["fields"] == "index_code"
            assert request.url.params["symbols"] == "SH600000"
        else:
            assert "fields" not in request.url.params
            assert "symbols" not in request.url.params
        metadata = _identity(dataset_id="pit_csi800", snapshot_id=f"snapshot-{calls}")
        metadata["price_basis"] = "not_applicable"
        metadata["data_kind"] = "point_in_time_membership"
        return httpx.Response(
            200,
            content=_arrow_bytes(table),
            headers={"X-Quant-Data-Metadata": json.dumps(metadata)},
        )

    client = _client(handler)
    first = client.query_partitioned_artifact(
        "pit_csi800",
        "daily",
        start="2025-01-02",
        end="2025-01-03",
        fields=["index_code"],
        symbols=["SH600000"],
    )

    with pytest.raises(QuantDataSnapshotChanged, match="pit_csi800"):
        client.query_partitioned_artifact(
            "pit_csi800",
            "daily",
            start="2025-01-02",
            end="2025-01-03",
            expected_identity=first.identity,
        )


def test_single_bar_json_and_file_artifact_preserve_identity() -> None:
    membership = b"parquet-content"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/bars/ashare_daily_qfq/SH600000":
            return httpx.Response(
                200,
                json={
                    "metadata": _identity(
                        dataset_id="ashare_daily_qfq", snapshot_id="ashare-1"
                    ),
                    "columns": ["date", "close"],
                    "data": [{"date": "2026-08-04", "close": 10.5}],
                },
            )
        assert request.url.path == "/v1/artifacts/pit_csi800/membership_intervals"
        metadata = _identity(dataset_id="pit_csi800", snapshot_id="pit-1")
        metadata["price_basis"] = "not_applicable"
        metadata["data_kind"] = "point_in_time_membership"
        return httpx.Response(
            200,
            content=membership,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Quant-Data-Metadata": json.dumps(metadata),
            },
        )

    client = _client(handler)
    bars = client.query_bars(
        "ashare_daily_qfq", "SH600000", basis="qfq", fields=["close"]
    )
    artifact = client.get_artifact("pit_csi800", "membership_intervals")

    assert bars.frame.to_dict("records") == [{"date": "2026-08-04", "close": 10.5}]
    assert bars.identity.dataset_id == "ashare_daily_qfq"
    assert artifact.content == membership
    assert artifact.identity.dataset_id == "pit_csi800"


def test_client_fails_closed_for_service_errors_and_missing_identity() -> None:
    responses = iter(
        [
            httpx.Response(
                503,
                json={"error": {"code": "snapshot_changed", "message": "changed"}},
            ),
            httpx.Response(200, content=_arrow_bytes(pa.table({"date": []}))),
        ]
    )
    client = _client(lambda _request: next(responses))

    with pytest.raises(QuantDataClientError) as service_error:
        client.query_bars_batch("cn_etf_daily", ["SZ159915"], basis="qfq")
    assert service_error.value.code == "snapshot_changed"
    assert service_error.value.status_code == 503

    with pytest.raises(QuantDataIdentityError, match="metadata"):
        client.query_bars_batch("cn_etf_daily", ["SZ159915"], basis="qfq")


def test_ready_describe_and_resolve_return_validated_dataset_identities() -> None:
    etf_identity = _identity()
    ashare_identity = _identity(
        dataset_id="ashare_daily_qfq", snapshot_id="ashare-1"
    )
    ashare_identity["data_kind"] = "daily_bars"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "validated_datasets": 2,
                    "datasets": [
                        {**etf_identity, "role": "canonical", "status": "ready"},
                        {
                            **ashare_identity,
                            "role": "canonical",
                            "status": "ready",
                        },
                    ],
                },
            )
        if request.url.path == "/v1/datasets/cn_etf_daily/resolve":
            assert request.url.params["basis"] == "qfq"
            return httpx.Response(
                200,
                json={
                    "status": "ready",
                    "metadata": {**etf_identity, "validated": True},
                    "source": {"provider": "tencent"},
                },
            )
        assert request.url.path == "/v1/datasets/cn_etf_daily"
        return httpx.Response(
            200,
            json={
                "metadata": etf_identity,
                "components": {"raw": {"kind": "partitioned_parquet"}},
            },
        )

    client = _client(handler)

    ready = client.ready()
    described = client.describe_dataset("cn_etf_daily")
    resolved = client.resolve_dataset(
        "cn_etf_daily", basis="qfq", expected_identity=described.identity
    )

    assert ready.status == "ready"
    assert ready.validated_datasets == 2
    assert [identity.dataset_id for identity in ready.identities] == [
        "cn_etf_daily",
        "ashare_daily_qfq",
    ]
    assert described.identity == resolved.identity
    assert described.payload["components"]["raw"]["kind"] == "partitioned_parquet"
    assert resolved.payload["source"]["provider"] == "tencent"


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "starting", "validated_datasets": 0, "datasets": []},
        {"status": "ready", "validated_datasets": 2, "datasets": [_identity()]},
    ],
)
def test_ready_fails_closed_for_an_inconsistent_response(payload) -> None:
    client = _client(lambda _request: httpx.Response(200, json=payload))

    with pytest.raises(QuantDataClientError, match="readiness response"):
        client.ready()


def _native_metadata(dataset_id: str = "cn_etf_daily") -> dict:
    return {
        "authority": "active",
        "dataset_set_version": "production.v2",
        "generation": 10,
        "pointer_sha256": "a" * 64,
        "release_set_sha256": "b" * 64,
        "dataset_id": dataset_id,
        "module_id": dataset_id,
        "module_release_sha256": "c" * 64,
        "content_manifest_sha256": "d" * 64,
        "requested_through": "2026-09-11",
        "component_identities": {"qfq": "e" * 64, "raw": "f" * 64},
    }


def test_native_bars_lock_release_and_require_strict_coverage() -> None:
    meta = _native_metadata()
    release = NativeReleaseIdentity.from_mapping(meta)
    dataset = NativeDatasetIdentity.from_mapping(meta)
    table = pa.table({
        "symbol": ["SH513100"], "date": ["2026-09-11"],
        "open": [2.21], "close": [2.20], "volume": [100.0],
    })
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/bars/query"
        body = json.loads(request.content)
        assert body["mode"] == "strict"
        assert body["requested_through"] == "2026-09-11"
        returned = dict(meta)
        returned.update({
            "basis": "qfq",
            "staleness": {
                "mode": "strict", "complete": True, "uncovered": [],
                "requested_through": "2026-09-11",
                "summary": {"included_symbol_count": 1, "excluded_symbol_count": 0},
            },
        })
        if calls == 2:
            returned["release_set_sha256"] = "1" * 64
        if calls == 3:
            returned["staleness"]["complete"] = False
        return httpx.Response(200, content=_arrow_bytes(table), headers={
            "X-Quant-Data-Metadata": json.dumps(returned),
        })

    client = _client(handler)
    result = client.native_bars_batch(
        dataset, ["SH513100"], basis="qfq",
        fields=["open", "close", "volume"],
        start="2026-09-10", end="2026-09-11",
    )
    assert result.identity.release == release
    assert result.frame["close"].tolist() == [2.20]
    with pytest.raises(QuantDataSnapshotChanged):
        client.native_bars_batch(
            dataset, ["SH513100"], basis="qfq", fields=["close"],
            start="2026-09-10", end="2026-09-11",
        )
    with pytest.raises(QuantDataClientError, match="coverage is incomplete"):
        client.native_bars_batch(
            dataset, ["SH513100"], basis="qfq", fields=["close"],
            start="2026-09-10", end="2026-09-11",
        )


def test_native_ready_and_dataset_use_production_v2_identity() -> None:
    meta = _native_metadata()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ready":
            return httpx.Response(200, json={
                **meta, "status": "ready", "validated_datasets": 1,
                "datasets": [{"dataset_id": "cn_etf_daily", "module_id": "cn_etf_daily",
                              "status": "ready", "availability": "AVAILABLE"}],
            })
        assert request.url.path == "/v1/datasets/cn_etf_daily"
        return httpx.Response(200, json={
            "metadata": {**meta, "availability": {"cn_etf_daily": "AVAILABLE"}},
            "coverage": {"symbols": ["SH513100", "SZ159915"]},
        })

    client = _client(handler)
    release = client.native_ready()
    readiness = client.ready()
    dataset = client.native_dataset("cn_etf_daily", expected_release=release)
    assert readiness.native_release == release
    assert readiness.datasets[0]["module_release_sha256"] == "c" * 64
    assert dataset.symbols == ("SH513100", "SZ159915")
    assert dataset.identity.component_identities["qfq"] == "e" * 64
