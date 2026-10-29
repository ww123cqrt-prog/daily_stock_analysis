"""Universe normalization and immutable-input bundle selection."""
import numpy as np
import json
import pytest

from src.services.script_universe_loader import _normalize_ashare_symbol, load_market_view
from tests.test_qmt_native_backtest import save_input


@pytest.mark.parametrize("value, expected", [("600519", "SH600519"), ("000001", "SZ000001"),
    ("300750", "SZ300750"), ("159915", "SZ159915"), ("sh600519", "SH600519"),
    ("920149", "BJ920149"), ("600519.SH", "SH600519")])
def test_normalize_symbol(value, expected):
    assert _normalize_ashare_symbol(value) == expected


def test_custom_selection_cannot_reuse_another_pools_bundle(tmp_path):
    directory = save_input(tmp_path)
    stock, a = load_market_view("custom", custom_symbols=["600519"], quant_data_root=directory,
                                cache_dir=tmp_path / "bundles")
    etf, b = load_market_view("custom", custom_symbols=["159915"], quant_data_root=directory,
                              cache_dir=tmp_path / "bundles")
    assert a["bundle_dir"] != b["bundle_dir"]
    assert list(stock.symbols) == ["SH600519"]
    assert list(etf.symbols) == ["SZ159915"]


def test_warm_bundle_preserves_same_experiment_and_missing_values(tmp_path):
    directory = save_input(tmp_path, missing=True)
    first, a = load_market_view("etf", quant_data_root=directory, cache_dir=tmp_path / "bundles")
    second, b = load_market_view("etf", quant_data_root=directory, cache_dir=tmp_path / "bundles")
    assert a["experiment_id"] == b["experiment_id"]
    assert b["snapshot_id"] is None
    np.testing.assert_equal(first.close, second.close)


def test_changed_membership_cannot_reuse_a_warm_bundle(tmp_path):
    directory = save_input(tmp_path, universe_kind="historical", membership_intervals=[
        {"symbol": "SH600519", "start_date": "2024-01-02", "end_date": "2024-01-04"},
    ])
    first, a = load_market_view("csi800", quant_data_root=directory, cache_dir=tmp_path / "bundles")
    from quant_qmt_data import load_dataset, save_dataset

    frame = load_dataset(directory)
    metadata = frame.attrs
    changed = save_dataset(frame, tmp_path / "changed-membership", calendar=metadata["calendar"],
        universe=metadata["universe"], universe_kind="historical", membership_intervals=[
            {"symbol": "SH600519", "start_date": "2024-01-02", "end_date": "2024-01-03"},
        ])
    second, b = load_market_view("csi800", quant_data_root=changed, cache_dir=tmp_path / "bundles")
    assert a["bundle_dir"] != b["bundle_dir"]
    assert first.members[:, 0].tolist() == [True, True, True, False]
    assert second.members[:, 0].tolist() == [True, True, False, False]


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
