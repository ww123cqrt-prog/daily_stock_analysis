import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quant_workflow.tests.test_download_cn_etf_data import load_module, _seed_current_paired_snapshot
from tests.test_miniqmt_fetcher import bar, reply


def test_actual_cli_bootstraps_project_when_launched_from_another_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/quant_data_cli.py"
    result = subprocess.run([sys.executable, "-I", str(script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "update" in result.stdout


def source_transport(calls):
    def transport(request):
        calls.append(request)
        if request["operation"] == "calendar":
            return reply(request, {m: {"dates": ["2026-08-07"]} for m in request["symbols"]})
        assert request["operation"] == "update_pair"
        rows = [bar(d, c) for d, c in zip(["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"], [1, 1.1, 1.2, 1.3])]
        for r in rows:
            r["volume"] = 100
            r["high"] = r["close"] * 1.01
            r["low"] = r["close"] * 0.99
        return reply(request, {s: {"raw": rows, "qfq": rows, "actions": [],
                                   "calendar": [r["date"] for r in rows]} for s in request["symbols"]})
    return transport


def test_default_etf_update_uses_batched_miniqmt_and_replays_sealed_inputs(tmp_path, monkeypatch):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    module = load_module()
    import miniqmt_etf_source
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls = []
    network_allowed = True
    provider = source_transport(calls)

    def guarded(request):
        assert network_allowed, "sealed inputs must resume without reconnecting HOME"
        return provider(request)
    monkeypatch.setattr(miniqmt_etf_source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=guarded))
    prepared = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07", publish=False)
    assert [r["symbols"] for r in calls if r["operation"] == "update_pair"] == [["510050.SH", "159915.SZ"]]
    assert (root / "parquet/etf_1d/cn_etf_daily/LATEST.json").is_file()
    assert json.loads((prepared / "DATASET_MANIFEST.json").read_text())["source"]["increment_provider"] == "miniqmt"
    calls.clear()
    network_allowed = False
    new = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07")
    assert not [r for r in calls if r["operation"] == "update_pair"]
    assert new != old
    for symbol in ["SH510050", "SZ159915"]:
        before = pd.read_parquet(old / "raw" / f"{symbol}.parquet")
        after = pd.read_parquet(new / "raw" / f"{symbol}.parquet")
        assert len(after) == len(before) + 1
        pd.testing.assert_frame_equal(after.iloc[:len(before)][before.columns].reset_index(drop=True), before, check_dtype=False)
    assert list((new / "source_receipts/miniqmt").glob("*.json"))


def test_miniqmt_shared_source_receipt_tampering_is_rejected(tmp_path, monkeypatch):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    module = load_module()
    import miniqmt_etf_source
    root, _ = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    monkeypatch.setattr(miniqmt_etf_source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=source_transport([])))
    module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07", publish=False)
    raw_response = next((root / "cache/cn-etf-incremental/miniqmt-responses").glob("*.json"))
    raw_response.write_text("{}")
    with pytest.raises(Exception, match="response.*(changed|digest)|source.*changed"):
        module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07")


def test_actual_cli_updates_temp_snapshot_through_default_source(tmp_path, monkeypatch, capsys):
    import quant_qmt_data as provider
    from quant_qmt_data import miniqmt_etf_source as source
    from quant_qmt_data.miniqmt_bridge import MiniQMTBridgeClient
    from quant_workflow.tests.test_quant_data_cli import load_module as load_cli
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    previous = provider.read_bars("cn_etf_daily", "SH510050", quant_data_root=root, price_basis="qfq")
    old_files = {path: path.read_bytes() for path in old.rglob("*") if path.is_file()}
    calls = []
    network_allowed = True
    days = ["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"]

    def transport(request):
        assert network_allowed, "the committed update must replay without reconnecting MiniQMT"
        calls.append(request)
        selected = [day for day in days
                    if pd.Timestamp(request["start"]) <= pd.Timestamp(day) <= pd.Timestamp(request["end"])]
        if request["operation"] == "universe":
            data = {asset: {"symbols": ["510050.SH", "159915.SZ"], "status": "ok"}
                    for asset in request["symbols"]}
        elif request["operation"] == "instruments":
            data = {symbol: {"name": symbol, "open_date": days[0], "expire_date": None,
                             "security_types": {"stock": False, "etf": True}}
                    for symbol in request["symbols"]}
        elif request["operation"] == "calendar":
            data = {market: {"dates": selected} for market in request["symbols"]}
        else:
            assert request["operation"] == "update_pair"
            rows = [dict(bar(day, close), high=close * 1.01, low=close * .99, volume=100)
                    for day, close in zip(days, [1., 1.1, 1.2, 1.3]) if day in selected]
            data = {symbol: dict(raw=rows, qfq=rows, actions=[], calendar=selected,
                                 volume_unit="lots", action_history_through=days[-1])
                    for symbol in request["symbols"]}
        return reply(request, data)

    monkeypatch.setattr(source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=transport))
    cli = load_cli()
    command = ["--root", str(root), "update", "cn_etf_daily", "--through", days[-1]]
    assert cli.main(command) == 0
    pointer_path = root / "parquet/etf_1d/cn_etf_daily/LATEST.json"
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    assert pointer["snapshot_id"] != old.name
    assert {"universe", "calendar", "update_pair"} <= {request["operation"] for request in calls}
    updated = provider.read_bars("cn_etf_daily", "SH510050", quant_data_root=root, price_basis="qfq")
    assert updated["close"].tolist() == [1., 1.1, 1.2, 1.3]
    columns = ["date", "open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(updated.iloc[:len(previous)][columns].reset_index(drop=True), previous[columns])
    assert all(path.read_bytes() == body for path, body in old_files.items())
    calls.clear()
    network_allowed = False
    assert cli.main(command) == 0
    assert calls == []
    assert pointer_path.read_bytes() == pointer_bytes
    pd.testing.assert_frame_equal(
        provider.read_bars("cn_etf_daily", "SH510050", quant_data_root=root, price_basis="qfq"), updated)


def test_bad_symbol_keeps_old_history_while_healthy_symbol_updates(tmp_path, monkeypatch):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    module = load_module()
    import miniqmt_etf_source
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    original = source_transport([])

    def transport(request):
        response = original(request)
        if request["operation"] == "update_pair":
            response["data"]["510050.SH"]["raw"] = []
            return reply(request, response["data"])
        return response
    monkeypatch.setattr(miniqmt_etf_source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=transport))
    new = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07")
    assert (old / "raw/SH510050.parquet").read_bytes() == (new / "raw/SH510050.parquet").read_bytes()
    assert len(pd.read_parquet(new / "raw/SZ159915.parquet")) == 4
    status = json.loads((root / "runtime/status/cn_etf_daily_update_latest.json").read_text())
    assert status["status"] == "published_partial"
    assert list(status["symbol_update_failures"]) == ["SH510050"]


def test_historical_action_revision_outside_overlap_is_isolated_on_next_update(tmp_path, monkeypatch):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    from quant_workflow.tests import test_download_cn_etf_data as helpers
    module = load_module()
    import miniqmt_etf_source
    old_factory = helpers.paired_frame

    # Avoid the helper's ``dates or ...`` ambiguity for a DatetimeIndex.
    def seed(symbol, **kwargs):
        days = pd.bdate_range("2026-07-01", "2026-08-06").strftime("%Y-%m-%d").tolist()
        return old_factory(symbol, name=kwargs["name"], source=kwargs["source"],
                           closes=[1 + i * .01 for i in range(len(days))], dates=days)
    monkeypatch.setattr(helpers, "paired_frame", seed)
    root, _ = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    revised = False

    def transport(request):
        days = pd.bdate_range(pd.Timestamp(request["start"]), pd.Timestamp(request["end"]))
        if request["operation"] == "calendar":
            return reply(request, {m: {"dates": days.strftime("%Y-%m-%d").tolist()} for m in request["symbols"]})
        rows = []
        origin = pd.bdate_range("2026-07-01", "2026-08-10").strftime("%Y-%m-%d").tolist()
        for day in days.strftime("%Y-%m-%d"):
            c = 1 + origin.index(day) * .01
            rows.append(dict(bar(day, c), high=c * 1.01, low=c * .99, volume=100))
        data = {s: {"raw": rows, "qfq": rows, "actions": [], "calendar": [r["date"] for r in rows]}
                for s in request["symbols"]}
        if revised:
            data["510050.SH"]["actions"] = [{"date": "2026-07-15", "interest": .1, "stockBonus": 0,
                                            "stockGift": 0, "allotNum": 0, "allotPrice": 0, "gugai": 0, "dr": 1}]
        return reply(request, data)
    monkeypatch.setattr(miniqmt_etf_source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=transport))
    first = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07")
    revised = True
    second = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-10")
    assert (first / "raw/SH510050.parquet").read_bytes() == (second / "raw/SH510050.parquet").read_bytes()
    assert pd.to_datetime(pd.read_parquet(second / "raw/SZ159915.parquet")["date"]).max() == pd.Timestamp("2026-08-10")
    status = json.loads((root / "runtime/status/cn_etf_daily_update_latest.json").read_text())
    assert "historical cash/share events" in status["symbol_update_failures"]["SH510050"]["error"]


def test_nontrading_target_uses_calendar_without_downloading_existing_history(tmp_path, monkeypatch):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    module = load_module()
    import miniqmt_etf_source
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls = []

    def transport(request):
        calls.append(request)
        assert request["operation"] == "calendar"
        return reply(request, {m: {"dates": ["2026-08-06"]} for m in request["symbols"]})
    monkeypatch.setattr(miniqmt_etf_source, "MiniQMTBridgeClient", lambda: MiniQMTBridgeClient(transport=transport))
    assert module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07") == old
    assert len(calls) == 1
