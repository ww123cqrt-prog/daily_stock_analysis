"""MiniQMT uses the existing manager contract, with no trading API or live SDK in tests."""
import hashlib
import json

import pandas as pd
import pytest


def reply(request, data):
    def encode(value):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {"status": "ok", "provider": "miniqmt", "request": request,
            "request_sha256": hashlib.sha256(encode(request)).hexdigest(),
            "data": data, "data_sha256": hashlib.sha256(encode(data)).hexdigest(),
            "mode": "read_existing_cache", "network_download_requested": False}


def bar(day, close=10.0):
    return {"date": day, "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 12, "amount": 12000, "suspendFlag": 0}


def test_client_rejects_changed_response_and_wrong_request():
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient, MiniQMTError

    def transport(request):
        result = reply(request, {"600000.SH": {"rows": [bar("2026-10-16")]}})
        result["data"]["600000.SH"]["rows"][0]["close"] = 999
        return result
    with pytest.raises(MiniQMTError, match="digest"):
        MiniQMTBridgeClient(transport=transport).request("bars", ["600000.SH"], "2026-10-16", "2026-10-16")


def test_daily_client_keeps_exact_window_and_source_receipt():
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    calls = []
    rows = [bar("2026-10-14"), bar("2026-10-15"), bar("2026-10-16")]

    def transport(request):
        calls.append(request.copy())
        assert request["operation"] == "update_pair"
        assert request["start"] == "20261014" and request["end"] == "20261016"
        return reply(request, {"600000.SH": {"raw": rows, "qfq": rows, "actions": [],
                                             "calendar": [r["date"] for r in rows]}})
    frames, receipts = MiniQMTBridgeClient(transport=transport).daily(
        ["600000.SH"], "2026-10-14", "2026-10-16")
    assert frames["600000.SH"]["date"].tolist() == ["2026-10-14", "2026-10-15", "2026-10-16"]
    assert len(calls) == len(receipts) == 1
    assert receipts[0]["request"] == calls[0]


def test_stock_fetcher_converts_lots_to_shares_and_excludes_suspended_rows():
    from data_provider.miniqmt_fetcher import MiniQMTFetcher

    class Client:
        def daily(self, symbols, start, end, **kwargs):
            assert symbols == ["600000.SH"]
            assert kwargs["adjustment"] == "front"
            return {symbols[0]: pd.DataFrame([bar("2026-10-14"), dict(bar("2026-10-15"), suspendFlag=1), bar("2026-10-16", 11)])}, []
    frame = MiniQMTFetcher(client=Client()).get_daily_data("SH600000", "2026-10-14", "2026-10-16")
    assert frame["volume"].tolist() == [1200, 1200]
    assert frame["close"].tolist() == [10, 11]
    assert frame["pct_chg"].iloc[-1] == pytest.approx(10)
    assert {"ma5", "ma10", "ma20"}.issubset(frame)


def test_manager_prefers_configured_miniqmt_and_falls_back_without_changing_other_markets(monkeypatch):
    from data_provider.base import BaseFetcher, DataFetcherManager
    from data_provider.miniqmt_fetcher import MiniQMTFetcher

    class Client:
        def daily(self, *args, **kwargs):
            raise RuntimeError("offline HOME")

    class Fallback(BaseFetcher):
        name, priority = "Fallback", 0

        def _fetch_raw_data(self, *args):
            return pd.DataFrame([bar("2026-10-16")])

        def _normalize_data(self, frame, code):
            return frame.assign(code=code, pct_chg=0)
    manager = DataFetcherManager(fetchers=[Fallback(), MiniQMTFetcher(client=Client())])
    frame, provider = manager.get_daily_data("600000", "2026-10-16", "2026-10-16")
    assert provider == "Fallback" and len(frame) == 1
    assert [f.name for f in manager._filter_daily_fetchers_for_market(manager._get_fetchers_snapshot(), "us")] == ["Fallback"]


@pytest.mark.filterwarnings("error:Setting an item of incompatible dtype:FutureWarning")
def test_affine_actions_apply_cash_and_split_in_order_without_using_dr():
    from data_provider.miniqmt_bridge import adjusted_prices
    raw = pd.DataFrame([bar("2026-10-12", 10), bar("2026-10-13", 12), bar("2026-10-14", 7)])
    actions = [{"date": "2026-10-13", "interest": 2, "stockBonus": 0, "stockGift": 0,
                "allotNum": 0, "allotPrice": 0, "gugai": 0, "dr": 0.123},
               {"date": "2026-10-14", "interest": 0, "stockBonus": 1, "stockGift": 0,
                "allotNum": 0, "allotPrice": 0, "gugai": 0, "dr": 99}]
    qfq = adjusted_prices(raw, actions)
    assert qfq["close"].tolist() == [4, 6, 7]
    assert qfq["volume"].tolist() == [12, 12, 12]
    pd.testing.assert_frame_equal(raw, pd.DataFrame([bar("2026-10-12", 10), bar("2026-10-13", 12), bar("2026-10-14", 7)]))


@pytest.mark.parametrize("event_day", ["2006-05-12", "2026-10-12"])
def test_irrelevant_share_conversion_does_not_block_later_raw_window(event_day):
    from data_provider.miniqmt_bridge import adjusted_prices
    raw = pd.DataFrame([bar("2026-10-12", 10), bar("2026-10-13", 12)])
    event = dict(date=event_day, interest=0, stockBonus=0.3, stockGift=0,
                 allotNum=0, allotPrice=0, gugai=1)
    pd.testing.assert_frame_equal(adjusted_prices(raw, [event]), raw.astype(
        {field: float for field in ("open", "high", "low", "close")}))


def test_relevant_share_conversion_still_rejects_without_a_supported_rule():
    from data_provider.miniqmt_bridge import adjusted_prices, MiniQMTError
    raw = pd.DataFrame([bar("2026-10-12", 10), bar("2026-10-13", 12)])
    event = dict(date="2026-10-13", interest=0, stockBonus=0.3, stockGift=0,
                 allotNum=0, allotPrice=0, gugai=1)
    with pytest.raises(MiniQMTError, match="unsupported MiniQMT share conversion"):
        adjusted_prices(raw, [event])


def test_default_stock_manager_selects_miniqmt_when_private_profile_is_available(monkeypatch, tmp_path):
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    from data_provider.base import DataFetcherManager
    monkeypatch.setattr(MiniQMTBridgeClient, "configured", staticmethod(lambda: True))
    monkeypatch.setenv("QUANT_DATA_ROOT", "")
    monkeypatch.setattr("src.services.quant_workflow_paths.quant_data_root", lambda: tmp_path)
    manager = DataFetcherManager()
    assert manager.available_fetchers[0] == "MiniQMTFetcher"


def test_configured_local_history_is_available_without_a_quote_connection(monkeypatch, tmp_path):
    from data_provider.base import DataFetcherManager
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    monkeypatch.setenv("QUANT_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(MiniQMTBridgeClient, "configured", staticmethod(lambda: False))
    manager = DataFetcherManager()
    assert manager.available_fetchers[0] == "QuantQMTDataFetcher"
    assert "QuantQMTDataFetcher" not in [f.name for f in manager._filter_daily_fetchers_for_market(
        manager._get_fetchers_snapshot(), "hk")]


def test_local_history_failure_does_not_implicitly_download_from_another_source(tmp_path):
    from data_provider.base import BaseFetcher, DataFetcherManager, DataFetchError
    from data_provider.miniqmt_fetcher import QuantQMTDataFetcher
    class NetworkFallback(BaseFetcher):
        name, priority = "NetworkFallback", 0
        def _fetch_raw_data(self, *args):
            pytest.fail("local snapshot error triggered a supplier request")
        def _normalize_data(self, data, code):
            return data
    manager = DataFetcherManager(fetchers=[QuantQMTDataFetcher(quant_data_root=tmp_path), NetworkFallback()])
    with pytest.raises(DataFetchError):
        manager.get_daily_data("000001", "2026-10-14", "2026-10-16")


def test_miniqmt_snapshot_quote_retains_provider_timestamp():
    from data_provider.miniqmt_fetcher import MiniQMTFetcher

    class Client:
        def request(self, operation, symbols, start, end, **kwargs):
            assert operation == "quotes"
            return {"data": {"600000.SH": {
                "time": 1792128000000, "lastPrice": 10, "lastClose": 9,
                "volume": 12, "amount": 12000, "open": 9.5, "high": 10.5, "low": 9}}}
    q = MiniQMTFetcher(client=Client()).get_realtime_quote("600000")
    assert q.source.value == "miniqmt"
    assert q.volume == 1200 and q.price == 10
    assert q.provider_timestamp is not None


def test_stock_front_read_uses_one_bounded_pair_and_rejects_inconsistent_adjustment():
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient, MiniQMTError
    calls = []

    def transport(request):
        calls.append(request["operation"])
        assert request["operation"] == "update_pair"
        return reply(request, {"600000.SH": {"raw": [bar("2026-10-16")],
                                             "qfq": [bar("2026-10-16", 9)], "actions": [],
                                             "calendar": ["2026-10-16"]}})
    with pytest.raises(MiniQMTError, match="QFQ"):
        MiniQMTBridgeClient(transport=transport).daily(["600000.SH"], "2026-10-16", "2026-10-16", adjustment="front")
    assert calls == ["update_pair"]


def test_default_realtime_uses_miniqmt_before_paid_source_and_respects_explicit_override(monkeypatch):
    from src.config import Config
    from data_provider.miniqmt_bridge import MiniQMTBridgeClient
    monkeypatch.setattr(MiniQMTBridgeClient, "configured", staticmethod(lambda: True))
    monkeypatch.delenv("REALTIME_SOURCE_PRIORITY", raising=False)
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only")
    assert Config._resolve_realtime_source_priority().startswith("miniqmt,tushare,")
    monkeypatch.setenv("REALTIME_SOURCE_PRIORITY", "tencent,efinance")
    assert Config._resolve_realtime_source_priority() == "tencent,efinance"


@pytest.mark.parametrize("change", ["gap", "duplicate", "cash_revision", "volume"])
def test_pair_rejects_missing_keys_and_changed_adjusted_values(change):
    from data_provider.miniqmt_bridge import verified_pair, MiniQMTError
    data = {"raw": [bar("2026-10-15"), bar("2026-10-16")],
            "qfq": [bar("2026-10-15"), bar("2026-10-16")], "actions": [],
            "calendar": ["2026-10-15", "2026-10-16"]}
    if change == "gap":
        data["raw"].pop(0)
    if change == "duplicate":
        data["raw"][1]["date"] = "2026-10-15"
    if change == "cash_revision":
        data["qfq"][0]["close"] = 9
    if change == "volume":
        data["qfq"][0]["volume"] = 13
    with pytest.raises(MiniQMTError):
        verified_pair(data, "2026-10-15", "2026-10-16")
