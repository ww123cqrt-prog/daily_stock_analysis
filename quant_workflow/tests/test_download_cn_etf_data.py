from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "download_cn_etf_data.py"
    )
    spec = importlib.util.spec_from_file_location("download_cn_etf_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paired_frame(
    symbol: str,
    *,
    name: str,
    source: str,
    closes: list[float],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    trade_dates = pd.to_datetime(
        dates or ["2026-08-04", "2026-08-05", "2026-08-06"]
    )
    assert len(trade_dates) == len(closes)
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(trade_dates),
            "name": [name] * len(trade_dates),
            "date": trade_dates,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [100.0] * len(trade_dates),
            "source": [source] * len(trade_dates),
        }
    )


def test_build_universe_reconciles_sina_and_recent_ths_listings() -> None:
    module = load_module()
    sina = pd.DataFrame(
        {
            "代码": ["sh510050", "sz159915", "bad"],
            "名称": ["上证50ETF", "创业板ETF", "无效"],
            "成交额": [300.0, 200.0, 999.0],
        }
    )
    ths = pd.DataFrame(
        {
            "基金代码": ["510050", "159915", "159058", "158000"],
            "基金名称": ["上证50ETF", "创业板ETF", "证券ETF", "待上市ETF"],
            "最新-交易日": ["2026-08-06", "2026-08-06", "2026-08-05", None],
        }
    )

    result = module.build_etf_universe(
        sina,
        ths,
        through="2026-08-06",
        recent_listing_days=10,
    )

    assert result["symbol"].tolist() == ["SH510050", "SZ159058", "SZ159915"]
    extra = result.set_index("symbol").loc["SZ159058"]
    assert not bool(extra["in_sina"])
    assert bool(extra["in_ths"])
    assert extra["market"] == "sz"
    assert "158000" not in set(result["code"])


def test_select_trial_universe_keeps_unified_required_codes() -> None:
    module = load_module()
    assert module.DEFAULT_LIMIT == 51
    assert set(module.DEFAULT_REQUIRED_CODES) == {
        "512890",
        "513010",
        "510720",
        "159915",
        "513100",
    }
    rows = []
    for index in range(60):
        code = f"51{index:04d}"
        rows.append(
            {
                "symbol": f"SH{code}",
                "code": code,
                "market": "sh",
                "name": code,
                "sina_code": f"sh{code}",
                "in_sina": True,
                "in_ths": True,
                "ths_latest_trade_date": "2026-08-06",
                "sina_amount": float(1000 - index),
            }
        )
    for code, name in (
        ("512890", "红利低波ETF华泰柏瑞"),
        ("513010", "恒生科技ETF易方达"),
        ("510720", "红利国企ETF国泰"),
        ("159915", "创业板ETF"),
        ("513100", "纳指ETF"),
    ):
        market = module._market_for_code(code)
        rows.append(
            {
                "symbol": f"{market.upper()}{code}",
                "code": code,
                "market": market,
                "name": name,
                "sina_code": f"{market}{code}",
                "in_sina": True,
                "in_ths": True,
                "ths_latest_trade_date": "2026-08-06",
                "sina_amount": 1.0,
            }
        )

    selected = module.select_trial_universe(
        pd.DataFrame(rows),
        limit=51,
        required_codes=("512890", "513010", "510720", "159915", "513100"),
    )

    required = {"512890", "513010", "510720", "159915", "513100"}
    assert len(selected) == 51
    assert required.issubset(set(selected["code"]))
    assert set(
        selected.loc[
            selected["code"].isin(required),
            "selection_reason",
        ]
    ) == {"required"}


def test_tencent_history_pages_backwards_until_listing_date() -> None:
    module = load_module()
    calls: list[str] = []

    class Response:
        def __init__(self, rows):
            self.rows = rows

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "code": 0,
                "msg": "",
                "data": {"sh510050": {"qfqday": self.rows}},
            }

    class Session:
        def get(self, _url: str, *, params, timeout):
            assert timeout == 20.0
            param = params["param"]
            calls.append(param)
            if ",2026-08-06," in param:
                return Response(
                    [
                        ["2026-08-05", "1.0", "1.1", "1.2", "0.9", "10"],
                        ["2026-08-06", "1.1", "1.2", "1.3", "1.0", "11"],
                    ]
                )
            return Response(
                [["2026-08-04", "0.9", "1.0", "1.1", "0.8", "9"]]
            )

    result = module.fetch_tencent_history(
        "SH510050",
        {"code": "510050", "market": "sh", "name": "上证50ETF"},
        adjustment="qfq",
        start_date="2026-08-04",
        end_date="2026-08-06",
        session=Session(),
        request_gate=module.RequestGate(interval=0, budget=10),
        page_size=2,
        retries=1,
    )

    assert calls == [
        "sh510050,day,,2026-08-06,2,qfq",
        "sh510050,day,,2026-08-04,2,qfq",
    ]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
    ]
    assert result["source"].unique().tolist() == [
        module.TENCENT_ADJUSTED_SOURCE_NAME
    ]


def test_request_gate_counts_failed_calls_against_budget() -> None:
    module = load_module()
    gate = module.RequestGate(interval=0, budget=1)

    with pytest.raises(ConnectionError):
        gate.call(lambda: (_ for _ in ()).throw(ConnectionError("offline")))
    with pytest.raises(module.CnEtfDownloadError, match="budget"):
        gate.call(lambda: None)


def test_build_paired_adjustments_supports_cash_distribution_offset() -> None:
    module = load_module()
    raw = paired_frame(
        "SH510300",
        name="沪深300ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[2.604, 2.644, 2.630],
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:1, field] = adjusted.loc[:1, field] - 0.88
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME

    evidence = module.build_paired_adjustments("SH510300", raw, adjusted)

    assert evidence["factor"].round(8).tolist() == [1.0, 1.0, 1.0]
    assert evidence["offset"].round(8).tolist() == [-0.88, -0.88, 0.0]
    assert evidence["adjustment_formula"].unique().tolist() == [
        "qfq_ohlc=raw_ohlc*factor+offset"
    ]


def test_build_paired_adjustments_supports_share_split_scale() -> None:
    module = load_module()
    raw = paired_frame(
        "SH513100",
        name="纳指ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[5.0, 5.2, 5.4],
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:1, field] = adjusted.loc[:1, field] * 0.2
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME

    evidence = module.build_paired_adjustments("SH513100", raw, adjusted)

    assert evidence["factor"].round(8).tolist() == [0.2, 0.2, 1.0]
    assert evidence["offset"].round(8).tolist() == [0.0, 0.0, 0.0]


def test_build_paired_adjustments_rejects_non_normalized_latest_date() -> None:
    module = load_module()
    raw = paired_frame(
        "SH510300",
        name="沪深300ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[2.604, 2.644, 2.630],
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted[field] = adjusted[field] - 0.88
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME

    with pytest.raises(module.DataContractError, match="latest transform is not normalized"):
        module.build_paired_adjustments("SH510300", raw, adjusted)


def test_build_paired_adjustments_keeps_one_transform_per_adjustment_segment() -> None:
    module = load_module()
    dates = pd.date_range("2026-07-10", periods=20, freq="B")
    closes = [2.5 + index * 0.017 + (index % 3) * 0.006 for index in range(20)]
    raw = pd.DataFrame(
        {
            "symbol": ["SH510500"] * len(dates),
            "name": ["中证500ETF"] * len(dates),
            "date": dates,
            "open": closes,
            "high": [value + 0.023 for value in closes],
            "low": [value - 0.019 for value in closes],
            "close": [value + 0.007 for value in closes],
            "volume": [100.0] * len(dates),
            "source": [module.TENCENT_RAW_SOURCE_NAME] * len(dates),
        }
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:14, field] = (
            adjusted.loc[:14, field] * 0.87 - 0.38
        ).round(3)
        adjusted.loc[15:, field] = adjusted.loc[15:, field].round(3)
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME

    evidence = module.build_paired_adjustments("SH510500", raw, adjusted)
    segments = evidence[["factor", "offset"]].round(8).drop_duplicates()
    actions = module._corporate_actions_from_adjustments(
        {"SH510500": evidence}, provider="tencent"
    )

    assert len(segments) == 2
    assert len(actions) == 1
    assert evidence["segment_id"].nunique() == 2


def test_cache_requires_matching_affine_factor_and_offset(tmp_path) -> None:
    module = load_module()
    symbol = "SH510300"
    raw = paired_frame(
        symbol,
        name="沪深300ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[2.604, 2.644, 2.630],
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:1, field] = adjusted.loc[:1, field] - 0.88
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME
    adjusted["provider_response_key"] = "qfqday"
    factors = module.build_paired_adjustments(symbol, raw, adjusted)

    for directory in ("raw", "qfq", "factors"):
        (tmp_path / directory).mkdir()
    raw.to_parquet(tmp_path / "raw" / f"{symbol}.parquet", index=False)
    adjusted.to_parquet(tmp_path / "qfq" / f"{symbol}.parquet", index=False)
    factors.to_parquet(tmp_path / "factors" / f"{symbol}.parquet", index=False)
    assert module._cache_symbol_complete(tmp_path, symbol)

    missing_segment = factors.drop(columns="segment_id")
    missing_segment.to_parquet(
        tmp_path / "factors" / f"{symbol}.parquet", index=False
    )
    assert not module._cache_symbol_complete(tmp_path, symbol)

    stale_factors = factors.drop(columns="offset")
    stale_factors.to_parquet(tmp_path / "factors" / f"{symbol}.parquet", index=False)
    assert not module._cache_symbol_complete(tmp_path, symbol)

    corrupted_factors = factors.copy()
    corrupted_factors.loc[0, "offset"] += 0.1
    corrupted_factors.to_parquet(tmp_path / "factors" / f"{symbol}.parquet", index=False)
    assert not module._cache_symbol_complete(tmp_path, symbol)


def test_resume_upgrades_legacy_factor_cache_without_downloading(tmp_path) -> None:
    module = load_module()
    symbol = "SH510300"
    raw = paired_frame(
        symbol,
        name="沪深300ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[2.604, 2.644, 2.630],
    )
    adjusted = raw.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:1, field] = adjusted.loc[:1, field] - 0.88
    adjusted["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME
    adjusted["provider_response_key"] = "qfqday"
    legacy_factors = pd.DataFrame(
        {
            "date": raw["date"],
            "factor": adjusted["close"] / raw["close"],
            "factor_source": "derived_same_provider_raw_qfq_close_ratio",
        }
    )

    for directory in ("raw", "qfq", "factors"):
        (tmp_path / directory).mkdir()
    raw.to_parquet(tmp_path / "raw" / f"{symbol}.parquet", index=False)
    adjusted.to_parquet(tmp_path / "qfq" / f"{symbol}.parquet", index=False)
    legacy_factors.to_parquet(
        tmp_path / "factors" / f"{symbol}.parquet", index=False
    )

    assert module._upgrade_cached_adjustment_evidence(tmp_path, symbol)
    upgraded = pd.read_parquet(tmp_path / "factors" / f"{symbol}.parquet")
    assert upgraded["offset"].round(8).tolist() == [-0.88, -0.88, 0.0]
    assert module._cache_symbol_complete(tmp_path, symbol)


def test_publish_cn_etf_snapshot_writes_one_paired_dataset(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    universe = pd.DataFrame(
        {
            "symbol": ["SH510050", "SZ159915"],
            "code": ["510050", "159915"],
            "market": ["sh", "sz"],
            "name": ["上证50ETF", "创业板ETF"],
            "sina_code": ["sh510050", "sz159915"],
            "in_sina": [True, True],
            "in_ths": [True, True],
            "ths_latest_trade_date": ["2026-08-06", "2026-08-06"],
        }
    )

    monkeypatch.setattr(
        module,
        "fetch_etf_universe",
        lambda **_kwargs: (universe, {"sina_count": 2, "ths_count": 2}),
    )

    def fake_pair(symbol, metadata, **_kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )
        return raw, qfq

    monkeypatch.setattr(module, "fetch_tencent_pair", fake_pair)
    monkeypatch.setattr(module, "_sync_root_manifest", lambda _root: None)
    lock_state = {"held": False}

    @contextmanager
    def tracked_lock(_path):
        assert not lock_state["held"]
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    original_publish = module.publish_snapshot_group

    def guarded_publish(*args, **kwargs):
        assert lock_state["held"]
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(module, "_run_lock", tracked_lock)
    monkeypatch.setattr(module, "publish_snapshot_group", guarded_publish)

    data_root = tmp_path / "quant_data"
    data_root.mkdir()
    (data_root / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )

    published = module.publish_cn_etf_snapshot(
        quant_data_dir=data_root,
        start_date="2004-01-01",
        end_date="2026-08-06",
        generated_at="2026-08-07T10:00:00+08:00",
        request_interval=0.2,
        request_budget=100,
        confirm_severe_current_corruption=True,
        confirm_backup_unrecoverable=True,
        confirm_user_approved=True,
    )

    qfq_manifest = json.loads(
        (published / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    universe_path = published / "ETF_UNIVERSE.parquet"

    assert qfq_manifest["dataset_id"] == "cn_etf_daily"
    assert qfq_manifest["data_kind"] == "paired_daily_bars"
    assert qfq_manifest["symbol_count"] == 2
    assert qfq_manifest["adjustment"]["paired_raw_component_path"] == "raw"
    assert (published / "raw" / "SH510050.parquet").is_file()
    assert (published / "qfq" / "SH510050.parquet").is_file()
    assert (published / "factors" / "SH510050.parquet").is_file()
    assert universe_path.is_file()
    assert len(pd.read_parquet(universe_path)) == 2
    assert qfq_manifest["source"]["universe_path"] == "ETF_UNIVERSE.parquet"
    dataset_parent = published.parents[1]
    assert dataset_parent.name == "cn_etf_daily"
    assert not (dataset_parent.parent / "cn_etf_daily_raw").exists()
    assert not (dataset_parent.parent / "cn_etf_daily_qfq").exists()


def test_prune_snapshot_history_keeps_current_and_one_backup(tmp_path) -> None:
    module = load_module()
    dataset_root = tmp_path / "cn_etf_daily"
    snapshots = dataset_root / "snapshots"
    for index in range(3):
        snapshot_id = f"snapshot-{index}"
        snapshot = snapshots / snapshot_id
        snapshot.mkdir(parents=True)
        (snapshot / "DATASET_MANIFEST.json").write_text(
            json.dumps(
                {
                    "snapshot_id": snapshot_id,
                    "source_retrieved_at": f"2026-08-0{index + 1}T10:00:00+08:00",
                }
            ),
            encoding="utf-8",
        )
    (dataset_root / "LATEST.json").write_text(
        json.dumps({"snapshot_id": "snapshot-2"}), encoding="utf-8"
    )

    removed = module.prune_snapshot_history(dataset_root, keep=2)

    assert removed == ["snapshot-0"]
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "snapshot-1",
        "snapshot-2",
    ]


def test_prune_snapshot_history_rejects_symlinked_dataset_root(tmp_path) -> None:
    module = load_module()
    external = tmp_path / "external"
    snapshot = external / "snapshots" / "snapshot-1"
    snapshot.mkdir(parents=True)
    (snapshot / "DATASET_MANIFEST.json").write_text(
        json.dumps(
            {
                "snapshot_id": "snapshot-1",
                "source_retrieved_at": "2026-08-01T10:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    (external / "LATEST.json").write_text(
        json.dumps({"snapshot_id": "snapshot-1"}), encoding="utf-8"
    )
    linked = tmp_path / "cn_etf_daily"
    linked.symlink_to(external, target_is_directory=True)

    with pytest.raises(module.CnEtfDownloadError, match="symlink"):
        module.prune_snapshot_history(linked, keep=2)

    assert snapshot.is_dir()


def test_publish_cn_etf_snapshot_keeps_cache_and_does_not_publish_on_failure(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    universe = pd.DataFrame(
        {
            "symbol": ["SH510050", "SZ159915"],
            "code": ["510050", "159915"],
            "market": ["sh", "sz"],
            "name": ["上证50ETF", "创业板ETF"],
            "sina_code": ["sh510050", "sz159915"],
            "in_sina": [True, True],
            "in_ths": [True, True],
            "ths_latest_trade_date": ["2026-08-06", "2026-08-06"],
        }
    )
    monkeypatch.setattr(
        module,
        "fetch_etf_universe",
        lambda **_kwargs: (universe, {"sina_count": 2, "ths_count": 2}),
    )

    def fail_second(symbol, metadata, **_kwargs):
        if symbol == "SZ159915":
            raise ConnectionError("provider unavailable")
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return raw, raw.assign(source=module.TENCENT_ADJUSTED_SOURCE_NAME)

    monkeypatch.setattr(module, "fetch_tencent_pair", fail_second)

    with pytest.raises(module.CnEtfDownloadError, match="incomplete"):
        module.publish_cn_etf_snapshot(
            quant_data_dir=tmp_path / "quant_data",
            start_date="2004-01-01",
            end_date="2026-08-06",
            generated_at="2026-08-07T10:00:00+08:00",
            request_interval=0.2,
            request_budget=100,
            max_consecutive_failures=3,
            confirm_severe_current_corruption=True,
            confirm_backup_unrecoverable=True,
            confirm_user_approved=True,
        )

    root = tmp_path / "quant_data"
    assert not list((root / "parquet" / "etf_1d").rglob("LATEST.json"))
    assert list((root / "cache" / "cn_etf_daily" / "v1").rglob("SH510050.parquet"))
    assert list((root / "cache" / "cn_etf_daily" / "v1").rglob("failures.json"))


def _seed_current_paired_snapshot(module, monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    universe = pd.DataFrame(
        {
            "symbol": ["SH510050", "SZ159915"],
            "code": ["510050", "159915"],
            "market": ["sh", "sz"],
            "name": ["上证50ETF", "创业板ETF"],
            "sina_code": ["sh510050", "sz159915"],
            "in_sina": [True, True],
            "in_ths": [True, True],
            "ths_latest_trade_date": ["2026-08-06", "2026-08-06"],
        }
    )
    monkeypatch.setattr(
        module,
        "fetch_etf_universe",
        lambda **_kwargs: (
            universe,
            {
                "selected_count": 2,
                "selection_method": "test_frozen_universe",
            },
        ),
    )

    def seed_pair(symbol, metadata, **_kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )
        return raw, qfq

    monkeypatch.setattr(module, "fetch_tencent_pair", seed_pair)
    monkeypatch.setattr(module, "_sync_root_manifest", lambda _root: None)
    root = tmp_path / "quant_data"
    root.mkdir()
    (root / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {},
            }
        ),
        encoding="utf-8",
    )
    published = module.publish_cn_etf_snapshot(
        quant_data_dir=root,
        start_date="2004-01-01",
        end_date="2026-08-06",
        generated_at="2026-08-07T10:00:00+08:00",
        request_interval=0.2,
        request_budget=100,
        confirm_severe_current_corruption=True,
        confirm_backup_unrecoverable=True,
        confirm_user_approved=True,
    )
    return root, published


def test_incremental_update_is_zero_request_noop_when_target_is_already_present(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)

    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("an up-to-date ETF snapshot must not use the network")

    monkeypatch.setattr(module, "fetch_tencent_pair", forbidden_download)
    result = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-06",
        generated_at="2026-08-07T11:00:00+08:00",
    )

    assert result == current
    snapshots = list((root / "parquet" / "etf_1d" / "cn_etf_daily" / "snapshots").iterdir())
    assert snapshots == [current]


def test_incremental_update_reuses_frozen_universe_and_fetches_only_recent_window(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls: list[tuple[str, str, str, int]] = []

    def forbidden_universe_refresh(**_kwargs):
        raise AssertionError("ordinary ETF updates must reuse the frozen universe")

    def incremental_pair(symbol, metadata, *, start_date, end_date, **_kwargs):
        calls.append((symbol, start_date, end_date, _kwargs["page_size"]))
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )
        return raw, qfq

    monkeypatch.setattr(module, "fetch_etf_universe", forbidden_universe_refresh)
    monkeypatch.setattr(module, "fetch_tencent_pair", incremental_pair)
    updated = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-07",
        generated_at="2026-08-07T11:00:00+08:00",
        overlap_days=14,
    )

    assert updated != current
    assert {symbol for symbol, _start, _end, _page in calls} == {
        "SH510050",
        "SZ159915",
    }
    assert all(start != "2004-01-01" for _symbol, start, _end, _page in calls)
    assert all(end == "2026-08-07" for _symbol, _start, end, _page in calls)
    assert all(page == 64 for _symbol, _start, _end, page in calls)
    old_universe = pd.read_parquet(current / "ETF_UNIVERSE.parquet")
    new_universe = pd.read_parquet(updated / "ETF_UNIVERSE.parquet")
    pd.testing.assert_frame_equal(new_universe, old_universe)
    manifest = json.loads((updated / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"]["update_mode"] == "missing_dates_only"
    assert manifest["source"]["parent_snapshot_id"] == current.name
    assert manifest["source"]["targeted_full_refresh_symbols"] == []
    assert len(pd.read_parquet(updated / "qfq" / "SH510050.parquet")) == 4


def test_incremental_update_reports_post_publish_housekeeping_warning(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)

    def incremental_pair(symbol, metadata, *, start_date, end_date, **_kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )
        return raw, qfq

    def fail_housekeeping(*_args, **_kwargs):
        raise OSError("simulated retention failure")

    monkeypatch.setattr(module, "fetch_tencent_pair", incremental_pair)
    monkeypatch.setattr(module, "prune_snapshot_history", fail_housekeeping)

    updated = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-07",
        generated_at="2026-08-07T11:00:00+08:00",
    )

    assert updated != current
    status = json.loads(
        (root / "runtime" / "status" / "cn_etf_daily_update_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "published_with_warnings"
    assert status["snapshot_id"] == updated.name
    assert "retention" in status["post_publish_warnings"][0]


def test_incremental_update_normalizes_weekend_to_previous_weekday(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls: list[tuple[str, str]] = []

    def incremental_pair(symbol, metadata, *, start_date, end_date, **_kwargs):
        calls.append((start_date, end_date))
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )
        return raw, qfq

    monkeypatch.setattr(module, "fetch_tencent_pair", incremental_pair)
    updated = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-08",  # Saturday; effective target is Friday 08-07.
        generated_at="2026-08-08T10:00:00+08:00",
    )

    assert updated != current
    assert calls
    assert {end for _start, end in calls} == {"2026-08-07"}
    status = json.loads(
        (root / "runtime" / "status" / "cn_etf_daily_update_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["requested_through"] == "2026-08-08"
    assert status["through"] == "2026-08-07"


def test_snapshot_file_copies_do_not_share_inode(tmp_path) -> None:
    module = load_module()
    source = tmp_path / "source.parquet"
    destination = tmp_path / "destination.parquet"
    source.write_bytes(b"snapshot")

    module._link_or_copy(source, destination)

    assert source.stat().st_ino != destination.stat().st_ino
    destination.write_bytes(b"changed")
    assert source.read_bytes() == b"snapshot"


def test_incremental_update_refreshes_only_the_symbol_with_a_proven_revision(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, _current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    call_counts = {"SH510050": 0, "SZ159915": 0}

    def incremental_or_targeted_pair(
        symbol, metadata, *, start_date, end_date, **_kwargs
    ):
        del start_date, end_date
        call_counts[symbol] += 1
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        qfq = raw.copy()
        if symbol == "SH510050":
            for field in ("open", "high", "low", "close"):
                qfq.loc[:2, field] = qfq.loc[:2, field] - 0.1
        qfq["source"] = module.TENCENT_ADJUSTED_SOURCE_NAME
        qfq["provider_response_key"] = "qfqday"
        return raw, qfq

    monkeypatch.setattr(module, "fetch_tencent_pair", incremental_or_targeted_pair)
    updated = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-07",
        generated_at="2026-08-07T11:00:00+08:00",
    )

    assert call_counts == {"SH510050": 2, "SZ159915": 1}
    manifest = json.loads((updated / "DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"]["targeted_full_refresh_symbols"] == ["SH510050"]


def test_full_rebuild_cli_refuses_without_all_three_confirmations(
    monkeypatch, capsys
) -> None:
    module = load_module()

    def forbidden_rebuild(**_kwargs):
        raise AssertionError("the full rebuild must stop before any provider call")

    monkeypatch.setattr(module, "publish_cn_etf_snapshot", forbidden_rebuild)
    result = module.main(
        [
            "--through",
            "2026-08-07",
            "--full-rebuild",
            "--confirm-severe-current-corruption",
        ]
    )

    assert result == 2
    assert "备份无法恢复" in capsys.readouterr().err


def test_full_rebuild_function_refuses_without_all_three_confirmations() -> None:
    module = load_module()

    with pytest.raises(module.CnEtfDownloadError, match="explicit user approval"):
        module.publish_cn_etf_snapshot(end_date="2026-08-07")
