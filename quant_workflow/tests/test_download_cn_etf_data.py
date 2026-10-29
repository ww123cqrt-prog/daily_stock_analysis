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
        "sh510050,day,2026-08-04,2026-08-06,2,qfq",
        "sh510050,day,2026-08-04,2026-08-04,2,qfq",
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


def test_expansion_cache_rejects_ohlc_relationships_the_release_validator_rejects(
    tmp_path,
) -> None:
    module = load_module()
    symbol = "SH510180"
    raw = paired_frame(
        symbol,
        name="上证180ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2],
    )
    raw.loc[0, "high"] = 0.99
    qfq = raw.assign(
        source=module.TENCENT_ADJUSTED_SOURCE_NAME,
        provider_response_key="qfqday",
    )
    factors = module.build_paired_adjustments(symbol, raw, qfq)
    for component, frame in (("raw", raw), ("qfq", qfq), ("factors", factors)):
        path = tmp_path / component / f"{symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    assert not module._expansion_cache_symbol_complete(
        tmp_path,
        symbol,
        minimum_history_rows=3,
        through=pd.Timestamp("2026-08-06"),
    )


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


def test_prune_snapshot_history_can_preserve_the_expansion_parent_backup(
    tmp_path,
) -> None:
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

    removed = module.prune_snapshot_history(
        dataset_root,
        keep=2,
        preferred_backup_snapshot_id="snapshot-0",
    )

    assert removed == ["snapshot-1"]
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "snapshot-0",
        "snapshot-2",
    ]


def test_prune_snapshot_history_supports_three_immutable_snapshots(tmp_path) -> None:
    module = load_module()
    dataset_root = tmp_path / "cn_etf_daily"
    snapshots = dataset_root / "snapshots"
    for index in range(4):
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
        json.dumps({"snapshot_id": "snapshot-3"}), encoding="utf-8"
    )

    removed = module.prune_snapshot_history(dataset_root, keep=3)

    assert removed == ["snapshot-0"]
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "snapshot-1",
        "snapshot-2",
        "snapshot-3",
    ]


def test_prune_snapshot_history_rejects_symlinked_dataset_root(tmp_path, make_symlink) -> None:
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
    make_symlink(linked, external, target_is_directory=True)

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
        provider="tencent",
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
        provider="tencent",
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
        provider="tencent",
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
        provider="tencent",
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


def test_incremental_update_rebases_only_the_symbol_with_a_proven_basis_change(
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
        provider="tencent",
    )

    assert call_counts == {"SH510050": 1, "SZ159915": 1}
    manifest = json.loads(
        (updated / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["targeted_full_refresh_symbols"] == []
    assert set(manifest["source"]["historical_qfq_rebases"]) == {"SH510050"}


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


def test_incremental_cash_adjustment_rebases_cached_history_without_refetch(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    original = pd.read_parquet(old / "raw/SH510050.parquet")
    calls = []

    def incoming(symbol, metadata, **kwargs):
        calls.append((symbol, kwargs["start_date"]))
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        qfq = raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )
        if symbol == "SH510050":
            for field in ("open", "high", "low", "close"):
                qfq.loc[:2, field] -= 0.1
        return raw, qfq

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    new = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07", provider="tencent")
    assert len(calls) == 2, "a QFQ basis change must not request full history"
    pd.testing.assert_frame_equal(
        pd.read_parquet(old / "raw/SH510050.parquet"), original
    )
    assert pd.read_parquet(new / "qfq/SH510050.parquet")["close"].round(6).tolist() == [
        0.9,
        1.0,
        1.1,
        1.3,
    ]
    manifest = json.loads((new / "DATASET_MANIFEST.json").read_text())
    assert manifest["source"]["targeted_full_refresh_symbols"] == []
    assert manifest["source"]["historical_qfq_rebases"]["SH510050"][
        "offset"
    ] == pytest.approx(-0.1)


def test_incremental_raw_revision_stops_without_full_history_request(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls = []

    def revised(symbol, metadata, **kwargs):
        calls.append(symbol)
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.05, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", revised)
    with pytest.raises(module.HistoricalRevisionDetected, match="raw history"):
        module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07", provider="tencent")
    assert len(calls) == 1
    assert (
        module.resolve_dataset(
            "cn_etf_daily", quant_data_root=root, require_basis="qfq"
        ).snapshot_path
        == current
    )


def test_prepared_symbol_update_reuses_downloads_before_full_update(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    pointer = root / "parquet/etf_1d/cn_etf_daily/LATEST.json"
    original_pointer = pointer.read_bytes()
    calls = []

    def incoming(symbol, metadata, **kwargs):
        calls.append(symbol)
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    prepared = module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-07",
        base_snapshot=current,
        symbols=["SH510050"],
        publish=False,
        provider="tencent",
    )
    assert pointer.read_bytes() == original_pointer
    assert pd.read_parquet(prepared / "qfq/SH510050.parquet")["close"].tolist() == [
        1.0,
        1.1,
        1.2,
        1.3,
    ]
    new = module.update_cn_etf_snapshot(
        quant_data_dir=root, end_date="2026-08-07", base_snapshot=current,
        provider="tencent",
    )
    assert calls == ["SH510050", "SZ159915"]
    assert pd.read_parquet(new / "ETF_UNIVERSE.parquet")["symbol"].tolist() == [
        "SH510050",
        "SZ159915",
    ]


def test_incremental_rejects_inconsistent_qfq_rebase() -> None:
    module = load_module()
    old = paired_frame(
        "SH510050",
        name="ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2],
    )
    new = paired_frame(
        "SH510050",
        name="ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2, 1.3],
        dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
    )
    bad = new.copy()
    bad.loc[0, "close"] -= 0.1
    with pytest.raises(module.HistoricalRevisionDetected):
        module.merge_incremental_pair("SH510050", old, old.copy(), new, bad)


def test_incremental_scale_and_cash_rebase_preserves_volume() -> None:
    module = load_module()
    old = paired_frame(
        "SH510050",
        name="ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2],
    )
    new = paired_frame(
        "SH510050",
        name="ETF",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2, 1.3],
        dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
    )
    adjusted = new.copy()
    for field in ("open", "high", "low", "close"):
        adjusted.loc[:2, field] = adjusted.loc[:2, field] * 0.5 - 0.1
    raw, qfq, count = module.merge_incremental_pair(
        "SH510050", old, old.copy(), new, adjusted
    )
    assert count == 1
    assert qfq["close"].round(6).tolist() == [0.4, 0.45, 0.5, 1.3]
    assert qfq["volume"].tolist() == raw["volume"].tolist()


def test_incremental_prepared_cache_tampering_is_rejected(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)

    def incoming(symbol, metadata, **kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    module.update_cn_etf_snapshot(
        quant_data_dir=root,
        end_date="2026-08-07",
        base_snapshot=current,
        symbols=["SH510050"],
        publish=False,
        provider="tencent",
    )
    raw_path = next((root / "cache/cn-etf-incremental").glob("*/raw/SH510050.parquet"))
    changed = pd.read_parquet(raw_path)
    changed.loc[0, "close"] = 99.0
    changed.to_parquet(raw_path, index=False)
    monkeypatch.setattr(
        module,
        "fetch_tencent_pair",
        lambda *a, **kw: pytest.fail(
            "tampering must not trigger a replacement download"
        ),
    )
    with pytest.raises(module.DataContractError, match="cache changed"):
        module.update_cn_etf_snapshot(
            quant_data_dir=root,
            end_date="2026-08-07",
            base_snapshot=current,
            symbols=["SH510050"],
            publish=False,
            provider="tencent",
        )


def test_explicit_failure_isolation_keeps_old_symbol_and_updates_healthy_one(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    old_bytes = (old / "raw/SH510050.parquet").read_bytes()

    def incoming(symbol, metadata, **kwargs):
        if symbol == "SH510050":
            raise module.CnEtfDownloadError("no current rows from the supplier")
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    new = module.update_cn_etf_snapshot(
        quant_data_dir=root, end_date="2026-08-07", keep_failed_history=True,
        provider="tencent",
    )
    assert (new / "raw/SH510050.parquet").read_bytes() == old_bytes
    assert len(pd.read_parquet(new / "qfq/SZ159915.parquet")) == 4
    m = json.loads((new / "DATASET_MANIFEST.json").read_text())
    assert set(m["source"]["symbol_update_failures"]) == {"SH510050"}
    assert (
        m["source"]["symbol_update_failures"]["SH510050"]["last_verified_date"]
        == "2026-08-06"
    )


def test_incremental_publication_does_not_rescan_unrelated_data(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)

    def incoming(symbol, metadata, **kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    def unrelated_scan(*args, **kwargs):
        pytest.fail(
            "targeted ETF publication must not scan PIT, Qlib, or the project service"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    monkeypatch.setattr(module, "_sync_root_manifest", unrelated_scan)
    new = module.update_cn_etf_snapshot(quant_data_dir=root, end_date="2026-08-07", provider="tencent")
    assert (
        module.resolve_dataset(
            "cn_etf_daily", quant_data_root=root, require_basis="qfq"
        ).snapshot_path
        == new
    )


def test_eastmoney_pair_is_bounded_and_retains_amount() -> None:
    module = load_module()
    import requests

    calls = []

    class Session:
        def get(self, url, *, params, timeout):
            calls.append(dict(params))
            qfq = params["fqt"] == "1"
            first = (
                "2026-08-06,1.1,1.1,1.2,1.0,10,1200,0,0,0,0"
                if qfq
                else "2026-08-06,1.2,1.2,1.3,1.1,10,1200,0,0,0,0"
            )
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(
                {
                    "rc": 0,
                    "data": {
                        "klines": [first, "2026-08-07,1.3,1.3,1.4,1.2,11,1300,0,0,0,0"]
                    },
                }
            ).encode()
            return response

    raw, qfq = module.fetch_eastmoney_pair(
        "SH510050",
        {"code": "510050", "market": "sh", "name": "ETF"},
        start_date="2026-08-06",
        end_date="2026-08-07",
        session=Session(),
        request_gate=module.RequestGate(0, 4),
        retries=1,
        timeout=1,
    )
    assert raw["close"].tolist() == [1.2, 1.3]
    assert qfq["close"].tolist() == [1.1, 1.3]
    assert raw["amount"].tolist() == [1200, 1300]
    assert len(calls) == 2
    assert all(c["beg"] == "20260806" and c["end"] == "20260807" for c in calls)
    assert [c["fqt"] for c in calls] == ["0", "1"]


def test_eastmoney_update_preserves_history_and_source_then_replays_without_network(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    import requests

    calls = []

    class Session:
        def get(self, url, *, params, timeout):
            calls.append(dict(params))
            rows = [
                "2026-08-04,1,1,1.01,0.99,100,1000,0,0,0,0",
                "2026-08-05,1.1,1.1,1.111,1.089,100,1100,0,0,0,0",
                "2026-08-06,1.2,1.2,1.212,1.188,100,1200,0,0,0,0",
                "2026-08-07,1.3,1.3,1.313,1.287,100,1300,0,0,0,0",
            ]
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps({"rc": 0, "data": {"klines": rows}}).encode()
            return response

        def close(self):
            pass

    # The frozen fixture's volume is supplied literally by paired_frame.
    old_volume = pd.read_parquet(old / "raw/SH510050.parquet")["volume"].iloc[0]
    assert old_volume == 100
    monkeypatch.setattr(requests, "Session", Session)
    new = module.update_cn_etf_snapshot(
        quant_data_dir=root, end_date="2026-08-07", provider="eastmoney"
    )
    frame = pd.read_parquet(new / "raw/SH510050.parquet")
    assert frame["close"].tolist() == [1, 1.1, 1.2, 1.3]
    assert frame.iloc[-1]["source"].startswith("eastmoney")
    assert frame.iloc[-1]["source_id"].startswith("eastmoney:")
    m = json.loads((new / "DATASET_MANIFEST.json").read_text())
    assert m["source"]["increment_provider"] == "eastmoney"
    assert (
        module.update_cn_etf_snapshot(
            quant_data_dir=root, end_date="2026-08-07", provider="eastmoney"
        )
        == new
    )
    assert len(calls) == 4


def test_eastmoney_empty_pair_is_not_accepted() -> None:
    module = load_module()
    import requests

    class Session:
        def get(self, url, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"rc":0,"data":{"klines":[]}}'
            return response

    with pytest.raises(module.CnEtfDownloadError):
        module.fetch_eastmoney_pair(
            "SH510050",
            {"code": "510050", "market": "sh", "name": "ETF"},
            start_date="2026-08-06",
            end_date="2026-08-07",
            session=Session(),
            request_gate=module.RequestGate(0, 2),
            retries=1,
            timeout=1,
        )


def test_incomplete_increment_is_not_saved_as_complete(monkeypatch, tmp_path) -> None:
    module = load_module()
    raw = paired_frame("SH510050", name="ETF", source="tencent", closes=[1, 1.1, 1.2])
    monkeypatch.setattr(
        module, "fetch_tencent_pair", lambda *args, **kwargs: (raw, raw.copy())
    )
    with pytest.raises(module.CnEtfDownloadError, match="before requested"):
        module._cached_incremental_pair(
            tmp_path,
            "SH510050",
            {"code": "510050", "market": "sh", "name": "ETF"},
            base_manifest_sha256="a" * 64,
            start_date="2026-08-04",
            end_date="2026-08-07",
            session=object(),
            request_gate=module.RequestGate(0, 2),
            retries=1,
            timeout=1,
        )
    assert not list(tmp_path.rglob("COMPLETE.json"))


def test_eastmoney_cleanup_cannot_replace_cancellation() -> None:
    module = load_module()
    import requests

    class Response(requests.Response):
        def json(self, **kwargs):
            raise KeyboardInterrupt("cancelled while reading")

        def close(self):
            raise OSError("close failed")

    class Session:
        def get(self, *args, **kwargs):
            response = Response()
            response.status_code = 200
            return response

    with pytest.raises(KeyboardInterrupt, match="cancelled while reading") as caught:
        module.fetch_eastmoney_pair(
            "SH510050",
            {"code": "510050", "market": "sh", "name": "ETF"},
            start_date="2026-08-06",
            end_date="2026-08-07",
            session=Session(),
            request_gate=module.RequestGate(0, 2),
            retries=2,
            timeout=1,
        )
    assert "close failed" in caught.value.__notes__[0]


@pytest.mark.parametrize("status,expected_calls", [(501, 1), (503, 2)])
def test_tencent_retries_only_transient_http_failures(status, expected_calls) -> None:
    module = load_module()
    import requests

    calls = []

    class Session:
        def get(self, *args, **kwargs):
            calls.append(kwargs)
            response = requests.Response()
            response.status_code = status
            response._content = b"supplier unavailable"
            return response

    with pytest.raises(module.CnEtfDownloadError):
        module.fetch_tencent_pair(
            "SH510050",
            {"code": "510050", "market": "sh", "name": "ETF"},
            start_date="2026-08-06",
            end_date="2026-08-07",
            session=Session(),
            request_gate=module.RequestGate(0, 4),
            retries=2,
            timeout=1,
        )
    assert len(calls) == expected_calls


def test_symbol_scoped_preparation_cannot_publish_full_dataset(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "fetch_tencent_pair",
        lambda *a, **kw: pytest.fail("scope check must precede acquisition"),
    )
    with pytest.raises(ValueError, match="symbol-scoped"):
        module.update_cn_etf_snapshot(
            quant_data_dir=root,
            end_date="2026-08-07",
            base_snapshot=current,
            symbols=["SH510050"],
            provider="tencent",
        )


def test_tencent_incremental_http_request_has_explicit_start_boundary() -> None:
    module = load_module()
    seen = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "code": 0,
                "data": {
                    "sh510050": {"day": [["2026-08-07", "1", "1", "1", "1", "10"]]}
                },
            }

    class Session:
        def get(self, url, *, params, timeout):
            seen.append(params["param"])
            return Response()

    frame = module.fetch_tencent_history(
        "SH510050",
        {"code": "510050", "market": "sh", "name": "ETF"},
        adjustment="raw",
        start_date="2026-08-07",
        end_date="2026-08-07",
        session=Session(),
        request_gate=module.RequestGate(0, 2),
        page_size=64,
        retries=1,
    )
    assert len(frame) == 1
    assert seen == ["sh510050,day,2026-08-07,2026-08-07,64"]


@pytest.mark.parametrize("metadata_only_symbol", [False, True])
def test_tickflow_history_can_receive_tencent_increment_without_relabeling(
    monkeypatch, tmp_path, metadata_only_symbol
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    old_manifest = json.loads((current / "DATASET_MANIFEST.json").read_text())
    frozen = current.parent / "frozen-tickflow-base"
    frozen.mkdir()
    for sym in ["SH510050", "SZ159915"]:
        for kind in ["raw", "qfq"]:
            f = pd.read_parquet(current / kind / f"{sym}.parquet")
            f = f[["date", "open", "high", "low", "close", "volume"]]
            module._atomic_parquet(f, frozen / kind / f"{sym}.parquet")
        module._link_or_copy(
            current / "factors" / f"{sym}.parquet",
            frozen / "factors" / f"{sym}.parquet",
        )
    module._link_or_copy(
        current / "ETF_UNIVERSE.parquet", frozen / "ETF_UNIVERSE.parquet"
    )
    if metadata_only_symbol:
        universe = pd.read_parquet(frozen / "ETF_UNIVERSE.parquet")
        extra = universe.iloc[[0]].assign(symbol="SH510099", code="510099")
        module._atomic_parquet(
            pd.concat([universe, extra], ignore_index=True),
            frozen / "ETF_UNIVERSE.parquet",
        )
    module._link_or_copy(
        current / "corporate_actions.parquet", frozen / "corporate_actions.parquet"
    )
    module.finalize_paired_price_snapshot(
        frozen,
        dataset_id="cn_etf_daily",
        snapshot_id=frozen.name,
        role="canonical",
        status="ready",
        source={"provider": "tickflow"},
        adjustment=old_manifest["adjustment"],
        producer=old_manifest["producer"],
        source_retrieved_at=old_manifest["source_retrieved_at"],
    )

    def incoming(symbol, metadata, **kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2, 1.3],
            dates=["2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME, provider_response_key="qfqday"
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", incoming)
    new = module.update_cn_etf_snapshot(
        quant_data_dir=root, base_snapshot=frozen, end_date="2026-08-07",
        provider="tencent",
    )
    raw = pd.read_parquet(new / "raw/SH510050.parquet")
    assert raw["close"].tolist() == [1.0, 1.1, 1.2, 1.3]
    assert raw.iloc[0]["source"] == "tickflow"
    assert raw.iloc[0]["source_id"].startswith("tickflow:")
    assert raw.iloc[-1]["source"] == module.TENCENT_RAW_SOURCE_NAME
    assert raw["source_id"].notna().all()
    new_manifest = json.loads((new / "DATASET_MANIFEST.json").read_text())
    assert new_manifest["source"]["retained_history_provider"] == "tickflow"
    assert set(pd.read_parquet(new / "ETF_UNIVERSE.parquet")["symbol"]) == {
        "SH510050",
        "SZ159915",
    }
    if metadata_only_symbol:
        assert new_manifest["source"]["metadata_only_symbols_without_history"] == [
            "SH510099"
        ]


def test_full_rebuild_function_refuses_without_all_three_confirmations() -> None:
    module = load_module()

    with pytest.raises(module.CnEtfDownloadError, match="explicit user approval"):
        module.publish_cn_etf_snapshot(end_date="2026-08-07")


def _representative_catalog_row(
    module,
    code: str,
    name: str,
    amount: float,
) -> dict[str, object]:
    market = module._market_for_code(code)
    return {
        "symbol": f"{market.upper()}{code}",
        "code": code,
        "market": market,
        "name": name,
        "sina_code": f"{market}{code}",
        "in_sina": True,
        "in_ths": True,
        "ths_latest_trade_date": "2026-08-14",
        "sina_amount": amount,
        "fund_type": "股票型",
    }


def test_representative_selection_reaches_200_and_preserves_every_existing_symbol() -> (
    None
):
    module = load_module()
    targets = module.REPRESENTATIVE_CATEGORY_TARGETS
    prefixes = {
        "core_broad": "沪深300增强",
        "sector_theme": "半导体主题",
        "dividend_factor": "红利低波策略",
        "hong_kong": "恒生科技策略",
        "overseas": "纳指海外策略",
        "bond": "国债策略",
        "commodity": "黄金商品策略",
        "cash": "货币现金策略",
    }
    rows: list[dict[str, object]] = []
    existing_rows: list[dict[str, object]] = []
    next_sh = 510000
    next_sz = 159000
    for category, count in targets.items():
        for index in range(count):
            if category in {"hong_kong", "overseas"}:
                code = f"{next_sh:06d}"
                next_sh += 1
            else:
                code = f"{next_sz:06d}"
                next_sz += 1
            row = _representative_catalog_row(
                module,
                code,
                f"{prefixes[category]}{index:02d}ETF",
                amount=float(10_000_000 - len(rows)),
            )
            rows.append(row)
            if index == 0:
                existing_rows.append(row)

    selected = module.select_representative_universe(
        pd.DataFrame(rows),
        pd.DataFrame(existing_rows),
        target_size=200,
        category_targets=targets,
    )

    assert len(selected) == 200
    assert set(row["symbol"] for row in existing_rows).issubset(set(selected["symbol"]))
    assert selected["representative_category"].value_counts().to_dict() == targets
    existing_symbols = {str(row["symbol"]) for row in existing_rows}
    assert set(
        selected.loc[selected["symbol"].isin(existing_symbols), "selection_reason"]
    ) == {"existing_snapshot"}


def test_representative_selection_does_not_add_a_duplicate_tracking_target() -> None:
    module = load_module()
    current = pd.DataFrame(
        [_representative_catalog_row(module, "512480", "半导体ETF华夏", 100.0)]
    )
    catalog = pd.DataFrame(
        [
            *current.to_dict(orient="records"),
            _representative_catalog_row(module, "159995", "半导体ETF国泰", 1_000.0),
            _representative_catalog_row(module, "588200", "科创芯片ETF嘉实", 900.0),
        ]
    )

    selected = module.select_representative_universe(
        catalog,
        current,
        target_size=2,
        category_targets={"sector_theme": 2},
        minimum_turnover=0.0,
    )

    assert set(selected["code"]) == {"512480", "588200"}
    assert selected["tracking_key"].nunique() == 2


def test_representative_selection_keeps_an_explicit_duplicate_tracking_target() -> None:
    module = load_module()
    current = pd.DataFrame(
        [_representative_catalog_row(module, "159516", "半导体设备ETF国泰", 100.0)]
    )
    catalog = pd.DataFrame(
        [
            *current.to_dict(orient="records"),
            _representative_catalog_row(
                module, "159558", "半导体设备ETF易方达", 1_000.0
            ),
        ]
    )

    selected = module.select_representative_universe(
        catalog,
        current,
        target_size=2,
        category_targets={"sector_theme": 2},
        minimum_turnover=0.0,
        required_symbols=("SZ159558",),
    )

    assert set(selected["symbol"]) == {"SZ159516", "SZ159558"}
    required = selected.set_index("symbol").loc["SZ159558"]
    assert required["selection_reason"] == "user_required"
    assert selected["tracking_key"].nunique() == 1


def test_representative_selection_defaults_missing_optional_catalog_columns() -> None:
    module = load_module()
    minimal = pd.DataFrame(
        [
            {
                "symbol": "SH510300",
                "code": "510300",
                "market": "sh",
                "name": "沪深300ETF",
            }
        ]
    )

    selected = module.select_representative_universe(
        minimal,
        minimal,
        target_size=1,
        category_targets={"core_broad": 1},
        minimum_turnover=0.0,
    )

    assert selected["symbol"].tolist() == ["SH510300"]
    assert not bool(selected.iloc[0]["in_sina"])
    assert not bool(selected.iloc[0]["in_ths"])
    assert float(selected.iloc[0]["sina_amount"]) == 0.0


def test_representative_default_quotas_only_support_the_registered_200_size() -> None:
    module = load_module()

    with pytest.raises(
        ValueError, match="default representative quotas support target_size=200"
    ):
        module.select_representative_universe(
            pd.DataFrame(),
            pd.DataFrame(),
            target_size=150,
        )


def test_expansion_request_interval_has_a_point_three_second_floor(
    tmp_path: Path,
) -> None:
    module = load_module()

    with pytest.raises(ValueError, match="request_interval must be at least 0.3"):
        module.expand_cn_etf_universe(
            quant_data_dir=tmp_path / "quant_data",
            target_size=200,
            end_date="2026-08-14",
            request_interval=0.2,
        )


def test_expansion_rejects_missing_required_symbol_without_target_capacity(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    current_universe = pd.read_parquet(current / "ETF_UNIVERSE.parquet")

    with pytest.raises(module.CnEtfDownloadError, match="target_size has no capacity"):
        module.expand_cn_etf_universe(
            quant_data_dir=root,
            target_size=len(current_universe),
            category_targets={"core_broad": len(current_universe)},
            end_date="2026-08-06",
            required_symbols=("SH588200",),
            request_interval=0.3,
        )


def test_expand_universe_downloads_only_new_symbols_and_preserves_old_hashes(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    current_universe = pd.read_parquet(current / "ETF_UNIVERSE.parquet")
    additions = pd.DataFrame(
        [
            _representative_catalog_row(module, "512480", "半导体ETF华夏", 900.0),
            _representative_catalog_row(module, "588200", "科创芯片ETF嘉实", 800.0),
        ]
    )
    catalog = pd.concat([current_universe, additions], ignore_index=True)
    monkeypatch.setattr(
        module,
        "fetch_etf_catalog",
        lambda **_kwargs: (
            catalog,
            {"sina_count": 4, "ths_count": 4, "reconciled_count": 4},
        ),
    )
    calls: list[str] = []

    def added_pair(symbol, metadata, **_kwargs):
        calls.append(symbol)
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

    monkeypatch.setattr(module, "fetch_tencent_pair", added_pair)
    old_hashes = {
        component: {
            symbol: module.sha256_file(current / component / f"{symbol}.parquet")
            for symbol in current_universe["symbol"].astype(str)
        }
        for component in ("raw", "qfq", "factors")
    }

    expanded = module.expand_cn_etf_universe(
        quant_data_dir=root,
        target_size=4,
        category_targets={"core_broad": 2, "sector_theme": 2},
        end_date="2026-08-06",
        generated_at="2026-08-07T12:00:00+08:00",
        request_interval=0.3,
        request_budget=20,
        minimum_turnover=0.0,
        minimum_history_rows=3,
        required_symbols=("SH588200",),
    )

    assert set(calls) == {"SH512480", "SH588200"}
    expanded_universe = pd.read_parquet(expanded / "ETF_UNIVERSE.parquet")
    assert len(expanded_universe) == 4
    assert set(current_universe["symbol"]).issubset(set(expanded_universe["symbol"]))
    assert (
        expanded_universe.set_index("symbol").loc["SH588200", "selection_reason"]
        == "user_required"
    )
    for component, hashes in old_hashes.items():
        for symbol, digest in hashes.items():
            assert (
                module.sha256_file(expanded / component / f"{symbol}.parquet") == digest
            )
    manifest = json.loads((expanded / "DATASET_MANIFEST.json").read_text())
    universe_manifest = json.loads((expanded / "UNIVERSE_MANIFEST.json").read_text())
    assert manifest["source"]["update_mode"] == "universe_expansion"
    assert manifest["source"]["parent_snapshot_id"] == current.name
    assert manifest["source"]["retained_symbol_count"] == 2
    assert manifest["source"]["added_symbol_count"] == 2
    assert manifest["source"]["required_symbols"] == ["SH588200"]
    assert manifest["source"]["universe_sha256"] == universe_manifest["universe_sha256"]
    assert universe_manifest["selected_count"] == 4


def test_expand_universe_failure_keeps_latest_pointer_unchanged(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    current_universe = pd.read_parquet(current / "ETF_UNIVERSE.parquet")
    additions = pd.DataFrame(
        [
            _representative_catalog_row(module, "512480", "半导体ETF华夏", 900.0),
            _representative_catalog_row(module, "588200", "科创芯片ETF嘉实", 800.0),
        ]
    )
    catalog = pd.concat([current_universe, additions], ignore_index=True)
    monkeypatch.setattr(
        module,
        "fetch_etf_catalog",
        lambda **_kwargs: (catalog, {"reconciled_count": 4}),
    )

    def fail_one(symbol, metadata, **_kwargs):
        if symbol == "SH588200":
            raise ConnectionError("provider unavailable")
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", fail_one)
    latest_path = root / "parquet" / "etf_1d" / "cn_etf_daily" / "LATEST.json"
    before = latest_path.read_bytes()

    with pytest.raises(module.CnEtfDownloadError, match="incomplete"):
        module.expand_cn_etf_universe(
            quant_data_dir=root,
            target_size=4,
            category_targets={"core_broad": 2, "sector_theme": 2},
            end_date="2026-08-06",
            generated_at="2026-08-07T12:00:00+08:00",
            request_interval=0.3,
            request_budget=20,
            minimum_turnover=0.0,
            minimum_history_rows=3,
        )

    assert latest_path.read_bytes() == before
    assert json.loads(latest_path.read_text())["snapshot_id"] == current.name


def test_revised_expansion_reuses_prior_contract_cache_and_downloads_only_replacement(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, current = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    current_universe = pd.read_parquet(current / "ETF_UNIVERSE.parquet")
    candidates = pd.DataFrame(
        [
            _representative_catalog_row(module, "512480", "半导体ETF华夏", 900.0),
            _representative_catalog_row(module, "588200", "科创芯片ETF嘉实", 800.0),
            _representative_catalog_row(module, "515880", "通信ETF国泰", 700.0),
        ]
    )
    catalog = pd.concat([current_universe, candidates], ignore_index=True)
    monkeypatch.setattr(
        module,
        "fetch_etf_catalog",
        lambda **_kwargs: (catalog, {"reconciled_count": 5}),
    )
    calls: list[str] = []

    def first_pair(symbol, metadata, **_kwargs):
        calls.append(symbol)
        if symbol == "SH588200":
            raise ConnectionError("provider pair is incompatible")
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", first_pair)
    with pytest.raises(module.CnEtfDownloadError, match="incomplete"):
        module.expand_cn_etf_universe(
            quant_data_dir=root,
            target_size=4,
            category_targets={"core_broad": 2, "sector_theme": 2},
            end_date="2026-08-06",
            generated_at="2026-08-07T12:00:00+08:00",
            request_interval=0.3,
            request_budget=20,
            minimum_turnover=0.0,
            minimum_history_rows=3,
        )
    assert set(calls) == {"SH512480", "SH588200"}

    calls.clear()

    def replacement_pair(symbol, metadata, **_kwargs):
        calls.append(symbol)
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", replacement_pair)
    expanded = module.expand_cn_etf_universe(
        quant_data_dir=root,
        target_size=4,
        category_targets={"core_broad": 2, "sector_theme": 2},
        excluded_symbols=("SH588200",),
        end_date="2026-08-06",
        generated_at="2026-08-07T12:30:00+08:00",
        request_interval=0.3,
        request_budget=20,
        minimum_turnover=0.0,
        minimum_history_rows=3,
    )

    assert calls == ["SH515880"]
    final_universe = pd.read_parquet(expanded / "ETF_UNIVERSE.parquet")
    assert set(final_universe["symbol"]) == {
        "SH510050",
        "SZ159915",
        "SH512480",
        "SH515880",
    }
    manifest = json.loads((expanded / "DATASET_MANIFEST.json").read_text())
    assert manifest["source"]["excluded_symbols"] == ["SH588200"]
    assert manifest["source"]["reused_prior_contract_symbol_count"] == 1


def test_metadata_reissue_keeps_the_original_51_style_parent_in_immutable_history(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    root, parent = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    current_universe = pd.read_parquet(parent / "ETF_UNIVERSE.parquet")
    additions = pd.DataFrame(
        [
            _representative_catalog_row(module, "512480", "半导体ETF华夏", 900.0),
            _representative_catalog_row(module, "588200", "科创芯片ETF嘉实", 800.0),
        ]
    )
    catalog = pd.concat([current_universe, additions], ignore_index=True)
    monkeypatch.setattr(
        module,
        "fetch_etf_catalog",
        lambda **_kwargs: (catalog, {"reconciled_count": 4}),
    )

    def added_pair(symbol, metadata, **_kwargs):
        raw = paired_frame(
            symbol,
            name=metadata["name"],
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return raw, raw.assign(
            source=module.TENCENT_ADJUSTED_SOURCE_NAME,
            provider_response_key="qfqday",
        )

    monkeypatch.setattr(module, "fetch_tencent_pair", added_pair)
    expanded = module.expand_cn_etf_universe(
        quant_data_dir=root,
        target_size=4,
        category_targets={"core_broad": 2, "sector_theme": 2},
        end_date="2026-08-06",
        generated_at="2026-08-07T12:00:00+08:00",
        request_interval=0.3,
        request_budget=20,
        minimum_turnover=0.0,
        minimum_history_rows=3,
    )

    corrected = module.reissue_cn_etf_expansion_metadata(
        quant_data_dir=root,
        generated_at="2026-08-07T12:30:00+08:00",
    )

    assert corrected != expanded
    snapshots = corrected.parents[1] / "snapshots"
    assert {path.name for path in snapshots.iterdir()} == {
        parent.name,
        expanded.name,
        corrected.name,
    }
    pointer = json.loads((corrected.parents[1] / "LATEST.json").read_text())
    assert pointer["published_at"] == "2026-08-07T12:30:00+08:00"
    manifest = json.loads((corrected / "DATASET_MANIFEST.json").read_text())
    universe_manifest = json.loads((corrected / "UNIVERSE_MANIFEST.json").read_text())
    assert manifest["source"]["parent_snapshot_id"] == parent.name
    assert manifest["source"]["metadata_correction_of_snapshot_id"] == expanded.name
    assert manifest["source"]["universe_sha256"] == universe_manifest["universe_sha256"]
    assert universe_manifest["selected_count"] == 4


def _baostock_payload(symbol, flag, *, dates=None, closes=None):
    code = symbol[:2].lower() + '.' + symbol[2:]
    dates = dates or ['2026-08-04', '2026-08-05', '2026-08-06', '2026-08-07']
    closes = closes or [1, 1.1, 1.2, 1.3]
    return {
        'request': {'code': code, 'fields': 'date,code,open,high,low,close,volume,amount,tradestatus',
                    'start_date': '2026-08-04', 'end_date': '2026-08-07',
                    'frequency': 'd', 'adjustflag': flag},
        'error_code': '0', 'error_msg': 'success',
        'fields': ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'tradestatus'],
        'rows': [[day, code, str(c), str(c * 1.01), str(c * .99), str(c), '10043', '1000', '1']
                 for day, c in zip(dates, closes)],
    }


def test_baostock_pair_retains_native_evidence_and_converts_volume(monkeypatch):
    module = load_module()
    calls = []

    def sdk_window(request, *, timeout):
        calls.append(request)
        return _baostock_payload('SH510050', request['adjustflag'])

    monkeypatch.setattr(module, '_read_baostock_window', sdk_window)
    raw, qfq = module.fetch_baostock_pair(
        'SH510050', {'code': '510050', 'market': 'sh', 'name': 'ETF'},
        start_date='2026-08-04', end_date='2026-08-07', session=None,
        request_gate=module.RequestGate(0, 2), retries=1,
    )
    assert raw['volume'].tolist() == [100.0] * 4
    assert raw['native_volume'].tolist() == [10043] * 4
    assert raw['amount'].tolist() == [1000] * 4
    assert raw['source'].unique().tolist() == ['baostock.etf_daily_raw']
    assert qfq['provider_response_key'].unique().tolist() == ['baostock_adjustflag_2']
    assert [p['adjustflag'] for p in calls] == ['3', '2']
    assert raw.attrs['source_receipt']['responses'][0]['rows'][0][6] == '10043'


@pytest.mark.parametrize('bad', ['symbol', 'duplicate', 'outside', 'halted', 'volume'])
def test_baostock_pair_rejects_invalid_or_fake_trading_rows(monkeypatch, bad):
    module = load_module()

    def sdk_window(request, *, timeout):
        response = _baostock_payload('SH510050', request['adjustflag'])
        if bad == 'symbol':
            response['rows'][0][1] = 'sz.159915'
        if bad == 'duplicate':
            response['rows'].append(response['rows'][0])
        if bad == 'outside':
            response['rows'][0][0] = '2004-01-01'
        if bad == 'halted':
            for row in response['rows']:
                row[8] = '0'
        if bad == 'volume' and request['adjustflag'] == '2':
            response['rows'][0][6] = '20000'
        return response

    monkeypatch.setattr(module, '_read_baostock_window', sdk_window)
    with pytest.raises(module.CnEtfDownloadError):
        module.fetch_baostock_pair(
            'SH510050', {'code': '510050', 'market': 'sh', 'name': 'ETF'},
            start_date='2026-08-04', end_date='2026-08-07', session=None,
            request_gate=module.RequestGate(0, 2), retries=1,
        )


def test_baostock_update_replays_receipt_and_preserves_history(monkeypatch, tmp_path):
    module = load_module()
    root, old = _seed_current_paired_snapshot(module, monkeypatch, tmp_path)
    calls = []

    def sdk_window(request, *, timeout):
        calls.append(request)
        return _baostock_payload(request['code'][:2].upper() + request['code'][3:], request['adjustflag'])

    monkeypatch.setattr(module, '_read_baostock_window', sdk_window)
    prepared = module.update_cn_etf_snapshot(
        quant_data_dir=root, end_date='2026-08-07', provider='baostock', publish=False,
    )
    assert len(calls) == 4
    receipt = next((root / 'cache/cn-etf-incremental').glob('*/SOURCE_RESPONSE.json'))
    assert json.loads(receipt.read_text())['volume_rule'] == 'floor(shares / 100 + 0.5) to legacy whole lots'
    new = module.update_cn_etf_snapshot(
        quant_data_dir=root, end_date='2026-08-07', provider='baostock',
    )
    assert len(calls) == 4
    assert prepared != new
    for symbol in ['SH510050', 'SZ159915']:
        before = pd.read_parquet(old / 'raw' / f'{symbol}.parquet')
        after = pd.read_parquet(new / 'raw' / f'{symbol}.parquet')
        pd.testing.assert_frame_equal(after.iloc[:3][before.columns].reset_index(drop=True), before)
        assert after.iloc[-1]['source'].startswith('baostock')
    assert module.update_cn_etf_snapshot(quant_data_dir=root, end_date='2026-08-07', provider='baostock') == new
    assert len(calls) == 4
    # A source response must remain bound even on the cached preparation route.
    receipt.write_text('{}')
    with pytest.raises(module.DataContractError):
        module.update_cn_etf_snapshot(
            quant_data_dir=root, base_snapshot=old, end_date='2026-08-07',
            provider='baostock', publish=False,
        )


def test_baostock_sdk_child_bounds_logout_and_returns_exact_request(monkeypatch, tmp_path):
    module = load_module()
    # Only the external SDK boundary is replaced; run the actual owned child.
    (tmp_path / 'baostock.py').write_text('''
class Result:
    error_code = '0'
    error_msg = 'success'
    fields = ['date', 'code']
    def next(self): return False

def login(): return Result()
def query_history_k_data_plus(**kwargs): return Result()
def logout(): pass
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    request = {'code': 'sh.510050'}
    response = module._read_baostock_window(request, timeout=5)
    assert response['request'] == request
    assert response['rows'] == []
    (tmp_path / 'baostock.py').write_text((tmp_path / 'baostock.py').read_text().replace(
        'def logout(): pass', 'def logout():\n    import time\n    time.sleep(60)'))
    import time
    started = time.monotonic()
    with pytest.raises(module.CnEtfDownloadError, match='bounded request failed'):
        module._read_baostock_window(request, timeout=.3)
    assert time.monotonic() - started < 3


def test_baostock_half_lot_matches_legacy_half_up_precision(monkeypatch):
    module = load_module()

    def sdk_window(request, *, timeout):
        response = _baostock_payload('SH510050', request['adjustflag'])
        response['rows'][0][6] = '10050'
        return response

    monkeypatch.setattr(module, '_read_baostock_window', sdk_window)
    raw, qfq = module.fetch_baostock_pair(
        'SH510050', {'code': '510050', 'market': 'sh', 'name': 'ETF'},
        start_date='2026-08-04', end_date='2026-08-07', session=None,
        request_gate=module.RequestGate(0, 2), retries=1,
    )
    assert raw['native_volume'].iloc[0] == 10050
    assert raw['volume'].iloc[0] == qfq['volume'].iloc[0] == 101
