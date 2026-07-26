from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import akshare as ak
import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "download_etf_momentum_data.py"
    )
    spec = importlib.util.spec_from_file_location("download_etf_momentum_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fetch_one_normalizes_sina_rows(monkeypatch) -> None:
    module = load_module()
    raw = pd.DataFrame(
        {
            "date": ["2026-07-08", "2026-07-07"],
            "open": [3.0, 2.9],
            "high": [3.1, 3.0],
            "low": [2.9, 2.8],
            "close": [3.05, 2.95],
            "volume": [100, 90],
            "amount": [300, 260],
        }
    )
    monkeypatch.setattr(ak, "fund_etf_hist_sina", lambda symbol: raw)

    result = module.fetch_one(
        "SZ159915", start_date="2026-07-01", end_date="2026-07-08"
    )

    assert result["symbol"].unique().tolist() == ["SZ159915"]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-07",
        "2026-07-08",
    ]
    assert result["source"].unique().tolist() == [module.SOURCE_NAME]


def test_daily_pipeline_end_alias_is_supported(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        sys, "argv", ["download_etf_momentum_data.py", "--end", "20260724"]
    )

    args = module.parse_args()

    assert args.end_date == "20260724"


def test_download_writes_only_the_two_etf_files_and_source_manifest(
    monkeypatch, tmp_path
) -> None:
    module = load_module()

    def fake_fetch(symbol: str, *, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "name": [module.DEFAULT_SYMBOLS[symbol]["name"]],
                "date": [pd.Timestamp("2026-07-24")],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [10.0],
                "amount": [10.5],
                "source": [module.SOURCE_NAME],
            }
        )

    monkeypatch.setattr(module, "fetch_one", fake_fetch)
    manifest_path = module.download(
        output_dir=tmp_path / "parquet" / "etf_1d" / "etf_momentum_rotation" / "v1",
        start_date="2026-07-01",
        end_date="2026-07-24",
    )

    output_dir = manifest_path.parent
    assert sorted(path.name for path in output_dir.glob("*.parquet")) == [
        "SH513100.parquet",
        "SZ159915.parquet",
    ]
    assert manifest_path.name == "SOURCE_MANIFEST.json"
    assert not (tmp_path / "parquet" / "ashare_1d").exists()


def test_fetch_one_adjusted_uses_eastmoney_qfq_and_normalizes_rows(monkeypatch) -> None:
    module = load_module()
    calls: list[dict[str, str]] = []
    raw = pd.DataFrame(
        {
            "日期": ["2026-07-08", "2026-07-07"],
            "开盘": [3.0, 2.9],
            "最高": [3.1, 3.0],
            "最低": [2.9, 2.8],
            "收盘": [3.05, 2.95],
            "成交量": [100, 90],
            "成交额": [300, 260],
        }
    )

    def fake_hist(**kwargs):
        calls.append(kwargs)
        return raw

    monkeypatch.setattr(ak, "fund_etf_hist_em", fake_hist)

    result = module.fetch_one_adjusted(
        "SZ159915", start_date="2026-07-01", end_date="2026-07-08"
    )

    assert calls == [
        {
            "symbol": "159915",
            "period": "daily",
            "start_date": "20260701",
            "end_date": "20260708",
            "adjust": "qfq",
        }
    ]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-07-07",
        "2026-07-08",
    ]
    assert result["source"].unique().tolist() == [module.ADJUSTED_SOURCE_NAME]
    assert result["adjustment"].unique().tolist() == ["qfq"]


def test_fetch_one_tencent_adjusted_normalizes_qfq_rows() -> None:
    module = load_module()
    calls: list[dict[str, object]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "code": 0,
                "msg": "",
                "data": {
                    "sh513100": {
                        "qfqday": [
                            ["2022-01-12", "1.032", "1.038", "1.039", "1.032", "218919.000"],
                            ["2022-01-14", "1.010", "1.015", "1.016", "1.007", "1256441.000"],
                        ]
                    }
                },
            }

    class Session:
        def get(self, url: str, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    result = module.fetch_one_tencent_adjusted(
        "SH513100",
        start_date="2022-01-01",
        end_date="2022-01-31",
        session=Session(),
    )

    assert len(calls) == 1
    assert calls[0]["params"] == {
        "param": "sh513100,day,2022-01-01,2022-01-31,400,qfq"
    }
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-01-12",
        "2022-01-14",
    ]
    assert result["close"].tolist() == [1.038, 1.015]
    assert result["source"].unique().tolist() == [
        module.TENCENT_ADJUSTED_SOURCE_NAME
    ]
    assert result["adjustment"].unique().tolist() == ["qfq"]


def test_fetch_one_tencent_raw_uses_raw_kline_endpoint() -> None:
    module = load_module()
    calls: list[dict[str, object]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "code": 0,
                "msg": "",
                "data": {
                    "sz159915": {
                        "day": [
                            ["2026-07-24", "3.100", "3.120", "3.130", "3.090", "1000.000"]
                        ]
                    }
                },
            }

    class Session:
        def get(self, url: str, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    result = module.fetch_one_tencent_raw(
        "SZ159915",
        start_date="2026-07-24",
        end_date="2026-07-24",
        session=Session(),
    )

    assert calls[0]["url"] == module.TENCENT_RAW_KLINE_URL
    assert calls[0]["params"] == {
        "param": "sz159915,day,2026-07-24,2026-07-24,400"
    }
    assert result["source"].unique().tolist() == [module.TENCENT_RAW_SOURCE_NAME]


def test_adjusted_download_writes_isolated_manifest(monkeypatch, tmp_path) -> None:
    module = load_module()

    def fake_fetch(symbol: str, *, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "name": [module.DEFAULT_SYMBOLS[symbol]["name"]],
                "date": [pd.Timestamp("2026-07-24")],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [10.0],
                "amount": [10.5],
                "source": [module.ADJUSTED_SOURCE_NAME],
                "adjustment": ["qfq"],
            }
        )

    monkeypatch.setattr(module, "fetch_one_adjusted", fake_fetch)
    output_dir = tmp_path / "eastmoney_qfq"
    manifest_path = module.download_adjusted(
        output_dir=output_dir,
        start_date="2026-07-01",
        end_date="2026-07-24",
    )

    assert sorted(path.name for path in output_dir.glob("*.parquet")) == [
        "SH513100.parquet",
        "SZ159915.parquet",
    ]
    assert manifest_path == output_dir / "SOURCE_MANIFEST.json"
    assert '"adjustment": "qfq"' in manifest_path.read_text(encoding="utf-8")
    assert '"source": "akshare.fund_etf_hist_em_qfq"' in manifest_path.read_text(
        encoding="utf-8"
    )


def test_adjusted_download_failure_preserves_existing_cache(monkeypatch, tmp_path) -> None:
    module = load_module()
    output_dir = tmp_path / "eastmoney_qfq"
    output_dir.mkdir()
    existing = output_dir / "SZ159915.parquet"
    existing.write_bytes(b"existing-cache")

    def fail_fetch(symbol: str, *, start_date: str, end_date: str) -> pd.DataFrame:
        raise ConnectionError(f"offline: {symbol}")

    monkeypatch.setattr(module, "fetch_one_adjusted", fail_fetch)

    with pytest.raises(ConnectionError, match="offline"):
        module.download_adjusted(
            output_dir=output_dir,
            start_date="2026-07-01",
            end_date="2026-07-24",
        )

    assert existing.read_bytes() == b"existing-cache"
    assert not (output_dir / "SH513100.parquet").exists()


def test_main_keeps_raw_update_success_when_adjusted_source_is_unavailable(
    monkeypatch, tmp_path, capsys
) -> None:
    module = load_module()
    raw_manifest = tmp_path / "raw" / "SOURCE_MANIFEST.json"
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            output_dir=tmp_path / "raw",
            adjusted_output_dir=tmp_path / "eastmoney_qfq",
            tencent_adjusted_output_dir=tmp_path / "tencent_qfq",
            start_date="2019-01-01",
            end_date="2026-07-24",
            incremental=True,
            overlap_days=5,
            skip_adjusted=False,
        ),
    )

    def fake_raw(**kwargs):
        calls.append("raw")
        return raw_manifest

    def fail_adjusted(**kwargs):
        calls.append("adjusted")
        raise ConnectionError("Eastmoney unavailable")

    def fallback_adjusted(**kwargs):
        calls.append("tencent")
        return tmp_path / "tencent_qfq" / "SOURCE_MANIFEST.json"

    monkeypatch.setattr(module, "download", fake_raw)
    monkeypatch.setattr(module, "download_adjusted", fail_adjusted)
    monkeypatch.setattr(module, "download_tencent_adjusted", fallback_adjusted)

    assert module.main() == 0
    assert calls == ["raw", "adjusted", "tencent"]
    assert "Eastmoney qfq refresh unavailable" in capsys.readouterr().err


def test_tencent_adjusted_download_writes_separate_manifest(monkeypatch, tmp_path) -> None:
    module = load_module()

    def fake_fetch(symbol: str, *, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "name": [module.DEFAULT_SYMBOLS[symbol]["name"]],
                "date": [pd.Timestamp("2026-07-24")],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [10.0],
                "source": [module.TENCENT_ADJUSTED_SOURCE_NAME],
                "adjustment": ["qfq"],
            }
        )

    monkeypatch.setattr(module, "fetch_one_tencent_adjusted", fake_fetch)
    output_dir = tmp_path / "tencent_qfq"
    manifest_path = module.download_tencent_adjusted(
        output_dir=output_dir,
        start_date="2026-07-01",
        end_date="2026-07-24",
    )

    payload = manifest_path.read_text(encoding="utf-8")
    assert '"source": "tencent.ifzq_qfq"' in payload
    assert '"ratio": "1:5"' in payload
    assert sorted(path.name for path in output_dir.glob("*.parquet")) == [
        "SH513100.parquet",
        "SZ159915.parquet",
    ]


def paired_frame(
    symbol: str,
    *,
    source: str,
    closes: list[float],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    dates = dates or ["2022-01-12", "2022-01-14", "2026-07-24"]
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(dates),
            "name": [symbol] * len(dates),
            "date": pd.to_datetime(dates),
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [100.0] * len(dates),
            "source": [source] * len(dates),
        }
    )


def test_build_paired_factors_records_full_date_curve_and_split() -> None:
    module = load_module()
    raw = paired_frame(
        "SH513100",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[5.19, 1.015, 2.0],
    )
    adjusted = paired_frame(
        "SH513100",
        source=module.TENCENT_ADJUSTED_SOURCE_NAME,
        closes=[1.038, 1.015, 2.0],
    )

    factors = module.build_paired_factors("SH513100", raw, adjusted)

    assert factors["date"].tolist() == raw["date"].tolist()
    assert factors["factor"].round(4).tolist() == [0.2, 1.0, 1.0]
    assert factors["factor_source"].unique().tolist() == [
        "derived_same_provider_raw_qfq_close_ratio"
    ]


def test_build_paired_factors_rejects_ohlc_that_does_not_follow_factor() -> None:
    module = load_module()
    raw = paired_frame(
        "SZ159915",
        source=module.TENCENT_RAW_SOURCE_NAME,
        closes=[1.0, 1.1, 1.2],
    )
    adjusted = raw.assign(source=module.TENCENT_ADJUSTED_SOURCE_NAME)
    adjusted.loc[0, "open"] = 9.0

    with pytest.raises(module.DataContractError, match="open relationship"):
        module.build_paired_factors("SZ159915", raw, adjusted)


def test_corporate_actions_include_announcement_and_inferred_factor_events() -> None:
    module = load_module()
    actions = module._corporate_actions_frame(
        {
            "SH513100": pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2022-01-12", "2022-01-14", "2026-07-24"]
                    ),
                    "factor": [0.2, 1.0, 1.0],
                }
            ),
            "SZ159915": pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2026-01-02", "2026-01-05", "2026-07-24"]
                    ),
                    "factor": [0.98, 1.0, 1.0],
                }
            ),
        },
        provider="tencent",
    )

    split = actions.loc[actions["action_type"] == "share_split"].iloc[0]
    inferred = actions.loc[
        actions["action_type"] == "provider_adjustment_event"
    ].iloc[0]
    assert split["symbol"] == "SH513100"
    assert split["factor_before"] == pytest.approx(0.2)
    assert split["factor_after"] == pytest.approx(1.0)
    assert inferred["symbol"] == "SZ159915"
    assert inferred["evidence_method"] == "derived_from_same_provider_raw_qfq_pair"


def test_publish_canonical_snapshot_writes_paired_raw_and_qfq_snapshots(
    monkeypatch, tmp_path
) -> None:
    module = load_module()

    def fake_pair(provider, symbol, *, start_date, end_date):
        assert provider == "tencent"
        if symbol == "SH513100":
            raw_close = [5.19, 1.015, 2.0]
            qfq_close = [1.038, 1.015, 2.0]
        else:
            raw_close = [1.0, 1.1, 1.2]
            qfq_close = [1.0, 1.1, 1.2]
        return (
            paired_frame(
                symbol,
                source=module.TENCENT_RAW_SOURCE_NAME,
                closes=raw_close,
            ),
            paired_frame(
                symbol,
                source=module.TENCENT_ADJUSTED_SOURCE_NAME,
                closes=qfq_close,
            ),
        )

    monkeypatch.setattr(module, "fetch_provider_pair", fake_pair)
    monkeypatch.setattr(module, "_sync_root_manifest", lambda root: None)

    published = module.publish_canonical_snapshot(
        quant_data_dir=tmp_path / "quant_data",
        start_date="2019-01-01",
        end_date="2026-07-24",
        generated_at="2026-07-25T23:00:00+08:00",
    )

    qfq_manifest = json.loads(
        (published / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    raw_root = (
        tmp_path
        / "quant_data"
        / "parquet"
        / "etf_1d"
        / "etf_momentum_daily_raw"
    )
    raw_pointer = json.loads((raw_root / "LATEST.json").read_text(encoding="utf-8"))
    factors = pd.read_parquet(published / "factors" / "SH513100.parquet")

    assert qfq_manifest["price_basis"] == "qfq"
    assert qfq_manifest["adjustment"]["method"] == "provider_native_paired"
    assert (
        qfq_manifest["adjustment"]["paired_raw_snapshot_id"]
        == raw_pointer["snapshot_id"]
    )
    assert factors["factor"].round(4).tolist() == [0.2, 1.0, 1.0]
    assert qfq_manifest["symbol_count"] == 2
    assert qfq_manifest["latest_trade_date"] == "2026-07-24"


def test_publish_canonical_snapshot_allows_different_historical_etf_calendars(
    monkeypatch, tmp_path
) -> None:
    module = load_module()

    def fake_pair(provider, symbol, *, start_date, end_date):
        assert provider == "tencent"
        if symbol == "SH513100":
            dates = ["2022-01-12", "2022-01-14", "2026-07-24"]
            raw_close = [5.19, 1.015, 2.0]
            qfq_close = [1.038, 1.015, 2.0]
        else:
            dates = ["2022-01-13", "2022-01-14", "2026-07-24"]
            raw_close = [1.0, 1.1, 1.2]
            qfq_close = raw_close
        return (
            paired_frame(
                symbol,
                source=module.TENCENT_RAW_SOURCE_NAME,
                closes=raw_close,
                dates=dates,
            ),
            paired_frame(
                symbol,
                source=module.TENCENT_ADJUSTED_SOURCE_NAME,
                closes=qfq_close,
                dates=dates,
            ),
        )

    monkeypatch.setattr(module, "fetch_provider_pair", fake_pair)
    monkeypatch.setattr(module, "_sync_root_manifest", lambda root: None)

    published = module.publish_canonical_snapshot(
        quant_data_dir=tmp_path / "quant_data",
        start_date="2019-01-01",
        end_date="2026-07-24",
        provider_order=("tencent",),
        generated_at="2026-07-25T23:00:00+08:00",
    )

    manifest = json.loads(
        (published / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["latest_trade_date"] == "2026-07-24"
    assert manifest["symbol_count"] == 2


def test_publish_canonical_snapshot_does_not_publish_partial_two_etf_data(
    monkeypatch, tmp_path
) -> None:
    module = load_module()

    def fake_pair(provider, symbol, *, start_date, end_date):
        if symbol == "SH513100":
            raise ConnectionError("second ETF failed")
        frame = paired_frame(
            symbol,
            source=module.TENCENT_RAW_SOURCE_NAME,
            closes=[1.0, 1.1, 1.2],
        )
        return frame, frame.assign(source=module.TENCENT_ADJUSTED_SOURCE_NAME)

    monkeypatch.setattr(module, "fetch_provider_pair", fake_pair)

    with pytest.raises(RuntimeError, match="all ETF paired providers failed"):
        module.publish_canonical_snapshot(
            quant_data_dir=tmp_path / "quant_data",
            start_date="2019-01-01",
            end_date="2026-07-24",
            provider_order=("tencent",),
            generated_at="2026-07-25T23:00:00+08:00",
        )

    assert not list((tmp_path / "quant_data").rglob("LATEST.json"))
