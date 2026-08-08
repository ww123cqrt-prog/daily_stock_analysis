from __future__ import annotations

import importlib.util
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


def test_main_delegates_to_unified_etf_dataset(
    monkeypatch, tmp_path, capsys
) -> None:
    module = load_module()
    published = tmp_path / "cn_etf_daily" / "snapshots" / "snapshot-001"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            quant_data_root=tmp_path / "quant_data",
            start_date="2019-01-01",
            end_date="2026-07-24",
        ),
    )
    monkeypatch.setattr(
        module,
        "publish_canonical_snapshot",
        lambda **kwargs: calls.append(kwargs) or published,
    )

    assert module.main() == 0
    assert calls == [
        {
            "quant_data_dir": tmp_path / "quant_data",
            "start_date": "2019-01-01",
            "end_date": "2026-07-24",
        }
    ]
    assert str(published) in capsys.readouterr().out


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


def test_publish_canonical_snapshot_delegates_to_unified_dataset(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    published = tmp_path / "cn_etf_daily" / "snapshots" / "snapshot-001"
    calls: list[dict[str, object]] = []
    fake_module = SimpleNamespace(
        update_cn_etf_snapshot=lambda **kwargs: calls.append(kwargs) or published
    )
    monkeypatch.setitem(sys.modules, "download_cn_etf_data", fake_module)

    result = module.publish_canonical_snapshot(
        quant_data_dir=tmp_path / "quant_data",
        start_date="2019-01-01",
        end_date="2026-07-24",
        provider_order=("tencent",),
        generated_at="2026-07-25T23:00:00+08:00",
    )

    assert result == published
    assert calls == [
        {
            "quant_data_dir": tmp_path / "quant_data",
            "end_date": "2026-07-24",
            "generated_at": "2026-07-25T23:00:00+08:00",
        }
    ]
