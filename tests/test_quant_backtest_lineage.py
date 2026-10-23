# -*- coding: utf-8 -*-
"""Regression tests for quant_data lineage in strategy backtest reports."""

from datetime import date
from dataclasses import replace
from unittest.mock import patch

import pytest

from src.core.strategy_backtest import PriceBar
from src.repositories.quant_price_repo import QuantPriceLoad
from src.services.backtest_service import BacktestService


@pytest.fixture
def price_data() -> QuantPriceLoad:
    return QuantPriceLoad(
        symbols=("SZ159915",),
        bars=(
            PriceBar(
                symbol="SZ159915",
                date=date(2026, 8, 6),
                open=1.0,
                close=1.0,
            ),
            PriceBar(
                symbol="SZ159915",
                date=date(2026, 8, 7),
                open=1.1,
                close=1.1,
            ),
        ),
        dataset_id="cn_etf_daily",
        snapshot_id="snapshot-1",
        manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        latest_trade_date="2026-08-07",
        price_basis="qfq",
        data_kind="paired_daily_bars",
        provider="quant_data",
    )


def _assert_complete_identity(report: dict) -> None:
    assert {
        key: report["data_quality"][key]
        for key in (
            "dataset_id",
            "snapshot_id",
            "manifest_sha256",
            "content_sha256",
            "latest_trade_date",
            "price_basis",
            "data_kind",
        )
    } == {
        "dataset_id": "cn_etf_daily",
        "snapshot_id": "snapshot-1",
        "manifest_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "latest_trade_date": "2026-08-07",
        "price_basis": "qfq",
        "data_kind": "paired_daily_bars",
    }


def test_form_strategy_report_preserves_complete_quant_data_identity(price_data) -> None:
    service = BacktestService.__new__(BacktestService)

    with patch(
        "src.services.backtest_service.QuantPriceRepository"
    ) as repository_cls, patch(
        "src.services.backtest_service.run_portfolio_backtest",
        return_value={"status": "completed"},
    ):
        repository_cls.return_value.load.return_value = price_data
        report = service.run_strategy(
            strategy_id="buy_and_hold",
            symbols=["159915"],
        )

    _assert_complete_identity(report)


def test_form_strategy_report_preserves_production_v2_release_identity(price_data) -> None:
    service = BacktestService.__new__(BacktestService)
    native = replace(
        price_data, snapshot_id=None, manifest_sha256=None, content_sha256=None,
        latest_trade_date=None, data_kind="daily_bars",
        dataset_set_version="production.v2", generation=10,
        release_set_sha256="b" * 64,
        module_identities={"cn_etf_daily": {"module_release_sha256": "c" * 64}},
    )
    with patch("src.services.backtest_service.QuantPriceRepository") as repository_cls, patch(
        "src.services.backtest_service.run_portfolio_backtest",
        return_value={"status": "completed"},
    ):
        repository_cls.return_value.load.return_value = native
        report = service.run_strategy(
            strategy_id="buy_and_hold", symbols=["159915"],
            start_date=date(2026, 8, 6), end_date=date(2026, 8, 7),
        )
        repository_cls.return_value.load.assert_called_once_with(
            ["159915"], start_date=date(2026, 8, 6), end_date=date(2026, 8, 7),
        )
    quality = report["data_quality"]
    assert quality["dataset_set_version"] == "production.v2"
    assert quality["release_set_sha256"] == "b" * 64
    assert quality["module_identities"]["cn_etf_daily"]["module_release_sha256"] == "c" * 64
    assert quality["snapshot_id"] is None


def test_script_strategy_report_preserves_production_v2_release_identity() -> None:
    service = BacktestService.__new__(BacktestService)

    from src.core.strategy_backtest import TargetAllocation

    with patch(
        "src.services.backtest_service.load_market_view",
        return_value=(
            object(),
            {
                "bundle_dir": "/tmp/test-universe", "dataset_id": "cn_etf_daily",
                "dataset_set_version": "production.v2", "generation": 10,
                "release_set_sha256": "b" * 64,
                "release": {"generation": 10, "release_set_sha256": "b" * 64,
                            "pointer_sha256": "a" * 64},
                "datasets": {"cn_etf_daily": {"module_release_sha256": "c" * 64}},
            },
        ),
    ), patch(
        "src.services.backtest_service.load_benchmark_series", return_value={},
    ), patch(
        "src.services.backtest_service.run_script_in_sandbox",
        return_value=(
            [
                TargetAllocation(
                    source_id="script-1",
                    signal_date=date(2026, 8, 6),
                    target_weights={"SZ159915": 1.0},
                    reason="test",
                )
            ],
            [],
            {"api_version": "v1"},
        ),
    ), patch(
        "src.services.backtest_service.run_portfolio_backtest_view",
        return_value={"status": "completed", "metrics": {}, "strategy": {}},
    ):
        report = service.run_script_strategy(
            script="def on_bar(context, data):\n    pass",
            symbols=["159915"],
            pool="etf",
            start_date=date(2026, 8, 6), end_date=date(2026, 8, 7),
        )

    quality = report["data_quality"]
    assert quality["dataset_id"] == "cn_etf_daily"
    assert quality["dataset_set_version"] == "production.v2"
    assert quality["release_set_sha256"] == "b" * 64
    assert quality["datasets"]["cn_etf_daily"]["module_release_sha256"] == "c" * 64
    assert quality["snapshot_id"] is None
    assert quality["pool"] == "etf"
    assert quality["price_basis"] == "qfq"
