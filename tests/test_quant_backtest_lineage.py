# -*- coding: utf-8 -*-
"""Regression tests for quant_data lineage in strategy backtest reports."""

from datetime import date
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


def test_script_strategy_report_preserves_complete_quant_data_identity() -> None:
    service = BacktestService.__new__(BacktestService)

    from src.core.strategy_backtest import TargetAllocation

    with patch(
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
        )

    # The script-strategy report carries the complete quant_data snapshot
    # identity used to assemble the universe pool (dataset/snapshot/manifest
    # hash/content hash).
    quality = report["data_quality"]
    assert quality["dataset_id"] == "cn_etf_daily"
    assert quality["snapshot_id"]
    assert len(quality["manifest_sha256"]) == 64
    assert len(quality["content_sha256"]) == 64
    assert quality["pool"] == "etf"
    assert quality["price_basis"] == "qfq"
