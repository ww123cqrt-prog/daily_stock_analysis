from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def client() -> TestClient:
    from api.v1.endpoints.etf_momentum_strategy import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/research")
    return TestClient(app)


def test_etf_momentum_strategy_api_returns_canonical_report(tmp_path: Path) -> None:
    report_path = tmp_path / "strategy_report_latest.json"
    payload = {
        "schema_version": 1,
        "status": "manual_review_required",
        "summary": {"data_status": "unverified_data"},
        "chart_data": {"wealth": [{"date": "2026-07-24", "strategy": 1.2}]},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch(
        "api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_PATH", report_path
    ):
        response = client().get("/api/v1/research/strategies/etf-momentum")

    assert response.status_code == 200
    assert response.json() == payload


def test_etf_momentum_strategy_api_selects_three_etf_preview(tmp_path: Path) -> None:
    report_path = tmp_path / "three_etf" / "strategy_report_latest.json"
    report_path.parent.mkdir()
    payload = {
        "schema_version": 1,
        "status": "research_preview",
        "research_id": "etf_20d_three_asset_defensive_rotation",
        "summary": {
            "formal_candidate_status": None,
            "usable_for_trading": False,
        },
        "chart_data": {"equity": []},
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch(
        "api.v1.endpoints.etf_momentum_strategy.ETF_DEFENSIVE_REPORT_PATH",
        report_path,
    ):
        response = client().get(
            "/api/v1/research/strategies/etf-momentum?variant=three_etf"
        )

    assert response.status_code == 200
    assert response.json() == payload


def test_etf_momentum_strategy_api_rejects_unknown_variant() -> None:
    response = client().get(
        "/api/v1/research/strategies/etf-momentum?variant=unknown"
    )

    assert response.status_code == 422


def test_etf_momentum_strategy_api_reports_missing_artifact(tmp_path: Path) -> None:
    with patch(
        "api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_PATH",
        tmp_path / "missing.json",
    ):
        response = client().get("/api/v1/research/strategies/etf-momentum")

    assert response.status_code == 200
    assert response.json() == {
        "status": "missing",
        "reason_code": "report_not_generated",
        "message": "ETF momentum strategy report has not been generated yet.",
    }


def test_etf_momentum_strategy_api_reports_malformed_json_without_path(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "private" / "strategy_report_latest.json"
    report_path.parent.mkdir()
    report_path.write_text("{not-json", encoding="utf-8")

    with patch(
        "api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_PATH", report_path
    ):
        response = client().get("/api/v1/research/strategies/etf-momentum")

    body = response.json()
    assert body["status"] == "invalid"
    assert body["reason_code"] == "report_invalid_json"
    assert str(tmp_path) not in json.dumps(body)


def test_etf_momentum_strategy_api_rejects_oversized_report(tmp_path: Path) -> None:
    report_path = tmp_path / "strategy_report_latest.json"
    report_path.write_text('{"status":"manual_review_required"}', encoding="utf-8")

    with (
        patch(
            "api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_PATH", report_path
        ),
        patch("api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_MAX_BYTES", 8),
    ):
        response = client().get("/api/v1/research/strategies/etf-momentum")

    assert response.status_code == 200
    assert response.json()["reason_code"] == "report_too_large"
    assert str(report_path) not in response.text


def test_etf_momentum_strategy_api_rejects_non_object_json(tmp_path: Path) -> None:
    report_path = tmp_path / "strategy_report_latest.json"
    report_path.write_text("[]", encoding="utf-8")

    with patch(
        "api.v1.endpoints.etf_momentum_strategy.ETF_REPORT_PATH", report_path
    ):
        response = client().get("/api/v1/research/strategies/etf-momentum")

    assert response.status_code == 200
    assert response.json()["reason_code"] == "report_invalid_shape"
