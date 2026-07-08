# -*- coding: utf-8 -*-
"""API contract tests for /api/v1/stocks."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api.v1.endpoints.stocks as stocks_endpoint
import src.auth as auth
from api.app import create_app
from src.config import Config


def _reset_auth_globals() -> None:
    auth._auth_enabled = None
    auth._session_secret = None
    auth._password_hash_salt = None
    auth._password_hash_stored = None
    auth._rate_limit = {}


class _FakeSystemConfigService:
    def __init__(self, stock_list: str = "600519,000001") -> None:
        self.stock_list = stock_list
        self.updates: list[dict] = []

    def get_config(self, include_schema: bool = False) -> dict:
        return {
            "config_version": "test-config-v1",
            "items": [
                {
                    "key": "STOCK_LIST",
                    "value": self.stock_list,
                }
            ],
        }

    def update(self, *, config_version: str, items: list[dict], mask_token: str, reload_now: bool) -> dict:
        self.updates.append(
            {
                "config_version": config_version,
                "items": items,
                "mask_token": mask_token,
                "reload_now": reload_now,
            }
        )
        for item in items:
            if item.get("key") == "STOCK_LIST":
                self.stock_list = str(item.get("value", ""))
        return self.get_config(include_schema=False)


class _FakeStockService:
    quote_result: dict | None = None
    history_result: dict | Exception = {"stock_name": "测试股票", "data": []}

    def get_realtime_quote(self, stock_code: str) -> dict | None:
        if self.quote_result is None:
            return None
        result = dict(self.quote_result)
        result.setdefault("stock_code", stock_code)
        return result

    def get_history_data(self, *, stock_code: str, period: str, days: int) -> dict:
        if isinstance(self.history_result, Exception):
            raise self.history_result
        result = dict(self.history_result)
        result.setdefault("stock_name", "测试股票")
        result.setdefault("data", [])
        return result


class StocksApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        _reset_auth_globals()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.env_path = self.data_dir / ".env"
        self.db_path = self.data_dir / "stocks_api_test.db"
        self._old_env_file = os.environ.get("ENV_FILE")
        self._old_database_path = os.environ.get("DATABASE_PATH")
        self.env_path.write_text(
            "\n".join(
                [
                    "ADMIN_AUTH_ENABLED=false",
                    "STOCK_LIST=600519,000001",
                    f"DATABASE_PATH={self.db_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.environ["ENV_FILE"] = str(self.env_path)
        os.environ["DATABASE_PATH"] = str(self.db_path)
        Config.reset_instance()

        app = create_app(static_dir=self.data_dir / "empty-static")
        self.config_service = _FakeSystemConfigService()
        app.dependency_overrides[stocks_endpoint.get_system_config_service] = lambda: self.config_service
        self.client = TestClient(app, headers={"Origin": "http://127.0.0.1:8010"})

    def tearDown(self) -> None:
        Config.reset_instance()
        if self._old_env_file is None:
            os.environ.pop("ENV_FILE", None)
        else:
            os.environ["ENV_FILE"] = self._old_env_file
        if self._old_database_path is None:
            os.environ.pop("DATABASE_PATH", None)
        else:
            os.environ["DATABASE_PATH"] = self._old_database_path
        self.temp_dir.cleanup()
        _reset_auth_globals()

    def test_parse_import_accepts_json_text_and_deduplicates_codes(self) -> None:
        response = self.client.post(
            "/api/v1/stocks/parse-import",
            json={"text": "代码,名称\n600519,贵州茅台\nSH600519,贵州茅台\n000001,平安银行"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["codes"], ["600519", "000001"])
        self.assertEqual(body["items"][0]["name"], "贵州茅台")

    def test_parse_import_rejects_json_without_text(self) -> None:
        response = self.client.post("/api/v1/stocks/parse-import", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "bad_request")

    def test_extract_from_image_rejects_unsupported_content_type_before_llm_call(self) -> None:
        response = self.client.post(
            "/api/v1/stocks/extract-from-image",
            files={"file": ("stocks.txt", b"600519", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "unsupported_type")

    def test_extract_from_image_returns_items_and_optional_raw_text(self) -> None:
        with patch.object(
            stocks_endpoint,
            "extract_stock_codes_from_image",
            return_value=([("600519", "贵州茅台", "high")], "raw llm text"),
        ) as extractor:
            response = self.client.post(
                "/api/v1/stocks/extract-from-image?include_raw=true",
                files={"file": ("stocks.png", b"fake-png", "image/png")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        extractor.assert_called_once_with(b"fake-png", "image/png")
        body = response.json()
        self.assertEqual(body["codes"], ["600519"])
        self.assertEqual(body["items"], [{"code": "600519", "name": "贵州茅台", "confidence": "high"}])
        self.assertEqual(body["raw_text"], "raw llm text")

    def test_parse_import_accepts_multipart_csv_file(self) -> None:
        response = self.client.post(
            "/api/v1/stocks/parse-import",
            files={"file": ("stocks.csv", b"code,name\n00700,\xe8\x85\xbe\xe8\xae\xaf\xe6\x8e\xa7\xe8\x82\xa1\n", "text/csv")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["codes"], ["00700"])
        self.assertEqual(body["items"][0]["name"], "腾讯控股")

    def test_quote_returns_stock_service_payload(self) -> None:
        _FakeStockService.quote_result = {
            "stock_code": "600519",
            "stock_name": "贵州茅台",
            "current_price": 1800.5,
            "change": 12.3,
            "change_percent": 0.69,
            "volume": 1000,
            "update_time": "2026-07-06T09:30:00",
        }
        with patch.object(stocks_endpoint, "StockService", _FakeStockService):
            response = self.client.get("/api/v1/stocks/600519/quote")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["stock_code"], "600519")
        self.assertEqual(body["stock_name"], "贵州茅台")
        self.assertEqual(body["current_price"], 1800.5)
        self.assertEqual(body["change_percent"], 0.69)

    def test_quote_returns_404_when_service_has_no_payload(self) -> None:
        _FakeStockService.quote_result = None
        with patch.object(stocks_endpoint, "StockService", _FakeStockService):
            response = self.client.get("/api/v1/stocks/999999/quote")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "not_found")

    def test_quote_returns_500_when_stock_service_raises(self) -> None:
        class RaisingStockService(_FakeStockService):
            def get_realtime_quote(self, stock_code: str) -> dict | None:
                raise RuntimeError("quote backend unavailable")

        with patch.object(stocks_endpoint, "StockService", RaisingStockService):
            response = self.client.get("/api/v1/stocks/600519/quote")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "internal_error")

    def test_history_returns_kline_payload(self) -> None:
        _FakeStockService.history_result = {
            "stock_name": "贵州茅台",
            "data": [
                {
                    "date": "2026-07-03",
                    "open": 1800.0,
                    "high": 1810.0,
                    "low": 1790.0,
                    "close": 1805.0,
                    "volume": 10000,
                    "amount": 18050000.0,
                    "change_percent": 0.5,
                }
            ],
        }
        with patch.object(stocks_endpoint, "StockService", _FakeStockService):
            response = self.client.get("/api/v1/stocks/600519/history?period=daily&days=1")

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["stock_code"], "600519")
        self.assertEqual(body["stock_name"], "贵州茅台")
        self.assertEqual(body["data"][0]["close"], 1805.0)

    def test_history_returns_422_when_service_rejects_period(self) -> None:
        _FakeStockService.history_result = ValueError("weekly period is not supported")
        with patch.object(stocks_endpoint, "StockService", _FakeStockService):
            response = self.client.get("/api/v1/stocks/600519/history?period=weekly&days=5")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "unsupported_period")

    def test_watchlist_read_add_and_remove_uses_system_config_dependency(self) -> None:
        initial = self.client.get("/api/v1/stocks/watchlist")
        self.assertEqual(initial.status_code, 200, initial.text)
        self.assertEqual(initial.json()["stock_codes"], ["600519", "000001"])

        added = self.client.post("/api/v1/stocks/watchlist/add", json={"stock_code": "300750"})
        self.assertEqual(added.status_code, 200, added.text)
        self.assertEqual(added.json()["stock_codes"], ["600519", "000001", "300750"])
        self.assertEqual(self.config_service.stock_list, "600519,000001,300750")

        removed = self.client.post("/api/v1/stocks/watchlist/remove", json={"stock_code": "600519.SH"})
        self.assertEqual(removed.status_code, 200, removed.text)
        self.assertEqual(removed.json()["stock_codes"], ["000001", "300750"])
        self.assertEqual(self.config_service.stock_list, "000001,300750")

    def test_watchlist_rejects_invalid_stock_code(self) -> None:
        response = self.client.post("/api/v1/stocks/watchlist/add", json={"stock_code": "../../bad"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "invalid_stock_code")

    def test_watchlist_add_does_not_duplicate_equivalent_codes(self) -> None:
        self.config_service.stock_list = "600519"

        response = self.client.post("/api/v1/stocks/watchlist/add", json={"stock_code": "SH600519"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["stock_codes"], ["600519"])
        self.assertEqual(self.config_service.updates, [])


if __name__ == "__main__":
    unittest.main()
