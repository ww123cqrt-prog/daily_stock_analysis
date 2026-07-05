# -*- coding: utf-8 -*-
"""Tests for backtest performance empty-state API behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app


class BacktestPerformanceApiTestCase(unittest.TestCase):
    def _make_client(self, static_dir: Path) -> TestClient:
        static_dir.mkdir(parents=True, exist_ok=True)
        return TestClient(create_app(static_dir=static_dir))

    def test_overall_performance_empty_ok_returns_null_without_changing_default_404(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("api.v1.endpoints.backtest.BacktestService") as service_cls:
                service_cls.return_value.get_summary.return_value = None
                client = self._make_client(Path(temp_dir) / "static")

                default_resp = client.get("/api/v1/backtest/performance")
                empty_ok_resp = client.get("/api/v1/backtest/performance?empty_ok=true")

        self.assertEqual(default_resp.status_code, 404)
        self.assertEqual(empty_ok_resp.status_code, 200)
        self.assertIsNone(empty_ok_resp.json())

    def test_stock_performance_empty_ok_returns_null_without_changing_default_404(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("api.v1.endpoints.backtest.BacktestService") as service_cls:
                service_cls.return_value.get_summary.return_value = None
                client = self._make_client(Path(temp_dir) / "static")

                default_resp = client.get("/api/v1/backtest/performance/600519")
                empty_ok_resp = client.get("/api/v1/backtest/performance/600519?empty_ok=true")

        self.assertEqual(default_resp.status_code, 404)
        self.assertEqual(empty_ok_resp.status_code, 200)
        self.assertIsNone(empty_ok_resp.json())


if __name__ == "__main__":
    unittest.main()
