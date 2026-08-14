"""Tests for the read-only DSA proxy to the governed quant_data service."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app
from quant_workflow.scripts.quant_data_service_client import (
    QuantDataClientError,
    QuantDataIdentity,
    QuantDataReadyResult,
)


class QuantDataStatusApiTestCase(unittest.TestCase):
    def _client(self) -> tuple[tempfile.TemporaryDirectory[str], TestClient]:
        temp_dir = tempfile.TemporaryDirectory()
        return temp_dir, TestClient(create_app(static_dir=Path(temp_dir.name)))

    def test_returns_only_ready_canonical_dataset_identities(self) -> None:
        identity = QuantDataIdentity(
            dataset_id="ashare_daily_qfq",
            snapshot_id="ashare-snapshot-1",
            manifest_sha256="a" * 64,
            content_sha256="b" * 64,
            latest_trade_date="2026-08-04",
            price_basis="qfq",
            data_kind="daily_bars",
        )
        ready = QuantDataReadyResult(
            status="ready",
            validated_datasets=1,
            identities=(identity,),
            datasets=({**identity.as_dict(), "role": "canonical", "status": "ready"},),
        )

        with patch(
            "quant_workflow.scripts.quant_data_service_client.QuantDataServiceClient.ready",
            return_value=ready,
        ):
            temp_dir, client = self._client()
            try:
                response = client.get("/api/v1/data/status")
            finally:
                temp_dir.cleanup()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["validated_datasets"], 1)
        self.assertEqual(body["datasets"][0]["dataset_id"], "ashare_daily_qfq")
        self.assertEqual(body["datasets"][0]["snapshot_id"], "ashare-snapshot-1")
        self.assertEqual(body["datasets"][0]["content_sha256"], "b" * 64)
        self.assertNotIn("path", body["datasets"][0])
        self.assertIn("checked_at", body)

    def test_fails_closed_when_quant_data_is_unavailable(self) -> None:
        with patch(
            "quant_workflow.scripts.quant_data_service_client.QuantDataServiceClient.ready",
            side_effect=QuantDataClientError("private backend detail", code="service_unavailable"),
        ):
            temp_dir, client = self._client()
            try:
                response = client.get("/api/v1/data/status")
            finally:
                temp_dir.cleanup()

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "quant_data_unavailable")
        self.assertNotIn("private backend detail", response.text)


if __name__ == "__main__":
    unittest.main()
