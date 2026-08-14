from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app


def _client(tmp_path: Path) -> TestClient:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    return TestClient(create_app(static_dir=static_dir))


def test_local_service_status_uses_the_fixed_service_name_and_open_url(tmp_path: Path) -> None:
    with _client(tmp_path) as client, patch(
        "api.v1.endpoints.quant_jobs._probe_mlflow_status_payload",
        return_value={"status": "available", "url": "http://127.0.0.1:6006/"},
    ):
        response = client.get("/api/v1/quant/services/mlflow")

    assert response.status_code == 200
    assert response.json() == {
        "service": "mlflow",
        "status": "available",
        "url": "http://127.0.0.1:6006/",
        "open_url": "/experiments/",
    }


def test_local_service_start_uses_only_the_mlflow_whitelist(tmp_path: Path) -> None:
    with _client(tmp_path) as client, patch(
        "api.v1.endpoints.quant_jobs.start_local_mlflow_service",
        return_value={"status": "starting", "pid": 60060, "port": 6006},
    ) as start_mlflow:
        headers = {"Origin": "http://127.0.0.1:8014"}
        response = client.post("/api/v1/quant/services/mlflow/start", headers=headers)
        rejected = client.post(
            "/api/v1/quant/services/not-a-service/start",
            headers=headers,
        )

    assert response.status_code == 202
    assert response.json() == {
        "service": "mlflow",
        "status": "starting",
        "pid": 60060,
        "port": 6006,
        "open_url": "/experiments/",
    }
    start_mlflow.assert_called_once_with()
    assert rejected.status_code == 422
