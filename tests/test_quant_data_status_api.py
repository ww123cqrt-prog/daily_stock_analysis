"""Read-only metadata errors do not leak private paths or call old services."""
from unittest.mock import patch

from fastapi.testclient import TestClient as ApiClient
from api.app import create_app
from quant_qmt_data import DataContractError


def test_missing_local_store_returns_controlled_error(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_ROOT", str(tmp_path / "private-missing-data"))
    with ApiClient(create_app(static_dir=tmp_path / "static")) as client:
        response = client.get("/api/v1/data/status")
    assert response.status_code == 503
    assert response.json()["error"] == "quant_data_unavailable"
    assert "private-missing-data" not in response.text


def test_invalid_local_metadata_does_not_fall_back_to_http(tmp_path):
    with patch("quant_qmt_data.inspect_datasets", side_effect=DataContractError("private detail")):
        with ApiClient(create_app(static_dir=tmp_path / "static")) as client:
            response = client.get("/api/v1/data/status")
    assert response.status_code == 503
    assert "private detail" not in response.text
