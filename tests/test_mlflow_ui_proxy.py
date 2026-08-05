from __future__ import annotations

import json
import gzip

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.services.mlflow_ui_proxy as mlflow_proxy
from src.services.mlflow_ui_proxy import (
    build_runtime_script,
    create_mlflow_ui_router,
    inject_runtime_script,
)


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, payload: bytes):
        self.payload = payload

    async def __aiter__(self):
        yield self.payload


def _client(handler) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_mlflow_ui_router(
            upstream_url="http://127.0.0.1:6006/",
            transport=httpx.MockTransport(handler),
        ),
        prefix="/experiments",
    )
    return TestClient(app)


def test_proxy_injects_runtime_into_mlflow_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://127.0.0.1:6006/"
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "cache-control": "no-cache",
            },
            text="<!doctype html><html><body><div id='root'></div></body></html>",
        )

    response = _client(handler).get("/experiments/")

    assert response.status_code == 200
    assert response.text.count("/__dsa_i18n__/runtime.js") == 1
    assert response.headers["cache-control"] == "no-store"


def test_proxy_preserves_non_html_payload_and_encoding() -> None:
    payload = b"compressed-artifact"
    compressed_payload = gzip.compress(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "content-encoding": "gzip",
                "etag": '"artifact-v1"',
            },
            stream=_AsyncBytes(compressed_payload),
        )

    response = _client(handler).get("/experiments/ajax-api/2.0/mlflow-artifacts/artifacts")

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["etag"] == '"artifact-v1"'


def test_proxy_forwards_request_body() -> None:
    observed: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["body"] = request.read()
        return httpx.Response(201, json={"ok": True})

    response = _client(handler).post(
        "/experiments/ajax-api/2.0/mlflow/upload-artifact",
        content=b"artifact-body",
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 201
    assert observed["body"] == b"artifact-body"


def test_proxy_does_not_forward_or_overwrite_dsa_session_cookie() -> None:
    observed: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(
            200,
            headers=[
                ("content-type", "application/json"),
                ("set-cookie", "dsa_session=upstream-value; Path=/"),
                ("set-cookie", "mlflow_user=cq; Path=/"),
            ],
            json={"ok": True},
        )

    response = _client(handler).get(
        "/experiments/ajax-api/2.0/mlflow/users/current",
        headers={"cookie": "dsa_session=secret-value; mlflow_user=cq"},
    )

    assert response.status_code == 200
    assert observed["cookie"] == "mlflow_user=cq"
    assert response.headers.get_list("set-cookie") == ["mlflow_user=cq; Path=/experiments/"]


def test_proxy_rewrites_upstream_redirect_back_under_experiments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            303,
            headers={"location": "http://127.0.0.1:6006/#/experiments/1"},
        )

    response = _client(handler).get("/experiments/redirect", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/experiments/#/experiments/1"


def test_runtime_script_contains_catalog_and_syncs_dsa_language() -> None:
    script = build_runtime_script({"Experiments": "实验", "Settings": "设置"})

    assert json.dumps("Experiments", ensure_ascii=False) in script
    assert json.dumps("实验", ensure_ascii=False) in script
    assert "dsa.uiLanguage" in script
    assert "placeholder" in script
    assert "aria-label" in script
    assert "指标 ($1)" in script
    assert "参数 ($1)" in script


def test_runtime_script_preserves_complete_upstream_navigation() -> None:
    script = build_runtime_script({"Experiments": "实验"})

    assert "enforceRouteScope" not in script
    assert "scopedHash" not in script
    assert "data-dsa-scope-hidden" not in script
    assert "window.addEventListener('hashchange'" not in script


def test_html_injection_is_idempotent() -> None:
    original = b"<html><body>MLflow</body></html>"
    once = inject_runtime_script(original)
    twice = inject_runtime_script(once)

    assert once == twice


def test_proxy_returns_localized_gateway_page_when_mlflow_is_offline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private upstream detail", request=request)

    response = _client(handler).get("/experiments/")

    assert response.status_code == 502
    assert "MLflow 服务未启动" in response.text
    assert "启动 MLflow" in response.text
    assert "重新检查" in response.text
    assert "/experiments/__dsa_control__/start" in response.text
    assert "private upstream detail" not in response.text


def test_proxy_exposes_fixed_local_mlflow_start_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_start() -> dict[str, object]:
        calls.append(True)
        return {"status": "starting", "pid": 4321}

    monkeypatch.setattr(mlflow_proxy, "start_local_mlflow_service", fake_start)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"start request must not reach MLflow upstream: {request.url}")

    response = _client(handler).post("/experiments/__dsa_control__/start")

    assert response.status_code == 202
    assert response.json() == {"status": "starting", "pid": 4321}
    assert calls == [True]


def test_proxy_returns_visible_error_when_mlflow_process_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mlflow_proxy,
        "start_local_mlflow_service",
        lambda: {
            "status": "failed",
            "message": "MLflow 进程启动后立即退出，请检查 logs/mlflow_6006.log。",
            "pid": 9876,
            "port": 6006,
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"start request must not reach MLflow upstream: {request.url}")

    response = _client(handler).post("/experiments/__dsa_control__/start")

    assert response.status_code == 500
    assert response.json()["message"] == "MLflow 进程启动后立即退出，请检查 logs/mlflow_6006.log。"


def test_proxy_forwards_head_without_a_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream", "content-length": "128"},
        )

    response = _client(handler).head("/experiments/static-files/artifact.bin")

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-length"] == "128"


def test_proxy_rejects_non_local_upstream() -> None:
    with pytest.raises(ValueError, match="local"):
        create_mlflow_ui_router(upstream_url="https://mlflow.example.com/")
