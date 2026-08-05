# -*- coding: utf-8 -*-
"""Tests for FastAPI app CORS configuration."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from api.app import create_app


class AppCorsConfigTestCase(unittest.TestCase):
    """CORS configuration should stay browser-compatible."""

    def _build_app(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return create_app(static_dir=Path(temp_dir.name))

    def test_allow_all_disables_credentials(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False):
            app = self._build_app()

        cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
        self.assertEqual(cors.kwargs["allow_origins"], ["*"])
        self.assertFalse(cors.kwargs["allow_credentials"])
        self.assertEqual(cors.kwargs["allow_methods"], ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
        self.assertEqual(cors.kwargs["allow_headers"], ["*"])

    def test_preflight_accepts_existing_custom_client_headers(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
            patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.request(
                "OPTIONS",
                "/api/v1/quant/jobs",
                headers={
                    "Origin": "http://127.0.0.1:8010",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-DSA-Local-Client, Cache-Control, Content-Type",
                },
            )

        self.assertEqual(resp.status_code, 200)
        allow_headers = resp.headers.get("access-control-allow-headers", "")
        self.assertIn("X-DSA-Local-Client", allow_headers)
        self.assertIn("Cache-Control", allow_headers)
        self.assertIn("Content-Type", allow_headers)

    def test_allow_all_warns_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.app.logger.warning") as warning:
            self._build_app()

        warning.assert_called_once()
        self.assertIn("CORS_ALLOW_ALL=true", warning.call_args.args[0])
        self.assertIn("ADMIN_AUTH_ENABLED is false", warning.call_args.args[0])

    def test_allow_all_assumes_localhost_when_bind_host_is_missing(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true"}, clear=True), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.app.logger.warning") as warning:
            app = self._build_app()

        self.assertIsNotNone(app)
        warning.assert_called_once()

    def test_explicit_cors_assumes_localhost_when_bind_host_is_missing(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false"}, clear=True), \
             patch("api.app.is_auth_enabled", return_value=False):
            app = self._build_app()

        self.assertIsNotNone(app)

    def test_allow_all_rejects_non_localhost_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "0.0.0.0"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._build_app()

        self.assertIn("Unsafe web runtime configuration", str(ctx.exception))

    def test_allow_all_rejects_production_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1", "DSA_ENV": "production"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._build_app()

        self.assertIn("ADMIN_AUTH_ENABLED=true", str(ctx.exception))

    def test_explicit_cors_rejects_non_localhost_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false", "DSA_BIND_HOST": "0.0.0.0"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._build_app()

        self.assertIn("ADMIN_AUTH_ENABLED=true", str(ctx.exception))

    def test_explicit_cors_rejects_production_when_auth_is_disabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false", "DSA_BIND_HOST": "127.0.0.1", "DSA_ENV": "production"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                self._build_app()

        self.assertIn("production", str(ctx.exception))

    def test_allow_all_does_not_warn_when_auth_is_enabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=True), \
             patch("api.app.logger.warning") as warning:
            self._build_app()

        warning.assert_not_called()

    def test_explicit_origin_list_keeps_credentials_enabled(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false"}, clear=False):
            app = self._build_app()

        cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
        self.assertIn("http://localhost:5173", cors.kwargs["allow_origins"])
        self.assertTrue(cors.kwargs["allow_credentials"])

    def test_cross_origin_quant_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "https://evil.example"},
                json={"job_type": "check_qlib_data"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_mlflow_proxy_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/experiments/ajax-api/2.0/mlflow/runs/delete",
                headers={"Origin": "https://evil.example"},
                json={"run_id": "run-1"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_mlflow_proxy_read_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/experiments/ajax-api/2.0/mlflow/experiments/search",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_research_experiment_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.delete(
                "/api/v1/research/experiments/abc123/permanent",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_permanent_delete_with_confirm_query_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.delete(
                "/api/v1/research/experiments/abc123/permanent?confirm=permanent",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_auth_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/auth/settings",
                headers={"Origin": "https://evil.example"},
                json={"authEnabled": True, "password": "adminpass123", "passwordConfirm": "adminpass123"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_alphasift_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/alphasift/screen",
                headers={"Origin": "https://evil.example"},
                json={"strategy": "dual_low", "market": "cn", "maxResults": 10},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_general_api_mutations_are_rejected_even_when_cors_is_open(self):
        mutation_paths = [
            ("POST", "/api/v1/analysis/analyze"),
            ("DELETE", "/api/v1/history/by-code/000001"),
            ("POST", "/api/v1/stocks/extract-from-image"),
            ("POST", "/api/v1/portfolio/accounts"),
            ("POST", "/api/v1/alerts/rules"),
            ("POST", "/api/v1/backtest/run"),
            ("PUT", "/api/v1/system/config"),
            ("POST", "/api/v1/agent/chat"),
        ]
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            for method, path in mutation_paths:
                with self.subTest(method=method, path=path):
                    resp = client.request(
                        method,
                        path,
                        headers={"Origin": "https://evil.example"},
                        json={},
                    )
                    self.assertEqual(resp.status_code, 403)
                    self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_no_origin_general_api_mutation_can_reach_routing(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post("/api/v1/not-a-real-endpoint", json={})

        self.assertEqual(resp.status_code, 404)

    def test_cross_origin_quant_cancel_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs/qj_123/cancel",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_quant_stream_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/api/v1/quant/jobs/stream",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_quant_read_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/api/v1/quant/jobs",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_null_origin_quant_read_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/api/v1/quant/status",
                headers={"Origin": "null"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_null_origin_quant_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "null"},
                json={"job_type": "check_qlib_data"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_quant_status_read_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/api/v1/quant/status",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_research_status_read_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get(
                "/api/v1/research/status",
                headers={"Origin": "https://evil.example"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_no_origin_quant_read_is_allowed_for_local_cli_health_checks(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get("/api/v1/quant/jobs")

        self.assertEqual(resp.status_code, 200)

    def test_no_origin_research_status_read_is_allowed_for_local_cli_health_checks(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.get("/api/v1/research/status")

        self.assertEqual(resp.status_code, 200)

    def test_missing_origin_and_referer_quant_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                json={"job_type": "check_qlib_data"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_missing_origin_and_referer_backtest_mutation_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/backtest/strategy/trained",
                json={},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_local_client_header_does_not_allow_no_origin_mutation(self):
        for header_value in ("true", "1", "yes"):
            with self.subTest(header_value=header_value):
                with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
                     patch("api.app.is_auth_enabled", return_value=False), \
                     patch("api.middlewares.auth.is_auth_enabled", return_value=False):
                    client = TestClient(self._build_app())
                    resp = client.post(
                        "/api/v1/quant/jobs",
                        headers={"X-DSA-Local-Client": header_value},
                        json={"job_type": "not_a_real_job"},
                    )

                self.assertEqual(resp.status_code, 403)
                self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_local_client_header_does_not_allow_no_origin_auth_mutation(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False), \
             patch("api.v1.endpoints.auth.has_stored_password", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/auth/settings",
                headers={"X-DSA-Local-Client": "true"},
                json={"authEnabled": True},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_trusted_local_client_header_is_ignored_when_origin_is_present(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "https://evil.example", "X-DSA-Local-Client": "true"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_trusted_local_client_header_is_ignored_when_origin_header_is_empty(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "", "X-DSA-Local-Client": "true"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_trusted_local_client_header_is_ignored_when_referer_header_is_empty(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Referer": "", "X-DSA-Local-Client": "true"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_trusted_local_client_header_is_ignored_on_non_localhost_bind(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false", "DSA_BIND_HOST": "0.0.0.0"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=True), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"X-DSA-Local-Client": "true"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_cross_origin_referer_without_origin_is_rejected_even_when_cors_is_open(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Referer": "https://evil.example/page"},
                json={"job_type": "check_qlib_data"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_local_referer_without_origin_is_allowed_to_reach_business_validation(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Referer": "http://127.0.0.1:8010/workflow"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")

    def test_local_quant_mutation_origin_is_allowed_to_reach_business_validation(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "http://127.0.0.1:8010"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")

    def test_non_localhost_bind_does_not_trust_localhost_origin_shortcut(self):
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "false", "DSA_BIND_HOST": "0.0.0.0"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=True), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "http://127.0.0.1:8010"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"], "csrf_forbidden")

    def test_allow_all_keeps_explicit_origins_for_non_localhost_mutation_guard(self):
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOW_ALL": "true",
                "CORS_ORIGINS": "http://127.0.0.1:8010",
                "DSA_BIND_HOST": "0.0.0.0",
            },
            clear=False,
        ), \
             patch("api.app.is_auth_enabled", return_value=True), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            resp = client.post(
                "/api/v1/quant/jobs",
                headers={"Origin": "http://127.0.0.1:8010"},
                json={"job_type": "not_a_real_job"},
            )

        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid_params")

    def test_missing_origin_sensitive_mutations_are_rejected_consistently(self):
        mutation_paths = [
            ("POST", "/api/v1/portfolio/accounts"),
            ("POST", "/api/v1/alerts/rules"),
            ("POST", "/api/v1/stocks/extract-from-image"),
        ]
        with patch.dict(os.environ, {"CORS_ALLOW_ALL": "true", "DSA_BIND_HOST": "127.0.0.1"}, clear=False), \
             patch("api.app.is_auth_enabled", return_value=False), \
             patch("api.middlewares.auth.is_auth_enabled", return_value=False):
            client = TestClient(self._build_app())
            for method, path in mutation_paths:
                with self.subTest(method=method, path=path):
                    resp = client.request(method, path, json={})
                    self.assertEqual(resp.status_code, 403)
                    self.assertEqual(resp.json()["error"], "csrf_forbidden")


if __name__ == "__main__":
    unittest.main()
