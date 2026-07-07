# -*- coding: utf-8 -*-
"""Tests for browser-safe URL display helpers."""

from __future__ import annotations

import unittest

from src.safe_urls import public_http_url


class SafeUrlsTestCase(unittest.TestCase):
    def test_public_http_url_strips_userinfo_query_and_fragment(self) -> None:
        self.assertEqual(
            public_http_url(
                "http://fake_user:fake_pass@127.0.0.1:6006/path?token=secret#frag",
                "http://127.0.0.1:6006/",
            ),
            "http://127.0.0.1:6006/path",
        )

    def test_public_http_url_rejects_non_http_urls(self) -> None:
        self.assertEqual(
            public_http_url("file:///etc/passwd", "http://127.0.0.1:6006/"),
            "http://127.0.0.1:6006/",
        )

    def test_public_http_url_uses_default_for_blank_or_malformed_values(self) -> None:
        default = "http://127.0.0.1:6006/"

        self.assertEqual(public_http_url("", default), default)
        self.assertEqual(public_http_url("   ", default), default)
        self.assertEqual(public_http_url("not-a-url", default), default)
        self.assertEqual(public_http_url("http://localhost:not-a-port/path", default), default)
        self.assertEqual(public_http_url("http://localhost:/path", default), default)
        self.assertEqual(public_http_url("http://[::1]:/path", default), default)

    def test_public_http_url_preserves_safe_https_url_without_secret_parts(self) -> None:
        self.assertEqual(
            public_http_url("https://mlflow.example.com:8443/experiments?api_key=secret", "http://127.0.0.1:6006/"),
            "https://mlflow.example.com:8443/experiments",
        )

    def test_public_http_url_round_trips_ipv6_loopback_host(self) -> None:
        self.assertEqual(
            public_http_url("http://[::1]:6006/?token=secret#frag", "http://127.0.0.1:6006/"),
            "http://[::1]:6006/",
        )


if __name__ == "__main__":
    unittest.main()
