# -*- coding: utf-8 -*-
"""Helpers for exposing URLs in browser-facing API payloads."""

from __future__ import annotations

from urllib.parse import urlparse


def public_http_url(raw_url: str, default_url: str) -> str:
    """Return an http(s) URL safe to display in API responses.

    Userinfo, query parameters, and fragments are intentionally removed because
    local tool URLs are often copied from environments that may contain tokens.
    """
    clean = (raw_url or "").strip() or default_url
    parsed = urlparse(clean)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return default_url
    if parsed.scheme not in {"http", "https"} or not hostname:
        return default_url
    host_port = parsed.netloc.rsplit("@", 1)[-1]
    if host_port.endswith(":") or " " in host_port:
        return default_url

    host = hostname
    # urlparse strips brackets from IPv6 hostnames; re-add them before
    # rebuilding netloc so the optional port remains unambiguous.
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"

    path = parsed.path or "/"
    return parsed._replace(netloc=host, path=path, params="", query="", fragment="").geturl()
