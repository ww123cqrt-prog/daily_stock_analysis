# -*- coding: utf-8 -*-
"""Helpers for publishing the actual Web UI bind host to the API factory."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional


def infer_uvicorn_cli_host(argv: Optional[Iterable[str]] = None) -> str:
    """Best-effort extraction of ``uvicorn --host`` from the current CLI."""
    args = list(sys.argv if argv is None else argv)
    for index, arg in enumerate(args):
        if arg == "--":
            break
        if arg == "--host" and index + 1 < len(args):
            return args[index + 1].strip()
        if arg.startswith("--host="):
            return arg.split("=", 1)[1].strip()
    return ""


def publish_bind_host(host: str) -> None:
    """Expose the runtime bind host for app-level CORS/auth safety checks."""
    normalized = (host or "").strip()
    if normalized:
        os.environ["DSA_BIND_HOST"] = normalized


def declared_bind_host() -> str:
    """Return the bind host explicitly declared by the launcher or uvicorn CLI."""
    return (
        os.environ.get("DSA_BIND_HOST")
        or os.environ.get("UVICORN_HOST")
        or infer_uvicorn_cli_host()
        or ""
    ).strip()
