# -*- coding: utf-8 -*-
"""Shared path resolution for the embedded local quant workflow."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def dsa_root() -> Path:
    return env_path("DSA_ROOT", Path(__file__).resolve().parents[2])


def projects_root(root: Path | None = None) -> Path:
    dsa = root or dsa_root()
    return env_path("WORKING_PROJECTS_ROOT", dsa.parent)


def workflow_root(root: Path | None = None) -> Path:
    dsa = root or dsa_root()
    return env_path("QUANT_WORKFLOW_ROOT", dsa / "quant_workflow")


def reports_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_REPORTS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return workflow_root(root) / "reports"


def mlruns_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_MLRUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    return workflow_root(root) / "mlruns"


def quant_data_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()

    return projects_root(root) / "quant_qmt_data" / "data"
