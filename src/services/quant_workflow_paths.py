# -*- coding: utf-8 -*-
"""Shared path resolution for the embedded local quant workflow."""

from __future__ import annotations

import os
from pathlib import Path


REPORT_HISTORY_FILES = (
    "research_status_latest.json",
    "latest_training_result.json",
    "experiment_runs.jsonl",
    "experiment_log.md",
)


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


def _contains_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def _prefer_payload_root(primary: Path, sibling: Path, marker_names: tuple[str, ...]) -> Path:
    if _contains_any(primary, marker_names):
        return primary
    if _contains_any(sibling, marker_names):
        return sibling
    return primary


def reports_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_REPORTS_ROOT")
    if configured:
        return Path(configured).expanduser()

    workflow = workflow_root(root)
    primary = workflow / "reports"
    if os.environ.get("QUANT_WORKFLOW_ROOT"):
        return primary

    sibling = projects_root(root) / "quant_workflow" / "reports"
    return _prefer_payload_root(primary, sibling, REPORT_HISTORY_FILES)


def mlruns_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_MLRUNS_DIR")
    if configured:
        return Path(configured).expanduser()

    workflow = workflow_root(root)
    primary = workflow / "mlruns"
    if os.environ.get("QUANT_WORKFLOW_ROOT"):
        return primary

    sibling = projects_root(root) / "quant_workflow" / "mlruns"
    return _prefer_payload_root(primary, sibling, ("0", ".trash", "models"))


def quant_data_root(root: Path | None = None) -> Path:
    configured = os.environ.get("QUANT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()

    dsa = root or dsa_root()
    primary = dsa / "quant_data"
    sibling = projects_root(root) / "quant_data"
    if primary.exists():
        return primary
    if sibling.exists():
        return sibling
    return primary
