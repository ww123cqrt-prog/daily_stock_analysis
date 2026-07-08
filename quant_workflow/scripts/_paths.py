"""Shared path helpers for quant_workflow scripts.

The scripts are run from DSA, tests, and sometimes a shell.  Keep default
locations anchored to this file while allowing environment overrides for local
deployments and dry-run tests.
"""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKFLOW_ROOT = SCRIPT_DIR.parent
DEFAULT_DSA_ROOT = DEFAULT_WORKFLOW_ROOT.parent
DEFAULT_PROJECTS_ROOT = DEFAULT_DSA_ROOT.parent
REPORT_HISTORY_FILES = (
    "research_status_latest.json",
    "latest_training_result.json",
    "experiment_runs.jsonl",
    "experiment_log.md",
)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


def projects_root() -> Path:
    return _env_path("WORKING_PROJECTS_ROOT", DEFAULT_PROJECTS_ROOT)


def workflow_root() -> Path:
    return _env_path("QUANT_WORKFLOW_ROOT", DEFAULT_WORKFLOW_ROOT)


def _contains_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def _prefer_payload_root(primary: Path, sibling: Path, marker_names: tuple[str, ...]) -> Path:
    if _contains_any(primary, marker_names):
        return primary
    if _contains_any(sibling, marker_names):
        return sibling
    return primary


def reports_root() -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_REPORTS_ROOT")
    if configured:
        return Path(configured).expanduser()
    primary = workflow_root() / "reports"
    if os.environ.get("QUANT_WORKFLOW_ROOT"):
        return primary
    sibling = projects_root() / "quant_workflow" / "reports"
    return _prefer_payload_root(primary, sibling, REPORT_HISTORY_FILES)


def mlruns_root() -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_MLRUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    primary = workflow_root() / "mlruns"
    if os.environ.get("QUANT_WORKFLOW_ROOT"):
        return primary
    sibling = projects_root() / "quant_workflow" / "mlruns"
    return _prefer_payload_root(primary, sibling, ("0", ".trash", "models"))


def quant_data_root() -> Path:
    configured = os.environ.get("QUANT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()

    repo_local = DEFAULT_DSA_ROOT / "quant_data"
    sibling = projects_root() / "quant_data"
    if repo_local.exists():
        return repo_local
    if sibling.exists():
        return sibling
    return repo_local


def china_astock_root() -> Path:
    return _env_path("CHINA_ASTOCK_QUANT_ROOT", projects_root() / "china-astock-quant")


def qlib_provider_uri(provider: str = "canonical") -> Path:
    if provider in {"canonical", "raw"}:
        default = quant_data_root() / "qlib" / "cn_data_baostock_jqdata"
        return _env_path("QLIB_PROVIDER_URI", default)
    if provider == "cleaned":
        default = quant_data_root() / "qlib" / "cn_data_baostock_jqdata_cleaned"
        return _env_path("QLIB_CLEANED_PROVIDER_URI", default)
    raise ValueError(f"Unknown Qlib provider: {provider}")
