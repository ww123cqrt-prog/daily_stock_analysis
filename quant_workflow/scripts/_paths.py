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


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser()


def projects_root() -> Path:
    return _env_path("WORKING_PROJECTS_ROOT", DEFAULT_PROJECTS_ROOT)


def workflow_root() -> Path:
    return _env_path("QUANT_WORKFLOW_ROOT", DEFAULT_WORKFLOW_ROOT)


def reports_root() -> Path:
    return _env_path("QUANT_WORKFLOW_REPORTS_ROOT", workflow_root() / "reports")


def mlruns_root() -> Path:
    return _env_path("QUANT_WORKFLOW_MLRUNS_DIR", workflow_root() / "mlruns")


def quant_data_root() -> Path:
    return _env_path("QUANT_DATA_ROOT", DEFAULT_DSA_ROOT / "quant_data")


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
