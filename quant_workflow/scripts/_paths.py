"""Shared path helpers for quant_workflow scripts.

The scripts are run from DSA, tests, and sometimes a shell.  Keep default
locations anchored to this file while allowing environment overrides for local
deployments and dry-run tests.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant_data_contract import DataContractError, resolve_dataset  # noqa: E402


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
    configured = os.environ.get("QUANT_WORKFLOW_REPORTS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return workflow_root() / "reports"


def mlruns_root() -> Path:
    configured = os.environ.get("QUANT_WORKFLOW_MLRUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    return workflow_root() / "mlruns"


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


@lru_cache(maxsize=8)
def _resolve_native_qlib(root_text: str) -> Path:
    root = Path(root_text)
    try:
        resolved = resolve_dataset(
            "qlib_ashare_v2",
            quant_data_root=root,
            require_basis="qfq",
        )
    except DataContractError as exc:
        raise RuntimeError(f"native qlib_ashare_v2 is unavailable: {exc}") from exc
    if resolved.manifest.get("benchmark_complete") is not True:
        raise RuntimeError("native qlib_ashare_v2 benchmarks are incomplete")
    return Path(resolved.snapshot_path)


@lru_cache(maxsize=8)
def _resolve_native_ashare_qfq(root_text: str) -> Path:
    root = Path(root_text)
    try:
        resolved = resolve_dataset(
            "ashare_daily_qfq",
            quant_data_root=root,
            require_basis="qfq",
        )
    except DataContractError as exc:
        raise RuntimeError(f"native ashare_daily_qfq is unavailable: {exc}") from exc
    return Path(resolved.snapshot_path)


def ashare_qfq_snapshot_uri() -> Path:
    root = quant_data_root().expanduser().resolve()
    return _resolve_native_ashare_qfq(str(root))


def ashare_qfq_bars_dir() -> Path:
    return ashare_qfq_snapshot_uri() / "bars"


def qlib_provider_uri(provider: str = "canonical") -> Path:
    if provider in {"canonical", "raw"}:
        configured = os.environ.get("QLIB_PROVIDER_URI")
        if configured:
            return Path(configured).expanduser()
        root = quant_data_root().expanduser().resolve()
        return _resolve_native_qlib(str(root))
    if provider == "cleaned":
        default = quant_data_root() / "qlib" / "cn_data_baostock_jqdata_cleaned"
        return _env_path("QLIB_CLEANED_PROVIDER_URI", default)
    raise ValueError(f"Unknown Qlib provider: {provider}")


def qlib_compatibility_provider_uri(provider: str = "canonical") -> Path:
    """Return mutable compatibility outputs used only by legacy producers."""

    root = quant_data_root() / "qlib"
    if provider == "canonical":
        return _env_path(
            "QLIB_COMPATIBILITY_PROVIDER_URI",
            root / "cn_data_baostock_jqdata",
        )
    if provider == "cleaned":
        return _env_path(
            "QLIB_CLEANED_PROVIDER_URI",
            root / "cn_data_baostock_jqdata_cleaned",
        )
    raise ValueError(f"Unknown compatibility Qlib provider: {provider}")
