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

from quant_data_contract import (  # noqa: E402
    DataContractError,
    resolve_dataset,
    validate_qlib_snapshot,
)


DEFAULT_WORKFLOW_ROOT = SCRIPT_DIR.parent
DEFAULT_DSA_ROOT = DEFAULT_WORKFLOW_ROOT.parent
DEFAULT_PROJECTS_ROOT = DEFAULT_DSA_ROOT.parent
if str(DEFAULT_DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_DSA_ROOT))
NATIVE_QLIB_DATASET_ROOTS = {
    "qlib_ashare_v2": Path("qlib/cn_data_ashare_v2"),
    "qlib_ashare_cleaned_v2": Path("qlib/cn_data_ashare_cleaned_v2"),
}


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


@lru_cache(maxsize=16)
def _resolve_native_qlib(
    root_text: str, dataset_id: str = "qlib_ashare_v2"
) -> Path:
    root = Path(root_text)
    try:
        resolved = resolve_dataset(
            dataset_id,
            quant_data_root=root,
            require_basis="qfq",
        )
    except DataContractError as exc:
        raise RuntimeError(f"native {dataset_id} is unavailable: {exc}") from exc
    if resolved.manifest.get("benchmark_complete") is not True:
        raise RuntimeError(f"native {dataset_id} benchmarks are incomplete")
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


def _path_contains_symlink(root: Path, relative: Path) -> bool:
    current = Path(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate_native_provider_override(
    configured: str | Path,
    *,
    dataset_id: str,
) -> Path:
    """Accept an override only when it names a complete immutable V2 snapshot."""

    root = quant_data_root().expanduser().resolve()
    dataset_root = NATIVE_QLIB_DATASET_ROOTS[dataset_id]
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    expected_parent = (root / dataset_root / "snapshots").resolve()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"{dataset_id} override must stay under QUANT_DATA_ROOT and use an immutable V2 snapshot"
        ) from exc
    if (
        candidate.parent != expected_parent
        or candidate.name in {"", ".", ".."}
        or _path_contains_symlink(root, relative)
    ):
        raise RuntimeError(
            f"{dataset_id} override must point to {dataset_root}/snapshots/<snapshot_id>; "
            "legacy compatibility providers are rejected"
        )
    try:
        manifest = validate_qlib_snapshot(
            candidate,
            verify_hashes=True,
            evidence_root=root,
        )
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{dataset_id} override is not a valid immutable V2 snapshot: {exc}"
        ) from exc
    if (
        manifest.get("dataset_id") != dataset_id
        or manifest.get("snapshot_id") != candidate.name
        or manifest.get("role") != "canonical"
        or manifest.get("status") != "ready"
        or manifest.get("price_basis") != "qfq"
        or manifest.get("benchmark_complete") is not True
    ):
        raise RuntimeError(
            f"{dataset_id} override has an invalid native V2 identity; "
            "legacy compatibility providers are rejected"
        )
    return candidate


def qlib_provider_uri(
    provider: str = "canonical",
    *,
    snapshot_override: str | Path | None = None,
) -> Path:
    """Resolve the old physical ``.bin`` provider for compatibility only.

    New code must use :func:`qlib_workflow_provider_uri`.  The separate
    ``qlib_native_snapshot_uri`` name makes the remaining legacy call sites
    explicit without changing the old snapshot contract.
    """
    if provider in {"canonical", "raw"}:
        configured = snapshot_override or os.environ.get("QLIB_PROVIDER_URI")
        if configured:
            return _validate_native_provider_override(
                configured,
                dataset_id="qlib_ashare_v2",
            )
        root = quant_data_root().expanduser().resolve()
        return _resolve_native_qlib(str(root))
    if provider == "cleaned":
        configured = snapshot_override or os.environ.get("QLIB_CLEANED_PROVIDER_URI")
        if configured:
            return _validate_native_provider_override(
                configured,
                dataset_id="qlib_ashare_cleaned_v2",
            )
        root = quant_data_root().expanduser().resolve()
        return _resolve_native_qlib(str(root), "qlib_ashare_cleaned_v2")
    raise ValueError(f"Unknown Qlib provider: {provider}")


def _resolve_direct_qlib_descriptor(
    root_text: str,
    membership_text: str,
    index_dataset_id: str = "cn_index_daily",
    cleaning_mode: str = "canonical",
    cleaning_plan_text: str = "",
) -> Path:
    from quant_workflow.quant_data_qlib_provider import (
        build_direct_provider_descriptor,
    )

    return build_direct_provider_descriptor(
        quant_data_root=Path(root_text),
        membership_path=Path(membership_text),
        index_dataset_id=index_dataset_id,
        cleaning_mode=cleaning_mode,
        cleaning_plan_path=Path(cleaning_plan_text) if cleaning_plan_text else None,
    )


def qlib_direct_provider_descriptor(provider: str = "canonical") -> Path:
    """Return the metadata-only Provider descriptor for the current qfq snapshot."""

    provider = provider.strip().lower()
    if provider not in {"canonical", "cleaned"}:
        raise ValueError(f"Unknown direct Qlib provider: {provider}")
    if os.environ.get("QLIB_PROVIDER_URI") or os.environ.get("QLIB_CLEANED_PROVIDER_URI"):
        raise RuntimeError(
            "legacy compatibility providers are rejected; remove QLIB_PROVIDER_URI/"
            "QLIB_CLEANED_PROVIDER_URI and use the governed direct Provider"
        )
    root = quant_data_root().expanduser().resolve()
    membership = _env_path(
        "QLIB_DIRECT_MEMBERSHIP_PATH",
        root / "reference" / "pit_csi800" / "membership_intervals.parquet",
    ).expanduser()
    plan = _env_path(
        "QLIB_DIRECT_CLEANING_PLAN_PATH",
        root / "reference" / "qlib_cleaning" / "CLEANING_PLAN.json",
    ).expanduser()
    return _resolve_direct_qlib_descriptor(
        str(root),
        str(membership),
        "cn_index_daily",
        provider,
        str(plan) if provider == "cleaned" else "",
    )


def qlib_direct_provider_uri(provider: str = "canonical") -> Path:
    """Return Qlib's required URI directory without storing market-data bins."""

    return qlib_direct_provider_descriptor(provider).parent


def init_qlib_direct(
    *,
    provider: str = "canonical",
    kernels: int | None = None,
    **overrides,
) -> Path:
    """Initialize Qlib against the frozen quant_data Parquet snapshot."""

    from quant_workflow.quant_data_qlib_provider import init_qlib_from_quant_data

    descriptor = qlib_direct_provider_descriptor(provider)
    init_qlib_from_quant_data(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root(),
        kernels=kernels,
        **overrides,
    )
    return descriptor


def qlib_workflow_provider_descriptor(provider: str = "canonical") -> Path:
    """Return the formal direct Provider descriptor used by runtime workflows."""

    return qlib_direct_provider_descriptor(provider)


def qlib_workflow_provider_uri(provider: str = "canonical") -> Path:
    """Return the formal direct Provider URI used by runtime workflows."""

    return qlib_direct_provider_uri(provider)


def _direct_descriptor_from_uri(provider_uri: str | Path) -> Path | None:
    candidate = Path(provider_uri).expanduser()
    if candidate.name == "PROVIDER_DESCRIPTOR.json":
        return candidate
    descriptor = candidate / "PROVIDER_DESCRIPTOR.json"
    return descriptor if descriptor.is_file() else None


def qlib_workflow_init_config(
    *,
    provider: str = "canonical",
    provider_uri: str | Path | None = None,
    kernels: int | None = None,
) -> dict:
    """Return a YAML-safe Qlib config for the formal direct Provider."""

    from quant_workflow.quant_data_qlib_provider import qlib_direct_init_config

    descriptor = (
        _direct_descriptor_from_uri(provider_uri)
        if provider_uri is not None
        else qlib_workflow_provider_descriptor(provider)
    )
    if descriptor is None:
        raise RuntimeError(
            "formal Qlib workflow requires a direct Provider descriptor; "
            "the old physical .bin Provider is compatibility-only"
        )
    return qlib_direct_init_config(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root(),
        kernels=kernels,
    )


def init_qlib_workflow(
    *,
    provider: str = "canonical",
    provider_uri: str | Path | None = None,
    kernels: int | None = None,
    **overrides,
) -> Path:
    """Initialize Qlib for a workflow using the direct Parquet Provider."""

    from quant_workflow.quant_data_qlib_provider import init_qlib_from_quant_data

    descriptor = (
        _direct_descriptor_from_uri(provider_uri)
        if provider_uri is not None
        else qlib_workflow_provider_descriptor(provider)
    )
    if descriptor is None:
        raise RuntimeError(
            "formal Qlib workflow requires a direct Provider descriptor; "
            "the old physical .bin Provider is compatibility-only"
        )
    init_qlib_from_quant_data(
        descriptor_path=descriptor,
        quant_data_root=quant_data_root(),
        kernels=kernels,
        **overrides,
    )
    return descriptor


def qlib_init_config_for_uri(
    provider_uri: str | Path,
    *,
    provider: str = "canonical",
    kernels: int | None = None,
) -> dict:
    """Build config for direct URI; explicit old paths remain compatibility-only."""

    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is None:
        legacy = Path(provider_uri).expanduser().resolve()
        return {
            "provider_uri": str(legacy),
            "provider": "LocalProvider",
            "region": "cn",
        }
    return qlib_workflow_init_config(
        provider=provider,
        provider_uri=descriptor,
        kernels=kernels,
    )


def provider_calendar_days(provider_uri: str | Path) -> list[str]:
    """Read the frozen calendar from a direct Provider or an old snapshot."""

    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is not None:
        import pandas as pd

        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        context = _load_descriptor(descriptor, quant_data_root())
        return [value.date().isoformat() for value in context.calendar]

    calendar = Path(provider_uri) / "calendars" / "day.txt"
    if not calendar.is_file():
        raise FileNotFoundError(f"Provider calendar is missing: {calendar}")
    return [
        line.strip()
        for line in calendar.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def provider_instrument_ranges(
    provider_uri: str | Path,
) -> dict[str, tuple[str, str]]:
    """Read instrument coverage from direct metadata or a legacy snapshot."""

    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is not None:
        import pandas as pd

        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        context = _load_descriptor(descriptor, quant_data_root())
        return {
            str(row.symbol): (
                pd.Timestamp(row.first_date).date().isoformat(),
                pd.Timestamp(row.last_date).date().isoformat(),
            )
            for row in context.file_index.itertuples(index=False)
        }

    all_path = Path(provider_uri) / "instruments" / "all.txt"
    if not all_path.is_file():
        raise FileNotFoundError(f"Provider instruments file is missing: {all_path}")
    ranges: dict[str, tuple[str, str]] = {}
    for line in all_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            ranges[parts[0].upper()] = (parts[1], parts[2])
    return ranges


def qlib_native_snapshot_uri(
    provider: str = "canonical",
    *,
    snapshot_override: str | Path | None = None,
) -> Path:
    """Explicit name for the old physical Qlib snapshot compatibility path."""

    return qlib_provider_uri(provider, snapshot_override=snapshot_override)


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
            "QLIB_CLEANED_COMPATIBILITY_PROVIDER_URI",
            root / "cn_data_baostock_jqdata_cleaned",
        )
    raise ValueError(f"Unknown compatibility Qlib provider: {provider}")
