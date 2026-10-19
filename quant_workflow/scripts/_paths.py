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
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quant_data_contract import (  # noqa: E402
    DataContractError,
    resolve_dataset,
    sha256_file,
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


class QlibProviderHistoryError(RuntimeError):
    """Raised when a newer Provider rewrites data visible to a frozen Provider."""


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

    return projects_root() / "quant_qmt_data" / "data"


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
            f"{dataset_id} override is not a valid Qlib Provider snapshot: {exc}"
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
            f"{dataset_id} override has an invalid Qlib Provider identity; "
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
    index_dataset_id: str = "cn_index_daily",
    cleaning_mode: str = "canonical",
) -> Path:
    from quant_workflow.quant_data_qlib_provider import (
        build_direct_provider_descriptor,
    )

    return build_direct_provider_descriptor(
        quant_data_root=Path(root_text),
        index_dataset_id=index_dataset_id,
        cleaning_mode=cleaning_mode,
    )


def qlib_direct_provider_descriptor(provider: str = "canonical") -> Path:
    """Return the metadata-only Qlib Provider descriptor for the current qfq snapshot."""

    provider = provider.strip().lower()
    if provider not in {"canonical", "cleaned"}:
        raise ValueError(f"Unknown Qlib Provider: {provider}")
    if os.environ.get("QLIB_PROVIDER_URI") or os.environ.get("QLIB_CLEANED_PROVIDER_URI"):
        raise RuntimeError(
            "legacy compatibility providers are rejected; remove QLIB_PROVIDER_URI/"
            "QLIB_CLEANED_PROVIDER_URI and use the governed Qlib Provider"
        )
    root = quant_data_root().expanduser().resolve()
    # The local Parquet catalog is the selected source. An archived service
    # pointer in the same data directory does not select a different backend.
    return _resolve_direct_qlib_descriptor(
        str(root),
        "cn_index_daily",
        provider,
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
    """Return the formal Qlib Provider descriptor used by runtime workflows."""

    return qlib_direct_provider_descriptor(provider)


def _resolve_approved_workflow_provider_uri(
    provider: str,
    expected_provider_identity: Mapping[str, Any],
) -> Path:
    """Resolve a frozen Provider lineage inside the storage adapter boundary."""

    try:
        from _mlflow_run_provenance import resolve_approved_provider_identity
    except ImportError:  # pragma: no cover - package import from the API.
        from quant_workflow.scripts._mlflow_run_provenance import (  # type: ignore
            resolve_approved_provider_identity,
        )

    return resolve_approved_provider_identity(
        provider,
        expected_provider_identity,
        quant_data_root=quant_data_root(),
    )


def qlib_workflow_provider_uri(
    provider: str = "canonical",
    *,
    expected_provider_identity: Mapping[str, Any] | None = None,
    append_only_from: str | Path | None = None,
) -> Path:
    """Return the formal Qlib Provider URI used by runtime workflows."""

    if expected_provider_identity is not None:
        resolved = _resolve_approved_workflow_provider_uri(
            provider,
            expected_provider_identity,
        )
    else:
        resolved = qlib_direct_provider_uri(provider)
    if append_only_from is not None:
        _verify_direct_provider_history_is_append_only(
            Path(append_only_from),
            resolved,
        )
    return resolved


def _direct_descriptor_from_uri(provider_uri: str | Path) -> Path | None:
    candidate = Path(provider_uri).expanduser()
    if candidate.name == "PROVIDER_DESCRIPTOR.json":
        return candidate
    descriptor = candidate / "PROVIDER_DESCRIPTOR.json"
    return descriptor if descriptor.is_file() else None


def _load_direct_provider_context(provider_uri: str | Path) -> Any:
    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is None:
        raise QlibProviderHistoryError("Qlib Provider descriptor is unavailable")
    try:
        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        return _load_descriptor(descriptor, quant_data_root())
    except Exception as exc:
        raise QlibProviderHistoryError("Qlib Provider failed its integrity check") from exc


def _safe_indexed_bar_path(snapshot: Path, relative: object) -> Path:
    root = Path(snapshot).expanduser().resolve()
    relative_path = Path(str(relative or ""))
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise QlibProviderHistoryError("Qlib Provider bar path is invalid")
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise QlibProviderHistoryError("Qlib Provider bar path contains a symlink")
    candidate = current.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise QlibProviderHistoryError("Qlib Provider bar path escapes its snapshot") from exc
    if not candidate.is_file():
        raise QlibProviderHistoryError("Qlib Provider bar file is unavailable")
    return candidate


def _historical_bar_frame(path: Path, cutoff: Any):
    import pandas as pd

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise QlibProviderHistoryError("Qlib Provider bar file is unreadable") from exc
    if "date" not in frame.columns:
        raise QlibProviderHistoryError("Qlib Provider bar file has no date column")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise QlibProviderHistoryError("Qlib Provider bar dates are invalid")
    result = frame.loc[dates <= pd.Timestamp(cutoff).normalize()].copy()
    result["date"] = dates.loc[result.index]
    return result.reset_index(drop=True)


def _verify_indexed_bar_history(
    *,
    label: str,
    frozen_snapshot: Path,
    frozen_index: Any,
    current_snapshot: Path,
    current_index: Any,
    cutoff: Any,
) -> None:
    import pandas as pd

    required = {
        "symbol",
        "relative_path",
        "row_count",
        "first_date",
        "last_date",
        "sha256",
    }
    if not required.issubset(frozen_index.columns) or not required.issubset(
        current_index.columns
    ):
        raise QlibProviderHistoryError(f"{label} Provider file index is incomplete")
    if Path(frozen_snapshot).resolve() == Path(current_snapshot).resolve():
        return

    current_rows = {
        str(row.symbol): row for row in current_index.itertuples(index=False)
    }
    for frozen_row in frozen_index.itertuples(index=False):
        symbol = str(frozen_row.symbol)
        current_row = current_rows.get(symbol)
        if current_row is None:
            raise QlibProviderHistoryError(
                f"{label} historical bars changed for {symbol}"
            )
        try:
            frozen_first = pd.Timestamp(frozen_row.first_date).normalize()
            frozen_last = pd.Timestamp(frozen_row.last_date).normalize()
            current_first = pd.Timestamp(current_row.first_date).normalize()
            current_last = pd.Timestamp(current_row.last_date).normalize()
        except (TypeError, ValueError) as exc:
            raise QlibProviderHistoryError(
                f"{label} Provider file index dates are invalid"
            ) from exc
        if current_first != frozen_first or current_last < frozen_last:
            raise QlibProviderHistoryError(
                f"{label} historical bars changed for {symbol}"
            )
        if str(current_row.sha256) == str(frozen_row.sha256):
            continue

        frozen_path = _safe_indexed_bar_path(
            Path(frozen_snapshot), frozen_row.relative_path
        )
        current_path = _safe_indexed_bar_path(
            Path(current_snapshot), current_row.relative_path
        )
        if sha256_file(frozen_path) != str(frozen_row.sha256) or sha256_file(
            current_path
        ) != str(current_row.sha256):
            raise QlibProviderHistoryError(
                f"{label} Provider bar file failed its integrity check"
            )
        frozen_frame = _historical_bar_frame(frozen_path, cutoff)
        current_frame = _historical_bar_frame(current_path, cutoff)
        if set(frozen_frame.columns) != set(current_frame.columns):
            raise QlibProviderHistoryError(
                f"{label} historical bars changed for {symbol}"
            )
        try:
            pd.testing.assert_frame_equal(
                frozen_frame,
                current_frame.loc[:, frozen_frame.columns],
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise QlibProviderHistoryError(
                f"{label} historical bars changed for {symbol}"
            ) from exc


def _historical_membership(frame: Any, cutoff: Any):
    import pandas as pd

    required = ["symbol", "index_code", "start_date", "end_date"]
    if not set(required).issubset(frame.columns):
        raise QlibProviderHistoryError("Qlib Provider membership is incomplete")
    result = frame.loc[:, required].copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["index_code"] = result["index_code"].astype(str).str.upper()
    result["start_date"] = pd.to_datetime(
        result["start_date"], errors="coerce"
    ).dt.normalize()
    result["end_date"] = pd.to_datetime(
        result["end_date"], errors="coerce"
    ).dt.normalize()
    cutoff_date = pd.Timestamp(cutoff).normalize()
    result = result[result["start_date"] <= cutoff_date].copy()
    result.loc[result["end_date"] > cutoff_date, "end_date"] = cutoff_date
    return result.sort_values(required, kind="stable").reset_index(drop=True)


def _verify_direct_provider_history_is_append_only(
    frozen_provider: Path,
    current_provider: Path,
) -> None:
    import pandas as pd

    frozen = _load_direct_provider_context(frozen_provider)
    current = _load_direct_provider_context(current_provider)
    frozen_calendar = pd.DatetimeIndex(frozen.calendar).normalize()
    current_calendar = pd.DatetimeIndex(current.calendar).normalize()
    if (
        frozen_calendar.empty
        or len(current_calendar) < len(frozen_calendar)
        or not frozen_calendar.equals(current_calendar[: len(frozen_calendar)])
    ):
        raise QlibProviderHistoryError("Qlib Provider calendar history changed")
    cutoff = frozen_calendar[-1]
    if frozen.cleaning != current.cleaning:
        raise QlibProviderHistoryError("Qlib Provider cleaning history changed")
    try:
        pd.testing.assert_frame_equal(
            _historical_membership(frozen.membership, cutoff),
            _historical_membership(current.membership, cutoff),
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise QlibProviderHistoryError("Qlib Provider membership history changed") from exc

    _verify_indexed_bar_history(
        label="stock",
        frozen_snapshot=Path(frozen.snapshot_path),
        frozen_index=frozen.file_index,
        current_snapshot=Path(current.snapshot_path),
        current_index=current.file_index,
        cutoff=cutoff,
    )
    if frozen.index_snapshot_path is not None:
        if current.index_snapshot_path is None or current.index_file_index is None:
            raise QlibProviderHistoryError("index Provider history is unavailable")
        _verify_indexed_bar_history(
            label="index",
            frozen_snapshot=Path(frozen.index_snapshot_path),
            frozen_index=frozen.index_file_index,
            current_snapshot=Path(current.index_snapshot_path),
            current_index=current.index_file_index,
            cutoff=cutoff,
        )


def qlib_workflow_init_config(
    *,
    provider: str = "canonical",
    provider_uri: str | Path | None = None,
    kernels: int | None = None,
) -> dict:
    """Return a YAML-safe Qlib config for the formal Qlib Provider."""

    from quant_workflow.quant_data_qlib_provider import qlib_direct_init_config

    descriptor = (
        _direct_descriptor_from_uri(provider_uri)
        if provider_uri is not None
        else qlib_workflow_provider_descriptor(provider)
    )
    if descriptor is None:
        raise RuntimeError(
            "formal Qlib workflow requires a Qlib Provider descriptor; "
            "configured compatibility paths are not accepted"
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
    """Initialize Qlib for a workflow using the governed Qlib Provider."""

    from quant_workflow.quant_data_qlib_provider import init_qlib_from_quant_data

    descriptor = (
        _direct_descriptor_from_uri(provider_uri)
        if provider_uri is not None
        else qlib_workflow_provider_descriptor(provider)
    )
    if descriptor is None:
        raise RuntimeError(
            "formal Qlib workflow requires a Qlib Provider descriptor; "
            "configured compatibility paths are not accepted"
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
    """Build config for a Qlib Provider URI; compatibility paths remain explicit."""

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
    """Read the frozen calendar from a Qlib Provider or compatibility snapshot."""

    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is not None:
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


def provider_has_instrument_pool(provider_uri: str | Path, pool: str) -> bool:
    """Return whether a named instrument pool is present in the Provider."""

    normalized_pool = str(pool or "").strip().lower()
    descriptor = _direct_descriptor_from_uri(provider_uri)
    if descriptor is not None:
        from quant_workflow.quant_data_qlib_provider import _load_descriptor

        context = _load_descriptor(descriptor, quant_data_root())
        membership = context.membership
        if normalized_pool == "csi800":
            codes = set(membership["index_code"].astype(str).str.upper())
            return bool(
                codes.intersection({"000300.XSHG", "SH000300", "000300"})
                and codes.intersection({"000905.XSHG", "SH000905", "000905"})
            )
        if normalized_pool == "csi300":
            codes = set(membership["index_code"].astype(str).str.upper())
            return bool(codes.intersection({"000300.XSHG", "SH000300", "000300"}))
        if normalized_pool == "all":
            return not context.file_index.empty
        raise ValueError(f"Unknown Qlib instrument pool: {pool}")

    if normalized_pool not in {"all", "csi300", "csi800"}:
        raise ValueError(f"Unknown Qlib instrument pool: {pool}")
    return (Path(provider_uri) / "instruments" / f"{normalized_pool}.txt").is_file()


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
