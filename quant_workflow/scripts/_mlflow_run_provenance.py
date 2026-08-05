"""Fail-closed MLflow run selection bound to one Qlib provider snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

try:
    from quant_data_contract import DataContractError, validate_qlib_snapshot
except ImportError:  # pragma: no cover - package import when used by the API.
    from quant_workflow.scripts.quant_data_contract import (  # type: ignore
        DataContractError,
        validate_qlib_snapshot,
    )


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
TAG_PREFIX = "dsa.quant_data."
PROVIDER_IDENTITY_FIELDS = (
    "provider_contract",
    "dataset_id",
    "snapshot_id",
    "content_sha256",
    "manifest_sha256",
)
PREDICTION_ARTIFACTS = {"pred.pkl", "label.pkl"}
ARTIFACT_SHA256_TAGS = {
    "pred.pkl": "dsa.artifact.pred_sha256",
    "label.pkl": "dsa.artifact.label_sha256",
    "task": "dsa.artifact.task_sha256",
    "portfolio_analysis/report_normal_1day.pkl": (
        "dsa.artifact.portfolio_report_sha256"
    ),
}
NATIVE_QLIB_DATASET_ROOTS = {
    "qlib_ashare_v2": Path("qlib/cn_data_ashare_v2"),
    "qlib_ashare_cleaned_v2": Path("qlib/cn_data_ashare_cleaned_v2"),
}
PROVIDER_DATASET_IDS = {
    "canonical": "qlib_ashare_v2",
    "cleaned": "qlib_ashare_cleaned_v2",
}


class RunProvenanceError(RuntimeError):
    """Raised when a prediction Run cannot prove its data lineage."""


def resolve_immutable_provider_identity(
    identity: Mapping[str, Any],
    *,
    quant_data_root: Path,
) -> Path:
    """Resolve and fully validate the native snapshot recorded by an MLflow run.

    The saved absolute ``provider_uri`` is deliberately ignored.  A historical
    report may have been moved to another machine, and a mutable compatibility
    path must never regain authority merely because it is still present.
    """

    if identity.get("provider_contract") != "immutable_snapshot_v2":
        raise RunProvenanceError(
            "selected training run does not record an immutable native V2 provider"
        )
    dataset_id = str(identity.get("dataset_id") or "")
    snapshot_id = str(identity.get("snapshot_id") or "")
    content_sha256 = str(identity.get("content_sha256") or "")
    manifest_sha256 = str(identity.get("manifest_sha256") or "")
    if dataset_id not in NATIVE_QLIB_DATASET_ROOTS:
        raise RunProvenanceError("selected training run uses an unsupported Qlib dataset")
    if (
        not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or snapshot_id in {".", ".."}
        or not SHA256_RE.fullmatch(content_sha256)
        or not SHA256_RE.fullmatch(manifest_sha256)
    ):
        raise RunProvenanceError("selected training run has an incomplete provider identity")

    root = Path(quant_data_root).expanduser().resolve()
    dataset_root = (root / NATIVE_QLIB_DATASET_ROOTS[dataset_id]).resolve()
    try:
        dataset_root.relative_to(root)
    except ValueError as exc:
        raise RunProvenanceError(
            "selected provider dataset root must remain under QUANT_DATA_ROOT"
        ) from exc
    snapshot = (dataset_root / "snapshots" / snapshot_id).resolve()
    expected_parent = (dataset_root / "snapshots").resolve()
    if snapshot.parent != expected_parent or snapshot.is_symlink():
        raise RunProvenanceError("selected provider is not an immutable native snapshot")
    try:
        manifest = validate_qlib_snapshot(
            snapshot,
            verify_hashes=True,
            evidence_root=root,
        )
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise RunProvenanceError(
            "selected native Qlib snapshot failed its integrity check"
        ) from exc
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    actual_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if (
        manifest.get("dataset_id") != dataset_id
        or manifest.get("snapshot_id") != snapshot_id
        or manifest.get("role") != "canonical"
        or manifest.get("status") != "ready"
        or manifest.get("price_basis") != "qfq"
        or manifest.get("benchmark_complete") is not True
        or manifest.get("content_sha256") != content_sha256
        or actual_manifest_sha256 != manifest_sha256
    ):
        raise RunProvenanceError(
            "selected native Qlib snapshot does not match the MLflow provider identity"
        )
    return snapshot


def resolve_approved_provider_identity(
    provider: str,
    identity: Mapping[str, Any],
    *,
    quant_data_root: Path,
) -> Path:
    """Resolve one API-approved provider identity without consulting LATEST."""

    provider_name = str(provider or "canonical").strip().lower()
    expected_dataset_id = PROVIDER_DATASET_IDS.get(provider_name)
    if expected_dataset_id is None:
        raise RunProvenanceError("approved provider name is unsupported")
    if identity.get("dataset_id") != expected_dataset_id:
        raise RunProvenanceError(
            "approved provider identity does not match the selected provider"
        )
    snapshot = resolve_immutable_provider_identity(
        identity,
        quant_data_root=quant_data_root,
    )
    actual = provider_identity(snapshot)
    if any(actual.get(field) != identity.get(field) for field in PROVIDER_IDENTITY_FIELDS):
        raise RunProvenanceError(
            "approved provider identity changed before training started"
        )
    return snapshot


def _file_store_run_directory(mlruns_dir: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(str(run_id or "")):
        raise RunProvenanceError("MLflow run id is invalid")
    root = Path(mlruns_dir).expanduser().resolve()
    matches = [
        path
        for path in root.glob(f"*/{run_id}")
        if path.is_dir() and not path.is_symlink() and not path.parent.is_symlink()
    ]
    if len(matches) != 1:
        raise RunProvenanceError(
            f"MLflow run {run_id} must resolve to exactly one file-store directory"
        )
    return matches[0]


def _file_store_provider_identity(run_dir: Path) -> dict[str, str]:
    tags_dir = run_dir / "tags"
    if not tags_dir.is_dir() or tags_dir.is_symlink():
        raise RunProvenanceError("MLflow run provider identity tags are unavailable")
    identity: dict[str, str] = {}
    for field in (*PROVIDER_IDENTITY_FIELDS, "provider_uri"):
        tag_path = tags_dir / f"{TAG_PREFIX}{field}"
        if not tag_path.is_file() or tag_path.is_symlink():
            if field in PROVIDER_IDENTITY_FIELDS:
                raise RunProvenanceError(
                    f"MLflow run provider identity is missing {field}"
                )
            continue
        try:
            if tag_path.stat().st_size > 4096:
                raise RunProvenanceError(
                    f"MLflow run provider identity tag {field} is too large"
                )
            value = tag_path.read_text(encoding="utf-8").strip()
        except RunProvenanceError:
            raise
        except (OSError, UnicodeError) as exc:
            raise RunProvenanceError(
                f"MLflow run provider identity tag {field} is unreadable"
            ) from exc
        if not value and field in PROVIDER_IDENTITY_FIELDS:
            raise RunProvenanceError(
                f"MLflow run provider identity is missing {field}"
            )
        if value:
            identity[field] = value
    return identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_store_artifact_path(run_dir: Path, relative: Path) -> Path:
    relative_text = relative.as_posix()
    if relative_text not in ARTIFACT_SHA256_TAGS:
        raise RunProvenanceError("MLflow artifact is not an approved report input")
    artifacts_root = (run_dir / "artifacts").resolve()
    expected = run_dir / "artifacts" / relative
    current = run_dir
    for part in (Path("artifacts") / relative).parts:
        current = current / part
        if current.is_symlink():
            raise RunProvenanceError("MLflow artifact path cannot contain symlinks")
    try:
        resolved = expected.resolve()
        resolved.relative_to(artifacts_root)
    except (OSError, ValueError) as exc:
        raise RunProvenanceError("MLflow artifact path escapes its Run") from exc
    if not expected.is_file() or expected.is_symlink():
        raise RunProvenanceError("MLflow artifact is unavailable")
    return resolved


def tag_run_artifact_hashes(
    client: Any,
    run_id: str,
    *,
    mlruns_dir: Path,
) -> dict[str, str]:
    """Record hashes for supported local artifacts after a Run finishes."""

    run_dir = _file_store_run_directory(mlruns_dir, run_id)
    hashes: dict[str, str] = {}
    for relative_text, tag_name in ARTIFACT_SHA256_TAGS.items():
        candidate = run_dir / "artifacts" / Path(relative_text)
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved = _file_store_artifact_path(run_dir, Path(relative_text))
        digest = _sha256_file(resolved)
        client.set_tag(run_id, tag_name, digest)
        hashes[relative_text] = digest
    return hashes


def validate_file_store_run_artifact(
    *,
    artifact_path: Path,
    artifact_relative_path: Path,
    run_id: str,
    mlruns_dir: Path,
    quant_data_root: Path,
) -> dict[str, Any]:
    """Bind one local MLflow artifact to its Run and immutable V2 provider."""

    relative = Path(artifact_relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RunProvenanceError("MLflow artifact relative path is invalid")
    run_dir = _file_store_run_directory(mlruns_dir, run_id)
    expected_resolved = _file_store_artifact_path(run_dir, relative)
    if (
        Path(artifact_path).expanduser().resolve() != expected_resolved
    ):
        raise RunProvenanceError(
            "MLflow artifact does not belong to the requested Run"
        )
    artifact_tag = ARTIFACT_SHA256_TAGS[relative.as_posix()]
    artifact_tag_path = run_dir / "tags" / artifact_tag
    if not artifact_tag_path.is_file() or artifact_tag_path.is_symlink():
        raise RunProvenanceError("MLflow artifact content hash is unavailable")
    try:
        if artifact_tag_path.stat().st_size > 128:
            raise RunProvenanceError("MLflow artifact content hash is invalid")
        expected_artifact_sha256 = artifact_tag_path.read_text(
            encoding="utf-8"
        ).strip()
    except RunProvenanceError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RunProvenanceError("MLflow artifact content hash is unreadable") from exc
    if (
        not SHA256_RE.fullmatch(expected_artifact_sha256)
        or _sha256_file(expected_resolved) != expected_artifact_sha256
    ):
        raise RunProvenanceError("MLflow artifact content hash does not match")
    identity = _file_store_provider_identity(run_dir)
    provider_path = resolve_immutable_provider_identity(
        identity,
        quant_data_root=quant_data_root,
    )
    return {
        "provider_identity": identity,
        "provider_path": str(provider_path),
        "artifact_sha256": expected_artifact_sha256,
    }


def provider_identity(provider_uri: Path) -> dict[str, str]:
    provider = Path(provider_uri).expanduser().resolve()
    identity = {
        "provider_uri": str(provider),
        "provider_contract": "path_only_override",
    }
    manifest_path = provider / "DATASET_MANIFEST.json"
    if not manifest_path.is_file():
        return identity
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunProvenanceError("Qlib provider manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise RunProvenanceError("Qlib provider manifest must be an object")
    dataset_id = str(manifest.get("dataset_id") or "")
    snapshot_id = str(manifest.get("snapshot_id") or "")
    content_sha256 = str(manifest.get("content_sha256") or "")
    if (
        not dataset_id
        or not snapshot_id
        or provider.name != snapshot_id
        or not SHA256_RE.fullmatch(content_sha256)
    ):
        raise RunProvenanceError("Qlib provider snapshot identity is incomplete")
    identity.update(
        {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "content_sha256": content_sha256,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )
    return identity


def revalidate_provider_identity(
    provider_uri: Path,
    expected_identity: Mapping[str, Any],
    *,
    quant_data_root: Path,
) -> dict[str, str]:
    """Fully revalidate a pinned provider and require its identity to be unchanged."""

    provider = Path(provider_uri).expanduser().resolve()
    validated_provider = resolve_immutable_provider_identity(
        expected_identity,
        quant_data_root=quant_data_root,
    )
    if provider != validated_provider:
        raise RunProvenanceError(
            "selected provider path changed after training approval"
        )
    actual_identity = provider_identity(provider)
    if any(
        actual_identity.get(field) != expected_identity.get(field)
        for field in PROVIDER_IDENTITY_FIELDS
    ):
        raise RunProvenanceError(
            "selected provider identity changed during training"
        )
    return actual_identity


def provider_tags(identity: Mapping[str, str]) -> dict[str, str]:
    return {f"{TAG_PREFIX}{key}": str(value) for key, value in identity.items()}


def tag_run_with_provider(client: Any, run_id: str, identity: Mapping[str, str]) -> None:
    for key, value in provider_tags(identity).items():
        client.set_tag(run_id, key, value)


def experiment_run_ids(client: Any, experiment_id: str) -> set[str]:
    return {
        str(run.info.run_id)
        for run in client.search_runs(
            experiment_ids=[str(experiment_id)],
            order_by=["start_time DESC"],
            max_results=1000,
        )
    }


def _has_prediction_artifacts(client: Any, run_id: str) -> bool:
    paths = {str(item.path) for item in client.list_artifacts(run_id)}
    return PREDICTION_ARTIFACTS.issubset(paths)


def _is_finished(run: Any) -> bool:
    return str(getattr(run.info, "status", "")).upper() == "FINISHED"


def select_new_prediction_run(
    client: Any,
    experiment_id: str,
    before_run_ids: set[str],
) -> Any:
    runs = client.search_runs(
        experiment_ids=[str(experiment_id)],
        order_by=["start_time DESC"],
        max_results=1000,
    )
    candidates = [
        run
        for run in runs
        if str(run.info.run_id) not in before_run_ids
        and _is_finished(run)
        and _has_prediction_artifacts(client, str(run.info.run_id))
    ]
    if len(candidates) != 1:
        raise RunProvenanceError(
            "training did not produce exactly one identifiable prediction Run"
        )
    return candidates[0]


def select_matching_prediction_run(
    client: Any,
    experiment_id: str,
    identity: Mapping[str, str],
) -> Any | None:
    if identity.get("provider_contract") == "path_only_override":
        return None
    expected = provider_tags(identity)
    runs = client.search_runs(
        experiment_ids=[str(experiment_id)],
        order_by=["start_time DESC"],
        max_results=1000,
    )
    for run in runs:
        tags = getattr(run.data, "tags", {}) or {}
        if (
            _is_finished(run)
            and all(str(tags.get(key, "")) == value for key, value in expected.items())
            and _has_prediction_artifacts(client, str(run.info.run_id))
        ):
            return run
    return None
