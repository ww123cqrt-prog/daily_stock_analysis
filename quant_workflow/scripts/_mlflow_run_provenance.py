"""MLflow provenance for frozen QMT inputs and read-only historical identities."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from quant_qmt_data.dataset import load_dataset

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
TAG_PREFIX = "dsa.quant_data."
PROVIDER_IDENTITY_FIELDS = ("backend", "experiment_id", "bars_sha256", "inputs_sha256", "price_basis")
PROVIDER_OPTIONAL_IDENTITY_FIELDS = ("input_dir",)
HISTORICAL_IDENTITY_FIELDS = (
    "provider_contract", "dataset_id", "snapshot_id", "content_sha256", "manifest_sha256",
)
DIRECT_IDENTITY_FIELDS = (
    "descriptor_path", "descriptor_sha256", "cleaning_mode", "index_dataset_id",
    "index_snapshot_id", "index_content_sha256", "index_manifest_sha256",
)
NATIVE_QLIB_DATASET_ROOTS = {
    "qlib_ashare_v2": Path("qlib/cn_data_ashare_v2"),
    "qlib_ashare_cleaned_v2": Path("qlib/cn_data_ashare_cleaned_v2"),
}
PROVIDER_DATASET_IDS = {"canonical": "qlib_ashare_v2", "cleaned": "qlib_ashare_cleaned_v2"}
PREDICTION_ARTIFACTS = {"pred.pkl", "label.pkl"}
ARTIFACT_SHA256_TAGS = {
    "pred.pkl": "dsa.artifact.pred_sha256",
    "label.pkl": "dsa.artifact.label_sha256",
    "task": "dsa.artifact.task_sha256",
    "portfolio_analysis/report_normal_1day.pkl": "dsa.artifact.portfolio_report_sha256",
}


class RunProvenanceError(RuntimeError):
    """Raised when a prediction Run cannot prove its data lineage."""


def validate_qlib_snapshot(*args, **kwargs):
    """Load DSA's existing validator only for an explicit historical input."""
    from quant_workflow.scripts.quant_data_contract import validate_qlib_snapshot
    return validate_qlib_snapshot(*args, **kwargs)


def normalize_provider_lineage(identity: Mapping[str, Any]) -> dict[str, str]:
    """Return the public identity; storage paths never leave this boundary."""
    if not isinstance(identity, Mapping):
        raise RunProvenanceError("provider lineage must be an object")
    if identity.get("backend") == "qmt_native":
        normalized = {field: str(identity.get(field) or "") for field in PROVIDER_IDENTITY_FIELDS}
        if (not normalized["experiment_id"]
                or any(not SHA256_RE.fullmatch(normalized[field]) for field in ("bars_sha256", "inputs_sha256"))
                or normalized["price_basis"] not in ("none", "front", "back", "front_ratio", "back_ratio")):
            raise RunProvenanceError("QMT provider lineage is incomplete or invalid (including inputs_sha256)")
        return normalized

    # These are the two saved DSA contracts from before the QMT migration.
    contract = str(identity.get("provider_contract") or "")
    if contract == "direct_parquet_v1":
        fields = (*HISTORICAL_IDENTITY_FIELDS, *DIRECT_IDENTITY_FIELDS)
    elif contract == "immutable_snapshot_v2":
        fields = HISTORICAL_IDENTITY_FIELDS
    else:
        raise RunProvenanceError("provider lineage contract is unsupported")
    normalized = {field: str(identity.get(field) or "") for field in fields}
    if any(not value for value in normalized.values()):
        raise RunProvenanceError("provider lineage is incomplete")
    snapshot_id = normalized["snapshot_id"]
    if Path(snapshot_id).name != snapshot_id or snapshot_id in {".", ".."}:
        raise RunProvenanceError("provider lineage snapshot id is invalid")
    if any(not SHA256_RE.fullmatch(normalized[field]) for field in ("content_sha256", "manifest_sha256")):
        raise RunProvenanceError("provider lineage hashes are invalid")
    if contract == "direct_parquet_v1":
        descriptor_path = Path(normalized["descriptor_path"])
        index_snapshot_id = normalized["index_snapshot_id"]
        if (PurePosixPath(normalized["descriptor_path"]).anchor
                or PureWindowsPath(normalized["descriptor_path"]).anchor
                or ".." in descriptor_path.parts or descriptor_path.name != "PROVIDER_DESCRIPTOR.json"
                or Path(index_snapshot_id).name != index_snapshot_id or index_snapshot_id in {".", ".."}
                or normalized["cleaning_mode"] not in {"canonical", "cleaned"}
                or any(not SHA256_RE.fullmatch(normalized[field]) for field in
                       ("descriptor_sha256", "index_content_sha256", "index_manifest_sha256"))):
            raise RunProvenanceError("provider lineage metadata is invalid")
    return normalized


def provider_lineage_is_complete(identity: Mapping[str, Any] | None) -> bool:
    try:
        normalize_provider_lineage(identity)
    except RunProvenanceError:
        return False
    return True


def provider_lineages_match(actual, expected) -> bool:
    try:
        return normalize_provider_lineage(actual) == normalize_provider_lineage(expected)
    except RunProvenanceError:
        return False


def provider_identity(provider_uri: Path, *, quant_data_root: Path | None = None) -> dict[str, str]:
    directory = Path(provider_uri).expanduser().resolve()
    descriptor = directory if directory.name == "PROVIDER_DESCRIPTOR.json" else directory / "PROVIDER_DESCRIPTOR.json"
    if descriptor.is_file():
        try:
            payload = json.loads(descriptor.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunProvenanceError("Qlib Provider descriptor is invalid") from exc
        dataset = payload.get("dataset")
        if not isinstance(dataset, Mapping):
            raise RunProvenanceError("Qlib Provider stock dataset identity is missing")
        root = Path(quant_data_root).expanduser().resolve() if quant_data_root is not None else descriptor.parent
        if quant_data_root is None:
            for parent in [directory, *directory.parents]:
                if parent.name == "quant_data":
                    root = parent
                    break
        try:
            descriptor_path = descriptor.relative_to(root).as_posix()
        except ValueError:
            descriptor_path = descriptor.as_posix()
        identity = {
            "provider_uri": str(descriptor.parent), "provider_contract": "direct_parquet_v1",
            **{key: str(dataset.get(key) or "") for key in HISTORICAL_IDENTITY_FIELDS if key != "provider_contract"},
            "descriptor_path": descriptor_path, "descriptor_sha256": _sha256_file(descriptor),
            "cleaning_mode": str((payload.get("cleaning") or {}).get("mode") or "canonical"),
        }
        index_dataset = payload.get("index_dataset")
        if isinstance(index_dataset, Mapping):
            identity.update({f"index_{key}": str(index_dataset.get(key) or "")
                for key in ("dataset_id", "snapshot_id", "content_sha256", "manifest_sha256")})
        return identity
    manifest_path = directory / "DATASET_MANIFEST.json"
    if manifest_path.is_file():
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunProvenanceError("Qlib provider manifest is invalid") from exc
        if not isinstance(manifest, Mapping):
            raise RunProvenanceError("Qlib provider manifest must be an object")
        if (not manifest.get("dataset_id") or manifest.get("snapshot_id") != directory.name
                or not SHA256_RE.fullmatch(str(manifest.get("content_sha256") or ""))):
            raise RunProvenanceError("Qlib provider snapshot identity is incomplete")
        return dict(provider_uri=str(directory), provider_contract="immutable_snapshot_v2",
            dataset_id=str(manifest["dataset_id"]), snapshot_id=str(manifest["snapshot_id"]),
            content_sha256=str(manifest["content_sha256"]),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())

    metadata = load_dataset(directory).attrs
    return {"backend": "qmt_native", "experiment_id": metadata["experiment_id"],
            "bars_sha256": metadata["bars_sha256"], "inputs_sha256": metadata["inputs_sha256"],
            "price_basis": metadata["price_basis"],
            "input_dir": str(directory)}


def resolve_immutable_provider_identity(identity, *, quant_data_root=None) -> Path:
    normalized = normalize_provider_lineage(identity)
    if normalized.get("backend") == "qmt_native":
        configured = identity.get("input_dir") or os.environ.get("QMT_DATASET_DIR")
        if not configured:
            raise RunProvenanceError("QMT_DATASET_DIR is required to resolve public provider lineage")
        directory = Path(configured).expanduser().resolve()
        actual = provider_identity(directory)
        if not provider_lineages_match(actual, normalized):
            raise RunProvenanceError("frozen input identity changed")
        return directory

    if quant_data_root is None:
        raise RunProvenanceError("Historical Qlib Provider requires its quant_data_root")
    root = Path(quant_data_root).expanduser().resolve()
    if normalized["provider_contract"] == "direct_parquet_v1":
        descriptor = root / normalized["descriptor_path"]
        current = root
        for part in Path(normalized["descriptor_path"]).parts:
            current = current / part
            if current.is_symlink():
                raise RunProvenanceError("Qlib Provider descriptor cannot contain symlinks")
        if not descriptor.is_file():
            raise RunProvenanceError("Qlib Provider descriptor is unavailable")
        # Resolve the saved descriptor identity, without reviving its retired
        # feature-provider loader or consulting today's QMT input.
        actual = provider_identity(descriptor, quant_data_root=root)
        if actual.get("dataset_id") != "ashare_daily_qfq" or not provider_lineages_match(actual, normalized):
            raise RunProvenanceError("Qlib Provider descriptor does not match the MLflow provider identity")
        return descriptor.parent

    dataset_id = normalized["dataset_id"]
    if dataset_id not in NATIVE_QLIB_DATASET_ROOTS:
        raise RunProvenanceError("selected training run uses an unsupported Qlib dataset")
    dataset_root = (root / NATIVE_QLIB_DATASET_ROOTS[dataset_id]).resolve()
    if not dataset_root.is_relative_to(root):
        raise RunProvenanceError("selected provider dataset root must remain under QUANT_DATA_ROOT")
    snapshot = (dataset_root / "snapshots" / normalized["snapshot_id"]).resolve()
    if snapshot.parent != (dataset_root / "snapshots").resolve() or snapshot.is_symlink():
        raise RunProvenanceError("selected Qlib Provider snapshot is invalid")
    from quant_workflow.scripts.quant_data_contract import DataContractError

    try:
        manifest = validate_qlib_snapshot(snapshot, verify_hashes=True, evidence_root=root)
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise RunProvenanceError("selected Qlib Provider snapshot failed its integrity check") from exc
    if (manifest.get("dataset_id") != dataset_id or manifest.get("snapshot_id") != normalized["snapshot_id"]
            or manifest.get("role") != "canonical" or manifest.get("status") != "ready"
            or manifest.get("price_basis") != "qfq" or manifest.get("benchmark_complete") is not True
            or manifest.get("content_sha256") != normalized["content_sha256"]
            or _sha256_file(snapshot / "DATASET_MANIFEST.json") != normalized["manifest_sha256"]):
        raise RunProvenanceError("selected Qlib Provider snapshot does not match the MLflow provider identity")
    return snapshot


def resolve_approved_provider_identity(provider, identity, *, quant_data_root=None) -> Path:
    normalized = normalize_provider_lineage(identity)
    if normalized.get("backend") == "qmt_native":
        matches = provider == "qmt_native"
    elif provider in PROVIDER_DATASET_IDS:
        matches = (normalized["cleaning_mode"] == provider if normalized["provider_contract"] == "direct_parquet_v1"
                   else normalized["dataset_id"] == PROVIDER_DATASET_IDS[provider])
    else:
        matches = False
    if not matches:
        raise RunProvenanceError("approved provider identity does not match the selected provider")
    return resolve_immutable_provider_identity(identity, quant_data_root=quant_data_root)


def revalidate_provider_identity(provider_uri, expected_identity, *, quant_data_root=None) -> dict[str, str]:
    if expected_identity.get("backend") != "qmt_native":
        resolved = resolve_immutable_provider_identity(expected_identity, quant_data_root=quant_data_root)
        if Path(provider_uri).expanduser().resolve() != resolved:
            raise RunProvenanceError("selected provider path changed after training approval")
    actual = provider_identity(provider_uri, quant_data_root=quant_data_root)
    if not provider_lineages_match(actual, expected_identity):
        raise RunProvenanceError("selected provider identity changed during training")
    return actual


def provider_tags(identity: Mapping[str, str]) -> dict[str, str]:
    public = normalize_provider_lineage(identity)
    location = "input_dir" if public.get("backend") == "qmt_native" else "provider_uri"
    if identity.get(location):
        public[location] = str(identity[location])
    return {f"{TAG_PREFIX}{key}": value for key, value in public.items()}


def tag_run_with_provider(client: Any, run_id: str, identity: Mapping[str, str]) -> None:
    for key, value in provider_tags(identity).items():
        client.set_tag(run_id, key, value)


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
    for field in (
        *PROVIDER_IDENTITY_FIELDS,
        *PROVIDER_OPTIONAL_IDENTITY_FIELDS,
        *HISTORICAL_IDENTITY_FIELDS,
        *DIRECT_IDENTITY_FIELDS,
        "provider_uri",
    ):
        tag_path = tags_dir / f"{TAG_PREFIX}{field}"
        if not tag_path.is_file() or tag_path.is_symlink():
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
        if value:
            identity[field] = value
    public = normalize_provider_lineage(identity)
    location = "input_dir" if public.get("backend") == "qmt_native" else "provider_uri"
    if location in identity:
        public[location] = identity[location]
    return public


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
    """Bind a local MLflow artifact to its Run and recorded input version."""

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
    expected = provider_tags(normalize_provider_lineage(identity))
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
