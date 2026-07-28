"""Fail-closed MLflow run selection bound to one Qlib provider snapshot."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TAG_PREFIX = "dsa.quant_data."
PREDICTION_ARTIFACTS = {"pred.pkl", "label.pkl"}


class RunProvenanceError(RuntimeError):
    """Raised when a prediction Run cannot prove its data lineage."""


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
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        }
    )
    return identity


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
