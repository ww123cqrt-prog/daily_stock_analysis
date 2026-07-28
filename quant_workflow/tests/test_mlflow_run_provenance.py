from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "_mlflow_run_provenance.py"
    )
    spec = importlib.util.spec_from_file_location("_mlflow_run_provenance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def fake_run(run_id: str, *, tags: dict[str, str] | None = None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status="FINISHED"),
        data=SimpleNamespace(tags=tags or {}, metrics={}),
    )


class FakeClient:
    def __init__(self, runs, artifacts=None):
        self.runs = list(runs)
        self.artifacts = artifacts or {
            run.info.run_id: {"pred.pkl", "label.pkl"} for run in runs
        }
        self.tags = {}

    def search_runs(self, **_kwargs):
        return self.runs

    def list_artifacts(self, run_id):
        return [SimpleNamespace(path=path) for path in self.artifacts.get(run_id, set())]

    def set_tag(self, run_id, key, value):
        self.tags[(run_id, key)] = value


def test_provider_identity_records_native_snapshot(tmp_path: Path) -> None:
    module = load_module()
    provider = tmp_path / "qlib-snapshot-001"
    provider.mkdir()
    (provider / "DATASET_MANIFEST.json").write_text(
        json.dumps(
            {
                "dataset_id": "qlib_ashare_v2",
                "snapshot_id": provider.name,
                "content_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    identity = module.provider_identity(provider)

    assert identity == {
        "provider_uri": str(provider.resolve()),
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": provider.name,
        "content_sha256": "a" * 64,
    }


def test_existing_run_must_match_every_provider_tag() -> None:
    module = load_module()
    identity = {
        "provider_uri": "/native/snapshot",
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": "snapshot-001",
        "content_sha256": "a" * 64,
    }
    legacy = fake_run("legacy", tags={})
    matching = fake_run("matching", tags=module.provider_tags(identity))
    client = FakeClient([legacy, matching])

    selected = module.select_matching_prediction_run(client, "1", identity)

    assert selected.info.run_id == "matching"


def test_mutable_path_override_never_reuses_an_existing_prediction_run() -> None:
    module = load_module()
    identity = {
        "provider_uri": "/legacy/mutable/provider",
        "provider_contract": "path_only_override",
    }
    matching = fake_run("matching", tags=module.provider_tags(identity))
    client = FakeClient([matching])

    assert module.select_matching_prediction_run(client, "1", identity) is None


def test_new_run_selection_fails_closed_when_concurrent_runs_are_ambiguous() -> None:
    module = load_module()
    client = FakeClient([fake_run("new-a"), fake_run("new-b")])

    with pytest.raises(module.RunProvenanceError, match="exactly one"):
        module.select_new_prediction_run(client, "1", {"old"})


def test_tag_run_writes_provider_identity() -> None:
    module = load_module()
    identity = {"provider_uri": "/native/snapshot", "snapshot_id": "snap-1"}
    client = FakeClient([fake_run("run-1")])

    module.tag_run_with_provider(client, "run-1", identity)

    assert client.tags[("run-1", "dsa.quant_data.provider_uri")] == "/native/snapshot"
    assert client.tags[("run-1", "dsa.quant_data.snapshot_id")] == "snap-1"
