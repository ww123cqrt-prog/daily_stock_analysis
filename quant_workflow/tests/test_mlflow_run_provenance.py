from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    manifest_path = provider / "DATASET_MANIFEST.json"
    manifest_path.write_text(
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
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def test_approved_provider_identity_is_rechecked_before_training(
    tmp_path: Path,
) -> None:
    module = load_module()
    snapshot = tmp_path / "canonical-snapshot"
    expected = {
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": snapshot.name,
        "content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }

    with (
        patch.object(
            module,
            "resolve_immutable_provider_identity",
            return_value=snapshot,
        ) as resolve_identity,
        patch.object(module, "provider_identity", return_value=expected),
    ):
        resolved = module.resolve_approved_provider_identity(
            "canonical",
            expected,
            quant_data_root=tmp_path,
        )

    assert resolved == snapshot
    resolve_identity.assert_called_once_with(expected, quant_data_root=tmp_path)

    with pytest.raises(module.RunProvenanceError, match="selected provider"):
        module.resolve_approved_provider_identity(
            "cleaned",
            expected,
            quant_data_root=tmp_path,
        )

    changed = {**expected, "content_sha256": "c" * 64}
    with (
        patch.object(
            module,
            "resolve_immutable_provider_identity",
            return_value=snapshot,
        ),
        patch.object(module, "provider_identity", return_value=changed),
        pytest.raises(module.RunProvenanceError, match="changed before training"),
    ):
        module.resolve_approved_provider_identity(
            "canonical",
            expected,
            quant_data_root=tmp_path,
        )


def test_revalidate_provider_identity_rejects_content_changed_during_training(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "qlib" / "cn_data_ashare_v2" / "snapshots" / "native-v2"
    provider.mkdir(parents=True)
    expected = {
        "provider_uri": str(provider),
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": "native-v2",
        "content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }

    with patch.object(
        module,
        "resolve_immutable_provider_identity",
        side_effect=module.RunProvenanceError(
            "selected native Qlib snapshot failed its integrity check"
        ),
    ):
        with pytest.raises(module.RunProvenanceError, match="integrity check"):
            module.revalidate_provider_identity(
                provider,
                expected,
                quant_data_root=tmp_path,
            )


def test_resolve_identity_uses_native_snapshot_and_ignores_saved_absolute_path(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = tmp_path / "quant_data"
    snapshot = (
        root
        / "qlib"
        / "cn_data_ashare_v2"
        / "snapshots"
        / "qlib-native-001"
    )
    snapshot.mkdir(parents=True)
    manifest = {
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": snapshot.name,
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "benchmark_complete": True,
        "content_sha256": "a" * 64,
    }
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    identity = {
        "provider_uri": "/old-machine/legacy-provider",
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": snapshot.name,
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }

    with patch.object(module, "validate_qlib_snapshot", return_value=manifest):
        resolved = module.resolve_immutable_provider_identity(
            identity,
            quant_data_root=root,
        )

    assert resolved == snapshot.resolve()


def test_resolve_identity_rejects_path_only_compatibility_report(
    tmp_path: Path,
) -> None:
    module = load_module()

    with pytest.raises(module.RunProvenanceError, match="immutable native V2"):
        module.resolve_immutable_provider_identity(
            {
                "provider_uri": str(
                    tmp_path / "qlib" / "cn_data_baostock_jqdata"
                ),
                "provider_contract": "path_only_override",
            },
            quant_data_root=tmp_path,
        )


def test_resolve_identity_rejects_dataset_root_symlink_outside_quant_data(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = tmp_path / "quant_data"
    outside_dataset = tmp_path / "outside-dataset"
    snapshot = outside_dataset / "snapshots" / "qlib-native-001"
    snapshot.mkdir(parents=True)
    dataset_link = root / "qlib" / "cn_data_ashare_v2"
    dataset_link.parent.mkdir(parents=True)
    dataset_link.symlink_to(outside_dataset, target_is_directory=True)
    manifest = {
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": snapshot.name,
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "benchmark_complete": True,
        "content_sha256": "a" * 64,
    }
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    identity = {
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": snapshot.name,
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }

    with (
        patch.object(module, "validate_qlib_snapshot", return_value=manifest),
        pytest.raises(module.RunProvenanceError, match="under QUANT_DATA_ROOT"),
    ):
        module.resolve_immutable_provider_identity(
            identity,
            quant_data_root=root,
        )


def test_file_store_artifact_is_bound_to_run_and_native_provider(
    tmp_path: Path,
) -> None:
    module = load_module()
    mlruns = tmp_path / "mlruns"
    run_dir = mlruns / "123" / "safe_run"
    artifact = run_dir / "artifacts" / "pred.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"prediction")
    identity = {
        "provider_contract": "immutable_snapshot_v2",
        "dataset_id": "qlib_ashare_v2",
        "snapshot_id": "snapshot-001",
        "content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    tags_dir = run_dir / "tags"
    tags_dir.mkdir()
    for key, value in identity.items():
        (tags_dir / f"{module.TAG_PREFIX}{key}").write_text(value, encoding="utf-8")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tags_dir / module.ARTIFACT_SHA256_TAGS["pred.pkl"]).write_text(
        artifact_sha256,
        encoding="utf-8",
    )
    provider = tmp_path / "quant_data" / "qlib" / "snapshot-001"

    with patch.object(
        module,
        "resolve_immutable_provider_identity",
        return_value=provider,
    ) as resolve_identity:
        provenance = module.validate_file_store_run_artifact(
            artifact_path=artifact,
            artifact_relative_path=Path("pred.pkl"),
            run_id="safe_run",
            mlruns_dir=mlruns,
            quant_data_root=tmp_path / "quant_data",
        )

    assert provenance == {
        "provider_identity": identity,
        "provider_path": str(provider),
        "artifact_sha256": artifact_sha256,
    }
    resolve_identity.assert_called_once_with(
        identity,
        quant_data_root=tmp_path / "quant_data",
    )

    artifact.write_bytes(b"tampered prediction")
    with pytest.raises(module.RunProvenanceError, match="artifact content hash"):
        module.validate_file_store_run_artifact(
            artifact_path=artifact,
            artifact_relative_path=Path("pred.pkl"),
            run_id="safe_run",
            mlruns_dir=mlruns,
            quant_data_root=tmp_path / "quant_data",
        )

    artifact.write_bytes(b"prediction")
    (tags_dir / f"{module.TAG_PREFIX}content_sha256").unlink()
    with pytest.raises(module.RunProvenanceError, match="provider identity"):
        module.validate_file_store_run_artifact(
            artifact_path=artifact,
            artifact_relative_path=Path("pred.pkl"),
            run_id="safe_run",
            mlruns_dir=mlruns,
            quant_data_root=tmp_path / "quant_data",
        )


def test_tag_run_artifact_hashes_records_every_supported_local_artifact(
    tmp_path: Path,
) -> None:
    module = load_module()
    mlruns = tmp_path / "mlruns"
    run_dir = mlruns / "123" / "safe_run"
    pred = run_dir / "artifacts" / "pred.pkl"
    report = (
        run_dir
        / "artifacts"
        / "portfolio_analysis"
        / "report_normal_1day.pkl"
    )
    pred.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    pred.write_bytes(b"prediction")
    report.write_bytes(b"portfolio")
    client = FakeClient([fake_run("safe_run")])

    hashes = module.tag_run_artifact_hashes(
        client,
        "safe_run",
        mlruns_dir=mlruns,
    )

    assert hashes == {
        "portfolio_analysis/report_normal_1day.pkl": hashlib.sha256(
            report.read_bytes()
        ).hexdigest(),
        "pred.pkl": hashlib.sha256(pred.read_bytes()).hexdigest(),
    }
    assert client.tags[
        ("safe_run", module.ARTIFACT_SHA256_TAGS["pred.pkl"])
    ] == hashes["pred.pkl"]
    assert client.tags[
        (
            "safe_run",
            module.ARTIFACT_SHA256_TAGS[
                "portfolio_analysis/report_normal_1day.pkl"
            ],
        )
    ] == hashes["portfolio_analysis/report_normal_1day.pkl"]


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
