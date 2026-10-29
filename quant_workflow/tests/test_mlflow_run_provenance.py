from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def frozen_input(tmp_path):
    from quant_qmt_data import save_dataset

    dates = pd.date_range("2024-01-02", periods=3)
    frame = pd.DataFrame({"date": dates, "symbol": "SH600519", "open": 10.0,
                          "high": 11.0, "low": 9.0, "close": 10.0,
                          "volume": 1000.0, "amount": 10000.0,
                          "preclose": 10.0, "suspend_flag": 0})
    frame.attrs = dict(provider="qmt", request={"symbols": ["SH600519"]},
        price_basis="front", volume_units={"SH600519": "shares"}, missing_symbols=[],
        read_at="2026-10-28T22:00:00+08:00", completed_through="2024-01-04",
        contains_partial=False, qmt_data_dir=str(tmp_path / "qmt-cache"),
        sdk_version="synthetic-test", actual_start="2024-01-02", actual_end="2024-01-04")
    return save_dataset(frame, tmp_path / "input", calendar=dates, universe=["SH600519"])


@pytest.fixture(params=["immutable_snapshot_v2", "direct_parquet_v1"])
def historical_identity(request):
    identity = dict(provider_contract=request.param, dataset_id="qlib_ashare_v2",
        snapshot_id="snapshot-001", content_sha256="a" * 64, manifest_sha256="b" * 64)
    if request.param == "direct_parquet_v1":
        identity.update(dataset_id="ashare_daily_qfq",
            descriptor_path="runtime/qlib_direct/provider/PROVIDER_DESCRIPTOR.json",
            descriptor_sha256="c" * 64, cleaning_mode="canonical",
            index_dataset_id="cn_index_daily", index_snapshot_id="index-001",
            index_content_sha256="d" * 64, index_manifest_sha256="e" * 64)
    return identity


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



def test_tag_run_artifact_hashes_records_every_supported_local_artifact(
    tmp_path: Path,
) -> None:
    module = load_module()
    mlruns = tmp_path / "mlruns"
    run_dir = mlruns / "123" / "safe_run"
    pred = run_dir / "artifacts" / "pred.pkl"
    task = run_dir / "artifacts" / "task"
    report = (
        run_dir
        / "artifacts"
        / "portfolio_analysis"
        / "report_normal_1day.pkl"
    )
    pred.parent.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    pred.write_bytes(b"prediction")
    task.write_bytes(b"task")
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
        "task": hashlib.sha256(task.read_bytes()).hexdigest(),
    }
    assert client.tags[
        ("safe_run", module.ARTIFACT_SHA256_TAGS["pred.pkl"])
    ] == hashes["pred.pkl"]
    assert client.tags[
        ("safe_run", module.ARTIFACT_SHA256_TAGS["task"])
    ] == hashes["task"]
    assert client.tags[
        (
            "safe_run",
            module.ARTIFACT_SHA256_TAGS[
                "portfolio_analysis/report_normal_1day.pkl"
            ],
        )
    ] == hashes["portfolio_analysis/report_normal_1day.pkl"]



def test_new_run_selection_fails_closed_when_concurrent_runs_are_ambiguous() -> None:
    module = load_module()
    client = FakeClient([fake_run("new-a"), fake_run("new-b")])

    with pytest.raises(module.RunProvenanceError, match="exactly one"):
        module.select_new_prediction_run(client, "1", {"old"})



def test_existing_run_must_match_the_public_frozen_identity():
    module = load_module()
    identity = dict(backend="qmt_native", experiment_id="experiment-1", bars_sha256="a"*64, inputs_sha256="c"*64,
                    price_basis="front", input_dir="private/input")
    wrong = fake_run("wrong", tags=module.provider_tags({**identity,"bars_sha256":"b"*64}))
    matching = fake_run("matching", tags=module.provider_tags(identity))
    selected = module.select_matching_prediction_run(FakeClient([wrong,matching]), "1", identity)
    assert selected.info.run_id == "matching"
    assert "input_dir" not in module.normalize_provider_lineage(identity)
    assert not module.provider_lineage_is_complete({"provider_contract":"immutable_snapshot_v2"})


def test_file_store_artifact_is_bound_to_run_and_frozen_input(tmp_path, frozen_input):
    module = load_module()
    directory = frozen_input
    identity = module.provider_identity(directory)
    store = tmp_path / "store"
    run_dir = store / "1" / "native-run"
    artifact = run_dir / "artifacts" / "pred.pkl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"prediction artifact")
    tags = run_dir / "tags"
    tags.mkdir()
    for key, value in module.provider_tags(identity).items():
        (tags/key).write_text(value,encoding="utf-8")
    (tags/module.ARTIFACT_SHA256_TAGS["pred.pkl"]).write_text(hashlib.sha256(artifact.read_bytes()).hexdigest())
    result = module.validate_file_store_run_artifact(artifact_path=artifact,
        artifact_relative_path=Path("pred.pkl"),run_id="native-run",mlruns_dir=store,
        quant_data_root=tmp_path)
    assert result["provider_identity"] == identity
    artifact.write_bytes(b"changed prediction")
    with pytest.raises(module.RunProvenanceError, match="hash"):
        module.validate_file_store_run_artifact(artifact_path=artifact,
            artifact_relative_path=Path("pred.pkl"),run_id="native-run",mlruns_dir=store,
            quant_data_root=tmp_path)


def test_historical_lineages_and_tags_coexist_with_native(tmp_path, historical_identity):
    module = load_module()
    identity = {**historical_identity, "provider_uri": str(tmp_path / "old-machine"),
                "input_dir": str(tmp_path / "must-not-use-qmt")}
    public = module.normalize_provider_lineage(identity)
    assert public == historical_identity
    assert module.provider_lineage_is_complete(identity)
    assert module.provider_lineages_match(public, identity)
    assert not module.provider_lineages_match({**public, "content_sha256": "f" * 64}, identity)

    tags = {f"dsa.quant_data.{key}": value for key, value in identity.items() if key != "input_dir"}
    run_dir = tmp_path / "run"
    (run_dir / "tags").mkdir(parents=True)
    for key, value in tags.items():
        (run_dir / "tags" / key).write_text(value, encoding="utf-8")
    parsed = module._file_store_provider_identity(run_dir)
    assert parsed["provider_uri"] == identity["provider_uri"]
    assert module.normalize_provider_lineage(parsed) == historical_identity

    native = dict(backend="qmt_native", experiment_id="experiment-1", bars_sha256="a" * 64,
                  inputs_sha256="b" * 64, price_basis="front")
    client = FakeClient([fake_run("native", tags=module.provider_tags(native)),
                         fake_run("historical", tags=tags)])
    assert module.select_matching_prediction_run(client, "1", public).info.run_id == "historical"
    assert module.select_matching_prediction_run(client, "1", native).info.run_id == "native"
    assert not module.provider_lineages_match(public, native)


@pytest.mark.parametrize("field,value", [("snapshot_id", "../escape"), ("content_sha256", "invalid")])
def test_historical_lineage_keeps_existing_validation(historical_identity, field, value):
    module = load_module()
    assert not module.provider_lineage_is_complete({**historical_identity, field: value})


@pytest.mark.parametrize("descriptor_path", ["/private/PROVIDER_DESCRIPTOR.json",
    "C:\\private\\PROVIDER_DESCRIPTOR.json", "../PROVIDER_DESCRIPTOR.json"])
def test_direct_public_lineage_rejects_private_or_escaping_descriptor_paths(descriptor_path):
    module = load_module()
    identity = dict(provider_contract="direct_parquet_v1", dataset_id="ashare_daily_qfq",
        snapshot_id="stock-001", content_sha256="a" * 64, manifest_sha256="b" * 64,
        descriptor_path=descriptor_path, descriptor_sha256="c" * 64, cleaning_mode="canonical",
        index_dataset_id="cn_index_daily", index_snapshot_id="index-001",
        index_content_sha256="d" * 64, index_manifest_sha256="e" * 64)
    assert not module.provider_lineage_is_complete(identity)


def test_historical_files_resolve_without_using_qmt(tmp_path, monkeypatch, historical_identity):
    module = load_module()
    root = tmp_path / "quant_data"
    monkeypatch.setenv("QMT_DATASET_DIR", str(tmp_path / "unrelated-qmt"))
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("Historical Run used QMT"))
    identity = dict(historical_identity)
    if identity["provider_contract"] == "immutable_snapshot_v2":
        provider = root / "qlib/cn_data_ashare_v2/snapshots" / identity["snapshot_id"]
        provider.mkdir(parents=True)
        manifest = dict(dataset_id=identity["dataset_id"], snapshot_id=provider.name,
            content_sha256=identity["content_sha256"], role="canonical", status="ready",
            price_basis="qfq", benchmark_complete=True)
        path = provider / "DATASET_MANIFEST.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        identity["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        # The existing snapshot validator owns full data validation; this test
        # exercises the real manifest read and pinned identity/path selection.
        monkeypatch.setattr(module, "validate_qlib_snapshot", lambda *_, **__: manifest, raising=False)
    else:
        path = root / identity["descriptor_path"]
        provider = path.parent
        provider.mkdir(parents=True)
        payload = dict(schema_version=1, adapter="quant_data_qlib_provider", storage_mode="direct_parquet",
            dataset={key: identity[key] for key in ("dataset_id", "snapshot_id", "content_sha256", "manifest_sha256")},
            index_dataset={key: identity[f"index_{key}"] for key in
                ("dataset_id", "snapshot_id", "content_sha256", "manifest_sha256")},
            cleaning={"mode": identity["cleaning_mode"]})
        path.write_text(json.dumps(payload), encoding="utf-8")
        identity["descriptor_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity["provider_uri"] = str(tmp_path / "old-machine")
    assert module.normalize_provider_lineage(module.provider_identity(provider, quant_data_root=root)) == module.normalize_provider_lineage(identity)
    assert module.resolve_immutable_provider_identity(identity, quant_data_root=root) == provider
    assert module.resolve_approved_provider_identity("canonical", identity, quant_data_root=root) == provider
    assert module.provider_lineages_match(module.revalidate_provider_identity(provider, identity, quant_data_root=root), identity)
    run_dir = tmp_path / "mlruns/1/historical-run"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "tags").mkdir()
    artifact = run_dir / "artifacts/pred.pkl"
    artifact.write_bytes(b"synthetic historical prediction")
    tags = module.provider_tags(identity)
    tags[module.ARTIFACT_SHA256_TAGS["pred.pkl"]] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    for key, value in tags.items():
        (run_dir / "tags" / key).write_text(value, encoding="utf-8")
    result = module.validate_file_store_run_artifact(artifact_path=artifact,
        artifact_relative_path=Path("pred.pkl"), run_id="historical-run",
        mlruns_dir=tmp_path / "mlruns", quant_data_root=root)
    assert result["provider_path"] == str(provider)
    assert module.normalize_provider_lineage(result["provider_identity"]) == module.normalize_provider_lineage(identity)
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    with pytest.raises(module.RunProvenanceError, match="identity|match|hash"):
        module.resolve_immutable_provider_identity(identity, quant_data_root=root)


@pytest.mark.parametrize("digest", [None, "", "invalid"])
def test_native_lineage_requires_valid_inputs_digest(digest):
    module = load_module()
    identity = dict(backend="qmt_native", experiment_id="experiment-1", bars_sha256="a" * 64, price_basis="front")
    if digest is not None:
        identity["inputs_sha256"] = digest
    assert not module.provider_lineage_is_complete(identity)
    with pytest.raises(module.RunProvenanceError):
        module.provider_tags(identity)


def test_native_run_tags_require_inputs_digest(tmp_path):
    module = load_module()
    tags_dir = tmp_path / "tags"
    tags_dir.mkdir()
    identity = dict(backend="qmt_native", experiment_id="experiment-1", bars_sha256="a" * 64, price_basis="front")
    for key, value in identity.items():
        (tags_dir / f"dsa.quant_data.{key}").write_text(value, encoding="utf-8")
    with pytest.raises(module.RunProvenanceError, match="inputs_sha256|incomplete"):
        module._file_store_provider_identity(tmp_path)


def test_native_digest_is_part_of_tags_matching_and_approval(tmp_path, frozen_input, monkeypatch):
    from quant_qmt_data import load_dataset

    module = load_module()
    directory = frozen_input
    identity = module.provider_identity(directory)
    assert identity["inputs_sha256"] == load_dataset(directory).attrs["inputs_sha256"]
    assert module.normalize_provider_lineage(identity) == {key: value for key, value in identity.items() if key != "input_dir"}
    assert module.provider_tags(identity)["dsa.quant_data.inputs_sha256"] == identity["inputs_sha256"]
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    public = module.normalize_provider_lineage(identity)
    assert module.resolve_approved_provider_identity("qmt_native", public) == directory
    changed = {**identity, "inputs_sha256": "f" * 64}
    assert not module.provider_lineages_match(changed, identity)
    with pytest.raises(module.RunProvenanceError, match="identity changed"):
        module.resolve_approved_provider_identity("qmt_native", changed)
    with pytest.raises(module.RunProvenanceError, match="identity changed"):
        module.revalidate_provider_identity(directory, changed)
    client = FakeClient([fake_run("changed", tags=module.provider_tags(changed)),
                         fake_run("matching", tags=module.provider_tags(identity))])
    assert module.select_matching_prediction_run(client, "1", identity).info.run_id == "matching"


@pytest.mark.parametrize("field,value", [("calendar", ["2024-01-02", "2024-01-04"]),
    ("universe", ["SH600519", "SZ000001"]), ("volume_units", {"SH600519": "lots"})])
def test_native_metadata_change_is_not_the_approved_input(frozen_input, field, value):
    module = load_module()
    identity = module.provider_identity(frozen_input)
    path = frozen_input / "dataset.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    # Old schema-1 inputs have no stored digest: loading must compute it from
    # their actual metadata and compare it with the Run's approved identity.
    metadata.pop("inputs_sha256", None)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(module.RunProvenanceError, match="identity changed"):
        module.resolve_approved_provider_identity("qmt_native", identity)
    with pytest.raises(module.RunProvenanceError, match="identity changed"):
        module.revalidate_provider_identity(frozen_input, identity)
