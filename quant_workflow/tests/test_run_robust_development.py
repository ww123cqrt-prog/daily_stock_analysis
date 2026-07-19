from __future__ import annotations

import importlib.util
import json
import pickle
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_robust_development.py"
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
MPS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "robust_strategy_mps_dnn.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_robust_development", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def seal_provider(provider: Path, key: str = "provider-v1") -> str:
    from prepare_robust_provider import (
        PROVIDER_CONTENT_FILENAME,
        PROVIDER_CONTENT_VERSION,
        PROVIDER_FORMAT_VERSION,
        PROVIDER_METADATA_FILENAME,
        _content_entries,
        _content_tree_sha256,
        _make_tree_read_only,
        _make_tree_writable,
        file_sha256,
    )

    if provider.exists():
        _make_tree_writable(provider)
        for filename in (PROVIDER_METADATA_FILENAME, PROVIDER_CONTENT_FILENAME):
            provider.joinpath(filename).unlink(missing_ok=True)
    (provider / "features").mkdir(parents=True, exist_ok=True)
    (provider / "instruments").mkdir(parents=True, exist_ok=True)
    (provider / "instruments" / "csi800.txt").write_text("", encoding="utf-8")
    entries = _content_entries(provider)
    tree_hash = _content_tree_sha256(entries)
    content_path = provider / PROVIDER_CONTENT_FILENAME
    content_path.write_text(
        json.dumps(
            {
                "version": PROVIDER_CONTENT_VERSION,
                "provider_key": key,
                "tree_sha256": tree_hash,
                "files": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (provider / PROVIDER_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "format_version": PROVIDER_FORMAT_VERSION,
                "provider_key": key,
                "provider_tree_sha256": tree_hash,
                "content_manifest_sha256": file_sha256(content_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _make_tree_read_only(provider)
    return tree_hash


def write_reference_source(quant_data: Path) -> dict:
    from _robust_membership_integrity import membership_read_lock

    metadata = {
        "requested": "auto",
        "selected": "baostock",
        "fallback_reason": "jqdata_permission_denied",
        "date_range": {
            "start_date": "2023-01-01",
            "end_date": "2025-12-31",
        },
        "industry_classification": "CSRC",
    }
    reference = quant_data / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    pit_dir = reference / "pit_csi800"
    (pit_dir / "daily").mkdir(parents=True, exist_ok=True)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2023-01-01",
                    "end_date": "2025-12-31",
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
            }
        ),
        encoding="utf-8",
    )
    (pit_dir / "membership_intervals.parquet").write_bytes(b"membership intervals")
    (pit_dir / "daily" / "2023.parquet").write_bytes(b"daily membership")
    with membership_read_lock(pit_dir) as snapshot:
        metadata["membership_snapshot_hash"] = snapshot.fingerprint()
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    for file_name in (
        "security_master.parquet",
        "industry_intervals.parquet",
        "st_status.parquet",
        "suspension_status.parquet",
    ):
        (reference / file_name).write_bytes(f"fixture:{file_name}".encode("ascii"))
    return metadata


def reference_context_fields(module, quant_data: Path) -> dict:
    reference = quant_data / "reference"
    with module.reference_read_lock(reference) as snapshot:
        snapshot.ensure_transaction_clean()
        metadata = snapshot.validate_source(
            start_date="2023-01-01", end_date="2025-12-31"
        )
        fingerprint = snapshot.fingerprint()
    return {
        "reference_source_metadata": metadata,
        "reference_snapshot_hash": fingerprint,
    }


def membership_context_fields(module, quant_data: Path) -> dict:
    pit_dir = quant_data / "reference" / "pit_csi800"
    with module.membership_read_lock(pit_dir) as snapshot:
        snapshot.ensure_transaction_clean()
        metadata = snapshot.validate_source(
            start_date="2023-01-01", end_date="2025-12-31"
        )
        fingerprint = snapshot.fingerprint()
    return {
        "membership_source_metadata": metadata,
        "membership_snapshot_hash": fingerprint,
    }


def build_resume_fixture(module, root: Path, *, run_id: str) -> dict:
    provider = root / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2018-01-02\n2023-01-03\n2025-12-31\n2026-07-15\n",
        encoding="utf-8",
    )
    provider_tree_sha256 = seal_provider(provider)
    provider_state = root / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = root / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    write_reference_source(quant_data)
    report_dir = root / "reports"
    context_path = (
        report_dir / "runs" / run_id / "development" / "development_context.json"
    )
    context_path.parent.mkdir(parents=True)
    context = {
        "config_hash": module.file_sha256(CONFIG_PATH),
        "git_commit": "a" * 40,
        "data_manifest_hash": module.file_sha256(manifest),
        "provider_uri": str(provider.resolve()),
        "provider_key": "provider-v1",
        "as_of_date": "2026-07-15",
        **reference_context_fields(module, quant_data),
        **membership_context_fields(module, quant_data),
    }
    runtime_environment = module.collect_runtime_environment(
        module.load_config(CONFIG_PATH)
    )
    context["runtime_environment"] = runtime_environment
    context["runtime_environment_hash"] = runtime_environment["environment_hash"]
    context["provider_tree_sha256"] = provider_tree_sha256
    context["context_hash"] = module.payload_sha256(context)
    context_path.write_text(json.dumps(context), encoding="utf-8")
    return {
        "provider": provider,
        "provider_state": provider_state,
        "quant_data": quant_data,
        "report_dir": report_dir,
        "context_path": context_path,
        "context": context,
    }


def test_membership_source_metadata_keeps_only_public_provenance(
    tmp_path: Path,
) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    pit_dir.mkdir()
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2014-01-02",
                    "end_date": "2026-07-16",
                },
                "indices": {
                    "csi300": "000300.XSHG",
                    "csi500": "000905.XSHG",
                },
                "private_path": "/Users/private/data",
            }
        ),
        encoding="utf-8",
    )

    result = module._read_membership_source_metadata(pit_dir)

    assert result["selected"] == "baostock"
    assert result["fallback_reason"] == "jqdata_permission_denied"
    assert "private_path" not in result


def completed_artifact_state(module, root: Path, job) -> dict:
    prediction = root / f"{job.key}.parquet"
    model = root / f"{job.key}.pkl"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2023-01-03")],
            "instrument": ["SH600000"],
            "score": [1.0],
            "label": [0.01],
        }
    ).to_parquet(prediction, index=False)
    model.write_bytes(pickle.dumps({"model_id": job.model_id}))
    return {
        "status": "completed",
        "model_id": job.model_id,
        "window_days": job.window_days,
        "quarter": job.quarter,
        "seed": job.seed,
        "model_path": str(model.resolve()),
        "model_sha256": module.file_sha256(model),
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": module.file_sha256(prediction),
        "rows": 1,
        "split": {
            "train_start": "2020-01-02",
            "train_end": "2021-12-31",
            "validation_start": "2022-01-03",
            "validation_end": "2022-06-24",
            "test_start": "2023-01-03",
            "test_end": "2023-03-31",
        },
    }


def test_training_matrix_contains_396_development_jobs_and_no_2026_period() -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)

    jobs = module.build_training_jobs(config)

    assert len(jobs) == 396
    assert len({job.key for job in jobs}) == 396
    assert {job.window_days for job in jobs} == {504, 756, 1008}
    assert {job.quarter for job in jobs} == set(
        config["rolling_validation"]["test_quarters"]
    )
    assert all(not job.quarter.startswith("2026") for job in jobs)
    linear = [job for job in jobs if job.model_id in {"linear_ols", "linear_ridge"}]
    trees = [job for job in jobs if job.model_id not in {"linear_ols", "linear_ridge"}]
    assert len(linear) == 72
    assert len(trees) == 324
    assert {job.seed for job in linear} == {None}
    assert {job.seed for job in trees} == {42, 2026, 3407}


def test_mps_dnn_config_adds_three_seed_matrix_without_changing_classical_jobs() -> (
    None
):
    module = load_module()
    classical_config = module.load_config(CONFIG_PATH)
    mps_config = module.load_config(MPS_CONFIG_PATH)

    classical_jobs = module.build_training_jobs(classical_config)
    jobs = module.build_training_jobs(mps_config)
    dnn_jobs = [job for job in jobs if job.model_id == "mps_dnn"]

    assert len(classical_jobs) == 396
    assert len(jobs) == 504
    assert len(dnn_jobs) == 108
    assert {job.window_days for job in dnn_jobs} == {504, 756, 1008}
    assert {job.seed for job in dnn_jobs} == {42, 2026, 3407}
    assert {job.quarter for job in dnn_jobs} == set(
        mps_config["rolling_validation"]["test_quarters"]
    )
    assert all(not job.quarter.startswith("2026") for job in dnn_jobs)


def test_model_config_applies_tree_seed_without_mutating_linear_models() -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    xgb_job = next(
        job
        for job in module.build_training_jobs(config)
        if job.model_id == "xgboost" and job.seed == 2026
    )
    linear_job = next(
        job
        for job in module.build_training_jobs(config)
        if job.model_id == "linear_ridge"
    )

    xgb = module.model_config_for_job(config, xgb_job)
    linear = module.model_config_for_job(config, linear_job)

    assert xgb["kwargs"]["seed"] == 2026
    assert "random_seed" not in xgb["kwargs"]
    assert linear["kwargs"]["estimator"] == "ridge"
    assert all(
        name not in linear["kwargs"] for name in ["seed", "random_seed", "random_state"]
    )

    lgb_job = next(
        job
        for job in module.build_training_jobs(config)
        if job.model_id == "lightgbm" and job.seed == 3407
    )
    cat_job = next(
        job
        for job in module.build_training_jobs(config)
        if job.model_id == "catboost" and job.seed == 42
    )
    lgb = module.model_config_for_job(config, lgb_job)
    cat = module.model_config_for_job(config, cat_job)
    assert lgb["kwargs"]["seed"] == 3407
    assert lgb["kwargs"]["bagging_seed"] == 3407
    assert lgb["kwargs"]["feature_fraction_seed"] == 3407
    assert cat["kwargs"]["random_seed"] == 42


def test_model_config_applies_neural_seed_and_keeps_mps_device() -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)
    job = next(
        item
        for item in module.build_training_jobs(config)
        if item.model_id == "mps_dnn" and item.seed == 2026
    )

    model = module.model_config_for_job(config, job)

    assert model["kwargs"]["seed"] == 2026
    assert model["kwargs"]["device"] == "mps"


def test_runtime_environment_metadata_is_hashed_and_contains_no_paths() -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)

    metadata = module.collect_runtime_environment(config)

    assert metadata["python_version"]
    assert metadata["platform"]
    assert metadata["machine"]
    assert metadata["requested_device"] == "mps"
    assert set(metadata["packages"]) == {
        "catboost",
        "cloudpickle",
        "lightgbm",
        "mlflow",
        "numpy",
        "pandas",
        "pyarrow",
        "pyqlib",
        "pyyaml",
        "scikit-learn",
        "torch",
        "xgboost",
    }
    assert metadata["environment_hash"] == module.payload_sha256(
        {key: value for key, value in metadata.items() if key != "environment_hash"}
    )
    serialized = json.dumps(metadata, sort_keys=True)
    assert "/Users/" not in serialized
    assert "site-packages" not in serialized


def test_build_dataset_config_uses_pit_csi800_five_day_open_label_and_purged_split() -> (
    None
):
    module = load_module()
    calendar = pd.bdate_range("2019-01-01", "2025-12-31")
    split = module.make_job_split(
        calendar, "2023Q1", window_days=504, validation_days=126, label_horizon_days=5
    )

    dataset = module.build_dataset_config(split)

    handler = dataset["kwargs"]["handler"]
    assert handler["kwargs"]["instruments"] == "csi800"
    assert handler["kwargs"]["label"] == [
        ["Ref($open, -6) / Ref($open, -1) - 1"],
        ["LABEL0"],
    ]
    assert dataset["class"] == "BoardAwareDataset"
    assert dataset["module_path"] == "robust_dataset"
    assert dataset["kwargs"]["add_market_state_features"] is True
    assert dataset["kwargs"]["segments"]["test"] == [
        str(split.test_start.date()),
        str(split.test_end.date()),
    ]


def test_percentile_ensemble_equal_weights_models_per_date() -> None:
    module = load_module()
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    first = pd.Series([1, 2, 3, 3, 2, 1], index=index)
    second = pd.Series([3, 2, 1, 1, 2, 3], index=index)

    ensemble = module.percentile_rank_ensemble({"first": first, "second": second})

    assert ensemble.index.equals(index)
    assert np.allclose(ensemble.values, [2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3])


def test_resume_selects_only_jobs_without_valid_completed_artifacts(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    jobs = module.build_training_jobs(config)[:3]
    completed = completed_artifact_state(module, tmp_path / "completed", jobs[0])
    corrupt = completed_artifact_state(module, tmp_path / "corrupt", jobs[1])
    Path(corrupt["model_path"]).write_bytes(b"corrupt model")
    progress = {
        jobs[0].key: completed,
        jobs[1].key: corrupt,
        jobs[2].key: {"status": "failed"},
    }

    pending = module.pending_jobs(jobs, progress)

    assert [job.key for job in pending] == [jobs[1].key, jobs[2].key]


def test_missing_provider_state_writes_development_failed_without_starting_training(
    tmp_path: Path,
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"

    return_code = module.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--quant-data-dir",
            str(tmp_path / "quant_data"),
            "--report-dir",
            str(report_dir),
            "--provider-state",
            str(tmp_path / "missing-provider.json"),
            "--as-of-date",
            "2026-07-15",
        ]
    )

    payload = json.loads(
        (report_dir / "development_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 2
    assert payload["status"] == "development_failed"
    assert payload["reason_code"] == "provider_state_missing"
    assert payload["training_jobs_started"] == 0


def test_plan_only_writes_matrix_without_claiming_a_research_candidate(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    provider.mkdir()
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    return_code = module.main(
        [
            "--config",
            str(CONFIG_PATH),
            "--quant-data-dir",
            str(tmp_path / "quant_data"),
            "--report-dir",
            str(report_dir),
            "--provider-state",
            str(provider_state),
            "--as-of-date",
            "2026-07-15",
            "--plan-only",
        ]
    )

    payload = json.loads(
        (report_dir / "development_plan_latest.json").read_text(encoding="utf-8")
    )
    assert return_code == 0
    assert payload["lifecycle_status"] == "planned"
    assert payload["official_status"] is None
    assert payload["training_job_count"] == 396
    assert not (report_dir / "development_latest.json").exists()


def test_fit_predict_job_aligns_scores_labels_and_writes_reusable_artifact(
    tmp_path: Path,
) -> None:
    module = load_module()
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            assert segment == "test"
            assert col_set == "label"
            return pd.DataFrame(
                {"LABEL0": [0.03, 0.02, 0.01, -0.01, 0.0, 0.01]}, index=index
            )

    class FakeModel:
        fitted = False

        def fit(self, dataset):
            self.fitted = True

        def predict(self, dataset, segment="test"):
            assert self.fitted
            return pd.Series([3, 2, 1, 1, 2, 3], index=index)

    artifact = tmp_path / "prediction.parquet"
    result = module.fit_predict_job(FakeModel(), FakeDataset(), artifact)

    saved = pd.read_parquet(artifact)
    assert result["rows"] == 6
    assert result["rank_ic_mean"] == 1.0
    assert list(saved.columns) == ["datetime", "instrument", "score", "label"]
    assert len(saved) == 6


def test_fit_predict_job_rejects_prediction_label_index_mismatch(
    tmp_path: Path,
) -> None:
    module = load_module()
    pred_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "A")], names=["datetime", "instrument"]
    )
    label_index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "B")], names=["datetime", "instrument"]
    )

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            return pd.DataFrame({"LABEL0": [0.1]}, index=label_index)

    class FakeModel:
        def fit(self, dataset):
            return None

        def predict(self, dataset, segment="test"):
            return pd.Series([1.0], index=pred_index)

    try:
        module.fit_predict_job(
            FakeModel(), FakeDataset(), tmp_path / "prediction.parquet"
        )
    except ValueError as exc:
        assert "overlap" in str(exc) or "index" in str(exc)
    else:
        raise AssertionError("mismatched prediction and label indexes were accepted")


def test_fit_predict_job_preserves_scores_without_future_labels(
    tmp_path: Path,
) -> None:
    module = load_module()
    prediction_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-03-31"), "SH600000"),
            (pd.Timestamp("2025-03-31"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )
    label_index = prediction_index[:1]

    class Model:
        def fit(self, dataset):
            return None

        def predict(self, dataset, segment):
            assert segment == "test"
            return pd.Series([2.0, 1.0], index=prediction_index)

    class Dataset:
        def prepare(self, segment, col_set, data_key):
            assert (segment, col_set, data_key) == ("test", "label", "learn")
            return pd.DataFrame({"LABEL0": [0.1]}, index=label_index)

    artifact = tmp_path / "prediction.parquet"
    result = module.fit_predict_job(Model(), Dataset(), artifact)
    saved = pd.read_parquet(artifact)

    assert result["rows"] == 2
    assert result["evaluable_rows"] == 1
    assert saved["instrument"].tolist() == ["SH600000", "SZ000001"]
    assert saved.loc[saved["instrument"] == "SZ000001", "label"].isna().all()


def test_mlflow_run_yields_active_recorder(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    job = module.TrainingJob("2023Q1", 504, "linear_ols", None)
    experiment = object()
    recorder = object()
    calls: list[dict[str, object]] = []

    class FakeR:
        @staticmethod
        def start(**kwargs):
            calls.append(kwargs)
            return nullcontext(experiment)

        @staticmethod
        def get_recorder():
            return recorder

    import qlib.workflow

    monkeypatch.setattr(qlib.workflow, "R", FakeR)
    monkeypatch.setattr(module, "mlruns_root", lambda: tmp_path / "mlruns")

    with module._mlflow_run(
        job,
        tmp_path / "provider",
        enabled=True,
        phase="development",
        experiment_name="robust_csi800_development",
    ) as active_run:
        assert active_run is recorder

    assert len(calls) == 1
    call = calls[0]
    assert call["experiment_name"] == "robust_csi800_development"
    assert str(call["recorder_name"]).startswith("2023Q1__w504__linear_ols__sfixed-")
    assert call["uri"] == f"file://{tmp_path / 'mlruns'}"


def test_mlflow_result_records_runtime_environment_hash_and_versions(
    tmp_path: Path,
) -> None:
    module = load_module()
    job = module.TrainingJob("2023Q1", 504, "mps_dnn", 2026)
    split = module.make_job_split(
        pd.bdate_range("2019-01-01", "2025-12-31"),
        "2023Q1",
        window_days=504,
        validation_days=126,
        label_horizon_days=5,
    )
    prediction_path = tmp_path / "prediction.parquet"
    model_path = tmp_path / "model.pkl"
    prediction_path.write_bytes(b"prediction")
    model_path.write_bytes(b"model")
    runtime = {
        "python_version": "3.11.15",
        "platform": "Darwin",
        "machine": "arm64",
        "requested_device": "mps",
        "mps_built": True,
        "mps_available": True,
        "packages": {"pyqlib": "0.9.7", "torch": "2.5.1"},
    }
    runtime["environment_hash"] = module.payload_sha256(runtime)

    class Recorder:
        id = "mlflow-run-id"

        def __init__(self):
            self.params = {}
            self.tags = {}
            self.metrics = {}
            self.artifacts = []

        def log_params(self, **kwargs):
            self.params.update(kwargs)

        def set_tags(self, **kwargs):
            self.tags.update(kwargs)

        def log_metrics(self, **kwargs):
            self.metrics.update(kwargs)

        def log_artifact(self, path, artifact_path=None):
            self.artifacts.append((path, artifact_path))

    recorder = Recorder()
    result = module._log_mlflow_result(
        recorder,
        job=job,
        model_config={"kwargs": {"device": "mps", "seed": 2026}},
        split=split,
        result={
            "rows": 3,
            "dates": 1,
            "rank_ic_mean": 0.1,
            "neural_repeatability_max_abs_diff": 0.0,
            "neural_repeatability_score_count": 3,
            "neural_repeatability_primary_score_sha256": "a" * 64,
            "neural_repeatability_repeat_score_sha256": "a" * 64,
        },
        prediction_path=prediction_path,
        model_path=model_path,
        phase="development",
        runtime_environment=runtime,
    )

    assert result == "mlflow-run-id"
    assert recorder.params["runtime.environment_hash"] == runtime["environment_hash"]
    assert recorder.params["runtime.package.pyqlib"] == "0.9.7"
    assert recorder.params["runtime.package.torch"] == "2.5.1"
    assert recorder.params["runtime.requested_device"] == "mps"
    assert recorder.params["neural_repeatability_primary_score_sha256"] == "a" * 64
    assert recorder.params["neural_repeatability_repeat_score_sha256"] == "a" * 64
    assert recorder.metrics["development.neural_repeatability_max_abs_diff"] == 0.0
    assert recorder.metrics["development.neural_repeatability_score_count"] == 3.0
    assert (
        recorder.tags["robust_strategy.runtime_environment_hash"]
        == runtime["environment_hash"]
    )


def test_run_training_jobs_reuses_dataset_per_quarter_window_and_checkpoints(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    jobs = [
        job
        for job in module.build_training_jobs(config)
        if job.quarter == "2023Q1" and job.window_days == 504
    ][:3]
    calendar = pd.bdate_range("2019-01-01", "2025-12-31")
    dataset_calls: list[tuple[str, int]] = []
    model_calls: list[str] = []
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-01-03", "2023-01-04"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            return pd.DataFrame(
                {"LABEL0": [0.03, 0.02, 0.01, -0.01, 0.0, 0.01]}, index=index
            )

    class FakeModel:
        def fit(self, dataset):
            return None

        def predict(self, dataset, segment="test"):
            return pd.Series([3, 2, 1, 1, 2, 3], index=index)

    def dataset_factory(provider_uri, dataset_config, split, group_jobs):
        dataset_calls.append((group_jobs[0].quarter, group_jobs[0].window_days))
        return FakeDataset()

    def model_factory(model_config, job):
        model_calls.append(job.key)
        return FakeModel()

    progress = module.run_training_jobs(
        config,
        jobs,
        provider_uri=tmp_path / "provider",
        calendar=calendar,
        run_dir=tmp_path / "run",
        dataset_factory=dataset_factory,
        model_factory=model_factory,
        mlflow_enabled=False,
    )

    assert dataset_calls == [("2023Q1", 504)]
    assert model_calls == [job.key for job in jobs]
    assert all(progress[job.key]["status"] == "completed" for job in jobs)
    assert all(Path(progress[job.key]["prediction_path"]).is_file() for job in jobs)
    for job in jobs:
        state = progress[job.key]
        split = module.make_job_split(
            calendar,
            job.quarter,
            window_days=job.window_days,
            validation_days=int(config["rolling_validation"]["validation_days"]),
            label_horizon_days=5,
        )
        assert state["model_id"] == job.model_id
        assert state["window_days"] == job.window_days
        assert state["quarter"] == job.quarter
        assert state["seed"] == job.seed
        assert state["split"] == {
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "validation_start": str(split.validation_start.date()),
            "validation_end": str(split.validation_end.date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
        }
        assert Path(state["model_path"]).is_file()
        assert state["model_sha256"] == module.file_sha256(Path(state["model_path"]))
        assert state["prediction_sha256"] == module.file_sha256(
            Path(state["prediction_path"])
        )
        assert state["rows"] == 6
    saved = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert set(saved["jobs"]) == {job.key for job in jobs}


def test_neural_training_repeats_each_job_and_records_prediction_consistency(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)
    job = next(
        item
        for item in module.build_training_jobs(config)
        if item.model_id == "mps_dnn"
    )
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-01-03", "2023-01-04"]), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    model_calls: list[str] = []

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            return pd.DataFrame({"LABEL0": [0.03, 0.02, -0.01, 0.01]}, index=index)

    class FakeModel:
        def fit(self, dataset):
            return None

        def predict(self, dataset, segment="test"):
            return pd.Series([0.4, 0.3, 0.2, 0.1], index=index)

    def model_factory(model_config, current_job):
        model_calls.append(current_job.key)
        return FakeModel()

    progress = module.run_training_jobs(
        config,
        [job],
        provider_uri=tmp_path / "provider",
        calendar=pd.bdate_range("2019-01-01", "2025-12-31"),
        run_dir=tmp_path / "run",
        dataset_factory=lambda *args: FakeDataset(),
        model_factory=model_factory,
        mlflow_enabled=False,
    )

    state = progress[job.key]
    assert model_calls == [job.key, job.key]
    assert state["status"] == "completed"
    assert state["neural_repeatability_passed"] is True
    assert state["neural_repeatability_max_abs_diff"] == pytest.approx(0.0)
    assert state["neural_repeatability_score_count"] == 4
    assert state["neural_repeatability_primary_score_sha256"]
    assert (
        state["neural_repeatability_primary_score_sha256"]
        == state["neural_repeatability_repeat_score_sha256"]
    )
    assert not (tmp_path / "run" / "repeatability" / f"{job.key}.parquet").exists()


def test_neural_training_fails_closed_when_repeat_prediction_changes(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)
    job = next(
        item
        for item in module.build_training_jobs(config)
        if item.model_id == "mps_dnn"
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-03"), "SH600000")],
        names=["datetime", "instrument"],
    )
    model_count = 0

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            return pd.DataFrame({"LABEL0": [0.01]}, index=index)

    class FakeModel:
        def __init__(self, score):
            self.score = score

        def fit(self, dataset):
            return None

        def predict(self, dataset, segment="test"):
            return pd.Series([self.score], index=index)

    def model_factory(model_config, current_job):
        nonlocal model_count
        model_count += 1
        return FakeModel(1.0 if model_count == 1 else 1.01)

    progress = module.run_training_jobs(
        config,
        [job],
        provider_uri=tmp_path / "provider",
        calendar=pd.bdate_range("2019-01-01", "2025-12-31"),
        run_dir=tmp_path / "run",
        dataset_factory=lambda *args: FakeDataset(),
        model_factory=model_factory,
        mlflow_enabled=False,
    )

    assert progress[job.key]["status"] == "failed"
    assert progress[job.key]["reason_code"] == "neural_repeatability_failed"
    assert progress[job.key]["error_type"] == "NeuralRepeatabilityError"


def test_run_training_jobs_resume_skips_completed_artifact(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    run_dir = tmp_path / "run"
    state = completed_artifact_state(module, tmp_path / "staging", job)
    model_path = run_dir / "models" / f"{job.key}.pkl"
    prediction_path = run_dir / "predictions" / f"{job.key}.parquet"
    model_path.parent.mkdir(parents=True)
    prediction_path.parent.mkdir(parents=True)
    Path(state["model_path"]).replace(model_path)
    Path(state["prediction_path"]).replace(prediction_path)
    state["model_path"] = str(model_path.resolve())
    state["prediction_path"] = str(prediction_path.resolve())
    state["model_sha256"] = module.file_sha256(model_path)
    state["prediction_sha256"] = module.file_sha256(prediction_path)
    calendar = pd.bdate_range("2019-01-01", "2025-12-31")
    split = module.make_job_split(
        calendar,
        job.quarter,
        window_days=job.window_days,
        validation_days=int(config["rolling_validation"]["validation_days"]),
        label_horizon_days=5,
    )
    state["split"] = module._split_record(split)
    state["context_hash"] = "a" * 64
    (tmp_path / "run" / "progress.json").write_text(
        json.dumps({"context_hash": "a" * 64, "jobs": {job.key: state}}),
        encoding="utf-8",
    )

    progress = module.run_training_jobs(
        config,
        [job],
        provider_uri=tmp_path / "provider",
        calendar=calendar,
        run_dir=run_dir,
        dataset_factory=lambda *args: (_ for _ in ()).throw(
            AssertionError("dataset must not rebuild")
        ),
        model_factory=lambda *args: (_ for _ in ()).throw(
            AssertionError("model must not rebuild")
        ),
        mlflow_enabled=False,
        resume=True,
        context_hash="a" * 64,
    )

    assert progress[job.key]["status"] == "completed"


def test_run_training_jobs_resume_retrains_completed_job_with_corrupt_model(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    state = completed_artifact_state(module, tmp_path / "run" / "artifacts", job)
    Path(state["model_path"]).write_bytes(b"corrupt model")
    (tmp_path / "run" / "progress.json").write_text(
        json.dumps({"jobs": {job.key: state}}),
        encoding="utf-8",
    )
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-01-03"), "SH600000")], names=["datetime", "instrument"]
    )
    model_calls: list[str] = []

    class FakeDataset:
        def prepare(self, segment, col_set, data_key=None):
            return pd.DataFrame({"LABEL0": [0.01]}, index=index)

    class FakeModel:
        def fit(self, dataset):
            return None

        def predict(self, dataset, segment="test"):
            return pd.Series([1.0], index=index)

    progress = module.run_training_jobs(
        config,
        [job],
        provider_uri=tmp_path / "provider",
        calendar=pd.bdate_range("2019-01-01", "2025-12-31"),
        run_dir=tmp_path / "run",
        dataset_factory=lambda *args: FakeDataset(),
        model_factory=lambda model_config, current_job: (
            model_calls.append(current_job.key) or FakeModel()
        ),
        mlflow_enabled=False,
        resume=True,
    )

    assert model_calls == [job.key]
    assert progress[job.key]["status"] == "completed"
    assert progress[job.key]["model_sha256"] == module.file_sha256(
        Path(progress[job.key]["model_path"])
    )


def test_pending_jobs_rejects_corrupt_or_unverified_prediction_artifact(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    jobs = module.build_training_jobs(config)[:2]
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not parquet")
    unverifiable = tmp_path / "unverified.parquet"
    pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2023-01-03")],
            "instrument": ["SH600000"],
            "score": [1.0],
            "label": [0.01],
        }
    ).to_parquet(unverifiable, index=False)
    first = completed_artifact_state(module, tmp_path / "first", jobs[0])
    first.update(
        prediction_path=str(corrupt.resolve()),
        prediction_sha256=module.file_sha256(corrupt),
    )
    second = completed_artifact_state(module, tmp_path / "second", jobs[1])
    second.update(prediction_path=str(unverifiable.resolve()))
    second.pop("prediction_sha256")
    progress = {jobs[0].key: first, jobs[1].key: second}

    assert module.pending_jobs(jobs, progress) == jobs


def test_completed_artifact_requires_expected_run_paths_and_context(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    state = completed_artifact_state(module, tmp_path / "outside", job)
    state["context_hash"] = "a" * 64

    assert (
        module.completed_job_artifact_is_valid(
            job,
            state,
            run_dir=tmp_path / "run",
            expected_split=state["split"],
            context_hash="a" * 64,
        )
        is False
    )


def test_completed_artifact_requires_exact_split_and_prediction_dates(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    run_dir = tmp_path / "run"
    state = completed_artifact_state(module, tmp_path / "staging", job)
    expected_model = run_dir / "models" / f"{job.key}.pkl"
    expected_prediction = run_dir / "predictions" / f"{job.key}.parquet"
    expected_model.parent.mkdir(parents=True)
    expected_prediction.parent.mkdir(parents=True)
    Path(state["model_path"]).replace(expected_model)
    Path(state["prediction_path"]).replace(expected_prediction)
    state["model_path"] = str(expected_model.resolve())
    state["prediction_path"] = str(expected_prediction.resolve())
    state["context_hash"] = "a" * 64
    state["model_sha256"] = module.file_sha256(expected_model)
    state["prediction_sha256"] = module.file_sha256(expected_prediction)
    expected_split = dict(state["split"])

    wrong_split = dict(expected_split)
    wrong_split["test_end"] = "2023-06-30"
    assert (
        module.completed_job_artifact_is_valid(
            job,
            state,
            run_dir=run_dir,
            expected_split=wrong_split,
            context_hash="a" * 64,
        )
        is False
    )

    frame = pd.read_parquet(expected_prediction)
    frame["datetime"] = pd.Timestamp("2024-01-03")
    frame.to_parquet(expected_prediction, index=False)
    state["prediction_sha256"] = module.file_sha256(expected_prediction)
    assert (
        module.completed_job_artifact_is_valid(
            job,
            state,
            run_dir=run_dir,
            expected_split=expected_split,
            context_hash="a" * 64,
        )
        is False
    )


def test_pending_jobs_treats_malformed_recorded_rows_as_invalid(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    state = completed_artifact_state(module, tmp_path / "artifacts", job)
    state["rows"] = "not-an-integer"

    assert module.pending_jobs([job], {job.key: state}) == [job]


def test_completed_neural_artifact_requires_repeatability_evidence(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)
    job = next(
        item
        for item in module.build_training_jobs(config)
        if item.model_id == "mps_dnn"
    )
    state = completed_artifact_state(module, tmp_path / "artifacts", job)

    assert module.completed_job_artifact_is_valid(job, state) is False

    state.update(
        neural_repeatability_passed=True,
        neural_repeatability_max_abs_diff=0.0,
        neural_repeatability_score_count=1,
        neural_repeatability_primary_score_sha256="a" * 64,
        neural_repeatability_repeat_score_sha256="a" * 64,
    )
    assert module.completed_job_artifact_is_valid(job, state) is True

    state["neural_repeatability_max_abs_diff"] = 1e-6
    assert module.completed_job_artifact_is_valid(job, state) is False


def test_run_development_writes_frozen_candidate_only_after_all_gates_pass(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in pd.bdate_range("2018-01-01", "2026-07-15"))
        + "\n",
        encoding="utf-8",
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 1}\n', encoding="utf-8")
    reference_source = write_reference_source(quant_data)
    report_dir = tmp_path / "reports"
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    model_prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )
    calendar = pd.bdate_range("2018-01-01", "2026-07-15")

    def training_runner(config, jobs, **kwargs):
        assert len(jobs) == 396
        run_dir = Path(kwargs["run_dir"])
        progress = {}
        for job in jobs:
            split = module.make_job_split(
                calendar,
                job.quarter,
                window_days=job.window_days,
                validation_days=int(config["rolling_validation"]["validation_days"]),
                label_horizon_days=5,
            )
            model_path = run_dir / "models" / f"{job.key}.pkl"
            prediction_path = run_dir / "predictions" / f"{job.key}.parquet"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(pickle.dumps({"model_id": job.model_id}))
            pd.DataFrame(
                {
                    "datetime": [split.test_start] * 3,
                    "instrument": ["A", "B", "C"],
                    "score": [3.0, 2.0, 1.0],
                    "label": [0.03, 0.02, 0.01],
                }
            ).to_parquet(prediction_path, index=False)
            progress[job.key] = {
                "status": "completed",
                "model_id": job.model_id,
                "window_days": job.window_days,
                "quarter": job.quarter,
                "seed": job.seed,
                "model_path": str(model_path.resolve()),
                "model_sha256": module.file_sha256(model_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": module.file_sha256(prediction_path),
                "rows": 3,
                "split": module._split_record(split),
                "context_hash": kwargs["context_hash"],
            }
        return progress

    def aggregator(records, topk, thresholds):
        assert len(records) == 396
        reports = [
            {
                "model_id": "lightgbm",
                "eligible": True,
                "development_score": 1.0,
                "stability": {"rank_correlation_median": 0.8, "topk_overlap_mean": 0.7},
            },
            {
                "model_id": "linear_ridge",
                "eligible": True,
                "development_score": 0.9,
                "stability": {
                    "rank_correlation_median": 0.75,
                    "topk_overlap_mean": 0.65,
                },
            },
        ]
        return {
            "model_reports": reports,
            "model_predictions": {
                "lightgbm": model_prediction,
                "linear_ridge": model_prediction,
            },
            "window_predictions": {},
        }

    def evaluator(**kwargs):
        return {
            "status": "research_candidate_ready",
            "reason_code": None,
            "selected_strategy": {
                "strategy_id": "full_risk_control",
                "acceptance": {"passed": True, "failed_gates": []},
                "baseline": {"metrics": {"annual_excess_return": 0.08}},
                "stress": {"metrics": {"annual_excess_return": 0.01}},
                "delayed": {"metrics": {"annual_excess_return": 0.005}},
                "quarter_metrics": [],
                "regime_metrics": {},
            },
            "strategies": [],
        }

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="ready-test",
        training_runner=training_runner,
        aggregator=aggregator,
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_evaluator=evaluator,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    latest = json.loads(
        (report_dir / "development_latest.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (report_dir / "frozen_strategy_latest.json").read_text(encoding="utf-8")
    )
    context = json.loads(
        (
            report_dir
            / "runs"
            / "ready-test"
            / "development"
            / "development_context.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "research_candidate_ready"
    assert latest["config_hash"]
    assert latest["data_manifest_hash"]
    assert latest["reference_source_metadata"] == reference_source
    assert len(latest["reference_snapshot_hash"]) == 64
    assert latest["membership_source_metadata"]["selected"] == "baostock"
    assert len(latest["membership_snapshot_hash"]) == 64
    assert frozen["selected_models"] == ["lightgbm", "linear_ridge"]
    assert frozen["selected_strategy"] == "full_risk_control"
    assert frozen["reference_source_metadata"] == reference_source
    assert frozen["reference_snapshot_hash"] == latest["reference_snapshot_hash"]
    assert frozen["membership_source_metadata"] == latest["membership_source_metadata"]
    assert frozen["membership_snapshot_hash"] == latest["membership_snapshot_hash"]
    assert frozen["runtime_environment"] == latest["runtime_environment"]
    assert frozen["runtime_environment_hash"] == latest["runtime_environment_hash"]
    artifacts = frozen["development_artifacts"]
    assert len(artifacts) == 144
    assert [item["job_key"] for item in artifacts] == sorted(
        item["job_key"] for item in artifacts
    )
    assert {item["model_id"] for item in artifacts} == {"lightgbm", "linear_ridge"}
    assert frozen["development_artifacts_hash"] == module.payload_sha256(artifacts)
    assert frozen["frozen_context_hash"] == module.frozen_context_hash(frozen)
    assert frozen["development_run_id"] == "ready-test"
    assert frozen["development_context_hash"] == context["context_hash"]
    run_dir = Path(frozen["development_run_dir"])
    assert run_dir == (report_dir / "runs" / "ready-test" / "development").resolve()
    assert all(
        Path(item["model_path"]).is_relative_to(run_dir)
        and Path(item["prediction_path"]).is_relative_to(run_dir)
        and Path(item["model_path"]).is_file()
        and Path(item["prediction_path"]).is_file()
        and item["model_sha256"] == module.file_sha256(Path(item["model_path"]))
        and item["prediction_sha256"]
        == module.file_sha256(Path(item["prediction_path"]))
        and item["context_hash"] == context["context_hash"]
        for item in artifacts
    )
    assert frozen["confirmation_run_started"] is False
    assert context["reference_source_metadata"] == reference_source
    assert context["reference_snapshot_hash"] == latest["reference_snapshot_hash"]
    assert context["membership_source_metadata"] == latest["membership_source_metadata"]
    assert context["membership_snapshot_hash"] == latest["membership_snapshot_hash"]
    assert context["runtime_environment"] == latest["runtime_environment"]
    assert context["runtime_environment_hash"] == latest["runtime_environment_hash"]
    assert (
        context["runtime_environment_hash"]
        == context["runtime_environment"]["environment_hash"]
    )


def test_run_development_rechecks_provider_before_freezing_candidate(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in pd.bdate_range("2018-01-01", "2026-07-15"))
        + "\n",
        encoding="utf-8",
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 1}\n', encoding="utf-8")
    write_reference_source(quant_data)
    report_dir = tmp_path / "reports"
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def training_runner(config, jobs, **kwargs):
        return {
            job.key: {
                "status": "completed",
                "prediction_path": str(tmp_path / f"{job.key}.parquet"),
            }
            for job in jobs
        }

    def aggregator(records, topk, thresholds):
        reports = [
            {"model_id": "lightgbm", "eligible": True, "development_score": 1.0},
            {"model_id": "linear_ridge", "eligible": True, "development_score": 0.9},
        ]
        return {
            "model_reports": reports,
            "model_predictions": {"lightgbm": prediction, "linear_ridge": prediction},
            "window_predictions": {},
        }

    def evaluator(**kwargs):
        provider_state.write_text(
            json.dumps(
                {
                    "status": "blocked_data",
                    "provider_uri": str(tmp_path / "redirected-provider"),
                    "provider_key": "provider-v2",
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "research_candidate_ready",
            "reason_code": None,
            "selected_strategy": {
                "strategy_id": "full_risk_control",
                "acceptance": {"passed": True, "failed_gates": []},
            },
            "strategies": [],
        }

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="provider-recheck",
        training_runner=training_runner,
        aggregator=aggregator,
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_evaluator=evaluator,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "reproducibility_state_changed_during_development"
    assert result["reproducibility_failures"] == [
        "provider_status",
        "provider_uri",
        "provider_key",
    ]
    assert not (report_dir / "frozen_strategy_latest.json").exists()


def test_run_development_refuses_dirty_worktree_before_training(tmp_path: Path) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in pd.bdate_range("2018-01-01", "2026-07-15"))
        + "\n",
        encoding="utf-8",
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="dirty-test",
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start from uncommitted code")
        ),
        git_state_provider=lambda: {
            "commit": "a" * 40,
            "clean": False,
            "changes": ["?? quant_workflow/scripts/new.py"],
        },
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "git_worktree_dirty"
    assert result["training_jobs_started"] == 0
    assert result["git_changes"] == ["?? quant_workflow/scripts/new.py"]
    assert not (report_dir / "frozen_strategy_latest.json").exists()


def test_run_development_resume_rejects_artifacts_from_different_commit(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="resume-test")
    seal_provider(fixture["provider"], key="provider-v2")
    fixture["provider_state"].write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(fixture["provider"]),
                "provider_key": "provider-v2",
            }
        ),
        encoding="utf-8",
    )

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="resume-test",
        resume=True,
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched artifacts must not be resumed")
        ),
        git_state_provider=lambda: {"commit": "b" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "resume_context_mismatch"
    assert result["context_mismatches"] == ["git_commit", "provider_key"]


def test_run_development_resume_accepts_matching_reference_context(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="matching-reference")
    calls: list[bool] = []

    def training_runner(config, jobs, **kwargs):
        calls.append(bool(kwargs["resume"]))
        return {}

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="matching-reference",
        resume=True,
        training_runner=training_runner,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert calls == [True]
    assert result["reason_code"] == "training_jobs_incomplete"


def test_run_development_resume_rejects_reference_snapshot_hash_change(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="changed-hash")
    reference = fixture["quant_data"] / "reference"
    (reference / "security_master.parquet").write_bytes(b"changed snapshot")

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="changed-hash",
        resume=True,
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start after reference hash change")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["reason_code"] == "resume_context_mismatch"
    assert result["context_mismatches"] == ["reference_snapshot_hash"]
    assert result["training_jobs_started"] == 0


def test_run_development_resume_rejects_reference_source_metadata_change(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="changed-source")
    reference = fixture["quant_data"] / "reference"
    source_path = reference / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(requested="auto", selected="jqdata")
    source.pop("fallback_reason")
    source_path.write_text(json.dumps(source), encoding="utf-8")
    stored_context = fixture["context"]
    stored_context["reference_snapshot_hash"] = reference_context_fields(
        module, fixture["quant_data"]
    )["reference_snapshot_hash"]
    fixture["context_path"].write_text(json.dumps(stored_context), encoding="utf-8")

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="changed-source",
        resume=True,
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start after reference source change")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["reason_code"] == "resume_context_mismatch"
    assert result["context_mismatches"] == ["reference_source_metadata"]
    assert result["training_jobs_started"] == 0


def test_run_development_resume_rejects_membership_snapshot_hash_change_before_training(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="changed-membership-hash")
    daily_path = (
        fixture["quant_data"] / "reference" / "pit_csi800" / "daily" / "2023.parquet"
    )
    daily_path.write_bytes(daily_path.read_bytes() + b"changed")

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="changed-membership-hash",
        resume=True,
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start after membership hash change")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["reason_code"] == "reference_membership_snapshot_mismatch"
    assert result["training_jobs_started"] == 0


def test_run_development_rejects_reference_from_different_membership_snapshot(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = build_resume_fixture(module, tmp_path, run_id="fixture-context")
    source_path = fixture["quant_data"] / "reference" / "REFERENCE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["membership_snapshot_hash"] = "f" * 64
    source_path.write_text(json.dumps(source), encoding="utf-8")

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=fixture["quant_data"],
        report_dir=fixture["report_dir"],
        provider_state_path=fixture["provider_state"],
        as_of_date="2026-07-15",
        run_id="wrong-reference-binding",
        training_runner=lambda *args, **kwargs: {},
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "reference_membership_snapshot_mismatch"
    assert result["training_jobs_started"] == 0


def test_development_exception_overwrites_stale_ready_report_with_failure(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in pd.bdate_range("2018-01-01", "2026-07-15"))
        + "\n",
        encoding="utf-8",
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 1}\n', encoding="utf-8")
    write_reference_source(quant_data)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "development_latest.json").write_text(
        json.dumps({"status": "research_candidate_ready", "run_id": "stale"}),
        encoding="utf-8",
    )

    def training_runner(config, jobs, **kwargs):
        return {
            job.key: {
                "status": "completed",
                "prediction_path": str(tmp_path / f"{job.key}.parquet"),
            }
            for job in jobs
        }

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="exception-test",
        training_runner=training_runner,
        aggregator=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    latest = json.loads(
        (report_dir / "development_latest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "development_execution_failed"
    assert result["error_type"] == "RuntimeError"
    assert latest["run_id"] == "exception-test"


def test_run_development_publishes_normal_result_to_run_scoped_path(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    report_dir = tmp_path / "reports"
    payload = {
        "status": "development_failed",
        "reason_code": "insufficient_stable_models_after_bounded_tuning",
        "as_of_date": "2026-07-15",
        "run_id": "normal-failure",
    }
    monkeypatch.setattr(module, "_run_development_impl", lambda **kwargs: payload)

    result = module.run_development(
        report_dir=report_dir,
        run_id="normal-failure",
        as_of_date="2026-07-15",
    )

    run_scoped = json.loads(
        (
            report_dir / "runs" / "normal-failure" / "development" / "development.json"
        ).read_text(encoding="utf-8")
    )
    latest = json.loads(
        (report_dir / "development_latest.json").read_text(encoding="utf-8")
    )
    assert result == payload
    assert run_scoped == payload
    assert latest == payload


def test_run_development_rejects_invalid_reference_source_before_training(
    tmp_path: Path,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2023-01-03\n2025-12-31\n", encoding="utf-8"
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 1}\n', encoding="utf-8")
    write_reference_source(quant_data)
    (quant_data / "reference" / "REFERENCE_SOURCE.json").unlink()
    report_dir = tmp_path / "reports"

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="invalid-reference-source",
        training_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start with invalid reference provenance")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    latest = json.loads(
        (report_dir / "development_latest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "reference_source_missing"
    assert result["training_jobs_started"] == 0
    assert "error" not in result
    assert latest == result
    assert not (report_dir / "frozen_strategy_latest.json").exists()


@pytest.mark.parametrize(
    ("changed_path", "reason_code"),
    [
        (("reference", "security_master.parquet"), "reference_snapshot_changed"),
        (
            ("reference", "pit_csi800", "daily", "2023.parquet"),
            "membership_snapshot_changed",
        ),
    ],
)
def test_run_development_detects_snapshot_change_before_evaluation(
    tmp_path: Path,
    monkeypatch,
    changed_path: tuple[str, ...],
    reason_code: str,
) -> None:
    module = load_module()
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "2023-01-03\n2025-12-31\n", encoding="utf-8"
    )
    seal_provider(provider)
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v1",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 1}\n', encoding="utf-8")
    write_reference_source(quant_data)
    report_dir = tmp_path / "reports"
    jobs = [
        module.TrainingJob("2025Q1", 504, "lightgbm", 42),
        module.TrainingJob("2025Q1", 504, "linear_ridge", None),
    ]
    monkeypatch.setattr(module, "build_training_jobs", lambda config: jobs)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"]), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame({"score": [2.0, 1.0], "label": [0.02, 0.01]}, index=index)

    def training_runner(config, selected_jobs, **kwargs):
        return {
            job.key: {
                "status": "completed",
                "prediction_path": str(tmp_path / f"{job.key}.parquet"),
            }
            for job in selected_jobs
        }

    def aggregator(records, topk, thresholds):
        reports = [
            {"model_id": "lightgbm", "eligible": True, "development_score": 1.0},
            {"model_id": "linear_ridge", "eligible": True, "development_score": 0.9},
        ]
        return {
            "model_reports": reports,
            "model_predictions": {
                "lightgbm": prediction,
                "linear_ridge": prediction,
            },
            "window_predictions": {},
        }

    def changing_market_loader(**kwargs):
        quant_data.joinpath(*changed_path).write_bytes(b"changed-after-training")
        return pd.DataFrame({"date": []})

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
        as_of_date="2026-07-15",
        run_id="reference-change",
        training_runner=training_runner,
        aggregator=aggregator,
        market_loader=changing_market_loader,
        benchmark_loader=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("benchmark loading must not start after snapshot drift")
        ),
        regime_builder=lambda panel: (_ for _ in ()).throw(
            AssertionError("regime evaluation must not start after snapshot drift")
        ),
        strategy_evaluator=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("strategy evaluation must not start after snapshot drift")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == reason_code
    assert result["training_jobs_started"] == 2
    assert not (report_dir / "frozen_strategy_latest.json").exists()


def test_bounded_tuning_trials_are_deterministic_capped_and_within_ranges() -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)

    for model_id in ["linear_ridge", "lightgbm", "xgboost", "catboost"]:
        model = next(item for item in config["models"] if item["id"] == model_id)
        first = module.generate_tuning_trials(model_id, model["kwargs"], maximum=20)
        second = module.generate_tuning_trials(model_id, model["kwargs"], maximum=20)
        assert first == second
        assert 1 <= len(first) <= 20
        assert len({json.dumps(item, sort_keys=True) for item in first}) == len(first)
    lgb = module.generate_tuning_trials(
        "lightgbm",
        next(item for item in config["models"] if item["id"] == "lightgbm")["kwargs"],
        maximum=20,
    )
    assert all(0.01 <= item["learning_rate"] <= 0.10 for item in lgb)
    assert all(4 <= item["max_depth"] <= 10 for item in lgb)
    assert (
        module.generate_tuning_trials("linear_ols", {"estimator": "ols"}, maximum=20)
        == []
    )

    mps_config = module.load_config(MPS_CONFIG_PATH)
    mps_model = next(item for item in mps_config["models"] if item["id"] == "mps_dnn")
    dnn = module.generate_tuning_trials("mps_dnn", mps_model["kwargs"], maximum=20)
    assert len(dnn) == 16
    assert all(item["device"] == "mps" for item in dnn)
    assert all(0.0003 <= item["lr"] <= 0.01 for item in dnn)
    assert all(1 <= len(item["hidden_layers"]) <= 3 for item in dnn)


def test_tuning_screen_jobs_use_development_quarters_only_one_window_and_one_seed() -> (
    None
):
    module = load_module()
    config = module.load_config(CONFIG_PATH)

    jobs = module.build_tuning_screen_jobs(config, "xgboost")

    assert len(jobs) == 12
    assert {job.window_days for job in jobs} == {756}
    assert {job.seed for job in jobs} == {42}
    assert all(
        job.quarter in config["rolling_validation"]["test_quarters"] for job in jobs
    )
    assert all(not job.quarter.startswith("2026") for job in jobs)


def test_mps_dnn_tuning_screen_uses_one_neural_seed_and_development_only() -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)

    jobs = module.build_tuning_screen_jobs(config, "mps_dnn")

    assert len(jobs) == 12
    assert {job.window_days for job in jobs} == {756}
    assert {job.seed for job in jobs} == {42}
    assert all(
        job.quarter in config["rolling_validation"]["test_quarters"] for job in jobs
    )
    assert all(not job.quarter.startswith("2026") for job in jobs)


def test_bounded_tuning_screens_20_then_reruns_best_with_full_matrix(
    tmp_path: Path,
) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    calls = []
    aggregate_calls = 0
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def training_runner(trial_config, jobs, **kwargs):
        calls.append((len(jobs), kwargs["run_dir"]))
        progress = {}
        for job in jobs:
            model_path = Path(kwargs["run_dir"]) / "models" / f"{job.key}.pkl"
            prediction_path = (
                Path(kwargs["run_dir"]) / "predictions" / f"{job.key}.parquet"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"model")
            prediction_path.write_bytes(b"prediction")
            progress[job.key] = {
                "status": "completed",
                "model_path": str(model_path),
                "model_sha256": module.file_sha256(model_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": module.file_sha256(prediction_path),
                "rows": 3,
                "split": {},
                "context_hash": str(kwargs.get("context_hash") or "test-context"),
            }
        return progress

    def aggregator(records, topk, thresholds):
        nonlocal aggregate_calls
        aggregate_calls += 1
        full = len({record["window_days"] for record in records}) == 3
        report = {
            "model_id": "xgboost",
            "eligible": full,
            "development_score": float(aggregate_calls),
            "rank_ic_mean": float(aggregate_calls),
            "positive_rank_ic_interval_ratio": 1.0,
            "stability": {
                "rank_correlation_median": 0.8 if full else None,
                "topk_overlap_mean": 0.7 if full else None,
            },
            "quarter_metrics": [{"quarter": "2025Q1", "rank_ic": 0.02}],
        }
        return {
            "model_reports": [report],
            "model_predictions": {"xgboost": prediction},
            "window_predictions": {},
        }

    result = module.run_bounded_tuning(
        config,
        ["xgboost"],
        provider_uri=tmp_path / "provider",
        calendar=pd.bdate_range("2019-01-01", "2025-12-31"),
        run_dir=tmp_path / "tuning",
        training_runner=training_runner,
        aggregator=aggregator,
    )

    assert len(calls) == 21
    assert all(count == 12 for count, _ in calls[:20])
    assert calls[-1][0] == 108
    assert result["model_reports"][0]["eligible"] is True
    assert result["model_reports"][0]["tuning"]["screened_trials"] == 20
    assert "xgboost" in result["model_predictions"]


def test_selected_development_artifacts_use_tuned_matrix_for_tuned_model() -> None:
    module = load_module()
    fixed = {
        "linear_ridge": [{"model_id": "linear_ridge", "model_path": "fixed-ridge"}],
        "xgboost": [{"model_id": "xgboost", "model_path": "fixed-xgb"}],
    }
    tuned = {
        "xgboost": [{"model_id": "xgboost", "model_path": "tuned-xgb"}],
    }
    reports = [
        {"model_id": "linear_ridge", "eligible": True},
        {
            "model_id": "xgboost",
            "eligible": True,
            "tuning": {"selected_trial": 3, "selected_kwargs": {"max_depth": 6}},
        },
    ]

    artifacts = module.select_development_artifacts(
        reports,
        fixed_artifacts_by_model=fixed,
        tuned_artifacts_by_model=tuned,
    )

    assert [item["model_path"] for item in artifacts] == [
        "fixed-ridge",
        "tuned-xgb",
    ]
