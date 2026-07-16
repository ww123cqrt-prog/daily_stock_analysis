from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_robust_development.py"
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("run_robust_development", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_run_training_jobs_resume_skips_completed_artifact(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    state = completed_artifact_state(module, tmp_path / "run" / "artifacts", job)
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    (tmp_path / "run" / "progress.json").write_text(
        json.dumps({"jobs": {job.key: state}}),
        encoding="utf-8",
    )

    progress = module.run_training_jobs(
        config,
        [job],
        provider_uri=tmp_path / "provider",
        calendar=pd.bdate_range("2019-01-01", "2025-12-31"),
        run_dir=tmp_path / "run",
        dataset_factory=lambda *args: (_ for _ in ()).throw(
            AssertionError("dataset must not rebuild")
        ),
        model_factory=lambda *args: (_ for _ in ()).throw(
            AssertionError("model must not rebuild")
        ),
        mlflow_enabled=False,
        resume=True,
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


def test_pending_jobs_treats_malformed_recorded_rows_as_invalid(tmp_path: Path) -> None:
    module = load_module()
    config = module.load_config(CONFIG_PATH)
    job = module.build_training_jobs(config)[0]
    state = completed_artifact_state(module, tmp_path / "artifacts", job)
    state["rows"] = "not-an-integer"

    assert module.pending_jobs([job], {job.key: state}) == [job]


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
    report_dir = tmp_path / "reports"
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    model_prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def training_runner(config, jobs, **kwargs):
        assert len(jobs) == 396
        return {
            job.key: {
                "status": "completed",
                "prediction_path": str(tmp_path / f"{job.key}.parquet"),
            }
            for job in jobs
        }

    def aggregator(records, topk):
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
    assert result["status"] == "research_candidate_ready"
    assert latest["config_hash"]
    assert latest["data_manifest_hash"]
    assert frozen["selected_models"] == ["lightgbm", "linear_ridge"]
    assert frozen["selected_strategy"] == "full_risk_control"
    assert frozen["confirmation_run_started"] is False


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

    def aggregator(records, topk):
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
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in pd.bdate_range("2018-01-01", "2026-07-15"))
        + "\n",
        encoding="utf-8",
    )
    provider_state = tmp_path / "provider.json"
    provider_state.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(provider),
                "provider_key": "provider-v2",
            }
        ),
        encoding="utf-8",
    )
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    report_dir = tmp_path / "reports"
    run_dir = report_dir / "runs" / "resume-test" / "development"
    run_dir.mkdir(parents=True)
    (run_dir / "development_context.json").write_text(
        json.dumps(
            {
                "config_hash": module.file_sha256(CONFIG_PATH),
                "git_commit": "a" * 40,
                "data_manifest_hash": module.file_sha256(manifest),
                "provider_uri": str(provider.resolve()),
                "provider_key": "provider-v1",
                "as_of_date": "2026-07-15",
            }
        ),
        encoding="utf-8",
    )

    result = module.run_development(
        config_path=CONFIG_PATH,
        quant_data_dir=quant_data,
        report_dir=report_dir,
        provider_state_path=provider_state,
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
        return {
            job.key: {
                "status": "completed",
                "prediction_path": str(tmp_path / f"{job.key}.parquet"),
            }
            for job in jobs
        }

    def aggregator(records, topk):
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
