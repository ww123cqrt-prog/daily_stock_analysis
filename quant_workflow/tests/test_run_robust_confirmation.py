from __future__ import annotations

import importlib.util
import hashlib
import json
import pickle
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_robust_confirmation.py"
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
MPS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robust_strategy_mps_dnn.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_robust_confirmation", SCRIPT_PATH
    )
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
        file_sha256,
    )

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


def frozen_fixture(
    module, root: Path, *, artifact_context_hash: str | None = None
) -> tuple[Path, Path, Path]:
    from _robust_membership_integrity import membership_read_lock
    from _robust_reference_integrity import reference_read_lock

    provider = root / "provider"
    (provider / "calendars").mkdir(parents=True)
    calendar = pd.bdate_range("2018-01-01", "2026-07-15")
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(str(day.date()) for day in calendar) + "\n", encoding="utf-8"
    )
    provider_tree_sha256 = seal_provider(provider)
    config = module.load_config(CONFIG_PATH)
    selected = ["linear_ridge", "lightgbm"]
    model_map = {item["id"]: item for item in config["models"]}
    selection_contract = {
        "selected_models": selected,
        "selected_model_configs": {
            model_id: {
                "class": model_map[model_id]["class"],
                "module_path": model_map[model_id]["module_path"],
                "kwargs": model_map[model_id]["kwargs"],
            }
            for model_id in selected
        },
        "selected_strategy": "full_risk_control",
    }
    selection_hash = hashlib.sha256(
        json.dumps(selection_contract, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    quant_data = root / "quant_data"
    quant_data.mkdir()
    manifest = quant_data / "DATA_MANIFEST.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")
    pit_dir = quant_data / "reference" / "pit_csi800"
    (pit_dir / "daily").mkdir(parents=True)
    (pit_dir / "MEMBERSHIP_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2014-01-02",
                    "end_date": "2026-07-15",
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
    (pit_dir / "daily" / "2026.parquet").write_bytes(b"membership daily")
    with membership_read_lock(pit_dir) as membership_snapshot:
        membership_snapshot_hash = membership_snapshot.fingerprint()

    reference = quant_data / "reference"
    (reference / "REFERENCE_SOURCE.json").write_text(
        json.dumps(
            {
                "requested": "auto",
                "selected": "baostock",
                "fallback_reason": "jqdata_permission_denied",
                "date_range": {
                    "start_date": "2014-01-02",
                    "end_date": "2026-07-15",
                },
                "industry_classification": "CSRC",
                "membership_snapshot_hash": membership_snapshot_hash,
            }
        ),
        encoding="utf-8",
    )
    for file_name in (
        "security_master.parquet",
        "industry_intervals.parquet",
        "st_status.parquet",
        "suspension_status.parquet",
    ):
        (reference / file_name).write_bytes(file_name.encode("ascii"))
    with reference_read_lock(reference) as reference_snapshot:
        reference_snapshot_hash = reference_snapshot.fingerprint()
    development_run_dir = root / "development"
    (development_run_dir / "models").mkdir(parents=True)
    (development_run_dir / "predictions").mkdir(parents=True)
    development_context_hash = "d" * 64
    development_artifacts = []
    for job in module.build_training_jobs(config):
        if job.model_id not in selected:
            continue
        split = module.make_job_split(
            calendar,
            job.quarter,
            window_days=job.window_days,
            validation_days=int(config["rolling_validation"]["validation_days"]),
            label_horizon_days=5,
        )
        model_path = development_run_dir / "models" / f"{job.key}.pkl"
        prediction_path = development_run_dir / "predictions" / f"{job.key}.parquet"
        model_path.write_bytes(
            pickle.dumps({"model_id": job.model_id, "seed": job.seed})
        )
        pd.DataFrame(
            {
                "datetime": [split.test_start],
                "instrument": ["SH600000"],
                "score": [1.0],
                "label": [0.01],
            }
        ).to_parquet(prediction_path, index=False)
        development_artifacts.append(
            {
                "status": "completed",
                "job_key": job.key,
                "model_id": job.model_id,
                "window_days": job.window_days,
                "quarter": job.quarter,
                "seed": job.seed,
                "model_path": str(model_path.resolve()),
                "model_sha256": module.file_sha256(model_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": module.file_sha256(prediction_path),
                "rows": 1,
                "split": module._split_record(split),
                "context_hash": artifact_context_hash or development_context_hash,
            }
        )
    frozen = {
        "status": "research_candidate_ready",
        "as_of_date": "2026-07-15",
        "config_hash": module.file_sha256(CONFIG_PATH),
        "git_commit": "a" * 40,
        "data_manifest_hash": module.file_sha256(manifest),
        "provider_uri": str(provider),
        "provider_key": "provider-v1",
        "provider_tree_sha256": provider_tree_sha256,
        "membership_snapshot_hash": membership_snapshot_hash,
        "reference_snapshot_hash": reference_snapshot_hash,
        "selected_models": selected,
        "selected_model_configs": selection_contract["selected_model_configs"],
        "selected_strategy": "full_risk_control",
        "selection_hash": selection_hash,
        "development_run_id": "development-one",
        "development_run_dir": str(development_run_dir.resolve()),
        "development_context_hash": development_context_hash,
        "development_artifacts": development_artifacts,
        "development_artifacts_hash": module.payload_sha256(development_artifacts),
        "confirmation_run_started": False,
        "shadow_tracking_started": False,
    }
    runtime_environment = module.collect_runtime_environment(config)
    frozen["runtime_environment"] = runtime_environment
    frozen["runtime_environment_hash"] = runtime_environment["environment_hash"]
    frozen["frozen_context_hash"] = module.frozen_context_hash(frozen)
    frozen_path = root / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    provider_state = root / "provider_state.json"
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
    return frozen_path, provider_state, provider


def confirmation_artifact_progress(module, jobs, root: Path) -> dict[str, dict]:
    root.mkdir(parents=True, exist_ok=True)
    config = module.load_config(CONFIG_PATH)
    calendar = pd.bdate_range("2018-01-01", "2026-07-15")
    confirmation_days = calendar[calendar >= pd.Timestamp("2026-01-01")]
    latest_signal_day = confirmation_days[-7]
    progress: dict[str, dict] = {}
    for job in jobs:
        prediction_path = (root / f"{job.key}.parquet").resolve()
        model_path = (root / f"{job.key}.pkl").resolve()
        pd.DataFrame(
            {
                "datetime": [
                    pd.Timestamp("2026-01-05"),
                    pd.Timestamp("2026-01-05"),
                    pd.Timestamp("2026-01-05"),
                ],
                "instrument": ["A", "B", "C"],
                "score": [3.0, 2.0, 1.0],
                "label": [0.03, 0.02, 0.01],
            }
        ).to_parquet(prediction_path, index=False)
        model_path.write_bytes(
            pickle.dumps({"model_id": job.model_id, "seed": job.seed})
        )
        split = module.make_confirmation_split(
            calendar,
            train_days=job.window_days,
            validation_days=int(config["rolling_validation"]["validation_days"]),
            label_horizon_days=5,
            test_start="2026-01-01",
            test_end=str(latest_signal_day.date()),
        )
        progress[job.key] = {
            "status": "completed",
            "model_id": job.model_id,
            "window_days": job.window_days,
            "quarter": job.quarter,
            "seed": job.seed,
            "model_path": str(model_path),
            "model_sha256": module.file_sha256(model_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": module.file_sha256(prediction_path),
            "rows": 3,
            "split": {
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "validation_start": str(split.validation_start.date()),
                "validation_end": str(split.validation_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
            },
        }
    return progress


def test_confirmation_jobs_keep_frozen_models_and_full_window_seed_matrix(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path)
    config = module.load_config(CONFIG_PATH)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    jobs = module.build_confirmation_jobs(config, frozen)

    assert len(jobs) == 12
    assert {job.model_id for job in jobs} == {"linear_ridge", "lightgbm"}
    assert {job.window_days for job in jobs} == {504, 756, 1008}
    assert len([job for job in jobs if job.model_id == "linear_ridge"]) == 3
    assert len([job for job in jobs if job.model_id == "lightgbm"]) == 9
    assert {job.quarter for job in jobs} == {"confirmation_2026"}


def test_confirmation_jobs_use_neural_seed_matrix_for_mps_dnn() -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)
    frozen = {"selected_models": ["linear_ridge", "mps_dnn"]}

    jobs = module.build_confirmation_jobs(config, frozen)
    dnn_jobs = [job for job in jobs if job.model_id == "mps_dnn"]

    assert len(jobs) == 12
    assert len(dnn_jobs) == 9
    assert {job.window_days for job in dnn_jobs} == {504, 756, 1008}
    assert {job.seed for job in dnn_jobs} == {42, 2026, 3407}


def test_confirmation_refuses_missing_frozen_candidate(tmp_path: Path) -> None:
    module = load_module()

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=tmp_path / "missing.json",
        provider_state_path=tmp_path / "missing-provider.json",
        run_id="confirmation-one",
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "frozen_candidate_missing"


def test_confirmation_is_single_use_across_run_ids(tmp_path: Path) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path)
    state_path = tmp_path / "state.json"

    first = module.claim_confirmation(
        frozen_path, state_path, run_id="first", resume=False
    )

    assert first["lifecycle_status"] == "running"
    try:
        module.claim_confirmation(
            frozen_path, state_path, run_id="second", resume=False
        )
    except RuntimeError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("a second confirmation run id was accepted")


def test_confirmation_claim_lock_is_bound_to_frozen_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path / "candidate")
    observed: list[Path] = []

    @contextmanager
    def fake_lock(directory: Path):
        observed.append(Path(directory))
        yield

    monkeypatch.setattr(module, "_claim_lock", fake_lock)
    module.claim_confirmation(
        frozen_path,
        tmp_path / "different-report-root" / "state.json",
        run_id="first",
        resume=False,
    )

    assert observed == [frozen_path.parent]


def test_same_confirmation_run_can_resume_after_recoverable_stop(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path)
    state_path = tmp_path / "state.json"
    module.claim_confirmation(frozen_path, state_path, run_id="first", resume=False)
    stopped = json.loads(state_path.read_text(encoding="utf-8"))
    stopped["lifecycle_status"] = "stopped"
    state_path.write_text(json.dumps(stopped), encoding="utf-8")

    resumed = module.claim_confirmation(
        frozen_path, state_path, run_id="first", resume=True
    )

    assert resumed["lifecycle_status"] == "running"
    assert resumed["run_id"] == "first"


def test_running_confirmation_claim_cannot_resume_concurrently(tmp_path: Path) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path)
    state_path = tmp_path / "state.json"
    module.claim_confirmation(frozen_path, state_path, run_id="first", resume=False)

    try:
        module.claim_confirmation(frozen_path, state_path, run_id="first", resume=True)
    except RuntimeError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("a running confirmation claim was resumed concurrently")


def test_resume_recovers_claim_when_frozen_write_succeeded_but_state_is_missing(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, _, _ = frozen_fixture(module, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["confirmation_run_started"] = True
    frozen["confirmation_run_id"] = "first"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    state_path = tmp_path / "missing-state.json"

    resumed = module.claim_confirmation(
        frozen_path, state_path, run_id="first", resume=True
    )

    assert resumed["lifecycle_status"] == "running"
    assert state_path.is_file()


def test_passing_confirmation_enters_shadow_without_changing_selection(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(
        module, tmp_path, artifact_context_hash="e" * 64
    )
    report_dir = tmp_path / "reports"
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def training_runner(config, jobs, **kwargs):
        return confirmation_artifact_progress(module, jobs, tmp_path / "artifacts")

    def aggregator(records, topk, thresholds):
        return {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        }

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-one",
        training_runner=training_runner,
        aggregator=aggregator,
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
        strategy_runner=lambda **kwargs: {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        },
    )

    frozen_after = json.loads(frozen_path.read_text(encoding="utf-8"))
    state = json.loads(
        (report_dir / "confirmation_state_latest.json").read_text(encoding="utf-8")
    )
    latest = json.loads(
        (report_dir / "confirmation_latest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "shadow_observation"
    assert result["selected_models"] == ["linear_ridge", "lightgbm"]
    assert result["selected_strategy"] == "full_risk_control"
    assert frozen_after["selected_models"] == ["linear_ridge", "lightgbm"]
    assert frozen_after["selected_strategy"] == "full_risk_control"
    assert frozen_after["confirmation_run_started"] is True
    assert state["status"] == "shadow_observation"
    artifacts = frozen_after["confirmation_model_artifacts"]
    assert len(artifacts) == 12
    assert [item["job_key"] for item in artifacts] == sorted(
        item["job_key"] for item in artifacts
    )
    assert all(
        set(item)
        == {
            "job_key",
            "model_id",
            "window_days",
            "seed",
            "model_path",
            "model_sha256",
            "split",
        }
        for item in artifacts
    )
    assert all(
        Path(item["model_path"]).is_absolute() and Path(item["model_path"]).is_file()
        for item in artifacts
    )
    assert all(
        item["model_sha256"] == module.file_sha256(Path(item["model_path"]))
        for item in artifacts
    )
    assert all(
        set(item["split"])
        == {
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "test_start",
            "test_end",
        }
        for item in artifacts
    )
    ensemble_path = Path(frozen_after["confirmation_ensemble_prediction_path"])
    assert ensemble_path.is_absolute() and ensemble_path.is_file()
    assert frozen_after["confirmation_ensemble_prediction_hash"] == module.file_sha256(
        ensemble_path
    )
    assert frozen_after["shadow_start_after"] == "2026-07-15"
    ensemble = pd.read_parquet(ensemble_path)
    assert list(ensemble.columns) == ["datetime", "instrument", "score", "label"]
    assert len(ensemble) == 3
    assert latest["confirmation_model_artifact_count"] == 12
    assert latest["confirmation_ensemble_prediction_path"] == str(ensemble_path)
    assert latest["confirmation_ensemble_prediction_hash"] == module.file_sha256(
        ensemble_path
    )
    assert "confirmation_model_artifacts" not in latest


def test_confirmation_publishes_final_result_under_snapshot_and_claim_locks(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    report_dir = tmp_path / "reports"
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )
    active = {"membership": 0, "reference": 0, "claim": 0}
    observed_publications: list[tuple[int, int, int]] = []
    original_membership_lock = module.membership_read_lock
    original_reference_lock = module.reference_read_lock
    original_claim_lock = module._claim_lock
    original_write_json = module._write_json

    @contextmanager
    def tracking_membership_lock(path):
        with original_membership_lock(path) as snapshot:
            active["membership"] += 1
            try:
                yield snapshot
            finally:
                active["membership"] -= 1

    @contextmanager
    def tracking_reference_lock(path):
        with original_reference_lock(path) as snapshot:
            active["reference"] += 1
            try:
                yield snapshot
            finally:
                active["reference"] -= 1

    @contextmanager
    def tracking_claim_lock(path):
        with original_claim_lock(path):
            active["claim"] += 1
            try:
                yield
            finally:
                active["claim"] -= 1

    def tracking_write_json(path, payload):
        is_final_report = payload.get("status") == "shadow_observation"
        is_final_frozen = (
            Path(path) == frozen_path
            and payload.get("confirmation_status") == "shadow_observation"
        )
        if is_final_report or is_final_frozen:
            observed_publications.append(
                (active["membership"], active["reference"], active["claim"])
            )
        original_write_json(path, payload)

    monkeypatch.setattr(module, "membership_read_lock", tracking_membership_lock)
    monkeypatch.setattr(module, "reference_read_lock", tracking_reference_lock)
    monkeypatch.setattr(module, "_claim_lock", tracking_claim_lock)
    monkeypatch.setattr(module, "_write_json", tracking_write_json)

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-lock-publication",
        training_runner=lambda config, jobs, **kwargs: confirmation_artifact_progress(
            module, jobs, tmp_path / "artifacts"
        ),
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=lambda **kwargs: {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        },
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "shadow_observation"
    assert observed_publications
    assert all(depths == (1, 1, 1) for depths in observed_publications)


def test_confirmation_rechecks_membership_snapshot_before_publishing_shadow(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def mutate_membership_and_simulate(**kwargs):
        daily = (
            tmp_path
            / "quant_data"
            / "reference"
            / "pit_csi800"
            / "daily"
            / "2026.parquet"
        )
        daily.write_bytes(daily.read_bytes() + b"changed-during-confirmation")
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        }

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-membership-recheck",
        training_runner=lambda config, jobs, **kwargs: confirmation_artifact_progress(
            module, jobs, tmp_path / "artifacts"
        ),
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=mutate_membership_and_simulate,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "reproducibility_state_changed_during_confirmation"
    assert "membership_snapshot_hash" in result["reproducibility_failures"]
    assert frozen["shadow_tracking_started"] is False


def test_confirmation_rejects_completed_jobs_with_missing_model_or_invalid_split(
    tmp_path: Path,
) -> None:
    module = load_module()
    for invalid_field in ("model_path", "split", "wrong_split"):
        root = tmp_path / invalid_field
        frozen_path, provider_state, _ = frozen_fixture(module, root)
        config = module.load_config(CONFIG_PATH)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        jobs = module.build_confirmation_jobs(
            module._frozen_config(config, frozen), frozen
        )
        progress = confirmation_artifact_progress(module, jobs, root / "artifacts")
        if invalid_field == "wrong_split":
            progress[jobs[0].key]["split"]["test_end"] = "2026-01-30"
        else:
            progress[jobs[0].key].pop(invalid_field)

        result = module.run_confirmation(
            config_path=CONFIG_PATH,
            quant_data_dir=root / "quant_data",
            report_dir=root / "reports",
            frozen_path=frozen_path,
            provider_state_path=provider_state,
            run_id=f"confirmation-invalid-{invalid_field}",
            training_runner=lambda *args, _progress=progress, **kwargs: _progress,
            aggregator=lambda records, topk, thresholds: (_ for _ in ()).throw(
                AssertionError("invalid artifacts reached aggregation")
            ),
            git_state_provider=lambda: {
                "commit": "a" * 40,
                "clean": True,
                "changes": [],
            },
        )

        assert result["status"] == "development_failed"
        assert result["reason_code"] == "confirmation_artifact_invalid"
        assert (
            json.loads(frozen_path.read_text(encoding="utf-8"))[
                "shadow_tracking_started"
            ]
            is False
        )


def test_confirmation_rejects_completed_job_with_damaged_model_hash(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    config = module.load_config(CONFIG_PATH)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    jobs = module.build_confirmation_jobs(module._frozen_config(config, frozen), frozen)
    progress = confirmation_artifact_progress(module, jobs, tmp_path / "artifacts")
    Path(progress[jobs[0].key]["model_path"]).write_bytes(b"damaged after hashing")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-damaged-model",
        training_runner=lambda *args, **kwargs: progress,
        aggregator=lambda records, topk, thresholds: (_ for _ in ()).throw(
            AssertionError("invalid artifacts reached aggregation")
        ),
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "confirmation_artifact_invalid"


def test_confirmation_rechecks_model_artifacts_before_publishing_shadow(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )
    captured: dict[str, dict] = {}

    def training_runner(config, jobs, **kwargs):
        captured.update(
            confirmation_artifact_progress(module, jobs, tmp_path / "artifacts")
        )
        return captured

    def damage_model_and_simulate(**kwargs):
        first = captured[sorted(captured)[0]]
        Path(first["model_path"]).write_bytes(b"damaged during confirmation")
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        }

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-artifact-recheck",
        training_runner=training_runner,
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=damage_model_and_simulate,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    latest = json.loads(
        (tmp_path / "reports" / "confirmation_latest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "confirmation_artifact_invalid"
    assert latest["run_id"] == "confirmation-artifact-recheck"
    assert frozen["shadow_tracking_started"] is False
    assert "confirmation_model_artifacts" not in frozen


def test_confirmation_refuses_dirty_or_different_code_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    report_dir = tmp_path / "reports"

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=report_dir,
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-dirty",
        git_state_provider=lambda: {
            "commit": "c" * 40,
            "clean": False,
            "changes": [" M quant_workflow/scripts/run_robust_confirmation.py"],
        },
    )

    frozen_after = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "frozen_git_state_mismatch"
    assert frozen_after["confirmation_run_started"] is False
    assert not (report_dir / "confirmation_state_latest.json").exists()


def test_confirmation_refuses_modified_frozen_selection_contract(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["selected_strategy"] = "weekly_top50"
    frozen["frozen_context_hash"] = module.frozen_context_hash(frozen)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-selection-mismatch",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "frozen_selection_mismatch"


def test_recoverable_provider_preflight_failure_does_not_consume_confirmation(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    provider_state.write_text(json.dumps({"status": "blocked_data"}), encoding="utf-8")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-one",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "provider_state_missing"
    assert frozen["confirmation_run_started"] is False
    assert not (tmp_path / "reports" / "confirmation_state_latest.json").exists()


def test_confirmation_rechecks_git_state_before_publishing_shadow_result(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )
    git_states = iter(
        [
            {"commit": "a" * 40, "clean": True, "changes": []},
            {"commit": "a" * 40, "clean": False, "changes": [" M changed.py"]},
        ]
    )

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-one",
        training_runner=lambda config, jobs, **kwargs: confirmation_artifact_progress(
            module, jobs, tmp_path / "artifacts"
        ),
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=lambda **kwargs: {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        },
        git_state_provider=lambda: next(git_states),
    )

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "reproducibility_state_changed_during_confirmation"


def test_confirmation_rechecks_full_provider_identity_before_publishing_shadow(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def mutate_provider_and_simulate(**kwargs):
        provider_state.write_text(
            json.dumps(
                {
                    "status": "blocked_data",
                    "provider_uri": str(tmp_path / "redirected-provider"),
                    "provider_key": "provider-v1",
                }
            ),
            encoding="utf-8",
        )
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        }

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-provider-recheck",
        training_runner=lambda config, jobs, **kwargs: confirmation_artifact_progress(
            module, jobs, tmp_path / "artifacts"
        ),
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=mutate_provider_and_simulate,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["status"] == "development_failed"
    assert result["reason_code"] == "reproducibility_state_changed_during_confirmation"
    assert "provider_status" in result["reproducibility_failures"]
    assert "provider_uri" in result["reproducibility_failures"]
    assert frozen["shadow_tracking_started"] is False


def test_confirmation_refuses_provider_identity_change_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    state = json.loads(provider_state.read_text(encoding="utf-8"))
    state["provider_key"] = "provider-v2"
    provider_state.write_text(json.dumps(state), encoding="utf-8")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-one",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_provider_mismatch"
    assert frozen["confirmation_run_started"] is False


def test_confirmation_refuses_changed_membership_snapshot_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    daily = (
        tmp_path / "quant_data" / "reference" / "pit_csi800" / "daily" / "2026.parquet"
    )
    daily.write_bytes(daily.read_bytes() + b"changed")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-membership-changed",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_membership_snapshot_mismatch"
    assert frozen["confirmation_run_started"] is False


def test_confirmation_refuses_changed_reference_snapshot_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    security = tmp_path / "quant_data" / "reference" / "security_master.parquet"
    security.write_bytes(security.read_bytes() + b"changed")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-reference-changed",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_reference_snapshot_mismatch"
    assert frozen["confirmation_run_started"] is False


def test_confirmation_refuses_tampered_frozen_context_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["development_run_id"] = "substituted-run"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-frozen-context-tampered",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen_after = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_context_mismatch"
    assert frozen_after["confirmation_run_started"] is False


def test_confirmation_refuses_runtime_environment_mismatch_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["runtime_environment_hash"] = "f" * 64
    frozen["frozen_context_hash"] = module.frozen_context_hash(frozen)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-runtime-mismatch",
        git_state_provider=lambda: {
            "commit": "a" * 40,
            "clean": True,
            "changes": [],
        },
    )

    frozen_after = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_runtime_environment_mismatch"
    assert frozen_after["confirmation_run_started"] is False


def test_confirmation_refuses_tampered_development_artifact_before_claim(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    Path(frozen["development_artifacts"][0]["model_path"]).write_bytes(b"tampered")

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-development-artifact-tampered",
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    frozen_after = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert result["reason_code"] == "frozen_development_artifacts_mismatch"
    assert frozen_after["confirmation_run_started"] is False


def test_confirmation_recomputes_selection_hash_before_publishing(
    tmp_path: Path,
) -> None:
    module = load_module()
    frozen_path, provider_state, _ = frozen_fixture(module, tmp_path)
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    prediction = pd.DataFrame(
        {"score": [3.0, 2.0, 1.0], "label": [0.03, 0.02, 0.01]}, index=index
    )

    def mutate_frozen_and_simulate(**kwargs):
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen["selected_strategy"] = "weekly_top50"
        frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(),
            "metrics": {
                "annual_excess_return": 0.02,
                "information_ratio": 0.3,
                "max_drawdown": -0.10,
            },
        }

    result = module.run_confirmation(
        config_path=CONFIG_PATH,
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        frozen_path=frozen_path,
        provider_state_path=provider_state,
        run_id="confirmation-one",
        training_runner=lambda config, jobs, **kwargs: confirmation_artifact_progress(
            module, jobs, tmp_path / "artifacts"
        ),
        aggregator=lambda records, topk, thresholds: {
            "model_reports": [],
            "model_predictions": {"linear_ridge": prediction, "lightgbm": prediction},
            "window_predictions": {},
        },
        market_loader=lambda **kwargs: pd.DataFrame({"date": []}),
        benchmark_loader=lambda *args, **kwargs: pd.DataFrame({"date": [], "open": []}),
        regime_builder=lambda panel: pd.Series(dtype="object"),
        strategy_runner=mutate_frozen_and_simulate,
        git_state_provider=lambda: {"commit": "a" * 40, "clean": True, "changes": []},
    )

    assert result["reason_code"] == "reproducibility_state_changed_during_confirmation"
    assert "selection_hash" in result["reproducibility_failures"]
