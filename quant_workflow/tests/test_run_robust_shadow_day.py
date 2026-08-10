from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import cloudpickle
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_robust_shadow_day.py"
)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
MPS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "robust_strategy_mps_dnn.yaml"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_robust_shadow_day", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_provider(
    root: Path,
    dates: list[str],
    feature_bytes: bytes,
    *,
    canonical_provider: Path | None = None,
) -> Path:
    (root / "calendars").mkdir(parents=True)
    (root / "features" / "sh600000").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text(
        "\n".join(dates) + "\n", encoding="utf-8"
    )
    (root / "features" / "sh600000" / "close.day.bin").write_bytes(feature_bytes)
    (root / "ROBUST_PROVIDER.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "provider_key": root.name,
                "canonical_provider": str((canonical_provider or root).resolve()),
            }
        ),
        encoding="utf-8",
    )
    return root


def prediction(day: str, scores: tuple[float, float] = (2.0, 1.0)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime([day, day]),
            "instrument": ["SH600000", "SZ000001"],
            "score": list(scores),
        }
    )


def make_run_fixture(
    module, tmp_path: Path, *, dates: list[str], shadow_start_after: str = "2026-07-15"
) -> dict:
    canonical = write_provider(tmp_path / "canonical", dates, b"abcdef")
    frozen_provider = write_provider(
        tmp_path / "frozen-provider",
        [day for day in dates if day <= shadow_start_after],
        b"abc",
        canonical_provider=canonical,
    )
    current_provider = write_provider(
        tmp_path / "current-provider", dates, b"abcdef", canonical_provider=canonical
    )

    quant_data = tmp_path / "quant-data"
    intervals = quant_data / "reference" / "pit_csi800" / "membership_intervals.parquet"
    intervals.parent.mkdir(parents=True)
    intervals.write_bytes(b"current-pit-intervals")
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 2}\n', encoding="utf-8")

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    config = module.load_config(CONFIG_PATH)
    selected = ["linear_ridge", "lightgbm"]
    model_configs = {
        model_id: {
            "class": next(item for item in config["models"] if item["id"] == model_id)[
                "class"
            ],
            "module_path": next(
                item for item in config["models"] if item["id"] == model_id
            )["module_path"],
            "kwargs": next(item for item in config["models"] if item["id"] == model_id)[
                "kwargs"
            ],
        }
        for model_id in selected
    }
    artifacts = []
    for model_id in selected:
        seeds = (
            config["training"]["tree_seeds"]
            if model_id in module.TREE_MODELS
            else [None]
        )
        for window in config["training"]["windows_trading_days"]:
            for seed in seeds:
                suffix = "fixed" if seed is None else str(seed)
                model_path = tmp_path / f"{model_id}-{window}-{suffix}.pkl"
                model_path.write_bytes(f"frozen-{model_id}-{window}-{suffix}".encode())
                artifacts.append(
                    {
                        "job_key": f"confirmation_2026__w{window}__{model_id}__s{suffix}",
                        "model_id": model_id,
                        "window_days": window,
                        "seed": seed,
                        "model_path": str(model_path),
                        "model_sha256": module.file_sha256(model_path),
                        "split": {
                            "train_start": "2022-01-04",
                            "train_end": "2023-12-29",
                            "validation_start": "2024-01-08",
                            "validation_end": "2024-07-01",
                            "test_start": "2026-01-01",
                            "test_end": shadow_start_after,
                        },
                    }
                )
    ensemble_path = tmp_path / "confirmation_ensemble_predictions.parquet"
    prediction(shadow_start_after).to_parquet(ensemble_path, index=False)
    ensemble_hash = module.file_sha256(ensemble_path)
    frozen = {
        "status": "research_candidate_ready",
        "confirmation_status": "shadow_observation",
        "config_hash": module.file_sha256(CONFIG_PATH),
        "git_commit": "a" * 40,
        "provider_uri": str(frozen_provider),
        "provider_key": "frozen-provider",
        "selected_models": selected,
        "selected_model_configs": model_configs,
        "selected_strategy": "full_risk_control",
        "selection_hash": module.selection_contract_hash(
            selected, model_configs, "full_risk_control"
        ),
        "confirmation_model_artifacts": artifacts,
        "confirmation_ensemble_prediction_path": str(ensemble_path),
        "confirmation_ensemble_prediction_hash": ensemble_hash,
        "confirmation_run_id": "confirmation-one",
        "shadow_start_after": shadow_start_after,
    }
    runtime_environment = module.collect_runtime_environment(config)
    frozen["runtime_environment"] = runtime_environment
    frozen["runtime_environment_hash"] = runtime_environment["environment_hash"]
    frozen["frozen_context_hash"] = module.frozen_context_hash(frozen)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    confirmation = {
        "status": "shadow_observation",
        "run_id": "confirmation-one",
        "config_hash": frozen["config_hash"],
        "selected_models": selected,
        "selected_strategy": "full_risk_control",
        "latest_market_day": shadow_start_after,
        "confirmation_model_artifact_count": len(artifacts),
        "confirmation_ensemble_prediction_path": str(ensemble_path),
        "confirmation_ensemble_prediction_hash": ensemble_hash,
        "runtime_environment_hash": frozen["runtime_environment_hash"],
    }
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_path.write_text(json.dumps(confirmation), encoding="utf-8")
    provider_latest = tmp_path / "provider_latest.json"
    provider_latest.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider_uri": str(frozen_provider),
                "provider_key": "frozen-provider",
            }
        ),
        encoding="utf-8",
    )
    return {
        "config_path": CONFIG_PATH,
        "quant_data_dir": quant_data,
        "report_dir": report_dir,
        "frozen_path": frozen_path,
        "confirmation_path": confirmation_path,
        "provider_state_path": provider_latest,
        "current_provider": current_provider,
        "frozen_provider": frozen_provider,
        "frozen": frozen,
        "dates": dates,
    }


def run_kwargs(
    module,
    fixture: dict,
    *,
    as_of_date: str,
    inference_runner=None,
    daily_dates: list[str] | None = None,
):
    observed = {}
    daily_dates = daily_dates or [as_of_date]

    def market_loader(**kwargs):
        observed["market_start"] = kwargs["start_date"]
        observed["market_end"] = kwargs["end_date"]
        return pd.DataFrame({"date": pd.to_datetime(daily_dates)})

    def benchmark_loader(provider, **kwargs):
        observed["benchmark_start"] = kwargs["start_date"]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(daily_dates),
                "open": range(100, 100 + len(daily_dates)),
            }
        )

    def strategy_runner(**kwargs):
        observed["predictions"] = kwargs["predictions"].copy()
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(
                {
                    "date": pd.to_datetime(daily_dates),
                    "portfolio_return": [0.001] * len(daily_dates),
                    "benchmark_return": [0.0002] * len(daily_dates),
                }
            ),
            "metrics": {},
        }

    kwargs = {
        "config_path": fixture["config_path"],
        "quant_data_dir": fixture["quant_data_dir"],
        "report_dir": fixture["report_dir"],
        "frozen_path": fixture["frozen_path"],
        "confirmation_path": fixture["confirmation_path"],
        "provider_state_path": fixture["provider_state_path"],
        "as_of_date": as_of_date,
        "provider_resolver": lambda provider, *, append_only_from: (
            observed.setdefault(
                "provider_resolution", (provider, Path(append_only_from))
            )
            and fixture["current_provider"]
        ),
        "calendar_loader": lambda _provider: list(fixture["dates"]),
        "inference_runner": inference_runner
        or (lambda **kwargs: prediction(as_of_date)),
        "market_loader": market_loader,
        "benchmark_loader": benchmark_loader,
        "regime_builder": lambda market: pd.Series(
            "neutral", index=pd.to_datetime(daily_dates)
        ),
        "strategy_runner": strategy_runner,
        "git_state_provider": lambda: {
            "commit": "a" * 40,
            "clean": True,
            "changes": [],
        },
    }
    return kwargs, observed


def test_provider_prefix_change_is_blocked_before_inference(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    calls = []
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-16",
        inference_runner=lambda **kwargs: calls.append(kwargs),
    )
    kwargs["provider_resolver"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        module.QlibProviderHistoryError("historical bars changed")
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "provider_history_changed"
    assert calls == []
    assert not (fixture["report_dir"] / "shadow_state_latest.json").exists()


def test_expected_artifact_keys_use_neural_seeds_for_mps_dnn() -> None:
    module = load_module()
    config = module.load_config(MPS_CONFIG_PATH)

    keys = module._expected_artifact_keys(config, ["linear_ridge", "mps_dnn"])
    dnn_keys = {item for item in keys if item[0] == "mps_dnn"}

    assert len(keys) == 12
    assert len(dnn_keys) == 9
    assert {item[1] for item in dnn_keys} == {504, 756, 1008}
    assert {item[2] for item in dnn_keys} == {42, 2026, 3407}


def test_runtime_environment_mismatch_blocks_before_inference(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(
        module, tmp_path, dates=["2026-07-15", "2026-07-16"]
    )
    frozen = json.loads(fixture["frozen_path"].read_text(encoding="utf-8"))
    frozen["runtime_environment_hash"] = "f" * 64
    frozen["frozen_context_hash"] = module.frozen_context_hash(frozen)
    fixture["frozen_path"].write_text(json.dumps(frozen), encoding="utf-8")
    calls = []
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-16",
        inference_runner=lambda **kwargs: calls.append(kwargs),
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "frozen_runtime_environment_mismatch"
    assert calls == []


def test_current_provider_is_resolved_through_the_workflow_adapter(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(
        module, tmp_path, dates=["2026-07-15", "2026-07-16"]
    )
    kwargs, observed = run_kwargs(module, fixture, as_of_date="2026-07-16")

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "shadow_observation"
    assert observed["provider_resolution"] == (
        "cleaned",
        fixture["frozen_provider"],
    )


def test_unavailable_current_provider_is_reported_without_starting_inference(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    calls = []
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-16",
        inference_runner=lambda **values: calls.append(values),
    )
    kwargs["provider_resolver"] = lambda *_args, **_kwargs: tmp_path / "missing"

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "provider_prepare_failed"
    assert calls == []


def test_default_inference_loads_models_without_fit_and_only_predicts_as_of(
    tmp_path: Path,
) -> None:
    module = load_module()
    as_of = "2026-07-16"
    captured = []

    class PredictOnlyModel:
        def fit(self, dataset):
            raise AssertionError("fit must never be called during shadow inference")

        def predict(self, dataset, segment="test"):
            assert segment == "test"
            index = pd.MultiIndex.from_product(
                [pd.to_datetime([as_of]), ["SH600000", "SZ000001"]],
                names=["datetime", "instrument"],
            )
            return pd.Series([2.0, 1.0], index=index)

    artifacts = []
    for model_id in ["linear_ridge", "lightgbm"]:
        path = tmp_path / f"{model_id}.pkl"
        with path.open("wb") as stream:
            cloudpickle.dump(PredictOnlyModel(), stream)
        artifacts.append(
            {
                "job_key": model_id,
                "model_id": model_id,
                "model_path": str(path),
                "model_sha256": module.file_sha256(path),
                "split": {
                    "train_start": "2022-01-04",
                    "train_end": "2023-12-29",
                    "validation_start": "2024-01-08",
                    "validation_end": "2024-07-01",
                    "test_start": "2026-01-01",
                    "test_end": "2026-07-15",
                },
            }
        )

    result = module.infer_frozen_models(
        provider_uri=tmp_path / "provider",
        artifacts=artifacts,
        selected_models=["linear_ridge", "lightgbm"],
        as_of_date=as_of,
        dataset_factory=lambda provider, config, split: (
            captured.append((provider, config, split)) or object()
        ),
        qlib_initializer=lambda provider: None,
    )

    assert set(result["datetime"].dt.strftime("%Y-%m-%d")) == {as_of}
    assert len(captured) == 2
    assert all(
        str(split.test_start.date()) == as_of and str(split.test_end.date()) == as_of
        for _, _, split in captured
    )
    assert all(
        config["kwargs"]["segments"]["test"] == [as_of, as_of]
        for _, config, _ in captured
    )


def test_two_level_percentile_aggregation_is_equal_weighted() -> None:
    module = load_module()
    frames = {
        "ridge-w1": ("ridge", prediction("2026-07-16", (100.0, 0.0))),
        "ridge-w2": ("ridge", prediction("2026-07-16", (0.0, 100.0))),
        "tree-w1": ("tree", prediction("2026-07-16", (10.0, 0.0))),
    }

    result = module.aggregate_daily_predictions(
        frames, selected_models=["ridge", "tree"]
    )

    scores = result.set_index("instrument")["score"].to_dict()
    assert scores["SH600000"] == pytest.approx(0.875)
    assert scores["SZ000001"] == pytest.approx(0.625)


def test_aggregation_rejects_mismatched_indexes_and_non_finite_scores() -> None:
    module = load_module()
    mismatch = prediction("2026-07-16").iloc[:1]
    with pytest.raises(ValueError, match="indexes"):
        module.aggregate_daily_predictions(
            {"a": ("ridge", prediction("2026-07-16")), "b": ("tree", mismatch)},
            selected_models=["ridge", "tree"],
        )
    invalid = prediction("2026-07-16")
    invalid.loc[0, "score"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        module.aggregate_daily_predictions(
            {"a": ("ridge", invalid), "b": ("tree", prediction("2026-07-16"))},
            selected_models=["ridge", "tree"],
        )


def test_only_today_is_inferred_and_two_missed_days_are_recorded_failed(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = make_run_fixture(
        module, tmp_path, dates=["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-20"]
    )
    inference_dates = []

    def infer(**kwargs):
        inference_dates.append(kwargs["as_of_date"])
        return prediction(kwargs["as_of_date"])

    kwargs, observed = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-20",
        inference_runner=infer,
        daily_dates=["2026-07-16", "2026-07-17", "2026-07-20"],
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "shadow_observation"
    assert result["appended_days"] == 3
    assert inference_dates == ["2026-07-20"]
    state = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )
    assert [(item["date"], item["task_success"]) for item in state["observations"]] == [
        ("2026-07-16", False),
        ("2026-07-17", False),
        ("2026-07-20", True),
    ]
    signals = pd.read_parquet(
        fixture["report_dir"] / "shadow_signal_predictions.parquet"
    )
    assert set(signals["datetime"].dt.strftime("%Y-%m-%d")) == {"2026-07-20"}
    assert set(observed["predictions"]["datetime"].dt.strftime("%Y-%m-%d")) == {
        "2026-07-15",
        "2026-07-20",
    }
    assert observed["market_start"] == "2026-01-01"
    assert observed["benchmark_start"] == "2026-01-01"


def test_replay_missing_an_expected_trading_day_fails_closed(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(
        module, tmp_path, dates=["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-20"]
    )
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-20",
        daily_dates=["2026-07-16", "2026-07-20"],
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "shadow_calendar_incomplete"
    assert result["appended_days"] == 0
    assert not (fixture["report_dir"] / "shadow_state_latest.json").exists()


def test_replay_with_an_extra_non_trading_day_fails_closed(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(
        module, tmp_path, dates=["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-20"]
    )
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-20",
        daily_dates=["2026-07-16", "2026-07-17", "2026-07-18", "2026-07-20"],
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "shadow_calendar_incomplete"
    assert result["appended_days"] == 0
    assert not (fixture["report_dir"] / "shadow_state_latest.json").exists()


def test_runner_and_state_cli_share_one_state_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    import robust_shadow_tracking

    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    kwargs, _ = run_kwargs(module, fixture, as_of_date="2026-07-16")
    state_path = fixture["report_dir"] / "shadow_state_latest.json"
    runner_at_write = threading.Event()
    release_runner = threading.Event()
    cli_done = threading.Event()
    errors: list[BaseException] = []
    original_write = module._write_json

    def delayed_write(path: Path, payload: dict) -> None:
        dates = [item["date"] for item in payload.get("observations", [])]
        if path == state_path and dates == ["2026-07-16"]:
            runner_at_write.set()
            if not release_runner.wait(timeout=5):
                raise TimeoutError("test did not release shadow-day runner")
        original_write(path, payload)

    monkeypatch.setattr(module, "_write_json", delayed_write)

    def run_daily() -> None:
        try:
            module.run_shadow_day(**kwargs)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def run_cli_update() -> None:
        try:
            robust_shadow_tracking.update_shadow_state_file(
                state_path,
                confirmation_run_id="confirmation-one",
                frozen_config_hash=fixture["frozen"]["config_hash"],
                start_after="2026-07-15",
                observation={
                    "date": "2026-07-17",
                    "portfolio_return": 0.001,
                    "benchmark_return": 0.0002,
                    "task_success": True,
                },
                acceptance_thresholds=module.load_config(CONFIG_PATH)["acceptance"],
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            cli_done.set()

    runner = threading.Thread(target=run_daily)
    cli = threading.Thread(target=run_cli_update)
    runner.start()
    assert runner_at_write.wait(timeout=5)
    cli.start()
    cli_finished_while_runner_was_writing = cli_done.wait(timeout=0.2)
    release_runner.set()
    runner.join(timeout=5)
    cli.join(timeout=5)

    assert cli_finished_while_runner_was_writing is False
    assert errors == []
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert [item["date"] for item in saved["observations"]] == [
        "2026-07-16",
        "2026-07-17",
    ]


def test_backlog_stops_at_the_existing_60_day_verdict(tmp_path: Path) -> None:
    module = load_module()
    start = pd.Timestamp("2026-07-15")
    shadow_days = pd.bdate_range(start + pd.offsets.BDay(1), periods=61)
    dates = [str(start.date()), *(str(day.date()) for day in shadow_days)]
    fixture = make_run_fixture(module, tmp_path, dates=dates)
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date=str(shadow_days[-1].date()),
        daily_dates=[str(day.date()) for day in shadow_days],
    )

    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["appended_days"] == 60
    state = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )
    assert state["metrics"]["trading_days"] == 60
    assert state["status"] == "development_failed"
    assert state["verdict"]["remaining_days"] == 0


def test_repeated_run_is_idempotent(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    kwargs, _ = run_kwargs(module, fixture, as_of_date="2026-07-16")

    first = module.run_shadow_day(**kwargs)
    state_before = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )
    second = module.run_shadow_day(**kwargs)
    state_after = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )

    assert first["appended_days"] == 1
    assert second["appended_days"] == 0
    assert state_after == state_before
    assert (
        len(
            pd.read_parquet(fixture["report_dir"] / "shadow_signal_predictions.parquet")
        )
        == 2
    )


def test_replay_cannot_change_an_already_recorded_return(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    kwargs, _ = run_kwargs(module, fixture, as_of_date="2026-07-16")
    first = module.run_shadow_day(**kwargs)
    assert first["status"] == "shadow_observation"

    _, observed = run_kwargs(module, fixture, as_of_date="2026-07-16")

    def changed_strategy(**kwargs):
        observed["predictions"] = kwargs["predictions"].copy()
        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-07-16"]),
                    "portfolio_return": [0.002],
                    "benchmark_return": [0.0002],
                }
            ),
            "metrics": {},
        }

    kwargs["strategy_runner"] = changed_strategy
    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "shadow_observation_conflict"
    state = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )
    assert state["observations"][0]["portfolio_return"] == 0.001


def test_corrupted_model_hash_blocks_before_inference(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])
    Path(
        fixture["frozen"]["confirmation_model_artifacts"][0]["model_path"]
    ).write_bytes(b"corrupt")
    calls = []
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-16",
        inference_runner=lambda **kwargs: calls.append(kwargs),
    )

    result = module.run_shadow_day(**kwargs)

    assert result["reason_code"] == "model_artifact_invalid"
    assert calls == []
    assert not (fixture["report_dir"] / "shadow_signal_predictions.parquet").exists()


def test_non_trading_day_is_skipped_without_observation(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-17"])
    calls = []
    kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-18",
        inference_runner=lambda **kwargs: calls.append(kwargs),
    )

    result = module.run_shadow_day(**kwargs)

    assert result["outcome"] == "skipped"
    assert result["reason_code"] == "non_trading_day"
    assert result["appended_days"] == 0
    assert calls == []
    assert not (fixture["report_dir"] / "shadow_state_latest.json").exists()


def test_inference_failure_records_today_unsuccessful_and_returns_failure(
    tmp_path: Path,
) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])

    def fail_inference(**kwargs):
        raise RuntimeError("private absolute path must not leak: /tmp/model.pkl")

    kwargs, _ = run_kwargs(
        module, fixture, as_of_date="2026-07-16", inference_runner=fail_inference
    )
    result = module.run_shadow_day(**kwargs)

    assert result["status"] == "development_failed"
    assert result["reason_code"] == "inference_failed"
    assert result["error_type"] == "RuntimeError"
    assert module.exit_code_for_result(result) != 0
    assert "/tmp" not in json.dumps(result)
    state = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )
    assert state["observations"][0]["task_success"] is False
    assert state["status"] == "shadow_observation"


def test_failed_day_cannot_be_retried_into_success(tmp_path: Path) -> None:
    module = load_module()
    fixture = make_run_fixture(module, tmp_path, dates=["2026-07-15", "2026-07-16"])

    def fail_inference(**kwargs):
        raise RuntimeError("inference failed")

    first_kwargs, _ = run_kwargs(
        module, fixture, as_of_date="2026-07-16", inference_runner=fail_inference
    )
    first = module.run_shadow_day(**first_kwargs)
    assert first["reason_code"] == "inference_failed"
    state_before = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )

    retry_calls = []
    retry_kwargs, _ = run_kwargs(
        module,
        fixture,
        as_of_date="2026-07-16",
        inference_runner=lambda **kwargs: (
            retry_calls.append(kwargs) or prediction("2026-07-16")
        ),
    )
    retry = module.run_shadow_day(**retry_kwargs)
    state_after = json.loads(
        (fixture["report_dir"] / "shadow_state_latest.json").read_text(encoding="utf-8")
    )

    assert retry["status"] == "development_failed"
    assert retry["reason_code"] == "inference_failed"
    assert retry["inference_success"] is False
    assert module.exit_code_for_result(retry) != 0
    assert retry_calls == []
    assert state_after == state_before
    assert not (fixture["report_dir"] / "shadow_signal_predictions.parquet").exists()
