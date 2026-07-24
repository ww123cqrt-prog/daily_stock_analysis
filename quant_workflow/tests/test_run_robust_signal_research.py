from pathlib import Path
import json

import pandas as pd
import pytest

from robust_market_data import build_market_regimes
from run_robust_signal_research import (
    QlibJobRunner,
    SignalJob,
    _aggregate_signal_prediction_frames,
    _dataset_feature_options,
    _factor_contract,
    _failed_signal_outcome,
    _goal1_safe_split,
    _load_fundamental_source_context,
    _load_canonical_market_states,
    _load_prediction,
    _load_market_states,
    _summarize_factor_quality,
    _write_failed_status,
    build_jobs,
    load_config,
    run_research,
    run_job_loop,
    write_immutable_factor_contract,
)
from robust_strategy_core import make_rolling_split


def _job(number: int) -> SignalJob:
    return SignalJob("screening", "signal", "ridge", f"202{number}Q1", 756)


def test_job_loop_pauses_only_after_completed_atomic_job(tmp_path: Path) -> None:
    jobs = [_job(1), _job(2)]
    completed: list[str] = []

    def run_job(job: SignalJob):
        completed.append(job.key)
        prediction = tmp_path / f"{job.key}.parquet"
        model = tmp_path / f"{job.key}.pkl"
        prediction.touch()
        model.touch()
        return {"prediction_path": str(prediction), "model_path": str(model)}

    progress = run_job_loop(
        jobs,
        progress_path=tmp_path / "progress.json",
        context_hash="context",
        resume=False,
        run_job=run_job,
        pause_requested=lambda: len(completed) == 1,
    )

    assert completed == [jobs[0].key]
    assert progress["jobs"][jobs[0].key]["status"] == "completed"
    assert progress["pause_state"] == "paused"


def test_resume_requires_same_context_and_skips_valid_artifact(tmp_path: Path) -> None:
    jobs = [_job(1), _job(2)]
    progress_path = tmp_path / "progress.json"
    calls: list[str] = []

    def run_job(job: SignalJob):
        calls.append(job.key)
        prediction = tmp_path / f"{job.key}.parquet"
        model = tmp_path / f"{job.key}.pkl"
        prediction.touch()
        model.touch()
        return {"prediction_path": str(prediction), "model_path": str(model)}

    run_job_loop(
        jobs[:1],
        progress_path=progress_path,
        context_hash="context",
        resume=False,
        run_job=run_job,
        pause_requested=lambda: False,
    )
    run_job_loop(
        jobs,
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=run_job,
        pause_requested=lambda: False,
    )

    assert calls == [jobs[0].key, jobs[1].key]
    with pytest.raises(ValueError, match="resume_context_mismatch"):
        run_job_loop(
            jobs,
            progress_path=progress_path,
            context_hash="different",
            resume=True,
            run_job=run_job,
            pause_requested=lambda: False,
        )


def test_build_jobs_never_accepts_2026_quarter(signal_config) -> None:
    with pytest.raises(ValueError, match="2026"):
        build_jobs(signal_config, phase="screening", quarters=["2026Q1"])


def test_build_jobs_groups_shared_data_before_switching_quarters() -> None:
    config = {
        "signals": [{"id": "first"}, {"id": "second"}],
        "models": {
            "screening": [{"id": "equal"}, {"id": "ridge"}],
            "validation": [],
        },
        "periods": {
            "screening_quarters": ["2020Q1", "2020Q2"],
            "screening_window_days": 756,
        },
    }

    jobs = build_jobs(config, phase="screening")

    assert [
        (job.quarter, job.signal_id, job.model_id) for job in jobs
    ] == [
        ("2020Q1", "first", "equal"),
        ("2020Q1", "first", "ridge"),
        ("2020Q1", "second", "equal"),
        ("2020Q1", "second", "ridge"),
        ("2020Q2", "first", "equal"),
        ("2020Q2", "first", "ridge"),
        ("2020Q2", "second", "equal"),
        ("2020Q2", "second", "ridge"),
    ]


def test_qlib_job_runner_reuses_handler_for_same_split_and_horizon(
    tmp_path: Path, monkeypatch
) -> None:
    runner = object.__new__(QlibJobRunner)
    runner.config = {
        "signals": [
            {
                "id": "first",
                "feature_family": "first",
                "horizon_days": 5,
                "normalization_scope": "industry",
                "target": "csi800_excess",
            },
            {
                "id": "second",
                "feature_family": "second",
                "horizon_days": 5,
                "normalization_scope": "industry",
                "target": "csi800_excess",
            },
        ],
        "feature_families": {
            "first": {
                "include_handler_features": False,
                "external_prefixes": ["FUND_FIRST_"],
            },
            "second": {
                "include_handler_features": False,
                "external_prefixes": ["FUND_SECOND_"],
            },
        },
        "paths": {
            "fundamental_features": "reference/fundamentals",
            "industry_intervals": "reference/industry_intervals.parquet",
        },
        "periods": {
            "screening_quarters": ["2020Q1"],
            "validation_quarters": ["2023Q1"],
            "validation_days": 126,
        },
        "evaluation": {
            "benchmark": "SH000906",
            "market": "csi800",
            "market_state_history_start": "2014-01-02",
            "market_state_lookback_days": 60,
        },
    }
    runner.quant_data_dir = tmp_path
    runner.calendar = pd.bdate_range("2014-01-02", "2025-12-31")
    runner._cached_key = None
    runner._cached_dataset = None
    runner._cached_shared_key = None
    runner._cached_handler = None
    runner._cached_benchmark = None
    runner._cached_market_states = None
    runner._market_states_by_end = {}

    class FakeDataApi:
        @classmethod
        def features(cls, instruments, fields, **kwargs):
            dates = pd.bdate_range(kwargs["start_time"], kwargs["end_time"])
            index = pd.MultiIndex.from_product(
                [instruments, dates], names=["instrument", "datetime"]
            )
            return pd.DataFrame({fields[0]: 0.01}, index=index)

    monkeypatch.setattr("qlib.data.D", FakeDataApi)
    monkeypatch.setattr(
        "run_robust_signal_research._load_canonical_market_states",
        lambda _api, **kwargs: pd.Series(
            "neutral",
            index=pd.bdate_range(kwargs["history_start"], kwargs["end_date"]),
            dtype="string",
        ),
    )
    handler_calls = 0
    dataset_handlers = []
    shared_handler = object()

    def fake_init(config):
        nonlocal handler_calls
        if config.get("kind") == "handler":
            handler_calls += 1
            return shared_handler
        dataset_handlers.append(config["kwargs"]["handler"])
        return object()

    monkeypatch.setattr("qlib.utils.init_instance_by_config", fake_init)
    monkeypatch.setattr(
        "run_robust_signal_research.build_signal_handler_config",
        lambda **_kwargs: {"kind": "handler"},
    )
    monkeypatch.setattr(
        "run_robust_signal_research.build_signal_dataset_config",
        lambda **kwargs: {"kind": "dataset", "kwargs": {"handler": kwargs["handler"]}},
    )

    runner._dataset(SignalJob("screening", "first", "ridge", "2020Q1", 756))
    runner._dataset(SignalJob("screening", "second", "ridge", "2020Q1", 756))

    assert handler_calls == 1
    assert dataset_handlers == [shared_handler, shared_handler]


def test_fundamental_extension_failure_is_terminal_signal_not_found(
    signal_config,
) -> None:
    signal_config["fundamentals"]["research_round"] = "single_extension"

    outcome = _failed_signal_outcome(
        signal_config, reason_code="fundamental_validation_failed"
    )

    assert outcome == {
        "status": "signal_not_found",
        "reason_code": "fundamental_validation_failed",
        "next_stage": "goal_1_complete_no_signal",
        "next_gate": "close_goal_1_without_goal_2",
    }


def test_fundamental_source_context_requires_pass_and_verifies_artifact_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reference" / "fundamentals"
    (root / "daily").mkdir(parents=True)
    artifact = root / "daily" / "2025.parquet"
    artifact.write_bytes(b"immutable fundamental artifact")
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    source = {
        "status": "pass",
        "selected_source": "eastmoney_hsf10",
        "mixed_sources": False,
        "data_range": {"start": "2014-01-02", "end": "2025-12-31"},
        "artifacts": {"daily/2025.parquet": digest},
    }
    (root / "FUNDAMENTAL_SOURCE.json").write_text(json.dumps(source))
    config = {
        "paths": {"fundamental_features": "reference/fundamentals"},
        "fundamentals": {
            "research_round": "single_extension",
            "selected_source": "eastmoney_hsf10",
            "allow_mixed_sources": False,
        },
    }

    context = _load_fundamental_source_context(config, tmp_path)

    assert context["fundamental_source"] == "eastmoney_hsf10"
    assert context["fundamental_source_manifest_hash"]
    assert context["fundamental_artifacts_hash"]
    source["status"] = "blocked_data"
    (root / "FUNDAMENTAL_SOURCE.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="status=pass"):
        _load_fundamental_source_context(config, tmp_path)


def test_fundamental_source_context_rejects_tampered_artifact(tmp_path: Path) -> None:
    root = tmp_path / "reference" / "fundamentals"
    root.mkdir(parents=True)
    artifact = root / "artifact.parquet"
    artifact.write_bytes(b"actual")
    source = {
        "status": "pass",
        "selected_source": "eastmoney_hsf10",
        "mixed_sources": False,
        "data_range": {"start": "2014-01-02", "end": "2025-12-31"},
        "artifacts": {"artifact.parquet": "0" * 64},
    }
    (root / "FUNDAMENTAL_SOURCE.json").write_text(json.dumps(source))
    config = {
        "paths": {"fundamental_features": "reference/fundamentals"},
        "fundamentals": {
            "research_round": "single_extension",
            "selected_source": "eastmoney_hsf10",
            "allow_mixed_sources": False,
        },
    }

    with pytest.raises(ValueError, match="hash mismatch"):
        _load_fundamental_source_context(config, tmp_path)


def test_dataset_feature_options_resolve_fundamental_family(tmp_path: Path) -> None:
    config = {
        "paths": {"fundamental_features": "reference/fundamentals"},
        "feature_families": {
            "quality": {
                "include_handler_features": False,
                "handler_prefixes": [],
                "external_prefixes": ["FUND_QUALITY_"],
            }
        },
    }
    signal = {"feature_family": "quality"}

    options = _dataset_feature_options(config, signal, tmp_path)

    assert options == {
        "feature_prefixes": [],
        "include_handler_features": False,
        "external_features_path": tmp_path / "reference" / "fundamentals",
        "external_feature_prefixes": ["FUND_QUALITY_"],
    }


def test_preregistered_price_volume_and_fundamental_configs_are_distinct() -> None:
    config_dir = Path(__file__).parents[1] / "configs"

    price_volume = load_config(config_dir / "robust_signal_research.yaml")
    fundamentals = load_config(config_dir / "robust_signal_fundamentals.yaml")

    assert price_volume["fundamentals"]["research_round"] == "price_volume_baseline"
    assert fundamentals["fundamentals"]["research_round"] == "single_extension"
    assert len(fundamentals["signals"]) == 7
    assert fundamentals["artifacts"]["experiment_name"] == (
        "robust_csi800_signal_fundamentals"
    )
    assert fundamentals["paths"]["fundamental_features"] == (
        "reference/fundamentals"
    )


def test_ten_day_label_uses_ten_trading_day_purge_on_both_boundaries() -> None:
    calendar = pd.bdate_range("2017-01-02", "2025-12-31")

    split = make_rolling_split(
        calendar,
        "2023Q1",
        train_days=756,
        validation_days=126,
        label_horizon_days=10,
    )

    assert calendar.get_loc(split.validation_start) - calendar.get_loc(split.train_end) - 1 == 10
    assert calendar.get_loc(split.test_start) - calendar.get_loc(split.validation_end) - 1 == 10


def test_goal1_split_excludes_forward_labels_crossing_research_boundaries() -> None:
    calendar = pd.bdate_range("2017-01-02", "2026-03-31")
    config = {
        "periods": {
            "screening_quarters": ["2018Q1", "2022Q4"],
            "validation_quarters": ["2023Q1", "2025Q4"],
        }
    }

    screening = make_rolling_split(
        calendar,
        "2022Q4",
        train_days=756,
        validation_days=126,
        label_horizon_days=5,
    )
    safe_screening = _goal1_safe_split(
        config=config,
        job=SignalJob("screening", "signal", "ridge", "2022Q4", 756),
        calendar=calendar,
        split=screening,
        horizon_days=5,
    )
    pre_validation = calendar[calendar < pd.Timestamp("2023-01-01")]
    assert safe_screening.test_end == pre_validation[-7]
    assert calendar[calendar.get_loc(safe_screening.test_end) + 6] < pd.Timestamp(
        "2023-01-01"
    )

    validation = make_rolling_split(
        calendar,
        "2025Q4",
        train_days=756,
        validation_days=126,
        label_horizon_days=10,
    )
    safe_validation = _goal1_safe_split(
        config=config,
        job=SignalJob("validation", "signal", "ridge", "2025Q4", 756),
        calendar=calendar,
        split=validation,
        horizon_days=10,
    )
    pre_confirmation = calendar[calendar < pd.Timestamp("2026-01-01")]
    assert safe_validation.test_end == pre_confirmation[-12]
    assert calendar[calendar.get_loc(safe_validation.test_end) + 11] < pd.Timestamp(
        "2026-01-01"
    )

    ordinary = make_rolling_split(
        calendar,
        "2025Q3",
        train_days=756,
        validation_days=126,
        label_horizon_days=5,
    )
    assert _goal1_safe_split(
        config=config,
        job=SignalJob("validation", "signal", "ridge", "2025Q3", 756),
        calendar=calendar,
        split=ordinary,
        horizon_days=5,
    ) == ordinary


def test_prediction_reader_rejects_any_2026_rows(tmp_path: Path) -> None:
    path = tmp_path / "prediction.parquet"
    pd.DataFrame(
        {
            "datetime": ["2025-12-31", "2026-01-02"],
            "instrument": ["SH600000", "SH600000"],
            "score": [0.1, 0.2],
            "label": [0.01, 0.02],
            "benchmark_return": [0.0, 0.0],
            "market_state": ["neutral", "neutral"],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="2026"):
        _load_prediction(str(path))


def test_market_state_loader_resolves_csi800_as_a_qlib_universe() -> None:
    class FakeDataApi:
        market: str | None = None
        feature_instruments = None

        @classmethod
        def instruments(cls, market: str):
            cls.market = market
            return {"market": market, "filter_pipe": []}

        @classmethod
        def features(cls, instruments, _fields, **_kwargs):
            cls.feature_instruments = instruments
            dates = pd.bdate_range("2022-10-03", "2023-03-31")
            index = pd.MultiIndex.from_product(
                [["SH600000", "SZ000001"], dates],
                names=["instrument", "datetime"],
            )
            return pd.DataFrame({"$close": range(1, len(index) + 1)}, index=index)

    states = _load_market_states(
        FakeDataApi,
        market="csi800",
        start_date="2022-10-03",
        end_date="2023-03-31",
    )

    assert FakeDataApi.market == "csi800"
    assert FakeDataApi.feature_instruments == {
        "market": "csi800",
        "filter_pipe": [],
    }
    assert not states.empty


def test_market_regime_forgets_a_peak_older_than_one_trading_year() -> None:
    dates = pd.bdate_range("2020-01-02", periods=320)
    closes = pd.Series(100.0, index=dates)
    closes.iloc[0] = 200.0
    closes.iloc[-40:] = pd.Series(
        range(100, 140), index=dates[-40:], dtype="float64"
    )
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SH600000",
            "close": closes.to_numpy(),
        }
    )

    regimes = build_market_regimes(panel)

    assert regimes.iloc[-1] == "risk_on"


def test_canonical_market_states_use_one_history_start_and_are_future_invariant() -> None:
    class FakeDataApi:
        calls: list[tuple[str, str]] = []

        @classmethod
        def instruments(cls, market: str):
            return {"market": market, "filter_pipe": []}

        @classmethod
        def features(cls, instruments, _fields, **kwargs):
            del instruments
            cls.calls.append((str(kwargs["start_time"]), str(kwargs["end_time"])))
            dates = pd.bdate_range(kwargs["start_time"], kwargs["end_time"])
            symbols = ["SH600000", "SZ000001"]
            rows = []
            for position, date_value in enumerate(dates):
                for symbol_offset, symbol in enumerate(symbols):
                    rows.append(
                        {
                            "instrument": symbol,
                            "datetime": date_value,
                            "$close": 100.0 + position + symbol_offset,
                        }
                    )
            frame = pd.DataFrame(rows)
            return frame.set_index(["instrument", "datetime"])

    calendar = pd.bdate_range("1991-01-02", "2025-12-31")
    early = _load_canonical_market_states(
        FakeDataApi,
        market="csi800",
        calendar=calendar,
        history_start="2014-01-02",
        end_date="2023-03-31",
    )
    later = _load_canonical_market_states(
        FakeDataApi,
        market="csi800",
        calendar=calendar,
        history_start="2014-01-02",
        end_date="2023-06-30",
    )

    assert FakeDataApi.calls == [
        ("2014-01-02", "2023-03-31"),
        ("2014-01-02", "2023-06-30"),
    ]
    pd.testing.assert_series_equal(early, later.loc[early.index])


def test_mlflow_context_yields_active_recorder_not_experiment(monkeypatch) -> None:
    events = []
    recorder = object()

    class FakeStart:
        def __enter__(self):
            events.append("enter")
            return object()

        def __exit__(self, *_args):
            events.append("exit")

    class FakeR:
        @staticmethod
        def start(**_kwargs):
            return FakeStart()

        @staticmethod
        def get_recorder():
            events.append("recorder")
            return recorder

    monkeypatch.setattr("qlib.workflow.R", FakeR)
    runner = object.__new__(QlibJobRunner)
    runner.mlflow_enabled = True
    runner.config = {"artifacts": {"experiment_name": "test"}}
    job = SignalJob("screening", "signal", "model", "2023Q1", 756)

    with runner._mlflow_context(job) as active_run:
        assert active_run is recorder

    assert events == ["enter", "recorder", "exit"]


def test_failed_job_updates_top_level_status_before_reraising(tmp_path: Path) -> None:
    status = _write_failed_status(
        run_id="run",
        context={"context_hash": "context", "git_commit": "commit"},
        report_dir=tmp_path,
        current_stage="screening",
        completed_count=1,
        expected_count=2,
        error=RuntimeError("provider failed"),
    )

    assert status["lifecycle_status"] == "failed"
    assert status["pause_state"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert json.loads(
        (tmp_path / "robust_signal_status_latest.json").read_text()
    )["reason_code"] == "job_failed"


def test_smoke_failure_does_not_replace_latest_formal_status(tmp_path: Path) -> None:
    latest = tmp_path / "robust_signal_status_latest.json"
    latest.write_text(json.dumps({"run_id": "formal"}), encoding="utf-8")

    _write_failed_status(
        run_id="smoke",
        context={"context_hash": "context", "git_commit": "commit"},
        report_dir=tmp_path,
        current_stage="screening",
        completed_count=0,
        expected_count=1,
        error=RuntimeError("smoke failed"),
        publish_latest=False,
    )

    assert json.loads(latest.read_text(encoding="utf-8")) == {"run_id": "formal"}
    assert json.loads(
        (tmp_path / "runs" / "smoke" / "status.json").read_text(encoding="utf-8")
    )["lifecycle_status"] == "failed"


def test_validation_summary_failure_updates_run_and_latest_status(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "robust_signal_research.yaml"
    )
    context = {
        "config_hash": "config",
        "context_hash": "context",
        "git_commit": "commit",
        "git_clean": True,
        "data_manifest_hash": "manifest",
        "provider_tree_sha256": "provider",
        "provider_uri": str(tmp_path / "provider"),
    }
    screening_job = SignalJob("screening", "signal", "ridge", "2020Q1", 756)
    validation_job = SignalJob("validation", "signal", "ridge", "2023Q1", 756)

    monkeypatch.setattr(
        "run_robust_signal_research.build_context", lambda **_kwargs: context
    )
    monkeypatch.setattr(
        "run_robust_signal_research.QlibJobRunner", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        "run_robust_signal_research.build_jobs",
        lambda _config, *, phase, **_kwargs: (
            [screening_job] if phase == "screening" else [validation_job]
        ),
    )

    def completed_progress(_jobs, *, progress_path, **_kwargs):
        job = screening_job if "screening" in str(progress_path) else validation_job
        return {
            "pause_state": "completed",
            "jobs": {
                job.key: {
                    "status": "completed",
                    "phase": job.phase,
                    "signal_id": job.signal_id,
                    "model_id": job.model_id,
                    "quarter": job.quarter,
                    "window_days": job.window_days,
                }
            },
        }

    monkeypatch.setattr("run_robust_signal_research.run_job_loop", completed_progress)
    monkeypatch.setattr(
        "run_robust_signal_research.select_screening_candidates",
        lambda _reports, *, maximum: ["signal"],
    )
    evaluation_calls = 0

    def evaluate(_records, *, config, phase):
        nonlocal evaluation_calls
        del config
        evaluation_calls += 1
        if phase == "validation":
            raise ValueError("market_state values differ across signal windows")
        return [{"phase": phase, "passed": True}]

    monkeypatch.setattr("run_robust_signal_research.evaluate_records", evaluate)

    with pytest.raises(ValueError, match="market_state"):
        run_research(
            config_path=config_path,
            quant_data_dir=tmp_path / "quant_data",
            report_dir=tmp_path / "reports",
            run_id="run",
            resume=False,
            smoke=False,
            smoke_quarter="2023Q1",
            smoke_signal_id="signal",
            mlflow_enabled=False,
        )

    assert evaluation_calls == 2
    for path in (
        tmp_path / "reports" / "runs" / "run" / "status.json",
        tmp_path / "reports" / "robust_signal_status_latest.json",
    ):
        status = json.loads(path.read_text())
        assert status["lifecycle_status"] == "failed"
        assert status["current_stage"] == "validation_summary"
        assert status["reason_code"] == "summary_failed"
        assert status["completed_count"] == 2
        assert status["expected_count"] == 2


def test_signal_window_aggregation_preserves_and_validates_context_columns() -> None:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "SH600000")],
        names=["datetime", "instrument"],
    )
    left = pd.DataFrame(
        {
            "score": [0.1],
            "label": [0.02],
            "benchmark_return": [0.01],
            "market_state": ["neutral"],
        },
        index=index,
    )
    right = left.copy()
    right["score"] = 0.3

    result = _aggregate_signal_prediction_frames({"504": left, "756": right})

    assert result.loc[index[0], "benchmark_return"] == pytest.approx(0.01)
    assert result.loc[index[0], "market_state"] == "neutral"
    changed = right.copy()
    changed["market_state"] = "risk_on"
    with pytest.raises(ValueError, match="market_state"):
        _aggregate_signal_prediction_frames({"504": left, "756": changed})


def test_factor_contract_is_hash_stable_and_run_copy_is_immutable(
    tmp_path: Path,
) -> None:
    config = {
        "signals": [
            {
                "id": "signal",
                "feature_family": "full",
                "horizon_days": 5,
                "normalization_scope": "board",
                "target": "csi800_excess",
            }
        ],
        "feature_families": {"full": {"prefixes": []}},
    }
    context = {
        "config_hash": "config",
        "context_hash": "context",
        "git_commit": "commit",
        "provider_tree_sha256": "provider",
        "membership_snapshot_hash": "membership",
    }
    report = {"signal_id": "signal", "model_id": "ridge", "values": {"rank_ic": 0.02}}
    first = _factor_contract(config=config, context=context, report=report)
    second = _factor_contract(config=config, context=context, report=report)
    path = tmp_path / "factor_contract.json"

    write_immutable_factor_contract(path, first)

    assert first["contract_hash"] == second["contract_hash"]
    write_immutable_factor_contract(path, second)
    with pytest.raises(ValueError, match="immutable_factor_contract_mismatch"):
        write_immutable_factor_contract(path, {**second, "model_id": "ols"})


def test_factor_quality_summary_reads_completed_job_diagnostics(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "raw": {
                    "feature_count": 4,
                    "missing_rate_by_feature": {"a": 0.10, "b": 0.20},
                    "constant_features": ["constant_a"],
                    "low_variance_features": ["quiet_a"],
                    "nonfinite_value_count": 3,
                    "high_correlation_pairs": [{"left": "a", "right": "b"}],
                },
                "processed": {
                    "feature_count": 8,
                    "missing_rate_by_feature": {"a": 0.0},
                    "constant_features": [],
                    "low_variance_features": [],
                    "nonfinite_value_count": 0,
                    "high_correlation_pairs": [],
                },
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "raw": {
                    "feature_count": 6,
                    "missing_rate_by_feature": {"a": 0.40},
                    "constant_features": ["constant_b"],
                    "low_variance_features": ["quiet_b", "quiet_c"],
                    "nonfinite_value_count": 5,
                    "high_correlation_pairs": [
                        {"left": "a", "right": "b"},
                        {"left": "c", "right": "d"},
                    ],
                },
                "processed": {
                    "feature_count": 10,
                    "missing_rate_by_feature": {"a": 0.0},
                    "constant_features": [],
                    "low_variance_features": [],
                    "nonfinite_value_count": 0,
                    "high_correlation_pairs": [],
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_factor_quality(
        [{"diagnostics_path": str(first)}, {"diagnostics_path": str(second)}]
    )

    assert summary == {
        "diagnostics_count": 2,
        "feature_count_min": 4,
        "feature_count_max": 6,
        "minimum_feature_coverage": pytest.approx(0.60),
        "max_missing_rate": pytest.approx(0.40),
        "constant_features": ["constant_a", "constant_b"],
        "constant_feature_count_max": 1,
        "low_variance_features": ["quiet_a", "quiet_b", "quiet_c"],
        "low_variance_feature_count_max": 2,
        "nonfinite_value_count_total": 8,
        "high_correlation_pair_count_max": 2,
        "processed": {
            "diagnostics_count": 2,
            "feature_count_min": 8,
            "feature_count_max": 10,
            "minimum_feature_coverage": 1.0,
            "max_missing_rate": 0.0,
            "constant_features": [],
            "constant_feature_count_max": 0,
            "low_variance_features": [],
            "low_variance_feature_count_max": 0,
            "nonfinite_value_count_total": 0,
            "high_correlation_pair_count_max": 0,
        },
    }


def test_factor_quality_summary_fails_closed_when_diagnostics_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="factor diagnostics"):
        _summarize_factor_quality(
            [{"diagnostics_path": str(tmp_path / "missing.json")}]
        )


@pytest.fixture
def signal_config():
    return {
        "fundamentals": {"research_round": "price_volume_baseline"},
        "signals": [
            {
                "id": "signal",
            }
        ],
        "models": {
            "screening": [{"id": "ridge"}],
            "validation": [{"id": "ridge"}],
        },
        "periods": {
            "screening_quarters": ["2020Q1"],
            "validation_quarters": ["2023Q1"],
            "screening_window_days": 756,
            "validation_windows_days": [504, 756, 1008],
        },
    }
