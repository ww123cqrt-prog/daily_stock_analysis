from pathlib import Path
import json

import pandas as pd
import pytest

from run_robust_signal_research import (
    QlibJobRunner,
    SignalJob,
    _aggregate_signal_prediction_frames,
    _factor_contract,
    _load_prediction,
    _load_market_states,
    _summarize_factor_quality,
    _write_failed_status,
    build_jobs,
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
