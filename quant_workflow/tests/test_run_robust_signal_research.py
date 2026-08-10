from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import pandas as pd
import pytest

from robust_market_data import build_market_regimes
from robust_signal_data_contract import RobustSignalDataBlocked
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
    build_context,
    build_jobs,
    load_config,
    run_research,
    run_job_loop,
    request_pause,
    write_immutable_factor_contract,
)
from robust_strategy_core import make_rolling_split


_RUN_CONTEXT_FIELDS = {
    "context_hash",
    "run_mode",
    "smoke_quarter",
    "smoke_signal_id",
}


def _signed_context(payload: dict[str, object]) -> dict[str, object]:
    context = dict(payload)
    immutable = {
        key: value for key, value in context.items() if key not in _RUN_CONTEXT_FIELDS
    }
    context["context_hash"] = hashlib.sha256(
        json.dumps(
            immutable,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return context


def _job(number: int) -> SignalJob:
    return SignalJob("screening", "signal", "ridge", f"202{number}Q1", 756)


def _job_split(job: SignalJob) -> dict[str, str]:
    year = int(job.quarter[:4])
    return {
        "train_start": f"{year - 3}-01-02",
        "train_end": f"{year - 1}-06-30",
        "validation_start": f"{year - 1}-07-01",
        "validation_end": f"{year - 1}-12-31",
        "test_start": f"{year}-01-04",
        "test_end": f"{year}-03-31",
    }


def _write_job_artifacts(run_dir: Path, job: SignalJob) -> dict[str, object]:
    split = _job_split(job)
    prediction = run_dir / job.phase / "predictions" / f"{job.key}.parquet"
    model = run_dir / job.phase / "models" / f"{job.key}.pkl"
    diagnostics = (
        run_dir
        / job.phase
        / "diagnostics"
        / f"{job.signal_id}__{job.quarter}__w{job.window_days}.json"
    )
    prediction.parent.mkdir(parents=True, exist_ok=True)
    model.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "datetime": split["test_start"],
                "instrument": "SH600000",
                "score": 0.25,
                "label": 0.02,
                "benchmark_return": 0.01,
                "market_state": "neutral",
            }
        ]
    ).to_parquet(prediction, index=False)
    model.write_bytes(f"model:{job.key}".encode())
    diagnostic_section = {
        "rows": 1,
        "feature_count": 1,
        "missing_rate_by_feature": {"feature": 0.0},
        "constant_features": [],
        "low_variance_features": [],
        "nonfinite_value_count": 0,
        "high_correlation_pairs": [],
    }
    diagnostics.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "raw": diagnostic_section,
                "processed": diagnostic_section,
            }
        ),
        encoding="utf-8",
    )
    return {
        "prediction_path": str(prediction),
        "model_path": str(model),
        "diagnostics_path": str(diagnostics),
        "rows": 1,
        "split": split,
    }


def _complete_job(
    *, run_dir: Path, job: SignalJob, calls: list[str]
) -> tuple[Path, dict[str, object]]:
    progress_path = run_dir / job.phase / "progress.json"

    def run_job(current: SignalJob) -> dict[str, object]:
        calls.append(current.key)
        return _write_job_artifacts(run_dir, current)

    progress = run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=False,
        run_job=run_job,
        pause_requested=lambda: False,
    )
    return progress_path, progress


def _context_config() -> dict[str, object]:
    return {
        "paths": {
            "data_manifest": "DATA_MANIFEST.json",
        },
        "fundamentals": {"research_round": "price_volume_baseline"},
    }


def _write_context_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "robust_signal.yaml"
    config_path.write_text("name: robust-signal-test\n", encoding="utf-8")
    return config_path


def _frozen_readiness_kwargs(tmp_path: Path) -> dict[str, object]:
    artifact_sha256 = "9" * 64
    return {
        "readiness_artifact_path": (
            tmp_path
            / "reports"
            / "robust_signal"
            / "readiness_artifacts"
            / f"robust_signal_data_readiness_{artifact_sha256}.json"
        ),
        "readiness_artifact_sha256": artifact_sha256,
    }


def test_build_context_reports_missing_robust_data_readiness_without_csi800_misdirection(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        build_context(
            config_path=_write_context_config(tmp_path),
            config=_context_config(),
            quant_data_dir=quant_data,
            report_root=tmp_path / "reports",
            **_frozen_readiness_kwargs(tmp_path),
        )

    assert getattr(caught.value, "reason_code", None) == "readiness_artifact_missing"
    assert "does not contain PIT csi800 instruments" not in str(caught.value)


def test_build_context_rejects_legacy_readiness_without_data_module_attestation(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text('{"version": 2}\n', encoding="utf-8")
    provider = quant_data / "runtime" / "robust_strategy" / "providers" / "provider-v1"
    (provider / "instruments").mkdir(parents=True)
    (provider / "instruments" / "csi800.txt").write_text(
        "SH600000\t2020-01-02\t2025-12-31\n", encoding="utf-8"
    )
    legacy_payload = {
        "contract_version": "robust_signal_data_v2",
        "status": "ready",
        "trainable": True,
        "provider_uri": str(provider),
        "provider_key": "provider-v1",
        "provider_tree_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
    }
    legacy_bytes = (
        json.dumps(legacy_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    legacy_sha256 = hashlib.sha256(legacy_bytes).hexdigest()
    artifact_path = (
        tmp_path
        / "reports"
        / "robust_signal"
        / "readiness_artifacts"
        / f"robust_signal_data_readiness_{legacy_sha256}.json"
    )
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(legacy_bytes)

    with pytest.raises(ValueError) as caught:
        build_context(
            config_path=_write_context_config(tmp_path),
            config=_context_config(),
            quant_data_dir=quant_data,
            report_root=tmp_path / "reports",
            readiness_artifact_path=artifact_path,
            readiness_artifact_sha256=legacy_sha256,
        )

    assert getattr(caught.value, "reason_code", None) == "robust_readiness_invalid"


def test_build_context_rejects_readiness_outside_controlled_report_root(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    readiness_bytes = b"{}\n"
    readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    artifact = (
        tmp_path
        / "attacker"
        / "readiness_artifacts"
        / f"robust_signal_data_readiness_{readiness_sha256}.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(readiness_bytes)

    with pytest.raises(RobustSignalDataBlocked) as caught:
        build_context(
            config_path=_write_context_config(tmp_path),
            config=_context_config(),
            quant_data_dir=quant_data,
            report_root=tmp_path / "reports",
            readiness_artifact_path=artifact,
            readiness_artifact_sha256=readiness_sha256,
        )

    assert caught.value.reason_code == "robust_readiness_invalid"


def test_build_context_uses_versioned_readiness_provenance_fields(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    (quant_data / "DATA_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    provider = tmp_path / "provider"
    provider.mkdir()
    provider_status = {
        "provider_tree_sha256": "a" * 64,
        "provider_key": "b" * 64,
        "membership_intervals_sha256": "c" * 64,
        "quality_report_id": "data-quality-" + "d" * 16,
        "quality_evidence_artifact_name": (
            "data_quality_report_20260805_120000_" + "d" * 16 + ".json"
        ),
        "quality_evidence_sha256": "d" * 64,
        "cleaned_provider_identity": {
            "dataset_id": "qlib_ashare_cleaned_v2",
            "snapshot_id": "cleaned-v2",
            "manifest_sha256": "e" * 64,
            "content_sha256": "f" * 64,
        },
    }

    with (
        patch(
            "run_robust_signal_research._resolve_provider",
            return_value=(provider, provider_status),
        ),
        patch("run_robust_signal_research._git_commit", return_value="commit"),
        patch("run_robust_signal_research._git_clean", return_value=True),
    ):
        context = build_context(
            config_path=_write_context_config(tmp_path),
            config=_context_config(),
            quant_data_dir=quant_data,
            report_root=tmp_path / "reports",
            **_frozen_readiness_kwargs(tmp_path),
        )

    assert context["membership_intervals_sha256"] == "c" * 64
    assert context["quality_report_id"] == "data-quality-" + "d" * 16
    assert context["quality_evidence_artifact_name"].startswith(
        "data_quality_report_"
    )
    assert context["quality_evidence_sha256"] == "d" * 64
    assert (
        context["cleaned_provider_identity"]
        == provider_status["cleaned_provider_identity"]
    )
    assert context["readiness_artifact_sha256"] == "9" * 64
    assert "membership_snapshot_hash" not in context
    assert "reference_snapshot_hash" not in context


def test_build_context_consumes_frozen_manifest_hash_without_current_manifest(
    tmp_path: Path,
) -> None:
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    provider = tmp_path / "provider"
    provider.mkdir()
    provider_status = {
        "provider_tree_sha256": "a" * 64,
        "provider_key": "b" * 64,
        "membership_intervals_sha256": "c" * 64,
        "data_manifest_sha256": "d" * 64,
        "quality_report_id": "data-quality-" + "e" * 16,
        "quality_evidence_artifact_name": (
            "data_quality_report_20260805_120000_" + "e" * 16 + ".json"
        ),
        "quality_evidence_sha256": "e" * 64,
        "cleaned_provider_identity": {
            "dataset_id": "qlib_ashare_cleaned_v2",
            "snapshot_id": "cleaned-v2",
            "manifest_sha256": "f" * 64,
            "content_sha256": "a" * 64,
        },
    }

    with (
        patch(
            "run_robust_signal_research._resolve_provider",
            return_value=(provider, provider_status),
        ),
        patch("run_robust_signal_research._git_commit", return_value="commit"),
        patch("run_robust_signal_research._git_clean", return_value=True),
    ):
        context = build_context(
            config_path=_write_context_config(tmp_path),
            config=_context_config(),
            quant_data_dir=quant_data,
            report_root=tmp_path / "reports",
            **_frozen_readiness_kwargs(tmp_path),
        )

    assert context["data_manifest_hash"] == "d" * 64


def test_resolve_provider_consumes_only_the_frozen_readiness_artifact(
    tmp_path: Path,
) -> None:
    from run_robust_signal_research import _resolve_provider

    artifact = tmp_path / "readiness.json"
    artifact.write_text("{}\n", encoding="utf-8")
    provider = (
        tmp_path / "quant_data" / "runtime" / "robust_strategy" / "providers" / "a"
    )
    status = {"provider_uri": str(provider)}

    with patch(
        "run_robust_signal_research.consume_accepted_robust_signal_data",
        return_value=status,
    ) as consume:
        resolved, returned = _resolve_provider(
            _context_config(),
            tmp_path / "reports",
            quant_data_dir=tmp_path / "quant_data",
            readiness_artifact_path=artifact,
            readiness_artifact_sha256="a" * 64,
        )

    assert resolved == provider
    assert returned == status
    consume.assert_called_once_with(
        artifact_path=artifact,
        artifact_sha256="a" * 64,
        provider_root=tmp_path / "quant_data" / "runtime",
        reports_dir=tmp_path / "reports",
    )


def test_runner_publishes_blocked_data_if_readiness_changes_after_web_submission(
    tmp_path: Path,
) -> None:
    blocked = RobustSignalDataBlocked(
        "data_manifest_changed",
        "The robust provider was built from an older data manifest.",
        "鲁棒 CSI800 provider 绑定的是旧数据清单，请重新验收并准备。",
    )
    report_dir = tmp_path / "reports"

    with (
        patch("run_robust_signal_research.load_config", return_value={}),
        patch("run_robust_signal_research.build_context", side_effect=blocked),
    ):
        payload = run_research(
            config_path=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            run_id="robust_signal_smoke",
            resume=False,
            smoke=True,
            smoke_quarter="2025Q4",
            smoke_signal_id="alpha158_raw_5d",
            mlflow_enabled=False,
            **_frozen_readiness_kwargs(tmp_path),
        )

    assert payload["status"] == "blocked_data"
    assert payload["reason_code"] == "data_manifest_changed"
    assert payload["current_stage"] == "data_readiness"
    assert json.loads(
        (report_dir / "runs" / "robust_signal_smoke" / "status.json").read_text(
            encoding="utf-8"
        )
    ) == payload
    assert json.loads(
        (report_dir / "robust_signal_smoke_status_latest.json").read_text(
            encoding="utf-8"
        )
    ) == payload
    assert not (report_dir / "robust_signal_status_latest.json").exists()


def test_job_loop_pauses_only_after_completed_atomic_job(tmp_path: Path) -> None:
    jobs = [_job(1), _job(2)]
    completed: list[str] = []
    run_dir = tmp_path / "run"

    def run_job(job: SignalJob):
        completed.append(job.key)
        return _write_job_artifacts(run_dir, job)

    progress = run_job_loop(
        jobs,
        progress_path=run_dir / "screening" / "progress.json",
        context_hash="context",
        resume=False,
        run_job=run_job,
        pause_requested=lambda: len(completed) == 1,
    )

    assert completed == [jobs[0].key]
    assert progress["jobs"][jobs[0].key]["status"] == "completed"
    assert progress["pause_state"] == "paused"


def test_request_pause_rejects_completed_run_without_rewriting_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "completed-run"
    run_dir.mkdir(parents=True)
    completed = {
        "run_id": "completed-run",
        "lifecycle_status": "completed",
        "pause_state": "completed",
        "status": "signal_diagnosis_complete",
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(completed), encoding="utf-8")

    with pytest.raises(ValueError, match="not actively running"):
        request_pause(tmp_path, "completed-run")

    assert json.loads(status_path.read_text(encoding="utf-8")) == completed
    assert not (run_dir / "pause.request.json").exists()


def test_resume_requires_same_context_and_skips_valid_artifact(tmp_path: Path) -> None:
    jobs = [_job(1), _job(2)]
    run_dir = tmp_path / "runs" / "run-a"
    progress_path = run_dir / "screening" / "progress.json"
    calls: list[str] = []

    def run_job(job: SignalJob):
        calls.append(job.key)
        return _write_job_artifacts(run_dir, job)

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
    state = json.loads(progress_path.read_text(encoding="utf-8"))["jobs"][jobs[0].key]
    assert state["checkpoint_schema_version"] == 1
    assert state["job_identity"] == {
        "run_id": "run-a",
        "job_key": jobs[0].key,
        "context_hash": "context",
    }
    assert state["prediction_path"] == (
        f"screening/predictions/{jobs[0].key}.parquet"
    )
    assert state["model_path"] == f"screening/models/{jobs[0].key}.pkl"
    assert state["diagnostics_path"] == (
        f"screening/diagnostics/signal__{jobs[0].quarter}__w756.json"
    )
    assert state["row_count"] == 1
    assert state["split"] == _job_split(jobs[0])
    assert state["artifact_schema"]["prediction"]["columns"] == [
        "datetime",
        "instrument",
        "score",
        "label",
        "benchmark_return",
        "market_state",
    ]
    assert len(state["prediction_sha256"]) == 64
    assert len(state["model_sha256"]) == 64
    assert len(state["diagnostics_sha256"]) == 64
    with pytest.raises(ValueError, match="resume_context_mismatch"):
        run_job_loop(
            jobs,
            progress_path=progress_path,
            context_hash="different",
            resume=True,
            run_job=run_job,
            pause_requested=lambda: False,
        )


def test_resume_reruns_job_when_model_artifact_is_tampered(tmp_path: Path) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    model_path = run_dir / job.phase / "models" / f"{job.key}.pkl"
    model_path.write_bytes(b"tampered-model")

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_model_artifact_is_empty(tmp_path: Path) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    model_path = run_dir / job.phase / "models" / f"{job.key}.pkl"
    model_path.write_bytes(b"")

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_checkpoint_uses_wrong_artifact_path(
    tmp_path: Path,
) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    wrong_path = tmp_path / "other" / "prediction.parquet"
    wrong_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "datetime": _job_split(job)["test_start"],
                "instrument": "SH600000",
                "score": 0.1,
                "label": 0.01,
                "benchmark_return": 0.0,
                "market_state": "neutral",
            }
        ]
    ).to_parquet(wrong_path, index=False)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["jobs"][job.key]["prediction_path"] = str(wrong_path)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_checkpoint_has_wrong_job_identity(
    tmp_path: Path,
) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["jobs"][job.key]["job_identity"] = {
        "run_id": "another-run",
        "job_key": job.key,
        "context_hash": "context",
    }
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_prediction_schema_is_wrong(tmp_path: Path) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    prediction_path = (
        run_dir / job.phase / "predictions" / f"{job.key}.parquet"
    )
    pd.DataFrame([{"datetime": _job_split(job)["test_start"], "score": 0.1}]).to_parquet(
        prediction_path, index=False
    )

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_prediction_date_exceeds_locked_split(
    tmp_path: Path,
) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    prediction_path = (
        run_dir / job.phase / "predictions" / f"{job.key}.parquet"
    )
    frame = pd.read_parquet(prediction_path)
    frame["datetime"] = "2026-01-02"
    frame.to_parquet(prediction_path, index=False)

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_resume_reruns_job_when_diagnostics_schema_is_wrong(tmp_path: Path) -> None:
    job = _job(1)
    run_dir = tmp_path / "runs" / "run-a"
    calls: list[str] = []
    progress_path, _ = _complete_job(run_dir=run_dir, job=job, calls=calls)
    diagnostics_path = (
        run_dir
        / job.phase
        / "diagnostics"
        / f"{job.signal_id}__{job.quarter}__w{job.window_days}.json"
    )
    diagnostics_path.write_text(
        json.dumps({"schema_version": 2, "raw": {}, "processed": {}}),
        encoding="utf-8",
    )

    run_job_loop(
        [job],
        progress_path=progress_path,
        context_hash="context",
        resume=True,
        run_job=lambda current: (
            calls.append(current.key) or _write_job_artifacts(run_dir, current)
        ),
        pause_requested=lambda: False,
    )

    assert calls == [job.key, job.key]


def test_research_resume_rejects_formal_smoke_mode_mismatch(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    run_dir = report_dir / "runs" / "run"
    run_dir.mkdir(parents=True)
    rebuilt = _signed_context(
        {
            "git_clean": True,
            "provider_uri": str(tmp_path / "provider"),
        }
    )
    (run_dir / "context.json").write_text(
        json.dumps(
            {
                **rebuilt,
                "run_mode": "smoke",
                "smoke_quarter": "2023Q1",
                "smoke_signal_id": "trend_momentum_excess_5d",
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("run_robust_signal_research.load_config", return_value={}),
        patch("run_robust_signal_research.build_context", return_value=rebuilt),
        pytest.raises(ValueError, match="resume_mode_mismatch"),
    ):
        run_research(
            config_path=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            run_id="run",
            resume=True,
            smoke=False,
            smoke_quarter="2023Q1",
            smoke_signal_id="trend_momentum_excess_5d",
            mlflow_enabled=False,
            **_frozen_readiness_kwargs(tmp_path),
        )


def test_smoke_resume_emits_the_stored_run_mode(tmp_path: Path) -> None:
    class StatusCaptured(RuntimeError):
        pass

    report_dir = tmp_path / "reports"
    run_dir = report_dir / "runs" / "run"
    run_dir.mkdir(parents=True)
    rebuilt = _signed_context(
        {
            "git_clean": True,
            "provider_uri": str(tmp_path / "provider"),
        }
    )
    stored = {
        **rebuilt,
        "run_mode": "smoke",
        "smoke_quarter": "2023Q1",
        "smoke_signal_id": "trend_momentum_excess_5d",
    }
    (run_dir / "context.json").write_text(json.dumps(stored), encoding="utf-8")
    captured: dict[str, object] = {}

    def capture_status(**kwargs):
        captured.update(kwargs["status"])
        raise StatusCaptured

    with (
        patch("run_robust_signal_research.load_config", return_value={}),
        patch("run_robust_signal_research.build_context", return_value=rebuilt),
        patch("run_robust_signal_research.QlibJobRunner"),
        patch("run_robust_signal_research.build_jobs", return_value=[]),
        patch(
            "run_robust_signal_research._write_run_status",
            side_effect=capture_status,
        ),
        pytest.raises(StatusCaptured),
    ):
        run_research(
            config_path=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            run_id="run",
            resume=True,
            smoke=True,
            smoke_quarter="2023Q1",
            smoke_signal_id="trend_momentum_excess_5d",
            mlflow_enabled=False,
            **_frozen_readiness_kwargs(tmp_path),
        )

    assert captured["run_mode"] == "smoke"


def test_resume_rejects_tampered_context_with_stale_hash(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    run_dir = report_dir / "runs" / "run"
    run_dir.mkdir(parents=True)
    rebuilt = _signed_context(
        {
            "git_clean": True,
            "provider_uri": str(tmp_path / "trusted-provider"),
            "provider_tree_sha256": "a" * 64,
            "cleaned_provider_identity": {"snapshot_id": "trusted"},
        }
    )
    stored = {
        **rebuilt,
        "provider_uri": str(tmp_path / "tampered-provider"),
        "run_mode": "smoke",
        "smoke_quarter": "2023Q1",
        "smoke_signal_id": "trend_momentum_excess_5d",
    }
    (run_dir / "context.json").write_text(json.dumps(stored), encoding="utf-8")

    with (
        patch("run_robust_signal_research.load_config", return_value={}),
        patch("run_robust_signal_research.build_context", return_value=rebuilt),
        pytest.raises(ValueError, match="resume_context_integrity_mismatch"),
    ):
        run_research(
            config_path=tmp_path / "config.yaml",
            quant_data_dir=tmp_path / "quant_data",
            report_dir=report_dir,
            run_id="run",
            resume=True,
            smoke=True,
            smoke_quarter="2023Q1",
            smoke_signal_id="trend_momentum_excess_5d",
            mlflow_enabled=False,
            **_frozen_readiness_kwargs(tmp_path),
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


def test_build_jobs_groups_signals_by_shared_handler_shape() -> None:
    config = {
        "signals": [
            {"id": "five_first", "horizon_days": 5, "feature_family": "full"},
            {"id": "ten", "horizon_days": 10, "feature_family": "full"},
            {"id": "five_second", "horizon_days": 5, "feature_family": "full"},
        ],
        "feature_families": {"full": {"include_handler_features": True}},
        "models": {
            "screening": [{"id": "ridge"}],
            "validation": [],
        },
        "periods": {
            "screening_quarters": ["2020Q1"],
            "screening_window_days": 756,
        },
    }

    jobs = build_jobs(config, phase="screening")

    assert [job.signal_id for job in jobs] == [
        "five_first",
        "five_second",
        "ten",
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


def test_smoke_failure_publishes_separate_status_without_replacing_formal(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "robust_signal_status_latest.json"
    latest.write_text(json.dumps({"run_id": "formal"}), encoding="utf-8")
    smoke_latest = tmp_path / "robust_signal_smoke_status_latest.json"

    _write_failed_status(
        run_id="smoke",
        context={"context_hash": "context", "git_commit": "commit"},
        report_dir=tmp_path,
        current_stage="screening",
        completed_count=0,
        expected_count=1,
        error=RuntimeError("smoke failed"),
        publish_latest=False,
        latest_status_path=smoke_latest,
    )

    assert json.loads(latest.read_text(encoding="utf-8")) == {"run_id": "formal"}
    assert json.loads(smoke_latest.read_text(encoding="utf-8"))["run_id"] == "smoke"
    assert json.loads(
        (tmp_path / "runs" / "smoke" / "status.json").read_text(encoding="utf-8")
    )["lifecycle_status"] == "failed"


def test_smoke_forces_mlflow_off_even_when_caller_enables_it(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    context = {
        "context_hash": "context",
        "git_commit": "commit",
        "git_clean": False,
        "provider_uri": str(tmp_path / "provider"),
    }
    job = SignalJob("screening", "signal", "equal_weight_rank", "2023Q1", 756)

    class FakeRunner:
        @staticmethod
        def split_contract(current: SignalJob) -> dict[str, str]:
            return _job_split(current)

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr("run_robust_signal_research.load_config", lambda _path: {})
    monkeypatch.setattr(
        "run_robust_signal_research.build_context", lambda **_kwargs: context
    )
    monkeypatch.setattr("run_robust_signal_research.QlibJobRunner", fake_runner)
    monkeypatch.setattr(
        "run_robust_signal_research.build_jobs", lambda *_args, **_kwargs: [job]
    )
    monkeypatch.setattr(
        "run_robust_signal_research.run_job_loop",
        lambda *_args, **_kwargs: {
            "pause_state": "paused",
            "completed_count": 0,
            "jobs": {},
        },
    )

    run_research(
        config_path=tmp_path / "config.yaml",
        quant_data_dir=tmp_path / "quant_data",
        report_dir=tmp_path / "reports",
        run_id="smoke",
        resume=False,
        smoke=True,
        smoke_quarter="2023Q1",
        smoke_signal_id="signal",
        mlflow_enabled=True,
        **_frozen_readiness_kwargs(tmp_path),
    )

    assert captured["mlflow_enabled"] is False


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

    class FakeRunner:
        @staticmethod
        def split_contract(current: SignalJob) -> dict[str, str]:
            return _job_split(current)

    monkeypatch.setattr(
        "run_robust_signal_research.build_context", lambda **_kwargs: context
    )
    monkeypatch.setattr(
        "run_robust_signal_research.QlibJobRunner", lambda **_kwargs: FakeRunner()
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
                    "prediction_path": (
                        f"{job.phase}/predictions/{job.key}.parquet"
                    ),
                    "model_path": f"{job.phase}/models/{job.key}.pkl",
                    "diagnostics_path": (
                        f"{job.phase}/diagnostics/"
                        f"{job.signal_id}__{job.quarter}__w{job.window_days}.json"
                    ),
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
            **_frozen_readiness_kwargs(tmp_path),
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
        "membership_intervals_sha256": "membership",
    }
    report = {"signal_id": "signal", "model_id": "ridge", "values": {"rank_ic": 0.02}}
    first = _factor_contract(config=config, context=context, report=report)
    second = _factor_contract(config=config, context=context, report=report)
    path = tmp_path / "factor_contract.json"

    write_immutable_factor_contract(path, first)

    assert first["contract_hash"] == second["contract_hash"]
    assert first["schema_version"] == 2
    assert first["membership_intervals_sha256"] == "membership"
    assert "membership_snapshot_hash" not in first
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
