"""Frozen QMT inputs drive status and approved tasks without a live SDK."""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.v1.endpoints import data_status, quant_jobs, research_status
from src.services import quant_job_service
from src.services.quant_job_service import QuantJobService


@pytest.fixture
def qmt_input(tmp_path, monkeypatch):
    from quant_qmt_data import save_dataset

    dates = pd.bdate_range("2021-01-04", "2024-12-31")
    frame = pd.DataFrame({
        "date": dates, "symbol": "SH600519", "open": 100.0, "high": 102.0,
        "low": 99.0, "close": 101.0, "volume": 10000.0, "amount": 1000000.0,
    })
    frame.attrs = {
        "provider": "qmt", "request": {"symbols": ["SH600519"]},
        "price_basis": "front", "volume_units": {"SH600519": "shares"},
        "missing_symbols": [], "read_at": "2026-10-28T10:00:00+08:00",
        "completed_through": "2026-10-27", "contains_partial": False,
        "qmt_data_dir": str(tmp_path / "private-qmt-cache"),
        "sdk_version": "test-sdk", "actual_start": "2021-01-04",
        "actual_end": "2024-12-31",
    }
    directory = save_dataset(
        frame, tmp_path / "private-frozen-input",
        calendar=dates, universe=["SH600519"],
    )
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    workflow = tmp_path / "workflow"
    (workflow / "configs").mkdir(parents=True)
    (workflow / "configs" / "baseline_qmt_native.yaml").write_text(
        "task:\n  dataset:\n    kwargs:\n      handler:\n        kwargs: {}\n      segments:\n"
        "        train: ['2021-01-01', '2022-12-31']\n"
        "        valid: ['2023-01-01', '2023-12-31']\n"
        "        test: ['2024-01-01', '2024-12-31']\n", encoding="utf-8",
    )
    monkeypatch.setattr(research_status, "QUANT_WORKFLOW_ROOT", workflow)
    monkeypatch.setattr(quant_job_service, "QUANT_WORKFLOW_ROOT", workflow)
    return directory


def test_ready_input_status_is_offline_and_exposes_only_public_metadata(qmt_input, tmp_path):
    with patch("quant_qmt_data.QMTClient.from_env", side_effect=AssertionError("No SDK")):
        with TestClient(create_app(static_dir=tmp_path / "static")) as client:
            response = client.get("/api/v1/data/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "qmt_native"
    assert payload["status"] == "ready"
    assert payload["sdk"]["status"] == "not_checked"
    assert payload["experiment"]["experiment_id"]
    assert payload["experiment"]["price_basis"] == "front"
    assert payload["experiment"]["calendar_end"] == "2024-12-31"
    assert payload["experiment"]["symbol_count"] == 1
    assert "private-" not in response.text
    assert "input_dir" not in response.text


@pytest.mark.parametrize("damage", ["json", "hash", "partial"])
def test_incomplete_or_changed_input_is_not_trainable(qmt_input, damage):
    path = qmt_input / "dataset.json"
    if damage == "json":
        path.unlink()
    elif damage == "hash":
        with (qmt_input / "bars.parquet").open("ab") as stream:
            stream.write(b"changed")
    else:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["contains_partial"] = True
        path.write_text(json.dumps(metadata), encoding="utf-8")
    payload = research_status.build_research_status_payload(provider="qmt_native")
    assert payload["trainable"] is False
    assert payload["training_preflight"]["status"] == "blocked"
    assert "private-" not in json.dumps(payload)


def test_preflight_rejects_overlap_without_a_complete_label_window(qmt_input, tmp_path):
    config = tmp_path / "too-short.yaml"
    config.write_text(
        "task:\n  dataset:\n    kwargs:\n      handler:\n        kwargs: {}\n      segments:\n"
        "        train: ['2022-12-30', '2022-12-30']\n"
        "        valid: ['2023-01-01', '2023-12-31']\n"
        "        test: ['2024-01-01', '2024-12-31']\n", encoding="utf-8",
    )
    payload = research_status.build_research_status_payload(provider="qmt_native", training_config=config)
    assert payload["trainable"] is False
    assert payload["training_preflight"]["status"] == "blocked"


def test_experiment_notebook_hides_internal_paths_at_public_boundary(qmt_input, tmp_path, monkeypatch):
    private = {"runs": [{"input_dir": str(qmt_input),
        "runtime_config": "D:/private-source/runtime/model.yaml"}]}
    monkeypatch.setattr(research_status, "_experiment_notebook_payload", lambda _: private)
    with TestClient(create_app(static_dir=tmp_path / "static")) as client:
        response = client.get("/api/v1/research/experiments")
    assert response.status_code == 200
    assert "input_dir" not in response.text
    assert "private-source" not in response.text


def test_native_preflight_needs_no_legacy_catalog_or_ready(qmt_input):
    with patch.object(research_status, "_resolve_native_qlib_provider", side_effect=AssertionError("Legacy")):
        payload = research_status.build_research_status_payload(provider="qmt_native")
    assert payload["trainable"] is True
    assert payload["training_preflights"]["qmt_native"]["status"] == "passed"
    identity = payload["current_provider_identities"]["qmt_native"]
    assert identity["experiment_id"]
    assert identity["bars_sha256"]
    assert "snapshot_id" not in identity
    assert "private-" not in json.dumps(payload)


def test_native_training_submission_pins_server_input(qmt_input, tmp_path):
    service = QuantJobService(jobs_root=tmp_path / "jobs")
    with patch.object(service, "_start_job"), patch.object(quant_jobs, "get_quant_job_service", return_value=service):
        with TestClient(create_app(static_dir=tmp_path / "static"), headers={"Origin": "http://127.0.0.1:8010"}) as client:
            response = client.post("/api/v1/quant/jobs", json={
                "job_type": "train_baseline_lgb", "params": {"provider": "qmt_native"},
            })
    assert response.status_code == 200, response.text
    job = service.get_job(response.json()["job_id"])
    assert job.command[job.command.index("--provider") + 1] == "qmt_native"
    assert job.command[job.command.index("--config") + 1].endswith("baseline_qmt_native.yaml")
    approved = json.loads(job.command[job.command.index("--provider-identity-json") + 1])
    assert approved["backend"] == "qmt_native"
    assert approved["input_dir"] == str(qmt_input.resolve())
    assert approved["experiment_id"]
    assert approved["bars_sha256"]
    assert "--no-publish-latest" in job.command
    assert "private-" not in response.text


def test_qmt_env_reaches_child_without_arbitrary_secrets(monkeypatch):
    source = {"QMT_DATASET_DIR": "input", "QMT_PYTHON": "sdk-python", "QMT_PORT": "58610",
              "QMT_DATA_DIR": "cache", "QMT_TIMEOUT": "30", "PRIVATE_TOKEN": "secret"}
    for name, value in source.items():
        monkeypatch.setenv(name, value)
    child = quant_job_service._quant_job_child_env({})
    assert {name: child.get(name) for name in source if name.startswith("QMT_")} == {
        name: value for name, value in source.items() if name.startswith("QMT_")
    }
    assert "PRIVATE_TOKEN" not in child


def test_retired_mutations_are_not_registered(tmp_path):
    service = QuantJobService(jobs_root=tmp_path / "jobs")
    assert not {
        "update_ashare_miniqmt", "download_etf_momentum", "merge_to_qlib",
        "apply_data_cleaning", "repair_qlib_provider", "validate_latest_snapshot",
    }.intersection(service.available_job_types())
    assert "train_baseline_lgb" in service.available_job_types()
    assert "report_qlib_model_comparison" in service.available_job_types()


def test_training_preflight_checks_selected_config_dates(qmt_input, tmp_path):
    config = tmp_path / "future-training.yaml"
    config.write_text("task:\n  dataset:\n    kwargs:\n      segments:\n        train: ['2030-01-01', '2030-12-31']\n        valid: ['2031-01-01', '2031-06-30']\n        test: ['2031-07-01', '2031-12-31']\n", encoding="utf-8")
    payload = research_status.build_research_status_payload(provider="qmt_native", training_config=config)
    assert payload["trainable"] is False
    assert "private-" not in json.dumps(payload)


def test_old_status_artifact_cannot_approve_a_broken_frozen_input(qmt_input, tmp_path, monkeypatch):
    saved_status = tmp_path / "research_status_latest.json"
    saved_status.write_text(json.dumps({"trainable": True}), encoding="utf-8")
    monkeypatch.setattr(research_status, "RESEARCH_STATUS_PATH", saved_status)
    (qmt_input / "dataset.json").unlink()
    payload = research_status.build_research_status_payload()
    assert payload["trainable"] is False
    assert payload["training_preflight"]["status"] == "blocked"


@pytest.mark.parametrize("same_input", [False, True])
def test_saved_run_is_matched_only_by_its_frozen_input(qmt_input, tmp_path, monkeypatch, same_input):
    monkeypatch.setattr(research_status, "RESEARCH_STATUS_PATH", tmp_path / "status.json")
    metadata = json.loads((qmt_input / "dataset.json").read_text(encoding="utf-8"))
    (tmp_path / "latest_training_result.json").write_text(json.dumps({
        "status": "completed", "run_id": "historical-run",
        "provider_identity": {"backend": "qmt_native", "experiment_id": metadata["experiment_id"] if same_input else "different",
            "bars_sha256": metadata["bars_sha256"] if same_input else "b" * 64,
            "inputs_sha256": metadata["inputs_sha256"], "price_basis": "front", "input_dir": "private-location"},
    }), encoding="utf-8")
    payload = research_status.build_research_status_payload()
    assert payload["latest_training"]["provider_current"] is same_input
    assert payload["latest_training"]["run_id"] == "historical-run"
    assert "input_dir" not in json.dumps(payload)
