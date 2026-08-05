import importlib.util
import json
import tempfile
from pathlib import Path


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "compare_training_equivalence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_training_equivalence",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def result(channel: str, run_id: str) -> dict:
    return {
        "status": "completed",
        "run_id": run_id,
        "experiment_id": "42",
        "experiment_name": "dsa_training_equivalence",
        "result_scope": "training_equivalence",
        "test_only": True,
        "equivalence_channel": channel,
        "provider_identity": {
            "provider_contract": "immutable_snapshot_v2",
            "dataset_id": "qlib_ashare_v2",
            "snapshot_id": "snapshot-1",
            "manifest_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        },
        "replay_contract": {
            "source_config_sha256": "c" * 64,
            "runtime_config_sha256": "d" * 64,
            "config_hash": "e" * 64,
            "periods": {
                "training": ["2024-01-02", "2024-06-28"],
                "validation": ["2024-07-01", "2024-09-30"],
                "test": ["2024-10-08", "2024-12-31"],
            },
        },
        "artifact_sha256": {
            "pred.pkl": "f" * 64,
            "label.pkl": "1" * 64,
        },
        "metrics": {
            "1day.excess_return_with_cost.annualized_return": 0.123456,
            "1day.excess_return_with_cost.information_ratio": 0.654321,
            "1day.excess_return_with_cost.max_drawdown": -0.12,
        },
    }


def test_identical_web_and_script_runs_pass_every_contract_check() -> None:
    module = load_module()

    report = module.compare_training_equivalence_results(
        result("web", "web-run"),
        result("script", "script-run"),
    )

    assert report["status"] == "pass"
    assert report["test_only"] is True
    assert report["web_run_id"] == "web-run"
    assert report["script_run_id"] == "script-run"
    assert all(check["passed"] for check in report["checks"])
    assert {check["id"] for check in report["checks"]} >= {
        "mlflow_ownership",
        "provider_identity",
        "config_hashes",
        "prediction_hash",
        "label_hash",
        "metrics_and_backtest",
        "bounded_period",
    }


def test_prediction_hash_mismatch_fails_equivalence() -> None:
    module = load_module()
    script = result("script", "script-run")
    script["artifact_sha256"]["pred.pkl"] = "9" * 64

    report = module.compare_training_equivalence_results(
        result("web", "web-run"),
        script,
    )

    prediction = next(
        check for check in report["checks"] if check["id"] == "prediction_hash"
    )
    assert report["status"] == "fail"
    assert prediction["passed"] is False


def test_matching_but_silently_extended_periods_fail_equivalence() -> None:
    module = load_module()
    web = result("web", "web-run")
    script = result("script", "script-run")
    for payload in (web, script):
        payload["replay_contract"]["periods"]["test"] = [
            "2024-10-08",
            "2026-07-23",
        ]

    report = module.compare_training_equivalence_results(web, script)

    bounded = next(
        check for check in report["checks"] if check["id"] == "bounded_period"
    )
    assert report["status"] == "fail"
    assert bounded["passed"] is False


def test_missing_or_malformed_evidence_cannot_pass_equivalence() -> None:
    module = load_module()
    web = result("web", "web-run")
    script = result("script", "script-run")
    for payload in (web, script):
        payload["provider_identity"] = {}
        payload["replay_contract"] = {
            "source_config_sha256": "",
            "runtime_config_sha256": "not-a-sha256",
            "config_hash": "",
            "periods": {
                "training": ["2024-01-02", "2024-06-28"],
                "validation": ["2024-07-01", "2024-09-30"],
                "test": ["2024-10-08", "2024-12-31"],
            },
        }
        payload["artifact_sha256"] = {
            "pred.pkl": "same-but-not-a-sha256",
            "label.pkl": "same-but-not-a-sha256",
        }
        payload["metrics"] = {}

    report = module.compare_training_equivalence_results(web, script)
    checks = {check["id"]: check["passed"] for check in report["checks"]}

    assert report["status"] == "fail"
    assert checks["provider_identity"] is False
    assert checks["config_hashes"] is False
    assert checks["prediction_hash"] is False
    assert checks["label_hash"] is False
    assert checks["metrics_and_backtest"] is False


def test_non_finite_or_non_numeric_metrics_fail_equivalence() -> None:
    module = load_module()
    web = result("web", "web-run")
    script = result("script", "script-run")
    web["metrics"] = {"annualized_return": float("nan"), "information_ratio": "0.5"}
    script["metrics"] = {"annualized_return": float("nan"), "information_ratio": "0.5"}

    report = module.compare_training_equivalence_results(web, script)
    metrics = next(
        check for check in report["checks"] if check["id"] == "metrics_and_backtest"
    )

    assert report["status"] == "fail"
    assert metrics["passed"] is False


def test_comparison_report_is_written_only_under_test_namespace() -> None:
    module = load_module()
    report = module.compare_training_equivalence_results(
        result("web", "web-run"),
        result("script", "script-run"),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        paths = module.write_comparison_reports(
            report,
            reports_root=root,
            timestamp="20260802_120000",
        )
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))

        assert paths["json"].is_relative_to(
            root / "test_runs" / "training_equivalence" / "comparisons"
        )
        assert paths["latest"].is_relative_to(
            root / "test_runs" / "training_equivalence"
        )
        assert payload["status"] == "pass"
        assert "测试数据" in paths["markdown"].read_text(encoding="utf-8")
