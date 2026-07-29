import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKFLOW_ROOT / "scripts" / "report_qlib_predictions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("report_qlib_predictions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_prediction_frame(path: Path) -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-07-01"), "SH600000"),
            (pd.Timestamp("2026-07-01"), "SZ000001"),
            (pd.Timestamp("2026-07-01"), "SZ000002"),
            (pd.Timestamp("2026-07-02"), "SH600000"),
            (pd.Timestamp("2026-07-02"), "SZ000001"),
            (pd.Timestamp("2026-07-02"), "SZ000002"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"score": [0.10, -0.20, 0.30, 0.05, -0.40, 0.25]}, index=index)
    frame.to_pickle(path)


class ReportQlibPredictionsTest(unittest.TestCase):
    def test_find_prediction_path_uses_latest_training_run_and_rejects_unsafe_run_id(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            pred_path = mlruns / "123" / "safe_run" / "artifacts" / "pred.pkl"
            pred_path.parent.mkdir(parents=True)
            pred_path.write_bytes(b"placeholder")
            latest = root / "latest_training_result.json"
            latest.write_text(json.dumps({"run_id": "safe_run"}), encoding="utf-8")

            found = module.find_prediction_path(latest_training_path=latest, mlruns_dir=mlruns)

            self.assertEqual(found, pred_path)
            with self.assertRaises(ValueError):
                module.find_prediction_path(latest_training_path=latest, mlruns_dir=mlruns, run_id="../unsafe")

    def test_find_prediction_path_does_not_fall_back_to_another_run(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            other_pred = mlruns / "123" / "other_run" / "artifacts" / "pred.pkl"
            other_pred.parent.mkdir(parents=True)
            other_pred.write_bytes(b"placeholder")
            latest = root / "latest_training_result.json"
            latest.write_text(json.dumps({"run_id": "missing_run"}), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "missing_run"):
                module.find_prediction_path(
                    latest_training_path=latest,
                    mlruns_dir=mlruns,
                )

    def test_prediction_provenance_requires_matching_native_v2_run_tags(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mlruns = root / "mlruns"
            run_dir = mlruns / "123" / "safe_run"
            pred_path = run_dir / "artifacts" / "pred.pkl"
            pred_path.parent.mkdir(parents=True)
            pred_path.write_bytes(b"placeholder")
            provider = root / "quant_data" / "qlib" / "snapshot-001"
            expected = {
                "provider_identity": {
                    "provider_contract": "immutable_snapshot_v2",
                    "dataset_id": "qlib_ashare_v2",
                    "snapshot_id": "snapshot-001",
                },
                "provider_path": str(provider),
            }

            with patch.object(
                module,
                "validate_file_store_run_artifact",
                return_value=expected,
            ) as validate:
                provenance = module.validate_prediction_provenance(
                    prediction_path=pred_path,
                    run_id="safe_run",
                    mlruns_dir=mlruns,
                    quant_data_root_path=root / "quant_data",
                )

            self.assertEqual(provenance, expected)
            validate.assert_called_once_with(
                artifact_path=pred_path,
                artifact_relative_path=Path("pred.pkl"),
                run_id="safe_run",
                mlruns_dir=mlruns,
                quant_data_root=root / "quant_data",
            )

            with (
                patch.object(
                    module,
                    "validate_file_store_run_artifact",
                    side_effect=module.RunProvenanceError("missing provider tag"),
                ),
                self.assertRaisesRegex(ValueError, "provider identity"),
            ):
                module.validate_prediction_provenance(
                    prediction_path=pred_path,
                    run_id="safe_run",
                    mlruns_dir=mlruns,
                    quant_data_root_path=root / "quant_data",
                )

    def test_generates_latest_prediction_summary_reports(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pred_path = root / "pred.pkl"
            out_dir = root / "reports"
            write_prediction_frame(pred_path)

            payload = module.generate_report(
                prediction_path=pred_path,
                out_dir=out_dir,
                run_id="run_1",
                experiment_name="baseline_lgb_alpha158_cleaned",
                topk=2,
                provenance={
                    "artifact_sha256": "a" * 64,
                    "provider_identity": {"dataset_id": "qlib_ashare_v2"},
                    "provider_path": str(root / "provider"),
                },
            )

            latest_json = out_dir / "qlib_predictions_latest.json"
            latest_md = out_dir / "qlib_predictions_latest.md"
            serialized = json.loads(latest_json.read_text(encoding="utf-8"))
            markdown = latest_md.read_text(encoding="utf-8")

        self.assertEqual(payload["mode"], "read_only_qlib_prediction_summary")
        self.assertEqual(payload["run_id"], "run_1")
        self.assertEqual(payload["artifact_sha256"], "a" * 64)
        self.assertEqual(payload["latest_date"], "2026-07-02")
        self.assertEqual([item["instrument"] for item in payload["top"]], ["SZ000002", "SH600000"])
        self.assertEqual([item["instrument"] for item in payload["bottom"]], ["SZ000001", "SH600000"])
        self.assertEqual(serialized["latest_date"], "2026-07-02")
        self.assertIn("Qlib Prediction Summary", markdown)
        self.assertIn("read-only", markdown)
        self.assertIn("不自动下单", markdown)

    def test_generates_empty_prediction_summary_when_pred_artifact_is_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            out_dir = root / "reports"

            payload = module.generate_report(
                prediction_path=root / "missing_pred.pkl",
                out_dir=out_dir,
                run_id="run_1",
                experiment_name="baseline_lgb_alpha158_cleaned",
                topk=3,
            )

            markdown = (out_dir / "qlib_predictions_latest.md").read_text(encoding="utf-8")

        self.assertEqual(payload["latest_date"], "n/a")
        self.assertEqual(payload["summary"]["rows"], 0)
        self.assertIn("missing_pred.pkl", payload["unavailable_reason"])
        self.assertIn("Qlib Prediction Summary", markdown)
        self.assertIn("不自动下单", markdown)

    def test_preflight_failure_does_not_read_unvalidated_prediction_artifact(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pred_path = root / "pred.pkl"
            pred_path.write_bytes(b"untrusted pickle")

            with patch.object(
                module,
                "load_prediction_scores",
                side_effect=AssertionError("unvalidated pickle was read"),
            ):
                payload = module.generate_report(
                    prediction_path=pred_path,
                    out_dir=root / "reports",
                    run_id="safe_run",
                    preflight_unavailable_reason="provider validation failed",
                )

        self.assertEqual(payload["summary"]["rows"], 0)
        self.assertEqual(payload["provenance_status"], "unavailable")
        self.assertEqual(payload["unavailable_reason"], "provider validation failed")

    def test_direct_report_call_refuses_existing_pickle_without_provenance(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pred_path = root / "pred.pkl"
            pred_path.write_bytes(b"untrusted pickle")

            with patch.object(
                module,
                "load_prediction_scores",
                side_effect=AssertionError("unvalidated pickle was read"),
            ) as load_scores:
                payload = module.generate_report(
                    prediction_path=pred_path,
                    out_dir=root / "reports",
                    run_id="safe_run",
                )

        self.assertEqual(payload["provenance_status"], "unavailable")
        self.assertIn("provenance was not validated", payload["unavailable_reason"])
        load_scores.assert_not_called()


if __name__ == "__main__":
    unittest.main()
