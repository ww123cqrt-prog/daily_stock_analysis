"""Checks for the frozen TQM-20 comparison control."""

import json
from types import SimpleNamespace

from quant_workflow.scripts import compare_production_v2_tqm20 as comparison


def test_legacy_control_replay_checks_full_result(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "frozen-snapshot" / "qfq"
    old = {
        "script_sha256": "a" * 64, "metrics": {"final_nav": 1.0},
        "daily": [{"date": "2026-07-01"}],
        "snapshot_id": "frozen-snapshot", "data_path": str(snapshot),
    }
    runner = SimpleNamespace(OUT=None, REQUESTED_END=None)

    def run() -> None:
        (runner.OUT / "results.json").write_text(json.dumps(old), encoding="utf-8")

    runner.main = run
    monkeypatch.setattr(comparison, "_load_runner", lambda: runner)

    result = comparison._verify_legacy_control(old)

    assert result == {"status": "matched", "daily_rows": 1, "script_sha256": "a" * 64}
    assert runner.REQUESTED_END.isoformat() == "2026-08-21"
    assert runner.SNAPSHOT_ID == "frozen-snapshot"
    assert runner.SNAP == snapshot
