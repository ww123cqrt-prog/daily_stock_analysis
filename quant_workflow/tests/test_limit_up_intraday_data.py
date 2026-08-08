from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "limit_up_intraday_data.py"
    )
    assert path.exists(), "limit-up intraday data contract module is missing"
    spec = importlib.util.spec_from_file_location("limit_up_intraday_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bars_for_day(day: str = "2025-06-10") -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            f"{day} 09:31:00",
            f"{day} 14:00:00",
            f"{day} 14:01:00",
            f"{day} 14:30:00",
            f"{day} 15:00:00",
        ]
    )
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": [9.5, 10.0, 9.98, 10.0, 9.99],
            "high": [9.6, 10.0, 10.0, 10.0, 10.0],
            "low": [9.4, 10.0, 9.96, 9.98, 9.97],
            "close": [9.55, 10.0, 9.98, 10.0, 9.99],
            "volume": [1000.0, 100.0, 500.0, 200.0, 300.0],
            "money": [9550.0, 1000.0, 4990.0, 2000.0, 2997.0],
            "high_limit": 10.0,
            "low_limit": 8.2,
            "pre_close": 9.09,
        }
    )


def build_stage(tmp_path: Path, *, recall_status: str = "passed") -> Path:
    stage = tmp_path / "snapshot"
    bar_path = stage / "bars" / "2025" / "2025-06-10" / "SH600000.parquet"
    bar_path.parent.mkdir(parents=True)
    bars_for_day().to_parquet(bar_path, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "date": "2025-06-10",
                "relative_path": "bars/2025/2025-06-10/SH600000.parquet",
                "retrieval_ever_touched_limit": True,
                "final_status": "broken_board",
                "listing_date": "1999-11-10",
                "is_st": False,
                "board": "main",
                "listing_age_sessions": 5000,
                "next_date": "2025-06-11",
                "next_open": 10.2,
                "next_close": 10.4,
            }
        ]
    ).to_parquet(stage / "events.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-06-10",
                "decision_time": "14:00",
                "sealed_count": 12,
                "broken_board_rate": 0.25,
                "max_consecutive_limit_days": 4,
                "index_trend_20d": 0.03,
                "advance_decline_ratio": 1.2,
                "turnover_activity": 1.1,
                "regime": "normal",
            },
            {
                "date": "2025-06-10",
                "decision_time": "14:30",
                "sealed_count": 10,
                "broken_board_rate": 0.35,
                "max_consecutive_limit_days": 4,
                "index_trend_20d": 0.03,
                "advance_decline_ratio": 1.1,
                "turnover_activity": 1.2,
                "regime": "normal",
            },
        ]
    ).to_parquet(stage / "market_state.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-06-10",
                "decision_time": "14:00",
                "status": recall_status,
                "expected_count": 1,
                "recovered_count": 1 if recall_status == "passed" else 0,
                "missing_symbols": "[]" if recall_status == "passed" else '["SH600000"]',
            }
        ]
    ).to_parquet(stage / "recall_audit.parquet", index=False)
    return stage


def seal_test_snapshot(module, stage: Path) -> None:
    events = module._read_events(stage)
    recall = module._read_recall(stage)
    market_state = module._read_market_state(stage, events["date"])
    index = module._build_file_index(stage, events)
    index.to_parquet(stage / "FILE_INDEX.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "dataset_id": "ashare_limit_up_intraday_jqdata",
        "snapshot_id": "limit-up-jqdata-20250610-test",
        "data_kind": "limit_up_intraday_events",
        "role": "research_input",
        "status": "ready",
        "source": {"provider": "jqdata", "interface": "get_price_1m"},
        "source_retrieved_at": "2026-08-07T14:00:00+08:00",
        "price_basis": "raw",
        "frequency": "1m",
        "decision_times": ["14:00", "14:30"],
        "date_range": {"start": "2025-06-10", "end": "2025-06-10"},
        "event_count": int(len(events)),
        "broken_board_event_count": int(
            events["final_status"].astype(str).eq("broken_board").sum()
        ),
        "bar_row_count": int(index["row_count"].sum()),
        "events_path": "events.parquet",
        "recall_audit_path": "recall_audit.parquet",
        "market_state_path": "market_state.parquet",
        "file_index_path": "FILE_INDEX.parquet",
        "recall_audit": {
            "passed": True,
            "sample_count": int(len(recall)),
            "minimum_recall": 1.0,
        },
        "forbidden_model_features": list(module.FORBIDDEN_MODEL_FEATURES),
        "market_state": {
            "point_in_time": True,
            "decision_times": ["14:00", "14:30"],
            "row_count": int(len(market_state)),
        },
    }
    manifest["content_sha256"] = module._content_hash(stage, manifest)
    (stage / module.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_snapshot_round_trip_keeps_broken_boards_and_hashes_every_bar(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)

    seal_test_snapshot(module, stage)
    manifest = module.validate_intraday_snapshot(
        stage,
        required_start="2025-06-10",
        required_end="2025-06-10",
        verify_hashes=True,
    )

    assert manifest["status"] == "ready"
    assert manifest["event_count"] == 1
    assert manifest["broken_board_event_count"] == 1
    assert manifest["recall_audit"]["passed"] is True
    assert manifest["market_state"]["point_in_time"] is True
    assert manifest["market_state"]["decision_times"] == ["14:00", "14:30"]
    assert "retrieval_ever_touched_limit" in manifest["forbidden_model_features"]
    index = pd.read_parquet(stage / manifest["file_index_path"])
    assert index.iloc[0]["sha256"]


def test_snapshot_is_not_published_when_candidate_recall_misses_a_stock(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path, recall_status="blocked_data")

    with pytest.raises(module.IntradayDataBlocked, match="candidate recall"):
        seal_test_snapshot(module, stage)


def test_snapshot_with_short_history_is_blocked_for_formal_research(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)

    with pytest.raises(module.IntradayDataBlocked, match="does not cover"):
        module.validate_intraday_snapshot(
            stage,
            required_start="2018-01-01",
            required_end="2025-12-31",
        )


def test_snapshot_hash_validation_detects_bar_mutation(tmp_path: Path) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)
    path = stage / "bars" / "2025" / "2025-06-10" / "SH600000.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "close"] = 9.56
    changed.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        module.validate_intraday_snapshot(stage, verify_hashes=True)


def test_snapshot_rejects_manifest_paths_that_do_not_match_validated_files(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)
    manifest_path = stage / module.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events_path"] = "../outside-events.parquet"
    manifest["content_sha256"] = module._content_hash(stage, manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="events_path"):
        module.validate_intraday_snapshot(stage, verify_hashes=True)


def test_snapshot_rejects_incomplete_recall_evidence_even_when_marked_passed(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    audit_path = stage / "recall_audit.parquet"
    audit = pd.read_parquet(audit_path)
    audit.loc[0, "recovered_count"] = 0
    audit.loc[0, "missing_symbols"] = '["SH600000"]'
    audit.to_parquet(audit_path, index=False)

    with pytest.raises(module.IntradayDataBlocked, match="candidate recall"):
        seal_test_snapshot(module, stage)


def test_snapshot_rejects_recall_audit_without_any_positive_candidate(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    audit_path = stage / "recall_audit.parquet"
    audit = pd.read_parquet(audit_path)
    audit.loc[0, ["expected_count", "recovered_count"]] = 0
    audit.loc[0, "missing_symbols"] = "[]"
    audit.to_parquet(audit_path, index=False)

    with pytest.raises(module.IntradayDataBlocked, match="candidate recall"):
        seal_test_snapshot(module, stage)


def test_snapshot_rejects_manifest_date_range_that_exceeds_event_coverage(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)
    manifest_path = stage / module.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["date_range"] = {"start": "2018-01-01", "end": "2025-06-10"}
    manifest["content_sha256"] = module._content_hash(stage, manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="date range"):
        module.validate_intraday_snapshot(
            stage,
            required_start="2018-01-01",
            required_end="2025-06-10",
            verify_hashes=True,
        )


def test_snapshot_rejects_recall_audit_for_an_unrelated_event_date(
    tmp_path: Path,
) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)
    audit_path = stage / "recall_audit.parquet"
    audit = pd.read_parquet(audit_path)
    audit.loc[0, "date"] = "2025-06-11"
    audit.to_parquet(audit_path, index=False)
    manifest_path = stage / module.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["content_sha256"] = module._content_hash(stage, manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall audit date"):
        module.validate_intraday_snapshot(stage, verify_hashes=True)


def test_snapshot_rejects_manifest_with_unlocked_decision_times(tmp_path: Path) -> None:
    module = load_module()
    stage = build_stage(tmp_path)
    seal_test_snapshot(module, stage)
    manifest_path = stage / module.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_times"] = ["14:00"]
    manifest["content_sha256"] = module._content_hash(stage, manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="decision times"):
        module.validate_intraday_snapshot(stage, verify_hashes=True)
