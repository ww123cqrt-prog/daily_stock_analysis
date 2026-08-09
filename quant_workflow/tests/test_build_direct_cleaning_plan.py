from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_workflow.scripts.build_direct_cleaning_plan import build_cleaning_plan


def test_build_cleaning_plan_freezes_anomaly_rules(tmp_path: Path) -> None:
    stocks = tmp_path / "stocks.csv"
    events = tmp_path / "events.csv"
    output = tmp_path / "CLEANING_PLAN.json"
    pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "severity": "review",
                "cleaning_action": "keep_but_mask_or_impute_volume",
            },
            {
                "symbol": "SZ000001",
                "severity": "blocker",
                "cleaning_action": "exclude_symbol_from_training_until_repaired",
            },
        ]
    ).to_csv(stocks, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "SH600000",
                "issue_type": "abs_return_gt_50pct",
                "sample_dates": "2026-01-06,2026-01-07",
            }
        ]
    ).to_csv(events, index=False)

    result = build_cleaning_plan(
        stocks_csv=stocks,
        events_csv=events,
        output_path=output,
        source_snapshot={"dataset_id": "ashare_daily_qfq", "snapshot_id": "qfq-1"},
    )

    assert result == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["excluded_symbols"] == ["SZ000001"]
    assert payload["volume_mask_symbols"] == ["SH600000"]
    assert payload["return_mask_symbols"] == ["SH600000"]
    assert payload["event_dates"]["SH600000"] == ["2026-01-06", "2026-01-07"]
