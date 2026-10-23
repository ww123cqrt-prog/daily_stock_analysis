from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_workflow.scripts import build_direct_cleaning_plan as module
from quant_workflow.scripts.build_direct_cleaning_plan import build_cleaning_plan


@pytest.mark.parametrize("windows", [False, True])
def test_build_cleaning_plan_freezes_anomaly_rules(tmp_path: Path, monkeypatch, windows) -> None:
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

    modes, flushed = {}, []
    real_open, real_fsync, real_replace = Path.open, os.fsync, os.replace

    def record_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        modes[handle.fileno()] = mode
        return handle

    def flush_file(fd):
        if windows:
            assert modes[fd] == "r+b", "Windows file fsync needs a writable handle"
        flushed.append(fd)
        return real_fsync(fd)

    def replace_file(source, target):
        assert flushed, "the temporary plan must be fsynced before replacement"
        return real_replace(source, target)

    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(
        module, "os", SimpleNamespace(**{
            **vars(os), "name": "nt" if windows else os.name,
            "fsync": flush_file, "replace": replace_file,
        })
    )
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
    assert len(flushed) == 1
    assert not list(tmp_path.glob(".CLEANING_PLAN.json.tmp-*"))


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_cleaning_plan_write_failure_preserves_existing_plan(tmp_path, monkeypatch, operation):
    stocks, events, output = (tmp_path / name for name in ("stocks.csv", "events.csv", "CLEANING_PLAN.json"))
    stocks.write_text("symbol,severity,cleaning_action\nSH600000,review,keep_but_mask_or_impute_volume\n")
    events.write_text("symbol,issue_type,sample_dates\nSH600000,abs_return_gt_50pct,2026-01-06\n")
    output.write_bytes(b"previous plan")

    def fail(*_args):
        raise OSError(f"{operation} failed")

    monkeypatch.setattr(module, "os", SimpleNamespace(**{**vars(os), operation: fail}))
    with pytest.raises(OSError, match=f"{operation} failed"):
        build_cleaning_plan(
            stocks_csv=stocks, events_csv=events, output_path=output,
            source_snapshot={"dataset_id": "ashare_daily_qfq", "snapshot_id": "qfq-1"},
        )
    assert output.read_bytes() == b"previous plan"
    assert not list(tmp_path.glob(".CLEANING_PLAN.json.tmp-*"))
