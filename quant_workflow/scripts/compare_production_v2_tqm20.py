"""Replay the frozen TQM-20 interval against strict production.v2 ETF bars."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

DSA_ROOT = Path(__file__).resolve().parents[2]
if str(DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_ROOT))

from quant_workflow.scripts.quant_data_service_client import (  # noqa: E402
    DEFAULT_QUANT_DATA_SERVICE_URL,
    QuantDataClientError,
    QuantDataServiceClient,
)

OLD_DIR = DSA_ROOT / "quant_workflow/reports/experiments/tqm20_period_20260701_20260821"
RUNNER = OLD_DIR.parent / "tqm20_period_20260701_20260814/run_backtest.py"
SYMBOLS = ("SZ159915", "SH513100", "SH512890")
WARMUP_START = "2026-04-01"
END = "2026-08-21"


def _write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_runner():
    spec = importlib.util.spec_from_file_location("frozen_tqm20_interval", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen TQM-20 runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_legacy_control(old: dict) -> dict:
    runner = _load_runner()
    snapshot = Path(str(old["data_path"]))
    snapshot_id = str(old["snapshot_id"])
    if snapshot.name != "qfq" or snapshot.parent.name != snapshot_id:
        raise ValueError("frozen TQM-20 snapshot path does not match its identity")
    with tempfile.TemporaryDirectory(prefix="tqm20-legacy-control-") as directory:
        runner.OUT = Path(directory)
        runner.SNAP = snapshot
        runner.SNAPSHOT_ID = snapshot_id
        runner.REQUESTED_END = date.fromisoformat(END)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        replay = json.loads((runner.OUT / "results.json").read_text(encoding="utf-8"))
    result = {
        "status": "matched" if replay == old else "different",
        "daily_rows": len(replay.get("daily", [])),
        "script_sha256": replay.get("script_sha256"),
    }
    if replay != old:
        result["different_keys"] = [
            key for key in sorted(set(old) | set(replay)) if old.get(key) != replay.get(key)
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or (
        DSA_ROOT / "quant_workflow/reports/experiments"
        / f"tqm20_production_v2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    output.mkdir(parents=True, exist_ok=False)
    old_path = OLD_DIR / "results.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    report: dict = {
        "comparison": "TQM-20 V1.6 frozen interval",
        "old_result": str(old_path),
        "old_result_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "period": {"start": "2026-07-01", "end": END},
        "warmup_start": WARMUP_START,
        "symbols": list(SYMBOLS),
        "old_metrics": old["metrics"],
    }
    try:
        report["old_control"] = _verify_legacy_control(old)
    except Exception as exc:
        report["status"] = "old_control_unavailable"
        report["old_control_error"] = str(exc)
        _write_report(output / "comparison.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
        return 2
    if report["old_control"]["status"] != "matched":
        report["status"] = "old_control_mismatch"
        _write_report(output / "comparison.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False))
        return 1
    try:
        with QuantDataServiceClient(
            base_url=os.getenv("QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL)
        ) as client:
            release = client.native_ready()
            dataset = client.native_dataset("cn_etf_daily", expected_release=release)
            report["release"] = release.as_dict()
            report["module"] = dataset.identity.as_dict()
            frames: dict[str, pd.DataFrame] = {}
            failures: list[dict] = []
            for symbol in SYMBOLS:
                try:
                    queried = client.native_bars_batch(
                        dataset.identity, [symbol], basis="qfq", fields=("close",),
                        start=WARMUP_START, end=END,
                    )
                except QuantDataClientError as exc:
                    failures.append({
                        "symbol": symbol, "code": exc.code,
                        "http_status": exc.status_code,
                        "failed_symbols": exc.details.get("failed_symbols", []),
                        "uncovered": exc.details.get("uncovered", []),
                    })
                    continue
                frame = queried.frame[["date", "close"]].copy()
                frame["date"] = pd.to_datetime(frame["date"]).dt.date
                frames[symbol] = (
                    frame.sort_values("date").drop_duplicates("date").set_index("date")
                )
            if failures:
                report["status"] = "blocked_by_quant_data"
                report["failed_queries"] = failures
                _write_report(output / "comparison.json", report)
                print(json.dumps({"status": report["status"], "output": str(output), "failed_queries": failures}, ensure_ascii=False))
                return 2
    except QuantDataClientError as exc:
        details = exc.details
        report["status"] = "blocked_by_quant_data"
        report["error"] = {
            "code": exc.code,
            "http_status": exc.status_code,
            "message": str(exc),
            "failed_symbols": details.get("failed_symbols", []),
            "uncovered": details.get("uncovered", []),
        }
        _write_report(output / "comparison.json", report)
        print(json.dumps({"status": report["status"], "output": str(output), "error": report["error"]}, ensure_ascii=False))
        return 2

    runner = _load_runner()
    runner.OUT = output
    runner.SNAPSHOT_ID = f"production.v2:{release.release_set_sha256}"
    runner.SNAP = "quant_data read service"
    runner.REQUESTED_END = date.fromisoformat(END)
    runner.load_frames = lambda: frames
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    current = json.loads((output / "results.json").read_text(encoding="utf-8"))
    old_days = {item["date"]: item for item in old["daily"]}
    current_days = {item["date"]: item for item in current["daily"]}
    different_days = [
        day for day in sorted(set(old_days) | set(current_days))
        if day not in old_days or day not in current_days
        or abs(float(current_days[day]["nav"]) - float(old_days[day]["nav"])) > 0.01
        or current_days[day]["position"] != old_days[day]["position"]
        or current_days[day]["trades"] != old_days[day]["trades"]
    ]
    report.update({
        "status": "matched" if current["metrics"] == old["metrics"] and not different_days else "different",
        "new_metrics": current["metrics"],
        "metric_differences": {
            key: {"old": old["metrics"].get(key), "new": current["metrics"].get(key)}
            for key in sorted(set(old["metrics"]) | set(current["metrics"]))
            if old["metrics"].get(key) != current["metrics"].get(key)
        },
        "old_trading_days": len(old_days),
        "new_trading_days": len(current_days),
        "different_dates": different_days,
    })
    _write_report(output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "output": str(output), "metric_differences": report["metric_differences"], "different_dates": different_days}, ensure_ascii=False))
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
