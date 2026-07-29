#!/usr/bin/env python3
"""Prepare a targeted Baostock refresh list from the latest anomaly report."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import reports_root, workflow_root  # noqa: E402


WORKFLOW = workflow_root()
REPORT_DIR = reports_root()
ANOMALY_STOCKS = REPORT_DIR / "data_anomaly_stocks_latest.csv"
SUMMARY_COLUMNS = ["symbol", "end", "severity"]


def normalize_symbol(value: str) -> str:
    raw = str(value).strip().upper()
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")):
        raw = raw[2:]
    if raw.isdigit() and len(raw) < 6:
        raw = raw.zfill(6)
    return raw


def select_refresh_symbols(summary: pd.DataFrame, target_end: pd.Timestamp) -> pd.DataFrame:
    selected = summary.copy()
    selected["symbol"] = selected["symbol"].map(normalize_symbol)
    selected["_end_date"] = pd.to_datetime(selected["end"], errors="coerce")
    selected["_missing_end"] = selected["_end_date"].isna()
    selected = selected[selected["_missing_end"] | (selected["_end_date"] < target_end)]
    selected = selected.sort_values(
        by=["_missing_end", "_end_date", "symbol"],
        ascending=[False, True, True],
        na_position="first",
    )
    return selected.drop(columns=["_end_date", "_missing_end"])


def write_latest(path: Path, latest_name: str) -> Path:
    latest = path.with_name(latest_name)
    shutil.copyfile(path, latest)
    return latest


def load_anomaly_summary(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            f"Missing anomaly stock report: `{path}`. Run `report_data_anomalies.py` first.",
        )

    summary = pd.read_csv(path, dtype={"symbol": str})
    missing = [column for column in SUMMARY_COLUMNS if column not in summary.columns]
    if "symbol" in missing:
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            f"Anomaly stock report is missing the required `symbol` column: `{path}`.",
        )

    if "end" in missing:
        summary["end"] = ""
    if "severity" in missing:
        summary["severity"] = "review"
    return summary, ""


def build_plan(
    selected: pd.DataFrame,
    target_end: pd.Timestamp,
    symbols_txt: Path,
    symbols_csv: Path,
    source_note: str = "",
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    oldest = selected["end"].replace("", pd.NA).dropna().min() if not selected.empty else ""
    newest = selected["end"].replace("", pd.NA).dropna().max() if not selected.empty else ""
    severity_counts = selected["severity"].value_counts().to_dict() if "severity" in selected else {}
    note_section = f"\n## Source Note\n\n{source_note}\n" if source_note else ""
    return f"""# Baostock Refresh Plan

Generated: {generated}

## Scope

- Target end date: {target_end.date()}
- Symbols to refresh: {len(selected)}
- Current selected end-date range: {oldest} ~ {newest}
- Severity counts: {severity_counts}
{note_section}

## Strategy

- Refresh source: Baostock daily qfq bars.
- Refresh mode: full-history re-download for selected symbols.
- Reason: 前复权 prices can shift after corporate actions; full-history refresh avoids mixing old qfq factors with newly appended rows.
- This list drives the legacy Baostock audit-cache downloader only; it does not publish formal V2 data.
- To update formal data, build a complete raw/factor/qfq bundle with `download_baostock_v2.py`, then run `quant_data_cli.py update ashare_daily_qfq --bundle-dir <verified-bundle>`.
- Do not run `merge_to_qlib.py` or refresh fixed-path compatibility output unless an explicitly identified old consumer still requires it.

## Artifacts

- Symbols TXT: `{symbols_txt}`
- Symbols CSV: `{symbols_csv}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly-stocks", type=Path, default=ANOMALY_STOCKS)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--target-end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    target_end = pd.Timestamp(args.target_end)
    summary, source_note = load_anomaly_summary(args.anomaly_stocks)
    selected = select_refresh_symbols(summary, target_end)
    if args.limit > 0:
        selected = selected.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbols_txt = args.out_dir / f"baostock_refresh_symbols_{ts}.txt"
    symbols_csv = args.out_dir / f"baostock_refresh_symbols_{ts}.csv"
    plan_path = args.out_dir / f"baostock_refresh_plan_{ts}.md"

    symbols_txt.write_text("\n".join(selected["symbol"].tolist()) + "\n", encoding="utf-8")
    selected.to_csv(symbols_csv, index=False)
    plan_path.write_text(build_plan(selected, target_end, symbols_txt, symbols_csv, source_note), encoding="utf-8")

    latest_txt = write_latest(symbols_txt, "baostock_refresh_symbols_latest.txt")
    latest_csv = write_latest(symbols_csv, "baostock_refresh_symbols_latest.csv")
    latest_plan = write_latest(plan_path, "baostock_refresh_plan_latest.md")

    print(f"Symbols: {symbols_txt}")
    print(f"CSV: {symbols_csv}")
    print(f"Plan: {plan_path}")
    print(f"Latest symbols: {latest_txt}")
    print(f"Latest CSV: {latest_csv}")
    print(f"Latest plan: {latest_plan}")
    print(f"Count: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
