#!/usr/bin/env python3
"""Apply anomaly-derived cleaning rules and build a cleaned Qlib provider."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import (  # noqa: E402
    ashare_qfq_bars_dir,
    qlib_native_snapshot_uri,
    quant_data_root,
    reports_root,
    workflow_root,
)
from merge_to_qlib import QLIB_FIELDS, convert_to_bin  # noqa: E402


QUANT_DATA = quant_data_root()
WORKFLOW = workflow_root()
MERGED_DIR: Path | None = None
CLEANED_MERGED_DIR = (
    QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
)
CANONICAL_QLIB_DIR: Path | None = None
CLEANED_QLIB_DIR: Path | None = None
REPORT_DIR = reports_root()
ANOMALY_STOCKS_CSV = REPORT_DIR / "data_anomaly_stocks_latest.csv"
ANOMALY_EVENTS_CSV = REPORT_DIR / "data_anomaly_events_latest.csv"

BENCHMARK_SYMBOLS = {
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
}
EXCLUDE_ACTIONS = {"exclude_symbol_from_training_until_repaired"}
VOLUME_MASK_ACTIONS = {"keep_but_mask_or_impute_volume"}
RETURN_MASK_ACTIONS = {
    "review_and_winsorize_or_mask_event_dates",
    "winsorize_returns_and_mark_event_dates",
}
RETURN_EVENT_TYPES = {"abs_return_gt_50pct", "abs_return_gt_30pct"}
VOLUME_EVENT_TYPES = {"non_positive_volume"}
EVENT_MASK_FIELDS = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "turn",
    "pctChg",
]


class CleaningPlan:
    def __init__(
        self,
        *,
        excluded_symbols: set[str] | None = None,
        volume_mask_symbols: set[str] | None = None,
        event_dates_by_symbol: dict[str, set[pd.Timestamp]] | None = None,
        return_mask_symbols: set[str] | None = None,
    ) -> None:
        self.excluded_symbols = excluded_symbols or set()
        self.volume_mask_symbols = volume_mask_symbols or set()
        self.event_dates_by_symbol = event_dates_by_symbol or {}
        self.return_mask_symbols = return_mask_symbols or set()


class CleaningStats:
    def __init__(
        self,
        *,
        symbol: str,
        source_rows: int,
        output_rows: int = 0,
        excluded: bool = False,
        masked_volume_rows: int = 0,
        masked_event_rows: int = 0,
    ) -> None:
        self.symbol = symbol
        self.source_rows = int(source_rows)
        self.output_rows = int(output_rows)
        self.excluded = bool(excluded)
        self.masked_volume_rows = int(masked_volume_rows)
        self.masked_event_rows = int(masked_event_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "excluded": self.excluded,
            "masked_volume_rows": self.masked_volume_rows,
            "masked_event_rows": self.masked_event_rows,
        }


def normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def parse_sample_dates(value: Any) -> set[pd.Timestamp]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    dates: set[pd.Timestamp] = set()
    for raw in str(value).replace(";", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = pd.to_datetime(item, errors="coerce")
        if not pd.isna(parsed):
            dates.add(pd.Timestamp(parsed).normalize())
    return dates


def load_cleaning_plan(stocks_csv: Path, events_csv: Path) -> CleaningPlan:
    stocks = (
        pd.read_csv(stocks_csv, dtype={"symbol": str})
        if stocks_csv.exists()
        else pd.DataFrame()
    )
    events = (
        pd.read_csv(events_csv, dtype={"symbol": str})
        if events_csv.exists()
        else pd.DataFrame()
    )
    excluded_symbols: set[str] = set()
    volume_mask_symbols: set[str] = set()
    return_mask_symbols: set[str] = set()
    event_dates_by_symbol: dict[str, set[pd.Timestamp]] = {}

    if not stocks.empty:
        for _, row in stocks.iterrows():
            symbol = normalize_symbol(row.get("symbol"))
            if not symbol:
                continue
            action = str(row.get("cleaning_action") or "").strip()
            severity = str(row.get("severity") or "").strip().lower()
            if action in EXCLUDE_ACTIONS or severity == "blocker":
                excluded_symbols.add(symbol)
            if action in VOLUME_MASK_ACTIONS:
                volume_mask_symbols.add(symbol)
            if action in RETURN_MASK_ACTIONS:
                return_mask_symbols.add(symbol)

    if not events.empty:
        for _, row in events.iterrows():
            symbol = normalize_symbol(row.get("symbol"))
            issue_type = str(row.get("issue_type") or "").strip()
            if not symbol:
                continue
            if issue_type in VOLUME_EVENT_TYPES:
                volume_mask_symbols.add(symbol)
            if issue_type in RETURN_EVENT_TYPES:
                event_dates_by_symbol.setdefault(symbol, set()).update(
                    parse_sample_dates(row.get("sample_dates"))
                )
                return_mask_symbols.add(symbol)

    return CleaningPlan(
        excluded_symbols=excluded_symbols,
        volume_mask_symbols=volume_mask_symbols,
        event_dates_by_symbol=event_dates_by_symbol,
        return_mask_symbols=return_mask_symbols,
    )


def detect_large_return_dates(
    frame: pd.DataFrame, threshold: float = 0.50
) -> set[pd.Timestamp]:
    if "close" not in frame.columns or "date" not in frame.columns:
        return set()
    ordered = frame.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered["_close"] = pd.to_numeric(ordered["close"], errors="coerce")
    ordered = ordered.dropna(subset=["date"]).sort_values("date")
    abs_return = (
        ordered["_close"]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .abs()
    )
    dates = ordered.loc[abs_return > threshold, "date"]
    return {pd.Timestamp(value).normalize() for value in dates.dropna()}


def clean_frame(
    symbol: str, frame: pd.DataFrame, plan: CleaningPlan
) -> tuple[pd.DataFrame | None, CleaningStats]:
    normalized_symbol = normalize_symbol(symbol)
    stats = CleaningStats(symbol=normalized_symbol, source_rows=len(frame))
    if normalized_symbol in plan.excluded_symbols:
        stats.excluded = True
        return None, stats

    cleaned = frame.copy()
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned = (
            cleaned.dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(subset=["date"])
        )
    if "factor" not in cleaned.columns:
        cleaned["factor"] = 1.0

    masked_volume_rows = pd.Series(False, index=cleaned.index)
    if normalized_symbol in plan.volume_mask_symbols and "volume" in cleaned.columns:
        volume = pd.to_numeric(cleaned["volume"], errors="coerce")
        masked_volume_rows = volume <= 0
        cleaned.loc[masked_volume_rows.fillna(False), "volume"] = np.nan

    event_dates = set(plan.event_dates_by_symbol.get(normalized_symbol, set()))
    if normalized_symbol in plan.return_mask_symbols:
        event_dates.update(detect_large_return_dates(cleaned))
    normalized_dates = (
        cleaned["date"].dt.normalize()
        if "date" in cleaned.columns
        else pd.Series([], dtype="datetime64[ns]")
    )
    masked_event_rows = (
        normalized_dates.isin(event_dates)
        if event_dates
        else pd.Series(False, index=cleaned.index)
    )
    fields_to_mask = [field for field in EVENT_MASK_FIELDS if field in cleaned.columns]
    if fields_to_mask and not masked_event_rows.empty:
        cleaned.loc[masked_event_rows.fillna(False), fields_to_mask] = np.nan

    stats.output_rows = len(cleaned)
    stats.masked_volume_rows = int(masked_volume_rows.fillna(False).sum())
    stats.masked_event_rows = int(masked_event_rows.fillna(False).sum())
    return cleaned, stats


def build_cleaned_parquet(
    src_dir: Path,
    dst_dir: Path,
    plan: CleaningPlan,
    overwrite: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        overwrite = False
    if overwrite and dst_dir.exists():
        shutil.rmtree(dst_dir)
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(src_dir.glob("*.parquet"))
    stats_rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        frame = pd.read_parquet(path)
        cleaned, stats = clean_frame(path.stem, frame, plan)
        stats_rows.append(stats.to_dict())
        if cleaned is not None and not cleaned.empty and not dry_run:
            output_path = dst_dir / path.name
            if output_path.exists() and not overwrite:
                stats_rows[-1]["skipped_existing"] = True
                if index <= 3 or index % 1000 == 0 or index == len(paths):
                    print(f"  skipped existing {index}/{len(paths)}: {path.stem}")
                continue
            keep = ["date"] + [
                field
                for field in QLIB_FIELDS + ["amount", "turn", "pctChg"]
                if field in cleaned.columns
            ]
            cleaned[keep].to_parquet(output_path, index=False)
            stats_rows[-1]["written"] = True
        if index <= 3 or index % 1000 == 0 or index == len(paths):
            action = "would clean" if dry_run else "cleaned"
            print(f"  {action} {index}/{len(paths)}: {path.stem}")

    total_source_rows = sum(row["source_rows"] for row in stats_rows)
    total_output_rows = sum(row["output_rows"] for row in stats_rows)
    summary = {
        "source_dir": str(src_dir),
        "cleaned_dir": str(dst_dir),
        "dry_run": bool(dry_run),
        "source_symbols": len(paths),
        "excluded_symbols": sum(1 for row in stats_rows if row["excluded"]),
        "would_write_symbols": sum(1 for row in stats_rows if not row["excluded"]),
        "skipped_existing_symbols": sum(
            1 for row in stats_rows if row.get("skipped_existing")
        ),
        "written_symbols": sum(1 for row in stats_rows if row.get("written")),
        "source_rows": total_source_rows,
        "output_rows": total_output_rows,
        "masked_volume_rows": sum(row["masked_volume_rows"] for row in stats_rows),
        "masked_event_rows": sum(row["masked_event_rows"] for row in stats_rows),
        "stats": stats_rows,
    }
    return summary


def write_latest(path: Path, latest_name: str) -> Path:
    latest = path.with_name(latest_name)
    shutil.copyfile(path, latest)
    return latest


def remove_symbol_from_instrument_file(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(f"{symbol}\t")
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_qlib_calendar(provider: Path) -> list[str]:
    path = provider / "calendars" / "day.txt"
    days = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not days:
        raise ValueError(f"empty Qlib calendar: {path}")
    return days


def align_feature_bin(source_path: Path, source_calendar: list[str], target_calendar: list[str]) -> np.ndarray:
    raw = np.fromfile(source_path, dtype="<f4")
    if len(raw) < 2:
        raise ValueError(f"invalid Qlib feature bin: {source_path}")
    start_idx = int(raw[0])
    values = raw[1:]
    if start_idx < 0 or start_idx + len(values) > len(source_calendar):
        raise ValueError(f"Qlib feature bin is outside its source calendar: {source_path}")
    source_positions = {day: index for index, day in enumerate(source_calendar)}
    missing_dates = [day for day in target_calendar if day not in source_positions]
    if missing_dates:
        raise ValueError(
            "target calendar dates missing from source provider: "
            f"count={len(missing_dates)} first={missing_dates[0]} last={missing_dates[-1]}"
        )

    aligned = np.full(len(target_calendar), np.nan, dtype="<f4")
    for target_idx, day in enumerate(target_calendar):
        source_idx = source_positions[day]
        value_idx = source_idx - start_idx
        if 0 <= value_idx < len(values):
            aligned[target_idx] = values[value_idx]
    return aligned


def write_qlib_feature_bin(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.array([0], dtype="<f4").tofile(handle)
        np.asarray(values, dtype="<f4").tofile(handle)


def copy_benchmark_features(source_provider: Path, target_provider: Path, symbols: set[str]) -> list[str]:
    source_calendar = read_qlib_calendar(source_provider)
    target_calendar = read_qlib_calendar(target_provider)

    copied: list[str] = []
    for symbol in sorted(symbols):
        feature_name = symbol.lower()
        source_dir = source_provider / "features" / feature_name
        if not source_dir.exists():
            continue

        target_dir = target_provider / "features" / feature_name
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = target_dir.with_name(f".{target_dir.name}.{uuid.uuid4().hex}.tmp")
        staging_dir.mkdir(parents=True)
        try:
            for source_path in sorted(source_dir.glob("*.day.bin")):
                aligned = align_feature_bin(source_path, source_calendar, target_calendar)
                write_qlib_feature_bin(staging_dir / source_path.name, aligned)

            backup_dir = target_dir.with_name(f".{target_dir.name}.{uuid.uuid4().hex}.old")
            if target_dir.exists():
                target_dir.replace(backup_dir)
            try:
                staging_dir.replace(target_dir)
            except Exception:
                if backup_dir.exists() and not target_dir.exists():
                    backup_dir.replace(target_dir)
                raise
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        copied.append(symbol)

        for instrument_file in (target_provider / "instruments").glob("*.txt"):
            remove_symbol_from_instrument_file(instrument_file, symbol)

    return copied


def require_mutable_qlib_output(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if (candidate / "LATEST.json").exists() or (candidate / "snapshots").is_dir():
        raise ValueError(
            "refusing to overwrite an immutable V2 dataset root; use "
            "build_cleaned_qlib_v2.py --publish"
        )
    if candidate.parent.name == "snapshots" or (
        candidate / "DATASET_MANIFEST.json"
    ).exists():
        raise ValueError(
            "refusing to overwrite an immutable V2 snapshot; use "
            "build_cleaned_qlib_v2.py --publish"
        )
    return candidate


def convert_cleaned_to_qlib(
    cleaned_parquet_dir: Path,
    cleaned_qlib_dir: Path,
    *,
    canonical_provider: Path,
    overwrite: bool,
) -> int:
    cleaned_qlib_dir = require_mutable_qlib_output(cleaned_qlib_dir)
    calendar_path = Path(canonical_provider) / "calendars" / "day.txt"
    if not calendar_path.is_file():
        raise ValueError(f"Canonical Qlib calendar is unavailable: {calendar_path}")
    return convert_to_bin(
        cleaned_parquet_dir,
        cleaned_qlib_dir,
        overwrite,
        dropna_fields=[],
        calendar_path=calendar_path,
    )


def write_reports(
    *,
    summary: dict[str, Any],
    qlib_dir: Path,
    qlib_return_code: int | None,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "generated_at": generated_at,
        "dry_run": bool(summary.get("dry_run")),
        "cleaned_qlib_dir": str(qlib_dir),
        "qlib_return_code": qlib_return_code,
        "summary": {key: value for key, value in summary.items() if key != "stats"},
    }
    title = (
        "Data Cleaning Dry Run" if summary.get("dry_run") else "Applied Data Cleaning"
    )
    status_line = (
        "- Mode: dry run, no cleaned parquet or Qlib files were written."
        if summary.get("dry_run")
        else "- Mode: applied cleaning outputs."
    )
    json_path = out_dir / f"data_cleaning_applied_{ts}.json"
    md_path = out_dir / f"data_cleaning_applied_{ts}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Generated: {generated_at}",
                "",
                "## Outputs",
                "",
                status_line,
                f"- Cleaned merged parquet: `{summary['cleaned_dir']}`",
                f"- Cleaned Qlib provider: `{qlib_dir}`",
                "- Original canonical provider is not overwritten.",
                "",
                "## Summary",
                "",
                f"- Source symbols: {summary['source_symbols']}",
                f"- Would write symbols: {summary.get('would_write_symbols', summary['written_symbols'])}",
                f"- Written symbols: {summary['written_symbols']}",
                f"- Skipped existing symbols: {summary.get('skipped_existing_symbols', 0)}",
                f"- Excluded blocker symbols: {summary['excluded_symbols']}",
                f"- Source rows: {summary['source_rows']:,}",
                f"- Output rows: {summary['output_rows']:,}",
                f"- Masked non-positive volume rows: {summary['masked_volume_rows']:,}",
                f"- Masked extreme-return event rows: {summary['masked_event_rows']:,}",
                f"- Qlib conversion return code: {qlib_return_code}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    latest_json = write_latest(json_path, "data_cleaning_applied_latest.json")
    latest_md = write_latest(md_path, "data_cleaning_applied_latest.md")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged-dir", type=Path, default=MERGED_DIR or ashare_qfq_bars_dir()
    )
    parser.add_argument("--stocks-csv", type=Path, default=ANOMALY_STOCKS_CSV)
    parser.add_argument("--events-csv", type=Path, default=ANOMALY_EVENTS_CSV)
    parser.add_argument("--cleaned-merged-dir", type=Path, default=None)
    parser.add_argument(
        "--qlib-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--benchmark-provider-dir",
        type=Path,
        default=CANONICAL_QLIB_DIR or qlib_native_snapshot_uri(),
    )
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--skip-qlib-convert", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the cleaning plan report without writing cleaned data.",
    )
    args = parser.parse_args()

    if not args.dry_run and args.cleaned_merged_dir is None:
        parser.error(
            "--cleaned-merged-dir is required for mutable compatibility output; "
            "use the governed quant_data Qlib Provider publishing workflow"
        )
    if not args.dry_run and not args.skip_qlib_convert and args.qlib_dir is None:
        parser.error(
            "--qlib-dir is required for mutable compatibility output; "
            "use the governed quant_data Qlib Provider publishing workflow"
        )

    overwrite = not args.no_overwrite
    cleaned_merged_dir = args.cleaned_merged_dir or (
        QUANT_DATA / "staging" / "cleaning-dry-run"
    )
    qlib_dir = args.qlib_dir or Path("not-generated")
    print("Loading anomaly cleaning plan...")
    plan = load_cleaning_plan(args.stocks_csv, args.events_csv)
    print(f"  excluded symbols: {len(plan.excluded_symbols)}")
    print(f"  volume-mask symbols: {len(plan.volume_mask_symbols)}")
    print(f"  return-mask symbols: {len(plan.return_mask_symbols)}")

    print("Building cleaned merged parquet...")
    summary = build_cleaned_parquet(
        args.merged_dir,
        cleaned_merged_dir,
        plan,
        overwrite,
        dry_run=args.dry_run,
    )

    qlib_return_code: int | None = None
    if args.dry_run:
        print("Dry run enabled; skipping cleaned parquet writes and Qlib conversion.")
    elif not args.skip_qlib_convert:
        print("Converting cleaned parquet to Qlib bin...")
        qlib_return_code = convert_cleaned_to_qlib(
            cleaned_merged_dir,
            qlib_dir,
            canonical_provider=args.benchmark_provider_dir,
            overwrite=overwrite,
        )
        if qlib_return_code == 0:
            copied = copy_benchmark_features(
                args.benchmark_provider_dir,
                qlib_dir,
                BENCHMARK_SYMBOLS,
            )
            print(f"Copied benchmark feature directories: {', '.join(copied) if copied else 'none'}")

    reports = write_reports(
        summary=summary,
        qlib_dir=qlib_dir,
        qlib_return_code=qlib_return_code,
        out_dir=args.out_dir,
    )
    print(
        f"Cleaned symbols: {summary['written_symbols']} / {summary['source_symbols']}"
    )
    print(f"Reports: {reports['latest_json']}, {reports['latest_markdown']}")
    return int(qlib_return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
