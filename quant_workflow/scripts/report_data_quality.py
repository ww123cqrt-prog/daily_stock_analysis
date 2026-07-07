"""Generate a data quality report for the local A-share Qlib dataset."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root, reports_root, workflow_root  # noqa: E402


QUANT_DATA = quant_data_root()
WORKFLOW = workflow_root()

BAOSTOCK_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "baostock_qfq"
JQDATA_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "jqdata"
AKSHARE_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "qfq_full"
AKSHARE_SAMPLE_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "qfq"
MERGED_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
QLIB_DIR = qlib_provider_uri()
STOCK_LIST = QUANT_DATA / "stock_lists.json"
EMPTY_CODES = QUANT_DATA / "baostock_empty_codes.json"
TIMEOUT_CODES = QUANT_DATA / "baostock_timeout_codes.json"
REPORT_DIR = reports_root()
LATEST_TRAINING_RESULT_FILENAME = "latest_training_result.json"
BENCHMARK_FEATURES = {"sh000300"}

NUMERIC_FIELDS = ["open", "close", "high", "low", "volume"]


@dataclass
class SourceStats:
    name: str
    files: int
    rows: int
    first_date: str
    last_date: str
    size_mb: float


def read_json_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def read_stock_universe(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    raw_codes = data.get("all", []) if isinstance(data, dict) else []
    return [str(code).split(".")[0] for code in raw_codes if str(code).strip()]


def file_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024 / 1024
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024 / 1024


def source_stats(name: str, directory: Path) -> SourceStats:
    files = sorted(directory.glob("*.parquet")) if directory.exists() else []
    rows = 0
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for path in files:
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            continue
        rows += len(df)
        mins.append(dates.min())
        maxs.append(dates.max())

    return SourceStats(
        name=name,
        files=len(files),
        rows=rows,
        first_date=str(min(mins).date()) if mins else "",
        last_date=str(max(maxs).date()) if maxs else "",
        size_mb=file_size_mb(directory),
    )


def load_source_index(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.parquet")} if directory.exists() else set()


def scan_merged_files(directory: Path, sample_limit: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    issue_rows: list[dict] = []
    files = sorted(directory.glob("*.parquet"))
    if sample_limit > 0:
        files = files[:sample_limit]

    for i, path in enumerate(files, 1):
        sym = path.stem
        df = pd.read_parquet(path)
        if "date" not in df.columns:
            issue_rows.append({"symbol": sym, "issue": "missing_date_column", "count": 1, "detail": ""})
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        row_count = len(df)
        duplicate_dates = int(df["date"].duplicated().sum())
        null_dates = int(df["date"].isna().sum())
        missing_cols = [c for c in NUMERIC_FIELDS if c not in df.columns]

        summary = {
            "symbol": sym,
            "rows": row_count,
            "start": str(df["date"].min().date()) if not df["date"].dropna().empty else "",
            "end": str(df["date"].max().date()) if not df["date"].dropna().empty else "",
            "duplicate_dates": duplicate_dates,
            "null_dates": null_dates,
            "missing_columns": ",".join(missing_cols),
        }

        for col in NUMERIC_FIELDS:
            if col not in df.columns:
                summary[f"null_{col}"] = row_count
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            summary[f"null_{col}"] = int(values.isna().sum())
            summary[f"non_positive_{col}"] = int((values <= 0).sum())

        if {"open", "high", "low", "close"}.issubset(df.columns):
            open_ = pd.to_numeric(df["open"], errors="coerce")
            high = pd.to_numeric(df["high"], errors="coerce")
            low = pd.to_numeric(df["low"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            bad_ohlc = (
                (high < low)
                | (high < open_)
                | (high < close)
                | (low > open_)
                | (low > close)
            )
            summary["bad_ohlc_rows"] = int(bad_ohlc.fillna(False).sum())
        else:
            summary["bad_ohlc_rows"] = row_count

        if "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            pct = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
            summary["abs_return_gt_30pct"] = int((pct.abs() > 0.30).sum())
            summary["abs_return_gt_50pct"] = int((pct.abs() > 0.50).sum())
        else:
            summary["abs_return_gt_30pct"] = 0
            summary["abs_return_gt_50pct"] = 0

        summary_rows.append(summary)

        for field in [
            "duplicate_dates",
            "null_dates",
            "missing_columns",
            "bad_ohlc_rows",
            "abs_return_gt_50pct",
        ]:
            value = summary.get(field)
            if value and value != "":
                issue_rows.append({"symbol": sym, "issue": field, "count": value, "detail": summary.get("missing_columns", "")})

        if i % 1000 == 0:
            print(f"  scanned {i}/{len(files)} merged files")

    return pd.DataFrame(summary_rows), pd.DataFrame(issue_rows)


def overlap_consistency(sample_limit: int = 0) -> pd.DataFrame:
    bs_files = load_source_index(BAOSTOCK_DIR)
    jq_files = load_source_index(JQDATA_DIR)
    symbols = sorted(bs_files & jq_files)
    if sample_limit > 0:
        symbols = symbols[:sample_limit]

    rows: list[dict] = []
    for i, sym in enumerate(symbols, 1):
        bs = pd.read_parquet(BAOSTOCK_DIR / f"{sym}.parquet")
        jq = pd.read_parquet(JQDATA_DIR / f"{sym}.parquet")
        bs["date"] = pd.to_datetime(bs["date"], errors="coerce")
        jq["date"] = pd.to_datetime(jq["date"], errors="coerce")
        joined = bs.merge(jq, on="date", suffixes=("_bs", "_jq"))
        if joined.empty:
            rows.append({"symbol": sym, "overlap_days": 0})
            continue

        row = {"symbol": sym, "overlap_days": len(joined)}
        for col in ["open", "close", "high", "low"]:
            diff = (pd.to_numeric(joined[f"{col}_bs"], errors="coerce") - pd.to_numeric(joined[f"{col}_jq"], errors="coerce")).abs()
            base = pd.to_numeric(joined[f"{col}_jq"], errors="coerce").abs().replace(0, np.nan)
            rel = (diff / base).replace([np.inf, -np.inf], np.nan)
            row[f"{col}_max_abs_diff"] = float(diff.max()) if not diff.empty else math.nan
            row[f"{col}_median_rel_diff"] = float(rel.median()) if not rel.dropna().empty else math.nan

        if "volume_bs" in joined.columns and "volume_jq" in joined.columns:
            vol_bs = pd.to_numeric(joined["volume_bs"], errors="coerce")
            vol_jq = pd.to_numeric(joined["volume_jq"], errors="coerce").replace(0, np.nan)
            ratio = (vol_bs / vol_jq).replace([np.inf, -np.inf], np.nan)
            row["volume_ratio_median"] = float(ratio.median()) if not ratio.dropna().empty else math.nan
            row["volume_ratio_min"] = float(ratio.min()) if not ratio.dropna().empty else math.nan
            row["volume_ratio_max"] = float(ratio.max()) if not ratio.dropna().empty else math.nan

        rows.append(row)
        if i % 1000 == 0:
            print(f"  compared {i}/{len(symbols)} overlap files")

    return pd.DataFrame(rows)


def qlib_provider_stats() -> dict:
    cal = QLIB_DIR / "calendars" / "day.txt"
    inst = QLIB_DIR / "instruments" / "all.txt"
    calendar = cal.read_text(encoding="utf-8").splitlines() if cal.exists() else []
    instruments = inst.read_text(encoding="utf-8").splitlines() if inst.exists() else []
    instrument_names = {line.split("\t")[0].lower() for line in instruments if line.strip()}
    feature_dirs = [p for p in (QLIB_DIR / "features").iterdir() if p.is_dir()] if (QLIB_DIR / "features").exists() else []
    feature_names = {p.name for p in feature_dirs}
    extra_features = sorted(feature_names - instrument_names)
    unexpected_extra = [name for name in extra_features if name not in BENCHMARK_FEATURES]
    return {
        "calendar_days": len(calendar),
        "calendar_start": calendar[0] if calendar else "",
        "calendar_end": calendar[-1] if calendar else "",
        "instruments": len(instruments),
        "feature_dirs": len(feature_dirs),
        "benchmark_feature_dirs": len([name for name in extra_features if name in BENCHMARK_FEATURES]),
        "unexpected_extra_feature_dirs": len(unexpected_extra),
        "unexpected_extra_feature_sample": ", ".join(unexpected_extra[:20]),
        "size_mb": file_size_mb(QLIB_DIR),
    }


def source_status(name: str, stats: SourceStats, merged_symbols: set[str]) -> str:
    if stats.files == 0 or stats.rows == 0:
        return "missing"
    if name in {"akshare", "jqdata"}:
        return "partial"
    if name == "merged" and stats.files < max(len(merged_symbols), 1):
        return "partial"
    return "available"


def load_latest_training_result(reports_dir: Path) -> dict | None:
    latest_training_result = reports_dir / LATEST_TRAINING_RESULT_FILENAME
    if not latest_training_result.exists():
        return None
    try:
        payload = json.loads(latest_training_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_training_readiness(
    qlib_stats: dict,
    merged_success: set[str],
    issue_counts: dict[str, int],
) -> tuple[bool, dict[str, str]]:
    blockers = []
    if qlib_stats["calendar_days"] < 252:
        blockers.append("Qlib calendar has fewer than 252 trading days")
    if qlib_stats["instruments"] < 100:
        blockers.append("Qlib provider has fewer than 100 instruments")
    if len(merged_success) < 100:
        blockers.append("Merged local dataset has fewer than 100 instruments")
    for key in ["duplicate_dates", "null_dates", "missing_columns", "bad_ohlc_rows"]:
        if issue_counts.get(key, 0) > 0:
            blockers.append(f"Structural issue remains: {key}={issue_counts[key]}")

    if blockers:
        return False, {
            "status": "limited",
            "summary": "数据仍需修复后再正式训练：" + "; ".join(blockers[:4]),
        }

    warnings = []
    if issue_counts.get("non_positive_volume", 0) > 0:
        warnings.append(f"{issue_counts['non_positive_volume']} symbols have non-positive volume rows")
    if issue_counts.get("abs_return_gt_50pct", 0) > 0:
        warnings.append(f"{issue_counts['abs_return_gt_50pct']} symbols have absolute daily returns above 50%")

    summary = "Qlib provider has enough calendar/instrument coverage for the baseline run."
    if warnings:
        summary += " Warnings: " + "; ".join(warnings[:3])

    return True, {
        "status": "trainable",
        "summary": summary,
    }


def write_research_status(
    out_dir: Path,
    source_rows: list[dict],
    qlib_stats: dict,
    merged_success: set[str],
    issue_counts: dict[str, int],
) -> Path:
    trainable, training_readiness = build_training_readiness(
        qlib_stats=qlib_stats,
        merged_success=merged_success,
        issue_counts=issue_counts,
    )
    latest_training = load_latest_training_result(out_dir)

    data_sources = []
    for item in source_rows:
        stats = SourceStats(**item)
        data_sources.append(
            {
                "name": stats.name,
                "files": stats.files,
                "rows": stats.rows,
                "date_range": f"{stats.first_date} ~ {stats.last_date}",
                "size_mb": round(stats.size_mb, 2),
                "status": source_status(stats.name, stats, merged_success),
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "trainable": trainable,
        "training_readiness": training_readiness,
        "data_sources": data_sources,
        "qlib_provider": {
            "path": str(QLIB_DIR),
            "calendar_days": qlib_stats["calendar_days"],
            "calendar_range": f"{qlib_stats['calendar_start']} ~ {qlib_stats['calendar_end']}",
            "instruments": qlib_stats["instruments"],
            "feature_dirs": qlib_stats["feature_dirs"],
            "size_mb": round(qlib_stats["size_mb"], 2),
        },
        "latest_training": latest_training,
        "quality_issues": issue_counts,
    }

    status_path = out_dir / "research_status_latest.json"
    status_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return status_path


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--overlap-sample", type=int, default=0, help="0 means all overlaps")
    parser.add_argument("--merged-sample", type=int, default=0, help="0 means all merged files")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    stock_codes = read_stock_universe(STOCK_LIST)
    stock_set = set(stock_codes)
    empty_codes = set(read_json_list(EMPTY_CODES))
    timeout_codes = set(read_json_list(TIMEOUT_CODES))

    source_rows = []
    for name, directory in [
        ("akshare", AKSHARE_DIR if AKSHARE_DIR.exists() else AKSHARE_SAMPLE_DIR),
        ("baostock", BAOSTOCK_DIR),
        ("jqdata", JQDATA_DIR),
        ("merged", MERGED_DIR),
    ]:
        stats = source_stats(name, directory)
        source_rows.append(stats.__dict__)

    source_df = pd.DataFrame(source_rows)
    source_csv = out_dir / f"data_quality_sources_{ts}.csv"
    source_df.to_csv(source_csv, index=False)

    print("Scanning merged files...")
    merged_summary, merged_issues = scan_merged_files(MERGED_DIR, sample_limit=args.merged_sample)
    summary_csv = out_dir / f"data_quality_merged_summary_{ts}.csv"
    issues_csv = out_dir / f"data_quality_issues_{ts}.csv"
    merged_summary.to_csv(summary_csv, index=False)
    merged_issues.to_csv(issues_csv, index=False)

    print("Comparing Baostock/JQData overlap...")
    overlap = overlap_consistency(args.overlap_sample)
    overlap_csv = out_dir / f"data_quality_overlap_{ts}.csv"
    overlap.to_csv(overlap_csv, index=False)

    qlib_stats = qlib_provider_stats()
    bs_success = load_source_index(BAOSTOCK_DIR)
    jq_success = load_source_index(JQDATA_DIR)
    merged_success = load_source_index(MERGED_DIR)

    missing_success = sorted(stock_set - merged_success)
    classified = bs_success | empty_codes | timeout_codes
    unclassified = sorted(stock_set - classified)

    issue_counts = {}
    for col in ["duplicate_dates", "null_dates", "bad_ohlc_rows", "abs_return_gt_50pct", "non_positive_volume"]:
        issue_counts[col] = int((merged_summary[col] > 0).sum()) if col in merged_summary.columns else 0
    missing_cols_count = int((merged_summary["missing_columns"].astype(str) != "").sum()) if "missing_columns" in merged_summary.columns else 0
    issue_counts["missing_columns"] = missing_cols_count

    overlap_metrics = {}
    if not overlap.empty:
        overlap_metrics = {
            "symbols": len(overlap),
            "median_close_rel_diff": float(overlap["close_median_rel_diff"].median()),
            "p95_close_rel_diff": float(overlap["close_median_rel_diff"].quantile(0.95)),
            "median_volume_ratio": float(overlap["volume_ratio_median"].median()),
            "p05_volume_ratio": float(overlap["volume_ratio_median"].quantile(0.05)),
            "p95_volume_ratio": float(overlap["volume_ratio_median"].quantile(0.95)),
        }

    rows = []
    for item in source_rows:
        rows.append(
            {
                "source": item["name"],
                "files": item["files"],
                "rows": item["rows"],
                "date_range": f"{item['first_date']} ~ {item['last_date']}",
                "size_mb": f"{item['size_mb']:.1f}",
            }
        )

    report = f"""# A-Share Data Quality Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

- Stock universe: {len(stock_set)}
- Merged usable instruments: {len(merged_success)}
- Baostock empty/no-data instruments: {len(empty_codes)}
- Baostock timeout instruments: {len(timeout_codes)}
- Unclassified instruments: {len(unclassified)}
- Merged rows: {int(source_df[source_df['name'] == 'merged']['rows'].iloc[0]):,}
- Qlib provider: `{QLIB_DIR}`

## Source Coverage

{markdown_table(rows, ["source", "files", "rows", "date_range", "size_mb"])}

## Qlib Provider

- Calendar days: {qlib_stats["calendar_days"]}
- Calendar range: {qlib_stats["calendar_start"]} ~ {qlib_stats["calendar_end"]}
- Instruments: {qlib_stats["instruments"]}
- Feature directories: {qlib_stats["feature_dirs"]}
- Benchmark-only feature directories: {qlib_stats["benchmark_feature_dirs"]}
- Unexpected extra feature directories: {qlib_stats["unexpected_extra_feature_dirs"]} `{qlib_stats["unexpected_extra_feature_sample"]}`
- Size: {qlib_stats["size_mb"]:.1f} MB

## Structural Checks

- Symbols with duplicate dates: {issue_counts["duplicate_dates"]}
- Symbols with null dates: {issue_counts["null_dates"]}
- Symbols with missing required columns: {issue_counts["missing_columns"]}
- Symbols with OHLC inconsistencies: {issue_counts["bad_ohlc_rows"]}
- Symbols with non-positive volume rows: {issue_counts["non_positive_volume"]}
- Symbols with absolute daily return above 50%: {issue_counts["abs_return_gt_50pct"]}

## Baostock vs JQData Overlap

- Compared symbols: {overlap_metrics.get("symbols", 0)}
- Median close relative difference: {overlap_metrics.get("median_close_rel_diff", math.nan):.6f}
- P95 close relative difference: {overlap_metrics.get("p95_close_rel_diff", math.nan):.6f}
- Median volume ratio, Baostock/JQData: {overlap_metrics.get("median_volume_ratio", math.nan):.6f}
- P05/P95 volume ratio: {overlap_metrics.get("p05_volume_ratio", math.nan):.6f} / {overlap_metrics.get("p95_volume_ratio", math.nan):.6f}

Interpretation: small close differences are expected because Baostock and JQData
round adjusted prices differently. Volume ratio below 1 for some stocks is also
expected in adjusted-volume space after corporate actions.

## Missing / Skipped Instruments

- Merged missing success count: {len(missing_success)}
- Missing success sample: `{", ".join(missing_success[:40])}`
- Empty/no-data sample: `{", ".join(sorted(empty_codes)[:40])}`
- Timeout codes: `{", ".join(sorted(timeout_codes))}`

## Artifacts

- Source summary CSV: `{source_csv}`
- Merged per-symbol summary CSV: `{summary_csv}`
- Issues CSV: `{issues_csv}`
- Baostock/JQData overlap CSV: `{overlap_csv}`
"""

    report_path = out_dir / f"data_quality_report_{ts}.md"
    report_path.write_text(report, encoding="utf-8")
    latest_path = out_dir / "data_quality_report_latest.md"
    latest_path.write_text(report, encoding="utf-8")
    research_status_path = write_research_status(
        out_dir=out_dir,
        source_rows=source_rows,
        qlib_stats=qlib_stats,
        merged_success=merged_success,
        issue_counts=issue_counts,
    )

    print(f"Report: {report_path}")
    print(f"Latest: {latest_path}")
    print(f"Research status: {research_status_path}")
    print(f"Sources: {source_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Issues: {issues_csv}")
    print(f"Overlap: {overlap_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
