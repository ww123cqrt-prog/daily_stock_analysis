#!/usr/bin/env python3
"""Generate abnormal-stock lists and a cleaning strategy report."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root, workflow_root  # noqa: E402


QUANT_DATA = quant_data_root()
WORKFLOW = workflow_root()

MERGED_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
REPORT_DIR = reports_root()

REQUIRED_FIELDS = ["open", "close", "high", "low", "volume"]
PRICE_FIELDS = ["open", "close", "high", "low"]
SEVERITY_RANK = {"ok": 0, "warning": 1, "review": 2, "blocker": 3}


def format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(pd.Timestamp(value).date())


def sample_dates(dates: pd.Series, limit: int = 8) -> str:
    clean = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()
    return ",".join(str(value.date()) for value in clean.head(limit))


def max_severity(severities: list[str]) -> str:
    if not severities:
        return "ok"
    return max(severities, key=lambda item: SEVERITY_RANK[item])


def cleaning_action(issue_types: set[str]) -> str:
    blockers = {
        "missing_date_column",
        "missing_required_columns",
        "null_dates",
        "duplicate_dates",
        "null_required_values",
        "non_positive_price",
        "bad_ohlc",
    }
    if issue_types & blockers:
        return "exclude_symbol_from_training_until_repaired"
    if "abs_return_gt_50pct" in issue_types:
        return "review_and_winsorize_or_mask_event_dates"
    if "non_positive_volume" in issue_types:
        return "keep_but_mask_or_impute_volume"
    if issue_types & {"stale_end_date", "very_low_row_count", "sparse_history"}:
        return "keep_for_history_but_filter_from_recent_training"
    if "abs_return_gt_30pct" in issue_types:
        return "winsorize_returns_and_mark_event_dates"
    return "keep"


def add_event(
    events: list[dict],
    *,
    symbol: str,
    issue_type: str,
    severity: str,
    count: int,
    dates: pd.Series | None = None,
    detail: str = "",
) -> None:
    events.append(
        {
            "symbol": symbol,
            "issue_type": issue_type,
            "severity": severity,
            "count": int(count),
            "sample_dates": sample_dates(dates) if dates is not None else "",
            "detail": detail,
        }
    )


def analyze_frame(
    symbol: str,
    df: pd.DataFrame,
    dataset_end: pd.Timestamp,
    stale_days: int = 252,
    sparse_rows: int = 252,
) -> tuple[dict, list[dict]]:
    events: list[dict] = []
    summary: dict = {
        "symbol": symbol,
        "rows": len(df),
        "start": "",
        "end": "",
        "stale_days": "",
        "duplicate_dates": 0,
        "null_dates": 0,
        "missing_columns": "",
        "null_required_values": 0,
        "non_positive_price_rows": 0,
        "non_positive_volume_rows": 0,
        "bad_ohlc_rows": 0,
        "abs_return_gt_30pct": 0,
        "abs_return_gt_50pct": 0,
    }

    if "date" not in df.columns:
        add_event(
            events,
            symbol=symbol,
            issue_type="missing_date_column",
            severity="blocker",
            count=1,
            detail="date column is required for Qlib alignment",
        )
        issue_types = {event["issue_type"] for event in events}
        summary["severity"] = "blocker"
        summary["cleaning_action"] = cleaning_action(issue_types)
        return summary, events

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    valid_dates = frame["date"].dropna()
    if not valid_dates.empty:
        start = valid_dates.min()
        end = valid_dates.max()
        summary["start"] = format_date(start)
        summary["end"] = format_date(end)
        lag = int((dataset_end - end).days)
        summary["stale_days"] = max(lag, 0)
        if lag > stale_days:
            add_event(
                events,
                symbol=symbol,
                issue_type="stale_end_date",
                severity="warning",
                count=1,
                detail=f"last date is {lag} calendar days before dataset end {format_date(dataset_end)}",
            )

    duplicate_dates = int(frame["date"].duplicated().sum())
    null_dates = int(frame["date"].isna().sum())
    summary["duplicate_dates"] = duplicate_dates
    summary["null_dates"] = null_dates
    if duplicate_dates:
        add_event(
            events,
            symbol=symbol,
            issue_type="duplicate_dates",
            severity="blocker",
            count=duplicate_dates,
            dates=frame.loc[frame["date"].duplicated(keep=False), "date"],
            detail="one trading day appears more than once",
        )
    if null_dates:
        add_event(
            events,
            symbol=symbol,
            issue_type="null_dates",
            severity="blocker",
            count=null_dates,
            detail="date could not be parsed",
        )

    missing = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    summary["missing_columns"] = ",".join(missing)
    if missing:
        add_event(
            events,
            symbol=symbol,
            issue_type="missing_required_columns",
            severity="blocker",
            count=len(missing),
            detail=",".join(missing),
        )

    numeric: dict[str, pd.Series] = {}
    for field in REQUIRED_FIELDS:
        if field in frame.columns:
            numeric[field] = pd.to_numeric(frame[field], errors="coerce")

    null_required = int(sum(series.isna().sum() for series in numeric.values()))
    summary["null_required_values"] = null_required
    if null_required:
        add_event(
            events,
            symbol=symbol,
            issue_type="null_required_values",
            severity="blocker",
            count=null_required,
            detail="required OHLCV fields contain null or non-numeric values",
        )

    if all(field in numeric for field in PRICE_FIELDS):
        non_positive_price = pd.Series(False, index=frame.index)
        for field in PRICE_FIELDS:
            non_positive_price = non_positive_price | (numeric[field] <= 0)
        count = int(non_positive_price.fillna(False).sum())
        summary["non_positive_price_rows"] = count
        if count:
            add_event(
                events,
                symbol=symbol,
                issue_type="non_positive_price",
                severity="blocker",
                count=count,
                dates=frame.loc[non_positive_price.fillna(False), "date"],
                detail="open/close/high/low should be positive",
            )

        bad_ohlc = (
            (numeric["high"] < numeric["low"])
            | (numeric["high"] < numeric["open"])
            | (numeric["high"] < numeric["close"])
            | (numeric["low"] > numeric["open"])
            | (numeric["low"] > numeric["close"])
        )
        bad_count = int(bad_ohlc.fillna(False).sum())
        summary["bad_ohlc_rows"] = bad_count
        if bad_count:
            add_event(
                events,
                symbol=symbol,
                issue_type="bad_ohlc",
                severity="blocker",
                count=bad_count,
                dates=frame.loc[bad_ohlc.fillna(False), "date"],
                detail="OHLC violates high/low envelope",
            )

    if "volume" in numeric:
        non_positive_volume = numeric["volume"] <= 0
        count = int(non_positive_volume.fillna(False).sum())
        summary["non_positive_volume_rows"] = count
        if count:
            ratio = count / max(len(frame), 1)
            severity = "review" if count >= 20 or ratio >= 0.01 else "warning"
            add_event(
                events,
                symbol=symbol,
                issue_type="non_positive_volume",
                severity=severity,
                count=count,
                dates=frame.loc[non_positive_volume.fillna(False), "date"],
                detail=f"{ratio:.2%} of rows have volume <= 0; often caused by停牌或复权成交量口径",
            )

    if "close" in numeric:
        ordered = frame.assign(_close=numeric["close"]).sort_values("date")
        daily_return = ordered["_close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        abs_return = daily_return.abs()
        gt30 = abs_return > 0.30
        gt50 = abs_return > 0.50
        summary["abs_return_gt_30pct"] = int(gt30.sum())
        summary["abs_return_gt_50pct"] = int(gt50.sum())
        if summary["abs_return_gt_30pct"]:
            add_event(
                events,
                symbol=symbol,
                issue_type="abs_return_gt_30pct",
                severity="warning",
                count=summary["abs_return_gt_30pct"],
                dates=ordered.loc[gt30.fillna(False), "date"],
                detail=f"max_abs_return={abs_return.max(skipna=True):.4f}",
            )
        if summary["abs_return_gt_50pct"]:
            add_event(
                events,
                symbol=symbol,
                issue_type="abs_return_gt_50pct",
                severity="review",
                count=summary["abs_return_gt_50pct"],
                dates=ordered.loc[gt50.fillna(False), "date"],
                detail=f"max_abs_return={abs_return.max(skipna=True):.4f}",
            )

    if len(frame) < sparse_rows:
        issue_type = "very_low_row_count" if len(frame) < 60 else "sparse_history"
        severity = "review" if len(frame) < 60 else "warning"
        add_event(
            events,
            symbol=symbol,
            issue_type=issue_type,
            severity=severity,
            count=len(frame),
            detail=f"row count below {sparse_rows}",
        )

    issue_types = {event["issue_type"] for event in events}
    summary["severity"] = max_severity([event["severity"] for event in events])
    summary["cleaning_action"] = cleaning_action(issue_types)
    return summary, events


def dataset_date_range(paths: list[Path]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for path in paths:
        try:
            dates = pd.read_parquet(path, columns=["date"])
        except Exception:
            continue
        series = pd.to_datetime(dates["date"], errors="coerce").dropna()
        if series.empty:
            continue
        starts.append(series.min())
        ends.append(series.max())
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def indexed_date_range(directory: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    index_path = Path(directory) / ".metadata" / "FILE_INDEX.parquet"
    if not index_path.is_file():
        return None, None
    try:
        index = pd.read_parquet(index_path, columns=["first_date", "last_date"])
    except Exception:
        return None, None
    starts = pd.to_datetime(index["first_date"], errors="coerce").dropna()
    ends = pd.to_datetime(index["last_date"], errors="coerce").dropna()
    if starts.empty or ends.empty:
        return None, None
    return starts.min(), ends.max()


def scan_merged_files(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    paths = sorted(directory.glob("*.parquet"))
    dataset_start, dataset_end = indexed_date_range(directory)
    if dataset_end is None:
        dataset_start, dataset_end = dataset_date_range(paths)
    if dataset_end is None:
        dataset_end = pd.Timestamp.today().normalize()

    summary_rows: list[dict] = []
    event_rows: list[dict] = []
    for index, path in enumerate(paths, 1):
        try:
            frame = pd.read_parquet(path)
            summary, events = analyze_frame(path.stem, frame, dataset_end)
        except Exception as exc:
            summary = {
                "symbol": path.stem,
                "rows": 0,
                "start": "",
                "end": "",
                "stale_days": "",
                "severity": "blocker",
                "cleaning_action": "exclude_symbol_from_training_until_repaired",
            }
            events = [
                {
                    "symbol": path.stem,
                    "issue_type": "read_parquet_failed",
                    "severity": "blocker",
                    "count": 1,
                    "sample_dates": "",
                    "detail": str(exc),
                }
            ]

        summary_rows.append(summary)
        event_rows.extend(events)
        if index % 1000 == 0:
            print(f"  scanned {index}/{len(paths)} merged files")

    summary_df = pd.DataFrame(summary_rows)
    events_df = pd.DataFrame(event_rows)
    coverage = {
        "files": len(paths),
        "rows": int(summary_df["rows"].sum()) if not summary_df.empty and "rows" in summary_df else 0,
        "start": format_date(dataset_start),
        "end": format_date(dataset_end),
    }
    return summary_df, events_df, coverage


def write_latest(path: Path, latest_name: str) -> Path:
    latest = path.with_name(latest_name)
    shutil.copyfile(path, latest)
    return latest


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    *,
    summary_df: pd.DataFrame,
    events_df: pd.DataFrame,
    coverage: dict,
    stock_csv: Path,
    events_csv: Path,
    merged_dir: Path,
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    severity_counts = (
        summary_df["severity"].value_counts().reindex(["blocker", "review", "warning", "ok"], fill_value=0).to_dict()
        if not summary_df.empty
        else {}
    )
    issue_counts = (
        events_df.groupby("issue_type")["symbol"].nunique().sort_values(ascending=False).to_dict()
        if not events_df.empty
        else {}
    )

    issue_rows = [{"issue": key, "symbols": value} for key, value in issue_counts.items()]
    top_blockers = summary_df[summary_df["severity"] == "blocker"].head(20).to_dict("records") if not summary_df.empty else []
    top_reviews = summary_df[summary_df["severity"] == "review"].head(20).to_dict("records") if not summary_df.empty else []

    structural_blockers = int(severity_counts.get("blocker", 0))
    if structural_blockers:
        conclusion = "当前数据存在结构性 blocker，正式训练前应先排除或修复这些股票。"
    else:
        conclusion = (
            "当前合并数据没有发现结构性 blocker，可以先跑基线训练；"
            "review 级事件建议在干净训练集里做极端收益截尾或事件日 mask。"
        )

    return f"""# Abnormal Stock List and Cleaning Strategy

Generated: {generated}

## Coverage

- Merged parquet directory: `{merged_dir}`
- Files scanned: {coverage["files"]}
- Rows scanned: {coverage["rows"]:,}
- Dataset date range: {coverage["start"]} ~ {coverage["end"]}

## Severity Summary

- Blocker symbols: {severity_counts.get("blocker", 0)}
- Review symbols: {severity_counts.get("review", 0)}
- Warning symbols: {severity_counts.get("warning", 0)}
- Clean/OK symbols: {severity_counts.get("ok", 0)}

{conclusion}

## Issue Counts

{markdown_table(issue_rows, ["issue", "symbols"])}

## Cleaning Strategy

- `blocker`: 先从训练股票池排除，修复后再纳入。包括日期缺失/重复、必需列缺失、OHLC 结构错误、价格非正数。
- `review`: 不直接删除整只股票，优先人工复核事件日；训练时对极端收益做 winsorize，或 mask 异常事件日标签/特征。
- `warning`: 一般可保留。零成交量多半对应停牌或数据源成交量口径，建议成交量特征单独 mask/impute；过旧或过短历史在最近窗口训练中降权或过滤。
- 场内基金/ETF 后续接入时，应单独定义基金专用异常规则，不把公司财报类清洗规则套到 ETF。

## Top Blocker Symbols

{markdown_table(top_blockers, ["symbol", "rows", "start", "end", "severity", "cleaning_action", "missing_columns", "duplicate_dates", "bad_ohlc_rows"])}

## Top Review Symbols

{markdown_table(top_reviews, ["symbol", "rows", "start", "end", "severity", "cleaning_action", "abs_return_gt_50pct", "non_positive_volume_rows"])}

## Artifacts

- Abnormal stock summary CSV: `{stock_csv}`
- Abnormal event detail CSV: `{events_csv}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Scanning merged parquet files for anomaly events...")
    summary_df, events_df, coverage = scan_merged_files(args.merged_dir)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["severity", "symbol"],
            key=lambda col: col.map(SEVERITY_RANK) if col.name == "severity" else col,
            ascending=[False, True],
        )
    if not events_df.empty:
        events_df = events_df.sort_values(
            by=["severity", "symbol", "issue_type"],
            key=lambda col: col.map(SEVERITY_RANK) if col.name == "severity" else col,
            ascending=[False, True, True],
        )

    stock_csv = args.out_dir / f"data_anomaly_stocks_{ts}.csv"
    events_csv = args.out_dir / f"data_anomaly_events_{ts}.csv"
    report_path = args.out_dir / f"data_cleaning_strategy_{ts}.md"

    summary_df.to_csv(stock_csv, index=False)
    events_df.to_csv(events_csv, index=False)
    report = build_report(
        summary_df=summary_df,
        events_df=events_df,
        coverage=coverage,
        stock_csv=stock_csv,
        events_csv=events_csv,
        merged_dir=args.merged_dir,
    )
    report_path.write_text(report, encoding="utf-8")

    latest_stocks = write_latest(stock_csv, "data_anomaly_stocks_latest.csv")
    latest_events = write_latest(events_csv, "data_anomaly_events_latest.csv")
    latest_report = write_latest(report_path, "data_cleaning_strategy_latest.md")

    print(f"Stocks: {stock_csv}")
    print(f"Events: {events_csv}")
    print(f"Report: {report_path}")
    print(f"Latest stocks: {latest_stocks}")
    print(f"Latest events: {latest_events}")
    print(f"Latest report: {latest_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
