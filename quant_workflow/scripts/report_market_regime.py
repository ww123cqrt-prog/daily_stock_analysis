"""Generate a read-only market regime and cash-ratio research report.

This report is deliberately advisory. It does not connect to broker software,
does not submit orders, and does not change any Qlib provider files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import ashare_qfq_bars_dir, quant_data_root, reports_root, workflow_root  # noqa: E402


QUANT_DATA = quant_data_root()
WORKFLOW = workflow_root()
MERGED_DIR: Path | None = None
REPORT_DIR = reports_root()


def _safe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_close_series(path: Path, lookback_days: int) -> pd.Series | None:
    try:
        frame = pd.read_parquet(path, columns=["date", "close"])
    except (OSError, KeyError, ValueError):
        return None
    if frame.empty:
        return None
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["close"] > 0]
    if frame.empty:
        return None
    frame = frame.sort_values("date").drop_duplicates(subset=["date"])
    tail = frame.tail(max(lookback_days + 1, 2))
    if len(tail) < 2:
        return None
    return pd.Series(tail["close"].to_numpy(dtype="float64"), index=tail["date"], name=path.stem)


def load_close_matrix(data_dir: Path, lookback_days: int = 252, limit: int = 0) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    series = []
    for path in files:
        item = _read_close_series(path, lookback_days)
        if item is not None:
            series.append(item)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index()


def _window_return(matrix: pd.DataFrame, days: int) -> pd.Series:
    if len(matrix) < days + 1:
        return pd.Series(dtype="float64")
    recent = matrix.iloc[-1]
    base = matrix.iloc[-days - 1]
    return (recent / base - 1).replace([np.inf, -np.inf], np.nan).dropna()


def classify_regime(metrics: dict[str, float | int | str | None]) -> tuple[str, float, str]:
    ret20 = float(metrics.get("return_20d") or 0)
    ret60 = float(metrics.get("return_60d") or 0)
    breadth20 = float(metrics.get("breadth_20d") or 0)
    drawdown = float(metrics.get("drawdown") or 0)
    volatility = float(metrics.get("volatility_20d") or 0)

    risk_score = 0
    if ret20 < -0.05:
        risk_score += 2
    elif ret20 < -0.02:
        risk_score += 1
    if ret60 < -0.10:
        risk_score += 2
    elif ret60 < -0.04:
        risk_score += 1
    if breadth20 < 0.35:
        risk_score += 2
    elif breadth20 < 0.45:
        risk_score += 1
    if drawdown < -0.12:
        risk_score += 2
    elif drawdown < -0.06:
        risk_score += 1
    if volatility > 0.035:
        risk_score += 1

    if risk_score >= 5:
        return "risk_off", 0.80, "Market trend and breadth are weak; use cash as a defensive state."
    if risk_score >= 3:
        return "defensive", 0.50, "Market risk is elevated; reduce gross exposure before relying on stock ranking."
    if ret20 > 0.03 and breadth20 > 0.55 and drawdown > -0.05:
        return "risk_on", 0.10, "Market trend and breadth are supportive; ranking models have a healthier backdrop."
    return "neutral", 0.30, "Market signal is mixed; keep position sizing moderate and verify strategy drawdown."


def analyze_market_regime(data_dir: Path, lookback_days: int = 252, min_symbols: int = 100, limit: int = 0) -> dict:
    matrix = load_close_matrix(data_dir, lookback_days=lookback_days, limit=limit)
    matrix = matrix.dropna(axis=1, thresh=max(2, min(len(matrix), lookback_days // 2)))
    if matrix.shape[1] < min_symbols:
        start_date = str(matrix.index.min().date()) if not matrix.empty else ""
        end_date = str(matrix.index.max().date()) if not matrix.empty else ""
        return {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "regime": "insufficient_data",
            "cash_ratio": 0.80,
            "position_ratio": 0.20,
            "interpretation": (
                f"Need at least {min_symbols} symbols with close data, got {matrix.shape[1]}; "
                "use a defensive cash setting until local data is prepared."
            ),
            "metrics": {
                "symbols": int(matrix.shape[1]),
                "calendar_days": int(len(matrix)),
                "start_date": start_date,
                "end_date": end_date,
                "return_20d": None,
                "return_60d": None,
                "breadth_20d": None,
                "breadth_60d": None,
                "drawdown": None,
                "volatility_20d": None,
            },
            "source_dir": str(data_dir),
            "mode": "read_only_research_report",
        }

    market_index = matrix.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0).add(1).cumprod()
    daily_returns = market_index.pct_change(fill_method=None).dropna()
    ret20_by_symbol = _window_return(matrix, min(20, max(1, len(matrix) - 1)))
    ret60_by_symbol = _window_return(matrix, min(60, max(1, len(matrix) - 1)))

    ret20_days = min(20, len(market_index) - 1)
    ret60_days = min(60, len(market_index) - 1)
    return_20d = float(market_index.iloc[-1] / market_index.iloc[-ret20_days - 1] - 1) if ret20_days > 0 else 0.0
    return_60d = float(market_index.iloc[-1] / market_index.iloc[-ret60_days - 1] - 1) if ret60_days > 0 else 0.0
    rolling_peak = market_index.cummax()
    drawdown = float(market_index.iloc[-1] / rolling_peak.iloc[-1] - 1)
    volatility_20d = float(daily_returns.tail(20).std()) if not daily_returns.empty else 0.0
    breadth_20d = float((ret20_by_symbol > 0).mean()) if not ret20_by_symbol.empty else 0.0
    breadth_60d = float((ret60_by_symbol > 0).mean()) if not ret60_by_symbol.empty else 0.0

    metrics = {
        "symbols": int(matrix.shape[1]),
        "calendar_days": int(len(matrix)),
        "start_date": str(matrix.index.min().date()),
        "end_date": str(matrix.index.max().date()),
        "return_20d": _safe_float(return_20d),
        "return_60d": _safe_float(return_60d),
        "breadth_20d": _safe_float(breadth_20d),
        "breadth_60d": _safe_float(breadth_60d),
        "drawdown": _safe_float(drawdown),
        "volatility_20d": _safe_float(volatility_20d),
    }
    regime, cash_ratio, interpretation = classify_regime(metrics)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "regime": regime,
        "cash_ratio": cash_ratio,
        "position_ratio": round(1 - cash_ratio, 4),
        "interpretation": interpretation,
        "metrics": metrics,
        "source_dir": str(data_dir),
        "mode": "read_only_research_report",
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def build_markdown(payload: dict) -> str:
    metrics = payload.get("metrics", {})
    return f"""# Market Regime Report

Generated: {payload.get("generated_at")}

## Signal

- Regime: `{payload.get("regime")}`
- Suggested cash_ratio: `{_fmt_pct(payload.get("cash_ratio"))}`
- Suggested position_ratio: `{_fmt_pct(payload.get("position_ratio"))}`
- Interpretation: {payload.get("interpretation")}

This is a read-only research signal. It does not submit orders, connect to broker
software, or change local Qlib data.

## Market Metrics

- Symbols: {metrics.get("symbols")}
- Calendar days: {metrics.get("calendar_days")}
- Date range: {metrics.get("start_date")} ~ {metrics.get("end_date")}
- Equal-weight return 20d: {_fmt_pct(metrics.get("return_20d"))}
- Equal-weight return 60d: {_fmt_pct(metrics.get("return_60d"))}
- Breadth 20d: {_fmt_pct(metrics.get("breadth_20d"))}
- Breadth 60d: {_fmt_pct(metrics.get("breadth_60d"))}
- Current drawdown: {_fmt_pct(metrics.get("drawdown"))}
- Daily volatility 20d: {_fmt_pct(metrics.get("volatility_20d"))}

## How To Use

- `risk_on`: ranking models can be evaluated with normal gross exposure.
- `neutral` / `defensive`: reduce position sizing and inspect drawdown before increasing exposure.
- `risk_off`: prioritize capital preservation; ranking-only stock selection is likely insufficient.
"""


def generate_report(data_dir: Path, out_dir: Path, lookback_days: int = 252, min_symbols: int = 100, limit: int = 0) -> dict:
    payload = analyze_market_regime(data_dir, lookback_days=lookback_days, min_symbols=min_symbols, limit=limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"market_regime_{ts}.json"
    md_path = out_dir / f"market_regime_{ts}.md"
    latest_json = out_dir / "market_regime_latest.json"
    latest_md = out_dir / "market_regime_latest.md"
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    markdown = build_markdown(payload)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--min-symbols", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="Optional symbol limit for smoke runs; 0 means all symbols.")
    args = parser.parse_args()
    data_dir = args.data_dir or MERGED_DIR or ashare_qfq_bars_dir()

    payload = generate_report(
        data_dir=data_dir,
        out_dir=args.out_dir,
        lookback_days=args.lookback_days,
        min_symbols=args.min_symbols,
        limit=args.limit,
    )
    print(f"Regime: {payload['regime']}")
    print(f"Cash ratio: {payload['cash_ratio']:.2f}")
    print(f"Latest markdown: {payload['artifacts']['latest_markdown']}")
    print(f"Latest JSON: {payload['artifacts']['latest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
