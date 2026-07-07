from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import china_astock_root as default_china_astock_root  # noqa: E402
from _paths import quant_data_root, reports_root  # noqa: E402


REQUIRED_PRICE_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
DEFAULT_INPUT_DIR = quant_data_root() / "parquet" / "ashare_1d" / "merged_baostock_jqdata_cleaned"
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_int(value: int | str | None, *, default: int, minimum: int, maximum: int | None = None) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"Expected integer >= {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"Expected integer <= {maximum}.")
    return parsed


def _parse_symbols(value: str | None) -> list[str]:
    if not value:
        return []
    symbols = []
    for part in str(value).replace(";", ",").split(","):
        symbol = part.strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _load_indicator_module(root: str | Path | None):
    project_root = Path(root) if root else default_china_astock_root()
    project_root = project_root.expanduser().resolve()
    indicator_file = project_root / "strategies" / "_indicators.py"
    if not indicator_file.is_file():
        raise FileNotFoundError(f"china-astock indicator module not found: {indicator_file}")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return importlib.import_module("strategies._indicators"), project_root


def _validate_price_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PRICE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Price data is missing required columns: {', '.join(missing)}")


def _read_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str})
    _validate_price_columns(df)
    if "symbol" not in df.columns:
        df["symbol"] = path.stem
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_input_dir(path: Path, symbols: list[str], limit: int | None) -> pd.DataFrame:
    if not path.is_dir():
        raise FileNotFoundError(f"Input parquet directory not found: {path}")
    wanted = {symbol.upper() for symbol in symbols}
    files = sorted(path.glob("*.parquet"))
    if wanted:
        files = [file for file in files if file.stem.upper() in wanted]
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError("No parquet files matched the requested symbols or limit.")

    frames = []
    for file in files:
        df = pd.read_parquet(file)
        _validate_price_columns(df)
        df = df.copy()
        df["symbol"] = file.stem.upper()
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _read_prices(
    *,
    input_csv: str | Path | None,
    input_dir: str | Path | None,
    symbols: list[str],
    limit: int | None,
) -> pd.DataFrame:
    if input_csv:
        df = _read_input_csv(Path(input_csv).expanduser().resolve())
        if symbols:
            df = df[df["symbol"].astype(str).str.upper().isin(symbols)]
        if limit is not None:
            keep = sorted(df["symbol"].astype(str).str.upper().unique())[:limit]
            df = df[df["symbol"].astype(str).str.upper().isin(keep)]
        if df.empty:
            raise ValueError("No CSV rows matched the requested symbols or limit.")
        return df
    return _read_input_dir(Path(input_dir or DEFAULT_INPUT_DIR).expanduser().resolve(), symbols, limit)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return round(parsed, 6)


def _tag_row(close: float | None, sma20: float | None, rsi14: float | None, bollinger_position: float | None) -> str:
    tags = []
    if close is not None and sma20 is not None:
        tags.append("trend_up" if close >= sma20 else "trend_down")
    if rsi14 is not None:
        if rsi14 >= 70:
            tags.append("rsi_hot")
        elif rsi14 <= 30:
            tags.append("rsi_cold")
        else:
            tags.append("rsi_neutral")
    if bollinger_position is not None:
        if bollinger_position >= 0.85:
            tags.append("near_upper_band")
        elif bollinger_position <= 0.15:
            tags.append("near_lower_band")
    return ",".join(tags) if tags else "insufficient_history"


def _latest_metric(series: pd.Series) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    return _safe_float(series.iloc[-1])


def _build_rows(df: pd.DataFrame, indicator_module: Any, lookback_days: int) -> list[dict[str, Any]]:
    rows = []
    normalized = df.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized = normalized.sort_values(["symbol", "date"])

    for symbol, group in normalized.groupby("symbol", sort=True):
        group = group.tail(lookback_days).copy()
        close = group["close"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        lower, mid, upper = indicator_module.bollinger(close, window=20, mult=2.0)
        band_width = upper - lower
        band_position = ((close - lower) / band_width.replace(0.0, pd.NA)).dropna()

        latest_close = _safe_float(close.iloc[-1]) if not close.empty else None
        sma20 = _latest_metric(indicator_module.sma(close, 20))
        ema20 = _latest_metric(indicator_module.ema(close, 20))
        rsi14 = _latest_metric(indicator_module.rsi(close, 14))
        atr14 = _latest_metric(indicator_module.atr(high, low, close, 14))
        bb_pos = _safe_float(band_position.iloc[-1]) if not band_position.empty else None
        rows.append(
            {
                "symbol": symbol,
                "last_date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
                "rows_used": int(len(group)),
                "close": latest_close,
                "sma20": sma20,
                "ema20": ema20,
                "rsi14": rsi14,
                "atr14": atr14,
                "bollinger_lower": _latest_metric(lower),
                "bollinger_mid": _latest_metric(mid),
                "bollinger_upper": _latest_metric(upper),
                "bollinger_position": bb_pos,
                "volume": _safe_float(group["volume"].iloc[-1]),
                "tags": _tag_row(latest_close, sma20, rsi14, bb_pos),
            }
        )
    return rows


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# china-astock 技术指标报告",
        "",
        f"- 生成时间：`{payload['generated_at']}`",
        "- 报告性质：只读研究报告，不连接券商，不读取密码，不自动下单。",
        f"- 数据来源：`{payload['source']}`",
        f"- china-astock-quant：`{payload['china_astock_root']}`",
        "",
        "| symbol | date | close | SMA20 | EMA20 | RSI14 | ATR14 | Bollinger位置 | tags |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {symbol} | {last_date} | {close} | {sma20} | {ema20} | {rsi14} | {atr14} | {bollinger_position} | {tags} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 边界",
            "- 本报告只复用 china-astock-quant 的指标函数，输出技术面观察。",
            "- 这里的标签不是买卖指令，也不是券商交易信号。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    *,
    input_csv: str | Path | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    china_astock_root: str | Path | None = None,
    timestamp: str | None = None,
    symbols: str | None = None,
    lookback_days: int | str | None = None,
    limit: int | str | None = None,
) -> dict[str, Any]:
    stamp = timestamp or _timestamp()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else reports_root().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_symbols = _parse_symbols(symbols)
    parsed_lookback = _as_int(lookback_days, default=120, minimum=20)
    parsed_limit = _as_int(limit, default=DEFAULT_LIMIT, minimum=1, maximum=MAX_LIMIT)
    indicators, project_root = _load_indicator_module(china_astock_root)
    prices = _read_prices(input_csv=input_csv, input_dir=input_dir, symbols=parsed_symbols, limit=parsed_limit)
    rows = _build_rows(prices, indicators, parsed_lookback)

    source = str(Path(input_csv).expanduser().resolve()) if input_csv else str(Path(input_dir or DEFAULT_INPUT_DIR).expanduser().resolve())
    payload = {
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "research_only": True,
        "trading_enabled": False,
        "source": source,
        "china_astock_root": str(project_root),
        "lookback_days": parsed_lookback,
        "limit": parsed_limit,
        "symbols_requested": parsed_symbols,
        "rows": rows,
    }

    out_json = out_dir / f"china_astock_indicator_report_{stamp}.json"
    out_md = out_dir / f"china_astock_indicator_report_{stamp}.md"
    latest_json = out_dir / "china_astock_indicator_report_latest.json"
    latest_md = out_dir / "china_astock_indicator_report_latest.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    copyfile(out_json, latest_json)
    copyfile(out_md, latest_md)

    status = {
        **payload,
        "output_json": str(out_json),
        "output_md": str(out_md),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a read-only china-astock technical indicator report.")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", default=None)
    parser.add_argument("--china-astock-root", default=None)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--lookback-days", default=None)
    parser.add_argument("--limit", default=None)
    args = parser.parse_args(argv)
    run_report(
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        china_astock_root=args.china_astock_root,
        symbols=args.symbols,
        lookback_days=args.lookback_days,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
