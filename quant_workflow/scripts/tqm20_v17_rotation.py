"""TQM-20 V1.7 four-ETF candidate-family research helpers.

V1.7 is one experiment: keep the frozen three-ETF TQM-20 rules and add
exactly one fourth asset. Gold, S&P 500, Germany DAX, Hang Seng Tech, and
China 30-year Treasury are candidate branches, not sequential versions.
Formal-data availability and verdicts are recorded by the Round 088 audit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


CASH = "CASH"
BASE_SYMBOLS: tuple[str, ...] = ("SZ159915", "SH513100", "SH512890")
V17_VARIANTS: dict[str, dict[str, str]] = {
    "gold": {"symbol": "SH518880", "name": "黄金ETF"},
    "sp500": {"symbol": "SH513500", "name": "标普500ETF"},
    "germany_dax": {"symbol": "SH513030", "name": "德国DAX ETF"},
    "hang_seng_tech": {"symbol": "SH513180", "name": "恒生科技ETF"},
    "china_30y_treasury": {"symbol": "SH511090", "name": "30年国债ETF"},
}
DEFAULT_VARIANT = "gold"


def build_v17_universe(variant: str = DEFAULT_VARIANT) -> tuple[str, ...]:
    """Return the base universe plus one audited V1.7 fourth-asset branch."""
    config = V17_VARIANTS.get(str(variant).strip().lower())
    if config is None:
        choices = ", ".join(sorted(V17_VARIANTS))
        raise ValueError(f"unknown V1.7 variant {variant!r}; expected one of: {choices}")
    return (*BASE_SYMBOLS, config["symbol"])


DEFAULT_SYMBOLS: tuple[str, ...] = build_v17_universe()
ELIGIBILITY_MODES = frozenset({"leader_then_filter", "filter_then_rank"})
REQUIRED_COLUMNS = {"date", "open", "close"}

# TQM-20 参数
TQM_MOMENTUM_DAYS = 20
TQM_K = 2
TQM_VOL_LIMIT = 0.40
TQM_PEAK_DAYS = 40
TQM_DRAWDOWN_LIMIT = 0.12


def normalize_price_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Normalize a source frame and reject duplicate or unusable bars."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol} is missing columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in ("open", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "open", "close"])
    result = result[(result["open"] > 0) & (result["close"] > 0)]
    result = result.sort_values("date", kind="stable").reset_index(drop=True)
    if result["date"].duplicated().any():
        duplicates = (
            result.loc[result["date"].duplicated(), "date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )
        raise ValueError(f"{symbol} contains duplicate dates: {duplicates[:5]}")
    if result.empty:
        raise ValueError(f"{symbol} has no usable price rows")
    return result


def _tqm_factor(closes: np.ndarray | list, k: float = 2, vol_limit: float = 0.40) -> dict | None:
    """计算TQM因子，返回因子明细或None（如果不符合条件）。"""
    values = np.asarray(closes, dtype=float)
    if len(values) < 21 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return None

    values = values[-21:]
    returns = values[1:] / values[:-1] - 1.0
    mean_return = float(np.mean(returns))
    variance = float(np.mean((returns - mean_return) ** 2))
    annualized_volatility = variance ** 0.5 * 15.87
    
    if annualized_volatility <= 0 or annualized_volatility >= vol_limit:
        return None

    momentum = float(values[-1] / values[0] - 1.0)
    path = float(np.sum(np.abs(returns)))
    efficiency = min(abs(momentum) / path, 1.0) if path > 0 else 0.0
    score = momentum - (1.0 - efficiency ** k) * abs(momentum)
    
    return {
        "momentum": momentum,
        "efficiency": efficiency,
        "annualized_volatility": annualized_volatility,
        "score": score,
    }


def build_tqm_signal_table(
    data: Mapping[str, pd.DataFrame],
    *,
    momentum_days: int = TQM_MOMENTUM_DAYS,
    k: float = TQM_K,
    vol_limit: float = TQM_VOL_LIMIT,
    peak_days: int = TQM_PEAK_DAYS,
    drawdown_limit: float = TQM_DRAWDOWN_LIMIT,
) -> pd.DataFrame:
    """构建TQM-20信号表。"""
    if len(data) < 2:
        raise ValueError("the strategy requires at least two ETF price series")

    normalized = {
        symbol: normalize_price_frame(frame, symbol=symbol)
        for symbol, frame in data.items()
    }
    symbols = tuple(data)
    merged: pd.DataFrame | None = None
    for symbol in symbols:
        frame = normalized[symbol][["date", "open", "close"]].rename(
            columns={"open": f"{symbol}_open", "close": f"{symbol}_close"}
        )
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on="date", how="inner", validate="one_to_one")
        )
    assert merged is not None
    merged = merged.sort_values("date", kind="stable").reset_index(drop=True)

    # 计算每个资产的TQM分数
    tqm_scores: dict[str, pd.Series] = {}
    tqm_details: dict[str, list[dict | None]] = {}
    
    for symbol in symbols:
        close = merged[f"{symbol}_close"]
        scores = []
        for i in range(len(close)):
            if i < momentum_days:
                scores.append(None)
            else:
                window = close.iloc[i - momentum_days:i + 1].values
                factor = _tqm_factor(window, k=k, vol_limit=vol_limit)
                scores.append(factor["score"] if factor else None)
        tqm_scores[symbol] = pd.Series(scores, index=merged.index)
        tqm_details[symbol] = scores

    # 构建信号
    signals = []
    for i in range(len(merged)):
        # 检查是否触发回撤止损
        held_symbol = signals[-1] if signals and signals[-1] != CASH else None
        
        if held_symbol and i >= peak_days:
            held_close = merged[f"{held_symbol}_close"].iloc[i]
            peak_close = merged[f"{held_symbol}_close"].iloc[i - peak_days:i + 1].max()
            if peak_close > 0 and held_close < peak_close * (1 - drawdown_limit):
                signals.append(CASH)
                continue
        
        # 选择TQM分数最高的资产
        best_symbol = None
        best_score = -float('inf')
        for symbol in symbols:
            score = tqm_scores[symbol].iloc[i]
            if score is not None and score > best_score:
                best_score = score
                best_symbol = symbol
        
        signals.append(best_symbol if best_symbol else CASH)

    merged["signal"] = signals
    merged["target_position"] = [CASH] + signals[:-1]  # 延迟一天执行
    return merged


def simulate_open_to_open(
    signal_table: pd.DataFrame,
    *,
    side_cost_bps: float = 0.0,
) -> dict[str, object]:
    """模拟开盘到开盘的收益。"""
    if side_cost_bps < 0:
        raise ValueError("side_cost_bps must be nonnegative")
    required = {"date", "target_position"}
    if not required.issubset(signal_table.columns):
        raise ValueError("signal_table must contain date and target_position")
    frame = signal_table.sort_values("date", kind="stable").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("at least two common trading days are required")

    symbols = [col.replace("_open", "") for col in frame.columns if col.endswith("_open")]
    wealth = 1.0
    previous_position = CASH
    curve: list[dict[str, object]] = [{"date": frame.iloc[0]["date"], "wealth": wealth}]
    total_cost = 0.0
    total_turnover = 0.0
    switches = 0
    invested_periods = 0
    period_rows: list[dict[str, object]] = []
    side_rate = side_cost_bps / 10000.0

    for index in range(len(frame) - 1):
        row = frame.iloc[index]
        next_row = frame.iloc[index + 1]
        position = str(row["target_position"])
        sides = (
            (int(previous_position != CASH) + int(position != CASH))
            if position != previous_position
            else 0
        )
        if position != previous_position:
            switches += 1
        turnover = float(sides)
        cost_rate = turnover * side_rate
        
        # 计算收益
        if position == CASH:
            gross_return = 0.0
        else:
            open_now = float(frame.iloc[index][f"{position}_open"])
            open_next = float(frame.iloc[index + 1][f"{position}_open"])
            gross_return = open_next / open_now - 1.0
        
        wealth_before = wealth
        wealth = wealth * (1.0 - cost_rate) * (1.0 + gross_return)
        net_return = wealth / wealth_before - 1.0
        total_cost += cost_rate
        total_turnover += turnover
        if position != CASH:
            invested_periods += 1
        period_rows.append(
            {
                "date": row["date"],
                "next_date": next_row["date"],
                "position": position,
                "previous_position": previous_position,
                "gross_return": gross_return,
                "transaction_cost_rate": cost_rate,
                "net_return": net_return,
                "wealth": wealth,
            }
        )
        curve.append({"date": next_row["date"], "wealth": wealth})
        previous_position = position

    curve_frame = pd.DataFrame(curve)
    running_max = curve_frame["wealth"].cummax()
    drawdown = curve_frame["wealth"] / running_max - 1.0
    period_frame = pd.DataFrame(period_rows)
    total_return = float(wealth - 1.0)
    periods = len(period_frame)
    
    def _annualized(total_return: float, periods: int) -> float:
        if periods <= 0:
            return float("nan")
        wealth = 1.0 + total_return
        if wealth <= 0:
            return -1.0
        return wealth ** (252.0 / periods) - 1.0
    
    return {
        "metrics": {
            "periods": periods,
            "start_date": str(pd.Timestamp(curve_frame.iloc[0]["date"]).date()),
            "end_date": str(pd.Timestamp(curve_frame.iloc[-1]["date"]).date()),
            "total_return": total_return,
            "annualized_return": _annualized(total_return, periods),
            "max_drawdown": float(drawdown.min()),
            "invested_fraction": invested_periods / periods
            if periods
            else float("nan"),
            "switch_count": switches,
            "total_turnover_sides": total_turnover,
            "total_transaction_cost_fraction": total_cost,
            "side_cost_bps": side_cost_bps,
        },
        "periods": period_frame,
        "equity_curve": curve_frame,
    }


def load_price_data(
    data_dir: Path, symbols: tuple[str, ...] = DEFAULT_SYMBOLS
) -> dict[str, pd.DataFrame]:
    """加载ETF价格数据。"""
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"ETF data is missing: {path}")
        result[symbol] = normalize_price_frame(pd.read_parquet(path), symbol=symbol)
    return result


def sanitize_json_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    return value
