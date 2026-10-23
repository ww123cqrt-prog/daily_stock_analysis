"""TQM-20 策略逻辑对比：简单动量 vs TQM因子

用同一套quant_data数据、同一回测框架，对比两种选股逻辑：
- 策略A（简单动量）：20日收益率排名 + 20日均线过滤
- 策略B（TQM因子）：动量×效率 + 波动率过滤 + 40日高点回撤止损
"""

import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import json
from datetime import datetime

QUANT_DATA_ROOT = Path("/Users/cq/WorkingProjects/quant_data")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CASH = "CASH"
SYMBOLS = ("SZ159915", "SH513100", "SH512890")
SNAPSHOT_PATH = (
    QUANT_DATA_ROOT / "parquet" / "etf_1d" / "cn_etf_daily" / "snapshots"
    / "cn-etf-tencent-incremental-20260816T014347-8a017ab2"
)


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in SYMBOLS:
        path = SNAPSHOT_PATH / "qfq" / f"{symbol}.parquet"
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "open", "close"])
        df = df[(df["open"] > 0) & (df["close"] > 0)]
        df = df.sort_values("date").reset_index(drop=True)
        df["symbol"] = symbol
        data[symbol] = df
    return data


def merge_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for symbol in SYMBOLS:
        frame = data[symbol][["date", "open", "close"]].rename(
            columns={"open": f"{symbol}_open", "close": f"{symbol}_close"}
        )
        merged = frame if merged is None else merged.merge(frame, on="date", how="inner")
    return merged.sort_values("date").reset_index(drop=True)


# ── 策略A：简单动量+均线 ──

def simple_momentum_signal(merged: pd.DataFrame, lookback=20, ma_days=20) -> list[str]:
    signals = []
    for i in range(len(merged)):
        if i < lookback:
            signals.append(CASH)
            continue
        best_sym = None
        best_ret = -float("inf")
        for sym in SYMBOLS:
            close_now = merged[f"{sym}_close"].iloc[i]
            close_past = merged[f"{sym}_close"].iloc[i - lookback]
            ret = close_now / close_past - 1.0
            ma = merged[f"{sym}_close"].iloc[i - ma_days + 1 : i + 1].mean()
            if ret > best_ret and close_now > ma:
                best_ret = ret
                best_sym = sym
        signals.append(best_sym if best_sym else CASH)
    return signals


# ── 策略B：TQM因子 ──

def tqm_factor(closes: np.ndarray, k=2, vol_limit=0.40):
    values = np.asarray(closes, dtype=float)
    if len(values) < 21 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        return None
    values = values[-21:]
    returns = values[1:] / values[:-1] - 1.0
    mean_ret = float(np.mean(returns))
    var = float(np.mean((returns - mean_ret) ** 2))
    ann_vol = var ** 0.5 * 15.87
    if ann_vol <= 0 or ann_vol >= vol_limit:
        return None
    momentum = float(values[-1] / values[0] - 1.0)
    path = float(np.sum(np.abs(returns)))
    efficiency = min(abs(momentum) / path, 1.0) if path > 0 else 0.0
    score = momentum - (1.0 - efficiency ** k) * abs(momentum)
    return score


def tqm_signal(merged: pd.DataFrame, momentum_days=20, k=2, vol_limit=0.40,
               peak_days=40, drawdown_limit=0.12) -> list[str]:
    signals = []
    for i in range(len(merged)):
        # 回撤止损
        held = signals[-1] if signals and signals[-1] != CASH else None
        if held and i >= peak_days:
            held_close = merged[f"{held}_close"].iloc[i]
            peak_close = merged[f"{held}_close"].iloc[i - peak_days:i + 1].max()
            if peak_close > 0 and held_close < peak_close * (1 - drawdown_limit):
                signals.append(CASH)
                continue

        best_sym = None
        best_score = -float("inf")
        for sym in SYMBOLS:
            if i < momentum_days:
                continue
            window = merged[f"{sym}_close"].iloc[i - momentum_days:i + 1].values
            score = tqm_factor(window, k=k, vol_limit=vol_limit)
            if score is not None and score > best_score:
                best_score = score
                best_sym = sym
        signals.append(best_sym if best_sym else CASH)
    return signals


# ── 模拟引擎 ──

def simulate(merged: pd.DataFrame, signals: list[str], side_cost_bps=20.0) -> dict:
    target = [CASH] + signals[:-1]  # 延迟一天执行
    side_rate = side_cost_bps / 10000.0
    wealth = 1.0
    prev_pos = CASH
    curve = [{"date": merged.iloc[0]["date"], "wealth": wealth}]
    switches = 0
    invested = 0

    for i in range(len(merged) - 1):
        pos = target[i]
        if pos != prev_pos:
            switches += 1
        sides = (int(prev_pos != CASH) + int(pos != CASH)) if pos != prev_pos else 0
        cost = sides * side_rate

        if pos == CASH:
            gross = 0.0
        else:
            o_now = float(merged.iloc[i][f"{pos}_open"])
            o_next = float(merged.iloc[i + 1][f"{pos}_open"])
            gross = o_next / o_now - 1.0

        w_before = wealth
        wealth = wealth * (1 - cost) * (1 + gross)
        if pos != CASH:
            invested += 1
        curve.append({"date": merged.iloc[i + 1]["date"], "wealth": wealth})
        prev_pos = pos

    curve_df = pd.DataFrame(curve)
    dd = curve_df["wealth"] / curve_df["wealth"].cummax() - 1.0
    total_ret = wealth - 1.0
    periods = len(merged) - 1
    ann_ret = (1 + total_ret) ** (252 / periods) - 1 if periods > 0 else 0

    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "max_drawdown": float(dd.min()),
        "switch_count": switches,
        "invested_fraction": invested / periods if periods else 0,
        "custom_return_drawdown_score": ann_ret / (abs(float(dd.min())) + 0.01),
        "equity_curve": curve_df,
    }


def main():
    print("加载quant_data数据...")
    data = load_data()
    merged = merge_data(data)
    print(f"共同交易日: {len(merged)} 天 ({merged.iloc[0]['date'].date()} ~ {merged.iloc[-1]['date'].date()})")

    print("\n" + "=" * 80)
    print("策略A: 简单动量 + 均线过滤")
    print("=" * 80)
    sig_a = simple_momentum_signal(merged)
    res_a = simulate(merged, sig_a)

    print("策略B: TQM因子 + 波动率过滤 + 回撤止损")
    print("=" * 80)
    sig_b = tqm_signal(merged)
    res_b = simulate(merged, sig_b)

    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    rows = [
        ("总收益", "total_return", "%"),
        ("年化收益", "annualized_return", "%"),
        ("最大回撤", "max_drawdown", "%"),
        ("自定义收益回撤评分", "custom_return_drawdown_score", ""),
        ("交易次数", "switch_count", ""),
        ("投资比例", "invested_fraction", "%"),
    ]

    print("自定义收益回撤评分 = 年化收益 / (最大回撤绝对值 + 0.01)")
    print(f"\n{'指标':<12} {'简单动量(A)':<20} {'TQM因子(B)':<20} {'B-A差异':<15}")
    print("-" * 70)
    for name, key, unit in rows:
        va = res_a[key]
        vb = res_b[key]
        diff = vb - va
        if unit == "%":
            print(f"{name:<12} {va*100:>12.2f}%       {vb*100:>12.2f}%       {diff*100:>+10.2f}%")
        else:
            print(f"{name:<12} {va:>12.3f}       {vb:>12.3f}       {diff:>+10.3f}")

    print("\n" + "=" * 80)
    print("分析")
    print("=" * 80)

    if res_b["annualized_return"] > res_a["annualized_return"]:
        print("✓ TQM因子年化收益更高")
    else:
        print("✗ 简单动量年化收益更高")

    if abs(res_b["max_drawdown"]) < abs(res_a["max_drawdown"]):
        print("✓ TQM因子最大回撤更小")
    else:
        print("✗ 简单动量最大回撤更小")

    if res_b["custom_return_drawdown_score"] > res_a["custom_return_drawdown_score"]:
        print("✓ TQM因子自定义收益回撤评分更高")
    else:
        print("✗ 简单动量自定义收益回撤评分更高")

    if res_b["switch_count"] < res_a["switch_count"]:
        print("✓ TQM因子交易次数更少")
    else:
        print("✗ 简单动量交易次数更少")

    print(f"\n结论（按自定义收益回撤评分）: {'TQM因子' if res_b['custom_return_drawdown_score'] > res_a['custom_return_drawdown_score'] else '简单动量'}")

    # 保存
    output_dir = SCRIPT_DIR / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "data_source": "quant_data",
        "simple_momentum": {k: v for k, v in res_a.items() if k != "equity_curve"},
        "tqm_factor": {k: v for k, v in res_b.items() if k != "equity_curve"},
    }
    for v in out.values():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (np.floating, float)) and math.isfinite(float(vv)):
                    v[kk] = float(vv)
    path = output_dir / f"tqm_vs_simple_momentum_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    main()
