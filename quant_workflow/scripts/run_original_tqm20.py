"""用DSA正式回测引擎运行原始TQM-20脚本，计算完整收益指标"""
import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd

PROJECT = Path("/Users/cq/WorkingProjects/daily_stock_analysis")
QUANT_DATA = Path("/Users/cq/WorkingProjects/quant_data")
SNAPSHOT = QUANT_DATA / "parquet" / "etf_1d" / "cn_etf_daily" / "snapshots" / "cn-etf-tencent-incremental-20260816T014347-8a017ab2"

sys.path.insert(0, str(PROJECT))

from src.core.strategy_backtest import PriceBar, StrategyBacktestConfig, run_portfolio_backtest
from src.core.script_strategy import execute_strategy_script, view_from_bars

# 读取原始TQM-20脚本
source = (PROJECT / "quant_workflow" / "scratch" / "tqm20_strategy_script.py").read_text(encoding="utf-8")
print("=== 原始TQM-20 V1.6 正式版回测 ===")
print(f"脚本: quant_workflow/scratch/tqm20_strategy_script.py")

# 加载数据
SYMBOLS = ["SZ159915", "SH513100", "SH512890"]
bars = []
for sym in SYMBOLS:
    df = pd.read_parquet(SNAPSHOT / "qfq" / f"{sym}.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    for _, row in df.iterrows():
        bars.append(PriceBar(symbol=sym, date=row["date"].date(), open=float(row["open"]), close=float(row["close"])))
    print(f"  {sym}: {len(df)} bars")

# 运行脚本获取信号
result = execute_strategy_script(source, bars=bars, symbols=SYMBOLS)
print(f"\n信号数量: {len(result.allocations)}")

# 用DSA回测引擎计算收益
config = StrategyBacktestConfig(
    initial_capital=1_000_000.0,
    side_cost_pct=0.06,  # 万6 单边，与聚宽一致
    lot_size=1,
)

report = run_portfolio_backtest(
    strategy_name="TQM-20 V1.6",
    allocations=result.allocations,
    bars=bars,
    config=config,
)

# 打印关键指标
m = report.get("metrics", {})
print(f"\n=== 回测结果 ===")
print(f"回测区间: {m.get('start_date', 'N/A')} ~ {m.get('end_date', 'N/A')}")
print(f"交易日数: {m.get('trading_days', 'N/A')}")
print(f"初始资金: {config.initial_capital:,.0f}")
print(f"最终净值: {m.get('final_equity', 0):,.2f}")
print(f"总收益:   {m.get('total_return_pct', 0):.2f}%")
print(f"年化收益: {m.get('annualized_return_pct', 0):.2f}%")
print(f"最大回撤: {m.get('max_drawdown_pct', 0):.2f}%")
print(f"夏普比率: {m.get('sharpe_ratio', 0):.3f}")
print(f"交易次数: {m.get('trade_count', 0)}")
print(f"总成本:   {m.get('total_cost', 0):,.2f}")

# 与聚宽结果对比
print(f"\n=== 与聚宽回测对比 ===")
print(f"{'指标':<12} {'本地DSA':<18} {'聚宽平台':<18}")
print("-" * 50)
print(f"{'总收益':<12} {m.get('total_return_pct', 0):>10.2f}%       {'1707.98%':>10}")
print(f"{'年化收益':<12} {m.get('annualized_return_pct', 0):>10.2f}%       {'46.72%':>10}")
print(f"{'最大回撤':<12} {m.get('max_drawdown_pct', 0):>10.2f}%       {'-18.78%':>10}")
print(f"{'夏普比率':<12} {m.get('sharpe_ratio', 0):>10.3f}        {'1.872':>10}")

# 保存
output_dir = PROJECT / "quant_workflow" / "scripts" / "backtest_results"
output_dir.mkdir(exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = {
    "timestamp": ts,
    "script": "quant_workflow/scratch/tqm20_strategy_script.py",
    "version": "TQM-20 V1.6 正式版",
    "data_source": "quant_data",
    "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in m.items()},
    "signal_count": len(result.allocations),
}
path = output_dir / f"original_tqm20_dsa_{ts}.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n结果已保存: {path}")
