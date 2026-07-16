#!/usr/bin/env python3
"""
Qlib策略优化实验：低换手 + 市场择时 + 风险控制

基于现有Alpha158+LGB基线，逐步添加：
1. 换手率限制（调仓频率 + turnover cap）
2. 市场择时（MA趋势 + 波动率仓位）
3. 风险控制（回撤保护 + 行业分散）

Usage:
    python scripts/optimize_strategy.py                    # 运行所有实验
    python scripts/optimize_strategy.py --phase 1          # 只运行Phase 1
    python scripts/optimize_strategy.py --skip-training     # 跳过训练，直接用已有预测
    python scripts/optimize_strategy.py --dry-run           # 只跑内置小样本预演
"""
import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import argparse
import shutil
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, qlib_provider_uri, quant_data_root, reports_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
CONFIGS_DIR = WORKFLOW_ROOT / "configs"
REPORT_DIR = reports_root()
MLRUNS_DIR = mlruns_root()
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
QUANT_DATA = quant_data_root()
DATA_DIR = qlib_provider_uri()

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

def configure_mlflow_env() -> None:
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI


def init_qlib(data_dir: Path = DATA_DIR):
    """Initialize Qlib lazily so browser-safe dry-runs can import this module."""
    configure_mlflow_env()
    import qlib

    qlib.init(provider_uri=str(data_dir), region="cn")
    return qlib


# ═══════════════════════════════════════════════════════════
#  Step 0: 训练基线模型并获取预测
# ═══════════════════════════════════════════════════════════

def train_and_predict(config_path: str, experiment_name: str):
    """训练模型并返回预测结果。"""
    configure_mlflow_env()
    from qlib.cli.run import workflow

    print(f"\n{'='*60}")
    print(f"[训练] Config: {config_path}")
    print(f"[训练] Experiment: {experiment_name}")
    print(f"{'='*60}\n")

    workflow(config_path, experiment_name=experiment_name, uri_folder=str(MLRUNS_DIR))

    # 从MLflow加载预测结果
    import mlflow
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        experiments = client.search_experiments()
        exp = next((item for item in experiments if item.name == experiment_name), None)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run = runs[0]

    # 下载预测结果
    pred_path = client.download_artifacts(run.info.run_id, "pred.pkl")
    pred = pd.read_pickle(pred_path)
    label_path = client.download_artifacts(run.info.run_id, "label.pkl")
    label = pd.read_pickle(label_path)

    return pred, label, run.data.metrics


def load_existing_predictions(experiment_name: str = "baseline_lgb_alpha158"):
    """从已有MLflow run加载预测，不重新训练。"""
    configure_mlflow_env()
    import mlflow
    import yaml as yaml_mod

    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        experiments = client.search_experiments()
        exp = next((item for item in experiments if item.name == experiment_name), None)
    if exp is None:
        return None, None, {}

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        # Fallback: 直接从文件系统读取
        exp_dir = MLRUNS_DIR / exp.experiment_id
        run_dirs = sorted(
            [p for p in exp_dir.iterdir() if p.is_dir() and (p / "meta.yaml").exists()],
            key=lambda p: (p / "meta.yaml").stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            return None, None, {}
        run_dir = run_dirs[0]
        pred = pd.read_pickle(run_dir / "artifacts" / "pred.pkl")
        label = pd.read_pickle(run_dir / "artifacts" / "label.pkl")
        return pred, label, {}

    run = runs[0]
    pred_path = client.download_artifacts(run.info.run_id, "pred.pkl")
    pred = pd.read_pickle(pred_path)
    label_path = client.download_artifacts(run.info.run_id, "label.pkl")
    label = pd.read_pickle(label_path)

    return pred, label, run.data.metrics


# ═══════════════════════════════════════════════════════════
#  市场指标获取
# ═══════════════════════════════════════════════════════════

def get_market_indicators(start_time: str, end_time: str):
    """获取沪深300指数的MA趋势和波动率指标。"""
    try:
        init_qlib()
        from qlib.data import D

        # 获取沪深300指数收盘价
        bench = D.features(
            instruments=["SH000300"],
            fields=["$close", "$volume"],
            start_time=start_time,
            end_time=end_time,
        )
        if bench.empty:
            print("[警告] 无法获取SH000300数据，跳过市场择时")
            return None

        bench = bench.droplevel("instrument")
        bench.columns = ["close", "volume"]

        # 计算技术指标
        bench["ma20"] = bench["close"].rolling(20).mean()
        bench["ma60"] = bench["close"].rolling(60).mean()
        bench["ma120"] = bench["close"].rolling(120).mean()
        bench["ma200"] = bench["close"].rolling(200).mean()

        # 趋势信号：收盘价在MA上方=1，下方=0
        bench["trend_20"] = (bench["close"] > bench["ma20"]).astype(float)
        bench["trend_60"] = (bench["close"] > bench["ma60"]).astype(float)
        bench["trend_200"] = (bench["close"] > bench["ma200"]).astype(float)

        # 20日波动率（年化）
        bench["volatility_20"] = bench["close"].pct_change().rolling(20).std() * np.sqrt(252)

        # 波动率分位数（滚动6个月）
        bench["vol_percentile"] = bench["volatility_20"].rolling(120).rank(pct=True)

        return bench
    except Exception as e:
        print(f"[警告] 获取市场指标失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  策略实现
# ═══════════════════════════════════════════════════════════

class OptimizedStrategy:
    """带市场择时和风控的TopK策略。"""

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        cost_bps: float = 15,
        rebalance_freq: int = 1,
        turnover_cap: float = 0.20,
        # 市场择时
        trend_filter: bool = False,
        trend_ma: str = "ma200",
        trend_reduce_pct: float = 0.5,
        vol_sizing: bool = False,
        vol_reduce_threshold: float = 0.8,
        # 风险控制
        max_drawdown_stop: float = -0.08,
        max_dd_cooldown_days: int = 20,
        max_single_weight: float = 0.05,
        max_sector_weight: float = 0.30,
        # 行业信息
        sector_map: dict = None,
    ):
        self.topk = topk
        self.n_drop = n_drop
        self.cost_bps = cost_bps
        self.rebalance_freq = rebalance_freq
        self.turnover_cap = turnover_cap
        self.trend_filter = trend_filter
        self.trend_ma = trend_ma
        self.trend_reduce_pct = trend_reduce_pct
        self.vol_sizing = vol_sizing
        self.vol_reduce_threshold = vol_reduce_threshold
        self.max_drawdown_stop = max_drawdown_stop
        self.max_dd_cooldown_days = max_dd_cooldown_days
        self.max_single_weight = max_single_weight
        self.max_sector_weight = max_sector_weight
        self.sector_map = sector_map or {}

    def run(
        self,
        pred: pd.DataFrame,
        label: pd.DataFrame,
        market_indicators: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """运行策略回测。"""
        pred_label = pd.concat([pred, label], axis=1).dropna()
        pred_label.columns = ["score", "label"]

        daily_results = []
        prev_holdings = set()
        cum_return = 1.0
        peak_return = 1.0
        in_cooldown = False
        cooldown_counter = 0
        day_counter = 0
        rebalance_day = 0

        dates = sorted(pred_label.index.get_level_values("datetime").unique())

        for dt in dates:
            day_data = pred_label.loc[dt].copy()
            if len(day_data) < self.topk:
                continue

            day_counter += 1

            # ── 市场择时：仓位系数 ──
            position_pct = 1.0

            if market_indicators is not None and dt in market_indicators.index:
                mi = market_indicators.loc[dt]

                # 趋势过滤
                if self.trend_filter and not pd.isna(mi.get(self.trend_ma, np.nan)):
                    trend_signal = mi.get(f"trend_{self.trend_ma.replace('ma', '')}", 1.0)
                    if trend_signal < 0.5:  # 趋势向下
                        position_pct *= (1.0 - self.trend_reduce_pct)

                # 波动率仓位调整
                if self.vol_sizing and not pd.isna(mi.get("vol_percentile", np.nan)):
                    vol_pct = mi["vol_percentile"]
                    if vol_pct > self.vol_reduce_threshold:
                        # 波动率越高，仓位越低
                        vol_factor = 1.0 - (vol_pct - self.vol_reduce_threshold) / (1.0 - self.vol_reduce_threshold) * 0.5
                        position_pct *= vol_factor

            # ── 回撤保护 ──
            if self.max_drawdown_stop < 0:
                current_dd = cum_return / peak_return - 1
                if current_dd < self.max_drawdown_stop and not in_cooldown:
                    in_cooldown = True
                    cooldown_counter = 0
                    position_pct = 0.0  # 清仓

                if in_cooldown:
                    cooldown_counter += 1
                    # 回撤恢复或冷却期结束后恢复
                    if current_dd > self.max_drawdown_stop * 0.5 or cooldown_counter > self.max_dd_cooldown_days:
                        in_cooldown = False
                    else:
                        position_pct = 0.0

            # ── 调仓频率 ──
            rebalance_day += 1
            if self.rebalance_freq > 1 and rebalance_day % self.rebalance_freq != 0 and prev_holdings:
                # 非调仓日，持有不变
                port_data = day_data.loc[day_data.index.isin(prev_holdings)]
                if len(port_data) > 0:
                    port_ret = port_data["label"].mean() * position_pct
                    cum_return *= (1 + port_ret)
                    peak_return = max(peak_return, cum_return)
                    daily_results.append({
                        "date": dt,
                        "return": port_ret,
                        "gross_return": port_data["label"].mean(),
                        "turnover": 0.0,
                        "position_pct": position_pct,
                        "n_stocks": len(prev_holdings),
                    })
                continue

            # ── 选股 ──
            if position_pct < 0.01:
                # 不持仓
                daily_results.append({
                    "date": dt,
                    "return": 0.0,
                    "gross_return": 0.0,
                    "turnover": len(prev_holdings) / max(self.topk, 1) if prev_holdings else 0.0,
                    "position_pct": 0.0,
                    "n_stocks": 0,
                })
                prev_holdings = set()
                continue

            top_stocks = set(day_data.nlargest(self.topk, "score").index)

            # ── 换手率限制 ──
            new_stocks = top_stocks - prev_holdings
            removed_stocks = prev_holdings - top_stocks
            turnover = (len(new_stocks) + len(removed_stocks)) / (2 * max(self.topk, 1))

            if self.turnover_cap and turnover > self.turnover_cap and prev_holdings:
                n_keep = int(self.topk * (1 - self.turnover_cap))
                keep = set(s for s in prev_holdings if s in day_data.index)
                if len(keep) > n_keep:
                    # 保留排名最高的旧持仓
                    keep_scores = day_data.loc[list(keep), "score"].nlargest(n_keep)
                    keep = set(keep_scores.index)
                n_new = self.topk - len(keep)
                candidates = day_data.drop(index=list(keep), errors="ignore").nlargest(n_new, "score")
                top_stocks = keep | set(candidates.index)
                turnover = min(turnover, self.turnover_cap)

            # ── 组合收益 ──
            port_data = day_data.loc[day_data.index.isin(top_stocks)]
            port_ret = port_data["label"].mean() * position_pct
            cost = turnover * (self.cost_bps / 10000)
            net_ret = port_ret - cost

            cum_return *= (1 + net_ret)
            peak_return = max(peak_return, cum_return)

            daily_results.append({
                "date": dt,
                "return": net_ret,
                "gross_return": port_data["label"].mean(),
                "turnover": turnover,
                "position_pct": position_pct,
                "n_stocks": len(top_stocks),
            })
            prev_holdings = top_stocks

        return pd.DataFrame(daily_results).set_index("date")


def print_stats(bt: pd.DataFrame, label: str):
    """打印回测统计指标。"""
    if bt.empty:
        print(f"  {label}: 数据不足")
        return {}

    cum_ret = (1 + bt["return"]).cumprod()
    total_ret = cum_ret.iloc[-1] - 1
    n_years = len(bt) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = bt["return"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    avg_turnover = bt["turnover"].mean()
    avg_position = bt["position_pct"].mean() if "position_pct" in bt.columns else 1.0

    # 年化交易成本估算
    annual_cost = avg_turnover * (15 / 10000) * 252
    gross_ann_ret = ann_ret + annual_cost

    stats = {
        "days": len(bt),
        "total_return": total_ret,
        "annual_return": ann_ret,
        "gross_annual_return": gross_ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "avg_position": avg_position,
        "annual_cost": annual_cost,
        "cost_ratio": annual_cost / gross_ann_ret if gross_ann_ret != 0 else 0,
    }

    print(f"\n  {label}:")
    print(f"    交易天数:     {stats['days']}")
    print(f"    总收益:       {stats['total_return']:.2%}")
    print(f"    年化收益(含成本): {stats['annual_return']:.2%}")
    print(f"    年化收益(不含成本): {stats['gross_annual_return']:.2%}")
    print(f"    年化波动率:   {stats['annual_vol']:.2%}")
    print(f"    Sharpe比率:   {stats['sharpe']:.2f}")
    print(f"    最大回撤:     {stats['max_drawdown']:.2%}")
    print(f"    平均换手率:   {stats['avg_turnover']:.2%}")
    print(f"    平均仓位:     {stats['avg_position']:.2%}")
    print(f"    年化交易成本: {stats['annual_cost']:.2%}")
    print(f"    成本占比:     {stats['cost_ratio']:.2%}")

    return stats


def validate_params(phase: int, topk: int) -> tuple[int, int]:
    phase = int(phase)
    topk = int(topk)
    if phase < 0 or phase > 4:
        raise ValueError("phase must be between 0 and 4")
    if topk < 1 or topk > 500:
        raise ValueError("topk must be between 1 and 500")
    return phase, topk


def build_dry_run_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create deterministic tiny prediction/label data for web smoke checks."""
    dates = pd.bdate_range("2026-01-02", periods=90)
    instruments = [f"SH600{i:03d}" for i in range(80)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    rows = []
    labels = []
    for day_idx, _day in enumerate(dates):
        market_wave = 0.001 if day_idx % 11 < 7 else -0.0015
        for inst_idx, _symbol in enumerate(instruments):
            score = (inst_idx % 17) / 17.0 + (day_idx % 5) * 0.01
            label_value = market_wave + ((inst_idx % 9) - 4) * 0.00035
            rows.append(score)
            labels.append(label_value)

    pred = pd.DataFrame({"score": rows}, index=index)
    label = pd.DataFrame({"label": labels}, index=index)
    close = pd.Series(100 + pd.RangeIndex(len(dates)).to_numpy() * 0.08, index=dates)
    market = pd.DataFrame(index=dates)
    market["close"] = close
    market["ma20"] = close.rolling(20, min_periods=1).mean()
    market["ma60"] = close.rolling(60, min_periods=1).mean()
    market["ma120"] = close.rolling(60, min_periods=1).mean()
    market["ma200"] = close.rolling(60, min_periods=1).mean()
    market["trend_20"] = 1.0
    market["trend_60"] = 1.0
    market["trend_120"] = 1.0
    market["trend_200"] = 1.0
    market["volatility_20"] = 0.18
    market["vol_percentile"] = 0.35
    return pred, label, market, {"dry_run.prediction_rows": float(len(pred))}


def run_phase_experiments(
    pred: pd.DataFrame,
    label: pd.DataFrame,
    market_indicators: pd.DataFrame | None,
    *,
    phase: int,
    topk: int,
) -> dict:
    all_results: dict = {}

    if phase == 0 or phase == 1:
        configs = [
            ("每日调仓+20%换手限制", 1, 0.20),
            ("每3日调仓+20%换手限制", 3, 0.20),
            ("每周调仓+20%换手限制", 5, 0.20),
            ("每2周调仓+10%换手限制", 10, 0.10),
            ("每月调仓+5%换手限制", 20, 0.05),
        ]
        for name, freq, cap in configs:
            strategy = OptimizedStrategy(topk=topk, rebalance_freq=freq, turnover_cap=cap, cost_bps=15)
            all_results[f"phase1_{name}"] = print_stats(strategy.run(pred, label), name)

    if phase == 0 or phase == 2:
        configs = [
            ("MA200趋势过滤(降仓50%)", True, "ma200", 0.5, False, 0.8),
            ("MA120趋势过滤(降仓50%)", True, "ma120", 0.5, False, 0.8),
            ("MA60趋势过滤(降仓30%)", True, "ma60", 0.3, False, 0.8),
            ("MA200+波动率仓位", True, "ma200", 0.5, True, 0.8),
            ("MA200+激进波动率", True, "ma200", 0.5, True, 0.6),
        ]
        for name, trend, ma, reduce, vol, vol_th in configs:
            strategy = OptimizedStrategy(
                topk=topk,
                rebalance_freq=5,
                turnover_cap=0.20,
                cost_bps=15,
                trend_filter=trend,
                trend_ma=ma,
                trend_reduce_pct=reduce,
                vol_sizing=vol,
                vol_reduce_threshold=vol_th,
            )
            all_results[f"phase2_{name}"] = print_stats(strategy.run(pred, label, market_indicators), name)

    if phase == 0 or phase == 3:
        configs = [
            ("-8%回撤止损+20天冷却", -0.08, 20),
            ("-5%回撤止损+10天冷却", -0.05, 10),
            ("-10%回撤止损+30天冷却", -0.10, 30),
        ]
        for name, mdd, cooldown in configs:
            strategy = OptimizedStrategy(
                topk=topk,
                rebalance_freq=5,
                turnover_cap=0.20,
                cost_bps=15,
                max_drawdown_stop=mdd,
                max_dd_cooldown_days=cooldown,
            )
            all_results[f"phase3_{name}"] = print_stats(strategy.run(pred, label, market_indicators), name)

    if phase == 0 or phase == 4:
        strategy = OptimizedStrategy(
            topk=topk,
            rebalance_freq=5,
            turnover_cap=0.15,
            cost_bps=15,
            trend_filter=True,
            trend_ma="ma200",
            trend_reduce_pct=0.5,
            vol_sizing=True,
            vol_reduce_threshold=0.8,
            max_drawdown_stop=-0.08,
            max_dd_cooldown_days=20,
        )
        all_results["phase4_combined"] = print_stats(strategy.run(pred, label, market_indicators), "综合最优策略")

        strategy_conservative = OptimizedStrategy(
            topk=min(topk, 30),
            rebalance_freq=10,
            turnover_cap=0.10,
            cost_bps=15,
            trend_filter=True,
            trend_ma="ma120",
            trend_reduce_pct=0.6,
            vol_sizing=True,
            vol_reduce_threshold=0.7,
            max_drawdown_stop=-0.06,
            max_dd_cooldown_days=30,
        )
        all_results["phase4_conservative"] = print_stats(
            strategy_conservative.run(pred, label, market_indicators),
            "保守综合策略",
        )

    return all_results


def _summary_rows(results: dict) -> list[dict]:
    rows = []
    for name, stats in results.items():
        if not stats:
            continue
        rows.append({
            "experiment": name,
            "annual_return": stats.get("annual_return", 0),
            "sharpe": stats.get("sharpe", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "avg_turnover": stats.get("avg_turnover", 0),
            "avg_position": stats.get("avg_position", 0),
            "cost_ratio": stats.get("cost_ratio", 0),
        })
    return rows


def write_reports(
    results: dict,
    *,
    output_dir: Path,
    timestamp: str,
    phase: int,
    topk: int,
    dry_run: bool,
    base_metrics: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / f"optimization_results_{timestamp}.json"
    result_md = output_dir / f"optimization_results_{timestamp}.md"
    latest_json = output_dir / "optimization_results_latest.json"
    latest_md = output_dir / "optimization_results_latest.md"
    payload = {
        "status": "completed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "topk": topk,
        "dry_run": dry_run,
        "metric_basis": {
            "candidate_return": "absolute_net_return",
            "base_metrics": "qlib_excess_return_vs_benchmark",
            "direct_comparison_allowed": False,
        },
        "base_metrics": base_metrics,
        "results": results,
        "summary": _summary_rows(results),
    }
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Qlib 策略优化实验报告",
        "",
        "只读策略优化预演" if dry_run else "只读策略优化实验",
        "",
        f"- Phase: {phase}",
        f"- TopK: {topk}",
        f"- Dry-run: {dry_run}",
        "",
        "口径：候选策略使用自定义模拟器计算绝对净收益；base_metrics 中的 Qlib excess_return 是相对沪深300的超额收益，两者不能直接横向比较。",
        "",
        "| 实验 | 年化收益 | Sharpe | 最大回撤 | 平均换手 | 平均仓位 | 成本占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {experiment} | {annual_return:.2%} | {sharpe:.2f} | {max_drawdown:.2%} | "
            "{avg_turnover:.2%} | {avg_position:.2%} | {cost_ratio:.2%} |".format(**row)
        )
    if not payload["summary"]:
        lines.append("| 无有效结果 | 0.00% | 0.00 | 0.00% | 0.00% | 0.00% | 0.00% |")
    lines.extend([
        "",
        "说明：本报告只读生成，不连接券商，不读取交易密码，不自动下单。",
    ])
    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(result_json, latest_json)
    shutil.copyfile(result_md, latest_md)
    return {
        "status": "completed",
        "dry_run": dry_run,
        "output_json": str(result_json.resolve()),
        "output_md": str(result_md.resolve()),
        "latest_json": str(latest_json.resolve()),
        "latest_md": str(latest_md.resolve()),
    }


def run_report(
    *,
    phase: int = 0,
    topk: int = 50,
    skip_training: bool = False,
    dry_run: bool = False,
    output_dir: Path = REPORT_DIR,
    timestamp: str | None = None,
) -> dict:
    phase, topk = validate_params(phase, topk)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    if dry_run:
        pred, label, market_indicators, base_metrics = build_dry_run_inputs()
    else:
        init_qlib()
        if skip_training:
            pred, label, base_metrics = load_existing_predictions()
            if pred is None:
                raise FileNotFoundError("未找到已有预测，请先运行训练，或使用 --dry-run 做网页预演")
        else:
            config_path = str(CONFIGS_DIR / "baseline_lgb_alpha158.yaml")
            pred, label, base_metrics = train_and_predict(config_path, "baseline_lgb_alpha158")

        dates = sorted(pred.index.get_level_values("datetime").unique())
        start_time = str(dates[0].date()) if hasattr(dates[0], "date") else str(dates[0])[:10]
        end_time = str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1])[:10]
        market_indicators = get_market_indicators(start_time, end_time)

    results = run_phase_experiments(pred, label, market_indicators, phase=phase, topk=topk)
    return write_reports(
        results,
        output_dir=Path(output_dir),
        timestamp=timestamp,
        phase=phase,
        topk=topk,
        dry_run=dry_run,
        base_metrics=base_metrics,
    )


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Qlib策略优化实验")
    parser.add_argument("--phase", type=int, default=0, help="运行特定Phase (0=全部)")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练，使用已有预测")
    parser.add_argument("--dry-run", action="store_true", help="使用内置小样本，只做浏览器安全预演")
    parser.add_argument("--topk", type=int, default=50, help="持仓数量")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR, help="报告输出目录")
    parser.add_argument("--timestamp", default=None, help="报告时间戳，默认使用当前时间")
    args = parser.parse_args()

    status = run_report(
        phase=args.phase,
        topk=args.topk,
        skip_training=args.skip_training,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        timestamp=args.timestamp,
    )
    print("\n[完成] 只读策略优化{}已生成".format("预演" if args.dry_run else "实验"))
    print(f"[保存] JSON: {status['output_json']}")
    print(f"[保存] Markdown: {status['output_md']}")
    return 0


def legacy_main():
    parser = argparse.ArgumentParser(description="Qlib策略优化实验")
    parser.add_argument("--phase", type=int, default=0, help="运行特定Phase (0=全部)")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练，使用已有预测")
    parser.add_argument("--topk", type=int, default=50, help="持仓数量")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  Qlib 策略优化实验")
    print("=" * 70)

    # ── Step 1: 获取预测 ──
    if args.skip_training:
        print("\n[Step 1] 加载已有预测结果...")
        pred, label, base_metrics = load_existing_predictions()
        if pred is None:
            print("[错误] 未找到已有预测，请先运行训练")
            sys.exit(1)
        print(f"  预测样本数: {len(pred)}")
        print(f"  预测日期范围: {pred.index.get_level_values('datetime').min()} ~ {pred.index.get_level_values('datetime').max()}")
    else:
        print("\n[Step 1] 训练基线模型...")
        config_path = str(CONFIGS_DIR / "baseline_lgb_alpha158.yaml")
        pred, label, base_metrics = train_and_predict(config_path, "baseline_lgb_alpha158")

    # ── Step 2: 获取市场指标 ──
    print("\n[Step 2] 获取市场指标...")
    dates = sorted(pred.index.get_level_values("datetime").unique())
    start_time = str(dates[0].date()) if hasattr(dates[0], 'date') else str(dates[0])[:10]
    end_time = str(dates[-1].date()) if hasattr(dates[-1], 'date') else str(dates[-1])[:10]
    market_indicators = get_market_indicators(start_time, end_time)
    if market_indicators is not None:
        print(f"  市场指标天数: {len(market_indicators)}")

    # ── Step 3: 运行实验 ──
    all_results = {}

    # Baseline (原始Qlib回测结果已在base_metrics中)
    print("\n" + "=" * 70)
    print("  基线结果 (来自MLflow)")
    print("=" * 70)
    for k, v in sorted(base_metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # ── Experiment 1: 低换手策略 ──
    if args.phase == 0 or args.phase == 1:
        print("\n" + "=" * 70)
        print("  Experiment 1: 低换手策略")
        print("=" * 70)

        configs = [
            # (名称, 换手率上限, 调仓频率)
            ("每日调仓+20%换手限制", 1, 0.20),
            ("每3日调仓+20%换手限制", 3, 0.20),
            ("每周调仓+20%换手限制", 5, 0.20),
            ("每2周调仓+10%换手限制", 10, 0.10),
            ("每月调仓+5%换手限制", 20, 0.05),
        ]

        for name, freq, cap in configs:
            strategy = OptimizedStrategy(
                topk=args.topk,
                rebalance_freq=freq,
                turnover_cap=cap,
                cost_bps=15,
            )
            bt = strategy.run(pred, label)
            stats = print_stats(bt, name)
            all_results[f"phase1_{name}"] = stats

    # ── Experiment 2: 市场择时策略 ──
    if args.phase == 0 or args.phase == 2:
        print("\n" + "=" * 70)
        print("  Experiment 2: 市场择时策略")
        print("=" * 70)

        configs = [
            ("MA200趋势过滤(降仓50%)", True, "ma200", 0.5, False, 0.8),
            ("MA120趋势过滤(降仓50%)", True, "ma120", 0.5, False, 0.8),
            ("MA60趋势过滤(降仓30%)", True, "ma60", 0.3, False, 0.8),
            ("MA200+波动率仓位", True, "ma200", 0.5, True, 0.8),
            ("MA200+激进波动率", True, "ma200", 0.5, True, 0.6),
        ]

        for name, trend, ma, reduce, vol, vol_th in configs:
            strategy = OptimizedStrategy(
                topk=args.topk,
                rebalance_freq=5,
                turnover_cap=0.20,
                cost_bps=15,
                trend_filter=trend,
                trend_ma=ma,
                trend_reduce_pct=reduce,
                vol_sizing=vol,
                vol_reduce_threshold=vol_th,
            )
            bt = strategy.run(pred, label, market_indicators)
            stats = print_stats(bt, name)
            all_results[f"phase2_{name}"] = stats

    # ── Experiment 3: 风险控制策略 ──
    if args.phase == 0 or args.phase == 3:
        print("\n" + "=" * 70)
        print("  Experiment 3: 风险控制策略")
        print("=" * 70)

        configs = [
            # (名称, 最大回撤止损, 冷却天数)
            ("-8%回撤止损+20天冷却", -0.08, 20),
            ("-5%回撤止损+10天冷却", -0.05, 10),
            ("-10%回撤止损+30天冷却", -0.10, 30),
        ]

        for name, mdd, cooldown in configs:
            strategy = OptimizedStrategy(
                topk=args.topk,
                rebalance_freq=5,
                turnover_cap=0.20,
                cost_bps=15,
                max_drawdown_stop=mdd,
                max_dd_cooldown_days=cooldown,
            )
            bt = strategy.run(pred, label, market_indicators)
            stats = print_stats(bt, name)
            all_results[f"phase3_{name}"] = stats

    # ── Experiment 4: 综合最优策略 ──
    if args.phase == 0 or args.phase == 4:
        print("\n" + "=" * 70)
        print("  Experiment 4: 综合最优策略 (低换手 + 择时 + 风控)")
        print("=" * 70)

        strategy = OptimizedStrategy(
            topk=args.topk,
            rebalance_freq=5,       # 每周调仓
            turnover_cap=0.15,      # 15%换手上限
            cost_bps=15,
            trend_filter=True,      # MA200趋势过滤
            trend_ma="ma200",
            trend_reduce_pct=0.5,   # 熊市降仓50%
            vol_sizing=True,        # 波动率仓位管理
            vol_reduce_threshold=0.8,
            max_drawdown_stop=-0.08,# 8%回撤止损
            max_dd_cooldown_days=20,
        )
        bt = strategy.run(pred, label, market_indicators)
        stats = print_stats(bt, "综合最优策略")
        all_results["phase4_combined"] = stats

        # 保守版
        strategy_conservative = OptimizedStrategy(
            topk=30,                # 更集中
            rebalance_freq=10,      # 每2周调仓
            turnover_cap=0.10,      # 10%换手上限
            cost_bps=15,
            trend_filter=True,
            trend_ma="ma120",
            trend_reduce_pct=0.6,   # 熊市降仓60%
            vol_sizing=True,
            vol_reduce_threshold=0.7,
            max_drawdown_stop=-0.06,
            max_dd_cooldown_days=30,
        )
        bt_c = strategy_conservative.run(pred, label, market_indicators)
        stats_c = print_stats(bt_c, "保守综合策略 (topk=30, 双周调仓)")
        all_results["phase4_conservative"] = stats_c

    # ── 汇总对比 ──
    print("\n" + "=" * 70)
    print("  实验结果汇总")
    print("=" * 70)

    summary_rows = []
    for name, stats in all_results.items():
        if stats:
            summary_rows.append({
                "实验": name,
                "年化收益": f"{stats.get('annual_return', 0):.2%}",
                "Sharpe": f"{stats.get('sharpe', 0):.2f}",
                "最大回撤": f"{stats.get('max_drawdown', 0):.2%}",
                "平均换手": f"{stats.get('avg_turnover', 0):.2%}",
                "成本占比": f"{stats.get('cost_ratio', 0):.2%}",
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        print(df_summary.to_string(index=False))

    # ── 保存结果 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = REPORT_DIR / f"optimization_results_{timestamp}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[保存] 结果已保存到: {result_path}")


if __name__ == "__main__":
    raise SystemExit(main())
