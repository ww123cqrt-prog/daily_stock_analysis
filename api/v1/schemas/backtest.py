# -*- coding: utf-8 -*-
"""Backtest API schemas."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from api.v1.schemas.market_phase import MarketPhaseSummary
from src.schemas.decision_action import DecisionAction


class BacktestRunRequest(BaseModel):
    code: Optional[str] = Field(None, description="仅回测指定股票")
    force: bool = Field(False, description="强制重新计算")
    eval_window_days: Optional[int] = Field(None, ge=1, le=120, description="评估窗口（交易日数）")
    min_age_days: Optional[int] = Field(None, ge=0, le=365, description="分析记录最小天龄（0=不限）")
    analysis_date_from: Optional[date] = Field(None, description="分析日期起始（含）")
    analysis_date_to: Optional[date] = Field(None, description="分析日期结束（含）")
    limit: int = Field(200, ge=1, le=2000, description="最多处理的分析记录数")


class BacktestRunResponse(BaseModel):
    processed: int = Field(..., description="候选记录数")
    saved: int = Field(..., description="写入回测结果数")
    completed: int = Field(..., description="完成回测数")
    insufficient: int = Field(..., description="数据不足数")
    errors: int = Field(..., description="错误数")
    applied_eval_window_days: Optional[int] = Field(
        ...,
        description="实际生效的评估窗口（交易日数）",
    )
    message: Optional[str] = Field(None, description="空结果或降级时的诊断说明")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="回测筛选与诊断信息")


class StrategyBacktestRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=20, description="标的代码（股票或ETF）")
    start_date: Optional[date] = Field(None, description="信号开始日期（含）")
    end_date: Optional[date] = Field(None, description="信号结束日期（含）")
    initial_capital: float = Field(1_000_000.0, gt=0, le=1_000_000_000_000.0)
    side_cost_pct: float = Field(
        0.0,
        ge=0,
        le=5,
        description="每次买入或卖出的成本百分比；0表示未计成本",
    )
    lot_size: int = Field(100, ge=1, le=1_000_000, description="最小交易数量")


class FormStrategyBacktestRunRequest(BaseModel):
    strategy_id: str = Field(..., min_length=1, max_length=64, description="内置策略标识")
    symbols: List[str] = Field(..., min_length=1, max_length=50, description="股票或ETF代码列表")
    start_date: Optional[date] = Field(None, description="信号开始日期（含）")
    end_date: Optional[date] = Field(None, description="信号结束日期（含）")
    initial_capital: float = Field(1_000_000.0, gt=0, le=1_000_000_000_000.0)
    side_cost_pct: float = Field(0.0, ge=0, le=5)
    lot_size: int = Field(100, ge=1, le=1_000_000)
    lookback_days: int = Field(20, ge=1, le=500)
    moving_average_days: int = Field(20, ge=1, le=500)
    fast_ma_days: int = Field(5, ge=1, le=500)
    slow_ma_days: int = Field(20, ge=2, le=1000)
    top_n: int = Field(1, ge=1, le=50)


class ScriptStrategyBacktestRunRequest(BaseModel):
    script: str = Field(..., min_length=1, max_length=50_000, description="DSA Python交易策略脚本")
    symbols: List[str] = Field(..., min_length=1, max_length=50, description="脚本可访问的标的列表")
    start_date: Optional[date] = Field(None, description="信号开始日期（含）")
    end_date: Optional[date] = Field(None, description="信号结束日期（含）")
    initial_capital: float = Field(1_000_000.0, gt=0, le=1_000_000_000_000.0)
    side_cost_pct: float = Field(0.0, ge=0, le=5)
    lot_size: int = Field(100, ge=1, le=1_000_000)


class TrainedStrategyBacktestRunRequest(BaseModel):
    run_id: str = Field(..., pattern=r"^[0-9a-f]{32}$", description="本地 MLflow Run ID")
    model_id: str = Field(..., min_length=1, max_length=64)
    strategy_id: str = Field(..., min_length=1, max_length=64)
    replay_mode: Literal["exact", "custom"] = "custom"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    topk: Optional[int] = Field(None, ge=1, le=500, description="每次持有的模型评分前K只股票")
    cost_bps: Optional[float] = Field(None, ge=0, le=1000, description="按换手扣除的成本；1基点=0.01%")


class TrainedStrategyCatalogStrategy(BaseModel):
    id: str
    label: str
    config: Dict[str, Any]
    exact_replay_available: bool = False
    replay_engine: Literal["qlib_portana", "dsa_optimized_strategy"] = (
        "dsa_optimized_strategy"
    )


class TrainedStrategyCatalogRun(BaseModel):
    run_id: str
    experiment_id: Optional[str] = None
    model_id: str
    model_label: str
    source_type: Literal["baseline", "candidate"]
    generated_at: str
    provider: str
    provider_identity: Dict[str, str]
    periods: Dict[str, List[str]]
    original_parameters: Dict[str, Any]
    strategies: List[TrainedStrategyCatalogStrategy] = Field(default_factory=list)
    config_hash: Optional[str] = None
    contract_version: Optional[int] = None
    exact_replay_available: bool
    custom_replay_available: bool
    replay_capability: Literal["exact_and_custom", "custom_only", "unavailable"]
    replay_unavailable_reason: Optional[str] = None


class TrainedStrategyCatalogResponse(BaseModel):
    status: Literal["completed", "missing"]
    generated_at: str
    runs: List[TrainedStrategyCatalogRun] = Field(default_factory=list)


class TrainedStrategyBacktestMetrics(BaseModel):
    days: int
    total_return: float
    benchmark_total_return: float
    annual_return: float
    benchmark_annual_return: float
    annual_excess_return: float
    information_ratio: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: float
    avg_turnover: float
    avg_position: float


class TrainedStrategyBacktestPoint(BaseModel):
    date: str
    strategy_cumulative_return: float
    benchmark_cumulative_return: float
    drawdown: float
    benchmark_drawdown: float
    daily_return: float
    benchmark_return: float
    turnover: float
    position_pct: float


class TrainedStrategyBacktestResponse(BaseModel):
    status: str
    generated_at: str
    run_id: str
    model_id: str
    model_label: str
    strategy_id: str
    strategy_label: str
    provider: str
    provider_identity: Dict[str, Any]
    artifact_sha256: Dict[str, str]
    strategy_config: Dict[str, Any]
    replay_mode: Literal["exact", "custom"]
    replay_engine: Literal["qlib_portana", "dsa_optimized_strategy"]
    source_replay_engine: Literal["qlib_portana", "dsa_optimized_strategy"]
    modified_from_original: bool
    config_hash: Optional[str] = None
    contract_version: Optional[int] = None
    original_parameters: Dict[str, Any]
    execution_assumptions: Dict[str, Any] = Field(default_factory=dict)
    exact_metric_validation: Optional[Dict[str, Any]] = None
    period: Dict[str, str]
    parameters: Dict[str, Any]
    metrics: TrainedStrategyBacktestMetrics
    equity_curve: List[TrainedStrategyBacktestPoint] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class StrategyBacktestMetrics(BaseModel):
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    max_drawdown_pct: float
    benchmark_max_drawdown_pct: float
    trade_count: int
    exposure_pct: float
    total_cost: float
    turnover_pct: float
    signal_count: int
    executed_signal_count: int
    skipped_signal_count: int
    trading_days: int
    annualized_volatility_pct: float = 0.0
    sharpe_ratio: Optional[float] = None
    alpha_pct: Optional[float] = None
    beta: Optional[float] = None
    win_rate_pct: float = 0.0


class StrategyBacktestEquityPoint(BaseModel):
    date: str
    strategy_equity: float
    benchmark_equity: float
    strategy_return_pct: float
    benchmark_return_pct: float
    drawdown_pct: float
    benchmark_drawdown_pct: float
    position: str


class StrategyBacktestTrade(BaseModel):
    source_id: str
    signal_date: str
    execution_date: str
    symbol: str
    action: str
    shares: int
    price: float
    notional: float
    fee: float
    reason: Optional[str] = None


class StrategyBacktestResponse(BaseModel):
    status: str
    strategy_name: str
    symbols: List[str] = Field(default_factory=list)
    period: Optional[Dict[str, str]] = None
    strategy: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    metrics: StrategyBacktestMetrics
    benchmark: Dict[str, Any] = Field(default_factory=dict)
    assumptions: Dict[str, Any] = Field(default_factory=dict)
    equity_curve: List[StrategyBacktestEquityPoint] = Field(default_factory=list)
    trades: List[StrategyBacktestTrade] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    script_check: Dict[str, Any] = Field(default_factory=dict)


class BacktestResultItem(BaseModel):
    analysis_history_id: int
    code: str
    stock_name: Optional[str] = None
    analysis_date: Optional[str] = None
    eval_window_days: int
    engine_version: str
    eval_status: str
    evaluated_at: Optional[str] = None
    operation_advice: Optional[str] = None
    action: Optional[DecisionAction] = None
    action_label: Optional[str] = None
    trend_prediction: Optional[str] = None
    market_phase: Optional[str] = None
    market_phase_summary: Optional[MarketPhaseSummary] = None
    position_recommendation: Optional[str] = None
    start_price: Optional[float] = None
    end_close: Optional[float] = None
    max_high: Optional[float] = None
    min_low: Optional[float] = None
    stock_return_pct: Optional[float] = None
    actual_return_pct: Optional[float] = None
    actual_movement: Optional[str] = None
    direction_expected: Optional[str] = None
    direction_correct: Optional[bool] = None
    outcome: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    hit_stop_loss: Optional[bool] = None
    hit_take_profit: Optional[bool] = None
    first_hit: Optional[str] = None
    first_hit_date: Optional[str] = None
    first_hit_trading_days: Optional[int] = None
    simulated_entry_price: Optional[float] = None
    simulated_exit_price: Optional[float] = None
    simulated_exit_reason: Optional[str] = None
    simulated_return_pct: Optional[float] = None


class BacktestResultsResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[BacktestResultItem] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    scope: str
    code: Optional[str] = None
    eval_window_days: int
    engine_version: str
    computed_at: Optional[str] = None

    total_evaluations: int
    completed_count: int
    insufficient_count: int
    long_count: int
    cash_count: int
    win_count: int
    loss_count: int
    neutral_count: int

    direction_accuracy_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    neutral_rate_pct: Optional[float] = None
    avg_stock_return_pct: Optional[float] = None
    avg_simulated_return_pct: Optional[float] = None

    stop_loss_trigger_rate: Optional[float] = None
    take_profit_trigger_rate: Optional[float] = None
    ambiguous_rate: Optional[float] = None
    avg_days_to_first_hit: Optional[float] = None

    advice_breakdown: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
