/**
 * Backtest API type definitions
 * Mirrors api/v1/schemas/backtest.py
 */
import type { DecisionAction, MarketPhaseSummary } from './analysis';

// ============ Request / Response ============

export type BacktestAnalysisPhase = 'premarket' | 'intraday' | 'postmarket' | 'unknown';
export type BacktestPhaseFilter = BacktestAnalysisPhase | 'all';

export interface BacktestRunRequest {
  code?: string;
  force?: boolean;
  evalWindowDays?: number;
  minAgeDays?: number;
  analysisDateFrom?: string;
  analysisDateTo?: string;
  limit?: number;
}

export interface BacktestRunResponse {
  processed: number;
  saved: number;
  completed: number;
  insufficient: number;
  errors: number;
  appliedEvalWindowDays?: number;
  message?: string | null;
  diagnostics?: Record<string, unknown>;
}

export interface StrategyBacktestRunRequest {
  code: string;
  startDate?: string;
  endDate?: string;
  initialCapital: number;
  sideCostPct: number;
  lotSize: number;
}

export type FormStrategyId = 'momentum_rotation' | 'dual_moving_average' | 'buy_and_hold';

export interface FormStrategyBacktestRunRequest {
  strategyId: FormStrategyId;
  symbols: string[];
  startDate?: string;
  endDate?: string;
  initialCapital: number;
  sideCostPct: number;
  lotSize: number;
  lookbackDays: number;
  movingAverageDays: number;
  fastMaDays: number;
  slowMaDays: number;
  topN: number;
}

export interface ScriptStrategyBacktestRunRequest {
  script: string;
  pool?: 'etf' | 'csi800' | 'all_a' | 'custom';
  symbols?: string[];
  benchmarks?: string[];
  comparisonStrategyIds?: string[];
  comparisonPresets?: Array<{ name: string; script: string }>;
  startDate?: string;
  endDate?: string;
  initialCapital: number;
  sideCostPct: number;
  lotSize: number;
  strategyAssetId?: string;
}

export interface TrainedStrategyBacktestRunRequest {
  runId: string;
  modelId: string;
  strategyId: string;
  replayMode: 'exact' | 'custom';
  startDate?: string;
  endDate?: string;
  topk?: number;
  costBps?: number;
}

export interface TrainedStrategyCatalogStrategy {
  id: string;
  label: string;
  config: Record<string, unknown>;
  exactReplayAvailable: boolean;
  replayEngine: 'qlib_portana' | 'dsa_optimized_strategy';
}

export interface TrainedStrategyCatalogRun {
  runId: string;
  experimentId?: string | null;
  modelId: string;
  modelLabel: string;
  sourceType: 'baseline' | 'candidate';
  generatedAt: string;
  provider: string;
  providerIdentity: Record<string, string>;
  periods: Record<string, string[]>;
  originalParameters: {
    strategyId?: string;
    startDate?: string;
    endDate?: string;
    topk?: number;
    costBps?: number;
  };
  strategies: TrainedStrategyCatalogStrategy[];
  configHash?: string | null;
  contractVersion?: number | null;
  exactReplayAvailable: boolean;
  customReplayAvailable: boolean;
  replayCapability: 'exact_and_custom' | 'custom_only' | 'unavailable';
  replayUnavailableReason?: string | null;
}

export interface TrainedStrategyCatalogResponse {
  status: 'completed' | 'missing';
  generatedAt: string;
  runs: TrainedStrategyCatalogRun[];
}

export interface TrainedStrategyBacktestMetrics {
  days: number;
  totalReturn: number;
  benchmarkTotalReturn: number;
  annualReturn: number;
  benchmarkAnnualReturn: number;
  annualExcessReturn: number;
  informationRatio?: number | null;
  sharpe?: number | null;
  maxDrawdown: number;
  avgTurnover: number;
  avgPosition: number;
}

export interface TrainedStrategyBacktestPoint {
  date: string;
  strategyCumulativeReturn: number;
  benchmarkCumulativeReturn: number;
  drawdown: number;
  benchmarkDrawdown: number;
  dailyReturn: number;
  benchmarkReturn: number;
  turnover: number;
  positionPct: number;
}

export interface TrainedStrategyBacktestResponse {
  status: string;
  generatedAt: string;
  runId: string;
  modelId: string;
  modelLabel: string;
  strategyId: string;
  strategyLabel: string;
  provider: string;
  providerIdentity: Record<string, string>;
  artifactSha256: Record<string, string>;
  strategyConfig: Record<string, unknown>;
  replayMode: 'exact' | 'custom';
  replayEngine: 'qlib_portana' | 'dsa_optimized_strategy';
  sourceReplayEngine: 'qlib_portana' | 'dsa_optimized_strategy';
  modifiedFromOriginal: boolean;
  configHash?: string | null;
  contractVersion?: number | null;
  originalParameters: Record<string, unknown>;
  executionAssumptions: Record<string, unknown>;
  exactMetricValidation?: {
    status: 'pass';
    tolerance: number;
    maxAbsDifference: number;
    saved: Record<string, number>;
    replayed: Record<string, number>;
  } | null;
  period: { startDate: string; endDate: string };
  parameters: {
    startDate: string;
    endDate: string;
    topk: number;
    costBps: number;
  };
  metrics: TrainedStrategyBacktestMetrics;
  equityCurve: TrainedStrategyBacktestPoint[];
  warnings: string[];
}

export interface StrategyBacktestMetrics {
  initialCapital: number;
  openingEquity: number;
  finalEquity: number;
  totalReturnPct: number;
  annualizedReturnPct: number;
  benchmarkReturnPct: number;
  benchmarkAnnualizedReturnPct: number;
  excessReturnPct: number;
  maxDrawdownPct: number;
  benchmarkMaxDrawdownPct: number;
  tradeCount: number;
  exposurePct: number;
  totalCost: number;
  turnoverPct: number;
  signalCount: number;
  executedSignalCount: number;
  skippedSignalCount: number;
  blockedSignalCount: number;
  tradingDays: number;
  annualizedVolatilityPct: number;
  benchmarkAnnualizedVolatilityPct: number;
  sharpeRatio?: number | null;
  benchmarkSharpeRatio?: number | null;
  alphaPct?: number | null;
  beta?: number | null;
  winRatePct: number;
}

export interface StrategyBacktestEquityPoint {
  date: string;
  strategyEquity: number;
  benchmarkEquity: number;
  strategyReturnPct: number;
  benchmarkReturnPct: number;
  drawdownPct: number;
  benchmarkDrawdownPct: number;
  position: string;
}

export interface StrategyBacktestTrade {
  sourceId: string;
  signalDate: string;
  executionDate: string;
  symbol: string;
  action: 'buy' | 'sell' | string;
  shares: number;
  price: number;
  notional: number;
  fee: number;
  reason?: string;
}

export interface ComparisonSeries {
  symbol: string;
  name?: string;
  equity: Array<number | null>;
  totalReturnPct: number;
  maxDrawdownPct: number;
  firstValidDate: string;
}

export interface ComparisonFailureItem {
  name: string;
  error: string;
}

export interface StrategyBacktestResponse {
  status: string;
  strategyName: string;
  runId?: string;
  symbols: string[];
  comparisons?: Record<string, ComparisonSeries>;
  comparisonFailures?: ComparisonFailureItem[];
  period?: { startDate: string; endDate: string };
  strategy: Record<string, unknown>;
  dataQuality: Record<string, unknown>;
  metrics: StrategyBacktestMetrics;
  benchmark: Record<string, unknown>;
  assumptions: {
    execution?: string;
    initialCapital?: number;
    sideCostPct?: number;
    lotSize?: number;
    [key: string]: unknown;
  };
  equityCurve: StrategyBacktestEquityPoint[];
  trades: StrategyBacktestTrade[];
  warnings: string[];
  logs: string[];
  scriptCheck: Record<string, unknown>;
}

// ============ Result Item ============

export interface BacktestResultItem {
  analysisHistoryId: number;
  code: string;
  stockName?: string;
  analysisDate?: string;
  evalWindowDays: number;
  engineVersion: string;
  evalStatus: string;
  evaluatedAt?: string;
  operationAdvice?: string;
  action?: DecisionAction | null;
  actionLabel?: string | null;
  trendPrediction?: string;
  marketPhase?: string | null;
  marketPhaseSummary?: MarketPhaseSummary | null;
  positionRecommendation?: string;
  startPrice?: number;
  endClose?: number;
  maxHigh?: number;
  minLow?: number;
  stockReturnPct?: number;
  actualReturnPct?: number;
  actualMovement?: string;
  directionExpected?: string;
  directionCorrect?: boolean;
  outcome?: string;
  stopLoss?: number;
  takeProfit?: number;
  hitStopLoss?: boolean;
  hitTakeProfit?: boolean;
  firstHit?: string;
  firstHitDate?: string;
  firstHitTradingDays?: number;
  simulatedEntryPrice?: number;
  simulatedExitPrice?: number;
  simulatedExitReason?: string;
  simulatedReturnPct?: number;
}

export interface BacktestResultsResponse {
  total: number;
  page: number;
  limit: number;
  items: BacktestResultItem[];
}

// ============ Performance Metrics ============

export interface PerformanceMetrics {
  scope: string;
  code?: string;
  evalWindowDays: number;
  engineVersion: string;
  computedAt?: string;

  totalEvaluations: number;
  completedCount: number;
  insufficientCount: number;
  longCount: number;
  cashCount: number;
  winCount: number;
  lossCount: number;
  neutralCount: number;

  directionAccuracyPct?: number;
  winRatePct?: number;
  neutralRatePct?: number;
  avgStockReturnPct?: number;
  avgSimulatedReturnPct?: number;

  stopLossTriggerRate?: number;
  takeProfitTriggerRate?: number;
  ambiguousRate?: number;
  avgDaysToFirstHit?: number;

  adviceBreakdown: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
}

// === Strategy assets and backtest run history ===

export interface StrategyAssetSummary {
  id: string;
  name: string;
  description: string;
  scriptSha256: string;
  apiVersion: string;
  archived: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface StrategyAssetDetail extends StrategyAssetSummary {
  script: string;
}

export interface StrategyAssetCreateRequest {
  name: string;
  description?: string;
  script: string;
}

export interface StrategyAssetUpdateRequest {
  name?: string;
  description?: string;
  script?: string;
  archived?: boolean;
}

export interface StabilityTestRequest {
  script: string;
  pool?: 'etf' | 'csi800' | 'all_a' | 'custom';
  symbols?: string[];
  startDate?: string;
  endDate?: string;
  initialCapital: number;
  sideCostPct: number;
  lotSize: number;
  minWindowDays?: number;
  iterations?: number;
  seed?: number;
}

export interface StabilitySample {
  startDate: string;
  endDate: string;
  tradingDays: number;
  totalReturnPct: number;
  maxDrawdownPct: number;
  benchmarkReturnPct: number;
  tradeCount: number;
  exposurePct: number;
}

export interface StabilityTestResponse {
  status: string;
  summary: Record<string, unknown>;
  samples: StabilitySample[];
  dataQuality: Record<string, unknown>;
  scriptCheck: Record<string, unknown>;
  logs: string[];
}

export interface BacktestRunSummary {
  id: string;
  strategyId?: string;
  strategyName: string;
  scriptSha256: string;
  symbols: string[];
  params: Record<string, unknown>;
  dataQuality: Record<string, unknown>;
  metrics: Record<string, unknown>;
  status: string;
  errorMessage?: string;
  createdAt: string;
}

export interface BacktestRunDetail extends BacktestRunSummary {
  report: StrategyBacktestResponse;
}
