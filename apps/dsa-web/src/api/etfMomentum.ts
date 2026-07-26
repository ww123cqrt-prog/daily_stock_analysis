import apiClient from './index';
import { toCamelCase } from './utils';

export type EtfMomentumMetrics = {
  periods?: number;
  startDate?: string;
  endDate?: string;
  totalReturn?: number | null;
  annualizedReturn?: number | null;
  maxDrawdown?: number | null;
  investedFraction?: number | null;
  switchCount?: number;
  totalTurnoverSides?: number;
  totalTransactionCostFraction?: number | null;
  sideCostBps?: number;
};

export type EtfMomentumReportState = {
  status: 'missing' | 'invalid' | string;
  reasonCode?: string;
  message?: string;
};

export type EtfMomentumStrategyReport = {
  schemaVersion: number;
  status: string;
  generatedAt: string;
  strategyId: string;
  summary: {
    status: string;
    verdictCode: string;
    dataStatus: string;
    usableForTrading: boolean;
    baselineTotalReturn: number | null;
    baselineAnnualizedReturn: number | null;
    baselineMaxDrawdown: number | null;
    stressAnnualizedReturn: number | null;
    delayedAnnualizedReturn: number | null;
    unmetReferenceCount: number;
    evidenceGapCount: number;
  };
  strategy: {
    symbols: string[];
    lookbackDays: number;
    movingAverageDays: number;
    signalTime: string;
    executionTime: string;
    buyRule: string;
    sellRule: string;
    cashRule: string;
    automaticOrders: boolean;
    brokerConnection: boolean;
  };
  latestSignal: {
    asOfDate: string;
    signal: string;
    positionAtLatestOpen: string;
    leader: string;
    nextOpenResearchTarget: string;
    reliability: string;
    usableForTrading: boolean;
    details: Record<string, {
      close: number | null;
      lookbackReturn: number | null;
      movingAverage: number | null;
      aboveMovingAverage: boolean;
      rank: number | null;
    }>;
  };
  performance: {
    dataRange: { startDate: string; endDate: string; tradingDays: number };
    variants: Record<string, { totalCostBps: number; metrics: EtfMomentumMetrics }>;
    delayedExecution: { additionalExecutionLagDays: number; metrics: EtfMomentumMetrics };
    benchmarks: Record<string, EtfMomentumMetrics>;
    quarterly: Array<EtfMomentumMetrics & { period: string; positive: boolean }>;
    yearly: Array<EtfMomentumMetrics & { period: string; positive: boolean }>;
    startDateSensitivity: Array<{ startDate: string; metrics: EtfMomentumMetrics }>;
    claimComparison?: Record<string, unknown>;
  };
  stability: {
    decisionMode: string;
    automaticRejectionDisabled: boolean;
    referenceChecks: Array<{
      id: string;
      value: number | null;
      threshold: number | null;
      operator: string;
      met: boolean;
      decision?: string;
    }>;
    advisoryNotMet: string[];
    quarterConsistency: {
      strictCompleteCount: number;
      strictCompletePositiveCount: number;
      strictCompletePositiveRatio: number | null;
      allAvailableCount: number;
      allAvailablePositiveCount: number;
      allAvailablePositiveRatio: number | null;
      earlierComparisonTotalBps: number;
      earlierComparisonCount: number;
      earlierComparisonPositiveCount: number;
      earlierComparisonPositiveRatio: number | null;
    };
    parameterGrid: {
      primary: { lookbackDays: number; movingAverageDays: number };
      selectionAllowed: boolean;
      variants: Array<{
        lookbackDays: number;
        movingAverageDays: number;
        metrics: EtfMomentumMetrics;
      }>;
    };
    costStress: Array<{
      name: string;
      totalCostBps: number;
      annualizedReturn: number | null;
      maxDrawdown: number | null;
    }>;
  };
  risk: {
    cashMetrics: {
      signalCash?: { cashDays?: number; cashFraction?: number | null; longestCashStreakDays?: number };
      nextOpenTargetCash?: { cashDays?: number; cashFraction?: number | null; longestCashStreakDays?: number };
      definition?: string;
    };
    holdingSummary: {
      episodeCount: number;
      investedEpisodeCount: number;
      cashEpisodeCount: number;
      averageTradingDays: number | null;
      medianTradingDays: number | null;
      investedWinRate: number | null;
    };
    drawdownEpisodes: Array<{
      startDate: string;
      troughDate: string;
      recoveryDate: string | null;
      maxDrawdown: number;
      tradingDays: number;
      recovered: boolean;
    }>;
    longestUnderwaterTradingDays: number;
    knownFailureConditions: string[];
  };
  chartData: {
    wealth: Array<{ date: string; strategy: number; SZ159915: number; SH513100: number; equalWeight: number }>;
    drawdown: Array<{ date: string; drawdown: number }>;
    quarterlyReturns: Array<{ period: string; return: number; positive: boolean }>;
    yearlyReturns: Array<{ period: string; return: number; positive: boolean }>;
    monthlyAllocations: Array<{ month: string; SZ159915: number; SH513100: number; CASH: number }>;
    costStress: Array<{ label: string; annualizedReturn: number; maxDrawdown: number }>;
    startDateSensitivity: Array<{ startDate: string; annualizedReturn: number; maxDrawdown: number }>;
    anomalyDates: string[];
  };
  dataQuality: {
    status: string;
    selectedSource: string;
    selectedAdjustment: string;
    rawSource?: string;
    adjustedSource?: string;
    adjustedSourceAvailable: boolean;
    coverageRatio: number;
    minimumCoverage?: number;
    returnAgreementRatio?: number | null;
    paperTrackingDaysCompleted: number;
    evidenceGaps: string[];
    rawPriceDiscontinuities?: {
      passed: boolean;
      events: Array<{ symbol: string; date: string; closeReturn: number }>;
    };
  };
  reproducibility: {
    configHash?: string | null;
    dataManifestHash?: string | null;
    gitCommit?: string | null;
    gitDirty?: boolean | null;
    generatedAt: string;
  };
};

export type EtfMomentumStrategyResponse = EtfMomentumStrategyReport | EtfMomentumReportState;

export function isEtfMomentumStrategyReport(
  value: EtfMomentumStrategyResponse | null,
): value is EtfMomentumStrategyReport {
  return Boolean(value && 'summary' in value && 'chartData' in value);
}

export const etfMomentumApi = {
  async getReport(): Promise<EtfMomentumStrategyResponse> {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/research/strategies/etf-momentum',
    );
    return toCamelCase<EtfMomentumStrategyResponse>(response.data);
  },
};
