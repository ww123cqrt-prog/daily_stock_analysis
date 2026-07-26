import type React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import EtfMomentumReportPage from '../EtfMomentumReportPage';

const { getReport, submitJob, getJob } = vi.hoisted(() => ({
  getReport: vi.fn(),
  submitJob: vi.fn(),
  getJob: vi.fn(),
}));

vi.mock('../../api/etfMomentum', () => ({
  etfMomentumApi: { getReport },
  isEtfMomentumStrategyReport: (value: Record<string, unknown> | null) => Boolean(value?.summary && value?.chartData),
}));

vi.mock('../../api/quant', () => ({
  quantApi: { submitJob, getJob },
}));

vi.mock('recharts', () => {
  const Wrapper = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return {
    Area: Wrapper,
    AreaChart: Wrapper,
    Bar: Wrapper,
    BarChart: Wrapper,
    CartesianGrid: Wrapper,
    Cell: Wrapper,
    ComposedChart: Wrapper,
    Legend: Wrapper,
    Line: Wrapper,
    LineChart: Wrapper,
    ReferenceLine: Wrapper,
    ResponsiveContainer: Wrapper,
    Tooltip: Wrapper,
    XAxis: Wrapper,
    YAxis: Wrapper,
  };
});

const metrics = {
  periods: 1810,
  totalReturn: 3.1192,
  annualizedReturn: 0.2179,
  maxDrawdown: -0.3132,
  investedFraction: 0.7448,
  switchCount: 273,
};

const report = {
  schemaVersion: 1,
  status: 'manual_review_required',
  generatedAt: '2026-07-25T21:48:09+08:00',
  strategyId: 'etf_20d_two_asset_momentum_rotation',
  summary: {
    status: 'manual_review_required',
    verdictCode: 'continue_research_data_unverified',
    dataStatus: 'unverified_data',
    usableForTrading: false,
    baselineTotalReturn: 3.1192,
    baselineAnnualizedReturn: 0.2179,
    baselineMaxDrawdown: -0.3132,
    stressAnnualizedReturn: 0.1595,
    delayedAnnualizedReturn: 0.1816,
    unmetReferenceCount: 1,
    evidenceGapCount: 4,
  },
  strategy: {
    symbols: ['SZ159915', 'SH513100'],
    lookbackDays: 20,
    movingAverageDays: 20,
    signalTime: 'close',
    executionTime: 'next_open',
    buyRule: 'buy',
    sellRule: 'sell',
    cashRule: 'cash',
    automaticOrders: false,
    brokerConnection: false,
  },
  latestSignal: {
    asOfDate: '2026-07-24',
    signal: 'CASH',
    positionAtLatestOpen: 'CASH',
    leader: 'SH513100',
    nextOpenResearchTarget: 'CASH',
    reliability: 'unverified_data',
    usableForTrading: false,
    details: {
      SZ159915: { close: 3.508, lookbackReturn: -0.1677, movingAverage: 3.8564, aboveMovingAverage: false, rank: 2 },
      SH513100: { close: 2.104, lookbackReturn: -0.025, movingAverage: 2.1564, aboveMovingAverage: false, rank: 1 },
    },
  },
  performance: {
    dataRange: { startDate: '2019-01-30', endDate: '2026-07-24', tradingDays: 1811 },
    variants: {
      zero: { totalCostBps: 0, metrics: { ...metrics, annualizedReturn: 0.2791 } },
      low: { totalCostBps: 10, metrics: { ...metrics, annualizedReturn: 0.2481 } },
      baseline: { totalCostBps: 20, metrics },
      stress: { totalCostBps: 40, metrics: { ...metrics, annualizedReturn: 0.1595, maxDrawdown: -0.3328 } },
    },
    delayedExecution: { additionalExecutionLagDays: 1, metrics: { ...metrics, annualizedReturn: 0.1816 } },
    benchmarks: {
      SZ159915: { totalReturn: 1.989, annualizedReturn: 0.1647, maxDrawdown: -0.5693 },
      SH513100: { totalReturn: -0.1022, annualizedReturn: -0.0149, maxDrawdown: -0.8573 },
      equalWeight: { totalReturn: 0.9434, annualizedReturn: 0.0969, maxDrawdown: -0.649 },
    },
    quarterly: [{ period: '2026Q1', totalReturn: 0.08, positive: true }],
    yearly: [{ period: '2025', totalReturn: -0.05, positive: false }],
    startDateSensitivity: [{ startDate: '2023-01-03', metrics }],
  },
  stability: {
    decisionMode: 'manual_review',
    automaticRejectionDisabled: true,
    referenceChecks: [
      ['positive_quarter_ratio', 0.5862, 0.7, false],
      ['baseline_annualized_return', 0.2179, 0.05, true],
      ['baseline_max_drawdown', 0.3132, 0.35, true],
      ['stress_annualized_return', 0.1595, 0, true],
      ['delayed_annualized_return', 0.1816, 0, true],
      ['parameter_positive_ratio', 1, 0.7, true],
      ['parameter_max_drawdown_pass_ratio', 1, 0.7, true],
    ].map(([id, value, threshold, met]) => ({ id, value, threshold, operator: '>=', met })),
    advisoryNotMet: ['positive_quarter_ratio'],
    quarterConsistency: {
      strictCompleteCount: 29,
      strictCompletePositiveCount: 17,
      strictCompletePositiveRatio: 0.5862069,
      allAvailableCount: 30,
      allAvailablePositiveCount: 18,
      allAvailablePositiveRatio: 0.6,
      earlierComparisonTotalBps: 15,
      earlierComparisonCount: 30,
      earlierComparisonPositiveCount: 19,
      earlierComparisonPositiveRatio: 0.6333333,
    },
    parameterGrid: {
      primary: { lookbackDays: 20, movingAverageDays: 20 },
      selectionAllowed: false,
      variants: [15, 20, 25].flatMap((lookbackDays) => [15, 20, 25].map((movingAverageDays) => ({
        lookbackDays,
        movingAverageDays,
        metrics: { ...metrics, annualizedReturn: 0.14 + movingAverageDays / 1000 },
      }))),
    },
    costStress: [
      { name: 'low', totalCostBps: 10, annualizedReturn: 0.2481, maxDrawdown: -0.3031 },
      { name: 'baseline', totalCostBps: 20, annualizedReturn: 0.2179, maxDrawdown: -0.3132 },
      { name: 'stress', totalCostBps: 40, annualizedReturn: 0.1595, maxDrawdown: -0.3328 },
    ],
  },
  risk: {
    cashMetrics: { signalCash: { cashDays: 463, cashFraction: 0.2557, longestCashStreakDays: 45 } },
    holdingSummary: { episodeCount: 274, investedEpisodeCount: 176, cashEpisodeCount: 98, averageTradingDays: 6.61, medianTradingDays: 3, investedWinRate: 0.4489 },
    drawdownEpisodes: [{ startDate: '2024-10-08', troughDate: '2025-04-17', recoveryDate: '2025-09-01', maxDrawdown: -0.3132, tradingDays: 222, recovered: true }],
    longestUnderwaterTradingDays: 456,
    knownFailureConditions: ['震荡市反复换仓', '快速反转', '隔夜跳空', 'QDII溢价、汇率和美股时差'],
  },
  chartData: {
    wealth: [{ date: '2026-07-24', strategy: 4.1, SZ159915: 2.99, SH513100: 0.9, equalWeight: 1.94 }],
    drawdown: [{ date: '2026-07-24', drawdown: -0.22 }],
    quarterlyReturns: [{ period: '2026Q1', return: 0.08, positive: true }],
    yearlyReturns: [{ period: '2025', return: -0.05, positive: false }],
    monthlyAllocations: [{ month: '2026-07', SZ159915: 0.3, SH513100: 0.4, CASH: 0.3 }],
    costStress: [{ label: '20bp', annualizedReturn: 0.2179, maxDrawdown: -0.3132 }],
    startDateSensitivity: [{ startDate: '2023-01-03', annualizedReturn: 0.2, maxDrawdown: -0.3 }],
    anomalyDates: ['2022-01-14'],
  },
  dataQuality: {
    status: 'unverified_data',
    selectedSource: 'akshare.fund_etf_hist_sina',
    selectedAdjustment: 'raw_close',
    adjustedSourceAvailable: false,
    coverageRatio: 0,
    paperTrackingDaysCompleted: 0,
    evidenceGaps: [
      'unresolved_price_discontinuity',
      'sixty_trading_day_paper_tracking_not_done',
      'independent_source_cross_check_not_done',
      'dividend_and_split_adjustment_not_verified',
    ],
    rawPriceDiscontinuities: { passed: false, events: [{ symbol: 'SH513100', date: '2022-01-14', closeReturn: -0.8045 }] },
  },
  reproducibility: { configHash: 'config', dataManifestHash: 'data', gitCommit: 'deadbeef', gitDirty: true, generatedAt: '2026-07-25T21:48:09+08:00' },
};

function renderPage(language: 'zh' | 'en' = 'zh') {
  localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, language);
  return render(
    <UiLanguageProvider>
      <MemoryRouter>
        <EtfMomentumReportPage />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
}

describe('EtfMomentumReportPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    getReport.mockResolvedValue(report);
    submitJob.mockResolvedValue({ jobId: 'qj_etf', status: 'completed' });
    getJob.mockResolvedValue({ jobId: 'qj_etf', status: 'completed' });
    Object.defineProperty(window, 'print', { value: vi.fn(), configurable: true });
  });

  it('shows the complete Chinese report with all evidence and charts', async () => {
    renderPage();

    expect(await screen.findByRole('heading', { name: 'ETF 动量轮动完整策略报告' })).toBeInTheDocument();
    expect(screen.getAllByText('不可用于交易').length).toBeGreaterThan(0);
    expect(screen.getByText('63.33%')).toBeInTheDocument();
    expect(screen.getByText('60.00%')).toBeInTheDocument();
    expect(screen.getByText('58.62%')).toBeInTheDocument();
    expect(screen.getAllByTestId('reference-check')).toHaveLength(7);
    expect(screen.getAllByTestId('evidence-gap')).toHaveLength(4);
    expect(screen.getByTestId('wealth-chart')).toBeInTheDocument();
    expect(screen.getByTestId('drawdown-chart')).toBeInTheDocument();
    expect(screen.getByTestId('quarterly-chart')).toBeInTheDocument();
    expect(screen.getByTestId('yearly-chart')).toBeInTheDocument();
    expect(screen.getByTestId('parameter-grid')).toBeInTheDocument();
    expect(screen.getByText('2022-01-14')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '打印或导出 PDF' }));
    expect(window.print).toHaveBeenCalledTimes(1);
  });

  it('supports English and regenerates through the whitelisted job', async () => {
    renderPage('en');

    expect(await screen.findByRole('heading', { name: 'Complete ETF Momentum Rotation Report' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Recalculate report' }));

    await waitFor(() => expect(submitJob).toHaveBeenCalledWith('generate_etf_momentum_report'));
    await waitFor(() => expect(getReport).toHaveBeenCalledTimes(2));
  });

  it('offers report generation when the artifact is missing', async () => {
    getReport.mockResolvedValue({
      status: 'missing',
      reasonCode: 'report_not_generated',
      message: 'not generated',
    });
    renderPage();

    expect(await screen.findByText('尚未生成 ETF 策略报告')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '生成报告' })).toBeInTheDocument();
  });
});
