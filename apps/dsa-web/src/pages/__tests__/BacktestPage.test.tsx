import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import BacktestPage from '../BacktestPage';

const { getResearchStatus, runFormStrategy, runScriptStrategy, runTrainedStrategy } = vi.hoisted(() => ({
  getResearchStatus: vi.fn(),
  runFormStrategy: vi.fn(),
  runScriptStrategy: vi.fn(),
  runTrainedStrategy: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    runFormStrategy,
    runScriptStrategy,
    runTrainedStrategy,
  },
}));

vi.mock('../../api/research', () => ({
  researchApi: {
    getStatus: getResearchStatus,
  },
}));

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Line: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  CartesianGrid: () => <div />,
  Tooltip: () => <div />,
  Legend: () => <div />,
}));

const report = {
  status: 'completed',
  strategyName: 'momentum_rotation',
  symbols: ['SH513100', 'SZ159915'],
  period: { startDate: '2020-01-01', endDate: '2020-06-30' },
  strategy: { strategyId: 'momentum_rotation' },
  dataQuality: {
    datasetId: 'etf_momentum_daily_qfq',
    snapshotId: 'snapshot-1',
    priceBasis: 'qfq',
    provider: 'tencent',
  },
  metrics: {
    initialCapital: 100000,
    finalEquity: 120000,
    totalReturnPct: 20,
    annualizedReturnPct: 18,
    benchmarkReturnPct: 12,
    excessReturnPct: 8,
    maxDrawdownPct: -10,
    benchmarkMaxDrawdownPct: -15,
    tradeCount: 4,
    exposurePct: 80,
    totalCost: 0,
    turnoverPct: 300,
    signalCount: 4,
    executedSignalCount: 4,
    skippedSignalCount: 0,
    tradingDays: 100,
    annualizedVolatilityPct: 22,
    sharpeRatio: 0.8,
    alphaPct: 5,
    beta: 0.7,
    winRatePct: 53,
  },
  benchmark: { kind: 'equal_weight_buy_and_hold' },
  assumptions: { sideCostPct: 0, lotSize: 100 },
  equityCurve: [
    {
      date: '2020-01-02',
      strategyEquity: 100000,
      benchmarkEquity: 100000,
      strategyReturnPct: 0,
      benchmarkReturnPct: 0,
      drawdownPct: 0,
      benchmarkDrawdownPct: 0,
      position: 'CASH',
    },
  ],
  trades: [],
  warnings: [],
  logs: [],
  scriptCheck: { status: 'passed' },
};

const trainedReport = {
  status: 'completed',
  generatedAt: '2026-07-26T10:00:00+08:00',
  runId: 'run-xgb',
  modelId: 'xgboost',
  modelLabel: 'XGBoost',
  strategyId: 'combined_risk',
  strategyLabel: '完整组合风控',
  provider: 'canonical',
  period: { startDate: '2024-01-02', endDate: '2026-07-10' },
  parameters: {
    startDate: '2024-01-01',
    endDate: '2026-07-10',
    topk: 50,
    costBps: 15,
  },
  metrics: {
    days: 610,
    totalReturn: 0.42,
    benchmarkTotalReturn: 0.18,
    annualReturn: 0.19,
    benchmarkAnnualReturn: 0.09,
    annualExcessReturn: 0.1,
    informationRatio: 1.1,
    sharpe: 0.9,
    maxDrawdown: -0.17,
    avgTurnover: 0.04,
    avgPosition: 0.8,
  },
  equityCurve: [
    {
      date: '2024-01-02',
      strategyCumulativeReturn: 0.01,
      benchmarkCumulativeReturn: 0.005,
      drawdown: 0,
      benchmarkDrawdown: 0,
      dailyReturn: 0.01,
      benchmarkReturn: 0.005,
      turnover: 0.2,
      positionPct: 1,
    },
  ],
  warnings: [],
};

function renderPage(initialEntry = '/backtest') {
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <UiLanguageProvider>
        <BacktestPage />
      </UiLanguageProvider>
    </MemoryRouter>,
  );
}

describe('BacktestPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    window.localStorage.setItem('dsa.uiLanguage', 'zh');
    runScriptStrategy.mockResolvedValue({ ...report, strategyName: 'python_script' });
    runFormStrategy.mockResolvedValue(report);
    runTrainedStrategy.mockResolvedValue(trainedReport);
    getResearchStatus.mockResolvedValue({
      generatedAt: '2026-07-15T18:00:00+08:00',
      trainable: true,
      trainingReadiness: { status: 'trainable', summary: 'ready' },
      dataSources: [],
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158',
        runId: 'latest-run',
        provider: 'canonical',
        metrics: { IC: 0.02 },
      },
      candidatePipeline: {
        status: 'completed',
        batchId: 'candidate-1',
        experimentName: 'candidate_models_alpha158',
        provider: 'canonical',
        periods: {
          training: ['2020-01-02', '2022-12-31'],
          selection: ['2024-01-01', '2025-12-31'],
          holdout: ['2026-01-01', '2026-07-10'],
        },
        models: [
          {
            id: 'xgboost',
            label: 'XGBoost',
            status: 'completed',
            runId: 'run-xgb',
          },
        ],
        strategyEvaluations: [
          {
            modelId: 'xgboost',
            strategyId: 'combined_risk',
            strategyLabel: '完整组合风控',
            selection: {
              annualReturn: 0.21,
              benchmarkAnnualReturn: 0.12,
              annualExcessReturn: 0.09,
              informationRatio: 0.8,
              maxDrawdown: -0.18,
            },
            holdout: {
              annualReturn: 0.08,
              benchmarkAnnualReturn: 0.02,
              annualExcessReturn: 0.06,
              informationRatio: 0.4,
              maxDrawdown: -0.12,
            },
          },
        ],
        selectedCombination: {
          modelId: 'xgboost',
          strategyId: 'combined_risk',
          strategyLabel: '完整组合风控',
        },
      },
    });
  });

  it('opens a script backtest workbench with actual strategy source', async () => {
    renderPage();

    expect(screen.getByRole('heading', { name: '交易策略回测工作台' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'ETF 动量完整报告' })).toHaveAttribute(
      'href',
      '/backtest/etf-momentum',
    );
    expect(screen.getByRole('tab', { name: '脚本交易策略' })).toHaveAttribute('aria-selected', 'true');
    const editor = screen.getByLabelText('Python交易策略脚本');
    expect((editor as HTMLTextAreaElement).value).toContain('def on_bar(context, data):');
    expect(screen.getByText('context.target_percent / target_equal / target_cash / log')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => expect(runScriptStrategy).toHaveBeenCalled());
    const request = runScriptStrategy.mock.calls[0][0];
    expect(request.symbols).toEqual(['159915', '513100']);
    expect(request.script).toContain('data.rank_by_return');
    expect(await screen.findByText('Alpha（年化）')).toBeInTheDocument();
    expect(screen.getByText('etf_momentum_daily_qfq')).toBeInTheDocument();
  });

  it('switches to a no-code form with explicit strategy rules and parameters', async () => {
    renderPage();
    fireEvent.click(screen.getByRole('tab', { name: '条件交易策略' }));

    expect(screen.getByRole('heading', { name: '创业板ETF-纳指ETF 20日动量轮动交易策略' })).toBeInTheDocument();
    expect(screen.getByText(/每天收盘比较过去20个交易日涨幅/)).toBeInTheDocument();
    expect(screen.getByText(/两者都不符合时空仓/)).toBeInTheDocument();
    expect(screen.getByLabelText('涨幅回看（交易日）')).toHaveValue(20);
    expect(screen.getByLabelText('均线周期（交易日）')).toHaveValue(20);

    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => expect(runFormStrategy).toHaveBeenCalledWith(expect.objectContaining({
      strategyId: 'momentum_rotation',
      symbols: ['159915', '513100'],
      lookbackDays: 20,
      movingAverageDays: 20,
      topN: 1,
    })));
    expect(await screen.findByText('交易策略总收益')).toBeInTheDocument();
  });

  it('loads a trained model and trading strategy from a workflow deep link', async () => {
    renderPage('/backtest?tab=trained&runId=run-xgb&modelId=xgboost&strategyId=combined_risk');

    expect(screen.getByRole('tab', { name: '选股模型与交易策略' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: '选股模型与交易策略' })).toBeInTheDocument();
    expect(screen.getByLabelText('选股模型')).toHaveValue('xgboost');
    expect(screen.getByLabelText('交易策略')).toHaveValue('combined_risk');
    expect(screen.getAllByText('XGBoost').length).toBeGreaterThan(0);
    expect(screen.getAllByText('完整组合风控').length).toBeGreaterThan(0);
    expect(screen.getAllByText('run-xgb').length).toBeGreaterThan(0);
    expect(screen.getByText('开发期年化超额')).toBeInTheDocument();
    expect(screen.getByText('9.00%')).toBeInTheDocument();
    expect(screen.getByText('留出期年化超额')).toBeInTheDocument();
    expect(screen.getByText('6.00%')).toBeInTheDocument();
    expect(screen.getByLabelText('回测开始日期')).toHaveValue('2024-01-01');
    expect(screen.getByLabelText('回测结束日期')).toHaveValue('2026-07-10');
    expect(screen.getByLabelText('每次持有股票数（TopK）')).toHaveValue(50);
    expect(screen.getByLabelText('回测成本（基点，1基点=0.01%）')).toHaveValue(15);

    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
    await waitFor(() => expect(runTrainedStrategy).toHaveBeenCalledWith({
      runId: 'run-xgb',
      modelId: 'xgboost',
      strategyId: 'combined_risk',
      startDate: '2024-01-01',
      endDate: '2026-07-10',
      topk: 50,
      costBps: 15,
    }));
    expect(await screen.findByRole('heading', { name: '本次重新计算结果' })).toBeInTheDocument();
    expect(screen.getByText('10.00%')).toBeInTheDocument();
    expect(screen.getByTestId('trained-strategy-equity-chart')).toBeInTheDocument();
    expect(runScriptStrategy).not.toHaveBeenCalled();
    expect(runFormStrategy).not.toHaveBeenCalled();
  });

  it('lets the latest trained model use any registered trading strategy', async () => {
    renderPage('/backtest?tab=trained&runId=latest-run&modelId=latest&strategyId=low_turnover_weekly');

    expect(screen.getByRole('tab', { name: '选股模型与交易策略' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: '选股模型与交易策略' })).toBeInTheDocument();
    expect(screen.getByLabelText('选股模型')).toHaveValue('latest');
    expect(screen.getByLabelText('交易策略')).toHaveValue('low_turnover_weekly');
    expect(screen.getByRole('option', { name: '每日 TopK 等权基线' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '每周低换手' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Qlib 默认组合回测' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
    await waitFor(() => expect(runTrainedStrategy).toHaveBeenCalledWith({
      runId: 'latest-run',
      modelId: 'latest',
      strategyId: 'low_turnover_weekly',
      startDate: '2024-01-01',
      endDate: '2026-07-10',
      topk: 50,
      costBps: 15,
    }));
  });
});
