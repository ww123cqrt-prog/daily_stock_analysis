import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, useNavigate } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import type { TrainedStrategyCatalogStrategy } from '../../types/backtest';
import BacktestPage from '../BacktestPage';

const { getResearchStatus, getTrainedStrategyCatalog, runFormStrategy, runScriptStrategy, runTrainedStrategy } = vi.hoisted(() => ({
  getResearchStatus: vi.fn(),
  getTrainedStrategyCatalog: vi.fn(),
  runFormStrategy: vi.fn(),
  runScriptStrategy: vi.fn(),
  runTrainedStrategy: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    getTrainedStrategyCatalog,
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
    datasetId: 'cn_etf_daily',
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
  providerIdentity: { snapshotId: 'snapshot-candidate' },
  artifactSha256: { 'pred.pkl': 'a'.repeat(64), 'label.pkl': 'b'.repeat(64) },
  strategyConfig: { rebalanceFreq: 5 },
  replayMode: 'exact',
  replayEngine: 'dsa_optimized_strategy',
  modifiedFromOriginal: false,
  configHash: 'c'.repeat(64),
  contractVersion: 1,
  originalParameters: {
    startDate: '2024-01-01',
    endDate: '2026-07-10',
    topk: 50,
    costBps: 15,
  },
  executionAssumptions: {},
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

const BacktestQuerySwitcher = () => {
  const navigate = useNavigate();
  return (
    <>
      <button
        type="button"
        onClick={() => navigate('/backtest?tab=trained&runId=latest-run&modelId=latest&strategyId=baseline_daily')}
      >
        Switch trained Run
      </button>
      <button
        type="button"
        onClick={() => navigate('/backtest?tab=trained&runId=run-xgb&modelId=latest&strategyId=combined_risk')}
      >
        Switch mismatched Run
      </button>
    </>
  );
};

function renderPageWithQuerySwitcher(initialEntry: string) {
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <UiLanguageProvider>
        <BacktestQuerySwitcher />
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
    getTrainedStrategyCatalog.mockResolvedValue({
      status: 'completed',
      generatedAt: '2026-08-02T12:00:00+08:00',
      runs: [
        {
          runId: 'run-xgb',
          modelId: 'xgboost',
          experimentId: '42',
          modelLabel: 'XGBoost',
          sourceType: 'candidate',
          generatedAt: '2026-08-02T10:00:00+08:00',
          provider: 'canonical',
          providerIdentity: { snapshotId: 'snapshot-candidate' },
          periods: {
            training: ['2020-01-02', '2022-12-31'],
            selection: ['2024-01-01', '2025-12-31'],
            holdout: ['2026-01-01', '2026-07-10'],
          },
          originalParameters: {
            startDate: '2024-01-01',
            endDate: '2025-12-31',
            topk: 50,
            costBps: 15,
          },
          strategies: [
            {
              id: 'combined_risk',
              label: '完整组合风控',
              config: { rebalanceFreq: 5 },
              exactReplayAvailable: true,
              replayEngine: 'dsa_optimized_strategy',
            },
          ],
          configHash: 'c'.repeat(64),
          contractVersion: 2,
          exactReplayAvailable: true,
          customReplayAvailable: true,
          replayCapability: 'exact_and_custom',
        },
        {
          runId: 'latest-run',
          modelId: 'latest',
          experimentId: '1',
          modelLabel: 'baseline_lgb_alpha158',
          sourceType: 'baseline',
          generatedAt: '2026-08-02T09:00:00+08:00',
          provider: 'canonical',
          providerIdentity: { snapshotId: 'snapshot-baseline' },
          periods: { test: ['2024-01-01', '2026-07-10'] },
          originalParameters: {
            strategyId: 'baseline_daily',
            startDate: '2024-01-01',
            endDate: '2026-07-10',
            topk: 50,
            costBps: 15,
          },
          strategies: [
            {
              id: 'baseline_daily',
              label: '每日 TopK 等权基线',
              config: { rebalanceFreq: 1 },
              exactReplayAvailable: true,
              replayEngine: 'qlib_portana',
            },
            {
              id: 'low_turnover_weekly',
              label: '每周低换手',
              config: { rebalanceFreq: 5 },
              exactReplayAvailable: false,
              replayEngine: 'dsa_optimized_strategy',
            },
          ],
          configHash: 'd'.repeat(64),
          contractVersion: 2,
          exactReplayAvailable: true,
          customReplayAvailable: true,
          replayCapability: 'exact_and_custom',
        },
      ],
    });
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
    expect(screen.getByText('cn_etf_daily')).toBeInTheDocument();
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
    expect(await screen.findByLabelText('选股模型')).toHaveValue('run-xgb:xgboost');
    expect(screen.getByLabelText('交易策略')).toHaveValue('combined_risk');
    expect(screen.getAllByText('XGBoost').length).toBeGreaterThan(0);
    expect(screen.getAllByText('完整组合风控').length).toBeGreaterThan(0);
    expect(screen.getAllByText('run-xgb').length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: '在 MLflow 中打开 Run run-xgb' })).toHaveAttribute(
      'href',
      '/experiments/#/experiments/42/runs/run-xgb',
    );
    expect(screen.getByText('开发期年化超额')).toBeInTheDocument();
    expect(screen.getByText('9.00%')).toBeInTheDocument();
    expect(screen.getByText('留出期年化超额')).toBeInTheDocument();
    expect(screen.getByText('6.00%')).toBeInTheDocument();
    expect(screen.getByLabelText('回测开始日期')).toHaveValue('2024-01-01');
    expect(screen.getByLabelText('回测结束日期')).toHaveValue('2025-12-31');
    expect(screen.getByLabelText('每次持有股票数（TopK）')).toHaveValue(50);
    expect(screen.getByLabelText('回测成本（基点，1基点=0.01%）')).toHaveValue(15);

    fireEvent.click(screen.getByRole('button', { name: '按原条件重放' }));
    await waitFor(() => expect(runTrainedStrategy).toHaveBeenCalledWith({
      runId: 'run-xgb',
      modelId: 'xgboost',
      strategyId: 'combined_risk',
      replayMode: 'exact',
    }));
    expect(await screen.findByRole('heading', { name: '本次重新计算结果' })).toBeInTheDocument();
    expect(screen.getByText('10.00%')).toBeInTheDocument();
    expect(screen.getByTestId('trained-strategy-equity-chart')).toBeInTheDocument();
    expect(runScriptStrategy).not.toHaveBeenCalled();
    expect(runFormStrategy).not.toHaveBeenCalled();
  });

  it('loads a different trained Run when the query changes without remounting the page', async () => {
    renderPageWithQuerySwitcher('/backtest?tab=trained&runId=run-xgb&modelId=xgboost&strategyId=combined_risk');

    expect(await screen.findByLabelText('选股模型')).toHaveValue('run-xgb:xgboost');
    fireEvent.click(screen.getByRole('button', { name: '按原条件重放' }));
    expect(await screen.findByRole('heading', { name: '本次重新计算结果' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Switch trained Run' }));

    await waitFor(() => {
      expect(screen.getByLabelText('选股模型')).toHaveValue('latest-run:latest');
    });
    expect(screen.getByLabelText('交易策略')).toHaveValue('baseline_daily');
    expect(screen.queryByRole('heading', { name: '本次重新计算结果' })).not.toBeInTheDocument();
  });

  it('clears the loaded selection and result when a changed deep link mismatches Run and model identity', async () => {
    renderPageWithQuerySwitcher('/backtest?tab=trained&runId=run-xgb&modelId=xgboost&strategyId=combined_risk');

    expect(await screen.findByLabelText('选股模型')).toHaveValue('run-xgb:xgboost');
    fireEvent.click(screen.getByRole('button', { name: '按原条件重放' }));
    expect(await screen.findByRole('heading', { name: '本次重新计算结果' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Switch mismatched Run' }));

    await waitFor(() => {
      expect(screen.getByLabelText('选股模型')).toHaveValue('');
    });
    expect(screen.getByLabelText('交易策略')).toHaveValue('');
    expect(screen.getByText('无法加载链接中的回测对象')).toBeInTheDocument();
    expect(screen.getByText(/链接指定的 Run、选股模型或交易策略不可用或彼此不匹配/)).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '本次重新计算结果' })).not.toBeInTheDocument();
  });

  it('marks a candidate result from the compatibility provider as historical', async () => {
    getResearchStatus.mockResolvedValueOnce({
      trainable: true,
      trainingReadiness: { status: 'trainable', summary: 'ready' },
      dataSources: [],
      candidatePipeline: {
        status: 'completed',
        provider: 'canonical',
        providerStatus: 'stale',
        providerCurrent: false,
        providerWarning: '该候选报告使用旧兼容数据，只能作为历史参考。',
        periods: {
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
          },
        ],
      },
    });

    renderPage('/backtest?tab=trained&runId=run-xgb&modelId=xgboost&strategyId=combined_risk');

    expect(
      await screen.findByText('该候选报告使用旧兼容数据，只能作为历史参考。'),
    ).toBeInTheDocument();
    expect(screen.getByText('历史候选结果已加载')).toBeInTheDocument();
  });

  it('shows a catalog failure instead of calling it old compatibility data', async () => {
    window.localStorage.setItem('dsa.uiLanguage', 'en');
    getResearchStatus.mockResolvedValueOnce({
      trainable: false,
      trainingReadiness: { status: 'blocked', summary: 'blocked' },
      dataSources: [],
      latestTraining: {
        status: 'completed',
        experimentName: 'baseline_lgb_alpha158',
        runId: 'latest-run',
        providerStatus: 'catalog_invalid',
        providerCurrent: false,
        providerWarning: '后端中文诊断不应直接显示',
      },
      candidatePipeline: {
        status: 'completed',
        models: [],
        strategyEvaluations: [],
      },
    });

    renderPage('/backtest?tab=trained&runId=latest-run&modelId=latest&strategyId=baseline_daily');

    expect(await screen.findByText('Native V2 blocked')).toBeInTheDocument();
    expect(screen.getByText('Native V2 catalog invalid')).toBeInTheDocument();
    expect(screen.getByText(/current canonical dataset cannot be verified/i)).toBeInTheDocument();
    expect(screen.queryByText(/old compatibility data/i)).not.toBeInTheDocument();
  });

  it('lets the latest trained model use any registered trading strategy', async () => {
    renderPage('/backtest?tab=trained&runId=latest-run&modelId=latest&strategyId=low_turnover_weekly');

    expect(screen.getByRole('tab', { name: '选股模型与交易策略' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: '选股模型与交易策略' })).toBeInTheDocument();
    expect(await screen.findByLabelText('选股模型')).toHaveValue('latest-run:latest');
    expect(screen.getByLabelText('交易策略')).toHaveValue('low_turnover_weekly');
    expect(screen.getByRole('option', { name: '每日 TopK 等权基线' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: '每周低换手' })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'Qlib 默认组合回测' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '按当前设置新回测' }));
    await waitFor(() => expect(runTrainedStrategy).toHaveBeenCalledWith({
      runId: 'latest-run',
      modelId: 'latest',
      strategyId: 'low_turnover_weekly',
      replayMode: 'custom',
      startDate: '2024-01-01',
      endDate: '2026-07-10',
      topk: 50,
      costBps: 15,
    }));
  });

  it('keeps two Runs of the same model separately selectable', async () => {
    const current = await getTrainedStrategyCatalog();
    getTrainedStrategyCatalog.mockResolvedValueOnce({
      ...current,
      runs: [
        current.runs[0],
        { ...current.runs[0], runId: 'run-xgb-old', generatedAt: '2026-07-01T10:00:00+08:00' },
      ],
    });

    renderPage('/backtest?tab=trained&runId=run-xgb-old&modelId=xgboost&strategyId=combined_risk');

    expect(await screen.findByLabelText('选股模型')).toHaveValue('run-xgb-old:xgboost');
    expect(screen.getByRole('option', { name: /run-xgb-old/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Run run-xgb$/ })).toBeInTheDocument();
  });

  it('excludes unavailable historical Runs from the backtest selector and explains why', async () => {
    const current = await getTrainedStrategyCatalog();
    getTrainedStrategyCatalog.mockResolvedValueOnce({
      ...current,
      runs: [
        {
          ...current.runs[0],
          runId: 'unavailable-run',
          providerIdentity: {},
          strategies: [],
          exactReplayAvailable: false,
          customReplayAvailable: false,
          replayCapability: 'unavailable',
          replayUnavailableReason: '该 Run 缺少不可变数据身份或冻结交易策略配置，不能回测。',
        },
        ...current.runs,
      ],
    });

    renderPage('/backtest?tab=trained&runId=unavailable-run&modelId=xgboost&strategyId=combined_risk');

    expect(await screen.findByLabelText('选股模型')).toHaveValue('');
    expect(screen.getByLabelText('交易策略')).toHaveValue('');
    expect(screen.getByText('无法加载链接中的回测对象')).toBeInTheDocument();
    expect(screen.getByText(/链接指定的 Run、选股模型或交易策略不可用或彼此不匹配/)).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /unavailable-run/ })).not.toBeInTheDocument();
    expect(screen.getByText('1 个历史 Run 缺少回测契约')).toBeInTheDocument();
    expect(screen.getByText('该 Run 缺少不可变数据身份或冻结交易策略配置，不能回测。')).toBeInTheDocument();
  });

  it('labels a legacy Run as custom-only and disables exact replay', async () => {
    const current = await getTrainedStrategyCatalog();
    getTrainedStrategyCatalog.mockResolvedValueOnce({
      ...current,
      runs: [{
        ...current.runs[1],
        strategies: current.runs[1].strategies.map((strategy: TrainedStrategyCatalogStrategy) => ({
          ...strategy,
          exactReplayAvailable: false,
        })),
        exactReplayAvailable: false,
        replayCapability: 'custom_only',
        replayUnavailableReason: '旧版 Run 没有可验证的精确重放契约，只能明确标记后自定义回测。',
      }],
    });

    renderPage('/backtest?tab=trained&runId=latest-run&modelId=latest&strategyId=baseline_daily');

    expect(await screen.findByText(/旧版 Run 没有可验证的精确重放契约/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '按原条件重放' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '按当前设置新回测' })).toBeEnabled();
  });
});
