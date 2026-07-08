import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import BacktestPage from '../BacktestPage';

const {
  mockGetResults,
  mockGetOverallPerformance,
  mockGetStockPerformance,
  mockRun,
  mockGetResearchStatus,
} = vi.hoisted(() => ({
  mockGetResults: vi.fn(),
  mockGetOverallPerformance: vi.fn(),
  mockGetStockPerformance: vi.fn(),
  mockRun: vi.fn(),
  mockGetResearchStatus: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    getResults: mockGetResults,
    getOverallPerformance: mockGetOverallPerformance,
    getStockPerformance: mockGetStockPerformance,
    run: mockRun,
  },
}));

vi.mock('../../api/research', () => ({
  researchApi: {
    getStatus: mockGetResearchStatus,
  },
}));

const basePerformance = {
  scope: 'overall',
  evalWindowDays: 10,
  engineVersion: 'test-engine',
  totalEvaluations: 3,
  completedCount: 2,
  insufficientCount: 1,
  longCount: 2,
  cashCount: 1,
  winCount: 1,
  lossCount: 1,
  neutralCount: 0,
  directionAccuracyPct: 66.7,
  winRatePct: 50,
  neutralRatePct: 0,
  avgStockReturnPct: 2.4,
  avgSimulatedReturnPct: 1.2,
  stopLossTriggerRate: 10,
  takeProfitTriggerRate: 20,
  ambiguousRate: 0,
  avgDaysToFirstHit: 3.5,
  adviceBreakdown: {},
  diagnostics: {},
};

const baseResultItem = {
  analysisHistoryId: 101,
  code: '600519',
  stockName: '贵州茅台',
  analysisDate: '2026-03-20',
  evalWindowDays: 10,
  engineVersion: 'test-engine',
  evalStatus: 'completed',
  operationAdvice: '继续持有',
  action: 'watch',
  actionLabel: '观望',
  trendPrediction: '震荡偏多',
  actualMovement: 'up',
  actualReturnPct: 3.8,
  directionExpected: 'long',
  directionCorrect: true,
  outcome: 'win',
  simulatedReturnPct: 3.8,
};

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  mockGetOverallPerformance.mockResolvedValue(basePerformance);
  mockGetStockPerformance.mockResolvedValue(null);
  mockGetResults.mockResolvedValue({
    total: 1,
    page: 1,
    limit: 20,
    items: [baseResultItem],
  });
  mockRun.mockResolvedValue({
    processed: 1,
    saved: 1,
    completed: 1,
    insufficient: 0,
    errors: 0,
  });
  mockGetResearchStatus.mockResolvedValue({
    generatedAt: '2026-07-03T21:00:00+08:00',
    trainable: true,
    trainingReadiness: { status: 'trainable', summary: 'ready' },
    dataSources: [],
    qlibProvider: {
      path: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata',
      calendarDays: 1500,
      calendarRange: '2020-01-02 ~ 2026-07-03',
      instruments: 5300,
    },
    latestTraining: {
      status: 'completed',
      experimentName: 'baseline_lgb_alpha158',
      runId: 'abc123',
      provider: 'cleaned',
      providerUri: '/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata_cleaned',
      sourceConfig: '/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/configs/baseline_lgb_alpha158.yaml',
      runtimeConfig: '/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/reports/runtime_configs/baseline_lgb_alpha158_cleaned_latest.yaml',
      artifacts: ['pred.pkl', 'portfolio_analysis', 'task'],
      metrics: {
        IC: 0.0123,
        '1day.excess_return_with_cost.annualized_return': 0.0339,
      },
    },
    cashControlBacktest: {
      generatedAt: '2026-07-05T09:00:00+08:00',
      mode: 'read_only_cash_control_overlay',
      comparison: {
        baseline: {
          cumulativeReturn: 0.20,
          annualizedReturn: 0.18,
          maxDrawdown: -0.30,
          sharpeRatio: 0.70,
        },
        cashControl: {
          cumulativeReturn: 0.12,
          annualizedReturn: 0.11,
          maxDrawdown: -0.12,
          sharpeRatio: 0.95,
        },
        delta: {
          cumulativeReturn: -0.08,
          annualizedReturn: -0.07,
          maxDrawdown: 0.18,
          sharpeRatio: 0.25,
        },
      },
      signalSummary: {
        latestRegime: 'risk_off',
        latestCashRatio: 0.80,
        averageCashRatio: 0.45,
      },
    },
    qlibPredictions: {
      generatedAt: '2026-07-05T10:00:00+08:00',
      mode: 'read_only_qlib_prediction_summary',
      runId: 'abc123',
      latestDate: '2026-07-02',
      topk: 2,
      top: [
        { instrument: 'SZ000002', score: 0.25 },
        { instrument: 'SH600000', score: 0.05 },
      ],
      bottom: [
        { instrument: 'SZ000001', score: -0.4 },
        { instrument: 'SH600000', score: 0.05 },
      ],
      summary: { latestRows: 3 },
    },
    mlflow: {
      url: 'http://127.0.0.1:6606/',
      role: 'experiment_raw_records',
    },
    experimentNotebook: {
      status: 'available',
      totalRuns: 1,
      latestNote: 'baseline note',
      runs: [
        {
          runId: 'abc123',
          experimentName: 'baseline_lgb_alpha158',
          status: 'completed',
          model: 'LGBModel',
          strategy: 'TopkDropoutStrategy',
          datasetRange: '2020-01-02 ~ 2026-07-01',
          conclusion: '收益质量偏弱，需要补风险控制。',
        },
      ],
    },
  });
});

describe('BacktestPage', () => {
  function renderPage() {
    window.localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    render(
      <MemoryRouter>
        <UiLanguageProvider>
          <BacktestPage />
        </UiLanguageProvider>
      </MemoryRouter>,
    );
  }

  function renderEnglishPage() {
    window.localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'en');
    render(
      <MemoryRouter>
        <UiLanguageProvider>
          <BacktestPage />
        </UiLanguageProvider>
      </MemoryRouter>,
    );
  }

  it('renders shared surface inputs and prediction tracking outputs', async () => {
    renderPage();

    const filterInput = await screen.findByPlaceholderText('按股票代码筛选（留空表示全部）');
    const windowInput = screen.getByPlaceholderText('10');

    expect(filterInput).toHaveClass('input-surface');
    expect(filterInput).toHaveClass('input-focus-glow');
    expect(windowInput).toHaveClass('input-surface');
    expect(windowInput).toHaveClass('input-focus-glow');

    expect(await screen.findByText('盈利')).toBeInTheDocument();
    expect(screen.getByText('已完成')).toBeInTheDocument();
    expect(screen.getByText('600519')).toBeInTheDocument();
    expect(screen.getByText('贵州茅台')).toBeInTheDocument();
    const resultRow = screen.getByText('600519').closest('tr');
    expect(resultRow).not.toBeNull();
    const rowScope = within(resultRow as HTMLElement);
    expect(rowScope.getByText('观望')).toBeInTheDocument();
    expect(rowScope.getByText('震荡偏多')).toBeInTheDocument();
    expect(rowScope.getByText('继续持有')).toBeInTheDocument();
    expect(screen.getByText('上涨')).toBeInTheDocument();
    expect(screen.getByText('窗口收益')).toBeInTheDocument();
    expect(screen.getByText('方向匹配')).toBeInTheDocument();
    expect(screen.getByText('做多')).toBeInTheDocument();
    expect(screen.getAllByLabelText('是').length).toBeGreaterThan(0);
    expect(screen.getByText('方向准确率')).toBeInTheDocument();
    expect(screen.getByText('平均模拟收益')).toBeInTheDocument();
  });

  it('shows an error instead of leaving initial backtest load unhandled', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    mockGetOverallPerformance.mockRejectedValueOnce(new Error('initial performance failed'));

    renderPage();

    expect(await screen.findByRole('alert')).toHaveTextContent('initial performance failed');
    expect(mockGetResults).not.toHaveBeenCalled();

    consoleErrorSpy.mockRestore();
  });

  it('shows Qlib model and strategy backtest results in a separate tab', async () => {
    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Qlib模型/策略回测' }));

    expect(await screen.findByText('Qlib 模型/策略回测')).toBeInTheDocument();
    expect(screen.getAllByText('baseline_lgb_alpha158').length).toBeGreaterThan(0);
    expect(screen.getAllByText('abc123').length).toBeGreaterThan(0);
    expect(screen.getByText('LGBModel')).toBeInTheDocument();
    expect(screen.getByText('TopkDropoutStrategy')).toBeInTheDocument();
    expect(screen.getByText(/收益质量偏弱/)).toBeInTheDocument();
    expect(screen.getByText('2020-01-02 ~ 2026-07-03')).toBeInTheDocument();
    expect(screen.getByText('现金仓位叠加回测')).toBeInTheDocument();
    expect(screen.getByText('risk_off')).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument();
    expect(screen.getByText('Baseline')).toBeInTheDocument();
    expect(screen.getByText('Cash-control')).toBeInTheDocument();
    expect(screen.getByText('最大回撤')).toBeInTheDocument();
    expect(screen.getByText('夏普比率')).toBeInTheDocument();
    expect(screen.getByText('0.95')).toBeInTheDocument();
    expect(screen.getByText('Qlib 预测摘要')).toBeInTheDocument();
    expect(screen.getAllByText('2026-07-02').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('SZ000002')).toBeInTheDocument();
    expect(screen.getByText('SZ000001')).toBeInTheDocument();
    expect(screen.getByText('0.2500')).toBeInTheDocument();
    expect(screen.getByText('-0.4000')).toBeInTheDocument();
    expect(screen.getByText(/只读模型评分摘要/)).toBeInTheDocument();
    expect(screen.getByText(/不自动下单/)).toBeInTheDocument();
    expect(screen.getByText('实验复现与产物')).toBeInTheDocument();
    expect(screen.getByText('数据版本')).toBeInTheDocument();
    expect(screen.getByText('cleaned')).toBeInTheDocument();
    expect(screen.getByText('Provider 路径')).toBeInTheDocument();
    expect(screen.getByText('/Users/cq/WorkingProjects/quant_data/qlib/cn_data_baostock_jqdata_cleaned')).toBeInTheDocument();
    expect(screen.getByText('运行配置')).toBeInTheDocument();
    expect(screen.getByText('/Users/cq/WorkingProjects/daily_stock_analysis/quant_workflow/reports/runtime_configs/baseline_lgb_alpha158_cleaned_latest.yaml')).toBeInTheDocument();
    expect(screen.getByText('MLflow Run')).toBeInTheDocument();
    expect(screen.getByText('pred.pkl')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '打开 MLflow' })).toHaveAttribute('href', 'http://127.0.0.1:6606/');
    expect(screen.getByRole('link', { name: '运行训练任务' })).toHaveAttribute('href', '/workflow');
  });

  it('shows an empty cash-control state when no cash-control report exists', async () => {
    mockGetResearchStatus.mockResolvedValueOnce({
      generatedAt: '2026-07-03T21:00:00+08:00',
      trainable: true,
      trainingReadiness: { status: 'trainable', summary: 'ready' },
      dataSources: [],
      latestTraining: null,
      qlibProvider: null,
      cashControlBacktest: null,
      experimentNotebook: { status: 'missing', totalRuns: 0, runs: [] },
    });

    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Qlib模型/策略回测' }));

    expect(await screen.findByText('暂无现金仓位叠加报告。先在工作流页面运行“生成现金仓位叠加回测”。')).toBeInTheDocument();
  });

  it('keeps cash-control card stable when optional metrics are missing', async () => {
    mockGetResearchStatus.mockResolvedValueOnce({
      generatedAt: '2026-07-03T21:00:00+08:00',
      trainable: true,
      trainingReadiness: { status: 'trainable', summary: 'ready' },
      dataSources: [],
      qlibProvider: null,
      latestTraining: null,
      cashControlBacktest: {
        generatedAt: '2026-07-05T09:00:00+08:00',
        mode: 'read_only_cash_control_overlay',
        comparison: {
          baseline: {
            cumulativeReturn: null,
          },
          cashControl: {
            sharpeRatio: null,
          },
          delta: {},
        },
        signalSummary: {
          latestRegime: null,
          latestCashRatio: null,
          averageCashRatio: null,
        },
      },
      experimentNotebook: { status: 'missing', totalRuns: 0, runs: [] },
    });

    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Qlib模型/策略回测' }));

    expect(await screen.findByText('现金仓位叠加回测')).toBeInTheDocument();
    expect(screen.getByText('夏普比率')).toBeInTheDocument();
    expect(screen.getAllByText('--').length).toBeGreaterThanOrEqual(5);
  });

  it('shows a localized Qlib status error when research status loading fails', async () => {
    mockGetResearchStatus.mockRejectedValueOnce(new Error('research failed'));

    renderPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Qlib模型/策略回测' }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Qlib 状态读取失败');
    expect(screen.getByRole('alert')).toHaveTextContent('research failed');
  });

  it('refreshes Qlib research status each time the Qlib backtest tab is reopened', async () => {
    renderPage();

    expect(await screen.findByText('600519')).toBeInTheDocument();
    expect(mockGetResearchStatus).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Qlib模型/策略回测' }));

    await waitFor(() => {
      expect(mockGetResearchStatus).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByText('Qlib 模型/策略回测')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'DSA分析准确性回测' }));
    fireEvent.click(screen.getByRole('button', { name: 'Qlib模型/策略回测' }));

    await waitFor(() => {
      expect(mockGetResearchStatus).toHaveBeenCalledTimes(2);
    });
  });

  it('renders the Qlib model and strategy tab in English UI mode', async () => {
    renderEnglishPage();

    fireEvent.click(await screen.findByRole('button', { name: 'Qlib model/strategy backtest' }));

    await waitFor(() => {
      expect(screen.getAllByText('Qlib model/strategy backtest').length).toBeGreaterThan(1);
    });
    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('Strategy')).toBeInTheDocument();
    expect(screen.getByText('Latest training metrics')).toBeInTheDocument();
    expect(screen.getByText('Qlib prediction summary')).toBeInTheDocument();
    expect(screen.getByText('Reproducibility and artifacts')).toBeInTheDocument();
    expect(screen.getByText('Recent experiment records')).toBeInTheDocument();
    expect(screen.queryByText('Qlib 模型/策略回测')).not.toBeInTheDocument();
  });

  it('falls back to the taxonomy label when backtest actionLabel is missing', async () => {
    mockGetResults.mockResolvedValueOnce({
      total: 1,
      page: 1,
      limit: 20,
      items: [
        {
          ...baseResultItem,
          action: 'watch',
          actionLabel: null,
        },
      ],
    });

    renderPage();

    const codeCell = await screen.findByText('600519');
    const resultRow = codeCell.closest('tr');
    expect(resultRow).not.toBeNull();
    const rowScope = within(resultRow as HTMLElement);
    expect(rowScope.getByText('观望')).toBeInTheDocument();
    expect(rowScope.getByText('继续持有')).toBeInTheDocument();
  });

  it('uses localized taxonomy labels before server labels in English UI mode', async () => {
    mockGetResults.mockResolvedValueOnce({
      total: 1,
      page: 1,
      limit: 20,
      items: [
        {
          ...baseResultItem,
          operationAdvice: 'continue holding',
          action: 'watch',
          actionLabel: '观望',
          trendPrediction: 'range-bound',
        },
      ],
    });

    renderEnglishPage();

    const codeCell = await screen.findByText('600519');
    const resultRow = codeCell.closest('tr');
    expect(resultRow).not.toBeNull();
    const rowScope = within(resultRow as HTMLElement);
    expect(rowScope.getByText('Watch')).toBeInTheDocument();
    expect(rowScope.getByText('continue holding')).toBeInTheDocument();
    expect(rowScope.queryByText('观望')).not.toBeInTheDocument();
  });

  it('keeps operation advice visible when backtest action fields are absent for multi-guard advice', async () => {
    mockGetResults.mockResolvedValueOnce({
      total: 1,
      page: 1,
      limit: 20,
      items: [
        {
          ...baseResultItem,
          operationAdvice: 'risk alert, avoid buying',
          action: null,
          actionLabel: null,
        },
      ],
    });

    renderPage();

    const codeCell = await screen.findByText('600519');
    const resultRow = codeCell.closest('tr');
    expect(resultRow).not.toBeNull();
    const rowScope = within(resultRow as HTMLElement);
    expect(rowScope.getByText('震荡偏多')).toBeInTheDocument();
    expect(rowScope.getByText('risk alert, avoid buying')).toBeInTheDocument();
    expect(rowScope.queryByText('回避')).not.toBeInTheDocument();
    expect(rowScope.queryByText('预警')).not.toBeInTheDocument();
  });

  it('renders backtest controls and result headings in English UI mode', async () => {
    renderEnglishPage();

    expect(await screen.findByPlaceholderText('Filter by stock code (leave empty for all)')).toBeInTheDocument();
    expect(screen.getByText('Evaluation window')).toBeInTheDocument();
    expect(screen.getAllByText('Phase').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: 'Run backtest' })).toBeInTheDocument();
    expect(screen.getByText('Window return')).toBeInTheDocument();
    expect(screen.getByText('Direction match')).toBeInTheDocument();
    expect(screen.getByText('Direction accuracy')).toBeInTheDocument();
    expect(screen.queryByText('运行回测')).not.toBeInTheDocument();
    expect(screen.queryByText('窗口收益')).not.toBeInTheDocument();
  });

  it('filters results with stock code, window, phase, and analysis date range when clicking Filter', async () => {
    renderPage();

    const filterInput = await screen.findByPlaceholderText('按股票代码筛选（留空表示全部）');
    const windowInput = screen.getByPlaceholderText('10');
    const phaseSelect = screen.getByDisplayValue('全部阶段');
    const fromInput = screen.getByLabelText('分析开始日期');
    const toInput = screen.getByLabelText('分析结束日期');

    fireEvent.change(filterInput, { target: { value: 'aapl' } });
    fireEvent.change(windowInput, { target: { value: '20' } });
    fireEvent.change(phaseSelect, { target: { value: 'intraday' } });
    fireEvent.change(fromInput, { target: { value: '2026-03-01' } });
    fireEvent.change(toInput, { target: { value: '2026-03-31' } });
    fireEvent.click(screen.getByRole('button', { name: '筛选' }));

    await waitFor(() => {
      expect(mockGetResults).toHaveBeenLastCalledWith({
        code: 'AAPL',
        evalWindowDays: 20,
        analysisDateFrom: '2026-03-01',
        analysisDateTo: '2026-03-31',
        analysisPhase: 'intraday',
        page: 1,
        limit: 20,
      });
      expect(mockGetStockPerformance).toHaveBeenLastCalledWith('AAPL', {
        evalWindowDays: 20,
        analysisDateFrom: '2026-03-01',
        analysisDateTo: '2026-03-31',
        analysisPhase: 'intraday',
      });
    });
  });

  it('runs a backtest and refreshes results using the shared filter values', async () => {
    renderPage();

    const filterInput = await screen.findByPlaceholderText('按股票代码筛选（留空表示全部）');
    const windowInput = screen.getByPlaceholderText('10');

    fireEvent.change(filterInput, { target: { value: 'tsla' } });
    fireEvent.change(windowInput, { target: { value: '15' } });
    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => {
      expect(mockRun).toHaveBeenCalledWith({
        code: 'TSLA',
        force: undefined,
        minAgeDays: undefined,
        evalWindowDays: 15,
      });
    });

    await waitFor(() => {
      expect(mockGetResults).toHaveBeenLastCalledWith({
        code: 'TSLA',
        evalWindowDays: 15,
        analysisDateFrom: undefined,
        analysisDateTo: undefined,
        analysisPhase: undefined,
        page: 1,
        limit: 20,
      });
      expect(mockGetStockPerformance).toHaveBeenLastCalledWith('TSLA', {
        evalWindowDays: 15,
        analysisDateFrom: undefined,
        analysisDateTo: undefined,
        analysisPhase: undefined,
      });
    });

    expect(await screen.findByText('已处理:')).toBeInTheDocument();
    expect(screen.getByText('已保存:')).toBeInTheDocument();
  });

  it('switches to next-day validation with the 1D shortcut', async () => {
    renderPage();

    await screen.findByPlaceholderText('按股票代码筛选（留空表示全部）');
    fireEvent.click(screen.getByRole('button', { name: '1 日验证' }));

    await waitFor(() => {
      expect(mockGetResults).toHaveBeenLastCalledWith({
        code: undefined,
        evalWindowDays: 1,
        analysisDateFrom: undefined,
        analysisDateTo: undefined,
        analysisPhase: undefined,
        page: 1,
        limit: 20,
      });
      expect(mockGetOverallPerformance).toHaveBeenLastCalledWith({
        evalWindowDays: 1,
        analysisDateFrom: undefined,
        analysisDateTo: undefined,
        analysisPhase: undefined,
      });
    });

    expect(screen.getByText('实际表现')).toBeInTheDocument();
    expect(screen.getByText('准确性')).toBeInTheDocument();
    expect(screen.getByText('1 日验证模式会用下一个交易日收盘表现校验 AI 预测。')).toBeInTheDocument();
  });
});
