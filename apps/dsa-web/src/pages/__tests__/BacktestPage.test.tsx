import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import BacktestPage from '../BacktestPage';

const { runFormStrategy } = vi.hoisted(() => ({
  runFormStrategy: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    runFormStrategy,
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
  },
  benchmark: { kind: 'equal_weight_buy_and_hold' },
  assumptions: { sideCostPct: 0, lotSize: 100 },
  equityCurve: [{
    date: '2020-01-02',
    strategyEquity: 100000,
    benchmarkEquity: 100000,
    strategyReturnPct: 0,
    benchmarkReturnPct: 0,
    drawdownPct: 0,
    benchmarkDrawdownPct: 0,
    position: 'CASH',
  }],
  trades: [],
  warnings: [],
  logs: [],
  scriptCheck: { status: 'passed' },
};

function renderPage() {
  return render(
    <UiLanguageProvider>
      <MemoryRouter initialEntries={['/backtest']}>
        <BacktestPage />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
}

describe('BacktestPage', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    vi.clearAllMocks();
    runFormStrategy.mockResolvedValue(report);
  });

  it('renders the form strategy backtest with one-line rule description', () => {
    renderPage();

    expect(screen.getByRole('heading', { name: '条件策略回测' })).toBeInTheDocument();
    expect(screen.getByText(/收盘比较过去 20 个交易日涨幅/)).toBeInTheDocument();
    expect(screen.getByLabelText('涨幅回看（交易日）')).toHaveValue(20);
    expect(screen.getByLabelText('均线周期（交易日）')).toHaveValue(20);
  });

  it('runs the form strategy and shows the result', async () => {
    renderPage();

    fireEvent.change(screen.getByLabelText('开始日期'), { target: { value: '2020-01-01' } });
    fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: '2020-06-30' } });
    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => expect(runFormStrategy).toHaveBeenCalledWith(expect.objectContaining({
      strategyId: 'momentum_rotation',
      symbols: ['159915', '513100'],
      startDate: '2020-01-01',
      endDate: '2020-06-30',
      lookbackDays: 20,
      movingAverageDays: 20,
      topN: 1,
    })));
    expect(await screen.findByText('cn_etf_daily')).toBeInTheDocument();
  });

  it.each([
    { startDate: '', endDate: '' },
    { startDate: '2020-07-01', endDate: '2020-06-30' },
  ])('rejects invalid dates before submitting ($startDate, $endDate)', ({ startDate, endDate }) => {
    renderPage();

    fireEvent.change(screen.getByLabelText('开始日期'), { target: { value: startDate } });
    fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: endDate } });
    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    expect(screen.getByRole('alert')).toHaveTextContent('请选择有效的开始和结束日期。');
    expect(runFormStrategy).not.toHaveBeenCalled();
  });
});
