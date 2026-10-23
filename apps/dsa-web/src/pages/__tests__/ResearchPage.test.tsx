import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import ResearchPage from '../ResearchPage';

const { runScriptStrategy, listBacktestRuns, listStrategyAssets, saveStrategyAsset, getStrategyAsset, updateStrategyAsset } = vi.hoisted(() => ({
  runScriptStrategy: vi.fn(),
  listBacktestRuns: vi.fn().mockResolvedValue([]),
  listStrategyAssets: vi.fn().mockResolvedValue([]),
  saveStrategyAsset: vi.fn(),
  getStrategyAsset: vi.fn(),
  updateStrategyAsset: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    runScriptStrategy,
    listBacktestRuns,
    listStrategyAssets,
    saveStrategyAsset,
    getStrategyAsset,
    updateStrategyAsset,
  },
}));

// CodeMirror needs layout measurement APIs unavailable in jsdom; render a
// plain textarea with the same aria-label/value contract instead.
vi.mock('@uiw/react-codemirror', () => ({
  default: (props: {
    value: string;
    onChange: (value: string) => void;
    'aria-label'?: string;
  }) => {
    const { value, onChange } = props;
    return (
      <textarea
        aria-label={props['aria-label']}
        value={value}
        onChange={event => onChange(event.target.value)}
      />
    );
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
  strategyName: 'python_script',
  runId: 'run-1',
  symbols: ['SH513100', 'SZ159915'],
  period: { startDate: '2020-01-01', endDate: '2020-06-30' },
  strategy: { strategyId: 'python_script' },
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
      <MemoryRouter initialEntries={['/research']}>
        <ResearchPage />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
}

describe('ResearchPage', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, 'zh');
    vi.clearAllMocks();
    runScriptStrategy.mockResolvedValue(report);
    listBacktestRuns.mockResolvedValue([]);
    listStrategyAssets.mockResolvedValue([]);
  });

  it('renders the research workbench with a Python strategy editor', () => {
    renderPage();

    expect(screen.getByRole('heading', { name: '量化研究' })).toBeInTheDocument();
    const editor = screen.getByLabelText('Python交易策略脚本');
    expect((editor as HTMLTextAreaElement).value).toContain('def on_bar(context, data):');
    expect(screen.getByLabelText('股票池')).toBeInTheDocument();
  });

  it('keeps the custom symbol input visible and enables it only for the custom pool', () => {
    renderPage();

    const symbolInput = screen.getByLabelText('标的代码（自选时启用，逗号分隔）') as HTMLInputElement;
    expect(symbolInput).toBeInTheDocument();
    expect(symbolInput.disabled).toBe(true); // 默认 ETF 池：输入框置灰

    fireEvent.change(screen.getByLabelText('股票池'), { target: { value: 'custom' } });
    expect(symbolInput.disabled).toBe(false);
  });

  it('offers preset scripts grouped with saved scripts', async () => {
    renderPage();

    const select = screen.getByLabelText('已保存的脚本') as HTMLSelectElement;
    await waitFor(() => expect(select).toBeEnabled());
    fireEvent.click(screen.getByRole('tab', { name: '脚本模板' }));
    const presetOptions = Array.from(select.querySelectorAll('optgroup[label="预置"] option')).map(
      option => (option as HTMLOptionElement).textContent,
    );
    expect(presetOptions).toEqual([
      '动量轮动',
      '多股动量轮动',
      '动量 Top-N 排名',
      '双均线择时',
      '均线金叉',
      '多头排列',
      '60 日新高突破',
      'RSI 超卖反转',
      '布林带回归',
      '等权买入持有',
    ]);

    fireEvent.change(select, { target: { value: 'preset:dual_ma_timing' } });
    const editor = screen.getByLabelText('Python交易策略脚本') as HTMLTextAreaElement;
    expect(editor.value).toContain('context.fast = 5');
    expect(editor.value).toContain('data.universe()');
  });

  it('runs the strategy script and shows the result', async () => {
    renderPage();

    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => expect(runScriptStrategy).toHaveBeenCalled());
    const request = runScriptStrategy.mock.calls[0][0];
    expect(request.pool).toBe('etf');
    expect(request.symbols).toBeUndefined();
    expect(request.script).toContain('data.rank_by_return');
    expect(await screen.findByText('cn_etf_daily')).toBeInTheDocument();
    expect(listBacktestRuns).toHaveBeenCalled();
  });

  it('shows the run history empty state', async () => {
    renderPage();

    await waitFor(() => expect(listBacktestRuns).toHaveBeenCalled());
    expect(screen.getByText('暂无回测记录')).toBeInTheDocument();
  });

  it('saves the current script as a named asset', async () => {
    saveStrategyAsset.mockResolvedValue({
      id: 's1',
      name: '动量选股',
      description: '',
      scriptSha256: 'x'.repeat(64),
      apiVersion: 'dsa-script-v1',
      createdAt: 't',
      updatedAt: 't',
      script: '',
    });
    renderPage();

    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '动量选股' } });
    fireEvent.change(screen.getByLabelText('开始日期'), { target: { value: '2020-01-01' } });
    fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: '2020-06-30' } });
    await waitFor(() => expect(screen.getByRole('button', { name: '保存脚本' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '保存脚本' }));

    await waitFor(() => expect(saveStrategyAsset).toHaveBeenCalledWith({
      name: '动量选股',
      script: expect.stringContaining('rank_by_return'),
    }));
    await waitFor(() => expect(listStrategyAssets).toHaveBeenCalled());
  });

  function mockLibrary() {
    const assets = [
      { id: 'main', name: 'TQM-20 V1.6', archived: false, script: 'def on_bar(c, d):\n    c.target_cash()' },
      { id: 'old', name: '3ETF 历史版本', archived: true, script: 'def on_bar(c, d):\n    c.target_equal([])' },
    ];
    listStrategyAssets.mockImplementation(async (params?: { archived?: boolean }) => (
      assets.filter(item => item.archived === (params?.archived ?? false))
    ));
    getStrategyAsset.mockImplementation(async (id: string) => assets.find(item => item.id === id));
    updateStrategyAsset.mockImplementation(async (id: string, patch: { archived: boolean }) => {
      const asset = assets.find(item => item.id === id)!;
      asset.archived = patch.archived;
      return asset;
    });
  }

  it('defaults to main strategies and loads archived scripts in a separate view', async () => {
    mockLibrary();
    renderPage();

    const select = screen.getByLabelText('已保存的脚本') as HTMLSelectElement;
    await waitFor(() => expect(select).toHaveTextContent('TQM-20 V1.6'));
    expect(select).not.toHaveTextContent('3ETF 历史版本');
    expect(select.querySelector('optgroup[label="预置"]')).toBeNull();

    fireEvent.click(screen.getByRole('tab', { name: '历史归档' }));
    expect(select).toHaveTextContent('3ETF 历史版本');
    expect(select).not.toHaveTextContent('TQM-20 V1.6');
    fireEvent.change(select, { target: { value: 'old' } });
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('3ETF 历史版本'));
    expect((screen.getByLabelText('Python交易策略脚本') as HTMLTextAreaElement).value).toContain('target_equal');

    fireEvent.click(screen.getByRole('tab', { name: '主策略' }));
    expect(select.value).toBe('');
    expect(screen.getByLabelText('名称')).toHaveValue('');
  });

  it('archives a selected main strategy without deleting it', async () => {
    mockLibrary();
    renderPage();

    const select = screen.getByLabelText('已保存的脚本') as HTMLSelectElement;
    await waitFor(() => expect(select).toHaveTextContent('TQM-20 V1.6'));
    fireEvent.change(select, { target: { value: 'main' } });
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('TQM-20 V1.6'));
    fireEvent.click(screen.getByRole('button', { name: '归档策略' }));

    await waitFor(() => expect(updateStrategyAsset).toHaveBeenCalledWith('main', { archived: true }));
    await waitFor(() => expect(select).not.toHaveTextContent('TQM-20 V1.6'));
    await waitFor(() => expect(screen.getByRole('tab', { name: '历史归档' })).toBeEnabled());
    fireEvent.click(screen.getByRole('tab', { name: '历史归档' }));
    expect(select).toHaveTextContent('TQM-20 V1.6');
    expect(screen.queryByRole('button', { name: '删除脚本' })).not.toBeInTheDocument();
  });

  it('restores an archived strategy to the main list', async () => {
    mockLibrary();
    renderPage();

    const select = screen.getByLabelText('已保存的脚本') as HTMLSelectElement;
    await waitFor(() => expect(select).toHaveTextContent('TQM-20 V1.6'));
    fireEvent.click(screen.getByRole('tab', { name: '历史归档' }));
    fireEvent.change(select, { target: { value: 'old' } });
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('3ETF 历史版本'));
    fireEvent.click(screen.getByRole('button', { name: '恢复到主列表' }));

    await waitFor(() => expect(updateStrategyAsset).toHaveBeenCalledWith('old', { archived: false }));
    await waitFor(() => expect(select).not.toHaveTextContent('3ETF 历史版本'));
    await waitFor(() => expect(screen.getByRole('tab', { name: '主策略' })).toBeEnabled());
    fireEvent.click(screen.getByRole('tab', { name: '主策略' }));
    expect(select).toHaveTextContent('3ETF 历史版本');
  });
});
