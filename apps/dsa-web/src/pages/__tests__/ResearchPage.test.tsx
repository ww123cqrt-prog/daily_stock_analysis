import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { UiLanguageProvider } from '../../contexts/UiLanguageContext';
import { UI_LANGUAGE_STORAGE_KEY } from '../../utils/uiLanguage';
import type { StrategyAssetDetail } from '../../types/backtest';
import ResearchPage from '../ResearchPage';

const { runScriptStrategy, runStabilityTest, listBacktestRuns, listStrategyAssets, saveStrategyAsset, getStrategyAsset, updateStrategyAsset } = vi.hoisted(() => ({
  runScriptStrategy: vi.fn(),
  runStabilityTest: vi.fn(),
  listBacktestRuns: vi.fn().mockResolvedValue([]),
  listStrategyAssets: vi.fn().mockResolvedValue([]),
  saveStrategyAsset: vi.fn(),
  getStrategyAsset: vi.fn(),
  updateStrategyAsset: vi.fn(),
}));

vi.mock('../../api/backtest', () => ({
  backtestApi: {
    runScriptStrategy,
    runStabilityTest,
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

async function renderPage() {
  const view = render(
    <UiLanguageProvider>
      <MemoryRouter initialEntries={['/research']}>
        <ResearchPage />
      </MemoryRouter>
    </UiLanguageProvider>,
  );
  await waitFor(() => expect(screen.getByLabelText('已保存的脚本')).toBeEnabled());
  return view;
}

function setBacktestDates(startDate: string, endDate: string) {
  fireEvent.change(screen.getByLabelText('开始日期'), { target: { value: startDate } });
  fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: endDate } });
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

  it('renders the research workbench with a Python strategy editor', async () => {
    await renderPage();

    expect(screen.getByRole('heading', { name: '量化研究' })).toBeInTheDocument();
    const editor = screen.getByLabelText('Python交易策略脚本');
    expect((editor as HTMLTextAreaElement).value).toContain('def on_bar(context, data):');
    expect(screen.getByLabelText('股票池')).toBeInTheDocument();
  });

  it('keeps the custom symbol input visible and enables it only for the custom pool', async () => {
    await renderPage();

    const symbolInput = screen.getByLabelText('标的代码（自选时启用，逗号分隔）') as HTMLInputElement;
    expect(symbolInput).toBeInTheDocument();
    expect(symbolInput.disabled).toBe(true); // 默认 ETF 池：输入框置灰

    fireEvent.change(screen.getByLabelText('股票池'), { target: { value: 'custom' } });
    expect(symbolInput.disabled).toBe(false);
  });

  it('offers preset scripts grouped with saved scripts', async () => {
    await renderPage();

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
    await renderPage();

    setBacktestDates('2020-01-01', '2020-06-30');
    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    await waitFor(() => expect(runScriptStrategy).toHaveBeenCalled());
    const request = runScriptStrategy.mock.calls[0][0];
    expect(request.pool).toBe('etf');
    expect(request.symbols).toBeUndefined();
    expect(request.script).toContain('data.rank_by_return');
    expect(request.startDate).toBe('2020-01-01');
    expect(request.endDate).toBe('2020-06-30');
    expect(await screen.findByText('cn_etf_daily')).toBeInTheDocument();
    expect(listBacktestRuns).toHaveBeenCalled();
  });

  it.each([
    { startDate: '', endDate: '' },
    { startDate: '', endDate: '2020-06-30' },
    { startDate: '2020-01-01', endDate: '' },
    { startDate: '2020-07-01', endDate: '2020-06-30' },
  ])('rejects invalid dates before running a script ($startDate, $endDate)', async ({ startDate, endDate }) => {
    await renderPage();

    setBacktestDates(startDate, endDate);
    fireEvent.click(screen.getByRole('button', { name: '运行回测' }));

    expect(screen.getByRole('alert')).toHaveTextContent('请选择有效的开始和结束日期。');
    expect(runScriptStrategy).not.toHaveBeenCalled();
  });

  it.each([
    { startDate: '', endDate: '' },
    { startDate: '2020-07-01', endDate: '2020-06-30' },
  ])('keeps date validation for stability tests ($startDate, $endDate)', async ({ startDate, endDate }) => {
    await renderPage();

    setBacktestDates(startDate, endDate);
    fireEvent.click(screen.getByRole('button', { name: '运行稳定性测试' }));

    expect(screen.getByRole('alert')).toHaveTextContent('请选择有效的开始和结束日期。');
    expect(runStabilityTest).not.toHaveBeenCalled();
  });

  it('shows the run history empty state', async () => {
    await renderPage();

    await waitFor(() => expect(listBacktestRuns).toHaveBeenCalled());
    expect(screen.getByText('暂无回测记录')).toBeInTheDocument();
  });

  it.each([
    { startDate: '', endDate: '' },
    { startDate: '2020-07-01', endDate: '2020-06-30' },
    { startDate: '2020-01-01', endDate: '2020-06-30' },
  ])('saves a named script independently of backtest dates ($startDate, $endDate)', async ({ startDate, endDate }) => {
    saveStrategyAsset.mockResolvedValue({
      id: 's1',
      name: '动量选股',
      description: '',
      scriptSha256: 'x'.repeat(64),
      apiVersion: 'dsa-script-v1',
      archived: false,
      createdAt: 't',
      updatedAt: 't',
      script: '',
    });
    await renderPage();

    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '动量选股' } });
    setBacktestDates(startDate, endDate);
    await waitFor(() => expect(screen.getByRole('button', { name: '保存脚本' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '保存脚本' }));

    await waitFor(() => expect(saveStrategyAsset).toHaveBeenCalledWith({
      name: '动量选股',
      script: expect.stringContaining('rank_by_return'),
    }));
    await waitFor(() => expect(screen.getByRole('button', { name: '保存修改' })).toBeEnabled());
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(updateStrategyAsset).not.toHaveBeenCalled();
  });

  it.each([
    { startDate: '', endDate: '' },
    { startDate: '2020-07-01', endDate: '2020-06-30' },
  ])('updates a saved script independently of backtest dates ($startDate, $endDate)', async ({ startDate, endDate }) => {
    const asset: StrategyAssetDetail = {
      id: 'editing',
      name: '原策略',
      description: '',
      scriptSha256: 'x'.repeat(64),
      apiVersion: 'dsa-script-v1',
      archived: false,
      createdAt: '2020-01-01T00:00:00Z',
      updatedAt: '2020-01-01T00:00:00Z',
      script: 'def on_bar(context, data):\n    context.target_cash()',
    };
    const editedScript = 'def on_bar(context, data):\n    context.target_equal(data.universe())';
    listStrategyAssets.mockImplementation(async (params?: { archived?: boolean }) => params?.archived ? [] : [asset]);
    getStrategyAsset.mockResolvedValue(asset);
    updateStrategyAsset.mockResolvedValue({ ...asset, name: '修改后策略', script: editedScript });
    await renderPage();

    const select = screen.getByLabelText('已保存的脚本');
    await waitFor(() => expect(select).toBeEnabled());
    fireEvent.change(select, { target: { value: 'editing' } });
    await waitFor(() => expect(screen.getByRole('button', { name: '保存修改' })).toBeEnabled());
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '  修改后策略  ' } });
    fireEvent.change(screen.getByLabelText('Python交易策略脚本'), { target: { value: editedScript } });
    setBacktestDates(startDate, endDate);
    fireEvent.click(screen.getByRole('button', { name: '保存修改' }));

    await waitFor(() => expect(updateStrategyAsset).toHaveBeenCalledWith('editing', {
      name: '修改后策略',
      script: editedScript,
    }));
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('修改后策略'));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(saveStrategyAsset).not.toHaveBeenCalled();
  });

  it('keeps saving disabled when the strategy name is blank', async () => {
    await renderPage();

    await waitFor(() => expect(screen.getByLabelText('名称')).toBeEnabled());
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '   ' } });
    const save = screen.getByRole('button', { name: '保存脚本' });
    expect(save).toBeDisabled();
    fireEvent.click(save);

    expect(saveStrategyAsset).not.toHaveBeenCalled();
    expect(updateStrategyAsset).not.toHaveBeenCalled();
  });

  it('rejects saving an empty strategy script', async () => {
    await renderPage();

    await waitFor(() => expect(screen.getByLabelText('名称')).toBeEnabled());
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '空策略' } });
    fireEvent.change(screen.getByLabelText('Python交易策略脚本'), { target: { value: '  \n ' } });
    setBacktestDates('2020-01-01', '2020-06-30');
    fireEvent.click(screen.getByRole('button', { name: '保存脚本' }));

    expect(screen.getByRole('alert')).toHaveTextContent('交易策略脚本不能为空。');
    expect(saveStrategyAsset).not.toHaveBeenCalled();
    expect(updateStrategyAsset).not.toHaveBeenCalled();
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
    await renderPage();

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
    await renderPage();

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
    await renderPage();

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
