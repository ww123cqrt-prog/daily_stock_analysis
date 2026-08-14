import { beforeEach, describe, expect, it, vi } from 'vitest';
import { backtestApi } from '../backtest';

const { get, post, patch, delete: del } = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  patch: vi.fn(),
  delete: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    get,
    post,
    patch,
    delete: del,
    defaults: { baseURL: '' },
  },
}));

describe('backtestApi', () => {
  beforeEach(() => {
    get.mockReset();
    post.mockReset();
    patch.mockReset();
    del.mockReset();
  });

  it('requests overall performance with empty_ok and treats null as empty metrics', async () => {
    get.mockResolvedValueOnce({ data: null });

    const result = await backtestApi.getOverallPerformance({
      evalWindowDays: 1,
      analysisPhase: 'all',
    });

    expect(result).toBeNull();
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/performance', {
      params: {
        empty_ok: true,
        eval_window_days: 1,
      },
    });
  });

  it('requests stock performance with empty_ok and preserves server metrics', async () => {
    get.mockResolvedValueOnce({
      data: {
        direction_accuracy_pct: 0.75,
        eval_window_days: 10,
      },
    });

    const result = await backtestApi.getStockPerformance('SH 600519', {
      analysisDateFrom: '2026-01-01',
    });

    expect(result?.directionAccuracyPct).toBe(0.75);
    expect(result?.evalWindowDays).toBe(10);
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/performance/SH%20600519', {
      params: {
        empty_ok: true,
        analysis_date_from: '2026-01-01',
      },
    });
  });

  it('submits explicit strategy execution assumptions without inventing costs', async () => {
    post.mockResolvedValueOnce({
      data: {
        status: 'completed',
        strategy_name: 'dsa_history_signal',
        symbols: ['600519'],
        metrics: { total_return_pct: 3.5 },
        assumptions: { side_cost_pct: 0 },
        equity_curve: [],
        trades: [],
        warnings: [],
      },
    });

    const result = await backtestApi.runStrategy({
      code: '600519',
      startDate: '2026-01-01',
      endDate: '2026-06-30',
      initialCapital: 100000,
      sideCostPct: 0,
      lotSize: 100,
    });

    expect(result.metrics.totalReturnPct).toBe(3.5);
    expect(result.assumptions.sideCostPct).toBe(0);
    expect(post).toHaveBeenCalledWith('/api/v1/backtest/strategy/dsa-signal', {
      code: '600519',
      start_date: '2026-01-01',
      end_date: '2026-06-30',
      initial_capital: 100000,
      side_cost_pct: 0,
      lot_size: 100,
    });
  });

  it('submits a no-code strategy and all visible form parameters', async () => {
    post.mockResolvedValueOnce({
      data: {
        status: 'completed',
        strategy_name: 'momentum_rotation',
        symbols: ['SZ159915', 'SH513100'],
        metrics: { total_return_pct: 8.5 },
        assumptions: { side_cost_pct: 0 },
        equity_curve: [],
        trades: [],
        warnings: [],
        logs: [],
        script_check: {},
      },
    });

    const result = await backtestApi.runFormStrategy({
      strategyId: 'momentum_rotation',
      symbols: ['159915', '513100'],
      startDate: '2020-01-01',
      endDate: '2026-07-24',
      initialCapital: 100000,
      sideCostPct: 0,
      lotSize: 100,
      lookbackDays: 20,
      movingAverageDays: 20,
      fastMaDays: 5,
      slowMaDays: 20,
      topN: 1,
    });

    expect(result.strategyName).toBe('momentum_rotation');
    expect(post).toHaveBeenCalledWith('/api/v1/backtest/strategy/run', {
      strategy_id: 'momentum_rotation',
      symbols: ['159915', '513100'],
      start_date: '2020-01-01',
      end_date: '2026-07-24',
      initial_capital: 100000,
      side_cost_pct: 0,
      lot_size: 100,
      lookback_days: 20,
      moving_average_days: 20,
      fast_ma_days: 5,
      slow_ma_days: 20,
      top_n: 1,
    });
  });

  it('submits the actual Python source to the script endpoint', async () => {
    post.mockResolvedValueOnce({
      data: {
        status: 'completed',
        strategy_name: 'python_script',
        symbols: ['SZ159915'],
        metrics: { total_return_pct: 1.25 },
        assumptions: { side_cost_pct: 0 },
        equity_curve: [],
        trades: [],
        warnings: [],
        logs: ['started'],
        script_check: { status: 'passed' },
      },
    });
    const script = 'def on_bar(context, data):\n    context.target_cash()';

    const result = await backtestApi.runScriptStrategy({
      script,
      pool: 'csi800',
      initialCapital: 50000,
      sideCostPct: 0,
      lotSize: 100,
    });

    expect(result.scriptCheck.status).toBe('passed');
    expect(post).toHaveBeenCalledWith('/api/v1/backtest/strategy/script', {
      script,
      pool: 'csi800',
      symbols: [],
      benchmarks: [],
      comparison_strategy_ids: [],
      comparison_presets: [],
      start_date: undefined,
      end_date: undefined,
      initial_capital: 50000,
      side_cost_pct: 0,
      lot_size: 100,
      strategy_asset_id: undefined,
    }, { timeout: 600000 });
  });

  it('reruns a selected trained model and strategy with the visible parameters', async () => {
    post.mockResolvedValueOnce({
      data: {
        status: 'completed',
        run_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        model_id: 'xgboost',
        strategy_id: 'low_turnover_weekly',
        parameters: { topk: 50, cost_bps: 15 },
        metrics: { annual_excess_return: 0.12 },
        equity_curve: [],
        warnings: [],
      },
    });

    const result = await backtestApi.runTrainedStrategy({
      runId: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      modelId: 'xgboost',
      strategyId: 'low_turnover_weekly',
      replayMode: 'custom',
      startDate: '2024-01-01',
      endDate: '2025-12-31',
      topk: 50,
      costBps: 15,
    });

    expect(result.metrics.annualExcessReturn).toBe(0.12);
    expect(post).toHaveBeenCalledWith(
      '/api/v1/backtest/strategy/trained',
      {
        run_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        model_id: 'xgboost',
        strategy_id: 'low_turnover_weekly',
        replay_mode: 'custom',
        start_date: '2024-01-01',
        end_date: '2025-12-31',
        topk: 50,
        cost_bps: 15,
      },
      { timeout: 120000 },
    );
  });

  it('loads the immutable historical trained-run catalog', async () => {
    get.mockResolvedValueOnce({
      data: {
        status: 'completed',
        generated_at: '2026-08-02T12:00:00+08:00',
        runs: [{
          run_id: 'a'.repeat(32),
          model_id: 'xgboost',
          model_label: 'XGBoost',
          source_type: 'candidate',
          exact_replay_available: true,
          custom_replay_available: true,
          strategies: [],
        }],
      },
    });

    const result = await backtestApi.getTrainedStrategyCatalog();

    expect(result.runs[0].runId).toBe('a'.repeat(32));
    expect(result.runs[0].exactReplayAvailable).toBe(true);
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/strategy/trained/catalog');
  });

  it('submits exact replay without browser-controlled backtest parameters', async () => {
    post.mockResolvedValueOnce({
      data: {
        status: 'completed',
        replay_mode: 'exact',
        metrics: {},
        equity_curve: [],
      },
    });

    await backtestApi.runTrainedStrategy({
      runId: 'a'.repeat(32),
      modelId: 'xgboost',
      strategyId: 'low_turnover_weekly',
      replayMode: 'exact',
    });

    expect(post).toHaveBeenCalledWith(
      '/api/v1/backtest/strategy/trained',
      {
        run_id: 'a'.repeat(32),
        model_id: 'xgboost',
        strategy_id: 'low_turnover_weekly',
        replay_mode: 'exact',
      },
      { timeout: 120000 },
    );
  });

  it('lists strategy assets without script bodies', async () => {
    get.mockResolvedValueOnce({
      data: [
        { id: 's1', name: 'Alpha', description: '', script_sha256: 'x'.repeat(64), api_version: 'dsa-script-v1', created_at: 't', updated_at: 't' },
      ],
    });

    const result = await backtestApi.listStrategyAssets();

    expect(result[0].id).toBe('s1');
    expect(result[0].scriptSha256).toBe('x'.repeat(64));
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/strategies');
  });

  it('saves a named strategy asset', async () => {
    post.mockResolvedValueOnce({
      data: { id: 's1', name: 'Alpha', script: 'def on_bar(c, d):\n    c.target_cash()', script_sha256: 'y'.repeat(64) },
    });

    const result = await backtestApi.saveStrategyAsset({
      name: 'Alpha',
      script: 'def on_bar(c, d):\n    c.target_cash()',
    });

    expect(result.id).toBe('s1');
    expect(post).toHaveBeenCalledWith('/api/v1/backtest/strategies', {
      name: 'Alpha',
      description: '',
      script: 'def on_bar(c, d):\n    c.target_cash()',
    });
  });

  it('updates and deletes a strategy asset', async () => {
    patch.mockResolvedValueOnce({ data: { id: 's1', name: 'Beta' } });
    await backtestApi.updateStrategyAsset('s1', { name: 'Beta' });
    expect(patch).toHaveBeenCalledWith('/api/v1/backtest/strategies/s1', {
      name: 'Beta',
      description: undefined,
      script: undefined,
    });

    del.mockResolvedValueOnce({ data: null });
    await backtestApi.deleteStrategyAsset('s1');
    expect(del).toHaveBeenCalledWith('/api/v1/backtest/strategies/s1');
  });

  it('lists backtest runs with filters and loads a full detail', async () => {
    get.mockResolvedValueOnce({
      data: [{ id: 'r1', metrics: { total_return_pct: 8.5 }, data_quality: { dataset_id: 'cn_etf_daily' } }],
    });

    const runs = await backtestApi.listBacktestRuns({ limit: 20, scriptSha256: 'z'.repeat(64) });

    expect(runs[0].metrics.totalReturnPct).toBe(8.5);
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/runs', {
      params: { limit: 20, script_sha256: 'z'.repeat(64) },
    });

    get.mockResolvedValueOnce({
      data: { id: 'r1', report: { status: 'completed', metrics: { total_return_pct: 8.5 } } },
    });
    const detail = await backtestApi.getBacktestRun('r1');
    expect(detail.report.status).toBe('completed');
    expect(get).toHaveBeenCalledWith('/api/v1/backtest/runs/r1');

    del.mockResolvedValueOnce({ data: null });
    await backtestApi.deleteBacktestRun('r1');
    expect(del).toHaveBeenCalledWith('/api/v1/backtest/runs/r1');
  });
});
