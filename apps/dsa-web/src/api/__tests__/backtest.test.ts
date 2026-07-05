import { beforeEach, describe, expect, it, vi } from 'vitest';
import { backtestApi } from '../backtest';

const { get, post } = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    get,
    post,
    defaults: { baseURL: '' },
  },
}));

describe('backtestApi', () => {
  beforeEach(() => {
    get.mockReset();
    post.mockReset();
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
});
