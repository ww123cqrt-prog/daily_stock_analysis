import { beforeEach, describe, expect, it, vi } from 'vitest';
import { etfMomentumApi } from '../etfMomentum';

const { get } = vi.hoisted(() => ({ get: vi.fn() }));

vi.mock('../index', () => ({
  default: { get },
}));

describe('etfMomentumApi', () => {
  beforeEach(() => {
    get.mockReset();
  });

  it('requests the selected variant and preserves instrument keys in chart data', async () => {
    get.mockResolvedValueOnce({
      data: {
        research_id: 'etf_20d_three_asset_defensive_rotation',
        chart_data: {
          equity: [{ date: '2026-08-06', SH000300: 1.4, SH512890: 2.1, two_etf_legacy_exact: 3.2 }],
        },
      },
    });

    const result = await etfMomentumApi.getReport('three_etf');

    expect(get).toHaveBeenCalledWith('/api/v1/research/strategies/etf-momentum', {
      params: { variant: 'three_etf' },
    });
    expect(result).toMatchObject({
      researchId: 'etf_20d_three_asset_defensive_rotation',
      chartData: {
        equity: [{ SH000300: 1.4, SH512890: 2.1, twoEtfLegacyExact: 3.2 }],
      },
    });
  });
});
