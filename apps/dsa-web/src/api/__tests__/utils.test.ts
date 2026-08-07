import { describe, expect, it } from 'vitest';
import { toCamelCase } from '../utils';

describe('toCamelCase', () => {
  it('can preserve security-code and CASH keys while converting report fields', () => {
    const result = toCamelCase<Record<string, unknown>>(
      {
        chart_data: [{ SH000300: 1, SZ159915: 2, CASH: 0.25, two_etf_legacy_exact: 3 }],
      },
      { exclude: [/^(?:SH|SZ)\d{6}$/, 'CASH'] },
    );

    expect(result).toEqual({
      chartData: [{ SH000300: 1, SZ159915: 2, CASH: 0.25, twoEtfLegacyExact: 3 }],
    });
  });
});
