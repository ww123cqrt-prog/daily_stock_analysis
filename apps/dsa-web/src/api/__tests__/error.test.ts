import { describe, expect, it } from 'vitest';
import { parseApiError } from '../error';

describe('parseApiError', () => {
  it('identifies a trained-strategy replay timeout as local computation', () => {
    const parsed = parseApiError({
      code: 'ECONNABORTED',
      message: 'timeout of 120000ms exceeded',
      config: {
        url: '/api/v1/backtest/strategy/trained',
      },
    });

    expect(parsed.category).toBe('local_operation_timeout');
    expect(parsed.title).toBe('本地回测等待超时');
    expect(parsed.message).toContain('不代表外部数据源或网络故障');
  });

  it('keeps an unclassified timeout in the upstream timeout category', () => {
    const parsed = parseApiError({
      code: 'ECONNABORTED',
      message: 'read timeout',
    });

    expect(parsed.category).toBe('upstream_timeout');
  });
});
