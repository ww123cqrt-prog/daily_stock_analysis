import { describe, expect, it } from 'vitest';

import { providerStatusNotice } from '../providerStatus';


describe('providerStatusNotice', () => {
  it('shows no notice when there is no artifact or the provider is verified current', () => {
    expect(providerStatusNotice(undefined, 'training', 'zh')).toBeNull();
    expect(providerStatusNotice({ providerCurrent: true }, 'training', 'zh')).toBeNull();
  });

  it('fails closed when an artifact has no authoritative provider-current result', () => {
    expect(providerStatusNotice({}, 'training', 'zh')).toEqual({
      badge: '数据身份未证实',
      title: '训练结果的数据身份未证实',
      message: '训练结果没有记录完整的不可变快照身份，无法证明它使用了当前原生 V2 数据。',
    });
    expect(providerStatusNotice({ providerStatus: 'unknown' }, 'candidate', 'en')).toEqual({
      badge: 'Data identity unverified',
      title: 'Candidate result data identity unverified',
      message: 'Candidate result does not record a complete immutable snapshot identity, so it cannot be verified against the current native V2 dataset.',
    });
  });

  it('distinguishes an invalid catalog from a stale historical artifact', () => {
    expect(providerStatusNotice({
      providerCurrent: false,
      providerStatus: 'catalog_invalid',
    }, 'training', 'en')?.badge).toBe('Native V2 blocked');

    expect(providerStatusNotice({
      providerCurrent: false,
      providerStatus: 'stale',
    }, 'candidate', 'zh')?.badge).toBe('历史候选结果已加载');
  });
});
