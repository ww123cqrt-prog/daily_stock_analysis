export type ProviderStatusArtifact = 'training' | 'candidate';

type ProviderStatusInput = {
  providerCurrent?: boolean;
  providerStatus?: string;
  providerWarning?: string;
};

export type ProviderStatusNotice = {
  badge: string;
  title: string;
  message: string;
};

export function providerStatusNotice(
  input: ProviderStatusInput | null | undefined,
  artifact: ProviderStatusArtifact,
  language: 'zh' | 'en',
): ProviderStatusNotice | null {
  if (!input || input.providerCurrent === true) return null;

  const zh = language === 'zh';
  const subjectZh = artifact === 'training' ? '训练结果' : '候选结果';
  const subjectEn = artifact === 'training' ? 'Training result' : 'Candidate result';

  if (input.providerStatus === 'catalog_invalid') {
    return {
      badge: zh ? '原生 V2 已阻断' : 'Native V2 blocked',
      title: zh ? '原生 V2 清单无效' : 'Native V2 catalog invalid',
      message: zh
        ? (input.providerWarning || '当前正式数据清单无法通过校验，因此不能把该结果当作当前结果。')
        : 'The current canonical dataset cannot be verified, so this result cannot be treated as current.',
    };
  }

  if (input.providerCurrent !== false || input.providerStatus === 'unknown') {
    return {
      badge: zh ? '数据身份未证实' : 'Data identity unverified',
      title: zh ? `${subjectZh}的数据身份未证实` : `${subjectEn} data identity unverified`,
      message: zh
        ? (input.providerWarning || `${subjectZh}没有记录完整的不可变快照身份，无法证明它使用了当前原生 V2 数据。`)
        : `${subjectEn} does not record a complete immutable snapshot identity, so it cannot be verified against the current native V2 dataset.`,
    };
  }

  return {
    badge: artifact === 'training'
      ? (zh ? '历史训练结果已加载' : 'Historical training result loaded')
      : (zh ? '历史候选结果已加载' : 'Historical candidate result loaded'),
    title: zh ? `${subjectZh}来自旧数据` : `${subjectEn} uses historical data`,
    message: zh
      ? (input.providerWarning || `${subjectZh}只能作为历史参考，不能代表当前原生 V2 数据。`)
      : `${subjectEn} is historical only and does not represent the current native V2 dataset.`,
  };
}
