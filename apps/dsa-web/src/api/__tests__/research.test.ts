import { beforeEach, describe, expect, it, vi } from 'vitest';
import { researchApi } from '../research';

const { get } = vi.hoisted(() => ({
  get: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    get,
    defaults: { baseURL: '' },
  },
}));

describe('researchApi', () => {
  beforeEach(() => {
    get.mockReset();
  });

  it('loads the backend capability matrix with frontend camelCase keys', async () => {
    get.mockResolvedValueOnce({
      data: {
        generated_at: '2026-07-05T12:00:00+08:00',
        trainable: true,
        training_readiness: { status: 'trainable', summary: 'ready' },
        data_sources: [],
        qlib_predictions: {
          generated_at: '2026-07-05T10:00:00+08:00',
          mode: 'read_only_qlib_prediction_summary',
          run_id: 'run_1',
          latest_date: '2026-07-02',
          top: [{ instrument: 'SZ000002', score: 0.25 }],
          bottom: [{ instrument: 'SZ000001', score: -0.4 }],
          summary: { latest_rows: 3 },
        },
        capability_matrix: [
          {
            id: 'trading_execution',
            module: '交易执行',
            status: 'not_planned',
            entry: '无网页入口',
            tests: '不提供交易 API',
            browser: '页面只展示研究、回测和观察能力',
            review: '安全边界需持续复审',
            next: '保持人工交易，不做自动下单',
            restrictions: ['不自动下单', '不读取券商密码'],
          },
        ],
        function_inventory: [
          {
            id: 'qlib_data_pipeline',
            area: 'Qlib 接入',
            name: '数据检查/转换/训练/回测',
            status: 'available',
            entry: '工作流 + 回测 + MLflow',
            route: '/workflow',
            owner: 'quant_workflow + Qlib',
            coverage: '网页可触发数据检查、Qlib 转换和训练。',
            gaps: '继续补模型质量解释。',
            next: '增加 Qlib 模型效果解释。',
            verification: 'train_baseline_lgb 测试。',
          },
        ],
        china_astock_reuse: [
          {
            id: 'paper_trading',
            capability: '纸面交易/券商适配',
            source: 'trading/*',
            status: 'not_planned',
            priority: '禁止项',
            decision: '本阶段不接入',
            target: '无网页入口',
            next: '保持为参考代码。',
            restrictions: ['不自动下单', '不读取券商密码'],
          },
        ],
      },
    });

    const status = await researchApi.getStatus();

    expect(get).toHaveBeenCalledWith('/api/v1/research/status');
    expect(status.generatedAt).toBe('2026-07-05T12:00:00+08:00');
    expect(status.qlibPredictions?.latestDate).toBe('2026-07-02');
    expect(status.qlibPredictions?.top?.[0].instrument).toBe('SZ000002');
    expect(status.qlibPredictions?.bottom?.[0].score).toBe(-0.4);
    expect(status.qlibPredictions?.summary?.latestRows).toBe(3);
    expect(status.capabilityMatrix?.[0].id).toBe('trading_execution');
    expect(status.capabilityMatrix?.[0].restrictions).toEqual(['不自动下单', '不读取券商密码']);
    expect(status.functionInventory?.[0].route).toBe('/workflow');
    expect(status.functionInventory?.[0].verification).toBe('train_baseline_lgb 测试。');
    expect(status.chinaAstockReuse?.[0].id).toBe('paper_trading');
    expect(status.chinaAstockReuse?.[0].restrictions).toEqual(['不自动下单', '不读取券商密码']);
  });
});
