import { beforeEach, describe, expect, it, vi } from 'vitest';
import { researchApi } from '../research';

const { del, get } = vi.hoisted(() => ({
  del: vi.fn(),
  get: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    delete: del,
    get,
    defaults: { baseURL: '' },
  },
}));

describe('researchApi', () => {
  beforeEach(() => {
    del.mockReset();
    get.mockReset();
  });

  it('loads the backend capability matrix with frontend camelCase keys', async () => {
    get.mockResolvedValueOnce({
      data: {
        generated_at: '2026-07-05T12:00:00+08:00',
        trainable: true,
        training_readiness: { status: 'trainable', summary: 'ready' },
        data_sources: [],
        robust_signal: {
          status: 'missing',
          data_readiness: {
            contract_version: 'robust_signal_data_v3',
            acceptance_owner: 'data_module',
            acceptance_contract: 'data_module_accepted_content_addressed_snapshot_v2',
            status: 'ready',
            trainable: true,
            readiness_artifact_name: `robust_signal_data_readiness_${'e'.repeat(64)}.json`,
            readiness_artifact_sha256: 'e'.repeat(64),
            quality_report_id: 'data-quality-0123456789abcdef',
            quality_evidence_artifact_name: 'data_quality_report_20260805_120000_0123456789abcdef.json',
            quality_evidence_sha256: 'f'.repeat(64),
            cleaned_provider_identity: {
              dataset_id: 'qlib_ashare_cleaned_v2',
              snapshot_id: 'cleaned-v2',
              manifest_sha256: 'a'.repeat(64),
              content_sha256: 'b'.repeat(64),
            },
          },
        },
        qlib_predictions: {
          generated_at: '2026-07-05T10:00:00+08:00',
          mode: 'read_only_qlib_prediction_summary',
          run_id: 'run_1',
          latest_date: '2026-07-02',
          top: [{ instrument: 'SZ000002', score: 0.25 }],
          bottom: [{ instrument: 'SZ000001', score: -0.4 }],
          summary: { latest_rows: 3 },
        },
        qlib_research: {
          status: 'available',
          available_sections: 1,
          total_sections: 6,
          sections: {
            risk_model: {
              section: 'risk_model',
              title: '风险模型集成诊断',
              status: 'available',
              generated_at: '2026-07-05T11:00:00+08:00',
              risk_metrics: { beta_to_benchmark: 0.8 },
            },
          },
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
    expect(status.robustSignal?.dataReadiness?.acceptanceOwner).toBe('data_module');
    expect(status.robustSignal?.dataReadiness?.acceptanceContract).toBe(
      'data_module_accepted_content_addressed_snapshot_v2',
    );
    expect(status.robustSignal?.dataReadiness?.readinessArtifactName).toBe(
      `robust_signal_data_readiness_${'e'.repeat(64)}.json`,
    );
    expect(status.robustSignal?.dataReadiness?.readinessArtifactSha256).toBe('e'.repeat(64));
    expect(status.robustSignal?.dataReadiness?.qualityReportId).toBe(
      'data-quality-0123456789abcdef',
    );
    expect(status.robustSignal?.dataReadiness?.qualityEvidenceArtifactName).toBe(
      'data_quality_report_20260805_120000_0123456789abcdef.json',
    );
    expect(status.robustSignal?.dataReadiness?.qualityEvidenceSha256).toBe('f'.repeat(64));
    expect(status.robustSignal?.dataReadiness?.cleanedProviderIdentity?.snapshotId).toBe(
      'cleaned-v2',
    );
    expect(status.qlibPredictions?.latestDate).toBe('2026-07-02');
    expect(status.qlibPredictions?.top?.[0].instrument).toBe('SZ000002');
    expect(status.qlibPredictions?.bottom?.[0].score).toBe(-0.4);
    expect(status.qlibPredictions?.summary?.latestRows).toBe(3);
    expect(status.qlibResearch?.availableSections).toBe(1);
    expect(status.qlibResearch?.sections.risk_model.riskMetrics?.betaToBenchmark).toBe(0.8);
    expect(status.capabilityMatrix?.[0].id).toBe('trading_execution');
    expect(status.capabilityMatrix?.[0].restrictions).toEqual(['不自动下单', '不读取券商密码']);
    expect(status.functionInventory?.[0].route).toBe('/workflow');
    expect(status.functionInventory?.[0].verification).toBe('train_baseline_lgb 测试。');
    expect(status.chinaAstockReuse?.[0].id).toBe('paper_trading');
    expect(status.chinaAstockReuse?.[0].restrictions).toEqual(['不自动下单', '不读取券商密码']);
  });

  it('preserves artifact file names while camel-casing training metadata', async () => {
    get.mockResolvedValueOnce({
      data: {
        generated_at: '2026-08-02T17:46:17+08:00',
        trainable: true,
        training_readiness: { status: 'trainable', summary: 'ready' },
        data_sources: [],
        latest_training: {
          status: 'completed',
          run_id: 'run-with-replay-artifacts',
          artifact_sha256: {
            'pred.pkl': 'a'.repeat(64),
            'label.pkl': 'b'.repeat(64),
            'portfolio_analysis/report_normal_1day.pkl': 'c'.repeat(64),
          },
        },
        candidate_pipeline: {
          status: 'completed',
          models: [
            {
              id: 'lightgbm',
              status: 'completed',
              artifact_sha256: {
                'pred.pkl': 'd'.repeat(64),
                'label.pkl': 'e'.repeat(64),
              },
            },
          ],
        },
      },
    });

    const status = await researchApi.getStatus();

    expect(status.latestTraining?.artifactSha256).toEqual({
      'pred.pkl': 'a'.repeat(64),
      'label.pkl': 'b'.repeat(64),
      'portfolio_analysis/report_normal_1day.pkl': 'c'.repeat(64),
    });
    expect(status.candidatePipeline?.models?.[0].artifactSha256).toEqual({
      'pred.pkl': 'd'.repeat(64),
      'label.pkl': 'e'.repeat(64),
    });
  });

  it('passes an explicit backend confirmation token for permanent experiment deletion', async () => {
    del.mockResolvedValueOnce({
      data: {
        removed_run: { run_id: 'abc123' },
        note_removed: true,
        experiment_notebook: { status: 'available', runs: [], total_runs: 0 },
      },
    });

    const result = await researchApi.permanentlyDeleteExperimentRun('abc123');

    expect(del).toHaveBeenCalledWith(
      '/api/v1/research/experiments/abc123/permanent',
      { params: { confirm: 'permanent' } },
    );
    expect(result.noteRemoved).toBe(true);
    expect(result.removedRun.runId).toBe('abc123');
  });
});
