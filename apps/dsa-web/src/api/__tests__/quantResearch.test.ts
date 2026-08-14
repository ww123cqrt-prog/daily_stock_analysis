/**
 * Tests for the quant research API client types and conversion.
 */

import { describe, expect, it } from 'vitest';
import type {
  Algorithm,
  Experiment,
  QuantResearchOverview,
  Research,
  Run,
} from '../quantResearch';

describe('quantResearch types', () => {
  it('Research has required fields', () => {
    const r: Research = {
      id: '123',
      name: 'Test',
      description: '',
      status: 'draft',
      researchType: 'stock_selection',
      config: {},
      createdAt: '2025-01-01T00:00:00Z',
      updatedAt: '2025-01-01T00:00:00Z',
    };
    expect(r.id).toBe('123');
    expect(r.status).toBe('draft');
  });

  it('Experiment has required fields', () => {
    const e: Experiment = {
      id: '456',
      researchId: '123',
      name: 'LightGBM',
      description: '',
      experimentType: 'stock_selection',
      methodKind: 'algorithm',
      algorithmId: 'lightgbm',
      protocol: {},
      protocolHash: null,
      status: 'draft',
      createdAt: '2025-01-01T00:00:00Z',
      updatedAt: '2025-01-01T00:00:00Z',
    };
    expect(e.algorithmId).toBe('lightgbm');
  });

  it('Run has required fields', () => {
    const r: Run = {
      id: '789',
      experimentId: '456',
      runLabel: 'First run',
      status: 'completed',
      startedAt: null,
      completedAt: null,
      metrics: { sharpe: 1.5 },
      artifacts: [],
      errorMessage: null,
      isLegacy: false,
      mlflowRunId: null,
      createdAt: '2025-01-01T00:00:00Z',
      updatedAt: '2025-01-01T00:00:00Z',
    };
    expect(r.metrics.sharpe).toBe(1.5);
  });

  it('Algorithm has required fields', () => {
    const a: Algorithm = {
      id: 'lightgbm',
      name: 'LightGBM',
      nameZh: 'LightGBM',
      category: 'tree_ensemble',
      description: 'A gradient boosting framework',
      descriptionZh: '梯度提升框架',
      capability: {
        supportsClassification: true,
        supportsRegression: true,
        supportsFeatureImportance: true,
        supportsOnlineLearning: false,
        requiresGpu: false,
        maxFeatures: null,
        supportedLossFunctions: ['mse', 'logloss'],
      },
      configSchema: {
        parameters: {},
        required: [],
      },
      defaultParams: {},
    };
    expect(a.id).toBe('lightgbm');
    expect(a.capability.supportsRegression).toBe(true);
  });

  it('QuantResearchOverview has required fields', () => {
    const o: QuantResearchOverview = {
      totalResearch: 5,
      totalExperiments: 10,
      totalRuns: 25,
      recentResearch: [],
    };
    expect(o.totalResearch).toBe(5);
  });
});
